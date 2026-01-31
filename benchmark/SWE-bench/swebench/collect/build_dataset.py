#!/usr/bin/env python3

import argparse
import json
import logging
import os
from typing import Optional

from swebench.collect.utils import (
    extract_patches,
    extract_problem_statement_and_hints,
    Repo,
    extract_commit_patches,
    get_commit_title,
    get_pull_request_conversation
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def _format_pr_conversation(data):
        lines = []
        for item in data or []:
            created_at = item.get("created_at", "")
            typ = item.get("type", "")
            author = item.get("author", "")
            body = (item.get("body") or "").replace("\r", " ").replace("\n", " ").strip()
            lines.append(f"{created_at}\t{typ}\t{author}\t{body}")
        return "\n".join(lines)

def create_instance(repo: Repo, pull: dict, token: Optional[str] = None) -> dict:
    """
    Create a single task instance from a pull request, where task instance is:

    {
        repo (str): owner/repo this task instance is from,
        pull_number (int): number of PR this task instance is from,
        base_commit (str): SHA of the base commit PR is based on,
        patch (str): reference solution as .patch (apply to base commit),
        test_patch (str): test suite as .patch (apply to base commit),
    }
    """
    patch, test_patch = extract_patches(pull, repo)
    problem_statement, hints = extract_problem_statement_and_hints(pull, repo)
    pr_conversation_raw = get_pull_request_conversation(repo.owner, repo.name, pull["number"], token=token)
    pr_conversation = _format_pr_conversation(pr_conversation_raw)
    commit_info = extract_commit_patches(pull, repo, include_merge=False, sleep=0.0)
    commits_fix = commit_info.get("commits_fix", [])
    commits_iter = repo.get_all_loop(repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True)
    commits_list = list(commits_iter)
    sha_to_commit = {c.sha: c for c in commits_list}
    commit_titles = []
    fix_shas_in_order = [e.get("sha") for e in commits_fix if e.get("sha")]
    for sha in fix_shas_in_order:
        c = sha_to_commit.get(sha)
        if not c:
            continue
        commit_url = getattr(
            c, "html_url", f"https://github.com/{repo.owner}/{repo.name}/commit/{sha}"
        )
        title = get_commit_title(commit_url)
        if not title and getattr(c, "commit", None) and getattr(c.commit, "message", ""):
            title = c.commit.message.split("\n", 1)[0]
        if title:
            commit_titles.append(title)
    return {
        "repo": repo.repo.full_name,
        "pull_number": pull["number"],
        "instance_id": (repo.repo.full_name + "-" + str(pull["number"])).replace(
            "/", "__"
        ),
        "issue_numbers": pull["resolved_issues"],
        "base_commit": pull["base"]["sha"],
        "patch": patch,
        "test_patch": test_patch,
        # "commit_count": commit_info.get("commit_count", 0),
        "commit_fix_count": len(commit_info.get("patch_fix", [])),
        "commit_patch_fix_list": commit_info.get("patch_fix", []),
        # "commit_patch_test_list": commit_info.get("patch_test", []),
        "commits_fix_sha": [e.get("sha") for e in commits_fix],
        "commits_fix_date": [e.get("date") for e in commits_fix],
        "commit_titles": commit_titles,
        "problem_statement": problem_statement,
        "hints_text": hints,
        "pr_conversation": pr_conversation,
        "created_at": pull["created_at"],
    }


def is_valid_pull(pull: dict) -> bool:
    """
    Check whether PR has an associated issue and is merged

    Args:
        pull (dict): pull request object
    Returns:
        bool: whether PR is valid
    """
    if pull["merged_at"] is None:
        return False
    if "resolved_issues" not in pull or len(pull["resolved_issues"]) < 1:
        return False
    return True


def is_valid_instance(instance: dict) -> bool:
    """
    Check whether task instance has all required fields for task instance creation

    Args:
        instance (dict): task instance object
    Returns:
        bool: whether task instance is valid
    """
    if instance["patch"] is None or instance["patch"] == "":
        return False
    if instance["problem_statement"] is None or instance["problem_statement"] == "":
        return False
    return True


def has_test_patch(instance: dict) -> bool:
    """
    Check whether task instance has a test suite

    Args:
        instance (dict): task instance object
    Returns:
        bool: whether task instance has a test suite
    """
    if instance["test_patch"] is None or instance["test_patch"].strip() == "":
        return False
    return True

def has_enough_fix_commits(instance: dict) -> bool:
    """
    过滤 commit_fix_count < 2 和 commit_fix_count >= 30 的实例（这些实例不进入主输出）
    """
    count = instance.get("commit_fix_count", 0)
    if count < 2 or count >= 30:
        return False
    return True

def has_commit_title(instance: dict) -> bool:
    """
    过滤掉没有 commit title 的实例
    """
    if not instance.get("commit_titles", []):
        return False
    return True

def create_at_after_2025(instance: dict) -> bool:
    """
    过滤掉创建时间在 2025 年之前的实例
    """
    created_at = instance.get("created_at", "")
    if created_at <= "2025-01-01T00:00:00Z":
        return False
    return True

def main(pr_file: str, output: str, token: Optional[str] = None):
    """
    Main thread for creating task instances from pull requests

    Args:
        pr_file (str): path to pull request JSONL file
        output (str): output file name
        token (str): GitHub token
    """
    if token is None:
        # Get GitHub token from environment variable if not provided
        token = os.environ.get("GITHUB_TOKEN")

    def load_repo(repo_name):
        # Return repo object for a given repo name
        owner, repo = repo_name.split("/")
        return Repo(owner, repo, token=token)

    repos = dict()
    completed = 0
    with_tests = 0
    total_instances = 0
    all_output = output + ".all"
    seen_prs = set()

    # Continue where we left off if output file already exists
    if os.path.exists(all_output):
        with open(all_output) as f:
            for line in f:
                pr = json.loads(line)
                if "instance_id" not in pr:
                    pr["instance_id"] = (
                        pr["repo"] + "-" + str(pr["pull_number"])
                    ).replace("/", "__")
                instance_id = pr["instance_id"]
                seen_prs.add(instance_id)
                if is_valid_instance(pr):
                    completed += 1
                    if has_test_patch(pr):
                        with_tests += 1
    logger.info(
        f"Will skip {len(seen_prs)} pull requests that have already been inspected"
    )

    # Write to .all file for all PRs
    write_mode_all = "w" if not os.path.exists(all_output) else "a"
    with open(all_output, write_mode_all) as all_output:
        # Write to output file for PRs with test suites
        write_mode = "w" if not os.path.exists(output) else "a"
        with open(output, write_mode) as output:
            for ix, line in enumerate(open(pr_file)):
                total_instances += 1
                pull = json.loads(line)
                if ix % 100 == 0:
                    logger.info(
                        f"[{pull['base']['repo']['full_name']}] (Up to {ix} checked) "
                        f"{completed} valid, {with_tests} with tests."
                    )
                # Construct instance fields
                instance_id = (
                    pull["base"]["repo"]["full_name"] + "-" + str(pull["number"])
                )
                instance_id = instance_id.replace("/", "__")
                if instance_id in seen_prs:
                    seen_prs -= {instance_id}
                    continue
                if not is_valid_pull(pull):
                    # Throw out invalid PRs
                    continue
                # Create task instance
                repo_name = pull["base"]["repo"]["full_name"]
                if repo_name not in repos:
                    repos[repo_name] = load_repo(repo_name)
                repo = repos[repo_name]
                instance = create_instance(repo, pull, token=token)
                if is_valid_instance(instance):
                    # If valid, write to .all output file
                    print(
                        json.dumps(instance), end="\n", flush=True, file=all_output
                    )  # write all instances to a separate file
                    completed += 1
                    # 同时需要测试补丁 & 足够的修复 commit 数量 & 有 commit title & 创建时间在 2025 年之后
                    if has_test_patch(instance) and has_enough_fix_commits(instance) and has_commit_title(instance) and create_at_after_2025(instance):
                        print(json.dumps(instance), end="\n", flush=True, file=output)
                        with_tests += 1
    logger.info(
        f"[{', '.join(repos.keys())}] Total instances: {total_instances}, completed: {completed}, with tests: {with_tests}"
    )
    logger.info(
        f"[{', '.join(repos.keys())}] Skipped {len(seen_prs)} pull requests that have already been inspected"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pr_file", type=str, help="Path to pull request JSONL file")
    parser.add_argument("output", type=str, help="Output file name")
    parser.add_argument("--token", type=str, help="GitHub token")
    args = parser.parse_args()
    main(**vars(args))
