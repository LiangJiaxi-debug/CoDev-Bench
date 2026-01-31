from __future__ import annotations


import logging
import re
import requests
import time

from bs4 import BeautifulSoup
from ghapi.core import GhApi
from fastcore.net import HTTP404NotFoundError, HTTP403ForbiddenError
from typing import Callable, Iterator, Optional, Dict, Any, List
from unidiff import PatchSet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests
PR_KEYWORDS = {
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
}


class Repo:
    def __init__(self, owner: str, name: str, token: Optional[str] = None):
        """
        Init to retrieve target repository and create ghapi tool

        Args:
            owner (str): owner of target repository
            name (str): name of target repository
            token (str): github token
        """
        self.owner = owner
        self.name = name
        self.token = token
        self.api = GhApi(token=token)
        self.repo = self.call_api(self.api.repos.get, owner=owner, repo=name)

    def call_api(self, func: Callable, **kwargs) -> dict | None:
        """
        API call wrapper with rate limit handling (checks every 5 minutes if rate limit is reset)

        Args:
            func (callable): API function to call
            **kwargs: keyword arguments to pass to API function
        Return:
            values (dict): response object of `func`
        """
        while True:
            try:
                values = func(**kwargs)
                return values
            except HTTP403ForbiddenError:
                while True:
                    rl = self.api.rate_limit.get()
                    logger.info(
                        f"[{self.owner}/{self.name}] Rate limit exceeded for token {self.token[:10]}, "
                        f"waiting for 5 minutes, remaining calls: {rl.resources.core.remaining}"
                    )
                    if rl.resources.core.remaining > 0:
                        break
                    time.sleep(60 * 5)
            except HTTP404NotFoundError:
                logger.info(f"[{self.owner}/{self.name}] Resource not found {kwargs}")
                return None

    def extract_resolved_issues(self, pull: dict) -> list[str]:
        """
        Extract list of issues referenced by a PR

        Args:
            pull (dict): PR dictionary object from GitHub
        Return:
            resolved_issues (list): list of issue numbers referenced by PR
        """
        # Define 1. issue number regex pattern 2. comment regex pattern 3. keywords
        issues_pat = re.compile(r"(\w+)\s+\#(\d+)")
        comments_pat = re.compile(r"(?s)<!--.*?-->")

        # Construct text to search over for issue numbers from PR body and commit messages
        text = pull.title if pull.title else ""
        text += "\n" + (pull.body if pull.body else "")
        commits = self.get_all_loop(
            self.api.pulls.list_commits, pull_number=pull.number, quiet=True
        )
        commit_messages = [commit.commit.message for commit in commits]
        commit_text = "\n".join(commit_messages) if commit_messages else ""
        text += "\n" + commit_text
        # Remove comments from text
        text = comments_pat.sub("", text)
        # Look for issue numbers in text via scraping <keyword, number> patterns
        references = issues_pat.findall(text)
        resolved_issues_set = set()
        if references:
            for word, issue_num in references:
                if word.lower() in PR_KEYWORDS:
                    resolved_issues_set.add(issue_num)
        return list(resolved_issues_set)

    def get_all_loop(
        self,
        func: Callable,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        quiet: bool = False,
        **kwargs,
    ) -> Iterator:
        """
        Return all values from a paginated API endpoint.

        Args:
            func (callable): API function to call
            per_page (int): number of values to return per page
            num_pages (int): number of pages to return
            quiet (bool): whether to print progress
            **kwargs: keyword arguments to pass to API function
        """
        page = 1
        args = {
            "owner": self.owner,
            "repo": self.name,
            "per_page": per_page,
            **kwargs,
        }
        while True:
            try:
                # Get values from API call
                values = func(**args, page=page)
                yield from values
                if len(values) == 0:
                    break
                if not quiet:
                    rl = self.api.rate_limit.get()
                    logger.info(
                        f"[{self.owner}/{self.name}] Processed page {page} ({per_page} values per page). "
                        f"Remaining calls: {rl.resources.core.remaining}"
                    )
                if num_pages is not None and page >= num_pages:
                    break
                page += 1
            except Exception as e:
                # Rate limit handling
                logger.error(
                    f"[{self.owner}/{self.name}] Error processing page {page} "
                    f"w/ token {self.token[:10]} - {e}"
                )
                while True:
                    rl = self.api.rate_limit.get()
                    if rl.resources.core.remaining > 0:
                        break
                    logger.info(
                        f"[{self.owner}/{self.name}] Waiting for rate limit reset "
                        f"for token {self.token[:10]}, checking again in 5 minutes"
                    )
                    time.sleep(60 * 5)
        if not quiet:
            logger.info(
                f"[{self.owner}/{self.name}] Processed {(page - 1) * per_page + len(values)} values"
            )

    def get_all_issues(
        self,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        direction: str = "desc",
        sort: str = "created",
        state: str = "closed",
        quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all issues from repo

        Args:
            per_page (int): number of issues to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort issues
            sort (str): field to sort issues by
            state (str): state of issues to look for
            quiet (bool): whether to print progress
        """
        issues = self.get_all_loop(
            self.api.issues.list_for_repo,
            num_pages=num_pages,
            per_page=per_page,
            direction=direction,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return issues

    def get_all_pulls(
        self,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        direction: str = "desc",
        sort: str = "created",
        state: str = "closed",
        quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all PRs from repo

        Args:
            per_page (int): number of PRs to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort PRs
            sort (str): field to sort PRs by
            state (str): state of PRs to look for
            quiet (bool): whether to print progress
        """
        pulls = self.get_all_loop(
            self.api.pulls.list,
            num_pages=num_pages,
            direction=direction,
            per_page=per_page,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return pulls


def extract_problem_statement_and_hints(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Extract problem statement from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    if repo.name == "django":
        return extract_problem_statement_and_hints_django(pull, repo)
    text = ""
    all_hint_texts = list()
    for issue_number in pull["resolved_issues"]:
        issue = repo.call_api(
            repo.api.issues.get,
            owner=repo.owner,
            repo=repo.name,
            issue_number=issue_number,
        )
        if issue is None:
            continue
        title = issue.title if issue.title else ""
        body = issue.body if issue.body else ""
        text += f"{title}\n{body}\n"
        issue_number = issue.number
        hint_texts = _extract_hints(pull, repo, issue_number)
        hint_text = "\n".join(hint_texts)
        all_hint_texts.append(hint_text)
    return text, "\n".join(all_hint_texts) if all_hint_texts else ""


def _extract_hints(pull: dict, repo: Repo, issue_number: int) -> list[str]:
    """
    Extract hints from comments associated with a pull request (before first commit)

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
        issue_number (int): issue number
    Return:
        hints (list): list of hints
    """
    # Get all commits in PR
    commits = repo.get_all_loop(
        repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
    )
    commits = list(commits)
    if len(commits) == 0:
        # If there are no comments, return no hints
        return []
    # Get time of first commit in PR
    commit_time = commits[0].commit.author.date  # str
    commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))
    # Get all comments in PR
    all_comments = repo.get_all_loop(
        repo.api.issues.list_comments, issue_number=issue_number, quiet=True
    )
    all_comments = list(all_comments)
    # Iterate through all comments, only keep comments created before first commit
    comments = list()
    for comment in all_comments:
        comment_time = time.mktime(
            time.strptime(comment.updated_at, "%Y-%m-%dT%H:%M:%SZ")
        )  # use updated_at instead of created_at
        if comment_time < commit_time:
            comments.append(comment)
        else:
            break
        # only include information available before the first commit was created
    # Keep text from comments
    comments = [comment.body for comment in comments]
    return comments


def extract_patches(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Get patch and test patch from PR

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """
    patch = requests.get(pull["diff_url"]).text
    patch_test = ""
    patch_fix = ""
    for hunk in PatchSet(patch):
        if any(
            test_word in hunk.path for test_word in ["test", "tests", "e2e", "testing"]
        ):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix, patch_test


### MARK: Repo Specific Parsing Functions ###
def extract_problem_statement_and_hints_django(
    pull: dict, repo: Repo
) -> tuple[str, list[str]]:
    """
    Get problem statement and hints from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    text = ""
    all_hints_text = list()
    for issue_number in pull["resolved_issues"]:
        url = f"https://code.djangoproject.com/ticket/{issue_number}"
        resp = requests.get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get problem statement (title + body)
        issue_desc = soup.find("div", {"id": "ticket"})
        title = issue_desc.find("h1", class_="searchable").get_text()
        title = re.sub(r"\s+", " ", title).strip()
        body = issue_desc.find("div", class_="description").get_text()
        body = re.sub(r"\n+", "\n", body)
        body = re.sub(r"    ", "\t", body)
        body = re.sub(r"[ ]{2,}", " ", body).strip()
        text += f"{title}\n{body}\n"

        # Get time of first commit in PR
        commits = repo.get_all_loop(
            repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
        )
        commits = list(commits)
        if len(commits) == 0:
            continue
        commit_time = commits[0].commit.author.date
        commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))

        # Get all comments before first commit
        comments_html = soup.find("div", {"id": "changelog"})
        div_blocks = comments_html.find_all("div", class_="change")
        # Loop through each div block
        for div_block in div_blocks:
            # Find the comment text and timestamp
            comment_resp = div_block.find("div", class_="comment")
            timestamp_resp = div_block.find("a", class_="timeline")
            if comment_resp is None or timestamp_resp is None:
                continue

            comment_text = re.sub(r"\s+", " ", comment_resp.text).strip()
            timestamp = timestamp_resp["title"]
            if timestamp.startswith("See timeline at "):
                timestamp = timestamp[len("See timeline at ") :]
            if "/" in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%m/%d/%y %H:%M:%S"))
            elif "," in timestamp:
                timestamp = time.mktime(
                    time.strptime(timestamp, "%b %d, %Y, %I:%M:%S %p")
                )
            else:
                raise ValueError(f"Timestamp format not recognized: {timestamp}")

            # Append the comment and timestamp as a tuple to the comments list
            if timestamp < commit_time:
                all_hints_text.append((comment_text, timestamp))

    return text, all_hints_text

def _split_patch(patch: str) -> tuple[str, str]:
    """
    将完整 patch 文本拆分成 代码修改部分 与 测试修改部分。
    """
    patch_test = ""
    patch_fix = ""
    for hunk in PatchSet(patch):
        if any(test_word in hunk.path for test_word in ["test", "tests", "e2e", "testing"]):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix, patch_test

_CODE_EXTS = {".py"}

def _filter_code_patch(patch_text: str) -> str:
    """
    仅保留属于代码文件(_CODE_EXTS)的 diff hunk.
    传入应为已去除测试文件后的 patch_fix。
    """
    def _is_code_file(path: str) -> bool:
        return any(path.endswith(ext) for ext in _CODE_EXTS)
    
    if not patch_text:
        return ""
    kept = ""
    for hunk in PatchSet(patch_text):
        if _is_code_file(hunk.path):
            kept += str(hunk)
    return kept

def extract_commit_patches(
    pull: dict,
    repo: Repo,
    include_merge: bool = False,
    sleep: float = 0.0,
) -> dict:
    """
    聚合获取 PR 中全部 commit 的 patch，并按时间顺序输出。

    Args:
        pull (dict): PR 对象（包含 number）
        repo (Repo): Repo 实例
        include_merge (bool): 是否包含 merge commit（多个 parent）
        sleep (float): 每次请求间隔秒数

    Return:
        {
            "commit_count": int,                # 该 PR 的 commit 总数（含/不含 merge 取决于 include_merge）
            "patch_fix": [str, ...],            # 按日期升序的非测试补丁列表
            "patch_test": [str, ...],           # 按日期升序的测试补丁列表
            "commits_fix_sha": [                        # 按日期升序
                {
                    "sha": str,
                }, ...
            ]
        }
    """
    raw_commits = list(
        repo.get_all_loop(
            repo.api.pulls.list_commits,
            pull_number=pull["number"],
            quiet=True,
        )
    )

    entries = []
    for commit in raw_commits:
        sha = commit.sha
        is_merge = len(getattr(commit, "parents", []) or []) > 1
        if (not include_merge) and is_merge:
            continue
        date = commit.commit.author.date if commit.commit and commit.commit.author else ""
        patch_url = f"https://github.com/{repo.owner}/{repo.name}/commit/{sha}.patch"
        patch_fix = ""
        patch_test = ""
        try:
            resp = requests.get(patch_url, timeout=30)
            if resp.status_code != 200:
                logger.warning(
                    f"[{repo.owner}/{repo.name}] 获取 commit patch 失败 {sha} status={resp.status_code}"
                )
            else:
                patch_fix, patch_test = _split_patch(resp.text)
                patch_fix = _filter_code_patch(patch_fix)
        except Exception as e:
            logger.error(f"[{repo.owner}/{repo.name}] 请求 commit patch 出错 sha={sha} err={e}")
        entries.append(
            {
                "sha": sha,
                "date": date,
                "is_merge": is_merge,
                "patch_url": patch_url,
                "_patch_fix": patch_fix,
                "_patch_test": patch_test,
            }
        )
        if sleep > 0:
            time.sleep(sleep)

    def _key(e):
        d = e["date"]
        if not d:
            return float("inf")
        try:
            return time.mktime(time.strptime(d, "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            return float("inf")

    entries.sort(key=_key)

    patch_fix_list = [e["_patch_fix"] for e in entries if e["_patch_fix"]]
    patch_test_list = [e["_patch_test"] for e in entries if e["_patch_test"]]

    result = {
        "commit_count": len(entries),
        "patch_fix": patch_fix_list,
        "patch_test": patch_test_list,
        "commits_fix": [{"sha": e["sha"] , "date": e["date"]} for e in entries if e["_patch_fix"]],
    }
    return result

def get_commit_title(commit_url: str) -> str:
    """
    提取单个 commit 页面中的标题文本。
    Args:
        commit_url (str): 形如 https://github.com/{owner}/{repo}/commit/{sha} 或
                          https://github.com/{owner}/{repo}/pull/{pr_number}/commits/{sha}
    Return:
        title (str): 提取到的标题；若失败返回空字符串
    """
    try:
        # 添加请求头模拟浏览器访问，避免被GitHub拒绝
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        resp = requests.get(commit_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"获取 commit 页面失败 {commit_url} status={resp.status_code}")
            return ""
        
        soup = BeautifulSoup(resp.text, "html.parser")
        title_text = ""
        
        # GitHub commit标题的准确选择器
        # 主选择器：针对commit页面的标题
        title_node = soup.select_one('p.commit-title > span')
        if title_node and title_node.get_text(strip=True):
            title_text = title_node.get_text(strip=True)
        else:
            # 备选选择器：处理不同页面结构
            for sel in [
                '.commit-title', 
                'div.commit-desc > h3',
                '.js-commit-title'
            ]:
                node = soup.select_one(sel)
                if node and node.get_text(strip=True):
                    title_text = node.get_text(strip=True)
                    break
        
        # 过滤掉可能的搜索框文本
        if "Search code" in title_text:
            title_text = ""
            
        return title_text
    except Exception as e:
        logger.error(f"解析 commit 标题失败 {commit_url}: {e}")
        return ""
    
def get_pull_request_conversation(
    owner: str,
    repo: str,
    number: int,
    token: Optional[str] = None,
    max_pages: int = 10,
    per_page: int = 100,
    sleep_seconds: float = 0.6,
    logger=None,
) -> Dict[str, Any]:
    """
    获取单一按时间排序的 PR 对话(类似网页 Conversation Tab).
    conversation 中每个元素结构:
    {
      "type": "pr_body" | "issue_comment" | "review" | "review_comment" | "timeline_event",
      "author": str | None,
      "created_at": ISO8601 或 None,
      "body": 文本(可能为空),
      "raw": 原始对象,
      "extra": 附加字段(dict)
    }
    """
    session = requests.Session()
    base_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "swebench-convo"
    }
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    def _req(url: str, params=None, headers=None, allow_404=False):
        hdrs = headers or base_headers
        for attempt in range(3):
            resp = session.get(url, headers=hdrs, params=params, timeout=30)
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait = max(0, reset - int(time.time()) + 1)
                if logger:
                    logger.warning(f"Rate limit hit. Sleep {wait}s")
                time.sleep(wait)
                continue
            if resp.ok:
                try:
                    return resp.json()
                except Exception:
                    return None
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        if logger:
            logger.error(f"Request failed {url}: {resp.status_code} {resp.text[:200]}")
        return None

    def _paginate(url: str, headers=None):
        out = []
        for page in range(1, max_pages + 1):
            data = _req(url, params={"page": page, "per_page": per_page}, headers=headers)
            if not isinstance(data, list):
                break
            out.extend(data)
            if len(data) < per_page:
                break
            time.sleep(sleep_seconds)
        return out

    def strip_pr_template_comments(text: str) -> str:
        if not text:
            return text
        # 去掉所有 HTML 注释块
        cleaned = re.sub(r'<!--.*?-->\s*', '', text, flags=re.DOTALL)
        # 去掉多余前后空行
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)
        return cleaned

    # 1. PR 基本信息
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    pr_data = _req(pr_url)
    if not pr_data:
        return {"pr_meta": {}, "conversation": []}

    # 清理 PR 模板注释
    pr_body_clean = strip_pr_template_comments(pr_data.get("body"))

    # 2. issue 评论
    issue_comments = _paginate(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments"
    )

    # 3. review 行级评论
    review_comments = _paginate(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/comments"
    )

    # 4. reviews (整体审核事件)
    reviews = _paginate(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/reviews"
    )

    conversation = []

    # PR body
    conversation.append({
        "type": "pr_body",
        "author": pr_data.get("user", {}).get("login") if pr_data.get("user") else None,
        "created_at": pr_data.get("created_at"),
        "body": pr_body_clean,
        "raw": pr_data,
        "extra": {
            "title": pr_data.get("title"),
            "state": pr_data.get("state"),
            "draft": pr_data.get("draft"),
            "merged": pr_data.get("merged"),
        }
    })

    # issue comments
    for c in issue_comments:
        conversation.append({
            "type": "issue_comment",
            "author": (c.get("user") or {}).get("login") if isinstance(c, dict) else None,
            "created_at": c.get("created_at") if isinstance(c, dict) else None,
            "body": c.get("body") if isinstance(c, dict) else None,
            "raw": c,
            "extra": {}
        })

    # reviews (一些 review 可能 body 为空)
    for r in reviews:
        conversation.append({
            "type": "review",
            "author": (r.get("user") or {}).get("login") if isinstance(r, dict) else None,
            "created_at": r.get("submitted_at") if isinstance(r, dict) else None,
            "body": r.get("body") if isinstance(r, dict) else None,
            "raw": r,
            "extra": {
                "state": r.get("state"),
                "commit_id": r.get("commit_id"),
            }
        })

    # review line comments
    for rc in review_comments:
        conversation.append({
            "type": "review_comment",
            "author": (rc.get("user") or {}).get("login") if isinstance(rc, dict) else None,
            "created_at": rc.get("created_at") if isinstance(rc, dict) else None,
            "body": rc.get("body") if isinstance(rc, dict) else None,
            "raw": rc,
            "extra": {
                "path": rc.get("path"),
                "position": rc.get("position"),
                "original_position": rc.get("original_position"),
                "diff_hunk": rc.get("diff_hunk"),
            }
        })

    # 排序
    def _ts(item):
        return item.get("created_at") or ""
    conversation.sort(key=_ts)
    return conversation