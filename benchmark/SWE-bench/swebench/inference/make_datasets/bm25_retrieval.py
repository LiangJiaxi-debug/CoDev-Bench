import json
import os
import ast
import jedi
import shutil
import traceback
import subprocess
from filelock import FileLock
from typing import Any
from datasets import load_from_disk, load_dataset
# from pyserini.search.lucene import LuceneSearcher
from git import Repo
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from datasets import DatasetDict
import sys

from swebench.inference.make_datasets.utils import list_files, string_to_bool

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_jvm_configured = False
_pyserini_imported = False
def _select_java_home():
    if not os.environ.get("JAVA_HOME"):
        for cand in [
            "/usr/lib/jvm/java-21-openjdk-amd64",
            "/usr/lib/jvm/java-21-openjdk",
            "/usr/lib/jvm/jdk-21",
            "/usr/lib/jvm/java-21",
        ]:
            if os.path.isdir(cand):
                os.environ["JAVA_HOME"] = cand
                break
    if os.environ.get("JAVA_HOME"):
        os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ.get("PATH","")

def _clean_java_env_vars():
    for v in ["JAVA_TOOL_OPTIONS","_JAVA_OPTIONS","JAVA_OPTS","PY4J_JAVA_OPTIONS","JDK_JAVA_OPTIONS"]:
        val = os.environ.get(v)
        if not val:
            continue
        if "jdk.incubator.vector" in val or "--add-modules" in val:
            del os.environ[v]

def _java_has_vector_module():
    jp = shutil.which("java")
    if not jp:
        return False
    try:
        r = subprocess.run([jp, "--list-modules"], capture_output=True, text=True, timeout=6)
        return "jdk.incubator.vector" in (r.stdout or "")
    except Exception:
        return False

def _configure_jvm_before_pyserini():
    """
    必须在 import pyserini 前执行。使用 pyjnius 的 jnius_config 设置 JVM 选项，避免错误 flag。
    幂等：多次调用只首次生效；若 JVM 已启动则跳过。
    """
    global _jvm_configured
    if _jvm_configured:
        return
    _select_java_home()
    _clean_java_env_vars()
    has_vector = _java_has_vector_module()
    try:
        import jnius_config
        if "jnius" in sys.modules:
            logging.getLogger(__name__).info("[JVM-CONFIG] JVM 已运行，跳过 set_options。")
            _jvm_configured = True
            return
        opts = [
            "-Xms512m",
            "-Xmx4g",
        ]
        if has_vector and os.environ.get("ENABLE_VECTOR_MODULE") == "1":
            opts += ["--add-modules", "jdk.incubator.vector"]
        try:
            jnius_config.set_options(*opts)
            logging.getLogger(__name__).info(f"[JVM-CONFIG] options={opts} JAVA_HOME={os.environ.get('JAVA_HOME')} has_vector={has_vector}")
        except ValueError as e:
            # JVM 已经启动的典型报错
            logging.getLogger(__name__).warning(f"[JVM-CONFIG] 跳过: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[JVM-CONFIG] 设置选项失败: {e}")
    except ModuleNotFoundError:
        logging.getLogger(__name__).warning("[JVM-CONFIG] 未找到 jnius_config，将按 pyserini 默认行为加载 JVM")
    finally:
        _jvm_configured = True

_pyserini_imported = False
def ensure_pyserini_import():
    global _pyserini_imported
    if _pyserini_imported:
        return
    _configure_jvm_before_pyserini()
    from pyserini.search.lucene import LuceneSearcher   # noqa
    globals()["LuceneSearcher"] = LuceneSearcher
    _pyserini_imported = True
class ContextManager:
    """
    A context manager for managing a Git repository at a specific commit.

    Args:
        repo_path (str): The path to the Git repository.
        base_commit (str): The commit hash to switch to.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Attributes:
        repo_path (str): The path to the Git repository.
        base_commit (str): The commit hash to switch to.
        verbose (bool): Whether to print verbose output.
        repo (git.Repo): The Git repository object.

    Methods:
        __enter__(): Switches to the specified commit and returns the context manager object.
        get_readme_files(): Returns a list of filenames for all README files in the repository.
        __exit__(exc_type, exc_val, exc_tb): Does nothing.
    """

    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.base_commit = base_commit
        self.verbose = verbose
        self.repo = Repo(self.repo_path)

    def __enter__(self):
        if self.verbose:
            print(f"Switching to {self.base_commit}")
        try:
            self.repo.git.reset("--hard", self.base_commit)
            self.repo.git.clean("-fdxq")
        except Exception as e:
            logger.error(f"Failed to switch to {self.base_commit}")
            logger.error(e)
            raise e
        return self

    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def file_name_and_contents(filename, relative_path):
    text = relative_path + "\n"
    with open(filename) as f:
        text += f.read()
    return text


def file_name_and_documentation(filename, relative_path):
    text = relative_path + "\n"
    try:
        with open(filename) as f:
            node = ast.parse(f.read())
        data = ast.get_docstring(node)
        if data:
            text += f"{data}"
        for child_node in ast.walk(node):
            if isinstance(
                child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                data = ast.get_docstring(child_node)
                if data:
                    text += f"\n\n{child_node.name}\n{data}"
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        with open(filename) as f:
            text += f.read()
    return text


def file_name_and_docs_jedi(filename, relative_path):
    text = relative_path + "\n"
    with open(filename) as f:
        source_code = f.read()
    try:
        script = jedi.Script(source_code, path=filename)
        module = script.get_context()
        docstring = module.docstring()
        text += f"{module.full_name}\n"
        if docstring:
            text += f"{docstring}\n\n"
        abspath = Path(filename).absolute()
        names = [
            name
            for name in script.get_names(
                all_scopes=True, definitions=True, references=False
            )
            if not name.in_builtin_module()
        ]
        for name in names:
            try:
                origin = name.goto(follow_imports=True)[0]
                if origin.module_name != module.full_name:
                    continue
                if name.parent().full_name != module.full_name:
                    if name.type in {"statement", "param"}:
                        continue
                full_name = name.full_name
                text += f"{full_name}\n"
                docstring = name.docstring()
                if docstring:
                    text += f"{docstring}\n\n"
            except:
                continue
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        text = f"{relative_path}\n{source_code}"
        return text
    return text


DOCUMENT_ENCODING_FUNCTIONS = {
    "file_name_and_contents": file_name_and_contents,
    "file_name_and_documentation": file_name_and_documentation,
    "file_name_and_docs_jedi": file_name_and_docs_jedi,
}


def clone_repo(repo, root_dir, token):
    """
    Clones a GitHub repository to a specified directory.

    Args:
        repo (str): The GitHub repository to clone.
        root_dir (str): The root directory to clone the repository to.
        token (str): The GitHub personal access token to use for authentication.

    Returns:
        Path: The path to the cloned repository directory.
    """
    repo_dir = Path(root_dir, f"repo__{repo.replace('/', '__')}")

    if not repo_dir.exists():
        repo_url = f"https://{token}@github.com/{repo}.git"
        logger.info(f"Cloning {repo} {os.getpid()}")
        Repo.clone_from(repo_url, repo_dir)
    return repo_dir


def build_documents(repo_dir, commit, document_encoding_func):
    """
    Builds a dictionary of documents from a given repository directory and commit.

    Args:
        repo_dir (str): The path to the repository directory.
        commit (str): The commit hash to use.
        document_encoding_func (function): A function that takes a filename and a relative path and returns the encoded document text.

    Returns:
        dict: A dictionary where the keys are the relative paths of the documents and the values are the encoded document text.
    """
    documents = dict()
    with ContextManager(repo_dir, commit):
        filenames = list_files(repo_dir, include_tests=False)
        for relative_path in filenames:
            filename = os.path.join(repo_dir, relative_path)
            text = document_encoding_func(filename, relative_path)
            documents[relative_path] = text
    return documents


def make_index(
    repo_dir,
    root_dir,
    query,
    commit,
    document_encoding_func,
    python,
    instance_id,
):
    """
    Builds an index for a given set of documents using Pyserini.

    Note: documents.jsonl will be rewritten AFTER indexing so that top-k (k=5)
    documents by BM25 score appear first (high score first). This rewrite does
    NOT affect the already-built Lucene index; it's only for the jsonl order.
    """
    ensure_pyserini_import()
    index_path = Path(root_dir, f"index__{str(instance_id)}", "index")
    if index_path.exists():
        return index_path
    thread_prefix = f"(pid {os.getpid()}) "
    documents_path = Path(root_dir, instance_id, "documents.jsonl")
    if not documents_path.parent.exists():
        documents_path.parent.mkdir(parents=True)

    documents = build_documents(repo_dir, commit, document_encoding_func)

    # 先写一个确定性顺序（用于索引构建）；不依赖 BM25 分数
    with open(documents_path, "w") as docfile:
        for relative_path in sorted(documents.keys()):
            print(
                json.dumps({"id": relative_path, "contents": documents[relative_path]}),
                file=docfile,
                flush=True,
            )

    cmd = [
        python,
        "-m",
        "pyserini.index",
        "--collection",
        "JsonCollection",
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "2",
        "--input",
        documents_path.parent.as_posix(),
        "--index",
        index_path.as_posix(),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output, error = proc.communicate()
    except KeyboardInterrupt:
        proc.kill()
        raise KeyboardInterrupt
    if proc.returncode == 130:
        logger.warning(thread_prefix + "Process killed by user")
        raise KeyboardInterrupt
    if proc.returncode != 0:
        logger.error(f"return code: {proc.returncode}")
        raise Exception(
            thread_prefix
            + f"Failed to build index for {instance_id} with error {error}"
        )

    # === NEW: 索引建好后，把 top-k(=5) 命中置顶（按 score 排序），其余路径排序追加 ===
    TOPK = 5
    try:
        searcher = LuceneSearcher(index_path.as_posix())
        cutoff = len(query)
        while True:
            try:
                hits = searcher.search(query[:cutoff], k=TOPK, remove_dups=True)
            except Exception as e:
                if "maxClauseCount" in str(e):
                    cutoff = int(round(cutoff * 0.8))
                    continue
                raise
            break

        top_docids = [h.docid for h in hits]  # 已按 score 从高到低
        top_set = set(top_docids)
        tail_docids = sorted(set(documents.keys()) - top_set)
        write_docids = top_docids + tail_docids

        with open(documents_path, "w") as docfile:
            for relative_path in write_docids:
                print(
                    json.dumps(
                        {"id": relative_path, "contents": documents.get(relative_path, "")}
                    ),
                    file=docfile,
                    flush=True,
                )
    except Exception as e:
        logger.warning(f"[ORDER] failed to rewrite ranked documents for {instance_id}: {e}")

    return index_path


def get_remaining_instances(instances, output_file):
    """
    Filters a list of instances to exclude those that have already been processed and saved in a file.

    Args:
        instances (List[Dict]): A list of instances, where each instance is a dictionary with an "instance_id" key.
        output_file (Path): The path to the file where the processed instances are saved.

    Returns:
        List[Dict]: A list of instances that have not been processed yet.
    """
    instance_ids = set()
    remaining_instances = list()
    if output_file.exists():
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file) as f:
                for line in f:
                    instance = json.loads(line)
                    instance_id = instance["instance_id"]
                    instance_ids.add(instance_id)
            logger.warning(
                f"Found {len(instance_ids)} existing instances in {output_file}. Will skip them."
            )
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return instances
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in instance_ids:
            remaining_instances.append(instance)
    return remaining_instances


def search(instance, index_path):
    """
    Searches for relevant documents in the given index for the given instance.

    Args:
        instance (dict): The instance to search for.
        index_path (str): The path to the index to search in.

    Returns:
        dict: A dictionary containing the instance ID and a list of hits, where each hit is a dictionary containing the
        document ID and its score.
    """
    try:
        instance_id = instance["instance_id"]
        searcher = LuceneSearcher(index_path.as_posix())
        cutoff = len(instance["problem_statement"])
        while True:
            try:
                hits = searcher.search(
                    instance["problem_statement"][:cutoff],
                    k=20,
                    remove_dups=True,
                )
            except Exception as e:
                if "maxClauseCount" in str(e):
                    cutoff = int(round(cutoff * 0.8))
                    continue
                else:
                    raise e
            break
        results = {"instance_id": instance_id, "hits": []}
        for hit in hits:
            results["hits"].append({"docid": hit.docid, "score": hit.score})
        return results
    except Exception:
        logger.error(f"Failed to process {instance_id}")
        logger.error(traceback.format_exc())
        return None


def search_indexes(remaining_instance, output_file, all_index_paths):
    """
    Searches the indexes for the given instances and writes the results to the output file.

    Args:
        remaining_instance (list): A list of instances to search for.
        output_file (str): The path to the output file to write the results to.
        all_index_paths (dict): A dictionary mapping instance IDs to the paths of their indexes.
    """
    for instance in tqdm(remaining_instance, desc="Retrieving"):
        instance_id = instance["instance_id"]
        if instance_id not in all_index_paths:
            continue
        index_path = all_index_paths[instance_id]
        results = search(instance, index_path)
        if results is None:
            continue
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file, "a") as out_file:
                print(json.dumps(results), file=out_file, flush=True)


def get_missing_ids(instances, output_file):
    """
    返回还未写入 output_file 的实例 ID 集合。
    文件不存在时直接返回全部实例 ID。
    """
    all_ids = {inst["instance_id"] for inst in instances}
    if not os.path.exists(output_file):
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return all_ids
    written_ids = set()
    with open(output_file) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
                iid=obj.get("instance_id")
                if iid:
                    written_ids.add(iid)
            except Exception:
                continue
    return all_ids - written_ids


def get_index_paths_worker(
    instance,
    root_dir_name,
    document_encoding_func,
    python,
    token,
):
    index_path = None
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    try:
        repo_dir = clone_repo(repo, root_dir_name, token)
        query = instance["problem_statement"]
        index_path = make_index(
            repo_dir=repo_dir,
            root_dir=root_dir_name,
            query=query,
            commit=commit,
            document_encoding_func=document_encoding_func,
            python=python,
            instance_id=instance_id,
        )
    except:
        logger.error(f"Failed to process {repo}/{commit} (instance {instance_id})")
        logger.error(traceback.format_exc())
    return instance_id, index_path


def get_index_paths(
    remaining_instances: list[dict[str, Any]],
    root_dir_name: str,
    document_encoding_func: Any,
    python: str,
    token: str,
    output_file: str,
) -> dict[str, str]:
    """
    Retrieves the index paths for the given instances using multiple processes.

    Args:
        remaining_instances: A list of instances for which to retrieve the index paths.
        root_dir_name: The root directory name.
        document_encoding_func: A function for encoding documents.
        python: The path to the Python executable.
        token: The token to use for authentication.
        output_file: The output file.
        num_workers: The number of worker processes to use.

    Returns:
        A dictionary mapping instance IDs to index paths.
    """
    all_index_paths = dict()
    for instance in tqdm(remaining_instances, desc="Indexing"):
        instance_id, index_path = get_index_paths_worker(
            instance=instance,
            root_dir_name=root_dir_name,
            document_encoding_func=document_encoding_func,
            python=python,
            token=token,
        )
        if index_path is None:
            continue
        all_index_paths[instance_id] = index_path
    return all_index_paths


def get_root_dir(dataset_name, output_dir, document_encoding_style):
    root_dir = Path(output_dir, dataset_name, document_encoding_style + "_indexes")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    root_dir_name = root_dir
    return root_dir, root_dir_name

def _reorder_documents_jsonl_topk(documents_path: Path, top_docids: list[str], k: int = 5):
    """对已存在的 documents.jsonl 做 top-k 置顶（保持 top_docids 的顺序），其余保持原文件顺序。"""
    if not documents_path.exists():
        return

    # 读取原有行（保持原顺序）
    rows = []
    by_id = {}
    with open(documents_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            docid = obj.get("id")
            if not docid:
                continue
            rows.append(obj)
            by_id[docid] = obj

    if not rows:
        return

    top = []
    seen = set()
    for docid in top_docids[:k]:
        if docid in by_id and docid not in seen:
            top.append(by_id[docid])
            seen.add(docid)

    tail = [r for r in rows if r.get("id") not in seen]

    tmp_path = documents_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as wf:
        for obj in top + tail:
            wf.write(json.dumps(obj) + "\n")
    os.replace(tmp_path, documents_path)


def reorder_existing_documents_topk(root_dir: Path, retrieval_file: Path, k: int = 5):
    """
    使用 retrieval_file 里已计算好的 hits/score，对对应 instance 的 documents.jsonl 做 top-k 置顶。
    不重新检索、不重建索引。
    """
    if not retrieval_file.exists():
        logger.info(f"[REORDER] retrieval file not found, skip: {retrieval_file}")
        return

    updated = 0
    skipped = 0
    with open(retrieval_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instance_id = obj.get("instance_id")
            hits = obj.get("hits") or []
            if not instance_id or not isinstance(hits, list) or not hits:
                skipped += 1
                continue

            # hits 在 search() 里已按 score 降序 append，但这里稳妥再排序一次
            try:
                hits_sorted = sorted(
                    hits,
                    key=lambda x: float(x.get("score", float("-inf"))),
                    reverse=True,
                )
            except Exception:
                hits_sorted = hits

            top_docids = [h.get("docid") for h in hits_sorted if h.get("docid")]
            if not top_docids:
                skipped += 1
                continue

            documents_path = Path(root_dir, instance_id, "documents.jsonl")
            if not documents_path.exists():
                skipped += 1
                continue

            _reorder_documents_jsonl_topk(documents_path, top_docids, k=k)
            updated += 1

    logger.info(f"[REORDER] updated={updated} skipped={skipped} k={k}")

def main(
    dataset_name_or_path,
    document_encoding_style,
    output_dir,
    shard_id,
    num_shards,
    splits,
    leave_indexes,
):
    document_encoding_func = DOCUMENT_ENCODING_FUNCTIONS[document_encoding_style]
    token = os.environ.get("GITHUB_TOKEN", "git")
    p = Path(dataset_name_or_path)
    if p.exists():
        if p.is_file() and p.suffix in {".jsonl", ".json"}:
            # 以 json 数据集加载
            dataset = load_dataset("json", data_files={"data": p.as_posix()})
            dataset_name = p.stem
        else:
            # 认为是 HF save_to_disk 目录
            dataset = load_from_disk(dataset_name_or_path)
            dataset_name = os.path.basename(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
        dataset_name = dataset_name_or_path.replace("/", "__")
    # 允许 splits=None 自动推断；若传入 ["none"] 也视为 None
    if splits and len(splits) == 1 and str(splits[0]).lower() == "none":
        splits = None
    if shard_id is not None:
        if isinstance(dataset, DatasetDict):
            # 若 splits 为 None 先推断全部 key
            tmp_splits = splits if splits else list(dataset.keys())
            for split in tmp_splits:
                dataset[split] = dataset[split].shard(num_shards, shard_id)
        else:
            dataset = dataset.shard(num_shards, shard_id)
    instances = []
    if isinstance(dataset, DatasetDict):
        if not splits:
            splits = list(dataset.keys())  # 自动使用所有可用 split
        else:
            unknown = set(splits) - set(dataset.keys())
            if unknown:
                raise ValueError(f"Unknown splits {unknown}; available={list(dataset.keys())}")
        for split in splits:
            instances += list(dataset[split])
    else:
        if splits:
            logger.warning("提供的是单一 Dataset，忽略 --splits 参数。")
        instances = list(dataset)
    python = subprocess.run("which python", shell=True, capture_output=True)
    python = python.stdout.decode("utf-8").strip()
    output_file = Path(
        output_dir, dataset_name, document_encoding_style + ".retrieval.jsonl"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[OUTPUT] retrieval file: {output_file}")
    remaining_instances = get_remaining_instances(instances, output_file)
    root_dir, root_dir_name = get_root_dir(
        dataset_name, output_dir, document_encoding_style
    )
    try:
        all_index_paths = get_index_paths(
            remaining_instances,
            root_dir_name,
            document_encoding_func,
            python,
            token,
            output_file,
        )
    except KeyboardInterrupt:
        logger.info(f"Cleaning up {root_dir}")
        del_dirs = list(root_dir.glob("repo__*"))
        if leave_indexes:
            index_dirs = list(root_dir.glob("index__*"))
            del_dirs += index_dirs
        for dirname in del_dirs:
            shutil.rmtree(dirname, ignore_errors=True)
    logger.info(f"Finished indexing {len(all_index_paths)} instances")
    search_indexes(remaining_instances, output_file, all_index_paths)

    # === NEW: 对“已经存在 output_file 中的 instance”（以及本次新写入的）统一用已计算 hits 重排 documents.jsonl ===
    reorder_existing_documents_topk(root_dir, output_file, k=5)

    missing_ids = get_missing_ids(instances, output_file)
    logger.warning(f"Missing indexes for {len(missing_ids)} instances.")
    logger.info(f"Saved retrieval results to {output_file}")
    del_dirs = list(root_dir.glob("repo__*"))
    logger.info(f"Cleaning up {root_dir}")
    if leave_indexes:
        index_dirs = list(root_dir.glob("index__*"))
        del_dirs += index_dirs
    for dirname in del_dirs:
        shutil.rmtree(dirname, ignore_errors=True)
    logger.info(f"Finished indexing {len(all_index_paths)} instances")
    search_indexes(remaining_instances, output_file, all_index_paths)
    missing_ids = get_missing_ids(instances, output_file)
    logger.warning(f"Missing indexes for {len(missing_ids)} instances.")
    logger.info(f"Saved retrieval results to {output_file}")
    del_dirs = list(root_dir.glob("repo__*"))
    logger.info(f"Cleaning up {root_dir}")
    if leave_indexes:
        index_dirs = list(root_dir.glob("index__*"))
        del_dirs += index_dirs
    for dirname in del_dirs:
        shutil.rmtree(dirname, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Dataset to use for test set from HuggingFace Datasets or path to a save_to_disk directory.",
    )
    parser.add_argument(
        "--document_encoding_style",
        choices=DOCUMENT_ENCODING_FUNCTIONS.keys(),
        default="file_name_and_contents",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--shard_id", type=int)
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--leave_indexes", type=string_to_bool, default=True)
    args = parser.parse_args()
    main(**vars(args))
