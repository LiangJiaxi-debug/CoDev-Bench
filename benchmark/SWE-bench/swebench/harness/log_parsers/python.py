import re

from swebench.harness.constants import TestStatus
from swebench.harness.test_spec.test_spec import TestSpec


def parse_log_pytest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = (
                TestStatus.PASSED.value
            )

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    解析 Sympy/pytest 输出的 verbose 测试行:
    path/to/test_file.py::test_name STATUS [progress]

    返回示例:
    {
        "sympy/functions/elementary/tests/test_complexes.py::test_re": "PASSED",
        "sympy/functions/elementary/tests/test_complexes.py::test_sign_issue_3068": "FAILED",  # 原 XFAIL 归为 FAILED
    }
    """
    test_status_map: dict[str, str] = {}
    pattern = re.compile(r"^(?P<case>\S+\.py::\S+)\s+(?P<status>PASSED|FAILED|XFAIL|XPASS|SKIPPED|ERROR)\b")

    for line in log.splitlines():
        line = line.strip()
        m = pattern.match(line)
        if not m:
            continue
        case = m.group("case")
        status = m.group("status")

        if status == "XFAIL":
            # 将 XFAIL 视为 FAILED
            status_value = TestStatus.FAILED.value
        else:
            enum_value = getattr(TestStatus, status, None)
            status_value = enum_value.value if enum_value else status

        test_status_map[case] = status_value

    return test_status_map


def parse_log_matplotlib(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2
parse_log_matplotlib = parse_log_pytest_v2
parse_log_conan = parse_log_pytest_v2
parse_log_faker = parse_log_pytest_v2
parse_log_falcon = parse_log_pytest_v2
parse_log_haystack = parse_log_pytest_v2
parse_log_hydra = parse_log_pytest_v2
parse_log_joblib = parse_log_pytest_v2
parse_log_kafka_python = parse_log_pytest_v2
parse_log_lm_evaluation_harness = parse_log_pytest_v2
parse_log_monai = parse_log_pytest_v2
parse_log_mteb = parse_log_pytest_v2
parse_log_accelerate = parse_log_pytest_v2
parse_log_bioregistry = parse_log_pytest_v2
parse_log_botocore = parse_log_pytest_v2
parse_log_cocotb = parse_log_pytest_v2
parse_log_datasets = parse_log_pytest_v2
parse_log_datumaro = parse_log_pytest_v2
parse_log_drf_spectacular = parse_log_pytest_v2
parse_log_gradio = parse_log_pytest_v2
parse_log_h2 = parse_log_pytest_v2
parse_log_hatch = parse_log_pytest_v2
parse_log_huggingface_hub = parse_log_pytest_v2
parse_log_pgmpy = parse_log_pytest_v2
parse_log_pyOCD = parse_log_pytest_v2
parse_log_pythainlp = parse_log_pytest_v2
parse_log_python_bigquery = parse_log_pytest_v2
parse_log_pytorch_image_models = parse_log_pytest_v2
parse_log_rdflib = parse_log_pytest_v2
parse_log_scanpy = parse_log_pytest_v2
parse_log_scikit_learn = parse_log_pytest_v2
parse_log_scrapy = parse_log_pytest_v2
parse_log_seaborn = parse_log_pytest_v2
parse_log_snowflake_connector_python = parse_log_pytest_v2
parse_log_softlayer_python = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2
parse_log_sqlglot = parse_log_pytest_v2
parse_log_stable_baselines3 = parse_log_pytest_v2
parse_log_statsmodels = parse_log_pytest_v2
parse_log_textual = parse_log_pytest_v2
parse_log_tornado = parse_log_pytest_v2
parse_log_tortoise_orm = parse_log_pytest_v2
parse_log_trl = parse_log_pytest_v2
parse_log_wagtail = parse_log_pytest_v2


MAP_REPO_TO_PARSER_PY = {
    "matplotlib/matplotlib": parse_log_matplotlib,
    "pallets/flask": parse_log_flask,
    "pylint-dev/pylint": parse_log_pylint,
    "pyvista/pyvista": parse_log_pyvista,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sympy/sympy": parse_log_sympy,
    "astropy/astropy": parse_log_astropy,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pylint-dev/astroid": parse_log_astroid,
    "conan-io/conan": parse_log_conan,
    "joke2k/faker": parse_log_faker,
    "falconry/falcon": parse_log_falcon,
    "deepset-ai/haystack": parse_log_haystack,
    "facebookresearch/hydra": parse_log_hydra,
    "joblib/joblib": parse_log_joblib,
    "dpkp/kafka-python": parse_log_kafka_python,
    "EleutherAI/lm-evaluation-harness": parse_log_lm_evaluation_harness,
    "Project-MONAI/MONAI": parse_log_monai,
    "embeddings-benchmark/mteb": parse_log_mteb,
    "huggingface/accelerate": parse_log_accelerate,
    "biopragmatics/bioregistry": parse_log_bioregistry,
    "boto/botocore": parse_log_botocore,
    "cocotb/cocotb": parse_log_cocotb,
    "huggingface/datasets": parse_log_datasets,
    "open-edge-platform/datumaro": parse_log_datumaro,
    "tfranzel/drf-spectacular": parse_log_drf_spectacular,
    "gradio-app/gradio": parse_log_gradio,
    "python-hyper/h2": parse_log_h2,
    "pypa/hatch": parse_log_hatch,
    "huggingface/huggingface_hub": parse_log_huggingface_hub,
    "pgmpy/pgmpy": parse_log_pgmpy,
    "pydicom/pydicom": parse_log_pydicom,
    "pyocd/pyOCD": parse_log_pyOCD,
    "PyThaiNLP/pythainlp": parse_log_pythainlp,
    "googleapis/python-bigquery": parse_log_python_bigquery,
    "huggingface/pytorch-image-models": parse_log_pytorch_image_models,
    "RDFLib/rdflib": parse_log_rdflib,
    "scverse/scanpy": parse_log_scanpy,
    "scikit-learn/scikit-learn": parse_log_scikit_learn,
    "scrapy/scrapy": parse_log_scrapy,
    "mwaskom/seaborn": parse_log_seaborn,
    "snowflakedb/snowflake-connector-python": parse_log_snowflake_connector_python,
    "softlayer/softlayer-python": parse_log_softlayer_python,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "tobymao/sqlglot": parse_log_sqlglot,
    "DLR-RM/stable-baselines3": parse_log_stable_baselines3,
    "statsmodels/statsmodels": parse_log_statsmodels,
    "Textualize/textual": parse_log_textual,
    "tornadoweb/tornado": parse_log_tornado,
    "tortoise/tortoise-orm": parse_log_tortoise_orm,
    "huggingface/trl": parse_log_trl,
    "wagtail/wagtail": parse_log_wagtail,
    "pydata/xarray": parse_log_xarray,
}

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test parse_log_pytest on a pytest log file.")
    parser.add_argument("--repo", "-r", type=str, required=True, help="代码库名称, 例如: matplotlib/matplotlib")
    parser.add_argument("--path", "-p", type=str, required=True, help="路径: pytest 日志文件 (test_output.txt)")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()

    log_path = args.path
    repo_name = args.repo
    path_obj = Path(log_path)
    instance_id = path_obj.parent.name  # 倒数第二部分作为 instance_id

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # 根据仓库名选择对应的解析函数
    if repo_name in MAP_REPO_TO_PARSER_PY:
        parser_func = MAP_REPO_TO_PARSER_PY[repo_name]
        status_map = parser_func(log_content, None)  # type: ignore
    else:
        print(f"未找到对应的解析函数，使用默认的 parse_log_pytest_v2 进行解析。")
        status_map = parse_log_pytest_v2(log_content, None)  # type: ignore

    output_path = args.output
    record = {
        "instance_id": instance_id,
        "status_map": status_map,
    }
    with open(output_path, "a", encoding="utf-8") as out_f:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
