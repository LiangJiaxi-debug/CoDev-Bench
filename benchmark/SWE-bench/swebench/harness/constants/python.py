# Constants - Testing Commands
TEST_PYTEST = "pytest --no-header -rA --tb=no -p no:cacheprovider"
TEST_PYTEST_VERBOSE = "pytest -rA --tb=long -p no:cacheprovider"
TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
TEST_DJANGO_NO_PARALLEL = "./tests/runtests.py --verbosity 2"
TEST_SEABORN = "pytest --no-header -rA"
TEST_SEABORN_VERBOSE = "pytest -rA --tb=long"
TEST_PYTEST = "pytest -rA"
TEST_PYTEST_VERBOSE = "pytest -rA --tb=long"
TEST_SPHINX = "tox --current-env -epy39 -v --"
TEST_SYMPY = (
    "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
)
TEST_SYMPY_VERBOSE = "bin/test -C --verbose"


# Constants - Installation Specifications
SPECS_SKLEARN = {
    k: {
        "python": "3.10",
        "install": "python -m pip install --no-build-isolation -e .",
        "pip_packages": [
            "numpy>=1.22.0",
            "scipy>=1.8.0",
            "joblib>=1.2.0",
            "threadpoolctl>=3.1.0",
            "matplotlib>=3.5.0",
            "scikit-image>=0.19.0",
            "pandas>=1.4.0",
            "pytest>=7.1.2",
            "pytest-cov>=2.9.0",
            "ruff>=0.11.7",
            "mypy>=1.15",
            "pyamg>=4.2.1",
            "polars>=0.20.30",
            "pyarrow>=12.0.0",
            "numpydoc>=1.2.0",
            "pooch>=1.6.0",
            # 构建依赖已在 pre_install 安装, 可选保留
            "ninja",
            "pytest-twisted",
            "pytest-asyncio",
            "twisted",
            "cython>=3.0.0",
            "meson-python>=0.15.0",
            "pybind11>=2.10.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
            "python -m pip install -U numpy scipy cython pybind11 meson-python ninja",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.19"]
}


SPECS_FLASK = {
    "2.0": {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "setuptools==70.0.0",
            "Werkzeug==2.3.7",
            "Jinja2==3.0.1",
            "itsdangerous==2.1.2",
            "click==8.0.1",
            "MarkupSafe==2.1.3",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2.1": {
        "python": "3.10",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "setuptools==70.0.0",
            "click==8.1.3",
            "itsdangerous==2.1.2",
            "Jinja2==3.1.2",
            "MarkupSafe==2.1.1",
            "Werkzeug==2.3.7",
        ],
        "test_cmd": TEST_PYTEST,
    },
}
SPECS_FLASK.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": [
                "setuptools==70.0.0",
                "click==8.1.3",
                "itsdangerous==2.1.2",
                "Jinja2==3.1.2",
                "MarkupSafe==2.1.1",
                "Werkzeug==2.3.7",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["2.2", "2.3", "3.0", "3.1"]
    }
)

SPECS_REQUESTS = {
    k: {
        "python": "3.9",
        "packages": "pytest",
        "install": "python -m pip install .",
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.7", "0.8", "0.9", "0.11", "0.13", "0.14", "1.1", "1.2", "2.0", "2.2"]
    + ["2.3", "2.4", "2.5", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.17"]
    + ["2.18", "2.19", "2.22", "2.26", "2.25", "2.27", "2.31", "3.0"]
}

SPECS_SEABORN = {
    k: {
        "python": "3.8",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.20,!=1.24.0",
            "pandas>=1.2",
            "matplotlib>=3.4,!=3.6.1",
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "flake8",
            "mypy",
            "pandas-stubs",
            "pre-commit",
            "flit",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_SEABORN,
    }
    for k in ["0.10"]
}

SPECS_PYTEST = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "4.4",
        "4.5",
        "4.6",
        "5.0",
        "5.1",
        "5.2",
        "5.3",
        "5.4",
        "6.0",
        "6.2",
        "6.3",
        "7.0",
        "7.1",
        "7.2",
        "7.4",
        "8.0",
        "8.1",
        "8.2",
        "8.3",
        "8.4",
    ]
}
SPECS_PYTEST["4.4"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
]
SPECS_PYTEST["4.5"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.11.0",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["4.6"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
for k in ["5.0", "5.1", "5.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "atomicwrites==1.4.1",
        "attrs==23.1.0",
        "more-itertools==10.1.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "wcwidth==0.2.6",
    ]
SPECS_PYTEST["5.3"]["pip_packages"] = [
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["5.4"]["pip_packages"] = [
    "py==1.11.0",
    "packaging==23.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
]
SPECS_PYTEST["6.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "toml==0.10.2",
]
for k in ["6.2", "6.3"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "toml==0.10.2",
    ]
SPECS_PYTEST["7.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
]
for k in ["7.1", "7.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "tomli==2.0.1",
    ]
for k in ["7.4", "8.0", "8.1", "8.2", "8.3", "8.4"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==1.3.0",
        "exceptiongroup==1.1.3",
        "tomli==2.0.1",
    ]
SPECS_PYTEST["6.3"]["pre_install"] = ["sed -i 's/>=>=/>=/' setup.cfg"]



SPECS_SPHINX = {
    "8.1": {
        "python": "3.11",
        "pip_packages": ["tox==4.16.0", "tox-current-env==0.0.11", "Jinja2==3.0.3"],
        "install": "python -m pip install -e .[test]",
        "pre_install": [
            "sed -i 's/pytest/pytest -rA/' tox.ini",
            "apt-get update && apt-get install -y graphviz"
        ],
        "test_cmd": TEST_SPHINX,
    },
    "8.2": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "sphinxcontrib-applehelp>=1.0.7",
            "sphinxcontrib-devhelp>=1.0.6",
            "sphinxcontrib-htmlhelp>=2.0.6",
            "sphinxcontrib-jsmath>=1.0.1",
            "sphinxcontrib-qthelp>=1.0.6",
            "sphinxcontrib-serializinghtml>=1.1.9",
            "Jinja2>=3.1",
            "Pygments>=2.17",
            "docutils>=0.20,<0.23",
            "snowballstemmer>=2.2",
            "babel>=2.13",
            "alabaster>=0.7.14",
            "imagesize>=1.3",
            "requests>=2.30.0",
            "roman-numerals-py>=1.0.0",
            "packaging>=23.0",
            "colorama>=0.4.6; sys_platform == 'win32'",
            "pytest>=8.0",
            "pytest-xdist[psutil]>=3.4",
            "cython>=3.0",
            "defusedxml>=0.7.1",
            "setuptools>=70.0",
            "typing_extensions>=4.9",
            "tox==4.16.0",
            "tox-current-env==0.0.11",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_SPHINX,
    }
}

SPECS_ASTROPY = {
    k: {
        "python": "3.11",
        # 禁用构建隔离，防止隔离环境拉取 NumPy 2.x
        "install": "python -m pip install -e .[test] --verbose --no-build-isolation",
        "pip_packages": [
            # 来自 test / 运行期常见依赖 (统一 pin)
            "numpy==1.26.4",
            "pytest==8.0.0",
            "pytest-xdist==3.6.1",
            "pytest-astropy==0.10.0",
            "pytest-astropy-header==0.2.2",
            "pytest-doctestplus==1.4.0",
            "coverage==7.4.4",
            "threadpoolctl==3.5.0",
            # 通用/基础依赖
            "attrs==23.2.0",
            "packaging==24.1",
            "pluggy==1.5.0",
            "iniconfig==2.0.0",
            "exceptiongroup==1.2.0",
            "psutil==5.9.8",
            "pyerfa==2.0.1.0",
            "sortedcontainers==2.4.0",
            "hypothesis==6.100.0",
            "execnet==2.1.1",
            "PyYAML==6.0.1",
            "setuptools==70.0.0",
            "tomli==2.0.1",
        ],
        "test_cmd": TEST_PYTEST,
        "pre_install": [
            # 安装 toml 解析器，用于稳定修改 pyproject.toml
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade tomlkit",

            # 兼容旧 pep621 校验：移除 [project].license-files，并将 license 规范化为仅 text
            "python - <<'PY'\nfrom pathlib import Path\nfrom tomlkit import parse, dumps, inline_table\npp=Path('pyproject.toml')\ndata=parse(pp.read_text())\nproj=data.get('project')\nif proj is None:\n    raise SystemExit('no [project] table in pyproject.toml')\n# 某些 setuptools/校验器不接受 license-files（PEP 639），直接移除\nif 'license-files' in proj:\n    del proj['license-files']\n    print('removed [project].license-files')\nlic=inline_table(); lic['text']='BSD-3-Clause'\nproj['license']=lic\npp.write_text(dumps(data))\nprint('patched [project].license -> {text=\"BSD-3-Clause\"}')\nPY",

            # 在 build-system 中加入 numpy<2，避免隔离环境拉取 2.x
            "python - <<'PY'\nfrom pathlib import Path\nimport re\npp=Path('pyproject.toml')\ns=pp.read_text()\ns=re.sub(r'requires = \\[(\"setuptools[^\"]*\".*?)\\]', lambda m: 'requires = ['+m.group(1)+', \"numpy<2\", \"wheel\"]', s)\npp.write_text(s)\nprint('patched build-system requires with numpy<2')\nPY",

            # 预装构建期所需的 numpy 头文件与工具
            "python -m pip install --upgrade wheel",
            "python -m pip install --upgrade \"numpy==1.26.4\" cython extension-helpers",
        ],
    }
    for k in ["7.1", "7.2"]
}

SPECS_PVLIB = {
    k: {
        "python": "3.11",
        "install": "python -m pip install -e .[test] --verbose",
        "pip_packages": [
            # 运行期依赖（按 pyproject dependencies，固定版本以保持可重复性）
            "numpy==1.26.4",
            "pandas==2.2.2",
            "pytz==2024.1",
            "requests==2.32.3",
            "scipy==1.11.4",
            "h5py==3.11.0",
            # 测试依赖（project.optional-dependencies.test）
            "pytest==8.0.0",
            "pytest-cov==4.1.0",
            "pytest-mock==3.12.0",
            "requests-mock==1.11.0",
            "pytest-timeout==2.2.0",
            "pytest-rerunfailures==13.0",
            "pytest-remotedata==0.4.1",
            "packaging==24.1",
            # 统一基础
            "coverage==7.4.4",
            "threadpoolctl==3.5.0",
            "attrs==23.2.0",
            "pluggy==1.5.0",
            "iniconfig==2.0.0",
            "exceptiongroup==1.2.0",
            "psutil==5.9.8",
            "sortedcontainers==2.4.0",
            "hypothesis==6.100.0",
            "execnet==2.1.1",
            "PyYAML==6.0.1",
            "setuptools==70.1.0",
            "setuptools_scm==8.1.0",
            "wheel==0.42.0",
            "tomli==2.0.1",
        ],
        "test_cmd": TEST_PYTEST,
        "pre_install": [
            'bash -c "command -v apt-get >/dev/null && (apt-get update && apt-get install -y ca-certificates) || true"',
            "python -m pip install --upgrade pip",
            "python -m pip install setuptools==70.1.0 setuptools_scm==8.1.0 wheel==0.42.0 packaging==24.1",
        ],
    }
    for k in ["0.11", "0.12", "0.13"]
}

SPECS_CONAN = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest","bottle","mock","webtest","jwt"],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.21","2.20","2.17","2.16","2.12"]
}

SPECS_FAKER = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest","freezegun"],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["37.6","37.1","37.0","35.2"]
}

SPECS_FALCON = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
            "python -m pip install 'cython>=3.0.8'",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["4.1","4.0"]
}

SPECS_HAYSTACK = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "pytest-asyncio",
            "pytest-bdd",
            "pytest-cov",
            "pytest-rerunfailures",
            "coverage",
            "mypy",
            "pylint",
            "transformers",
            "pdfminer.six",
            "nltk",
            "huggingface-hub",
            "azure-ai-formrecognizer",
            "ddtrace",
            "pandas",
            "arrow",
            "torch",
            "openapi3",
            "structlog",
            "colorama",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install 'hatchling>=1.8.0'",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.19", "2.18", "2.17", "2.16", "2.15", "2.14", "2.13", "2.12", "2.11", "2.10","2.9"]
}

SPECS_HYDRA = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "pre_install": [
            "python -m pip install --upgrade pip",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.3"]
}

SPECS_JOBLIB = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install 'setuptools>=61.2'",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.4"]
}

SPECS_KAFKA_PYTHON = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "pytest-mock",
            "pytest-timeout",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install 'setuptools>=61.2'",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.2", "2.1"]
}

SPECS_LM_EVAL_HARNESS = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.4"]
}

SPECS_MONAI = {
    "1.5": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "torch>=2.4.1; platform_system != 'Windows'",
            "torch>=2.4.1, !=2.7.0; platform_system == 'Windows'",
            "numpy>=1.24,<3.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.4": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "torch>=2.4.1,<2.7.0",
            "numpy>=1.24,<3.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_MTEB = {
    "1.39": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest>=8.3.4,<8.4.0",
            "pytest-xdist>=3.6.1,<3.7.0",
            "pytest-coverage>=0.0",
            "pytest-rerunfailures>=15.0,<16.0",
            "iso639>=0.1.4",
            "GitPython>=3.0.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.38": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest>=8.3.4,<8.4.0",
            "pytest-xdist>=3.6.1,<3.7.0",
            "pytest-coverage>=0.0",
            "pytest-rerunfailures>=15.0,<16.0",
            "iso639>=0.1.4",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.36": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest>=8.3.4,<8.4.0",
            "pytest-xdist>=3.6.1,<3.7.0",
            "pytest-coverage>=0.0",
            "pytest-rerunfailures>=15.0,<16.0",
            "iso639>=0.1.4",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.34": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest>=8.3.4,<8.4.0",
            "pytest-xdist>=3.6.1,<3.7.0",
            "pytest-coverage>=0.0",
            "pytest-rerunfailures>=15.0,<16.0",
            "iso639>=0.1.4",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_ACCELERATE = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "ruff>=0.0.282",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.13"]
}

SPECS_BIOREGISTRY = {
    "0.12": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "coverage",
            "more_itertools",
            "httpx",
            "scikit-learn",
            "rdflib",
            "pandas",
            "pyyaml",
            "fastapi",
            "a2wsgi",
            "defusedxml",
            "flask",
            "flask_bootstrap",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "0.11": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "coverage",
            "more_itertools",
            "httpx",
            "indra",
            "scikit-learn",
            "rdflib",
            "pandas",
            "pyyaml",
            "fastapi",
            "a2wsgi",
            "defusedxml",
            "flask",
            "flask_bootstrap",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_BOTOCORE = {
    k: {
        "python": "3.12",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "wheel==0.43.0",
            "behave==1.2.5",
            "jsonschema==4.23.0",
            "coverage==7.2.7",
            "setuptools==71.1.0",
            "packaging==24.1",
            "pytest==8.1.1",
            "pytest-cov==5.0.0",
            "pytest-xdist==3.5.0",
            "atomicwrites>=1.0",  # Windows requirement
            "colorama>0.3.0",     # Windows requirement
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.17"]
}

SPECS_COCOTB = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "matplotlib",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.0", "1.4"]
}

SPECS_DATASETS = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "pytest-xdist",
            "numpy",
            "pandas",
            "pyarrow",
            "requests",
            "dill",
            "xxhash",
            "fsspec",
            "decorator",
            "h5py",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["4.1", "4.0", "3.6", "3.5", "3.2"]
}

SPECS_DATUMARO = {
    "1.12": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest>=5.3.5",
            "pytest-cov>=4.0.0",
            "pytest-stress",
            "pytest-html",
            "coverage",
            "pytest-csv",
            "dill",
            "tifffile",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.11": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "attrs>=21.3.0",
            "defusedxml>=0.7.0",
            "h5py>=2.10.0",
            "imagesize>=1.4.1",
            "lxml==6.0.0",  # 原 lxml>=5.2.0,<7
            "matplotlib>=3.3.1",
            "networkx>=2.6",
            "nibabel>=3.2.1",
            "numpy==1.24.0",  # 原 numpy>=1.23.4,<2.2.0
            "orjson>=3.10.0",
            "Pillow>=10.3.0",
            "ruamel.yaml>=0.17.0",
            "shapely>=1.7",
            "typing_extensions>=3.7.4.3",
            "tqdm",
            "pycocotools>=2.0.4",
            "PyYAML==6.0.2",
            "tensorboardX==2.4.1",  # 原 tensorboardX>=1.8,!=2.3
            "scipy",
            "requests",
            "pandas==1.4.0",  # 原 pandas>=1.4.0
            "openvino==2023.2.0",  # 原 openvino>=2023.2.0
            "tokenizers",
            "cryptography==38.0.3",  # 原 cryptography>=38.0.3
            "pyarrow",
            "protobuf",
            "tabulate",
            "ovmsclient",
            "tritonclient[all]",
            "scikit-learn",
            "json-stream",
            "nltk",
            "portalocker",
            "dvc==3.49.0",
            "GitPython==3.1.18",  # 原 GitPython>=3.1.18,!=3.1.25
            "openvino-telemetry>=2022.1.0",  # 原 openvino-telemetry>=2022.1.0
            "opencv-python-headless==4.11.0.86",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "1.10": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "attrs>=21.3.0",
            "defusedxml>=0.7.0",
            "h5py>=2.10.0",
            "imagesize>=1.4.1",
            "lxml==5.2.0",  # 原 lxml>=5.2.0,<6
            "matplotlib>=3.3.1",
            "networkx>=2.6",
            "nibabel>=3.2.1",
            "numpy==1.24.0",  # 原 numpy>=1.23.4,<2
            "orjson==3.10.18",
            "Pillow>=10.3.0",
            "ruamel.yaml>=0.17.0",
            "shapely>=1.7",
            "typing_extensions>=3.7.4.3",
            "tqdm",
            "pycocotools>=2.0.4",
            "PyYAML==6.0.2",
            "tensorboardX==2.4.1",  # 原 tensorboardX>=1.8,!=2.3
            "scipy",
            "requests",
            "pandas==1.4.0",  # 原 pandas>=1.4.0
            "openvino==2023.2.0",  # 原 openvino>=2023.2.0
            "tokenizers",
            "cryptography==38.0.3",  # 原 cryptography>=38.0.3
            "pyarrow",
            "protobuf",
            "tabulate",
            "ovmsclient",
            "tritonclient[all]",
            "scikit-learn",
            "json-stream",
            "nltk",
            "portalocker",
            "dvc==3.49.0",
            "fsspec==2022.11.0",  # 原 fsspec<=2022.11.0; python_version < '3.8'
            "GitPython==3.1.18",  # 原 GitPython>=3.1.18,!=3.1.25
            "openvino-telemetry>=2022.1.0",  # 原 openvino-telemetry>=2022.1.0
            "openvino-dev>=2023.2.0",  # 原 openvino-dev>=2023.2.0
            "opencv-python-headless==4.11.0.86",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_DRF_SPECTACULAR = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "Django>=2.2",
            "djangorestframework>=3.10.3",
            "uritemplate>=2.0.0",
            "PyYAML>=5.1",
            "jsonschema>=2.6.0",
            "inflection>=0.3.1",
            "typing-extensions",
            "pytest>=5.3.5",
            "pytest-django>=3.8.0",
            "pytest-cov>=2.8.1",
            "flake8",
            "isort==5.12.0",
            "mypy==1.7.1",
            "django-stubs==4.2.3",
            "djangorestframework-stubs==3.14.2",
            "django-allauth<0.55.0",
            "drf-jwt>=0.13.0",
            "dj-rest-auth>=1.0.0",
            "djangorestframework-simplejwt>=4.4.0",
            "setuptools",
            "django-polymorphic>=2.1",
            "django-rest-polymorphic>=0.1.8",
            "django-oauth-toolkit>=1.2.0",
            "djangorestframework-camel-case>=1.1.2",
            "django-filter>=2.3.0",
            "psycopg2-binary>=2.7.3.2",
            "drf-nested-routers>=0.93.3",
            "djangorestframework-recursive>=0.1.2",
            "drf-spectacular-sidecar",
            "djangorestframework-dataclasses>=1.0.0",
            "djangorestframework-gis>=1.0.0",
            "pydantic>=2,<3",
            "django-rest-knox>=4.1",
            "twine>=3.1.1",
            "wheel>=0.34.2",
            "Sphinx>=4.1.0",
            "furo",
            "typing-extensions",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.28"]
}

SPECS_GRADIO = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "aiofiles>=22.0,<25.0",
            "anyio>=3.0,<5.0",
            "brotli>=1.1.0",
            "fastapi>=0.115.2,<1.0",
            "ffmpy",
            "groovy~=0.1",
            "gradio_client==2.0.0-dev.0",
            "httpx>=0.24.1,<1.0",
            "huggingface_hub>=0.33.5,<2.0",
            "Jinja2<4.0",
            "markupsafe>=2.0,<4.0",
            "numpy>=1.0,<3.0",
            "orjson~=3.0",
            "packaging",
            "pandas>=1.0,<3.0",
            "pillow>=8.0,<12.0",
            "pydantic>=2.0,<2.12",
            "python-multipart>=0.0.18",
            "pydub",
            "pyyaml>=5.0,<7.0",
            "safehttpx>=0.1.6,<0.2.0",
            "semantic_version~=2.0",
            "starlette>=0.40.0,<1.0",
            "tomlkit>=0.12.0,<0.14.0",
            "typer>=0.12,<1.0",
            "typing_extensions~=4.0",
            "uvicorn>=0.14.0",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.10"]
}

SPECS_H2 = {
    "4.3": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "hyperframe>=6.1,<7",
            "hpack>=4.1,<5",
            "pytest>=8.3.3,<9",
            "pytest-cov>=6.0.0,<7",
            "pytest-xdist>=3.6.1,<4",
            "hypothesis>=6.119.4,<7",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_HATCH = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "click>=8.0.6",
            "hatchling>=1.26.3",
            "httpx>=0.22.0",
            "hyperlink>=21.0.0",
            "keyring>=23.5.0",
            "packaging>=23.2",
            "pexpect~=4.8",
            "platformdirs>=2.5.0",
            "rich>=11.2.0",
            "shellingham>=1.4.0",
            "tomli-w>=1.0",
            "tomlkit>=0.11.1",
            "userpath~=1.7",
            "uv>=0.5.23",
            "virtualenv>=20.26.6",
            "zstandard<1",
            "pytest",
            "pytest-cov",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.15"]
}

SPECS_HUGGINGFACE_HUB = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "requests",
            "tqdm",
            "filelock",
            "packaging",
            "pyyaml",
            "typing-extensions",
            "fsspec",
            "numpy",
            "pytest",
            "pytest-xdist",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.0", "0.9"]
}

SPECS_MATPLOTLIB = {
    "3.10": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "contourpy>=1.0.1",
            "cycler>=0.10",
            "fonttools>=4.22.0",
            "kiwisolver>=1.3.1",
            "numpy>=1.25",
            "packaging>=20.0",
            "pillow>=9",
            "pyparsing>=3",
            "python-dateutil>=2.7",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_PGMPY = {
    "1.0": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "networkx>=3.0",
            "numpy>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.2",
            "pandas>=1.5",
            "statsmodels>=0.14.5",
            "tqdm>=4.64",
            "pyparsing>=3.0",
            "joblib>=1.2",
            "opt_einsum>=3.3",
            "scikit-base>=0.12.4",
            "xdoctest>=0.11.0",
            "pytest>=3.3.1",
            "pytest-cov",
            "pytest-xdist",
            "coverage>=4.3.4",
            "mock",
            "black",
            "pre-commit",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "0.1": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "networkx>=3.0",
            "numpy>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.2",
            "pandas>=1.5",
            "torch>=2.5",
            "statsmodels>=0.13",
            "tqdm>=4.64",
            "joblib>=1.2",
            "opt_einsum>=3.3",
            "pyro-ppl>=1.9.1",
            "xdoctest>=0.11.0",
            "pytest>=3.3.1",
            "pytest-cov",
            "pytest-xdist",
            "coverage>=4.3.4",
            "mock",
            "black",
            "pre-commit",
            "daft-pgm>=0.1.4",
            "xgboost>=2.0.3",
            "litellm==1.61.15",
            "pyparsing>=3.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
}

SPECS_PYOCD = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.39"]
}

SPECS_PYTHAINLP = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "PyYAML>=5.4.1",
            "numpy>=1.22",
            "pyicu>=2.3",
            "python-crfsuite>=0.9.7",
            "requests>=2.31",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["5.1"]
}

SPECS_BIGQUERY = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "google-api-core[grpc]>=2.11.1,<3.0.0",
            "google-auth>=2.14.1,<3.0.0",
            "google-cloud-core>=2.4.1,<3.0.0",
            "google-resumable-media>=2.0.0,<3.0.0",
            "packaging>=24.2.0",
            "python-dateutil>=2.8.2,<3.0.0",
            "requests>=2.21.0,<3.0.0",
            "pytest",
            "freezegun",
            "test_utils",
            "pandas"
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["3.38", "3.36", "3.35","3.29","3.32","3.34"]
}

SPECS_PYTORCH_IMAGE_MODELS = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "torch>=1.7",
            "torchvision",
            "pyyaml",
            "huggingface_hub>=0.17.0",
            "safetensors>=0.2",
            "numpy",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.0"]
}

SPECS_RDFLIB = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "isodate>=0.7.2,<1.0.0",
            "pyparsing>=3.2.0,<4",
            "pytest>=7.1.3,<9.0.0",
            "pytest-cov>=4,<7",
            "coverage>=7.8.2",
            "types-setuptools>=68.0.0.3,<72.0.0.0",
            "setuptools>=68,<72",
            "wheel>=0.42,<0.46",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["7.1"]
}

SPECS_SCANPY = {
    k: {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "anndata>=0.9.2",
            "numpy>=1.25.2",
            "fast-array-utils[accel,sparse]>=1.2.1",
            "matplotlib>=3.7.5",
            "pandas>=2.0.3",
            "scipy>=1.11.1,<1.16.0",
            "seaborn>=0.13.2",
            "h5py>=3.8.0",
            "tqdm",
            "scikit-learn>=1.1.3",
            "statsmodels>=0.14.4",
            "patsy!=1.0.0",
            "networkx>=2.8.8",
            "natsort",
            "joblib",
            "numba>=0.58.1",
            "umap-learn>=0.5.7",
            "pynndescent>=0.5.13",
            "packaging>=21.3",
            "session-info2",
            "legacy-api-wrap>=1.4.1",
            "typing-extensions",
            "pytest>=8.2",
            "pytest-mock",
            "pytest-cov",
            "pytest-xdist[psutil]",
            "pytest-randomly",
            "pytest-rerunfailures",
            "tuna",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["1.11"]
}

SPECS_SCRAPY = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "Twisted>=21.7.0",
            "cryptography>=37.0.0",
            "cssselect>=0.9.1",
            "defusedxml>=0.7.1",
            "itemadapter>=0.1.0",
            "itemloaders>=1.0.1",
            "lxml>=4.6.0",
            "packaging",
            "parsel>=1.5.0",
            "protego>=0.1.15",
            "pyOpenSSL>=22.0.0",
            "queuelib>=1.4.2",
            "service_identity>=18.1.0",
            "tldextract",
            "w3lib>=1.17.0",
            "zope.interface>=5.1.0",
            'PyDispatcher>=2.0.5',
            'PyPyDispatcher>=2.1.0',
            "pytest",
            "testfixtures",
            "pexpect",
            "uvloop",
            "pytest-asyncio",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.13", "2.12"]
}

SPECS_SNOWFLAKE_CONNECTOR = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "setuptools>=40.6.0",
            "wheel",
            "cython",
            "pytest",
            "parameterized",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["3.12", "3.17","3.14"]
}

SPECS_SOFTLAYER = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["6.2"]
}

SPECS_SYMPY = {
    k: {
            "python": "3.9",
            "packages": "mpmath flake8",
            "pip_packages": [
                "mpmath",
                "pytest",
                "pytest-xdist",
                "pytest-timeout",
                "pytest-split",
                "pytest-doctestplus",
                "hypothesis",
                "flake8",
                "flake8-comprehensions",
            ],
            "install": "python -m pip install -e .",
            "test_cmd": TEST_SYMPY,
        }
    for k in ["1.13", "1.14"]
}

SPECS_PYLINT = {
    k: {
        "python": "3.10",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
        "pip_packages": ["astroid==4.0.1", "setuptools"],
        "nano_cpus": int(2e9),
    }
    for k in ["3.3", "4.0"]
}

SPECS_XARRAY = {
    "2025.10": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.26",
            "packaging>=24.1",
            "pandas>=2.2",
            "hypothesis",
            "jinja2",
            "mypy==1.18.1",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-mypy-plugins",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-asyncio",
            "ruff>=0.8.0",
            "sphinx",
            "sphinx_autosummary_accessors",
            "scipy",
            "netCDF4>=1.7.0",
            "cftime",
            "dask",
            "zarr",
            "netcdf",
            "fsspec",
            "pydap.client",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2025.09": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.26",
            "packaging>=24.1",
            "pandas>=2.2",
            "hypothesis",
            "jinja2",
            "mypy==1.18.1",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-mypy-plugins",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-asyncio",
            "ruff>=0.8.0",
            "sphinx",
            "sphinx_autosummary_accessors",
            "scipy",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2025.08": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.26",
            "packaging>=24.1",
            "pandas>=2.2",
            "hypothesis",
            "jinja2",
            "mypy",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-mypy-plugins",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-asyncio",
            "ruff>=0.8.0",
            "sphinx",
            "sphinx_autosummary_accessors",
            "scipy",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2025.06": {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.24",
            "packaging>=23.2",
            "pandas>=2.1",
            "hypothesis",
            "jinja2",
            "mypy",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-mypy-plugins",
            "pytest-timeout",
            "pytest-xdist",
            "ruff>=0.8.0",
            "sphinx",
            "sphinx_autosummary_accessors",
            "scipy",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2025.07": {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.26",
            "packaging>=24.1",
            "pandas>=2.2",
            "hypothesis",
            "jinja2",
            "mypy",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-mypy-plugins",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-asyncio",
            "ruff>=0.8.0",
            "sphinx",
            "sphinx_autosummary_accessors",
            "scipy",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
}

SPECS_SQLFLUFF = {
    "3.5": {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "platformdirs",
            "chardet",
            "click<=8.3.0",
            "colorama>=0.3",
            "diff-cover>=2.5.0",
            "Jinja2",
            "pathspec",
            "pytest",
            "pyyaml>=5.1",
            "regex",
            "tblib",
            "tomli",
            "tqdm",
            "hypothesis",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "3.4": {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "platformdirs",
            "chardet",
            "click<=8.3.0",
            "colorama>=0.3",
            "diff-cover>=2.5.0",
            "Jinja2",
            "pathspec",
            "pytest",
            "pyyaml>=5.1",
            "regex",
            "tblib",
            "tomli",
            "tqdm",
            "hypothesis",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "3.3": {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "platformdirs",
            "chardet",
            "click",
            "colorama>=0.3",
            "diff-cover>=2.5.0",
            "Jinja2",
            "pathspec",
            "pytest",
            "pyyaml>=5.1",
            "regex",
            "tblib",
            "toml",
            "tqdm",
            "hypothesis",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
}

SPECS_SQLGLOT = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "27.28", "27.27", "27.21", "27.20", "27.19", "27.17", "27.16", "27.14",
        "27.13", "27.11", "27.8", "27.6", "27.3", "27.2", "27.1", "27.0", "26.33", "26.32","26.30", "26.29", "26.28", "26.26", "26.26", "26.25", "26.24", "26.18", "26.17"
    ]
}

SPECS_STABLE_BASELINES3 = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pytest",
            "ale_py"
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["2.7","2.6","2.5"]
}

SPECS_STATSMODELS = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy>=1.22.3,<3",
            "scipy>=1.8,!=1.9.2",
            "pandas>=1.4,!=2.1.0",
            "patsy>=0.5.6",
            "packaging>=21.3",
            "formulaic>=1.1.0",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.15"]
}

SPECS_TEXTUAL = {
    k: {
        "python": "3.8",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "markdown-it-py[plugins,linkify]>=2.1.0",
            "rich>=13.3.3",
            "typing-extensions>=4.4.0",
            "platformdirs>=3.6.0,<5",
            "pygments>=2.19.2",
            "pytest>=8.3.1",
            "pytest-xdist>=3.6.1",
            "pytest-asyncio",
            "pytest-cov>=5.0.0",
            "textual-dev>=1.7.0",
            "types-setuptools>=67.2.0.1",
            "isort>=5.13.2",
            "pytest-textual-snapshot>=1.0.0",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["5.2", "4.0","3.2","3.0","1.0"]
}

SPECS_TORNADO = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "alabaster==0.7.16",
            "babel==2.15.0",
            "black==24.4.2",
            "build==1.2.1",
            "cachetools==5.5.2",
            "certifi==2024.7.4",
            "chardet==5.2.0",
            "charset-normalizer==3.3.2",
            "click==8.1.7",
            "colorama==0.4.6",
            "distlib==0.3.8",
            "docutils==0.20.1",
            "filelock==3.18.0",
            "flake8==7.0.0",
            "idna==3.7",
            "imagesize==1.4.1",
            "jinja2==3.1.6",
            "markupsafe==2.1.5",
            "mccabe==0.7.0",
            "mypy==1.10.0",
            "mypy-extensions==1.0.0",
            "packaging==25.0",
            "pathspec==0.12.1",
            "pip-tools==7.4.1",
            "platformdirs==4.3.8",
            "pluggy==1.5.0",
            "pycodestyle==2.11.1",
            "pyflakes==3.2.0",
            "pygments==2.18.0",
            "pyproject-api==1.9.1",
            "pyproject-hooks==1.1.0",
            "requests==2.32.3",
            "snowballstemmer==2.2.0",
            "sphinx==7.3.7",
            "sphinx-rtd-theme==2.0.0",
            "sphinxcontrib-applehelp==1.0.8",
            "sphinxcontrib-devhelp==1.0.6",
            "sphinxcontrib-htmlhelp==2.0.5",
            "sphinxcontrib-jquery==4.1",
            "sphinxcontrib-jsmath==1.0.1",
            "sphinxcontrib-qthelp==1.0.7",
            "sphinxcontrib-serializinghtml==1.1.10",
            "tox==4.26.0",
            "types-pycurl==7.45.3.20240421",
            "typing-extensions==4.12.2",
            "urllib3==2.2.2",
            "virtualenv==20.31.2",
            "wheel==0.43.0",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["6.5","6.4"]
}

SPECS_WAGTAIL = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "Django>=4.1",
            "django-modelcluster>=6.2.1,<7.0",
            "django-permissionedforms>=0.1,<1.0",
            "django-taggit>=5.0,<7",
            "django-treebeard>=4.5.1,<5.0",
            "djangorestframework>=3.15.1,<4.0",
            "django-filter>=23.3",
            "draftjs_exporter>=2.1.5,<6.0",
            "Pillow>=9.1.0,<12.0.0",
            "beautifulsoup4>=4.8,<5",
            "Willow[heif]>=1.11.0,<2",
            "requests>=2.11.1,<3.0",
            "openpyxl>=3.0.10,<4.0",
            "anyascii>=0.1.5",
            "telepath>=0.3.1,<1",
            "laces>=0.1,<0.2",
            "django-tasks>=0.8,<0.9",
            "python-dateutil>=2.7",
            "Jinja2>=3.0,<3.2",
            "boto3>=1.28,<2",
            "freezegun>=0.3.8",
            "azure-mgmt-cdn>=12.0,<13.0",
            "azure-mgmt-frontdoor>=1.0,<1.1",
            "django-pattern-library>=0.7",
            "responses>=0.25,<1",
            "coverage>=3.7.0",
            "doc8==1.1.2",
            "ruff==0.9.6",
            "semgrep==1.132.0",
            "curlylint==0.13.1",
            "djhtml==3.0.6",
            "polib>=1.1,<2.0",
            "factory-boy>=3.2",
            "tblib>=2.0,<3.0",
            "pytest",
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["4.1"]
}

SPECS_TORTOISE_ORM = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "pypika-tortoise>=0.5.0,<1.0.0",
            "iso8601>=2.1.0,<3.0.0",
            "aiosqlite>=0.16.0,<1.0.0",
            "pytz",
            "pytest",
            "pytest-xdist",
            "pytest-cov",
            "pytest-codspeed",
            "pytest-asyncio>=0.24.0",
            "pydantic",
            "sanic_testing",
            "sanic"
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.24"]
}

SPECS_TRL = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "accelerate>=1.4.0",
            "datasets>=3.0.0",
            "transformers>=4.56.1",
            "pytest",
            "parameterized"
        ],
        "pre_install": [
            "python -m pip install --upgrade pip",
            "python -m pip install --upgrade setuptools wheel",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.23"]
}

SPECS_PYVISTA = {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST,
            "pre_install": [
                "apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender1"
            ],
        }
        for k in ["0.27"]
}


SPECS_ASTROID = {
    k: {
        "python": "3.10",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest", "typing-extensions>=4"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["4.0"]
}

SPECS_PYDICOM = {
    k: {
        "python": "3.11",
        "install": "python -m pip install -e .",
        "packages": "numpy",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["3.0",]
}

SPECS_HUMANEVAL = {k: {"python": "3.9", "test_cmd": "python"} for k in ["1.0"]}

# Constants - Task Instance Instllation Environment
MAP_REPO_VERSION_TO_SPECS_PY = {
    "astropy/astropy": SPECS_ASTROPY,
    "biopragmatics/bioregistry": SPECS_BIOREGISTRY,
    "boto/botocore": SPECS_BOTOCORE,
    "conan-io/conan" : SPECS_CONAN,
    "cocotb/cocotb": SPECS_COCOTB,
    "deepset-ai/haystack": SPECS_HAYSTACK,
    "dpkp/kafka-python": SPECS_KAFKA_PYTHON,
    "DLR-RM/stable-baselines3": SPECS_STABLE_BASELINES3,
    "EleutherAI/lm-evaluation-harness": SPECS_LM_EVAL_HARNESS,
    "embeddings-benchmark/mteb": SPECS_MTEB,
    "facebookresearch/hydra": SPECS_HYDRA,
    "falconry/falcon": SPECS_FALCON,
    "gradio-app/gradio": SPECS_GRADIO,
    "googleapis/python-bigquery": SPECS_BIGQUERY,
    "huggingface/accelerate": SPECS_ACCELERATE,
    "huggingface/datasets": SPECS_DATASETS,
    "huggingface/huggingface_hub": SPECS_HUGGINGFACE_HUB,
    "huggingface/pytorch-image-models": SPECS_PYTORCH_IMAGE_MODELS,
    "huggingface/trl": SPECS_TRL,
    "joke2k/faker": SPECS_FAKER,
    "joblib/joblib": SPECS_JOBLIB,
    "matplotlib/matplotlib": SPECS_MATPLOTLIB,
    "mwaskom/seaborn": SPECS_SEABORN,
    "open-edge-platform/datumaro": SPECS_DATUMARO,
    "Project-MONAI/MONAI": SPECS_MONAI,
    "pypa/hatch": SPECS_HATCH,
    "PyThaiNLP/pythainlp": SPECS_PYTHAINLP,
    "pallets/flask": SPECS_FLASK,
    "pgmpy/pgmpy": SPECS_PGMPY,
    "python-hyper/h2": SPECS_H2,
    "pyocd/pyOCD": SPECS_PYOCD,
    "psf/requests": SPECS_REQUESTS,
    "pvlib/pvlib-python": SPECS_PVLIB,
    "pydata/xarray": SPECS_XARRAY,
    "pydicom/pydicom": SPECS_PYDICOM,
    "pylint-dev/astroid": SPECS_ASTROID,
    "pylint-dev/pylint": SPECS_PYLINT,
    "pytest-dev/pytest": SPECS_PYTEST,
    "pyvista/pyvista": SPECS_PYVISTA,
    "RDFLib/rdflib": SPECS_RDFLIB,
    "scikit-learn/scikit-learn": SPECS_SKLEARN,
    "sphinx-doc/sphinx": SPECS_SPHINX,
    "sqlfluff/sqlfluff": SPECS_SQLFLUFF,
    "swe-bench/humaneval": SPECS_HUMANEVAL,
    "sympy/sympy": SPECS_SYMPY,
    "scverse/scanpy": SPECS_SCANPY,
    "scrapy/scrapy": SPECS_SCRAPY,
    "snowflakedb/snowflake-connector-python": SPECS_SNOWFLAKE_CONNECTOR,
    "softlayer/softlayer-python": SPECS_SOFTLAYER,
    "tfranzel/drf-spectacular": SPECS_DRF_SPECTACULAR,
    "statsmodels/statsmodels": SPECS_STATSMODELS,
    "tobymao/sqlglot": SPECS_SQLGLOT,
    "Textualize/textual": SPECS_TEXTUAL,
    "tornadoweb/tornado": SPECS_TORNADO,
    "tortoise/tortoise-orm": SPECS_TORTOISE_ORM,
    "wagtail/wagtail": SPECS_WAGTAIL,
}

# Constants - Repository Specific Installation Instructions
MAP_REPO_TO_INSTALL_PY = {}


# Constants - Task Instance Requirements File Paths
MAP_REPO_TO_REQS_PATHS = {
    "dbt-labs/dbt-core": ["dev-requirements.txt", "dev_requirements.txt"],
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": [
        "requirements/dev/dev-requirements.txt",
        "requirements/testing/travis_all.txt",
    ],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt", "requirements-test.txt"],
    "astropy/astropy": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pvlib/pvlib-python": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pylint-dev/astroid": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "conan-io/conan": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "joke2k/faker": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "falconry/falcon": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "deepset-ai/haystack": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "facebookresearch/hydra": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "joblib/joblib": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "dpkp/kafka-python": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "EleutherAI/lm-evaluation-harness": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "Project-MONAI/MONAI": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "embeddings-benchmark/mteb": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "huggingface/accelerate": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "biopragmatics/bioregistry": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "boto/botocore": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "cocotb/cocotb": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "huggingface/datasets": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "open-edge-platform/datumaro": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "tfranzel/drf-spectacular": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "gradio-app/gradio": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "python-hyper/h2": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pypa/hatch": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "huggingface/huggingface_hub": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pgmpy/pgmpy": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pydicom/pydicom": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pyocd/pyOCD": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "PyThaiNLP/pythainlp": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "googleapis/python-bigquery": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "huggingface/pytorch-image-models": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "RDFLib/rdflib": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "scverse/scanpy": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "scikit-learn/scikit-learn": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "scrapy/scrapy": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "mwaskom/seaborn": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "snowflakedb/snowflake-connector-python": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "softlayer/softlayer-python": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "sphinx-doc/sphinx": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "tobymao/sqlglot": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "DLR-RM/stable-baselines3": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "statsmodels/statsmodels": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "Textualize/textual": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "tornadoweb/tornado": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "tortoise/tortoise-orm": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "huggingface/trl": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "wagtail/wagtail": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
    "pydata/xarray": ["requirements.txt","pyproject.toml","setup.cfg","docs/requirements.txt",],
}


# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_ENV_YML_PATHS = {}

USE_X86_PY = {}
