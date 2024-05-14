"""All minimum dependencies for scikit-learn."""
import argparse

# TODO: flaky fails on pytest=8
PYTEST_MIN_VERSION = "7"

# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags, condition)
# tags can be: 'build', 'install', 'docs', 'examples', 'tests', 'benchmark',
# 'rich'
# example:
#     "tomli": ("1.1.0", "install", "python_full_version < '3.11.0a7'"),
dependent_packages = {
    "scikit-learn": ("0.24", "install", None),
    "huggingface_hub": ("0.17.0", "install", None),
    "tabulate": ("0.8.8", "install", None),
    "quantile-forest": ("1.0.0", "tests", None),
    "pytest": (PYTEST_MIN_VERSION, "tests", None),
    "pytest-cov": ("2.9.0", "tests", None),
    "flake8": ("3.8.2", "tests", None),
    "types-requests": ("2.28.5", "tests", None),
    "flaky": ("3.7.0", "tests", None),
    "sphinx": ("3.2.0", "docs", None),
    "sphinx-gallery": ("0.7.0", "docs", None),
    "sphinx-rtd-theme": ("1", "docs", None),
    "numpydoc": ("1.0.0", "docs", None),
    "sphinx-prompt": ("1.3.0", "docs", None),
    "sphinx-issues": ("1.2.0", "docs", None),
    "matplotlib": ("3.3", "docs, tests", None),
    "packaging": ("17.0", "install", None),
    "pandas": ("1", "docs, tests", None),
    # required for persistence tests of external libraries
    "lightgbm": ("3", "tests", None),
    "xgboost": ("1.6", "tests", None),
    "catboost": ("1.0", "tests", None),
    "fairlearn": ("0.7.0", "docs, tests", None),
    "rich": ("12", "tests, rich", None),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: []
    for extra in ["build", "install", "docs", "examples", "tests", "benchmark", "rich"]
}
for package, (min_version, extras, condition) in dependent_packages.items():
    for extra in extras.split(", "):
        spec = f"{package}>={min_version}"
        if condition:
            spec += f"; {condition}"
        tag_to_packages[extra].append(spec)


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
