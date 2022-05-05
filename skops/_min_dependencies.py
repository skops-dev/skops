"""All minimum dependencies for scikit-learn."""
import argparse

PYTEST_MIN_VERSION = "5.0.1"

# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma seperated tags)
# tags can be: 'build', 'install', 'docs', 'examples', 'tests', 'benchmark'
dependent_packages = {
    "scikit-learn": ("0.24", "install"),
    "huggingface_hub": ("0.5", "install"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "mypy": ("0.770", "tests"),
    "sphinx": ("3.2.0", "docs"),
    "sphinx-gallery": ("0.7.0", "docs"),
    "sphinx-rtd-theme": ("1", "docs"),
    "numpydoc": ("1.0.0", "docs"),
    "sphinx-prompt": ("1.3.0", "docs"),
    "matplotlib": ("3.3", "docs"),
    "pandas": ("1", "docs"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: []
    for extra in ["build", "install", "docs", "examples", "tests", "benchmark"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
