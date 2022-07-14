#! /usr/bin/env python
# License: 3-clause BSD
import builtins

from setuptools import setup

# This is a bit (!) hackish: we are setting a global variable so that the
# main modelcard __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components.
builtins.__SKOPS_SETUP__ = True


import skops  # noqa
import skops._min_dependencies as min_deps  # noqa

VERSION = skops.__version__

DISTNAME = "skops"
DESCRIPTION = (
    "A set of tools to push scikit-learn based models to and pull from Hugging Face Hub"
)
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Adrin Jalali"
MAINTAINER_EMAIL = "adrin.jalali@gmail.com"
URL = "http://github.com/skops-dev/skops"
DOWNLOAD_URL = "https://pypi.org/project/skops/#files"
LICENSE = "MIT"
PROJECT_URLS = {
    "Bug Tracker": "http://github.com/skops-dev/skops/issues",
    "Documentation": "http://github.com/skops-dev/skops",
    "Source Code": "http://github.com/skops-dev/skops",
}


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 1 - Planning",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython",
        ],
        python_requires=">=3.7",
        install_requires=min_deps.tag_to_packages["install"],
        extras_require={"docs": min_deps.tag_to_packages["docs"]},
        include_package_data=True,
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
