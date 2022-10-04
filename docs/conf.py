# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from packaging.version import parse

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import skops
import subprocess
import inspect
from operator import attrgetter
import os
import sys
from functools import partial



# -- Project information -----------------------------------------------------

project = "skops"
copyright = "2022, Adrin Jalali"
author = "Adrin Jalali"


# The full version, including alpha/beta/rc tags

parsed_version = parse(skops.__version__)
release = ".".join(parsed_version.base_version.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx.ext.intersphinx",  # link to other documentations, e.g. sklearn
]

autodoc_default_options = {"members": True, "inherited-members": True}
autodoc_typehints = "none"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

# sphinx-issues configuration
# Path to GitHub repo {group}/{project}
# (note that `group` is the GitHub user or organization)
issues_github_path = "skops-dev/skops"




def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/skops-dev/skops/blob/main/skops/hub_utils/_hf_hub.py"



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "images/logo.png"
html_theme_options = {
    "logo_only": True,
}

# See:
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "huggingface_hub": ("https://huggingface.co/docs/huggingface_hub/main/en", None),
}
