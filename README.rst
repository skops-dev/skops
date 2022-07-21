SKOPS
-----

This library helps you share your scikit-learn based models and put them in
production.

THIS LIBRARY IS NOT READY TO BE USED.

Sharing via HuggingFace Hub
===========================

Get model output via HuggingFace inference API
==============================================

Demo a model using HuggingFace Spaces
=====================================

Deploy a model using Seldon
===========================

DEVELOPMENT
===========

Setting up the dev environment
==============================

Follow these steps if you want to contribute to the skops development.

Using conda
-----------

.. code:: bash

          conda create -c conda-forge -n skops python=3.10
          conda activate skops
          python -m pip install -e ".[tests,docs]"
          # add pre-commit hooks
          conda install -c conda-forge pre-commit
          pre-commit install

Releases
========

Releases are created using `manual GitHub workflows <https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow>`_. Follow these steps:

1. Create a new branch
2. Bump the version defined in ``skops/__init__.py``
3. Git grep for any TODO's that need fixing before the release (e.g. deprecations)
4. Update the ``CHANGES.md``
5. Create a PR with all the changes and have it reviewed and merged
6. Use the GitHub action to create a new release on **TestPyPI**. Check it for correctness `on test.pypi <https://test.pypi.org/project/skops/>`_.
7. Use the GitHub action to create a new release on **PyPI**. Check it for correctness `pypi <https://pypi.org/project/skops/>`_.
8. Create a `new release <https://github.com/skops-dev/skops/releases>`_ on GitHub
9. Update the patch version of the package to a new dev version, e.g. from ``v0.3.0`` to ``v0.3.dev1``
10. Check that the new stable branch of documentation was built correctly on `readthedocs <https://readthedocs.org/projects/skops/builds/>`_
