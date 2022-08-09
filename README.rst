.. -*- mode: rst -*-

|readthedocs| |github-actions| |Codecov| |PyPi| |Black|

.. |readthedocs| image:: https://readthedocs.org/projects/skops/badge/?version=latest&style=flat
    :target: https://skops.readthedocs.io/en/latest/
    :alt: Documentation

.. |github-actions| image:: https://github.com/skops-dev/skops/workflows/pytest/badge.svg
    :target: https://github.com/skops-dev/skops/actions
    :alt: Linux, macOS, Windows tests

.. |Codecov| image:: https://codecov.io/gh/skops-dev/skops/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/skops-dev/skops
    :alt: Codecov

.. |PyPi| image:: https://img.shields.io/pypi/v/skops
    :target: https://pypi.org/project/skops
    :alt: PyPi

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black


SKOPS
=====

``skops`` is a Python library helping you share your `scikit-learn
<https://scikit-learn.org/stable/>`__ based models and put them in production.
At the moment, it includes the following features:

- ``skops.hub_utils``: tools to create a model repository to be stored on
  `Hugging Face Hub <https://hf.co/models>`__, mainly through
  ``skops.hub_utils.init`` and ``skops.hub_utils.push``.
- ``skops.card``: tools to create a model card explaining what the model does
  and how it should be used.

Please refer to our `documentation <https://skops.readthedocs.io/en/latest/>`_
on using the library as user, which includes user guides on the above topics as
well as complete examples explaining how the features can be used.

If you want to contribute to the library, please refer to our `contributing
<CONTRIBUTING.rst>`_ guidelines.

Installation
------------

You can install this library using:

    pip install skops

Bug Reports and Questions
-------------------------

Please send all your questions and report issues on this repository's issue
tracker as an issue. Try to look for existing ones before you create a new one.
