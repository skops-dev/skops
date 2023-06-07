.. -*- mode: rst -*-

|readthedocs| |github-actions| |Codecov| |PyPI| |Black|

.. |readthedocs| image:: https://readthedocs.org/projects/skops/badge/?version=latest&style=flat
    :target: https://skops.readthedocs.io/en/latest/
    :alt: Documentation

.. |github-actions| image:: https://github.com/skops-dev/skops/workflows/pytest/badge.svg
    :target: https://github.com/skops-dev/skops/actions
    :alt: Linux, macOS, Windows tests

.. |Codecov| image:: https://codecov.io/gh/skops-dev/skops/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/skops-dev/skops
    :alt: Codecov

.. |PyPI| image:: https://img.shields.io/pypi/v/skops
    :target: https://pypi.org/project/skops
    :alt: PyPi

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black

.. image:: https://raw.githubusercontent.com/skops-dev/skops/main/docs/images/logo.png
  :width: 500
  :target: https://skops.readthedocs.io/en/latest/

SKOPS
=====

``skops`` is a Python library helping you share your `scikit-learn
<https://scikit-learn.org/stable/>`__ based models and put them in production.
At the moment, it includes tools to easily integrate models on the Hugging Face
Hub, which allows you to share your models, make them discoverable, and use the
Hub's API inference and widgets to get outputs of the model without having to
download or load the model.

- ``skops.hub_utils``: tools to create a model repository to be stored on
  `Hugging Face Hub <https://hf.co/models>`__, mainly through
  ``skops.hub_utils.init`` and ``skops.hub_utils.push``. You can see all the
  models uploaded to the hub using this library `here
  <https://huggingface.co/models?other=skops>`__. Find out more `here
  <https://skops.readthedocs.io/en/stable/hf_hub.html>`__.
- ``skops.card``: tools to create a model card explaining what the model does
  and how it should be used. The model card can then be stored as the
  ``README.md`` file on the Hugging Face Hub, with pre-populated metadata to
  help Hub understand the model. More information can be found `here
  <https://skops.readthedocs.io/en/stable/model_card.html>`__.
- ``skops.io``: Secure persistence of sklearn estimators and more, without using
  ``pickle``. Visit `the docs
  <https://skops.readthedocs.io/en/latest/persistence.html>`__ for more
  information.

Please refer to our `documentation <https://skops.readthedocs.io/en/latest/>`_
on using the library as user, which includes user guides on the above topics as
well as complete examples explaining how the features can be used.

If you want to contribute to the library, please refer to our `contributing
<CONTRIBUTING.rst>`_ guidelines.

Installation
------------

You can install this library using:

.. code-block:: bash

    python -m pip install skops

Bug Reports and Questions
-------------------------

Please send all your questions and report issues on `this repository's issue
tracker <https://github.com/skops-dev/skops/issues>`_ as an issue. Try to look
for existing ones before you create a new one.
