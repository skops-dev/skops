.. _changelog:

skops Changelog
===============

.. contents:: Table of Contents
    :depth: 2
    :local:

v0.1
----

This is the first release of the library. It include two main modules:

- ``skops.hub_utils``: tools to create a model repository to be stored on
  `Hugging Face Hub <https://hf.co/models>`__, mainly through
  ``skops.hub_utils.init`` and ``skops.hub_utils.push``.
- ``skops.card``: tools to create a model card explaining what the model does
  and how it should be used. The model card can then be stored as the
  ``README.md`` file on the Hugging Face Hub, with pre-populated metadata to
  help Hub understand the model.
