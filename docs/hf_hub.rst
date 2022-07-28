.. _hf_hub:

HuggingFace Hub
===============

This library allows you to initialize and create a model repository compatible
with HuggingFace Hub, which you can push to the hub and call an API to get the
output of the model hosted on hte hub.

There are certain requirements in terms of information about the model for the
hub to be able to load and run the model. For scikit-learn compatible models,
this information is stored in two places:

- The metadata in ``README.md`` of the model repository, about which you can
  read `here <https://huggingface.co/docs/hub/models-cards>`__.
- The configuration stored in ``config.json``.

As a user of ``skops``, you can use the tools in the :py:mode:`skops.hub_utils`
to create and persist a ``config.json`` file, and then use it to populate
necessary metadata in the ``README.md`` file. The metadata in ``README.md`` is
used by the hub's backend to understand the type of the model and the kind of
task which the model tries to solve. An example of a task can be
``"tabular-classification"`` or ``"text-regression"``.
