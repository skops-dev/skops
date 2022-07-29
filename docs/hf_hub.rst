.. _hf_hub:

HuggingFace Hub
===============

This library allows you to initialize and create a model repository compatible
with HuggingFace Hub, which you can push to the hub and call an API to get the
output of the model hosted on the hub.

In terms of files, there are three which a scikit-learn model repo needs to
have on the hub:

- ``README.md``: includes certain metadata on top of the file and then a
  description of the model, aka model card.
- ``config.json``: contains the configuration needed to run the model.
- The persisted model file. There are no constraints on the name of the file
  and the name is configured in ``config.json``. The file needs to be loadable
  by ``joblib`` or ``pickle``.

There are certain requirements in terms of information about the model for the
hub to be able to load and run the model. For scikit-learn compatible models,
this information is stored in two places:

- The metadata in ``README.md`` of the model repository, about which you can
  read `here <https://huggingface.co/docs/hub/models-cards>`__.
- The configuration stored in ``config.json``.

As a user of ``skops``, you can use the tools in ``skops.hub_utils`` to create
and persist a ``config.json`` file, and then use it to populate necessary
metadata in the ``README.md`` file. The metadata in ``README.md`` is used by
the hub's backend to understand the type of the model and the kind of task
which the model tries to solve. An example of a task can be
``"tabular-classification"`` or ``"text-regression"``.

An example ``config.json`` file looks like this::

    {
        "sklearn": {
            "columns": [
                "petal length (cm)",
                "petal width (cm)",
                "sepal length (cm)",
                "sepal width (cm)",
            ],
            "environment": ['scikit-learn="1.1.1"', "numpy"],
            "example_input": {
                "petal length (cm)": [1.4, 1.4, 1.3],
                "petal width (cm)": [0.2, 0.2, 0.2],
                "sepal length (cm)": [5.1, 4.9, 4.7],
                "sepal width (cm)": [3.5, 3.0, 3.2],
            },
            "model": {"file": "model.pkl"},
            "task": "tabular-classification",
        }
    }

The key ``sklearn`` includes the following sub-keys:

- ``columns``: An ordered list of column names. The order is important as it is
  used to make sure the input given to the model is what the model expects.
- ``example_input``: A list of examples to the model. This is in the form of a
  dictionary of column names to list of values, and is used by the HuggingFace
  Hub backend to show them in the widget to test the model when visiting the
  model's page on the hub.
- ``environment``: A list of dependencies that the model requires. These
  packages must be available on conda-forge and are installed before loading
  the model.
- ``model.file``: The file name of the persisted model.
- ``task``: The task of the model.

You almost never need to create or touch this file manually, and it's created
when you call :func:`skops.hub_utils.init`.
