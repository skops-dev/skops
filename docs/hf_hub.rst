.. _hf_hub:

scikit-learn Models on Hugging Face Hub
=======================================

This library allows you to initialize and create a model repository compatible
with `Hugging Face Hub <https://huggingface.co/models>`__, which among other
things, gives you the following benefits:

- Inference API to get model output through REST calls
- A widget to try the model directly in the browser
- Metadata tags for better discoverability of the model
- Collaborating with others on a model through discussions and pull requests
- Convenient sharing of models with the community

You can see all the models uploaded to the Hugging Face Hub using this library
`here <https://huggingface.co/models?other=skops>`_.

In terms of files, there are three which a scikit-learn model repo needs to
have on the Hub:

- ``README.md``: includes certain metadata on top of the file and then a
  description of the model, aka model card.
- ``config.json``: contains the configuration needed to run the model.
- The persisted model file. There are no constraints on the name of the file
  and the name is configured in ``config.json``. The file needs to be loadable
  by :func:`joblib.load` or :func:`pickle.load`.

There are certain requirements in terms of information about the model for the
Hub to be able to load and run the model. For scikit-learn compatible models,
this information is stored in two places:

- The metadata in ``README.md`` of the model repository, about which you can
  read `here <https://huggingface.co/docs/hub/models-cards>`__.
- The configuration stored in ``config.json``.

As a user of ``skops``, you can use the tools in ``skops.hub_utils`` to create
and persist a ``config.json`` file, and then use it to populate necessary
metadata in the ``README.md`` file. The metadata in ``README.md`` is used by
the Hub's backend to understand the type of the model and the kind of task
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
  dictionary of column names to list of values, and is used by the Hugging Face
  Hub backend to show them in the widget to test the model when visiting the
  model's page on the Hub.
- ``environment``: A list of dependencies that the model requires. These
  packages must be available on conda-forge and are installed before loading
  the model.
- ``model.file``: The file name of the persisted model.
- ``task``: The task of the model.

You almost never need to create or touch this file manually, and it's created
when you call :func:`skops.hub_utils.init`.

It is recommended to include the script itself that creates the whole output in
the upload. This way, the results are easily reproducible for others. To achieve
this, call :func:`skops.hub_utils.add_files`:

.. code:: python

    # contents of train.py
    ...
    hub_utils.init(model, dst=local_repo)
    hub_utils.add_files(__file__, dst=local_repo)  # adds train.py to repo
    hub_utils.push(...)

You may of course add more files if they're useful.

.. _hf_hub_inference:

Inference without Downloading the Models
----------------------------------------

You can use the Hugging Face Hub's inference API to get model output without
downloading the models. The :func:`skops.hub_utils.get_model_output` function
returns the model output for a given input. It can be used as::

    import skops.hub_utils as hub_utils
    import pandas as pd
    data = pd.DataFrame(...)
    # Load the model from the Hub
    res = hub_utils.get_model_output("USER/MODEL_ID", data)

In the above code snippet, ``res`` will be a :class:`numpy.ndarray` containing
the model's output.

.. _hf_hub_gradio:
..
   TODO: replace gradio link once gradio provides object.inv
Easily build user interfaces to your scikit-learn models
--------------------------------------------------------
`gradio <https://gradio.app/>`__ is a python library that lets you create interfaces on your model.
It has a class called `Interface <https://gradio.app/docs/#interface>`__ that lets you create application
interfaces to your machine learning models. Using gradio can have some advantages over the using a plain
model repository, e.g. the Gradio dataframe component allows uploading a csv for tabular data, unlike the
widget in the model repository.

``gradio`` is integrated with skops, so you can load an interface with only one
line of code. During the initialization of the interface, call load method with
your repository identifier prepended with "huggingface/" will load an
interface for your model. The interface has a dataframe input that takes samples
and a dataframe output to return predictions. It also takes the example in the
repository that is previously pushed with skops.
Calling `gr.Interface.launch() <https://gradio.app/docs/#launch/>`__ will launch your application.

.. code:: python

    import gradio as gr
    repo_id = "scikit-learn/tabular-playground"
    gr.Interface.load(f"huggingface/{repo_id}").launch()


You can further customize your UI, add description, title, and more. If you'd
like to share your demo, you can set ``share`` to True in `gr.Interface.launch() <https://gradio.app/docs/#launch/>`__.

.. code:: python

    title = "Supersoaker Defective Product Prediction"
    description = ("This model predicts Supersoaker production line failures."
    "Drag and drop any slice from dataset or edit values as you wish in below"
    "dataframe component.")
    gr.Interface.load(f"huggingface/{repo_id}", title = title, description = description)

Sharing your local application has time limitations.
If you want to share your application continuously, you can deploy it to
Hugging Face Spaces. You can check out `this blog <https://huggingface.co/blog/gradio-spaces>`__
on how to do it.
For more information, please refer to documentation of `gradio <https://gradio.app/docs/>`__.

It's also possible to spawn a gradio space directly from the model repository.
To achieve this, from the model page, click on ``Deploy`` (top right corner) >
``Spaces`` > ``Create new Space``, then follow the instructions. After
finishing, you get a gradio space hosted on Hugging Face Hub, with all the
benefits that brings.
