.. _model_cards:

Model Cards for scikit-learn
============================

This library allows you to automatically create documentation for models, also
known as model cards. The model cards consist of two parts, a metadata part for
model discoverability and markdown part for information related to model. The
card itself is named as ``README.md``.

The metadata part of the file needs to follow the notation `here
<https://huggingface.co/docs/hub/models-cards#model-card-metadata>`__. It
includes simple attributes of your models such as the task you're solving,
dataset you trained the model with, evaluation results and more. The tasks have
keys, such as ``"tabular-classification"`` or ``"text-regression"``. When the model
is hosted on the Hub, information like task and dataset help your model be
discovered at the `Hugging Face Hub <https://huggingface.co/models>`__ and the
evaluation results stored in the metadata is automatically pushed to the
leaderboard of the task and the dataset on `Papers with Code
<paperswithcode.com>`__.

Metadata part looks like below:
.. code-block:: yaml
    ---
    tags:
    - tabular-classification
    license: mit
    datasets:
    - breast-cancer
    metrics:
    - accuracy
    ---

The markdown part of the model card can include:

- Simple description of the model,
- Intended use for the model, limitations and biases,
- Metrics of the model,
- Plots describing model and it's performance,
- Information related to training process, e.g. hyperparameters.

Model cards are based on templates with slots that you can pass information to.
``skops`` has a default template called ``default_template.md``. Each
information you ``add`` to the model card needs to be matching with the slots in
the template. This doesn't apply for plots. If you want a custom template, you can
pass the ``template_path`` with ``add``.
