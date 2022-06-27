from modelcards import CardData, ModelCard
from sklearn import set_config
from sklearn.utils import estimator_html_repr


def _plot_estimator(model):
    set_config(display="diagram")
    return estimator_html_repr(model)


def _extract_estimator_config(model):
    """Extracts estimator configuration and renders them into a vertical table.

    Args:
        model (estimator): scikit-learn pipeline or model.

    Returns:
        str: Markdown table of hyperparameters.
    """
    hyperparameter_dict = model.get_params(deep=True)
    table = "| Hyperparameters | Value |\n| :-- | :-- |\n"
    for hyperparameter, value in hyperparameter_dict.items():
        table += f"| {hyperparameter} | {value} |\n"
    return table


def create_model_card(model, path, model_id, license, tags, metrics):
    """Creates a model card for the model and saves it to the target directory.

    Args:
        model (estimator): scikit-learn pipeline or model.
        path (Path): Path to repository that the model card is generated in.
        model_id (str): Hugging Face Hub ID.
        license (str): Model license.
        tags (list): Tags about the model. e.g. tabular-classification
        metrics (list): Metrics model is evaluated on.
    """

    model_plot = _plot_estimator(model)
    hyperparameter_table = _extract_estimator_config(model)
    card = ModelCard.from_template(
        card_data=CardData(
            license=license,
            library_name="sklearn",
            tags=tags,
            metrics=metrics,
        ),
        template_path="default_template.md",
        model_id=model_id,
        hyperparameter_table=hyperparameter_table,
        model_plot=model_plot,
    )

    with open(path, "w+", encoding="utf-8") as f:
        f.write(card)
