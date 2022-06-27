import os
import re

from modelcards import ModelCard
from sklearn.utils import estimator_html_repr

import skops


def _extract_estimator_config(model):
    """Extracts estimator configuration and renders them into a vertical table.

    Parameters
    ----------
        model (estimator): scikit-learn pipeline or model.

    Returns
    -------
    str:
        Markdown table of hyperparameters.
    """
    hyperparameter_dict = model.get_params(deep=True)
    table = "| Hyperparameters | Value |\n| :-- | :-- |\n"
    for hyperparameter, value in hyperparameter_dict.items():
        table += f"| {hyperparameter} | {value} |\n"
    return table


def create_model_card(
    model,
    card_data,
    **card_kwargs,
):
    """Creates a model card for the model and saves it to the target directory.

    Parameters:
    ----------
    model: estimator
        scikit-learn compatible estimator.
    card_data: CardData
        CardData object.
    card_kwargs:
        Card kwargs are information you can pass to fill in the sections of the
        card template, e.g. model_description, citation_bibtex, get_started_code.
    """
    ROOT = skops.__path__
    model_plot = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
    hyperparameter_table = _extract_estimator_config(model)
    card_data.library_name = "sklearn"
    template_path = card_kwargs.get("template_path")
    if template_path is None:
        template_path = os.path.join(f"{ROOT[0]}", "card", "default_template.md")
    card_kwargs["template_path"] = template_path
    card = ModelCard.from_template(
        card_data=card_data,
        hyperparameter_table=hyperparameter_table,
        model_plot=model_plot,
        **card_kwargs,
    )

    return card
