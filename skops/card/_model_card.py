import re

from modelcards import ModelCard
from sklearn.inspection import permutation_importance
from sklearn.utils import estimator_html_repr


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
        scikit-learn pipeline or model.
    card_data: CardData
        CardData object. See the
        [docs](https://github.com/nateraw/modelcards/blob/main/modelcards/card_data.py#L76).
    card_kwargs:
        Card kwargs are information you can pass to fill in the sections of the
        card template, e.g. description of model
    """
    model_plot = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
    hyperparameter_table = _extract_estimator_config(model)
    card_data.library_name = "sklearn"
    template_path = card_kwargs.get("template_path")
    if template_path is None:
        template_path = "skops/card/default_template.md"
    card_kwargs["template_path"] = template_path
    card = ModelCard.from_template(
        card_data=card_data,
        hyperparameter_table=hyperparameter_table,
        model_plot=model_plot,
        **card_kwargs,
    )

    return card


def permutation_importances(model, X_test, y_test):
    importances = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=0
    )
    imp = "Below are permutation importances:\n\n"
    for i in importances.importances_mean.argsort()[::-1]:
        if importances.importances_mean[i] - 2 * importances.importances_std[i] > 0:
            imp += f"{X_test.columns[i]:<8}\n"
            imp += f"{importances.importances_mean[i]:.3f}"
            imp += f" +/- {importances.importances_std[i]:.3f}"
    return imp
