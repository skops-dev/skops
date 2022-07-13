import re

from modelcards import EvalResult, ModelCard
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
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
        [docs](https://github.com/nateraw/modelcards/blob/7cb1c427a75c0a796631c137c5690ceedab3b2f8/modelcards/card_data.py#L78).
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
    importance = ""
    for i in importances.importances_mean.argsort()[::-1]:
        if importances.importances_mean[i] - 2 * importances.importances_std[i] > 0:
            importance += f"{X_test.columns[i]:<8}\n"
            importance += f"{importances.importances_mean[i]:.3f}"
            importance += f" +/- {importances.importances_std[i]:.3f}"
    if importance != "":
        importance = "Below are permutation importances:\n\n" + importance

    return importance


def evaluate(model, X_test, y_test, metric, dataset_type, dataset_name, task_type):
    """Evaluates the model and returns the score and the metric.
    Parameters:
    ----------
    model: estimator
        scikit-learn pipeline or model.
    X_test: pandas.core.series.Series or numpy.ndarray
        Split consisting of features for validation.
    y_test: pandas.core.series.Series or numpy.ndarray
        Split consisting of targets for validation.
    metric: scorer, str, or list of such values
        sklearn metric key or list of sklearn metric keys. See available list of
        metrics
        [here](https://scikit-learn.org/stable/modules/model_evaluation.html).
    dataset_type: str
        Type of dataset.
    dataset_name: str
        Name of dataset.
    task_type: str
        Task type. e.g. tabular-regression
    Returns:
    ----------
        eval_results: list List of ``EvalResult`` objects to be passed to ``CardData``.
    """
    metric_values = {}
    if isinstance(metric, str):
        scorer = get_scorer(metric)
        metric_values[metric] = float(scorer(model, X_test, y_test))

    elif isinstance(metric, list):
        for metric_key in metric:
            scorer = get_scorer(metric_key)
            metric_values[metric_key] = float(scorer(model, X_test, y_test))
    else:
        raise ValueError("Metric should be a metric key or list of metric keys.")

    eval_results = []
    for metric_key, metric_value in metric_values.items():
        eval_results.append(
            EvalResult(
                task_type=task_type,
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                metric_type=metric_key,
                metric_value=metric_value,
            )
        )

    return eval_results
