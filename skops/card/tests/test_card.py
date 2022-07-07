import os
import tempfile
from pathlib import Path

from modelcards import CardData, RepoCard

from skops.card import create_model_card, evaluate, permutation_importances


def _get_cwd():
    """Return the current working directory.

    Only works if we're using pytest.
    """
    return Path(os.getenv("PYTEST_CURRENT_TEST").split("::")[0]).parent


def fit_model():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = LinearRegression().fit(X_train, y_train)
    return reg, X_test, y_test


def write_card():
    model, _, _ = fit_model()
    card_data = CardData(library_name="sklearn")

    model_card = create_model_card(
        model=model,
        card_data=card_data,
    )
    return model_card


def test_write_model_card():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model, _, _ = fit_model()
        card_data = CardData(library_name="sklearn")
        model_card = create_model_card(
            model, card_data=card_data, model_description="sklearn FTW"
        )
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "sklearn FTW" in model_card


def test_hyperparameter_table():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = write_card()
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "fit_intercept" in model_card


def test_plot_model():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = write_card()
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "<style>" in model_card


def test_permutation_importances():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model, X_test, y_test = fit_model()
        importances = permutation_importances(model, X_test, y_test)
        card_data = CardData(library_name="sklearn")
        model_card = create_model_card(
            model, card_data, permutation_importances=importances
        )
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "Below are permutation importances:" in model_card


def test_evaluate():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model, X_test, y_test = fit_model()

        eval_results = evaluate(
            model,
            X_test,
            y_test,
            "r2",
            "random_type",
            "dummy_dataset",
            "tabular-regression",
        )
        card_data = CardData(eval_results=eval_results, model_name="my-cool-model")
        card = create_model_card(model, card_data)
        card.save(os.path.join(f"{dir_path}", "README.md"))
        loaded_card = RepoCard.load(os.path.join(f"{dir_path}", "README.md"))
        assert loaded_card.data.eval_results[0].task_type == "tabular-regression"
