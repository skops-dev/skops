import os
import tempfile
from pathlib import Path

from modelcards import CardData

from skops.card import create_model_card


def _get_cwd():
    """Return the current working directory.

    Only works if we're using pytest.
    """
    return Path(os.getenv("PYTEST_CURRENT_TEST").split("::")[0]).parent


def fit_model():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


def write_card():
    model = fit_model()
    card_data = CardData(library_name="sklearn")

    model_card = create_model_card(
        model,
        card_data,
        template_path="skops/skops/card/default_template.md",
        model_description="sklearn FTW",
    )
    return model_card


def test_write_model_card():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model = fit_model()
        model_card = create_model_card(model, model_description="sklearn FTW")
        model_card.save(f"{dir_path}/README.md")
        with open(f"{dir_path}/README.md", "r") as f:
            model_card = f.read()
        assert "sklearn FTW" in model_card


def test_hyperparameter_table():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = write_card()
        model_card.save(f"{dir_path}/README.md")
        with open(f"{dir_path}/README.md", "r") as f:
            model_card = f.read()
        assert "fit_intercept" in model_card


def test_plot_model():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = write_card()
        model_card.save(f"{dir_path}/README.md")
        with open(f"{dir_path}/README.md", "r") as f:
            model_card = f.read()
        assert "<style>" in model_card
