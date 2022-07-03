import os
import tempfile
from pathlib import Path

from skops import card


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
    model_card = card.create_model_card(model, model_description="sklearn FTW")
    return model_card


def test_write_model_card():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model = fit_model()
        model_card = card.create_model_card(model, model_description="sklearn FTW")
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


def test_hyperparameter_table_search():
    """
    from sklearn.metrics import r2_score, make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeRegressor
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [1, 2]},
    scoring={"R_square": make_scorer(r2_score)},
    refit="R_square",
    n_jobs=1,
    )
    gs.fit(X, y)
    """
    pass


def test_plot_model():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = write_card()
        model_card.save(f"{dir_path}/README.md")
        with open(f"{dir_path}/README.md", "r") as f:
            model_card = f.read()
        assert "<style>" in model_card
