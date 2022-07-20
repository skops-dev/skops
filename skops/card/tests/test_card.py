import os
import tempfile

import numpy as np
from sklearn.linear_model import LinearRegression

from skops.card import Card


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


def generate_card():
    model = fit_model()

    model_card = Card(model)
    return model_card


def test_write_model_card():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model = fit_model()
        model_card = Card(model)
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "sklearn FTW" in model_card


def test_hyperparameter_table():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = generate_card()
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "fit_intercept" in model_card


def test_plot_model():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = generate_card()
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
        assert "<style>" in model_card
