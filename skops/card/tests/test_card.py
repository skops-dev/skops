import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sklearn
from modelcards import CardData
from sklearn.linear_model import LinearRegression

from skops import hub_utils
from skops.card import create_model_card


@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


@pytest.fixture
def model_folder(temp_path):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_path = Path(dir_path) / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(reg, f)
        hub_utils.init(
            model=model_path,
            requirements=[f"scikit-learn{sklearn.__version__}"],
            task="tabular-classification",
            data=X,
            dst=temp_path,
        )
    yield Path(temp_path)


@pytest.fixture
def sample_card(model_folder):
    card_data = CardData(library_name="sklearn")

    model_card = create_model_card(
        model_folder,
        card_data,
        template_path="skops/card/default_template.md",
        model_description="sklearn FTW",
    )
    return model_card


def test_write_model_card(model_folder):
    card_data = CardData(library_name="sklearn")
    model_card = create_model_card(
        model_folder,
        card_data=card_data,
        model_description="sklearn FTW",
    )
    card_path = model_folder / "README.md"
    model_card.save(card_path)
    with open(card_path, "r") as f:
        model_card = f.read()
    assert "sklearn FTW" in model_card


def test_hyperparameter_table(sample_card, temp_path):
    card_path = temp_path / "README.md"
    sample_card.save(card_path)
    with open(card_path, "r") as f:
        model_card = f.read()
    assert "fit_intercept" in model_card


def test_plot_model(sample_card, temp_path):
    card_path = temp_path / "README.md"
    sample_card.save(card_path)
    with open(card_path, "r") as f:
        model_card = f.read()
    assert "<style>" in model_card
