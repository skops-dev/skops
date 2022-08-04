import copy
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
from huggingface_hub import metadata_load
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

import skops
from skops import hub_utils
from skops.card import Card, metadata_from_config


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


@pytest.fixture
def model_card(model_diagram=True):
    model = fit_model()
    card = Card(model, model_diagram)
    yield card


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


def test_save_model_card(destination_path, model_card):
    model_card.save(Path(destination_path) / "README.md")
    assert (Path(destination_path) / "README.md").exists()


def test_hyperparameter_table(destination_path, model_card):
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        model_card = f.read()
    assert "fit_intercept" in model_card


def test_plot_model(destination_path, model_card):
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        model_card = f.read()
        assert "<style>" in model_card


def test_plot_model_false(destination_path, model_card):
    model = fit_model()
    model_card = Card(model, model_diagram=False)
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        model_card = f.read()
        assert "<style>" not in model_card


def test_add(destination_path, model_card):
    model_card.add(model_description="sklearn FTW")
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        model_card = f.read()
        assert "sklearn FTW" in model_card


def test_template_sections_not_mutated_by_save(destination_path, model_card):
    template_sections_before = copy.deepcopy(model_card._template_sections)
    model_card.save(Path(destination_path) / "README.md")
    template_sections_after = copy.deepcopy(model_card._template_sections)
    assert template_sections_before == template_sections_after


def test_add_plot(destination_path, model_card):
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card.add_plot(fig1="fig1.png")
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        model_card = f.read()
        assert "![fig1](fig1.png)" in model_card


def test_temporary_plot(destination_path, model_card):
    # test if the additions are made to a temporary template file
    # and not to default template or template provided
    root = skops.__path__
    # read original template
    with open(Path(root[0]) / "card" / "default_template.md") as f:
        default_template = f.read()
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card.add_plot(fig1="fig1.png")
    model_card.save(Path(destination_path) / "README.md")
    # check if default template is not modified
    with open(Path(root[0]) / "card" / "default_template.md") as f:
        default_template_post = f.read()
    assert default_template == default_template_post


def test_metadata_keys(destination_path, model_card):
    # test if the metadata is added on top of the card
    model_card.metadata.tags = "dummy"
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        assert "tags: dummy" in f.read()


def test_metadata_from_config_tabular_data(destination_path):
    # test if widget data is correctly set in the README
    X, y = load_iris(return_X_y=True, as_frame=True)
    est = LogisticRegression(solver="liblinear").fit(X, y)
    pkl_file = tempfile.mkstemp(suffix=".pkl", prefix="skops-test")[1]
    with open(pkl_file, "wb") as f:
        pickle.dump(est, f)
    hub_utils.init(
        model=pkl_file,
        requirements=[f"scikit-learn=={sklearn.__version__}"],
        dst=destination_path,
        task="tabular-classification",
        data=X,
    )
    card = Card(
        est, model_diagram=True, metadata=metadata_from_config(destination_path)
    )
    card.save(Path(destination_path) / "README.md")
    metadata = metadata_load(local_path=Path(destination_path) / "README.md")
    assert "widget" in metadata

    expected_data = {
        "structuredData": {
            "petal length (cm)": [1.4, 1.4, 1.3],
            "petal width (cm)": [0.2, 0.2, 0.2],
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
        }
    }
    assert metadata["widget"] == expected_data
