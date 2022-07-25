import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import skops
from skops.card import Card


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


def generate_card(model_diagram=True):
    model = fit_model()
    model_card = Card(model, model_diagram)
    return model_card


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


def test_save_model_card(destination_path):
    model = fit_model()
    model_card = Card(model)
    model_card.save((Path(destination_path) / "README.md"))
    assert (Path(destination_path) / "README.md").exists()


def test_hyperparameter_table(destination_path):
    model_card = generate_card()
    model_card.save((Path(destination_path) / "README.md"))
    with open((Path(destination_path) / "README.md"), "r") as f:
        model_card = f.read()
    assert "fit_intercept" in model_card


def test_plot_model(destination_path):
    model_card = generate_card()
    model_card.save((Path(destination_path) / "README.md"))
    with open((Path(destination_path) / "README.md"), "r") as f:
        model_card = f.read()
        assert "<style>" in model_card


def test_plot_model_false(destination_path):
    model_card = generate_card(model_diagram=False)
    model_card.save((Path(destination_path) / "README.md"))
    with open((Path(destination_path) / "README.md"), "r") as f:
        model_card = f.read()
        assert "<style>" not in model_card


def test_add(destination_path):
    model_card = generate_card()
    model_card.add(model_description="sklearn FTW")
    model_card.save((Path(destination_path) / "README.md"))
    with open((Path(destination_path) / "README.md"), "r") as f:
        model_card = f.read()
        assert "sklearn FTW" in model_card


def test_add_plot(destination_path):
    model_card = generate_card()
    plt.plot([4, 5, 6, 7])
    plt.savefig(f"{destination_path}/fig1.png")
    model_card.add_plot(fig1="fig1.png")
    model_card.save(Path(destination_path) / "README.md")
    with open((Path(destination_path) / "README.md"), "r") as f:
        model_card = f.read()
        assert "![fig1](fig1.png)" in model_card


def test_temporary_plot(destination_path):
    # test if the additions are made to a temporary template file
    # and not to default template or template provided
    root = skops.__path__
    # read original template
    with open((Path(root[0]) / "card" / "default_template.md")) as f:
        default_template = f.read()
        f.seek(0)
    model_card = generate_card()
    plt.plot([4, 5, 6, 7])
    plt.savefig((Path(destination_path) / "fig1.png"))
    model_card.add_plot(fig1="fig1.png")
    model_card.save((Path(destination_path) / "README.md"))
    # check if default template is not modified
    with open((Path(root[0]) / "card" / "default_template.md")) as f:
        default_template_post = f.read()
    assert default_template == default_template_post
