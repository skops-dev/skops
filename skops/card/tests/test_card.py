import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import skops
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


def test_save_model_card():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model = fit_model()
        model_card = Card(model)
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        assert os.path.exists(f"{dir_path}/README.md")


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


def test_add():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = generate_card()
        model_card.add(model_description="sklearn FTW")
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
            assert "sklearn FTW" in model_card


def test_add_plot():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        model_card = generate_card()
        plt.plot([4, 5, 6, 7])
        plt.savefig(f"{dir_path}/fig1.png")
        model_card.add_plot(fig1="fig1.png")
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
            assert "![fig1](fig1.png)" in model_card
        assert os.path.exists(f"{dir_path}/fig1.png")


def test_temporary_plot():
    # test if the additions are made to a temporary template file
    # and not to default template or template provided
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        root = skops.__path__
        # read original template
        with open(os.path.join(f"{root[0]}", "card", "default_template.md")) as f:
            default_template = f.read()
            f.seek(0)
        model_card = generate_card()
        plt.plot([4, 5, 6, 7])
        plt.savefig(f"{dir_path}/fig1.png")
        model_card.add_plot(fig1="fig1.png")
        model_card.save(os.path.join(f"{dir_path}", "README.md"))
        # make sure the card has plots
        with open(os.path.join(f"{dir_path}", "README.md"), "r") as f:
            model_card = f.read()
            assert "![fig1](fig1.png)" in model_card

        # check if default template is not modified
        with open(os.path.join(f"{root[0]}", "card", "default_template.md")) as f:
            default_template_post = f.read()
        assert default_template == default_template_post
