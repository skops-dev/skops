import copy
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
    template_sections_before = copy.deepcopy(model_card.template_sections)
    model_card.save(Path(destination_path) / "README.md")
    template_sections_after = copy.deepcopy(model_card.template_sections)
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
    model_card.add(tags="dummy")
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        assert "tags: dummy" in f.read()
        
        
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
        # TODO: Change Below
        card_data = CardData(eval_results=eval_results, model_name="my-cool-model")

        card = create_model_card(model, card_data)
        card.save(os.path.join(f"{dir_path}", "README.md"))
        loaded_card = RepoCard.load(os.path.join(f"{dir_path}", "README.md"))
        assert loaded_card.data.eval_results[0].task_type == "tabular-regression"

