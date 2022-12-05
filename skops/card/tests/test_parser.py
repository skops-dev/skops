import difflib
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from skops.card import Card, parse_modelcard
from skops.card._parser import check_pandoc_installed

try:
    check_pandoc_installed()
except FileNotFoundError:
    # not installed, skip
    pytest.skip(reason="These tests require pandoc", allow_module_level=True)


@pytest.fixture
def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


@pytest.fixture
def card(fit_model, tmp_path):
    card = Card(fit_model)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1])
        fig.savefig(tmp_path / "my-throwaway-plot.png")
        card.add_plot(**{"My plots/My first plot": "my-throwaway-plot.png"})
    except ImportError:
        pass

    card.add_table(**{"A table": {"col0": [0, 1], "col1": [2, 3]}})
    return card


EXAMPLE_CARDS = [
    "bert-base-uncased.md",
    "clip-vit-large-patch14.md",
    "gpt2.md",
    "specter.md",
    "vit-base-patch32-224-in21k.md",
]


def assert_readme_files_almost_equal(file0, file1, diff):
    """Check that the two model cards are identical, but allow differences as
    defined in the ``diff`` file"""
    with open(file0, "r") as f:
        readme0 = f.readlines()

    with open(file1, "r") as f:
        readme1 = f.readlines()

    # exclude trivial case of both being empty
    assert readme0
    assert readme1

    diff_actual = list(difflib.unified_diff(readme0, readme1, n=0))

    with open(diff, "r") as f:
        diff_expected = f.readlines()

    assert diff_actual == diff_expected


@pytest.mark.parametrize("file_name", EXAMPLE_CARDS, ids=EXAMPLE_CARDS)
def test_example_model_cards(tmp_path, file_name):
    """Test that the difference between original and parsed model card is
    acceptable

    For this test, model cards for some of the most popular models on HF Hub
    were retrieved and stored in the ./examples folder. This test checks that
    these model cards can be successfully parsed and that the output is *almost*
    the same.

    We don't expect the output to be 100% identical, see the limitations listed
    in ``parse_modelcard``. Instead, we assert that the diff corresponds to the
    expected diff, which is also checked in.

    So e.g. for "specter.md", we expect that the diff will be the same diff as
    in "specter.md.diff".

    """
    path = Path(os.getcwd()) / "skops" / "card" / "tests" / "examples"
    file0 = path / file_name
    diff = (path / file_name).with_suffix(".md.diff")

    parsed_card = parse_modelcard(file0)
    file1 = tmp_path / "readme-parsed.md"
    parsed_card.save(file1)

    assert_readme_files_almost_equal(file0, file1, diff)
