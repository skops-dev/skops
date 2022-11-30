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


def assert_readme_files_equal(file0, file1):
    """Check that the two model cards are identical, but allow differences in
    line breaks."""
    # exclude trivial case of both being empty
    assert file0
    assert file1

    with open(file0, "r") as f:
        readme0 = f.readlines()

    with open(file1, "r") as f:
        readme1 = f.readlines()

    # remove completely empty lines
    readme0 = [line.strip() for line in readme0 if line.strip()]
    readme1 = [line.strip() for line in readme1 if line.strip()]

    readme_str0 = "\n".join(readme0)
    readme_str1 = "\n".join(readme1)

    # a minuscule further difference is an excess empty line after </style>
    readme_str1 = readme_str1.replace("</style>\n", "</style>")

    assert readme_str0 == readme_str1


def test_parsed_card_identical(card, tmp_path):
    file0 = tmp_path / "readme-skops.md"
    card.save(file0)

    parsed_card = parse_modelcard(file0)
    file1 = tmp_path / "readme-parsed.md"
    parsed_card.save(file1)

    assert_readme_files_equal(file0, file1)
