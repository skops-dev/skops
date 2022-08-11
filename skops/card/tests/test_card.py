import copy
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

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
from skops.card._model_card import PlotSection, TableSection


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


def test_add_metrics(destination_path, model_card):
    model_card.add_metrics(**{"acc": 0.1})
    model_card.add_metrics(f1=0.1)
    model_card.save(Path(destination_path) / "README.md")
    with open(Path(destination_path) / "README.md", "r") as f:
        card = f.read()
        assert ("acc" in card) and ("f1" in card) and ("0.1" in card)


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

    for tag in ["sklearn", "skops", "tabular-classification"]:
        assert tag in metadata["tags"]


class TestCardRepr:
    """Test __str__ and __repr__ methods of Card, which are identical for now"""

    @pytest.fixture
    def card(self):
        model = LinearRegression(fit_intercept=False)
        card = Card(model=model)
        card.add(
            model_description="A description",
            model_card_authors="Jane Doe",
        )
        card.add_plot(
            roc_curve="ROC_curve.png",
            confusion_matrix="confusion_matrix.jpg",
        )
        card.add_table(search_results={"split": [1, 2, 3], "score": [4, 5, 6]})
        return card

    @pytest.mark.parametrize("meth", [repr, str])
    def test_card_repr(self, card: Card, meth):
        result = meth(card)
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  model_description='A description',\n"
            "  model_card_authors='Jane Doe',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_very_long_lines_are_shortened(self, card: Card, meth):
        card.add(my_section="very long line " * 100)
        result = meth(card)
        expected = (
            "Card(\n  model=LinearRegression(fit_intercept=False),\n"
            "  model_description='A description',\n  model_card_authors='Jane Doe',\n"
            "  my_section='very long line very lon...line very long line very long line"
            " ',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_without_model_attribute(self, card: Card, meth):
        del card.model
        result = meth(card)
        expected = (
            "Card(\n"
            "  model_description='A description',\n"
            "  model_card_authors='Jane Doe',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_no_template_sections(self, card: Card, meth):
        card._template_sections = {}
        result = meth(card)
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_no_extra_sections(self, card: Card, meth):
        card._extra_sections = []
        result = meth(card)
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  model_description='A description',\n"
            "  model_card_authors='Jane Doe',\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_template_section_val_not_str(self, card: Card, meth):
        card._template_sections["model_description"] = [1, 2, 3]  # type: ignore
        result = meth(card)
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  model_description=[1, 2, 3],\n"
            "  model_card_authors='Jane Doe',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_extra_sections_val_not_str(self, card: Card, meth):
        card._extra_sections.append(("some section", {1: 2}))
        result = meth(card)
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  model_description='A description',\n"
            "  model_card_authors='Jane Doe',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            "  some section={1: 2},\n"
            ")"
        )
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_with_metadata(self, card: Card, meth):
        from modelcards import CardData

        metadata = CardData(
            language="fr",
            license="bsd",
            library_name="sklearn",
            tags=["sklearn", "tabular-classification"],
            foo={"bar": 123},
            widget={"something": "very-long"},
        )
        card.metadata = metadata
        expected = (
            "Card(\n"
            "  model=LinearRegression(fit_intercept=False),\n"
            "  metadata.language=fr,\n"
            "  metadata.license=bsd,\n"
            "  metadata.library_name=sklearn,\n"
            "  metadata.tags=['sklearn', 'tabular-classification'],\n"
            "  metadata.foo={'bar': 123},\n"
            "  metadata.widget={...},\n"
            "  model_description='A description',\n"
            "  model_card_authors='Jane Doe',\n"
            "  roc_curve='ROC_curve.png',\n"
            "  confusion_matrix='confusion_matrix.jpg',\n"
            "  search_results=Table(3x2),\n"
            ")"
        )
        result = meth(card)
        assert result == expected


class TestPlotSection:
    def test_format_path_is_str(self):
        section = PlotSection(alt_text="some title", path="path/plot.png")
        expected = "![some title](path/plot.png)"
        assert section.format() == expected

    def test_format_path_is_pathlib(self):
        section = PlotSection(alt_text="some title", path=Path("path") / "plot.png")
        expected = f"![some title](path{os.path.sep}plot.png)"
        assert section.format() == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr(self, meth):
        section = PlotSection(alt_text="some title", path="path/plot.png")
        expected = "'path/plot.png'"
        assert meth(section) == expected

    def test_str(self):
        section = PlotSection(alt_text="some title", path="path/plot.png")
        expected = "'path/plot.png'"
        assert str(section) == expected


class TestTableSection:
    @pytest.fixture
    def table_dict(self):
        return {"split": [1, 2, 3], "score": [4, 5, 6]}

    def test_table_is_dict(self, table_dict):
        section = TableSection(table=table_dict)
        expected = """|   split |   score |
|---------|---------|
|       1 |       4 |
|       2 |       5 |
|       3 |       6 |"""
        assert section.format() == expected

    def test_table_is_dataframe(self, table_dict):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(table_dict)
        section = TableSection(table=df)
        expected = """|   split |   score |
|---------|---------|
|       1 |       4 |
|       2 |       5 |
|       3 |       6 |"""
        assert section.format() == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr_table_is_dict(self, table_dict, meth):
        section = TableSection(table=table_dict)
        expected = "Table(3x2)"
        assert meth(section) == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr_table_is_dataframe(self, table_dict, meth):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(table_dict)
        section = TableSection(table=df)
        expected = "Table(3x2)"
        assert meth(section) == expected

    def test_raise_error_if_no_cols(self):
        msg = "Empty table added"
        with pytest.raises(ValueError, match=msg):
            TableSection(table={})

    def test_raise_error_if_no_rows(self):
        msg = "Empty table added"
        with pytest.raises(ValueError, match=msg):
            TableSection(table={"col": []})

    def test_raise_error_df_empty(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame([])
        msg = "Empty table added"
        with pytest.raises(ValueError, match=msg):
            TableSection(table=df)

    def test_pandas_not_installed(self, table_dict):
        # patch import so that it raises an ImportError when trying to import
        # pandas. This works because pandas is only imported lazily.
        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            section = TableSection(table=table_dict)
            assert section._is_pandas_df is False
