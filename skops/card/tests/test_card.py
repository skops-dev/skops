import copy
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
from huggingface_hub import CardData, metadata_load
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

import skops
from skops import hub_utils
from skops.card import Card, metadata_from_config
from skops.card._model_card import PlotSection, TableSection, _load_model
from skops.io import dump


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


def save_model_to_file(model_instance, suffix):
    save_file = tempfile.mkstemp(suffix=suffix, prefix="skops-test")[1]
    if suffix in (".pkl", ".pickle"):
        with open(save_file, "wb") as f:
            pickle.dump(model_instance, f)
    elif suffix == ".skops":
        dump(model_instance, save_file)
    return save_file


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".skops"])
def test_load_model(suffix):
    model0 = LinearRegression(n_jobs=123)
    save_file = save_model_to_file(model0, suffix)
    loaded_model_str = _load_model(save_file)
    save_file_path = Path(save_file)
    loaded_model_path = _load_model(save_file_path)

    assert loaded_model_str.n_jobs is model0.n_jobs
    assert loaded_model_path.n_jobs is model0.n_jobs
    assert loaded_model_path.n_jobs is loaded_model_str.n_jobs


@pytest.fixture
def model_card(model_diagram=True):
    model = fit_model()
    card = Card(model, model_diagram)
    yield card


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    yield X, y


@pytest.fixture
def iris_estimator(iris_data):
    X, y = iris_data
    est = LogisticRegression(solver="liblinear").fit(X, y)
    yield est


@pytest.fixture
def iris_pkl_file(iris_estimator):
    pkl_file = tempfile.mkstemp(suffix=".pkl", prefix="skops-test")[1]
    with open(pkl_file, "wb") as f:
        pickle.dump(iris_estimator, f)
    yield pkl_file


@pytest.fixture
def iris_skops_file(iris_estimator):
    skops_folder = tempfile.mkdtemp()
    model_name = "model.skops"
    skops_path = Path(skops_folder) / model_name
    dump(iris_estimator, skops_path)
    yield skops_path


def _create_model_card_from_saved_model(
    destination_path,
    iris_estimator,
    iris_data,
    save_file,
):
    X, y = iris_data
    hub_utils.init(
        model=save_file,
        requirements=[f"scikit-learn=={sklearn.__version__}"],
        dst=destination_path,
        task="tabular-classification",
        data=X,
    )
    card = Card(iris_estimator, metadata=metadata_from_config(destination_path))
    card.save(Path(destination_path) / "README.md")
    return card


@pytest.fixture
def skops_model_card_metadata_from_config(
    destination_path, iris_estimator, iris_skops_file, iris_data
):
    yield _create_model_card_from_saved_model(
        destination_path, iris_estimator, iris_data, iris_skops_file
    )


@pytest.fixture
def pkl_model_card_metadata_from_config(
    destination_path, iris_estimator, iris_pkl_file, iris_data
):
    yield _create_model_card_from_saved_model(
        destination_path, iris_estimator, iris_data, iris_pkl_file
    )


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


def test_save_model_card(destination_path, model_card):
    model_card.save(Path(destination_path) / "README.md")
    assert (Path(destination_path) / "README.md").exists()


def test_hyperparameter_table(destination_path, model_card):
    model_card = model_card.render()
    assert "fit_intercept" in model_card


def _strip_multiple_chars(text, char):
    # _strip_multiple_chars("hi    there") == "hi there"
    # _strip_multiple_chars("|---|--|", "-") == "|-|-|"
    while char + char in text:
        text = text.replace(char + char, char)
    return text


def test_hyperparameter_table_with_line_break(destination_path):
    # Hyperparameters can contain values with line breaks, "\n", in them. In
    # that case, the markdown table is broken. Check that the hyperparameter
    # table we create properly replaces the "\n" with "<br />".
    class EstimatorWithLbInParams:
        def get_params(self, deep=False):
            return {"fit_intercept": True, "n_jobs": "line\nwith\nbreak"}

    model_card = Card(EstimatorWithLbInParams())
    model_card = model_card.render()
    # remove multiple whitespaces, as they're not important
    model_card = _strip_multiple_chars(model_card, " ")
    assert "| n_jobs | line<br />with<br />break |" in model_card


def test_plot_model(destination_path, model_card):
    model_card = model_card.render()
    assert "<style>" in model_card


def test_plot_model_false(destination_path, model_card):
    model = fit_model()
    model_card = Card(model, model_diagram=False).render()
    assert "<style>" not in model_card


def test_add(destination_path, model_card):
    model_card = model_card.add(model_description="sklearn FTW").render()
    assert "sklearn FTW" in model_card


def test_template_sections_not_mutated_by_save(destination_path, model_card):
    template_sections_before = copy.deepcopy(model_card._template_sections)
    model_card.save(Path(destination_path) / "README.md")
    template_sections_after = copy.deepcopy(model_card._template_sections)
    assert template_sections_before == template_sections_after


def test_add_plot(destination_path, model_card):
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card = model_card.add_plot(fig1="fig1.png").render()
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
    model_card = model_card.render()
    assert "tags: dummy" in model_card


def test_default_sections_save(model_card):
    # test if the plot and hyperparameters are only added during save
    assert "<style>" not in str(model_card)
    assert "fit_intercept" not in str(model_card)


def test_add_metrics(destination_path, model_card):
    model_card.add_metrics(**{"acc": 0.1})
    model_card.add_metrics(f1=0.1)
    card = model_card.render()
    assert ("acc" in card) and ("f1" in card) and ("0.1" in card)


def test_code_autogeneration(destination_path, pkl_model_card_metadata_from_config):
    # test if getting started code is automatically generated
    metadata = metadata_load(local_path=Path(destination_path) / "README.md")
    filename = metadata["model_file"]
    with open(Path(destination_path) / "README.md") as f:
        assert f"joblib.load({filename})" in f.read()


def test_code_autogeneration_skops(
    destination_path, skops_model_card_metadata_from_config
):
    # test if getting started code is automatically generated for skops format
    metadata = metadata_load(local_path=Path(destination_path) / "README.md")
    filename = metadata["model_file"]
    with open(Path(destination_path) / "README.md") as f:
        read_buffer = f.read()
        assert f'clf = load("{filename}")' in read_buffer

        # test if the model doesn't overflow the huggingface models page
        assert read_buffer.count("sk-top-container") == 1
        assert 'style="overflow: auto;' in read_buffer


def test_metadata_from_config_tabular_data(
    pkl_model_card_metadata_from_config, destination_path
):
    # test if widget data is correctly set in the README
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


class TestModelCardFromPath:
    @pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".skops"])
    def test_model_card_str(self, suffix):
        model0 = LinearRegression(n_jobs=123)
        save_file_str = save_model_to_file(model0, suffix)
        card_from_str = Card(save_file_str)
        card_from_model0 = Card(model0)

        assert card_from_model0.model.n_jobs == card_from_str.model.n_jobs

    @pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".skops"])
    def test_model_card_path(self, suffix):
        model0 = LinearRegression(n_jobs=123)
        save_file = save_model_to_file(model0, suffix)
        save_file_path = Path(save_file)
        card_from_path = Card(save_file_path)
        card_from_model0 = Card(model0)

        assert card_from_model0.model.n_jobs == card_from_path.model.n_jobs


class TestCardModelAttribute:
    def test_model_estimator(self):
        model0 = LinearRegression()

        card = Card(model0)
        assert card.model is model0

        # re-assigning the model works
        model1 = LogisticRegression()
        card.model = model1
        assert card.model is model1

        # re-assigning back to original works
        card.model = model0
        assert card.model is model0

    def test_model_is_str_pickle(self, destination_path):
        model0 = LinearRegression(n_jobs=123)
        f_name0 = destination_path / "lin_reg.pickle"
        with open(f_name0, "wb") as f:
            pickle.dump(model0, f)

        card = Card(f_name0)
        assert isinstance(card.model, LinearRegression)
        assert card.model.n_jobs == 123

        # re-assigning the model works
        model1 = LogisticRegression(n_jobs=456)
        f_name1 = destination_path / "log_reg.pickle"
        with open(f_name1, "wb") as f:
            pickle.dump(model1, f)

        card.model = f_name1
        assert isinstance(card.model, LogisticRegression)
        assert card.model.n_jobs == 456

        # re-assigning back to original works
        card.model = f_name0
        assert isinstance(card.model, LinearRegression)
        assert card.model.n_jobs == 123


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

    @pytest.mark.parametrize("folded", [True, False])
    def test_folded(self, folded):
        section = PlotSection(
            alt_text="some title", path="path/plot.png", folded=folded
        )
        output = section.format()
        if folded:
            assert "<details>" in output
        else:
            assert "<details>" not in output


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

    @pytest.mark.parametrize("table", [{}, {"col": []}, "pandas"])
    def test_raise_error_empty_table(self, table):
        # Test no columns, no rows, empty df
        if table == "pandas":
            pd = pytest.importorskip("pandas")
            table = pd.DataFrame([])

        msg = "Empty table added"
        with pytest.raises(ValueError, match=msg):
            TableSection(table=table)

    def test_pandas_not_installed(self, table_dict, pandas_not_installed):
        # use pandas_not_installed fixture from conftest.py to pretend that
        # pandas is not installed
        section = TableSection(table=table_dict)
        assert section._is_pandas_df is False

    @pytest.mark.parametrize("folded", [True, False])
    def test_folded(self, table_dict, folded):
        section = TableSection(table=table_dict, folded=folded)
        output = section.format()
        if folded:
            assert "<details>" in output
        else:
            assert "<details>" not in output

    def test_line_break_in_entry(self, table_dict):
        # Line breaks are not allowed inside markdown tables, so check that
        # they're removed. We test 3 conditions here:

        # 1. custom object with line breaks in repr
        # 2. string with line break in the middle
        # 3. string with line break at start, middle, and end

        # Note that for the latter, tabulate will automatically strip the line
        # breaks from the start and end.
        class LineBreakInRepr:
            """Custom object whose repr has a line break"""

            def __repr__(self) -> str:
                return "obj\nwith lb"

        table_dict["with break"] = [
            LineBreakInRepr(),
            "hi\nthere",
            """
entry with
line breaks
""",
        ]
        section = TableSection(table=table_dict)
        expected = """| split | score | with break |
|-|-|-|
| 1 | 4 | obj<br />with lb |
| 2 | 5 | hi<br />there |
| 3 | 6 | entry with<br />line breaks |"""

        result = section.format()
        # remove multiple whitespaces and dashes, as they're not important
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected
