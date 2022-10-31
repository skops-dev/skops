import copy
import os
import pickle
import tempfile
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
from huggingface_hub import CardData, metadata_load
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import skops
from skops import hub_utils
from skops.card import metadata_from_config
from skops.card._card_alternative import Card
from skops.card._model_card import PlotSection, TableSection
from skops.io import dump


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


def test_select_existing_section():
    # TODO
    pass


def test_select_non_existing_section_raises():
    # TODO
    pass


def test_hyperparameter_table(destination_path, model_card):
    section_name = "Model description/Training Procedure/Hyperparameters"
    text_hyperparams = model_card.select(section_name).content
    expected = "\n".join(
        [
            "The model is trained with below hyperparameters.",
            "",
            "<details>",
            "<summary> Click to expand </summary>",
            "",
            "| Hyperparameter   | Value      |",
            "|------------------|------------|",
            "| copy_X           | True       |",
            "| fit_intercept    | True       |",
            "| n_jobs           |            |",
            "| normalize        | deprecated |",
            "| positive         | False      |",
            "",
            "</details>",
        ]
    )
    assert text_hyperparams == expected


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
    section_name = "Model description/Training Procedure/Hyperparameters"
    text_hyperparams = model_card.select(section_name).content

    # remove multiple whitespaces, as they're not important
    text_cleaned = _strip_multiple_chars(text_hyperparams, " ")
    assert "| n_jobs | line<br />with<br />break |" in text_cleaned


def test_plot_model(destination_path, model_card):
    text_plot = model_card.select(
        "Model description/Training Procedure/Model Plot"
    ).content
    # don't compare whole text, as it's quite long and non-deterministic
    assert text_plot.startswith("The model plot is below.\n\n<style>#sk-container-id")
    assert "<style>" in text_plot
    assert text_plot.endswith(
        "<pre>LinearRegression()</pre></div></div></div></div></div>"
    )


def test_plot_model_false(destination_path, model_card):
    model = fit_model()
    model_card = Card(model, model_diagram=False)
    text_plot = model_card.select(
        "Model description/Training Procedure/Model Plot"
    ).content
    assert text_plot == "The model plot is below."


def test_add_new_section(destination_path, model_card):
    model_card = model_card.add(**{"A new section": "sklearn FTW"})
    section = model_card.select("A new section")
    assert section.content == "sklearn FTW"


def test_add_content_to_existing_section(destination_path, model_card):
    section = model_card.select("Model description")
    num_subsection_before = len(section.subsections)

    # add content to "Model description" section
    model_card = model_card.add(**{"Model description": "sklearn FTW"})
    section = model_card.select("Model description")
    num_subsection_after = len(section.subsections)

    assert num_subsection_before == num_subsection_after
    assert section.content == "sklearn FTW"


@pytest.mark.skip  # FIXME: remove
def test_template_sections_not_mutated_by_save(destination_path, model_card):
    template_sections_before = copy.deepcopy(model_card._template_sections)
    model_card.save(Path(destination_path) / "README.md")
    template_sections_after = copy.deepcopy(model_card._template_sections)
    assert template_sections_before == template_sections_after


def test_add_plot(destination_path, model_card):
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card = model_card.add_plot(fig1="fig1.png")
    plot_content = model_card.select("fig1").content.format()
    assert plot_content == "![fig1](fig1.png)"


@pytest.mark.skip  # FIXME: remove
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


def test_adding_metadata(destination_path, model_card):
    # test if the metadata is added to the card
    model_card.metadata.tags = "dummy"
    metadata = list(model_card._generate_metadata(model_card.metadata))
    assert len(metadata) == 1
    assert metadata[0] == "metadata.tags=dummy,"


@pytest.mark.xfail(reason="Waiting for update of model attribute")
def test_override_model(model_card):
    # test that the model can be overridden and dependent sections are updated
    hyperparams_before = model_card.select(
        "Model description/Training Procedure/Hyperparameters"
    ).content
    model_card.model = DecisionTreeClassifier()
    hyperparams_after = model_card.select(
        "Model description/Training Procedure/Hyperparameters"
    ).content

    assert hyperparams_before != hyperparams_after
    assert "fit_intercept" not in hyperparams_before
    assert "min_samples_leaf" in hyperparams_after


def test_add_metrics(destination_path, model_card):
    model_card.add_metrics(**{"acc": "0.1"})  # str
    model_card.add_metrics(f1=0.1)  # float
    model_card.add_metrics(awesomeness=123)  # int

    eval_metric_content = model_card.select(
        "Model description/Evaluation Results"
    ).content
    expected = "\n".join(
        [
            "| Metric      |   Value |",
            "|-------------|---------|",
            "| acc         |     0.1 |",
            "| f1          |     0.1 |",
            "| awesomeness |   123   |",
        ]
    )
    assert eval_metric_content.endswith(expected)


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
        assert f'model = load("{filename}")' in read_buffer

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


@pytest.mark.skip  # FIXME
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
        card._template_sections = {}  # type: ignore
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
        card._extra_sections = []  # type: ignore
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
        card._extra_sections.append(("some section", {1: 2}))  # type: ignore
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


def make_card(card_type, file_path: Path, fill_content: bool = True):
    import pickle

    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from skops import hub_utils
    from skops.card import Card as CardOld
    from skops.card import metadata_from_config

    if card_type == "old":
        card_cls = CardOld  # type: ignore
    else:
        card_cls = Card  # type: ignore

    destination_path = file_path.parent
    X, y = load_iris(return_X_y=True, as_frame=True)

    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=123))]
    ).fit(X, y)

    pkl_file = tempfile.mkstemp(suffix=".pkl", prefix="skops-test")[1]
    with open(pkl_file, "wb") as f:
        pickle.dump(model, f)

    hub_utils.init(
        model=pkl_file,
        requirements=[f"scikit-learn=={sklearn.__version__}"],
        dst=destination_path,
        task="tabular-classification",
        data=X,
    )
    card = card_cls(model, metadata=metadata_from_config(destination_path))

    if fill_content:
        # add metrics
        card.add_metrics(**{"acc": "0.1"})

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        if card_type == "old":
            card.add_plot(**{"A beautiful plot": "fig1.png"})
        else:
            # old card always adds additional content in an extra section
            card.add_plot(**{"Additional Content/A beautiful plot": "fig1.png"})

        # add table
        table = {"split": [1, 2, 3], "score": [4, 5, 6]}
        if card_type == "old":
            card.add_table(
                folded=True,
                **{"Yet another table": table},
            )
        else:
            # old card always adds additional content in an extra section
            card.add_table(
                folded=True, **{"Additional Content/Yet another table": table}
            )

        # add authors and contacts
        if card_type == "old":
            # old card requires to use the placeholder variable name
            card.add(
                **{
                    "model_card_authors": "Alice and Bob",
                    "model_card_contact": "alice@example.com",
                    "citation_bibtex": "Holy Cow, Nature, 2022-10",
                }
            )
        else:
            # new card uses the section titles instead and overrides the
            # existing content
            card.add(
                **{
                    "Model Card Authors": (
                        "This model card is written by following authors:\n\n"
                        "Alice and Bob"
                    ),
                    "Model Card Contact": (
                        "You can contact the model card authors through following"
                        " channels:\nalice@example.com"
                    ),
                    "Citation": (
                        "Below you can find information related to citation.\n\n"
                        "**BibTeX:**\n"
                        "```\nHoly Cow, Nature, 2022-10\n```\n"
                    ),
                }
            )

        # more metrics
        card.add_metrics(**{"f1": "0.2", "roc": "123"})

    card.save(file_path)


@pytest.mark.parametrize("fill_content", [False, True])
def test_old_and_new_card_identical(fill_content):
    import tempfile

    with tempfile.TemporaryDirectory(prefix="skops-test") as destination_path:
        file_path = Path(destination_path) / "README-old.md"
        make_card("old", file_path, fill_content=fill_content)
        card_old = file_path.read_text()

    with tempfile.TemporaryDirectory(prefix="skops-test") as destination_path:
        file_path = Path(destination_path) / "README-new.md"
        make_card("new", file_path, fill_content=fill_content)
        card_new = file_path.read_text()

    lines_old, lines_new = card_old.split("\n"), card_new.split("\n")
    for i, (line0, line1) in enumerate(zip_longest(lines_old, lines_new, fillvalue="")):
        # actual file name may differ, so only compare start of line
        if line0.startswith("model_file: skops-"):
            assert line1.startswith("model_file: skops-")
            continue
        if line0.startswith("model = joblib.load(skops-test"):
            assert line1.startswith("model = joblib.load(skops-test")
            continue

        # model diagram is not deterministic, e.g. ids
        if line0.startswith("<style>#sk-container-id"):
            assert line1.startswith("<style>#sk-container-id")
            continue

        assert line0 == line1
