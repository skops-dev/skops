import os
import pickle
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn
from huggingface_hub import CardData, metadata_load
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skops import hub_utils
from skops.card import metadata_from_config
from skops.card._model_card import Card, PlotSection, TableSection
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


def test_hyperparameter_table(model_card):
    section_name = "Model description/Training Procedure/Hyperparameters"
    result = model_card.select(section_name).content

    lines = [
        "The model is trained with below hyperparameters.",
        "",
        "<details>",
        "<summary> Click to expand </summary>",
        "",
        "| Hyperparameter   | Value   |",
        "|------------------|---------|",
        "| copy_X           | True    |",
        "| fit_intercept    | True    |",
        "| n_jobs           |         |",
        "| normalize        | False   |",
        "| positive         | False   |",
        "",
        "</details>",
    ]
    # TODO: remove when dropping sklearn v0.24 and when dropping v1.1 and
    # below. This is because the "normalize" parameter was changed after
    # v0.24 will be removed completely in sklearn v1.2.
    major, minor, *_ = sklearn.__version__.split(".")
    major, minor = int(major), int(minor)
    if (major >= 1) and (minor < 2):
        lines[10] = "| normalize        | deprecated |"
    elif (major >= 1) and (minor >= 2):
        del lines[10]
    expected = "\n".join(lines)

    # remove multiple whitespaces and dashes, as they're not important and may
    # differ depending on OS
    expected = _strip_multiple_chars(expected, " ")
    expected = _strip_multiple_chars(expected, "-")
    result = _strip_multiple_chars(result, " ")
    result = _strip_multiple_chars(result, "-")

    assert result == expected


def _strip_multiple_chars(text, char):
    # _strip_multiple_chars("hi    there") == "hi there"
    # _strip_multiple_chars("|---|--|", "-") == "|-|-|"
    while char + char in text:
        text = text.replace(char + char, char)
    return text


def test_hyperparameter_table_with_line_break():
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


def test_plot_model(model_card):
    text_plot = model_card.select(
        "Model description/Training Procedure/Model Plot"
    ).content
    # don't compare whole text, as it's quite long and non-deterministic
    assert text_plot.startswith("The model plot is below.\n\n<style>#sk-")
    assert "<style>" in text_plot
    assert text_plot.endswith(
        "<pre>LinearRegression()</pre></div></div></div></div></div>"
    )


def test_plot_model_false(model_card):
    model = fit_model()
    model_card = Card(model, model_diagram=False)
    text_plot = model_card.select(
        "Model description/Training Procedure/Model Plot"
    ).content
    assert text_plot == "The model plot is below."


def test_render(model_card, destination_path):
    file_name = destination_path / "README.md"
    model_card.save(file_name)
    with open(file_name, "r", encoding="utf-8") as f:
        loaded = f.read()

    rendered = model_card.render()
    assert loaded == rendered


def test_with_metadata(model_card):
    model_card.metadata.foo = "something"
    model_card.metadata.bar = "something else"
    rendered = model_card.render()
    expected = textwrap.dedent(
        """
        ---
        foo: something
        bar: something else
        ---
        """
    ).strip()
    assert rendered.startswith(expected)


class TestSelect:
    """Selecting sections from the model card"""

    def test_select_existing_section(self, model_card):
        section = model_card.select("Model description")
        assert section.title == "Model description"

    def test_select_existing_subsection(self, model_card):
        section = model_card.select("Model description/Training Procedure")
        assert section.title == "Training Procedure"

        section = model_card.select(["Model description", "Training Procedure"])
        assert section.title == "Training Procedure"

    def test_select_non_existing_section_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select("non-existing section")

    def test_select_non_existing_subsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select("Model description/non-existing subsection")

        with pytest.raises(KeyError):
            model_card.select(["Model description", "non-existing subsection"])

    def test_select_non_existing_subsubsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select(
                "Model description/Training Procedure/non-existing sub-subsection"
            )

        with pytest.raises(KeyError):
            model_card.select(
                [
                    "Model description",
                    "Training Procedure",
                    "non-existing sub-subsection",
                ]
            )

    def test_select_non_existing_section_and_subsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select(["non-existing section", "non-existing subsection"])

    def test_select_empty_key_raises(self, model_card):
        msg = r"Section name cannot be empty but got ''"
        with pytest.raises(KeyError, match=msg):
            model_card.select("")

        msg = r"Section name cannot be empty but got '\[\]'"
        with pytest.raises(KeyError, match=msg):
            model_card.select([])

    def test_select_empty_key_subsection_raises(self, model_card):
        msg = r"Section name cannot be empty but got 'Model description/'"
        with pytest.raises(KeyError, match=msg):
            model_card.select("Model description/")

        msg = r"Section name cannot be empty but got '\['Model description', ''\]'"
        with pytest.raises(KeyError, match=msg):
            model_card.select(["Model description", ""])

    def test_default_skops_sections_present(self, model_card):
        from skops.card._model_card import SKOPS_TEMPLATE

        # model_card (which is prefilled) contains all default sections
        for key in SKOPS_TEMPLATE:
            model_card.select(key)

    def test_default_hub_sections_present(self, model_card):
        from skops.card._model_card import HUB_TEMPLATE

        model = fit_model()
        model_card = Card(model, model_diagram=False, template="hub")

        # model_card contains all default sections
        for key in HUB_TEMPLATE:
            model_card.select(key)

    def test_custom_template_sections_present(self, model_card):
        template = {
            "My awesome model": "hello",
            "My awesome model/More details": "123",
            "More info": "Thanks",
        }
        model = fit_model()
        model_card = Card(model, model_diagram=False, template=template)

        # model_card contains all default sections
        for key in template:
            model_card.select(key)

        # no other top level sections as those defined in the template
        assert list(model_card._data.keys()) == ["My awesome model", "More info"]

    def test_default_skops_sections_empty_card(self, model_card):
        # Without prefilled template, the card should not contain the default sections
        from skops.card._model_card import SKOPS_TEMPLATE

        # empty card does not contain those sections
        model = fit_model()
        card_empty = Card(model, model_diagram=False, template=None)
        for key in SKOPS_TEMPLATE:
            with pytest.raises(KeyError):
                card_empty.select(key)

    def test_invalid_template_name_raises(self):
        msg = "Unknown template does-not-exist, must be one of"
        with pytest.raises(ValueError, match=msg):
            Card(model=None, template="does-not-exist")


class TestAdd:
    """Adding sections and subsections"""

    def test_add_new_section(self, model_card):
        model_card = model_card.add(**{"A new section": "sklearn FTW"})
        section = model_card.select("A new section")
        assert section.title == "A new section"
        assert section.content == "sklearn FTW"

    def test_add_new_subsection(self, model_card):
        model_card = model_card.add(
            **{"Model description/A new section": "sklearn FTW"}
        )
        section = model_card.select("Model description/A new section")
        assert section.title == "A new section"
        assert section.content == "sklearn FTW"

        # make sure that the new subsection is the last subsection
        subsections = model_card._data["Model description"].subsections
        assert len(subsections) > 1  # exclude trivial case of only one subsection

        last_subsection = list(subsections.values())[-1]
        assert last_subsection is section

    def test_add_new_section_and_subsection(self, model_card):
        model_card = model_card.add(**{"A new section/A new subsection": "sklearn FTW"})

        section = model_card.select("A new section")
        assert section.title == "A new section"
        assert section.content == ""

        subsection = model_card.select("A new section/A new subsection")
        assert subsection.title == "A new subsection"
        assert subsection.content == "sklearn FTW"

    def test_add_new_section_with_slash_in_name(self, model_card):
        model_card = model_card.add(**{"A new\\/section": "sklearn FTW"})
        section = model_card.select("A new\\/section")
        assert section.title == "A new/section"
        assert section.content == "sklearn FTW"

    def test_add_new_subsection_with_slash_in_name(self, model_card):
        model_card = model_card.add(
            **{"Model description/A new\\/section": "sklearn FTW"}
        )
        section = model_card.select("Model description/A new\\/section")
        assert section.title == "A new/section"
        assert section.content == "sklearn FTW"

    def test_add_content_to_existing_section(self, model_card):
        # Add content (not new sections) to an existing section. Make sure that
        # existing subsections are not affected by this
        section = model_card.select("Model description")
        num_subsection_before = len(section.subsections)
        assert num_subsection_before > 0  # exclude trivial case of empty sections

        # add content to "Model description" section
        model_card = model_card.add(**{"Model description": "sklearn FTW"})
        section = model_card.select("Model description")
        num_subsection_after = len(section.subsections)

        assert num_subsection_before == num_subsection_after
        assert section.content == "sklearn FTW"


class TestDelete:
    """Deleting sections and subsections"""

    def test_delete_section(self, model_card):
        model_card.select("Model description")
        model_card.delete("Model description")
        with pytest.raises(KeyError):
            model_card.select("Model description")

    def test_delete_subsection(self, model_card):
        model_card.select("Model description/Training Procedure")
        model_card.delete("Model description/Training Procedure")
        with pytest.raises(KeyError):
            model_card.select("Model description/Training Procedure")
        # parent section still exists
        model_card.delete("Model description")

    def test_delete_subsubsection(self, model_card):
        model_card.select("Model description/Training Procedure/Hyperparameters")
        model_card.delete("Model description/Training Procedure/Hyperparameters")
        with pytest.raises(KeyError):
            model_card.select("Model description/Training Procedure/Hyperparameters")
        # parent section still exists
        model_card.delete("Model description/Training Procedure")

    def test_delete_section_with_slash_in_name(self, model_card):
        model_card.add(**{"A new\\/section": "some content"})
        model_card.select("A new\\/section")
        model_card.delete("A new\\/section")
        with pytest.raises(KeyError):
            model_card.select("A new\\/section")

    def test_delete_non_existing_section_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.delete("non-existing section")

    def test_delete_non_existing_subsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.delete("Model description/non-existing subsection")

        with pytest.raises(KeyError):
            model_card.delete(["Model description", "non-existing subsection"])

    def test_delete_non_existing_subsubsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.delete(
                "Model description/Training Procedure/non-existing sub-subsection"
            )

        with pytest.raises(KeyError):
            model_card.delete(
                [
                    "Model description",
                    "Training Procedure",
                    "non-existing sub-subsection",
                ]
            )

    def test_delete_non_existing_section_and_subsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.delete(["non-existing section", "non-existing subsection"])

    def test_delete_empty_key_raises(self, model_card):
        msg = r"Section name cannot be empty but got ''"
        with pytest.raises(KeyError, match=msg):
            model_card.delete("")

        msg = r"Section name cannot be empty but got '\[\]'"
        with pytest.raises(KeyError, match=msg):
            model_card.delete([])

    def test_delete_empty_key_subsection_raises(self, model_card):
        msg = r"Section name cannot be empty but got 'Model description/'"
        with pytest.raises(KeyError, match=msg):
            model_card.delete("Model description/")

        msg = r"Section name cannot be empty but got '\['Model description', ''\]'"
        with pytest.raises(KeyError, match=msg):
            model_card.delete(["Model description", ""])


def test_add_plot(destination_path, model_card):
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card = model_card.add_plot(fig1="fig1.png")
    plot_content = model_card.select("fig1").content.format()
    assert plot_content == "![fig1](fig1.png)"


def test_add_plot_to_existing_section(destination_path, model_card):
    plt.plot([4, 5, 6, 7])
    plt.savefig(Path(destination_path) / "fig1.png")
    model_card = model_card.add_plot(**{"Model description/Figure 1": "fig1.png"})
    plot_content = model_card.select("Model description/Figure 1").content.format()
    assert plot_content == "![Figure 1](fig1.png)"


def test_adding_metadata(model_card):
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


def test_code_autogeneration(
    model_card, destination_path, pkl_model_card_metadata_from_config
):
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


class TestCardRepr:
    """Test __str__ and __repr__ methods of Card, which are identical for now"""

    @pytest.fixture
    def card(self):
        model = LinearRegression(fit_intercept=False)
        card = Card(model=model)
        card.add(Figures="")
        card.add(
            **{
                "Model Description": "A description",
                "Model Card Authors": "Jane Doe",
            }
        )
        card.add_plot(
            **{
                "Figures/ROC": "ROC.png",
                "Figures/Confusion matrix": "confusion_matrix.jpg",
            }
        )
        card.add_table(**{"Search Results": {"split": [1, 2, 3], "score": [4, 5, 6]}})
        return card

    @pytest.fixture
    def expected_lines(self):
        card_repr = """
        Card(
          model=LinearRegression(fit_intercept=False),
          Model description/Training Procedure/...ed | | positive | False | </details>,
          Model description/Training Procedure/...</pre></div></div></div></div></div>,
          Model description/Evaluation Results=...ric | Value | |----------|---------|,
          Model Card Authors=Jane Doe,
          Figures/ROC='ROC.png',
          Figures/Confusion matrix='confusion_matrix.jpg',
          Model Description=A description,
          Search Results=Table(3x2),
        )
        """
        expected = textwrap.dedent(card_repr).strip()
        lines = expected.split("\n")

        # TODO: remove when dropping sklearn v0.24 and when dropping v1.1 and
        # below. This is because the "normalize" parameter was changed after
        # v0.24 will be removed completely in sklearn v1.2.
        major, minor, *_ = sklearn.__version__.split(".")
        if int(major) < 1:
            # v0.24: "deprecated" -> "False"
            lines[2] = (
                "  Model description/Training Procedure/...se | | positive | False | "
                "</details>,"
            )
        elif int(minor) >= 2:
            # >= v1.2: remove argument completely
            lines[2] = (
                "  Model description/Training Procedure/... | | | positive | False | "
                "</details>,"
            )
        return lines

    @pytest.mark.parametrize("meth", [repr, str])
    def test_card_repr(self, card: Card, meth, expected_lines):
        result = meth(card)
        expected = "\n".join(expected_lines)
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_card_repr_empty_card(self, meth):
        """Without prefilled template, the repr should be empty"""
        model = fit_model()
        card = Card(model, model_diagram=False, template=None)
        result = meth(card)
        expected = textwrap.dedent(
            """
        Card(
          model=LinearRegression(),
        )
        """
        ).strip()
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_very_long_lines_are_shortened(self, card: Card, meth, expected_lines):
        card.add(my_section="very long line " * 100)

        # expected results contain 1 line at the very end
        extra_line = (
            "  my_section=very long line very long l... "
            "line very long line very long line ,"
        )
        expected_lines.insert(-1, extra_line)
        expected = "\n".join(expected_lines)

        result = meth(card)
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_without_model_attribute(self, card: Card, meth, expected_lines):
        del card.model

        # remove line 1 from expected results, which corresponds to the model
        del expected_lines[1]
        expected = "\n".join(expected_lines)

        result = meth(card)
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_with_metadata(self, card: Card, meth, expected_lines):
        metadata = CardData(
            language="fr",
            license="bsd",
            library_name="sklearn",
            tags=["sklearn", "tabular-classification"],
            foo={"bar": 123},
            widget={"something": "very-long"},
        )
        card.metadata = metadata

        # metadata comes after model line, i.e. position 2
        extra_lines = [
            "  metadata.language=fr,",
            "  metadata.license=bsd,",
            "  metadata.library_name=sklearn,",
            "  metadata.tags=['sklearn', 'tabular-classification'],",
            "  metadata.foo={'bar': 123},",
            "  metadata.widget={...},",
        ]
        expected = "\n".join(expected_lines[:2] + extra_lines + expected_lines[2:])

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

    @pytest.mark.parametrize("table", [{}, "pandas"])
    def test_raise_error_empty_table(self, table):
        # If there are no columns, raise
        if table == "pandas":
            pd = pytest.importorskip("pandas")
            table = pd.DataFrame([])

        msg = "Trying to add table with no columns"
        with pytest.raises(ValueError, match=msg):
            TableSection(table=table)

    @pytest.mark.parametrize("table", [{"col0": []}, "pandas"])
    def test_table_with_no_rows_works(self, table):
        # If there are no rows, it's okay
        if table == "pandas":
            pd = pytest.importorskip("pandas")
            table = pd.DataFrame(data=[], columns=["col0"])

        TableSection(table=table).format()  # no error raised

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
