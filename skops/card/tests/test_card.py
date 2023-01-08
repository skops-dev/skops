import os
import pickle
import re
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import sklearn
from huggingface_hub import CardData, metadata_load
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skops import hub_utils
from skops.card import Card, metadata_from_config
from skops.card._model_card import (
    SKOPS_TEMPLATE,
    PlotSection,
    TableSection,
    _load_model,
)
from skops.io import dump


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg


def save_model_to_file(model_instance, suffix):
    save_file_handle, save_file = tempfile.mkstemp(suffix=suffix, prefix="skops-test")
    if suffix in (".pkl", ".pickle"):
        with open(save_file, "wb") as f:
            pickle.dump(model_instance, f)
    elif suffix == ".skops":
        dump(model_instance, save_file)
    return save_file_handle, save_file


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".skops"])
def test_load_model(suffix):
    model0 = LinearRegression(n_jobs=123)
    _, save_file = save_model_to_file(model0, suffix)
    loaded_model_str = _load_model(save_file, trusted=True)
    save_file_path = Path(save_file)
    loaded_model_path = _load_model(save_file_path, trusted=True)
    loaded_model_instance = _load_model(model0, trusted=True)

    assert loaded_model_str.n_jobs == 123
    assert loaded_model_path.n_jobs == 123
    assert loaded_model_instance.n_jobs == 123


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


CUSTOM_TEMPLATES = [None, {}, {"A Title", "Another Title", "A Title/A Section"}]  # type: ignore


class TestAddModelPlot:
    """Tests for the sklearn model repr"""

    def test_default(self, model_card):
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        # don't compare whole text, as it's quite long and non-deterministic
        assert result.startswith("The model plot is below.\n\n<style>#sk-")
        assert "<style>" in result
        assert result.endswith(
            "<pre>LinearRegression()</pre></div></div></div></div></div>"
        )

    def test_no_overflow(self, model_card):
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        # test if the model doesn't overflow the huggingface models page
        assert result.count("sk-top-container") == 1
        assert 'style="overflow: auto;' in result

    def test_model_diagram_false(self):
        model = fit_model()
        model_card = Card(model, model_diagram=False)
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        assert result == "The model plot is below."

    def test_other_section(self, model_card):
        model_card.add_model_plot(section="Other section")
        result = model_card.select("Other section").content
        assert result.startswith("The model plot is below.\n\n<style>#sk-")
        assert "<style>" in result
        assert result.endswith(
            "<pre>LinearRegression()</pre></div></div></div></div></div>"
        )

    def test_other_description(self, model_card):
        model_card.add_model_plot(description="Awesome diagram below")
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        assert result.startswith("Awesome diagram below\n\n<style>#sk-")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_raises(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        msg = (
            "You are trying to add a model plot but you're using a custom template, "
            "please pass the 'section' argument to determine where to put the content"
        )
        with pytest.raises(ValueError, match=msg):
            model_card.add_model_plot()

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        text1 = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        model_card.add_model_plot(section="Other section")
        text2 = model_card.select("Other section").content

        # both are identical, except for numbers like "#sk-container-id-123",
        # thus compare everything but the numbers
        assert re.split(r"\d+", text1) == re.split(r"\d+", text2)


def _strip_multiple_chars(text, char):
    # utility function needed to compare tables across systems
    # _strip_multiple_chars("hi    there") == "hi there"
    # _strip_multiple_chars("|---|--|", "-") == "|-|-|"
    while char + char in text:
        text = text.replace(char + char, char)
    return text


class TestAddHyperparams:
    """Tests for the model hyperparameters"""

    @pytest.fixture
    def expected(self):
        lines = [
            "The model is trained with below hyperparameters.",
            "",
            "<details>",
            "<summary> Click to expand </summary>",
            "",
            "| Hyperparameter   | Value       |",
            "|------------------|-------------|",
            "| copy_X           | True        |",
            "| fit_intercept    | True        |",
            "| n_jobs           |             |",
            "| normalize        | deprecated  |",
            "| positive         | False       |",
            "",
            "</details>",
        ]
        # TODO: After dropping sklearn < 1.2, when the "normalize" parameter is
        # removed, remove it from the table above and remove the code below.
        major, minor, *_ = sklearn.__version__.split(".")
        major, minor = int(major), int(minor)
        if (major >= 1) and (minor >= 2):
            del lines[10]

        table = "\n".join(lines)
        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        table = _strip_multiple_chars(table, " ")
        table = _strip_multiple_chars(table, "-")
        return table

    def test_default(self, model_card, expected):
        result = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).content

        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_other_section(self, model_card, expected):
        model_card.add_hyperparams(section="Other section")
        result = model_card.select("Other section").content

        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_other_description(self, model_card, expected):
        model_card.add_hyperparams(description="Awesome hyperparams")
        result = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).content
        assert result.startswith("Awesome hyperparams")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_raises(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        msg = (
            "You are trying to add model hyperparameters but you're using a custom "
            "template, please pass the 'section' argument to determine where to put "
            "the content"
        )
        with pytest.raises(ValueError, match=msg):
            model_card.add_hyperparams()

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        text1 = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).content
        model_card.add_hyperparams(section="Other section")
        text2 = model_card.select("Other section").content

        assert text1 == text2

    def test_hyperparameter_table_with_line_break(self):
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


class TestAddMetrics:
    """Tests for adding metrics"""

    def test_default(self, model_card):
        # by default, don't add a table, as there are no metrics
        result = model_card.select("Model description/Evaluation Results").content
        expected = "[More Information Needed]"
        assert result == expected

    def test_empty_metrics_table(self, model_card):
        model_card.add_metrics()
        result = model_card.select("Model description/Evaluation Results").content
        expected = (
            "You can find the details about evaluation process and the evaluation "
            "results.\n\n"
            "| Metric   | Value   |\n"
            "|----------|---------|"
        )
        assert result == expected

    def test_multiple_metrics(self, model_card):
        model_card.add_metrics(**{"acc": "0.1"})  # str
        model_card.add_metrics(
            f1=0.1,  # float
            awesomeness=123,  # int
        )
        result = model_card.select("Model description/Evaluation Results").content
        expected = (
            "You can find the details about evaluation process and the evaluation "
            "results.\n\n"
            "| Metric      |   Value |\n"
            "|-------------|---------|\n"
            "| acc         |     0.1 |\n"
            "| f1          |     0.1 |\n"
            "| awesomeness |   123   |"
        )
        assert result == expected

    def test_other_section(self, model_card):
        model_card.add_metrics(accuracy=0.9, section="Other section")
        result = model_card.select("Other section").content
        expected = (
            "You can find the details about evaluation process and the evaluation "
            "results.\n\n"
            "| Metric   |   Value |\n"
            "|----------|---------|\n"
            "| accuracy |     0.9 |"
        )
        assert result == expected

    def test_other_description(self, model_card):
        model_card.add_metrics(accuracy=0.9, description="Awesome metrics")
        result = model_card.select("Model description/Evaluation Results").content
        assert result.startswith("Awesome metrics")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_raises(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        msg = (
            "You are trying to add metrics but you're using a custom template, "
            "please pass the 'section' argument to determine where to put the content"
        )
        with pytest.raises(ValueError, match=msg):
            model_card.add_metrics(accuracy=0.9)

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        model_card.add_metrics(accuracy=0.9)
        text1 = model_card.select("Model description/Evaluation Results").content
        model_card.add_metrics(section="Other section")
        text2 = model_card.select("Other section").content
        assert text1 == text2


class TestAddGetStartedCode:
    """Tests for getting started code"""

    @pytest.fixture
    def metadata(self):
        # dummy ModelCardData using pickle
        class Metadata:
            def to_dict(self):
                return {
                    "model_file": "my-model.pickle",
                    "sklearn": {
                        "model_format": "pickle",
                    },
                }

        return Metadata()

    @pytest.fixture
    def model_card(self, metadata):
        model = fit_model()
        card = Card(model, metadata=metadata)
        return card

    @pytest.fixture
    def metadata_skops(self):
        # dummy ModelCardData using skops
        class Metadata:
            def to_dict(self):
                return {
                    "model_file": "my-model.skops",
                    "sklearn": {
                        "model_format": "skops",
                    },
                }

        return Metadata()

    @pytest.fixture
    def model_card_skops(self, metadata_skops):
        model = fit_model()
        card = Card(model, metadata=metadata_skops)
        return card

    def test_default_pickle(self, model_card):
        # by default, don't add a table, as there are no metrics
        result = model_card.select("How to Get Started with the Model").content
        expected = (
            "Use the code below to get started with the model.\n\n"
            "```python\n"
            "import json\n"
            "import pandas as pd\n"
            "import joblib\n"
            'model = joblib.load("my-model.pickle")\n'
            'with open("config.json") as f:\n'
            "    config = json.load(f)\n"
            'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))\n'
            "```"
        )
        assert result == expected

    def test_default_skops(self, model_card_skops):
        # by default, don't add a table, as there are no metrics
        result = model_card_skops.select("How to Get Started with the Model").content
        expected = (
            "Use the code below to get started with the model.\n\n"
            "```python\n"
            "import json\n"
            "import pandas as pd\n"
            "import skops.io as sio\n"
            'model = sio.load("my-model.skops")\n'
            'with open("config.json") as f:\n'
            "    config = json.load(f)\n"
            'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))\n'
            "```"
        )
        assert result == expected

    def test_no_metadata_file_name(self):
        model = fit_model()
        card = Card(model, metadata=None)
        card.add_get_started_code()  # does not raise

    def test_no_metadata_file_format(self):
        class Metadata:
            def to_dict(self):
                return {
                    "model_file": "my-model.skops",
                    # missing file format entry
                }

        model = fit_model()
        card = Card(model, metadata=Metadata())
        card.add_get_started_code()  # does not raise

    def test_other_section(self, model_card):
        model_card.add_get_started_code(section="Other section")
        result = model_card.select("Other section").content
        expected = "Use the code below to get started with the model."
        assert result.startswith(expected)

    def test_other_description(self, model_card):
        model_card.add_get_started_code(description="Awesome code")
        result = model_card.select("How to Get Started with the Model").content
        assert result.startswith("Awesome code")

    def test_other_filename(self, model_card):
        model_card.add_get_started_code(file_name="foobar.pkl")
        result = model_card.select("How to Get Started with the Model").content
        expected = (
            "Use the code below to get started with the model.\n\n"
            "```python\n"
            "import json\n"
            "import pandas as pd\n"
            "import joblib\n"
            'model = joblib.load("foobar.pkl")\n'
            'with open("config.json") as f:\n'
            "    config = json.load(f)\n"
            'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))\n'
            "```"
        )
        assert result == expected

    def test_other_model_format(self, model_card):
        model_card.add_get_started_code(model_format="skops")
        result = model_card.select("How to Get Started with the Model").content
        expected = (
            "Use the code below to get started with the model.\n\n"
            "```python\n"
            "import json\n"
            "import pandas as pd\n"
            "import skops.io as sio\n"
            'model = sio.load("my-model.pickle")\n'
            'with open("config.json") as f:\n'
            "    config = json.load(f)\n"
            'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))\n'
            "```"
        )
        assert result == expected

    def test_invalid_model_format_passed(self, model_card):
        # json is not a valid model format
        msg = "Invalid model format 'json', should be one of 'pickle' or 'skops'"
        with pytest.raises(ValueError, match=msg):
            model_card.add_get_started_code(model_format="json")

    def test_invalid_model_format_passed_via_metadata(self):
        # metadata contains invalid model format json
        class Metadata:
            def to_dict(self):
                return {
                    "model_file": "my-model.skops",
                    "sklearn": {
                        "model_format": "json",
                    },
                }

        model = fit_model()

        msg = "Invalid model format 'json', should be one of 'pickle' or 'skops'"
        with pytest.raises(ValueError, match=msg):
            Card(model, metadata=Metadata())

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_raises(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        msg = (
            "You are trying to add get started code but you're using a custom template,"
            " please pass the 'section' argument to determine where to put the content"
        )
        with pytest.raises(ValueError, match=msg):
            model_card.add_get_started_code(file_name="foo.bar", model_format="skops")

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        text1 = model_card.select("How to Get Started with the Model").content
        model_card.add_get_started_code(section="Other section")
        text2 = model_card.select("Other section").content
        assert text1 == text2


class TestRender:
    def test_render(self, model_card, destination_path):
        file_name = destination_path / "README.md"
        model_card.save(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            loaded = f.read()

        rendered = model_card.render()
        assert loaded == rendered

    def test_render_with_metadata(self, model_card):
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

        section = model_card.select("Model description").select("Training Procedure")
        assert section.title == "Training Procedure"

    def test_select_existing_subsubsection(self, model_card):
        section = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        )
        assert section.title == "Hyperparameters"

        section = (
            model_card.select("Model description")
            .select("Training Procedure")
            .select("Hyperparameters")
        )
        assert section.title == "Hyperparameters"

    def test_select_non_existing_section_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select("non-existing section")

    def test_select_non_existing_subsection_raises(self, model_card):
        with pytest.raises(KeyError):
            model_card.select("Model description/non-existing subsection")

        with pytest.raises(KeyError):
            model_card.select("Model description").select("non-existing subsection")

    def test_select_non_existing_subsubsection_raises(self, model_card):
        msg = "non-existing sub-subsection"

        with pytest.raises(KeyError, match=msg):
            model_card.select(
                "Model description/Training Procedure/non-existing sub-subsection"
            )

        with pytest.raises(KeyError, match=msg):
            (
                model_card.select("Model description")
                .select("Training Procedure")
                .select("non-existing sub-subsection")
            )

    def test_select_non_existing_section_and_subsection_raises(self, model_card):
        msg = "non-existing section"

        with pytest.raises(KeyError, match=msg):
            model_card.select("non-existing section/non-existing subsection")

        with pytest.raises(KeyError, match=msg):
            model_card.select("non-existing section").select("non-existing subsection")

    def test_select_empty_key_raises(self, model_card):
        msg = r"Section name cannot be empty but got ''"
        with pytest.raises(KeyError, match=msg):
            model_card.select("")

    def test_select_empty_key_subsection_raises(self, model_card):
        msg = r"Section name cannot be empty but got 'Model description/'"
        with pytest.raises(KeyError, match=msg):
            model_card.select("Model description/")

        msg = r"Section name cannot be empty but got ''"
        with pytest.raises(KeyError, match=msg):
            model_card.select("Model description").select("")

    def test_default_skops_sections_present(self, model_card):
        # model_card (which is prefilled) contains all default sections
        for key in SKOPS_TEMPLATE:
            model_card.select(key)

    def test_default_skops_sections_empty_card(self, model_card):
        # Without prefilled template, the card should not contain the default sections

        # empty card does not contain any sections, so trying to select them
        # should raise a KeyError
        model = fit_model()
        card_empty = Card(model, model_diagram=False, template=None)
        for key in SKOPS_TEMPLATE:
            with pytest.raises(KeyError):
                card_empty.select(key)

    def test_invalid_template_name_raises(self):
        msg = "Unknown template 'does-not-exist', template must be one of the following"
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


class TestAddPlot:
    def test_add_plot(self, destination_path, model_card):
        # don't import matplotlib at root or else other tests may be affected
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(fig1="fig1.png")
        plot_content = model_card.select("fig1").content.format()
        assert plot_content == "![fig1](fig1.png)"

    def test_add_plot_to_existing_section(self, destination_path, model_card):
        # don't import matplotlib at root or else other tests may be affected
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(**{"Model description/Figure 1": "fig1.png"})
        plot_content = model_card.select("Model description/Figure 1").content.format()
        assert plot_content == "![Figure 1](fig1.png)"


class TestMetadata:
    def test_adding_metadata(self, model_card):
        # test if the metadata is added to the card
        model_card.metadata.tags = "dummy"
        metadata = list(model_card._generate_metadata(model_card.metadata))
        assert len(metadata) == 1
        assert metadata[0] == "metadata.tags=dummy,"

    def test_metadata_from_config_tabular_data(
        self, pkl_model_card_metadata_from_config, destination_path
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


@pytest.mark.xfail(reason="dynamic adjustment when model changes not implemented yet")
class TestModelDynamicUpdate:
    def test_model_related_sections_updated_dynamically_skops_template(
        self, model_card
    ):
        # Change the model to be a KNN classifier and check that the sections
        # related to the model, the plot and hyperparams, are updated correctly.
        # But first, as a sanity check, ensure that before the change, there is
        # no reference to KNN.
        model_plot_before = model_card.select(
            "Model description/Training Procedure/Model Plot"
        )
        assert "KNeighborsClassifier" not in model_plot_before.content

        hyperparams_before = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        )
        assert "n_neighbors" not in hyperparams_before.content.format()

        # change model to KNN
        model_after = KNeighborsClassifier()
        model_card.model = model_after

        model_plot_after = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).content
        assert "KNeighborsClassifier" in model_plot_after

        hyperparams_after = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        )
        assert "n_neighbors" in hyperparams_after.content.format()

    def test_model_related_sections_updated_dynamically_custom_template(
        self, model_card
    ):
        # same as previous test but using a custom template
        template = {"My model plot": ""}
        model = fit_model()
        model_card = Card(model, template=template)

        # add model plot and hyperparams -- section must be passed but it
        # doesn't need to already exist in the custom template
        model_card.add_model_plot(section="My model plot")
        model_card.add_hyperparams(section="My hyperparams")

        model_plot_before = model_card.select("My model plot")
        assert "KNeighborsClassifier" not in model_plot_before.content

        hyperparams_before = model_card.select("My hyperparams")
        assert "n_neighbors" not in hyperparams_before.content.format()

        # change model to KNN
        model_after = KNeighborsClassifier()
        model_card.model = model_after

        model_plot_after = model_card.select("My model plot")
        assert "KNeighborsClassifier" in model_plot_after.content

        hyperparams_after = model_card.select("My hyperparams")
        assert "n_neighbors" in hyperparams_after.content.format()


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


class TestCardModelAttributeIsPath:
    def path_to_card(self, path):
        card = Card(model=path, trusted=True)
        return card

    @pytest.mark.parametrize("meth", [repr, str])
    @pytest.mark.parametrize("suffix", [".pkl", ".skops"])
    def test_model_card_repr(self, meth, suffix):
        # Test that if the model is changed, Card takes this into account, if
        # the model argument is a path to a model file. First, we test that if
        # the model path changes, the Card changes. Then we test that if the
        # file on disk changes, the Card changes.
        model = LinearRegression(fit_intercept=False)
        file_handle, file_name = save_model_to_file(model, suffix)
        os.close(file_handle)
        card_from_path = self.path_to_card(file_name)

        result0 = meth(card_from_path)
        expected = "Card(\n  model=LinearRegression(fit_intercept=False),"
        assert result0.startswith(expected)

        # change file name, same card should show different result
        model = LinearRegression()
        file_handle, file_name = save_model_to_file(model, suffix)
        card_from_path.model = file_name
        result1 = meth(card_from_path)
        expected = "Card(\n  model=LinearRegression(),"
        assert result1.startswith(expected)

        # change model on disk but keep same file name, should show different
        # result
        model = LinearRegression(fit_intercept=None)
        with open(file_name, "wb") as f:
            dump_fn = pickle.dump if suffix == ".pkl" else dump
            dump_fn(model, f)
        result2 = meth(card_from_path)
        expected = "Card(\n  model=LinearRegression(fit_intercept=None),"
        assert result2.startswith(expected)

    @pytest.mark.parametrize("suffix", [".pkl", ".skops"])
    @pytest.mark.parametrize("meth", [repr, str])
    def test_load_model_exception(self, meth, suffix):
        file_handle, file_name = tempfile.mkstemp(suffix=suffix, prefix="skops-test")

        os.close(file_handle)

        with pytest.raises(Exception, match="occurred during model loading."):
            card = Card(file_name)
            meth(card)

    @pytest.mark.parametrize("meth", [repr, str])
    def test_load_model_file_not_found(self, meth):
        file_handle, file_name = tempfile.mkstemp(suffix=".pkl", prefix="skops-test")

        os.close(file_handle)
        os.remove(file_name)

        with pytest.raises(FileNotFoundError) as excinfo:
            card = Card(file_name)
            meth(card)

        assert file_name in str(excinfo.value)


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


class TestCustomTemplate:
    @pytest.fixture
    def template(self):
        return {
            "My description": "An awesome model",
            "Model": "Here goes model related stuff",
            "Model/Metrics": "",
            "Foo/Bar": "Baz",
        }

    @pytest.fixture
    def card(self, template):
        model = fit_model()
        card = Card(model, template=template)
        return card

    def test_add_model_plot(self, card):
        card.add_model_plot(section="Model/Model plot")
        content = card.select("Model/Model plot").content
        assert "LinearRegression" in content

    def test_add_hyperparams(self, card):
        card.add_hyperparams(section="Model/Hyperparams")
        content = card.select("Model/Hyperparams").content
        assert "fit_intercept" in content

    def test_add_metrics(self, card):
        card.add_metrics(accuracy=0.1, section="Model/Metrics")
        content = card.select("Model/Metrics").content
        assert "accuracy" in content
        assert "0.1" in content

    def test_add_get_started_code(self, card):
        card.add_get_started_code(
            section="Getting Started",
            file_name="foobar.skops",
            model_format="skops",
        )
        content = card.select("Getting Started").content
        assert "load" in content

    def test_custom_template_all_sections_present(self, template, card):
        # model_card contains all default sections
        for key in template:
            card.select(key)

        # no other top level sections as those defined in the template
        expected = ["My description", "Model", "Foo"]
        assert list(card._data.keys()) == expected
