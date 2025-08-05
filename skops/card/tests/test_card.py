import os
import pickle
import re
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier

from skops.card import Card
from skops.card._model_card import (
    CONTENT_PLACEHOLDER,
    SKOPS_TEMPLATE,
    PlotSection,
    Section,
    TableSection,
    _load_model,
)
from skops.io import dump, get_untrusted_types, load
from skops.utils.importutils import import_or_raise


class MyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, param_1=1, param_2="string", param_3=None):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.ones(X.shape[0])


class MyRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, param_1=1, param_2="value", param_3=None):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def fit(self, X, y):
        return self


def fit_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = MyRegressor().fit(X, y)
    return reg


def save_model_to_file(model_instance, suffix):
    save_file_handle, save_file = tempfile.mkstemp(suffix=suffix, prefix="skops-test")
    if suffix in (".pkl", ".pickle"):
        with open(save_file, "wb") as f:
            pickle.dump(model_instance, f)
    elif suffix == ".skops":
        dump(model_instance, save_file)
    return save_file_handle, save_file


def reprs_equal(repr1, repr2):
    """Check that repr1 and repr2 are basically equal.

    This ignores line order of what comes after the first and before the last line.
    """
    return sorted(repr1.split("\n")[1:-1]) == sorted(repr2.split("\n")[1:-1])


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".skops"])
def test_load_model(suffix):
    model0 = MyRegressor(param_1=10)
    _, save_file = save_model_to_file(model0, suffix)
    if suffix == ".skops":
        untrusted_types = get_untrusted_types(file=save_file)
        allow_pickle = False
    else:
        untrusted_types = None
        allow_pickle = True
    loaded_model_str = _load_model(
        save_file, trusted=untrusted_types, allow_pickle=allow_pickle
    )
    save_file_path = Path(save_file)
    loaded_model_path = _load_model(
        save_file_path, trusted=untrusted_types, allow_pickle=allow_pickle
    )
    loaded_model_instance = _load_model(model0, trusted=untrusted_types)

    assert loaded_model_str.param_1 == 10
    assert loaded_model_path.param_1 == 10
    assert loaded_model_instance.param_1 == 10


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
def test_load_model_exception_allow_pickle(suffix):
    model0 = MyRegressor(param_1=10)
    _, save_file = save_model_to_file(model0, suffix)

    with pytest.raises(
        RuntimeError,
        match=(
            "Model file is not a skops file, and allow_pickle is set to False. "
            "Please set allow_pickle=True to load the model."
            "This may lead to security issues if the model file is not trustworthy."
        ),
    ):
        _load_model(save_file, trusted=None, allow_pickle=False)

    with pytest.raises(
        ValueError,
        match=(
            "`allow_pickle` cannot be `True` if `trusted` is not empty. "
            "Pickles cannot be trusted or checked for security issues."
        ),
    ):
        _load_model(save_file, trusted=[""], allow_pickle=True)


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
    est = MyClassifier().fit(X, y)
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
):
    card = Card(iris_estimator)
    card.save(Path(destination_path) / "README.md")
    return card


@pytest.fixture
def skops_model_card(destination_path, iris_estimator):
    yield _create_model_card_from_saved_model(destination_path, iris_estimator)


@pytest.fixture
def pkl_model_card(destination_path, iris_estimator):
    yield _create_model_card_from_saved_model(destination_path, iris_estimator)


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


def test_save_model_card(destination_path, model_card):
    model_card.save(Path(destination_path) / "README.md")
    assert (Path(destination_path) / "README.md").exists()


def test_model_caching(skops_model_card, iris_skops_file, destination_path):
    """Tests that the model card caches the model to avoid loading it multiple times"""

    new_model = MyClassifier(param_1=10)
    # mock _load_model, it still loads the model but we can track call count
    mock_load_model = mock.Mock(
        side_effect=lambda path, trusted, _: load(path, trusted=trusted)
    )
    card = Card(iris_skops_file, trusted=[MyClassifier])
    with mock.patch("skops.card._model_card._load_model", mock_load_model):
        model1 = card.get_model()
        model2 = card.get_model()
        assert model1 is model2
        # model is cached, hence _load_model is not called
        mock_load_model.assert_not_called()

        # override model with new model
        dump(new_model, card.model)

        model3 = card.get_model()
        assert mock_load_model.call_count == 1
        assert model3.param_1 == 10
        model4 = card.get_model()

        assert model3 is model4
        assert mock_load_model.call_count == 1  # cached call


CUSTOM_TEMPLATES = [None, {}, {"A Title", "Another Title", "A Title/A Section"}]  # type: ignore


class TestAddModelPlot:
    """Tests for the sklearn model repr"""

    def test_default(self, model_card):
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        # don't compare whole text, as it's quite long and non-deterministic
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    def test_no_overflow(self, model_card):
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        # test if the model doesn't overflow the huggingface models page
        assert result.count("sk-top-container") == 1
        assert 'style="overflow: auto;' in result

    def test_model_diagram_false(self):
        model = fit_model()
        model_card = Card(model, model_diagram=False)
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        assert result == CONTENT_PLACEHOLDER

    def test_model_diagram_str(self):
        # if passing a str, use that as the section name
        model = fit_model()
        other_section_name = "Here is the model diagram"
        model_card = Card(model, model_diagram=other_section_name)

        # first check that default section only contains placeholder
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        assert result == CONTENT_PLACEHOLDER

        # now check that the actual model diagram is in the other section
        result = model_card.select(other_section_name).format()
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    def test_other_section(self, model_card):
        model_card.add_model_plot(section="Other section")
        result = model_card.select("Other section").content
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    def test_with_description(self, model_card):
        model_card.add_model_plot(description="Awesome diagram below")
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        assert result.startswith("Awesome diagram below\n\n<style>#sk-")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_uses_default(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        model_card.add_model_plot()
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()

        # don't compare whole text, as it's quite long and non-deterministic
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_init_str_works(self, template):
        model = fit_model()
        section_name = "Here is the model diagram"
        model_card = Card(model, template=template, model_diagram=section_name)

        result = model_card.select(section_name).format()
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    def test_default_template_and_model_diagram_true(self, model_card):
        # setting model_diagram=True should not change anything vs auto with the
        # default template
        model = fit_model()
        model_card = Card(model, model_diagram=True)
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        # don't compare whole text, as it's quite long and non-deterministic
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_and_model_diagram_true_uses_default(
        self, model_card, template
    ):
        # when using a custom template and not indicating a section, use the
        # default section
        model = fit_model()
        Card(model, template=template, model_diagram=True)
        result = model_card.select(
            "Model description/Training Procedure/Model Plot"
        ).format()
        # don't compare whole text, as it's quite long and non-deterministic
        assert result.startswith("<style>#sk-")
        assert "<style>" in result
        assert "MyRegressor()" in result

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
            "<details>",
            "<summary> Click to expand </summary>",
            "",
            "| Hyperparameter | Value |",
            "| :------------: | :---: |",
            "|    param_1     |   1   |",
            "|    param_2     | value |",
            "|    param_3     |  None |",
            "",
            "</details>",
        ]

        table = "\n".join(lines)
        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        table = _strip_multiple_chars(table, " ")
        table = _strip_multiple_chars(table, "-")
        return table

    def test_default(self, model_card, expected):
        result = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).format()
        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_other_section(self, model_card, expected):
        model_card.add_hyperparams(section="Other section")
        result = model_card.select("Other section").format()

        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_with_description(self, model_card, expected):
        model_card.add_hyperparams(description="Awesome hyperparams")
        result = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).format()
        assert result.startswith("Awesome hyperparams")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_uses_default(self, template, expected):
        model = fit_model()
        model_card = Card(model, template=template)
        model_card.add_hyperparams()
        result = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).format()

        # remove multiple whitespaces and dashes, as they're not important and may
        # differ depending on OS
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        text1 = model_card.select(
            "Model description/Training Procedure/Hyperparameters"
        ).format()
        model_card.add_hyperparams(section="Other section")
        text2 = model_card.select("Other section").format()

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
        text_hyperparams = model_card.select(section_name).format()

        # remove multiple whitespaces, as they're not important
        text_cleaned = _strip_multiple_chars(text_hyperparams, " ")
        assert "| n_jobs | line<br />with<br />break |" in text_cleaned


class TestAddMetrics:
    """Tests for adding metrics"""

    def test_default(self, model_card):
        # by default, don't add a table, as there are no metrics
        result = model_card.select("Model description/Evaluation Results").format()
        expected = CONTENT_PLACEHOLDER
        assert result == expected

    def test_empty_metrics_table(self, model_card):
        model_card.add_metrics()
        result = model_card.select("Model description/Evaluation Results").format()
        expected = (
            "| Metric | Value |\n"  # fmt: skip
            "| :----: | :---: |"
        )
        assert result == expected

    def test_multiple_metrics(self, model_card):
        model_card.add_metrics(**{"acc": "0.1"})  # str
        model_card.add_metrics(
            f1=0.1,  # float
            awesomeness=123,  # int
        )
        result = model_card.select("Model description/Evaluation Results").format()
        expected = (
            "|    Metric   | Value |\n"
            "| :---------: | :---: |\n"
            "|     acc     |  0.1  |\n"
            "|      f1     |  0.1  |\n"
            "| awesomeness |  123  |"
        )
        assert result == expected

    def test_other_section(self, model_card):
        model_card.add_metrics(accuracy=0.9, section="Other section")
        result = model_card.select("Other section").format()
        # fmt: off
        expected = (
            "|  Metric  | Value |\n"
            "| :------: | :---: |\n"
            "| accuracy |  0.9  |"
        )
        # fmt: on
        assert result == expected

    def test_with_description(self, model_card):
        model_card.add_metrics(accuracy=0.9, description="Awesome metrics")
        result = model_card.select("Model description/Evaluation Results").format()
        assert result.startswith("Awesome metrics\n\n|  Metric ")

    @pytest.mark.parametrize("template", CUSTOM_TEMPLATES)
    def test_custom_template_no_section_raises(self, template):
        model = fit_model()
        model_card = Card(model, template=template)
        model_card.add_metrics(accuracy=0.9)

        result = model_card.select("Model description/Evaluation Results").format()
        # fmt: off
        expected = (
            "|  Metric  | Value |\n"
            "| :------: | :---: |\n"
            "| accuracy |  0.9  |"
        )
        # fmt: on
        assert result == expected

    def test_add_twice(self, model_card):
        # it's possible to add the section twice, even if it doesn't make a lot
        # of sense
        model_card.add_metrics(accuracy=0.9)
        text1 = model_card.select("Model description/Evaluation Results").format()
        model_card.add_metrics(section="Other section")
        text2 = model_card.select("Other section").format()
        assert text1 == text2


class TestAddPermutationImportance:
    @pytest.fixture
    def importances(self, iris_estimator, iris_data):
        X, y = iris_data
        result = permutation_importance(
            iris_estimator, X, y, n_repeats=10, random_state=42, n_jobs=2
        )
        return result

    def test_permutation_importances(
        self, iris_data, importances, model_card, destination_path
    ):
        X, _ = iris_data
        model_card.add_permutation_importances(
            importances,
            columns=X.columns,
            plot_file=Path(destination_path) / "importance.png",
            plot_name="Permutation Importance",
        )
        temp_path = Path(destination_path) / "importance.png"
        section = model_card.select("Permutation Importance")
        expected = f"![Permutation Importance]({temp_path})"
        assert section.format() == expected

    def test_multiple_permutation_importances(
        self, iris_data, iris_estimator, importances, model_card, destination_path
    ):
        X, y = iris_data
        model_card.add_permutation_importances(
            importances, X.columns, plot_file=Path(destination_path) / "importance.png"
        )

        f1 = make_scorer(f1_score, average="micro")
        importances_f1 = permutation_importance(
            iris_estimator, X, y, scoring=f1, n_repeats=10, random_state=42, n_jobs=2
        )
        model_card.add_permutation_importances(
            importances_f1,
            columns=X.columns,
            plot_file=Path(destination_path) / "f1_importance.png",
            plot_name="Permutation Importance on f1",
        )

        # check for default one
        temp_path = Path(destination_path) / "importance.png"
        section = model_card.select("Permutation Importances")
        expected = f"![Permutation Importances]({temp_path})"
        assert section.format() == expected

        # check for F1
        temp_path_f1 = Path(destination_path) / "f1_importance.png"
        section = model_card.select("Permutation Importance on f1")
        expected = f"![Permutation Importance on f1]({temp_path_f1})"
        assert section.format() == expected

    def test_duplicate_permutation_importances(
        self, iris_data, importances, model_card, destination_path
    ):
        X, _ = iris_data
        plot_path = os.path.join(destination_path, "importance.png")
        model_card.add_permutation_importances(
            importances, X.columns, plot_file=plot_path
        )
        with pytest.raises(
            ValueError,
            match=(
                "already exists. Set `overwrite` to `True` or pass a"
                " different filename for the plot."
            ),
        ):
            model_card.add_permutation_importances(
                importances,
                columns=X.columns,
                plot_file=plot_path,
                plot_name="Permutation Importance on f1",
            )

    def test_duplicate_permutation_importances_overwrite(
        self, iris_data, importances, model_card, destination_path
    ):
        X, _ = iris_data
        plot_path = os.path.join(destination_path, "importance.png")
        model_card.add_permutation_importances(
            importances, X.columns, plot_file=plot_path
        )

        model_card.add_permutation_importances(
            importances,
            columns=X.columns,
            plot_file=plot_path,
            plot_name="Permutation Importance on f1",
            overwrite=True,
        )
        section = model_card.select("Permutation Importance on f1")
        expected = f"![Permutation Importance on f1]({plot_path})"
        assert section.format() == expected

    def test_permutation_importances_with_description(
        self, iris_data, importances, model_card, destination_path
    ):
        X, _ = iris_data
        model_card.add_permutation_importances(
            importances,
            columns=X.columns,
            plot_file=Path(destination_path) / "importance.png",
            description="Very important",
        )
        temp_path = Path(destination_path) / "importance.png"
        section = model_card.select("Permutation Importances")
        expected = f"Very important\n\n![Permutation Importances]({temp_path})"
        assert section.format() == expected


class TestRender:
    def test_render(self, model_card, destination_path):
        file_name = destination_path / "README.md"
        model_card.save(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            loaded = f.read()

        rendered = model_card.render()
        assert loaded == rendered


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

    @pytest.mark.parametrize("folded", [True, False])
    def test_add_new_section(self, model_card, folded):
        model_card = model_card.add(**{"A new section": "sklearn FTW"}, folded=folded)
        section = model_card.select("A new section")
        assert section.title == "A new section"
        assert section.content == "sklearn FTW"

        output = section.format()
        if folded:
            assert "<details>" in output
        else:
            assert "<details>" not in output

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

    def test_add_plain_section_works(self, model_card):
        # It is allowed to add a *Section object, but it's not documented and
        # users should normally not use that feature
        section = Section("title may differ from section name", "some content")
        model_card.add(
            a_string="normal string",
            a_section=section,
        )
        assert model_card.select("a_section") == section

    def test_add_section_preserves_subsections(self, model_card):
        # As explained in the previous test, users can theoretically add section
        # instances. If they override an existing section with a new section,
        # the subsections of the existing section should be preserved.

        # first let's add a section and a subsection
        model_card.add(**{"new section": "hello", "new section/subsection": "world"})
        assert model_card.select("new section").format() == "hello"
        assert model_card.select("new section/subsection").format() == "world"

        # now let's override the section, the subsection should be preserved
        new_section = Section("new section", "bonjour")
        model_card.add(**{"new section": new_section})
        assert model_card.select("new section").format() == "bonjour"
        assert model_card.select("new section/subsection").format() == "world"

    def test_add_section_with_identical_subsection_preserves_subsections(
        self, model_card
    ):
        # As explained in the previous tests, users can theoretically add
        # section instances. If they override an existing section with a new
        # section, the subsections of the existing section should be preserved.
        # If the new section they add has its own subsections, and these
        # subsections are identical to the old subsections, that should be fine.

        # first let's add a section and a subsection
        model_card.add(**{"new section": "hello", "new section/subsection": "world"})

        # now let's override the section using the same subsections
        old_subsection = model_card.select("new section").subsections
        new_section = Section("new section", "bonjour", subsections=old_subsection)
        model_card.add(**{"new section": new_section})
        assert model_card.select("new section").format() == "bonjour"
        assert model_card.select("new section/subsection").format() == "world"

    def test_add_section_with_different_subsection_raises(self, model_card):
        # This is the same as the previous test, but now the section used to
        # override the previous section has different subsections. Now we don't
        # know what to do and should raise. This is okay because normally, a
        # user shouldn't add section instances anyway.

        # first let's add a section and a subsection
        model_card.add(**{"new section": "hello", "new section/subsection": "world"})

        # now let's override the section using different subsections
        new_subsection = {"new subsection": Section("subsection", "mars")}
        new_section = Section("new section", "bonjour", subsections=new_subsection)

        match = (
            "Trying to override section 'new section' but found conflicting subsections"
        )
        with pytest.raises(ValueError, match=match):
            model_card.add(**{"new section": new_section})


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
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(fig1="fig1.png")
        plot_content = model_card.select("fig1").format()
        assert plot_content == "![fig1](fig1.png)"

    def test_add_plot_to_existing_section(self, destination_path, model_card):
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(**{"Model description/Figure 1": "fig1.png"})
        plot_content = model_card.select("Model description/Figure 1").format()
        assert plot_content == "![Figure 1](fig1.png)"

    def test_add_plot_with_description(self, destination_path, model_card):
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(description="My fancy plot", fig1="fig1.png")
        plot_content = model_card.select("fig1").format()
        assert plot_content == "My fancy plot\n\n![fig1](fig1.png)"

    def test_add_plot_with_alt_text(self, destination_path, model_card):
        import matplotlib.pyplot as plt

        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        model_card = model_card.add_plot(alt_text="the figure", fig1="fig1.png")
        plot_content = model_card.select("fig1").format()
        assert plot_content == "![the figure](fig1.png)"


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
        model = MyRegressor(param_1=10)
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
          model=MyRegressor(param_1=10),
          Model description/Training Procedure/Hyperparameters=TableSection(3x2),
          Model description/Training Procedure/Model Plot=__anything__,
          Model Card Authors=Jane Doe,
          Figures/ROC=PlotSection(ROC.png),
          Figures/Confusion matrix=PlotSection(confusion_matrix.jpg),
          Model Description=A description,
          Search Results=TableSection(3x2),
        )
        """  # noqa: E501
        expected = textwrap.dedent(card_repr).strip()
        expected = re.escape(expected)
        expected = expected.replace("__anything__", ".*")
        lines = expected.split("\n")
        return lines

    @pytest.mark.parametrize("meth", [repr, str])
    def test_card_repr(self, card: Card, meth, expected_lines):
        result = meth(card)
        expected = "\n".join(expected_lines)
        assert re.match(expected, result)

    @pytest.mark.parametrize("meth", [repr, str])
    def test_card_repr_empty_card(self, meth):
        """Without prefilled template, the repr should be empty"""
        model = fit_model()
        card = Card(model, model_diagram=False, template=None)
        result = meth(card)
        expected = textwrap.dedent("""
        Card(
          model=MyRegressor(),
        )
        """).strip()
        assert result == expected

    @pytest.mark.parametrize("meth", [repr, str])
    def test_very_long_lines_are_shortened(self, card: Card, meth, expected_lines):
        card.add(my_section="very long line " * 100)

        # expected results contain 1 line at the very end
        extra_line = (
            "  my_section=very long line very long line ve...e very long line "
            "very long line ,"
        )
        expected_lines.insert(-1, extra_line)
        expected = "\n".join(expected_lines)

        result = meth(card)
        assert re.match(expected, result)

    @pytest.mark.parametrize("meth", [repr, str])
    def test_without_model_attribute(self, card: Card, meth, expected_lines):
        del card.model

        # remove line 1 from expected results, which corresponds to the model
        del expected_lines[1]
        expected = "\n".join(expected_lines)

        result = meth(card)
        assert re.match(expected, result)


class TestCardModelAttributeIsPath:
    def path_to_card(self, path, suffix):
        if suffix == ".skops":
            card = Card(model=path, trusted=get_untrusted_types(file=path))
        else:
            card = Card(model=path, allow_pickle=True)
        return card

    @pytest.mark.parametrize("meth", [repr, str])
    @pytest.mark.parametrize("suffix", [".pkl", ".skops"])
    def test_model_card_repr(self, meth, suffix):
        # Test that if the model is changed, Card takes this into account, if
        # the model argument is a path to a model file. First, we test that if
        # the model path changes, the Card changes. Then we test that if the
        # file on disk changes, the Card changes.
        model = MyRegressor(param_1=10)
        file_handle, file_name = save_model_to_file(model, suffix)
        os.close(file_handle)
        card_from_path = self.path_to_card(file_name, suffix=suffix)

        result0 = meth(card_from_path)
        expected = "Card(\n  model=MyRegressor(param_1=10),"
        assert result0.startswith(expected)

        # change file name, same card should show different result
        model = MyRegressor()
        file_handle, file_name = save_model_to_file(model, suffix)
        card_from_path.model = file_name
        result1 = meth(card_from_path)
        expected = "Card(\n  model=MyRegressor(),"
        assert result1.startswith(expected)

        # change model on disk but keep same file name, should show different
        # result
        model = MyRegressor(param_2="new value")
        with open(file_name, "wb") as f:
            dump_fn = pickle.dump if suffix == ".pkl" else dump
            dump_fn(model, f)
        result2 = meth(card_from_path)
        expected = "Card(\n  model=MyRegressor(param_2='new value'),"
        assert result2.startswith(expected)

    @pytest.mark.parametrize("suffix", [".pkl", ".skops"])
    @pytest.mark.parametrize("meth", [repr, str])
    def test_load_model_exception(self, meth, suffix):
        file_handle, file_name = tempfile.mkstemp(suffix=suffix, prefix="skops-test")

        os.close(file_handle)

        with pytest.raises(Exception, match="occurred during model loading."):
            card = Card(file_name, allow_pickle=True)
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
        section = PlotSection(
            title="", content="", alt_text="some title", path="path/plot.png"
        )
        expected = "![some title](path/plot.png)"
        assert section.format() == expected

    def test_format_path_is_pathlib(self):
        section = PlotSection(
            title="", content="", alt_text="some title", path=Path("path") / "plot.png"
        )
        expected = f"![some title](path{os.path.sep}plot.png)"
        assert section.format() == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr(self, meth):
        section = PlotSection(
            title="", content="", alt_text="some title", path="path/plot.png"
        )
        expected = "PlotSection(path/plot.png)"
        assert meth(section) == expected

    def test_str(self):
        section = PlotSection(
            title="", content="", alt_text="some title", path="path/plot.png"
        )
        expected = "PlotSection(path/plot.png)"
        assert str(section) == expected

    @pytest.mark.parametrize("folded", [True, False])
    def test_folded(self, folded):
        section = PlotSection(
            title="",
            content="",
            alt_text="some title",
            path="path/plot.png",
            folded=folded,
        )
        output = section.format()
        if folded:
            assert "<details>" in output
        else:
            assert "<details>" not in output

    def test_add_with_description(self):
        # FIXME
        pass


class TestTableSection:
    @pytest.fixture
    def table_dict(self):
        return {"split": [1, 2, 3], "score": [4, 5, 6]}

    def test_table_is_dict(self, table_dict):
        section = TableSection(title="", content="", table=table_dict)
        expected = (
            "| split | score |\n"
            "| :---: | :---: |\n"
            "|   1   |   4   |\n"
            "|   2   |   5   |\n"
            "|   3   |   6   |"
        )
        assert section.format() == expected

    def test_table_is_dataframe(self, table_dict):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(table_dict)
        section = TableSection(title="", content="", table=df)
        expected = (
            "| split | score |\n"
            "| :---: | :---: |\n"
            "|   1   |   4   |\n"
            "|   2   |   5   |\n"
            "|   3   |   6   |"
        )
        assert section.format() == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr_table_is_dict(self, table_dict, meth):
        section = TableSection(title="", content="", table=table_dict)
        expected = "TableSection(3x2)"
        assert meth(section) == expected

    @pytest.mark.parametrize("meth", [str, repr])
    def test_str_and_repr_table_is_dataframe(self, table_dict, meth):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(table_dict)
        section = TableSection(title="", content="", table=df)
        expected = "TableSection(3x2)"
        assert meth(section) == expected

    @pytest.mark.parametrize("table", [{}, "pandas"])
    def test_raise_error_empty_table(self, table):
        # If there are no columns, raise
        if table == "pandas":
            pd = pytest.importorskip("pandas")
            table = pd.DataFrame([])

        msg = "Trying to add table with no columns"
        with pytest.raises(ValueError, match=msg):
            TableSection(title="", content="", table=table)

    @pytest.mark.parametrize("table", [{"col0": []}, "pandas"])
    def test_table_with_no_rows_works(self, table):
        # If there are no rows, it's okay
        if table == "pandas":
            pd = pytest.importorskip("pandas")
            table = pd.DataFrame(data=[], columns=["col0"])

        TableSection(title="", content="", table=table).format()  # no error raised

    def test_pandas_not_installed(self, table_dict, pandas_not_installed):
        # use pandas_not_installed fixture from conftest.py to pretend that
        # pandas is not installed
        section = TableSection(title="", content="", table=table_dict)
        assert section._is_pandas_df is False

    @pytest.mark.parametrize("folded", [True, False])
    def test_folded(self, table_dict, folded):
        section = TableSection(title="", content="", table=table_dict, folded=folded)
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
        section = TableSection(title="", content="", table=table_dict)
        expected = (
            "| split | score | with break |\n"
            "| :-: | :-: | :-: |\n"
            "| 1 | 4 | obj<br />with lb |\n"
            "| 2 | 5 | hi<br />there |\n"
            "| 3 | 6 | <br />entry with<br />line breaks<br /> |"
        )

        result = section.format()
        # remove multiple whitespaces and dashes, as they're not important
        result = _strip_multiple_chars(result, " ")
        result = _strip_multiple_chars(result, "-")
        assert result == expected

    def test_add_table_with_description(self, model_card, table_dict):
        model_card.add_table(description="My fancy table", **{"The table": table_dict})
        section = model_card.select("The table")
        content = section.format()
        expected = (
            "My fancy table\n\n"
            "| split | score |\n"
            "| :---: | :---: |\n"
            "|   1   |   4   |\n"
            "|   2   |   5   |\n"
            "|   3   |   6   |"
        )
        assert content == expected


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
        content = card.select("Model/Model plot").format()
        assert "MyRegressor" in content

    def test_add_hyperparams(self, card):
        card.add_hyperparams(section="Model/Hyperparams")
        content = card.select("Model/Hyperparams").format()
        assert "param_1" in content

    def test_add_metrics(self, card):
        card.add_metrics(accuracy=0.1, section="Model/Metrics")
        content = card.select("Model/Metrics").format()
        assert "accuracy" in content
        assert "0.1" in content

    def test_custom_template_all_sections_present(self, template, card):
        # model_card contains all default sections
        for key in template:
            card.select(key)

        # no other top level sections as those defined in the template
        expected = ["My description", "Model", "Foo"]
        assert list(card._data.keys()) == expected


class TestRenderedCardVisibility:
    """Check that visibility flag works

    Sections that are not visible should not be rendered, neither when calling
    model_card.render, nor when calling model_card.save.

    """

    @pytest.fixture
    def template(self):
        return {
            "Model": "Here goes model related stuff",
            "Model/Metrics": "123",
            "Model/Bar": "Baz",
            "Authors": "Jane Doe",
        }

    @pytest.fixture
    def card(self, template):
        model = fit_model()
        card = Card(model, template=template)
        return card

    def test_all_visible_by_default(self, card):
        rendered = card.render()
        expected = (
            "# Model\n\n"
            "Here goes model related stuff\n\n"
            "## Metrics\n\n"
            "123\n\n"
            "## Bar\n\n"
            "Baz\n\n"
            "# Authors\n\n"
            "Jane Doe"
        )
        assert rendered.strip() == expected

    def test_section_invisible(self, card):
        card.select("Model/Metrics").visible = False
        rendered = card.render()
        expected = (
            "# Model\n\n"
            "Here goes model related stuff\n\n"
            "## Bar\n\n"
            "Baz\n\n"
            "# Authors\n\n"
            "Jane Doe"
        )
        assert rendered.strip() == expected

    def test_restoring_visibility_works(self, card):
        card.select("Model/Metrics").visible = False
        card.select("Model/Metrics").visible = True
        expected = (
            "# Model\n\n"
            "Here goes model related stuff\n\n"
            "## Metrics\n\n"
            "123\n\n"
            "## Bar\n\n"
            "Baz\n\n"
            "# Authors\n\n"
            "Jane Doe"
        )
        rendered = card.render()
        assert rendered.strip() == expected

    def test_invisible_parent_section_hides_subsections(self, card):
        # By making the parent section "Model" invisible, all of the subsections
        # are also turned invisible
        card.select("Model").visible = False
        # fmt: off
        expected = (
            "# Authors\n\n"
            "Jane Doe"
        )
        # fmt: on
        rendered = card.render()
        assert rendered.strip() == expected

    def test_visibility_with_card_save(self, card):
        # Since .save and .render share the same functionality, it's not
        # necessary to repeat all the tests above with .save. Just do one test
        # to ensure that the same functionality is indeed being used.
        file = tempfile.mkstemp(suffix=".md", prefix="skops-model-card")[1]
        card.select("Model/Metrics").visible = False
        card.save(file)

        with open(file, "r") as f:
            loaded = f.read()

        expected = (
            "# Model\n\n"
            "Here goes model related stuff\n\n"
            "## Bar\n\n"
            "Baz\n\n"
            "# Authors\n\n"
            "Jane Doe"
        )
        assert loaded.strip() == expected


class TestAddFairlearnMetricFrame:
    @pytest.fixture
    def card(self):
        model = MyRegressor()
        card = Card(model=model)
        return card

    @pytest.fixture
    def metric_frame(self):
        metrics = import_or_raise(
            "fairlearn.metrics", "model card fairlearn metricframe"
        )

        y_true = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        sex = ["Female"] * 5 + ["Male"] * 5
        metric_dict = {"selection_rate": metrics.selection_rate}
        metric_frame = metrics.MetricFrame(
            y_true=y_true, y_pred=y_pred, sensitive_features=sex, metrics=metric_dict
        )
        return metric_frame

    @pytest.mark.parametrize("transpose", [True, False])
    def test_metric_table(self, card: Card, transpose, metric_frame):
        card.add_fairlearn_metric_frame(
            metric_frame=metric_frame,
            transpose=transpose,
            table_name="Metric Frame Table",
        )

        actual_table = card.select("Metric Frame Table").format()

        if transpose is True:
            expected_table = (
                "<details>\n"
                "<summary> Click to expand </summary>\n"
                "\n"
                "| selection_rate |\n"
                "| :------------: |\n"
                "|      0.4       |\n"
                "|      0.8       |\n"
                "|      0.4       |\n"
                "|      0.5       |\n"
                "\n"
                "</details>"
            )
        else:
            expected_table = (
                "<details>\n"
                "<summary> Click to expand </summary>\n"
                "\n"
                "| difference | group_max | group_min | ratio |\n"
                "| :--------: | :-------: | :-------: | :---: |\n"
                "|    0.4     |    0.8    |    0.4    |  0.5  |\n"
                "\n"
                "</details>"
            )

        assert expected_table == actual_table

    def test_metric_table_with_description(self, card: Card, metric_frame):
        card.add_fairlearn_metric_frame(
            description="An awesome table",
            metric_frame=metric_frame,
            table_name="Metric Frame Table",
        )

        actual_table = card.select("Metric Frame Table").format()
        expected_table = (
            "An awesome table\n\n"
            "<details>\n"
            "<summary> Click to expand </summary>\n\n"
            "| selection_rate |\n"
            "| :------------: |\n"
            "|      0.4       |\n"
            "|      0.8       |\n"
            "|      0.4       |\n"
            "|      0.5       |\n"
            "\n"
            "</details>"
        )

        assert expected_table == actual_table


class TestCardTableOfContents:
    @pytest.fixture
    def card(self):
        model = MyRegressor()
        card = Card(model=model)
        card.add_model_plot()
        card.add_hyperparams()
        card.add_metrics(accuracy=0.1)
        return card

    def test_toc(self, card):
        toc = card.get_toc()
        exptected_toc = [
            "- Model description",
            "  - Intended uses & limitations",
            "  - Training Procedure",
            "    - Hyperparameters",
            "    - Model Plot",
            "  - Evaluation Results",
            "- How to Get Started with the Model",
            "- Model Card Authors",
            "- Model Card Contact",
            "- Citation",
        ]

        assert toc == "\n".join(exptected_toc)

    def test_toc_with_invisible_section(self, card):
        section = card.select("Citation")
        section.visible = False
        toc = card.get_toc()
        exptected_toc = [
            "- Model description",
            "  - Intended uses & limitations",
            "  - Training Procedure",
            "    - Hyperparameters",
            "    - Model Plot",
            "  - Evaluation Results",
            "- How to Get Started with the Model",
            "- Model Card Authors",
            "- Model Card Contact",
        ]

        assert toc == "\n".join(exptected_toc)


class TestFoldedSection:
    def test_folded_section(self, destination_path, model_card):
        model_card.add(foo="Foo")
        model_card.add(**{"foo/bar": "Foo/Bar", "foo/baz": "Foo/Baz"})
        model_card.select("foo/baz").folded = True

        foo_details = (
            "<details>\n<summary> Click to expand </summary>\n\nFoo\n\n</details>\n"
        )
        foo_bar_details = (
            "<details>\n<summary> Click to expand </summary>\n\nFoo/Bar\n\n</details>\n"
        )
        foo_baz_details = (
            "<details>\n<summary> Click to expand </summary>\n\nFoo/Baz\n\n</details>\n"
        )

        output = model_card.render()
        assert foo_details not in output
        assert foo_bar_details not in output
        assert foo_baz_details in output

        model_card.select("foo").folded = True

        output = model_card.render()
        assert foo_details in output
        assert foo_bar_details not in output
        assert foo_baz_details not in output

        model_card.select("foo").folded = False

        output = model_card.render()
        assert foo_details not in output
        assert foo_bar_details not in output
        assert foo_baz_details in output

        model_card.select("foo/bar").folded = True
        model_card.select("foo/baz").folded = False

        output = model_card.render()
        assert foo_details not in output
        assert foo_bar_details in output
        assert foo_baz_details not in output


class TestCardSaveWithPlots:
    def test_copy_plots(self, destination_path, model_card):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory(prefix="skops-test-plots") as plot_path:
            plt.plot([4, 5, 6, 7])
            fig_1_path = Path(plot_path) / "fig1.png"
            plt.savefig(fig_1_path)
            model_card = model_card.add_plot(fig1=fig_1_path)

            plt.plot([7, 6, 5, 4])
            fig_2_path = "fig2.png"
            plt.savefig(fig_2_path)
            model_card = model_card.add_plot(fig2=fig_2_path)

            model_card.save(Path(destination_path) / "README.md", copy_files=True)

        assert (Path(destination_path) / "README.md").exists()
        assert (Path(destination_path) / "fig1.png").exists()
        assert (Path(destination_path) / "fig2.png").exists()
