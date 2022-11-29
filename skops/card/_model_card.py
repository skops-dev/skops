from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from reprlib import Repr
from typing import Any, Optional, Union

import joblib
from huggingface_hub import ModelCard, ModelCardData
from sklearn.utils import estimator_html_repr
from tabulate import tabulate  # type: ignore

import skops
from skops.io import load
from skops.utils.importutils import import_or_raise

# Repr attributes can be used to control the behavior of repr
aRepr = Repr()
aRepr.maxother = 79
aRepr.maxstring = 79


def wrap_as_details(text: str, folded: bool) -> str:
    if not folded:
        return text
    return f"<details>\n<summary> Click to expand </summary>\n\n{text}\n\n</details>"


def _clean_table(table: str) -> str:
    # replace line breaks "\n" with html tag <br />, however, leave end-of-line
    # line breaks (eol_lb) intact
    eol_lb = "|\n"
    placeholder = "$%!?"  # arbitrary sting that never appears naturally
    table = (
        table.replace(eol_lb, placeholder)
        .replace("\n", "<br />")
        .replace(placeholder, eol_lb)
    )
    return table


@dataclass
class PlotSection:
    """Adds a link to a figure to the model card"""

    alt_text: str
    path: str | Path
    folded: bool = False

    def format(self) -> str:
        text = f"![{self.alt_text}]({self.path})"
        return wrap_as_details(text, folded=self.folded)

    def __repr__(self) -> str:
        return repr(self.path)


@dataclass
class TableSection:
    """Adds a table to the model card"""

    table: dict[str, list[Any]]
    folded: bool = False

    def __post_init__(self) -> None:
        try:
            import pandas as pd

            self._is_pandas_df = isinstance(self.table, pd.DataFrame)
        except ImportError:
            self._is_pandas_df = False

        if self._is_pandas_df:
            if self.table.empty:  # type: ignore
                raise ValueError("Empty table added")
        else:
            ncols = len(self.table)
            if ncols == 0:
                raise ValueError("Empty table added")

            key = next(iter(self.table.keys()))
            nrows = len(self.table[key])
            if nrows == 0:
                raise ValueError("Empty table added")

    def format(self) -> str:
        if self._is_pandas_df:
            headers = self.table.columns  # type: ignore
        else:
            headers = self.table.keys()

        table = _clean_table(
            tabulate(self.table, tablefmt="github", headers=headers, showindex=False)
        )
        return wrap_as_details(table, folded=self.folded)

    def __repr__(self) -> str:
        if self._is_pandas_df:
            nrows, ncols = self.table.shape  # type: ignore
        else:
            # table cannot be empty, so no checks needed here
            ncols = len(self.table)
            key = next(iter(self.table.keys()))
            nrows = len(self.table[key])
        return f"Table({nrows}x{ncols})"


def metadata_from_config(config_path: Union[str, Path]) -> ModelCardData:
    """Construct a ``ModelCardData`` object from a ``config.json`` file.

    Most information needed for the metadata section of a ``README.md`` file on
    Hugging Face Hub is included in the ``config.json`` file. This utility
    function constructs a :class:`huggingface_hub.ModelCardData` object which
    can then be passed to the :class:`~skops.card.Card` object.

    This method populates the following attributes of the instance:

    - ``library_name``: It needs to be ``"sklearn"`` for scikit-learn
        compatible models.
    - ``tags``: Set to a list, containing ``"sklearn"`` and the task of the
        model. You can then add more tags to this list.
    - ``widget``: It is populated with the example data to be used by the
        widget component of the Hugging Face Hub widget, on the model's
        repository page.

    Parameters
    ----------
    config_path: str, or Path
        Filepath to the ``config.json`` file, or the folder including that
        file.

    Returns
    -------
    card_data: huggingface_hub.ModelCardData
        :class:`huggingface_hub.ModelCardData` object.

    """
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    card_data = ModelCardData()
    card_data.library_name = "sklearn"
    card_data.tags = ["sklearn", "skops"]
    task = config.get("sklearn", {}).get("task", None)
    if task:
        card_data.tags += [task]
    card_data.model_file = config.get("sklearn", {}).get("model", {}).get("file")  # type: ignore
    example_input = config.get("sklearn", {}).get("example_input", None)
    # Documentation on what the widget expects:
    # https://huggingface.co/docs/hub/models-widgets-examples
    if example_input:
        if "tabular" in task:
            card_data.widget = {"structuredData": example_input}  # type: ignore
        # TODO: add text data example here.

    return card_data


def _load_model(model: Any) -> Any:
    """Loads the mddel if provided a file path, if already a model instance return it
    unmodified.

    Parameters
    ----------
    model : pathlib.Path, str, or sklearn estimator
        Path/str or the actual model instance. if a Path or str, loads the model.

    Returns
    -------
    model : object
        Model instance.

    """

    if not isinstance(model, (Path, str)):
        return model

    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"File is not present: {model_path}")

    try:
        if zipfile.is_zipfile(model_path):
            model = load(model_path)
        else:
            model = joblib.load(model_path)
    except Exception as ex:
        msg = f'An "{type(ex).__name__}" occured during model loading.'
        raise RuntimeError(msg) from ex

    return model


class Card:
    """Model card class that will be used to generate model card.

    This class can be used to write information and plots to model card and save
    it. This class by default generates an interactive plot of the model and a
    table of hyperparameters. The slots to be filled are defined in the markdown
    template.

    Parameters
    ----------
    model: pathlib.Path, str, or sklearn estimator object
        ``Path``/``str`` of the model or the actual model instance that will be
        documented. If a ``Path`` or ``str`` is provided, model will be loaded.

    model_diagram: bool, default=True
        Set to True if model diagram should be plotted in the card.

    metadata: ModelCardData, optional
        :class:`huggingface_hub.ModelCardData` object. The contents of this
        object are saved as metadata at the beginning of the output file, and
        used by Hugging Face Hub.

        You can use :func:`~skops.card.metadata_from_config` to create an
        instance pre-populated with necessary information based on the contents
        of the ``config.json`` file, which itself is created by
        :func:`skops.hub_utils.init`.

    Attributes
    ----------
    model: estimator object
        The scikit-learn compatible model that will be documented.

    metadata: ModelCardData
        Metadata to be stored at the beginning of the saved model card, as
        metadata to be understood by the Hugging Face Hub.

    Notes
    -----
    The contents of the sections of the template can be set using
    :meth:`Card.add` method. Plots can be added to the model card using
    :meth:`Card.add_plot`. The key you pass to :meth:`Card.add_plot` will be
    used as the header of the plot.

    Examples
    --------
    >>> from sklearn.metrics import (
    ...     ConfusionMatrixDisplay,
    ...     confusion_matrix,
    ...     accuracy_score,
    ...     f1_score
    ... )
    >>> import tempfile
    >>> from pathlib import Path
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skops import card
    >>> X, y = load_iris(return_X_y=True)
    >>> model = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> model_card = card.Card(model)
    >>> model_card.metadata.license = "mit"
    >>> y_pred = model.predict(X)
    >>> model_card.add_metrics(**{
    ...     "accuracy": accuracy_score(y, y_pred),
    ...     "f1 score": f1_score(y, y_pred, average="micro"),
    ... })
    Card(
      model=LogisticRegression(random_state=0, solver='liblinear'),
      metadata.license=mit,
    )
    >>> cm = confusion_matrix(y, y_pred,labels=model.classes_)
    >>> disp = ConfusionMatrixDisplay(
    ...     confusion_matrix=cm,
    ...     display_labels=model.classes_
    ... )
    >>> disp.plot()
    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at ...>
    >>> tmp_path = Path(tempfile.mkdtemp(prefix="skops-"))
    >>> disp.figure_.savefig(tmp_path / "confusion_matrix.png")
    ...
    >>> model_card.add_plot(confusion_matrix="confusion_matrix.png")
    Card(
      model=LogisticRegression(random_state=0, solver='liblinear'),
      metadata.license=mit,
      confusion_matrix='...confusion_matrix.png',
    )
    >>> model_card.save(tmp_path / "README.md")

    """

    def __init__(
        self,
        model: Any,
        model_diagram: bool = True,
        metadata: Optional[ModelCardData] = None,
    ) -> None:
        self.model = model
        self.model_diagram = model_diagram
        self._eval_results = {}  # type: ignore
        self._template_sections: dict[str, str] = {}
        self._extra_sections: list[tuple[str, Any]] = []
        self.metadata = metadata or ModelCardData()

    def get_model(self) -> Any:
        """Returns sklearn estimator object if ``Path``/``str``
        is provided.

        Returns
        -------
        model : Object
            Model instance.
        """
        model = _load_model(self.model)
        return model

    def add(self, **kwargs: str) -> "Card":
        """Takes values to fill model card template.

        Parameters
        ----------
        **kwargs : dict
            Parameters to be set for the model card. These parameters
            need to be sections of the underlying `jinja` template used.

        Returns
        -------
        self : object
            Card object.
        """
        for section, value in kwargs.items():
            self._template_sections[section] = value
        return self

    def add_plot(self, folded=False, **kwargs: str) -> "Card":
        """Add plots to the model card.

        Parameters
        ----------
        folded: bool (default=False)
            If set to ``True``, the plot will be enclosed in a ``details`` tag.
            That means the content is folded by default and users have to click
            to show the content. This option is useful if the added plot is
            large.

        **kwargs : dict
            The arguments should be of the form `name=plot_path`, where `name`
            is the name of the plot and `plot_path` is the path to the plot,
            relative to the root of the project. The plots should have already
            been saved under the project's folder.

        Returns
        -------
        self : object
            Card object.
        """
        for plot_name, plot_path in kwargs.items():
            section = PlotSection(alt_text=plot_name, path=plot_path, folded=folded)
            self._extra_sections.append((plot_name, section))
        return self

    def add_table(self, folded: bool = False, **kwargs: dict["str", list[Any]]) -> Card:
        """Add a table to the model card.

        Add a table to the model card. This can be especially useful when you
        using cross validation with sklearn. E.g. you can directly pass the
        result from calling :func:`sklearn.model_selection.cross_validate` or
        the ``cv_results_`` attribute from any of the hyperparameter searches,
        such as :class:`sklearn.model_selection.GridSearchCV`.

        Morevoer, you can pass any pandas :class:`pandas.DataFrame` to this
        method and it will be rendered in the model card. You may consider
        selecting only a part of the table if it's too big:

        .. code:: python

            search = GridSearchCV(...)
            search.fit(X, y)
            df = pd.DataFrame(search.cv_results_)
            # show only top 10 highest scores
            df = df.sort_values(["mean_test_score"], ascending=False).head(10)
            model_card = skops.card.Card(...)
            model_card.add_table(**{"Hyperparameter search results top 10": df})

        Parameters
        ----------
        folded: bool (default=False)
            If set to ``True``, the table will be enclosed in a ``details`` tag.
            That means the content is folded by default and users have to click
            to show the content. This option is useful if the added table is
            large.

        **kwargs : dict
            The keys should be strings, which will be used as the section
            headers, and the values should be tables. Tables can be either dicts
            with the key being strings that represent the column name, and the
            values being lists that represent the entries for each row.
            Alternatively, the table can be a :class:`pandas.DataFrame`. The
            table must not be empty.

        Returns
        -------
        self : object
            Card object.

        """
        for key, val in kwargs.items():
            section = TableSection(table=val, folded=folded)
            self._extra_sections.append((key, section))
        return self

    def add_metrics(self, **kwargs: str) -> "Card":
        """Add metric values to the model card.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of the form `{metric name: metric value}`.

        Returns
        -------
        self : object
            Card object.
        """
        for metric, value in kwargs.items():
            self._eval_results[metric] = value
        return self

    def add_permutation_importances(
        self,
        permutation_importances,
        columns,
        plot_file="permutation_importances.png",
        plot_name="Permutation Importances",
        overwrite=False,
    ) -> "Card":
        """Plots permutation importance and saves it to model card.

        Parameters
        ----------
        permutation_importances : sklearn.utils.Bunch
            Output of :func:`sklearn.inspection.permutation_importance`.

        columns : str, list or pandas.Index
            Column names of the data used to generate importances.

        plot_file : str
            Filename for the plot.

        plot_name : str, or Path
            Name of the plot.

        overwrite : bool
            Whether to overwrite the permutation importance plot, if exists.

        Returns
        -------
        self : object
            Card object.
        """
        plt = import_or_raise("matplotlib.pyplot", "permutation importance")

        if Path(plot_file).exists() and overwrite is False:
            raise ValueError(
                f"{str(plot_file)} already exists. Set `overwrite` to `True` or pass a"
                " different filename for the plot."
            )
        sorted_importances_idx = permutation_importances.importances_mean.argsort()
        _, ax = plt.subplots()
        ax.boxplot(
            x=permutation_importances.importances[sorted_importances_idx].T,
            labels=columns[sorted_importances_idx],
            vert=False,
        )
        ax.set_title(plot_name)
        ax.set_xlabel("Decrease in Score")
        plt.savefig(plot_file)
        self.add_plot(**{plot_name: plot_file})

        return self

    def _generate_card(self) -> ModelCard:
        """Generate the ModelCard object

        Returns
        -------
        card : huggingface_hub.ModelCard
            The final :class:`huggingface_hub.ModelCard` object with all
            placeholders filled and all extra sections inserted.
        """
        root = skops.__path__

        # add evaluation results

        template_sections = copy.deepcopy(self._template_sections)

        if self.metadata:
            model_file = self.metadata.to_dict().get("model_file")
            if model_file and model_file.endswith(".skops"):
                template_sections["get_started_code"] = (
                    "from skops.io import load\nimport json\n"
                    "import pandas as pd\n"
                    f'clf = load("{model_file}")\n'
                    'with open("config.json") as f:\n   '
                    " config ="
                    " json.load(f)\n"
                    'clf.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))'
                )
            elif model_file is not None:
                template_sections["get_started_code"] = (
                    "import joblib\nimport json\nimport pandas as pd\nclf ="
                    f' joblib.load({model_file})\nwith open("config.json") as'
                    " f:\n   "
                    " config ="
                    " json.load(f)\n"
                    'clf.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))'
                )
        if self.model_diagram is True:
            model_plot_div = re.sub(
                r"\n\s+", "", str(estimator_html_repr(self.get_model()))
            )
            if model_plot_div.count("sk-top-container") == 1:
                model_plot_div = model_plot_div.replace(
                    "sk-top-container", 'sk-top-container" style="overflow: auto;'
                )
            model_plot: str | None = model_plot_div
        else:
            model_plot = None
        template_sections["eval_results"] = tabulate(
            list(self._eval_results.items()),
            headers=["Metric", "Value"],
            tablefmt="github",
        )

        # if template path is not given, use default
        if template_sections.get("template_path") is None:
            template_sections["template_path"] = str(
                Path(root[0]) / "card" / "default_template.md"
            )

        # copying the template so that the original template is not touched/changed
        # append plot_name if any plots are provided, at the end of the template
        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copyfile(
                template_sections["template_path"],
                f"{tmpdirname}/temporary_template.md",
            )
            #  create a temporary template with the additional plots
            template_sections["template_path"] = f"{tmpdirname}/temporary_template.md"
            # add extra sections at the end of the template
            with open(template_sections["template_path"], "a") as template:
                if self._extra_sections:
                    template.write("\n\n# Additional Content\n")

                for key, val in self._extra_sections:
                    formatted = val.format()
                    template.write(f"\n## {key}\n\n{formatted}\n")

            card = ModelCard.from_template(
                card_data=self.metadata,
                hyperparameter_table=self._extract_estimator_config(),
                model_plot=model_plot,
                **template_sections,
            )
        return card

    def save(self, path: str | Path) -> None:
        """Save the model card.

        This method renders the model card in markdown format and then saves it
        as the specified file.

        Parameters
        ----------
        path: str, or Path
            Filepath to save your card.

        Notes
        -----
        The keys in model card metadata can be seen `here
        <https://huggingface.co/docs/hub/models-cards#model-card-metadata>`__.
        """
        card = self._generate_card()
        card.save(path)

    def render(self) -> str:
        """Render the final model card as a string.

        Returns
        -------
        card : str
            The rendered model card with all placeholders filled and all extra
            sections inserted.
        """
        card = self._generate_card()
        return str(card)

    def _extract_estimator_config(self) -> str:
        """Extracts estimator hyperparameters and renders them into a vertical table.

        Returns
        -------
        str:
            Markdown table of hyperparameters.
        """
        hyperparameter_dict = self.get_model().get_params(deep=True)
        return _clean_table(
            tabulate(
                list(hyperparameter_dict.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="github",
            )
        )

    @staticmethod
    def _strip_blank(text) -> str:
        # remove new lines and multiple spaces
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", r" ", text)
        return text

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # create repr for model
        model = getattr(self, "model", None)
        if model:
            model_str = self._strip_blank(repr(self.get_model()))
            model_repr = aRepr.repr(f"  model={model_str},").strip('"').strip("'")
        else:
            model_repr = None

        # metadata
        metadata_reprs = []
        for key, val in self.metadata.to_dict().items() if self.metadata else {}:
            if key == "widget":
                metadata_reprs.append("  metadata.widget={...},")
                continue

            metadata_reprs.append(
                aRepr.repr(f"  metadata.{key}={val},").strip('"').strip("'")
            )
        metadata_repr = "\n".join(metadata_reprs)

        # normal sections
        template_reprs = []
        for key, val in self._template_sections.items():
            val = self._strip_blank(repr(val))
            template_reprs.append(aRepr.repr(f"  {key}={val},").strip('"').strip("'"))
        template_repr = "\n".join(template_reprs)

        # figures
        figure_reprs = []
        for key, val in self._extra_sections:
            val = self._strip_blank(repr(val))
            figure_reprs.append(aRepr.repr(f"  {key}={val},").strip('"').strip("'"))
        figure_repr = "\n".join(figure_reprs)

        complete_repr = "Card(\n"
        if model_repr:
            complete_repr += model_repr + "\n"
        if metadata_reprs:
            complete_repr += metadata_repr + "\n"
        if template_repr:
            complete_repr += template_repr + "\n"
        if figure_repr:
            complete_repr += figure_repr + "\n"
        complete_repr += ")"
        return complete_repr
