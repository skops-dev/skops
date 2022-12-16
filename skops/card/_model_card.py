from __future__ import annotations

import json
import re
import textwrap
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from reprlib import Repr
from typing import Any, Iterator, Literal, Protocol, Sequence, Union

import joblib
from huggingface_hub import ModelCardData
from sklearn.utils import estimator_html_repr
from tabulate import tabulate  # type: ignore

from skops.card._templates import CONTENT_PLACEHOLDER, SKOPS_TEMPLATE, Templates
from skops.io import load

# Repr attributes can be used to control the behavior of repr
aRepr = Repr()
aRepr.maxother = 79
aRepr.maxstring = 79


VALID_TEMPLATES = {item.value for item in Templates}
NEED_SECTION_ERR_MSG = (
    "You are trying to {action} but you're using a custom template, please pass the "
    "'section' argument to determine where to put the content"
)


def wrap_as_details(text: str, folded: bool) -> str:
    if not folded:
        return text
    return f"<details>\n<summary> Click to expand </summary>\n\n{text}\n\n</details>"


def _clean_table(table: str) -> str:
    # replace line breaks "\n" with html tag <br />, however, leave end-of-line
    # line breaks (eol_lb) intact
    eol_lb = "|\n"
    placeholder = "$%!?"  # arbitrary string that never appears naturally
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

    table: Mapping[str, Sequence[Any]]
    folded: bool = False

    def __post_init__(self) -> None:
        try:
            import pandas as pd

            self._is_pandas_df = isinstance(self.table, pd.DataFrame)
        except ImportError:
            self._is_pandas_df = False

        if self._is_pandas_df:
            ncols = len(self.table.columns)  # type: ignore
        else:
            ncols = len(self.table)
        if ncols == 0:
            raise ValueError("Trying to add table with no columns")

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


def split_subsection_names(key: str) -> list[str]:
    r"""Split a string containing multiple sections into a list of strings for
    each.

    The separator is ``"/"``. To avoid splitting on ``"/"``, escape it using
    ``"\\/"``.

    Examples
    --------
    >>> split_subsection_names("Section A")
    ['Section A']
    >>> split_subsection_names("Section A/Section B/Section C")
    ['Section A', 'Section B', 'Section C']
    >>> split_subsection_names("A section containg \\/ a slash")
    ['A section containg / a slash']
    >>> split_subsection_names("Spaces are / stripped")
    ['Spaces are', 'stripped']

    Parameters
    ----------
    key : str
        The section name consisting potentially of multiple subsections. It has
        to be ensured beforhand that this is not an empty string.

    Returns
    -------
    parts : list of str
        The individual (sub)sections.

    """
    placeholder = "$%!?"  # arbitrary sting that never appears naturally
    key = key.replace("\\/", placeholder)
    parts = (part.strip() for part in key.split("/"))
    return [part.replace(placeholder, "/") for part in parts]


def _getting_started_code(
    file_name: str, model_format: Literal["pickle", "skops"], indent="    "
) -> list[str]:
    # get lines of code required to load the model
    lines = [
        "import json",
        "import pandas as pd",
    ]
    if model_format == "skops":
        lines += ["import skops.io as sio"]
    else:
        lines += ["import joblib"]

    if model_format == "skops":
        lines += [f'model = sio.load("{file_name}")']
    else:  # pickle
        lines += [f'model = joblib.load("{file_name}")']

    lines += [
        'with open("config.json") as f:',
        indent + "config = json.load(f)",
        'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))',
    ]
    return lines


@dataclass
class Section:
    """Building block of the model card.

    The model card is represented internally as a dict with keys being strings
    and values being Sections. The key is identical to the section title.

    Additionally, the section may hold content in the form of strings (can be an
    empty string) or a ``Formattable``, which is simply an object with a
    ``format`` method that returns a string.

    Finally, the section can contain subsections, which again are dicts of
    string keys and section values (the dict can be empty). Therefore, the model
    card representation forms a tree structure, making use of the fact that dict
    order is preserved.

    """

    title: str
    content: Formattable | str
    subsections: dict[str, Section] = field(default_factory=dict)

    def select(self, key: str) -> Section:
        """Return a subsection or subsubsection of this section

        Parameters
        ----------
        key : str
            The name of the (sub)section to select. When selecting a subsection,
            either use a ``"/"`` in the name to separate the parent and child
            sections, chain multiple ``select`` calls.

        Returns
        -------
        section : Section
            A dataclass containing all information relevant to the selected
            section. Those are the title, the content, and subsections (in a
            dict).

        Raises
        ------
        KeyError
            If the given section name was not found, a ``KeyError`` is raised.
        """
        section_names = split_subsection_names(key)
        # check that no section name is empty
        if not all(bool(name) for name in section_names):
            msg = f"Section name cannot be empty but got '{key}'"
            raise KeyError(msg)

        section = self
        for section_name in section_names:
            section = section.subsections[section_name]
        return section


class Formattable(Protocol):
    def format(self) -> str:
        ...  # pragma: no cover


def _load_model(model: Any, trusted=False) -> Any:
    """Return a model instance.

    Loads the model if provided a file path, if already a model instance return
    it unmodified.

    Parameters
    ----------
    model : pathlib.Path, str, or sklearn estimator
        Path/str or the actual model instance. if a Path or str, loads the model.

    trusted : bool, default=False
        Passed to :func:`skops.io.load` if the model is a file path and it's
        a `skops` file.

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
            model = load(model_path, trusted=trusted)
        else:
            model = joblib.load(model_path)
    except Exception as ex:
        msg = f'An "{type(ex).__name__}" occurred during model loading.'
        raise RuntimeError(msg) from ex

    return model


class Card:
    """Model card class that will be used to generate model card.

    This class can be used to write information and plots to model card and save
    it. This class by default generates an interactive plot of the model and a
    table of hyperparameters. Some sections are added by default.

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

    template: "skops", dict, or None (default="skops")
        Whether to add default sections or not. The template can be a predefined
        template, which at the moment can only be the string ``"skops"``, which
        is a template provided by ``skops`` that is geared towards typical
        sklearn models. If you don't want any prefilled sections, just pass
        ``None``. If you want custom prefilled sections, pass a ``dict``, where
        keys are the sections and values are the contents of the sections. Note
        that when you use no template or a custom template, some methods will
        not work, e.g. :meth:`Card.add_metrics`, since it's not clear where to
        put the metrics when there is no template or a custom template.

    trusted: bool, default=False
        Passed to :func:`skops.io.load` if the model is a file path and it's
        a `skops` file.

    Attributes
    ----------
    model: estimator object
        The scikit-learn compatible model that will be documented.

    metadata: ModelCardData
        Metadata to be stored at the beginning of the saved model card, as
        metadata to be understood by the Hugging Face Hub.

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
    >>> from skops.card import Card
    >>> X, y = load_iris(return_X_y=True)
    >>> model = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> model_card = Card(model)
    >>> model_card.metadata.license = "mit"
    >>> y_pred = model.predict(X)
    >>> model_card.add_metrics(**{
    ...     "accuracy": accuracy_score(y, y_pred),
    ...     "f1 score": f1_score(y, y_pred, average="micro"),
    ... })
    Card(...)
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
    >>> model_card.add_plot(**{
    ...     "Model description/Confusion Matrix": tmp_path / "confusion_matrix.png"
    ... })
    Card(...)
    >>> # add new content to the existing section "Model description"
    >>> model_card.add(**{"Model description": "This is the best model"})
    Card(...)
    >>> # add content to a new section
    >>> model_card.add(**{"A new section": "Please rate my model"})
    Card(...)
    >>> # add new subsection to an existing section by using "/"
    >>> model_card.add(**{"Model description/Model name": "This model is called Bob"})
    Card(
      model=LogisticRegression(random_state=0, solver='liblinear'),
      metadata.license=mit,
      Model description=This is the best model,
      Model description/Training Procedure/... | | warm_start | False | </details>,
      Model description/Training Procedure/...</pre></div></div></div></div></div>,
      Model description/Evaluation Results=...ccuracy | 0.96 | | f1 score | 0.96 |,
      Model description/Confusion Matrix=...confusion_matrix.png'),
      Model description/Model name=This model is called Bob,
      A new section=Please rate my model,
    )
    >>> # save the card to a README.md file
    >>> model_card.save(tmp_path / "README.md")

    """

    def __init__(
        self,
        model,
        model_diagram: bool = True,
        metadata: ModelCardData | None = None,
        template: Literal["skops"] | dict[str, str] | None = "skops",
        trusted: bool = False,
    ) -> None:
        self.model = model
        self.model_diagram = model_diagram
        self.metadata = metadata or ModelCardData()
        self.template = template
        self.trusted = trusted

        self._data: dict[str, Section] = {}
        self._metrics: dict[str, str | float | int] = {}

        self._populate_template()

    def _populate_template(self):
        """If initialized with a template, use it to populate the card."""
        if not self.template:
            return

        if isinstance(self.template, str) and (self.template not in VALID_TEMPLATES):
            valid_templates = ", ".join(f"'{val}'" for val in sorted(VALID_TEMPLATES))
            msg = (
                f"Unknown template '{self.template}', "
                f"template must be one of the following values: {valid_templates}"
            )
            raise ValueError(msg)

        if self.template == Templates.skops.value:
            self.add(**SKOPS_TEMPLATE)
            # for the skops template, automatically add some default sections
            self.add_model_plot()
            self.add_hyperparams()
            self.add_get_started_code()
        elif isinstance(self.template, Mapping):
            self.add(**self.template)

    def get_model(self) -> Any:
        """Returns sklearn estimator object.

        If the ``model`` is already loaded, return it as is. If the ``model``
        attribute is a ``Path``/``str``, load the model and return it.

        Returns
        -------
        model : BaseEstimator
            The model instance.

        """
        model = _load_model(self.model, self.trusted)
        # Ideally, we would only call the method below if we *know* that the
        # model has changed, but at the moment we have no way of knowing that
        return model

    def add(self, **kwargs: str | Formattable) -> Card:
        """Add new section(s) to the model card.

        Add one or multiple sections to the model card. The section names are
        taken from the keys and the contents are taken from the values.

        To add to an existing section, use a ``"/"`` in the section name, e.g.:

        ``card.add(**{"Existing section/New section": "content"})``.

        If the parent section does not exist, it will be added automatically.

        To add a section with ``"/"`` in its title (i.e. not inteded as a
        subsection), escape the slash like so, ``"\\/"``, e.g.:

        ``card.add(**{"A section with\\/a slash in the title": "content"})``.

        If a section of the given name already exists, its content will be
        overwritten.

        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary serve as the section title and the values
            as the section content. It's possible to add to existing sections.

        Returns
        -------
        self : object
            Card object.

        """
        for key, val in kwargs.items():
            self._add_single(key, val)
        return self

    def _select(
        self, subsection_names: Sequence[str], create: bool = True
    ) -> dict[str, Section]:
        """Select a single section from the data.

        Parameters
        ----------
        subsection_names: list of str
            The subsection names, already split into individual subsections.

        create: bool (default=True)
            Whether to create the subsection if it does not already exist or
            not.

        Returns
        -------
        section: dict of Section
            A dict mapping the section key (identical to the title) to the
            actual ``Section``, which is a dataclass that contains the actual
            data of the section.

        Raises
        ------
        KeyError
            If the section does not exist and ``create=False``, raises a
            ``KeyError``.

        """
        section = self._data
        if not subsection_names:
            return section

        for subsection_name in subsection_names:
            section_maybe = section.get(subsection_name)

            # there are already subsections
            if section_maybe is not None:
                section = section_maybe.subsections
                continue

            if create:
                # no subsection, create
                entry = Section(title=subsection_name, content="")
                section[subsection_name] = entry
                section = entry.subsections
            else:
                raise KeyError(f"Section {subsection_name} does not exist")

        return section

    def select(self, key: str) -> Section:
        """Select a section from the model card.

        To select a subsection of an existing section, use a ``"/"`` in the
        section name, e.g.:

        ``card.select("Main section/Subsection")``.

        Alternatively, multiple ``select`` calls can be chained:

        ``card.select("Main section").select("Subsection")``.

        Parameters
        ----------
        key : str
            The name of the (sub)section to select. When selecting a subsection,
            either use a ``"/"`` in the name to separate the parent and child
            sections, chain multiple ``select`` calls.

        Returns
        -------
        self : Section
            A dataclass containing all information relevant to the selected
            section. Those are the title, the content, and subsections (in a
            dict).

        Raises
        ------
        KeyError
            If the given section name was not found, a ``KeyError`` is raised.

        """
        if not key:
            msg = f"Section name cannot be empty but got '{key}'"
            raise KeyError(msg)

        *subsection_names, leaf_node_name = split_subsection_names(key)

        if not leaf_node_name:
            msg = f"Section name cannot be empty but got '{key}'"
            raise KeyError(msg)

        parent_section = self._select(subsection_names, create=False)
        return parent_section[leaf_node_name]

    def delete(self, key: str | Sequence[str]) -> None:
        """Delete a section from the model card.

        To delete a subsection of an existing section, use a ``"/"`` in the
        section name, e.g.:

        ``card.delete("Existing section/New section")``.

        Alternatively, a list of strings can be passed:

        ``card.delete(["Existing section", "New section"])``.

        Parameters
        ----------
        key : str or list of str
            The name of the (sub)section to select. When selecting a subsection,
            either use a ``"/"`` in the name to separate the parent and child
            sections, or pass a list of strings.

        Raises
        ------
        KeyError
            If the given section name was not found, a ``KeyError`` is raised.

        """
        if not key:
            msg = f"Section name cannot be empty but got '{key}'"
            raise KeyError(msg)

        if isinstance(key, str):
            *subsection_names, leaf_node_name = split_subsection_names(key)
        else:
            *subsection_names, leaf_node_name = key

        if not leaf_node_name:
            msg = f"Section name cannot be empty but got '{key}'"
            raise KeyError(msg)

        parent_section = self._select(subsection_names, create=False)
        del parent_section[leaf_node_name]

    def _add_single(self, key: str, val: Formattable | str) -> Section:
        """Add a single section.

        If the (sub)section does not exist, it is created. Otherwise, the
        existing (sub)section is modified.

        Parameters
        ----------
        key: str
            The name of the (sub)section.

        val: str or Formattable
            The value to assign to the (sub)section.

        """
        *subsection_names, leaf_node_name = split_subsection_names(key)
        section = self._select(subsection_names)

        if leaf_node_name in section:
            # entry exists, only overwrite content
            section[leaf_node_name].content = val
        else:
            # entry does not exist, create a new one
            section[leaf_node_name] = Section(title=leaf_node_name, content=val)

        return section[leaf_node_name]

    def add_model_plot(
        self,
        section: str | None = None,
        description: str | None = None,
    ) -> Card:
        """Add a model plot

        Use sklearn model visualization to add create a diagram of the model.
        See the `sklearn model visualization docs
        <https://scikit-learn.org/stable/modules/compose.html#visualizing-composite-estimators>`_.

        The model diagram is not added if the card class was instantiated with
        ``model_diagram=False``.

        Parameters
        ----------
        section : str or None, default=None
            The section that the model plot should be added to. If you're using
            the default skops template, you can leave this parameter as
            ``None``, otherwise you have to indicate the section. If the section
            does not exist, it will be created for you.

        description : str or None, default=None
            An optional description to be added before the model plot. If you're
            using the default skops template, a standard text is used. Pass a
            string here if you want to use your own text instead. Leave this
            empty to not add any description.

        Returns
        -------
        self : object
            Card object.
        """
        if not self.model_diagram:
            return self

        if section is None:
            if self.template == Templates.skops.value:
                section = "Model description/Training Procedure/Model Plot"
            else:
                msg = NEED_SECTION_ERR_MSG.format(action="add a model plot")
                raise ValueError(msg)

        if description is None:
            if self.template == Templates.skops.value:
                description = "The model plot is below."

        self._add_model_plot(self.get_model(), section=section, description=description)

        return self

    def _add_model_plot(
        self, model: Any, section: str, description: str | None
    ) -> None:
        """Add model plot section

        The model should be a loaded sklearn model, not a path.

        """
        model_plot_div = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
        if model_plot_div.count("sk-top-container") == 1:
            model_plot_div = model_plot_div.replace(
                "sk-top-container", 'sk-top-container" style="overflow: auto;'
            )

        if description:
            content = f"{description}\n\n{model_plot_div}"
        else:
            content = model_plot_div

        self._add_single(section, content)

    def add_hyperparams(
        self, section: str | None = None, description: str | None = None
    ) -> Card:
        """Add the model's hyperparameters as a table

        Parameters
        ----------
        section : str or None, default=None
            The section that the hyperparamters should be added to. If you're
            using the default skops template, you can leave this parameter as
            ``None``, otherwise you have to indicate the section. If the section
            does not exist, it will be created for you.

        description : str or None, default=None
            An optional description to be added before the hyperparamters. If
            you're using the default skops template, a standard text is used.
            Pass a string here if you want to use your own text instead. Leave
            this empty to not add any description.

        Returns
        -------
        self : object
            Card object.

        """
        if section is None:
            if self.template == Templates.skops.value:
                section = "Model description/Training Procedure/Hyperparameters"
            else:
                msg = NEED_SECTION_ERR_MSG.format(action="add model hyperparameters")
                raise ValueError(msg)

        if description is None:
            if self.template == Templates.skops.value:
                description = "The model is trained with below hyperparameters."

        self._add_hyperparams(
            self.get_model(), section=section, description=description
        )
        return self

    def _add_hyperparams(
        self, model: Any, section: str, description: str | None
    ) -> None:
        """Add hyperparameter section.

        The model should be a loaded sklearn model, not a path.

        """
        hyperparameter_dict = model.get_params(deep=True)
        table = _clean_table(
            tabulate(
                list(hyperparameter_dict.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="github",
            )
        )
        table_folded = textwrap.dedent(
            """
            <details>
            <summary> Click to expand </summary>

            {}

            </details>"""
        ).format(table)

        if description:
            content = f"{description}\n{table_folded}"
        else:
            content = table_folded

        self._add_single(section, content)

    def add_get_started_code(
        self,
        section: str | None = None,
        description: str | None = None,
        file_name: str | None = None,
        model_format: Literal["pickle", "skops"] | None = None,
    ) -> Card:
        """Add getting started code

        This code can be copied by users to load the model and make predictions
        with it.

        Parameters
        ----------
        section : str or None, default=None
            The section that the code should be added to. If you're using the
            default skops template, you can leave this parameter as ``None``,
            otherwise you have to indicate the section. If the section does not
            exist, it will be created for you.

        description : str or None, default=None
            An optional description to be added before the code. If you're using
            the default skops template, a standard text is used. Pass a string
            here if you want to use your own text instead. Leave this empty to
            not add any description.

        file_name : str or None, default=None
            The file name of the model. If no file name is indicated, there will
            be an attempt to read the file name from the card's metadata. If
            that fails, an error is raised and you have to pass this argument
            explicitly.

        model_format : "skops", "pickle", or None, default=None
            The model format used to store the model.If format is indicated,
            there will be an attempt to read the model format from the card's
            metadata. If that fails, an error is raised and you have to pass
            this argument explicitly.

        Returns
        -------
        self : object
            Card object.

        """
        if file_name is None:
            file_name = self.metadata.to_dict().get("model_file")

        if model_format is None:
            model_format = (
                self.metadata.to_dict().get("sklearn", {}).get("model_format")
            )

        if model_format and (model_format not in ("pickle", "skops")):
            msg = (
                f"Invalid model format '{model_format}', should be one of "
                "'pickle' or 'skops'"
            )
            raise ValueError(msg)

        if (not file_name) or (not model_format):
            return self

        if section is None:
            if self.template == Templates.skops.value:
                section = "How to Get Started with the Model"
            else:
                msg = NEED_SECTION_ERR_MSG.format(action="add get started code")
                raise ValueError(msg)

        if description is None:
            if self.template == Templates.skops.value:
                description = "Use the code below to get started with the model."

        self._add_get_started_code(
            section,
            file_name=file_name,
            model_format=model_format,
            description=description,
        )

        return self

    def _add_get_started_code(
        self,
        section: str,
        file_name: str,
        model_format: Literal["pickle", "skops"],
        description: str | None,
        indent: str = "    ",
    ) -> None:
        """Add getting started code to the corresponding section"""
        lines = _getting_started_code(
            file_name, model_format=model_format, indent=indent
        )
        lines = ["```python"] + lines + ["```"]
        code = "\n".join(lines)

        if description:
            content = f"{description}\n\n{code}"
        else:
            content = code

        self._add_single(section, content)

    def add_plot(self, *, folded=False, **kwargs: str) -> Card:
        """Add plots to the model card.

        The plot should be saved on the file system and the path passed as
        value.

        Parameters
        ----------
        folded: bool (default=False)
            If set to ``True``, the plot will be enclosed in a ``details`` tag.
            That means the content is folded by default and users have to click
            to show the content. This option is useful if the added plot is
            large.

        **kwargs : dict
            The arguments should be of the form ``name=plot_path``, where
            ``name`` is the name of the plot and section, and ``plot_path`` is
            the path to the plot on the file system, relative to the root of the
            project. The plots should have already been saved under the
            project's folder.

        Returns
        -------
        self : object
            Card object.

        """
        for section_name, plot_path in kwargs.items():
            plot_name = split_subsection_names(section_name)[-1]
            section = PlotSection(alt_text=plot_name, path=plot_path, folded=folded)
            self._add_single(section_name, section)
        return self

    def add_table(
        self, *, folded: bool = False, **kwargs: dict["str", list[Any]]
    ) -> Card:
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
            self._add_single(key, section)
        return self

    def add_metrics(
        self,
        section: str | None = None,
        description: str | None = None,
        **kwargs: str | int | float,
    ) -> Card:
        """Add metric values to the model card.

        All metrics will be collected in, and then formatted to, a table.

        Parameters
        ----------
        section : str or None, default=None
            The section that the metrics should be added to. If you're using the
            default skops template, you can leave this parameter as ``None``,
            otherwise you have to indicate the section. If the section does not
            exist, it will be created for you.

        description : str or None, default=None
            An optional description to be added before the metrics. If you're
            using the default skops template, a standard text is used. Pass a
            string here if you want to use your own text instead. Leave this
            empty to not add any description.

        **kwargs : dict
            A dictionary of the form ``{metric name: metric value}``.

        Returns
        -------
        self : object
            Card object.
        """
        if section is None:
            if self.template == Templates.skops.value:
                section = "Model description/Evaluation Results"
            else:
                msg = NEED_SECTION_ERR_MSG.format(action="add metrics")
                raise ValueError(msg)

        if description is None:
            if self.template == Templates.skops.value:
                description = (
                    "You can find the details about evaluation process and "
                    "the evaluation results."
                )

        self._metrics.update(kwargs)
        self._add_metrics(section, self._metrics, description=description)
        return self

    def _add_metrics(
        self,
        section: str,
        metrics: dict[str, str | float | int],
        description: str | None,
    ) -> None:
        """Add metrics to the Evaluation Results section."""
        if self._metrics:
            data_transposed = zip(*self._metrics.items())  # make column oriented
            inp = {key: val for key, val in zip(["Metric", "Value"], data_transposed)}
            table = TableSection(inp).format()
        else:
            # create empty table
            table = TableSection({"Metric": [], "Value": []}).format()

        if description:
            content = f"{description}\n\n{table}"
        else:
            content = table

        self._add_single(section, content)

    def _generate_metadata(self, metadata: ModelCardData) -> Iterator[str]:
        """Yield metadata in yaml format"""
        for key, val in metadata.to_dict().items() if metadata else {}:
            yield aRepr.repr(f"metadata.{key}={val},").strip('"').strip("'")

    def _generate_content(
        self, data: dict[str, Section], depth: int = 1
    ) -> Iterator[str]:
        """Yield title and (formatted) contents.

        Recursively go through the data and consecutively yield the title with
        the appropriate number of "#"s (markdown format), then the associated
        content.

        """
        for val in data.values():
            title = f"{depth * '#'} {val.title}"
            yield title

            if isinstance(val.content, str):
                yield val.content
            else:  # is a Formattable
                yield val.content.format()

            if val.subsections:
                yield from self._generate_content(val.subsections, depth=depth + 1)

    def _iterate_content(
        self, data: dict[str, Section], parent_section: str = ""
    ) -> Iterator[tuple[str, Formattable | str]]:
        """Yield tuples of title and (non-formatted) content."""
        for val in data.values():
            if parent_section:
                title = "/".join((parent_section, val.title))
            else:
                title = val.title

            yield title, val.content

            if val.subsections:
                yield from self._iterate_content(val.subsections, parent_section=title)

    @staticmethod
    def _format_repr(text: str) -> str:
        # Remove new lines, multiple spaces, quotation marks, and cap line length
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", r" ", text)
        return aRepr.repr(text).strip('"').strip("'")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # repr for the model
        model = getattr(self, "model", None)
        if model:
            model_repr = self._format_repr(f"model={repr(self.get_model())},")
        else:
            model_repr = None

        # repr for metadata
        metadata_reprs = []
        for key, val in self.metadata.to_dict().items() if self.metadata else {}:
            if key == "widget":
                metadata_reprs.append("metadata.widget={...},")
                continue

            metadata_reprs.append(self._format_repr(f"metadata.{key}={val},"))
        metadata_repr = "\n".join(metadata_reprs)

        # repr for contents
        content_reprs = []
        for title, content in self._iterate_content(self._data):
            if not content:
                continue
            if isinstance(content, str) and content.rstrip("`").rstrip().endswith(
                CONTENT_PLACEHOLDER
            ):
                # if content is just some default text, no need to show it
                continue
            content_reprs.append(self._format_repr(f"{title}={content},"))
        content_repr = "\n".join(content_reprs)

        # combine all parts
        complete_repr = "Card(\n"
        if model_repr:
            complete_repr += textwrap.indent(model_repr, "  ") + "\n"
        if metadata_reprs:
            complete_repr += textwrap.indent(metadata_repr, "  ") + "\n"
        if content_reprs:
            complete_repr += textwrap.indent(content_repr, "  ") + "\n"
        complete_repr += ")"
        return complete_repr

    def _generate_card(self) -> Iterator[str]:
        """Yield sections of the model card, including the metadata."""
        if self.metadata.to_dict():
            yield f"---\n{self.metadata.to_yaml()}\n---"

        for line in self._generate_content(self._data):
            if line:
                yield "\n" + line

        # add an empty line add the end
        yield ""

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
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._generate_card()))

    def render(self) -> str:
        """Render the final model card as a string.

        Returns
        -------
        result : str
            The rendered model card with all placeholders filled and all extra
            sections inserted.
        """
        return "\n".join(self._generate_card())
