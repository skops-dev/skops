from __future__ import annotations

import re
import shutil
import sys
import textwrap
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from reprlib import Repr
from typing import Any, Iterator, List, Literal, Optional, Sequence

import joblib
from prettytable import PrettyTable, TableStyle
from sklearn.utils import estimator_html_repr

from skops.card._templates import CONTENT_PLACEHOLDER, SKOPS_TEMPLATE, Templates
from skops.io import load
from skops.utils._fixes import boxplot
from skops.utils.importutils import import_or_raise

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

VALID_TEMPLATES = {item.value for item in Templates}
NEED_SECTION_ERR_MSG = (
    "You are trying to {action} but you're using a custom template, please pass the "
    "'section' argument to determine where to put the content"
)


def wrap_as_details(text: str, folded: bool) -> str:
    if not folded:
        return text
    return f"<details>\n<summary> Click to expand </summary>\n\n{text}\n\n</details>"


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
    >>> split_subsection_names("A section containing \\/ a slash")
    ['A section containing / a slash']
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
    placeholder = "\x1f"  # unit separator control character (ASCII control char 31)
    key = key.replace("\\/", placeholder)
    parts = (part.strip() for part in key.split("/"))
    return [part.replace(placeholder, "/") for part in parts]


@dataclass
class Section:
    """Building block of the model card.

    The model card is represented internally as a dict with keys being strings
    and values being ``Section``s. The key is identical to the section title.

    Additionally, the section may hold content in the form of strings (can be an
    empty string) or a ``Formattable``, which is simply an object with a
    ``format`` method that returns a string.

    The section can contain subsections, which again are dicts of
    string keys and section values (the dict can be empty). Therefore, the model
    card representation forms a tree structure, making use of the fact that dict
    order is preserved.

    The section may also contain a ``visible`` flag, which determines if the
    section will be shown when the card is rendered.

    """

    title: str
    content: str
    subsections: dict[str, Section] = field(default_factory=dict)
    visible: bool = True
    folded: bool = False

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
            section. Those are the title, the content, subsections (in a dict),
            and additional fields that depend on the type of section.

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

        section: Section = self
        for section_name in section_names:
            section = section.subsections[section_name]
        return section

    def format(self) -> str:
        return wrap_as_details(self.content, folded=self.folded)

    def __repr__(self) -> str:
        """Generates the ``repr`` of this section.

        ``repr`` determines how the content of this section is shown in the
        Card's repr.
        """
        return self.content


@dataclass
class PlotSection(Section):
    """Adds a link to a figure to the model card"""

    path: str | Path = ""
    alt_text: str = ""
    folded: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise TypeError(f"{self.__class__.__name__} requires a path")

    def format(self) -> str:
        # if no alt text provided, fall back to figure path
        alt_text = self.alt_text or self.path
        text = f"![{alt_text}]({self.path})"
        val = wrap_as_details(text, folded=self.folded)
        if self.content:
            val = f"{self.content}\n\n{val}"
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"


@dataclass
class TableSection(Section):
    """Adds a table to the model card"""

    table: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    folded: bool = False

    def __post_init__(self) -> None:
        self._check_table()

    def _check_table(self) -> None:
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
        table = PrettyTable()
        table.set_style(TableStyle.MARKDOWN)
        for key, values in self.table.items():
            # replace \n with <br /> (html new line tag) so that line breaks are
            # not converted into new rows with PrettyTable.
            values = [str(value).replace("\n", "<br />") for value in values]
            table.add_column(key, values)

        table = table.get_string()

        val = wrap_as_details(table, folded=self.folded)

        if self.content:
            val = f"{self.content}\n\n{val}"
        return val

    def __repr__(self) -> str:
        if self._is_pandas_df:
            nrows, ncols = self.table.shape  # type: ignore
        else:
            # table cannot be empty, so no checks needed here
            ncols = len(self.table)
            key = next(iter(self.table.keys()))
            nrows = len(self.table[key])
        return f"{self.__class__.__name__}({nrows}x{ncols})"


def _load_model(
    model: Any, trusted: Optional[Sequence[str]] = None, allow_pickle: bool = False
) -> Any:
    """Return a model instance.

    Loads the model if provided a file path, if already a model instance return
    it unmodified.

    Parameters
    ----------
    model : pathlib.Path, str, or sklearn estimator
        Path/str or the actual model instance. if a Path or str, loads the model.

    trusted: list of str, default=None
        Passed to :func:`skops.io.load` if the model is a file path and it's
        a `skops` file.

    allow_pickle : bool, default=False
        If `True`, allows loading models using `joblib.load`. This may lead to
        security issues if the model file is not trustworthy.

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

    if trusted and allow_pickle:
        raise ValueError(
            "`allow_pickle` cannot be `True` if `trusted` is not empty. "
            "Pickles cannot be trusted or checked for security issues."
        )

    msg = ""
    try:
        if zipfile.is_zipfile(model_path):
            model = load(model_path, trusted=trusted)
        elif allow_pickle:
            model = joblib.load(model_path)
        else:
            msg = (
                "Model file is not a skops file, and allow_pickle is set to False. "
                "Please set allow_pickle=True to load the model."
                "This may lead to security issues if the model file is not trustworthy."
            )
            raise RuntimeError(msg)
    except Exception as ex:
        if not msg:
            msg = f'"{type(ex).__name__}" occurred during model loading.'
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
        Note that a a "get started" code block will be added to the card only if
        the model is a ``Path`` or ``str``.

    model_format: Literal["pickle", "skops"] or None (default=None)
        The format of the model file. If ``None``, the format will be inferred
        from the file extension of the model file if possible.

    model_diagram: bool or "auto" or str, default="auto"
        If using the skops template, setting this to ``True`` or ``"auto"`` will
        add the model diagram, as generated by sckit-learn, to the default
        section, i.e "Model description/Training Procedure/Model Plot". Passing
        a string to ``model_diagram`` will instead use that string as the
        section name for the diagram. Set to ``False`` to not include the model
        diagram.

        If using a non-skops template, passing ``"auto"`` won't add the model
        diagram because there is no pre-defined section to put it. The model
        diagram can, however, always be added later using
        :meth:`Card.add_model_plot`.

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

    trusted: list of str, default=None
        Passed to :func:`skops.io.load` if the model is a file path and it's
        a `skops` file.

    allow_pickle: bool, default=False
        If `True`, allows loading models using `joblib.load`. This may lead to
        security issues if the model file is not trustworthy.

    Attributes
    ----------
    model: estimator object
        The scikit-learn compatible model that will be documented.

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
    >>> model = LogisticRegression(solver="saga", random_state=0).fit(X, y)
    >>> model_card = Card(model)
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
      model=LogisticRegression(random_state=0, solver='saga'),
      ...
    )
    >>> # save the card to a README.md file
    >>> model_card.save(tmp_path / "README.md")

    """

    def __init__(
        self,
        model,
        model_format: Literal["pickle", "skops"] | None = None,
        model_diagram: bool | Literal["auto"] | str = "auto",
        template: Literal["skops"] | dict[str, str] | None = "skops",
        trusted: Optional[List[str]] = None,
        allow_pickle: bool = False,
    ) -> None:
        self.model = model
        self.model_format = model_format
        self.template = template
        self.trusted = trusted
        self.allow_pickle = allow_pickle

        self._data: dict[str, Section] = {}
        self._metrics: dict[str, str | float | int] = {}
        self._model_hash = ""

        self._populate_template(model_diagram=model_diagram)

    def _populate_template(self, model_diagram: bool | Literal["auto"] | str):
        """If initialized with a template, use it to populate the card.

        Parameters
        ----------
        model_diagram: bool or "auto" or str
            If using the default template, ``"auto"`` and ``True`` will add the
            diagram in its default section. If using a custom template,
            ``"auto"`` will not add the diagram, and passing ``True`` will
            result in an error. For either, passing ``False`` will result in the
            model diagram being omitted, and passing a string (other than
            ``"auto"``) will put the model diagram into a section corresponding
            to that string.

        """
        if isinstance(self.template, str) and (self.template not in VALID_TEMPLATES):
            valid_templates = ", ".join(f"'{val}'" for val in sorted(VALID_TEMPLATES))
            msg = (
                f"Unknown template '{self.template}', "
                f"template must be one of the following values: {valid_templates}"
            )
            raise ValueError(msg)

        # default template
        if self.template == Templates.skops.value:
            self.add(folded=False, **SKOPS_TEMPLATE)
            # for the skops template, automatically add some default sections
            self.add_hyperparams()

            if (model_diagram is True) or (model_diagram == "auto"):
                self.add_model_plot()
            elif isinstance(model_diagram, str):
                self.add_model_plot(section=model_diagram)
            return

        # non-default template
        if isinstance(self.template, Mapping):
            self.add(folded=False, **self.template)

        if isinstance(model_diagram, str) and (model_diagram != "auto"):
            self.add_model_plot(section=model_diagram)
        elif model_diagram is True:
            # will trigger an error
            self.add_model_plot()

    def get_model(self) -> Any:
        """Returns sklearn estimator object.

        If the ``model`` is already loaded, return it as is. If the ``model``
        attribute is a ``Path``/``str``, load the model and return it.

        Returns
        -------
        model : BaseEstimator
            The model instance.

        """
        if isinstance(self.model, (str, Path)) and hasattr(self, "_model"):
            hash_obj = sha256()
            buf_size = 2**20  # load in chunks to save memory
            with open(self.model, "rb") as f:
                for chunk in iter(lambda: f.read(buf_size), b""):
                    hash_obj.update(chunk)
            model_hash = hash_obj.hexdigest()

            # if hash changed, invalidate cache by deleting attribute
            if model_hash != self._model_hash:
                del self._model
                self._model_hash = model_hash

        return self._model

    @cached_property
    def _model(self):
        model = _load_model(self.model, self.trusted, self.allow_pickle)
        return model

    def add(self, folded: bool = False, **kwargs: str) -> Self:
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
        folded : bool
            Whether to fold the sections by default or not.

        **kwargs : dict
            The keys of the dictionary serve as the section title and the values
            as the section content. It's possible to add to existing sections.

        Returns
        -------
        self : object
            Card object.

        """
        for key, val in kwargs.items():
            self._add_single(key, val, folded=folded)
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

    def _add_single(
        self, key: str, val: str | Section, folded: bool = False
    ) -> Section:
        """Add a single section.

        If the (sub)section does not exist, it is created. Otherwise, the
        existing (sub)section is modified.

        Parameters
        ----------
        key: str
            The name of the (sub)section.

        val: str or Section
            The value to assign to the (sub)section. If this is already a
            section, leave it as it is. If it's a string, create a
            :class:`skops.card._model_card.Section`.

        folded: bool
            Whether the (sub)section should be folded or not.

        Returns
        -------
        Section instance
            The section that has been added or modified.

        """
        *subsection_names, leaf_node_name = split_subsection_names(key)
        section = self._select(subsection_names)

        if isinstance(val, str):
            # val is a str, create a Section
            new_section = Section(title=leaf_node_name, content=val, folded=folded)
        else:
            # val is already a section and can be used as is
            new_section = val

        if leaf_node_name in section:
            # entry exists, preserve its subsections
            old_section = section[leaf_node_name]
            if new_section.subsections and (
                new_section.subsections != old_section.subsections
            ):
                msg = (
                    f"Trying to override section '{leaf_node_name}' but found "
                    "conflicting subsections."
                )
                raise ValueError(msg)
            new_section.subsections = old_section.subsections

        section[leaf_node_name] = new_section
        return section[leaf_node_name]

    def add_model_plot(
        self,
        section: str = "Model description/Training Procedure/Model Plot",
        description: str | None = None,
    ) -> Self:
        """Add a model plot

        Use sklearn model visualization to add create a diagram of the model.
        See the `sklearn model visualization docs
        <https://scikit-learn.org/stable/modules/compose.html#visualizing-composite-estimators>`_.

        The model diagram is not added if the card class was instantiated with
        ``model_diagram=False``.

        Parameters
        ----------
        section : str (default="Model description/Training Procedure/Model Plot")
            The section that the model plot should be added to. By default, the
            section is set to fit the skops model card template. If you're using
            a different template, you may have to choose a different section name.

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
        self._add_model_plot(
            self.get_model(), section_name=section, description=description
        )

        return self

    def _add_model_plot(
        self, model: Any, section_name: str, description: str | None
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

        description = description or ""
        title = split_subsection_names(section_name)[-1]
        section = Section(title=title, content=content)
        self._add_single(section_name, section)

    def add_hyperparams(
        self,
        section: str = "Model description/Training Procedure/Hyperparameters",
        description: str | None = None,
    ) -> Self:
        """Add the model's hyperparameters as a table

        Parameters
        ----------
        section : str (default="Model description/Training Procedure/Hyperparameters")
            The section that the hyperparameters should be added to. By default,
            the section is set to fit the skops model card template. If you're
            using a different template, you may have to choose a different section
            name.

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
        self._add_hyperparams(
            self.get_model(), section_name=section, description=description
        )
        return self

    def _add_hyperparams(
        self, model: Any, section_name: str, description: str | None
    ) -> None:
        """Add hyperparameter section.

        The model should be a loaded sklearn model, not a path.

        """
        params = model.get_params(deep=True)
        table = {"Hyperparameter": list(params.keys()), "Value": list(params.values())}

        description = description or ""
        title = split_subsection_names(section_name)[-1]
        section = TableSection(
            title=title, content=description, table=table, folded=True
        )
        self._add_single(section_name, section)

    def add_plot(
        self,
        *,
        description: str | None = None,
        alt_text: str | None = None,
        folded=False,
        **kwargs: str | Path,
    ) -> Self:
        """Add plots to the model card.

        The plot should be saved on the file system and the path passed as
        value.

        Parameters
        ----------
        description: str or None (default=None)
            If a string is passed as description, it is shown before the figure.
            If multiple figures are added with one call, they all get the same
            description. To add multiple figures with different descriptions,
            call this method multiple times.

        alt_text: : str or None (default=None)
            If a string is passed as ``alt_text``, it is used as the alternative
            text for the figure (i.e. what is shown if the figure cannot be
            rendered). If this argument is ``None``, the alt_text will just be
            the same as the section title. If multiple figures are added with
            one call, they all get the same alt text. To add multiple figures
            with different alt texts, call this method multiple times.

        folded: bool (default=False)
            If set to ``True``, the plot will be enclosed in a ``details`` tag.
            That means the content is folded by default and users have to click
            to show the content. This option is useful if the added plot is
            large.

        **kwargs : dict
            The arguments should be of the form ``name=plot_path``, where
            ``name`` is the name of the plot and section, and ``plot_path`` is
            the path to the plot on the file system (either a str or
            ``pathlib.Path``), relative to the root of the project. The plots
            should have already been saved under the project's folder.

        Returns
        -------
        self : object
            Card object.

        """
        description = description or ""
        for section_name, plot_path in kwargs.items():
            title = split_subsection_names(section_name)[-1]
            alt_text = alt_text or title
            section = PlotSection(
                title=title,
                content=description,
                alt_text=alt_text,
                path=plot_path,
                folded=folded,
            )
            self._add_single(section_name, section)
        return self

    def add_table(
        self,
        *,
        description: str | None = None,
        folded: bool = False,
        **kwargs: dict["str", list[Any]],
    ) -> Self:
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
        description: str or None (default=None)
            If a string is passed as description, it is shown before the table.
            If multiple tables are added with one call, they all get the same
            description. To add multiple tables with different descriptions,
            call this method multiple times.

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
        description = description or ""
        for key, val in kwargs.items():
            section = TableSection(
                title=key, content=description, table=val, folded=folded
            )
            self._add_single(key, section)
        return self

    def add_metrics(
        self,
        section: str = "Model description/Evaluation Results",
        description: str | None = None,
        **kwargs: str | int | float,
    ) -> Self:
        """Add metric values to the model card.

        All metrics will be collected in, and then formatted to, a table.

        Parameters
        ----------
        section : str (default="Model description/Evaluation Results")
            The section that metrics should be added to. By default, the section
            is set to fit the skops model card template. If you're using a
            different template, you may have to choose a different section name.

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
        self._metrics.update(kwargs)
        self._add_metrics(section, description=description, metrics=self._metrics)
        return self

    def add_permutation_importances(
        self,
        permutation_importances,
        columns: Sequence[str],
        plot_file: str | Path = "permutation_importances.png",
        plot_name: str = "Permutation Importances",
        overwrite: bool = False,
        description: str | None = None,
    ) -> Self:
        """Plots permutation importance and saves it to model card.

        Parameters
        ----------
        permutation_importances : sklearn.utils.Bunch
            Output of :func:`sklearn.inspection.permutation_importance`.

        columns : str, list or pandas.Index
            Column names of the data used to generate importances.

        plot_file : str or pathlib.Path
            Filename for the plot.

        plot_name : str
            Name of the plot.

        overwrite : bool (default=False)
            Whether to overwrite the permutation importance plot file, if a plot by that
            name already exists.

        description : str | None (default=None)
            An optional description to be added before the plot.

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
        boxplot(
            ax,
            x=permutation_importances.importances[sorted_importances_idx].T,
            tick_labels=columns[sorted_importances_idx],
            vert="horizontal",
        )
        ax.set_title(plot_name)
        ax.set_xlabel("Decrease in Score")
        plt.savefig(plot_file)
        self.add_plot(description=description, alt_text=None, **{plot_name: plot_file})

        return self

    def add_fairlearn_metric_frame(
        self,
        metric_frame,
        table_name: str = "Fairlearn MetricFrame Table",
        transpose: bool = True,
        description: str | None = None,
    ) -> Self:
        """
        Add a :class:`fairlearn.metrics.MetricFrame` table to the model card.
        The table contains the difference, group_ma, group_min, and ratio for
        each metric.

        Parameters
        ----------
        metric_frame: MetricFrame
            The Fairlearn MetricFrame to add to the model card.

        table_name: str
            The desired name of the table section in the model card.

        transpose: bool, default=True
            Whether to transpose the table or not.


        description : str | None (default=None)
            An optional description to be added before the table.

        Returns
        -------
        self: Card
            The model card with the metric frame added.

        Notes
        --------
        You can check `fairlearn's documentation
        <https://fairlearn.org/v0.8/user_guide/assessment/index.html>`__ on how to
        work with `MetricFrame`s.

        """
        frame_dict = {
            "difference": metric_frame.difference(),
            "group_max": metric_frame.group_max(),
            "group_min": metric_frame.group_min(),
            "ratio": metric_frame.ratio(),
        }

        if transpose is True:
            pd = import_or_raise("pandas", "Pandas is used to pivot the table.")

            frame_dict = pd.DataFrame(frame_dict).T

        return self.add_table(
            folded=True, description=description, **{table_name: frame_dict}
        )

    def _add_metrics(
        self,
        section_name: str,
        description: str | None,
        metrics: dict[str, str | float | int],
    ) -> None:
        """Add metrics to the Evaluation Results section."""
        if self._metrics:
            # transpose from row oriented to column oriented
            data_transposed = zip(*self._metrics.items())
            table = {
                key: list(val) for key, val in zip(["Metric", "Value"], data_transposed)
            }
        else:
            # create empty table
            table = {"Metric": [], "Value": []}

        description = description or ""
        title = split_subsection_names(section_name)[-1]
        section = TableSection(title=title, content=description, table=table)
        self._add_single(section_name, section)

    def _generate_content(
        self,
        data: dict[str, Section],
        depth: int = 1,
        destination_path: Path | None = None,
    ) -> Iterator[str]:
        """Yield title and (formatted) contents.

        Recursively go through the data and consecutively yield the title with
        the appropriate number of "#"s (markdown format), then the associated
        content.

        """
        for section in data.values():
            if not section.visible:
                continue

            title = f"{depth * '#'} {section.title}"
            yield title

            yield section.format()

            if destination_path is not None and isinstance(section, PlotSection):
                shutil.copy(section.path, destination_path)

            if section.subsections and not section.folded:
                yield from self._generate_content(
                    section.subsections,
                    depth=depth + 1,
                    destination_path=destination_path,
                )

    def _iterate_content(
        self, data: dict[str, Section], parent_section: str = ""
    ) -> Iterator[tuple[str, Section]]:
        """Yield tuples of title and (non-formatted) content."""
        for val in data.values():
            if parent_section:
                title = "/".join((parent_section, val.title))
            else:
                title = val.title

            yield title, val

            if val.subsections:
                yield from self._iterate_content(val.subsections, parent_section=title)

    @staticmethod
    def _format_repr(title: str, content: str) -> str:
        # Remove new lines, multiple spaces, quotation marks, and cap line length
        content = content.replace("\n", " ")
        content = re.sub(r"\s+", r" ", content)

        # Repr attributes can be used to control the behavior of repr
        aRepr = Repr()
        aRepr.maxother = max(3, 79 - len(title))
        aRepr.maxstring = max(3, 79 - len(title))

        content = aRepr.repr(content).strip('"').strip("'")
        return f"{title}={content},"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # repr for the model
        model = getattr(self, "model", None)
        if model:
            model_repr = self._format_repr("model", repr(self.get_model()))
        else:
            model_repr = None

        # repr for contents
        content_reprs = []
        for title, section in self._iterate_content(self._data):
            content = section.format()
            if not content:
                continue
            if content.rstrip("`").rstrip().endswith(CONTENT_PLACEHOLDER):
                # if content is just some default text, no need to show it
                continue
            content_reprs.append(self._format_repr(title, repr(section)))
        content_repr = "\n".join(content_reprs)

        # combine all parts
        complete_repr = "Card(\n"
        if model_repr:
            complete_repr += textwrap.indent(model_repr, "  ") + "\n"
        if content_reprs:
            complete_repr += textwrap.indent(content_repr, "  ") + "\n"
        complete_repr += ")"
        return complete_repr

    def _generate_card(self, destination_path: Path | None = None) -> Iterator[str]:
        """Yield sections of the model card."""
        for line in self._generate_content(
            self._data, destination_path=destination_path
        ):
            if line:
                yield "\n" + line

        # add an empty line add the end
        yield ""

    def save(self, path: str | Path, copy_files: bool = False) -> None:
        """Save the model card.

        This method renders the model card in markdown format and then saves it
        as the specified file.

        Parameters
        ----------
        path: Path
            Filepath to save your card.

        plot_path: str
            Filepath to save the plots. Use this when saving the model card
            before creating the repository. Without this path the README will
            have an absolute path to the plot that won't exist in the
            repository.
        """
        with open(path, "w", encoding="utf-8") as f:
            if not isinstance(path, Path):
                path = Path(path)
            destination_path = path.parent if copy_files else None
            f.write("\n".join(self._generate_card(destination_path=destination_path)))

    def render(self) -> str:
        """Render the final model card as a string.

        Returns
        -------
        result : str
            The rendered model card with all placeholders filled and all extra
            sections inserted.
        """
        return "\n".join(self._generate_card())

    def _iterate_key_section_content(
        self,
        data: dict[str, Section],
        level: int = 0,
    ):
        """Iterate through the key sections and yield the title and level.

        Parameters
        ----------
        data : dict[str, Section]
            The card data to iterate through. This is usually the sections and
            subsections.

        level : int, optional
            The level of the section, by default 0. This keeps track of subsections.

        Returns
        -------
        table_of_contents : str
        """
        for key, val in data.items():
            if not getattr(val, "visible", True):
                continue

            title = val.title
            yield title, level

            if val.subsections:
                yield from self._iterate_key_section_content(
                    val.subsections,
                    level=level + 1,
                )

    def get_toc(self) -> str:
        """Get the table of contents for the model card.

        Returns
        -------
        toc : str
            The table of contents for the model card formatted as a markdown string.
            Example:
                - Model description
                    - Intended uses & limitations
                    - Training Procedure
                        - Hyperparameters
                        - Model Plot
                    - Evaluation Results
                - How to Get Started with the Model
                - Model Card Authors
                - Model Card Contact
        """
        sections = []
        for title, level in self._iterate_key_section_content(self._data):
            sections.append(f"{'  ' * level}- {title}")

        return "\n".join(sections)
