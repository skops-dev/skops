"""Contains the PandocParser

This class needs to know about the pandoc parse tree but should not have
knowledge of any particular markup syntex; everything related to markup should
be known by the mapping attribute.

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from packaging.version import Version

from skops.card import Card
from skops.card._model_card import Section

from ._markup import Markdown, PandocItem

PANDOC_MIN_VERSION = "2.0"


class PandocParser:
    """Create model cards from files parsed through pandoc.

    This class knows about the implementation details of the
    :class:`~skops.card.Card` and generates it by initializing an empty class
    and then calling its methods with the input provided by pandoc.

    ``PandocParser`` does not know about any specific markup type, such as
    markdown. Instead, it is initialized with a ``Mapping``, which is
    responsible to convert pandoc input into the desired markup language.

    After initializing this class, call
    :meth:`~skops.card._parser.PandocParser.generate` to generate the resulting
    :class:`~skops.card.Card` instance.

    Parameters
    ----------
    source : str
        The model card parsed using the ``pandoc -t json`` option.

    markup_type : "markdown"
        The type of markup that was used for the model card. Right now, only
        ``"markdown"`` is supported.

    """

    def __init__(
        self, source: str, markup_type: Literal["markdown"] = "markdown"
    ) -> None:
        self.source = source
        if markup_type.lower() == "markdown":
            self.mapping = Markdown()
        else:
            raise ValueError(f"Markup of type {markup_type} is not supported (yet)")

    def _add_section(
        self, section_name: str, card: Card, section_trace: list[str]
    ) -> Section:
        # Add a new section to the card, which can be a subsection, and return
        # it.
        section_name = "/".join(section_trace)
        cur_section = card._add_single(section_name, "")
        return cur_section

    def _add_content(self, content: str, section: Section | None) -> None:
        # Add content to the current section
        if section is None:
            # This may occur if the model card starts without a section. This is
            # not illegal in markdown, but we don't handle it yet.
            raise ValueError(
                "Trying to add content but there is no current section, "
                "this is probably a bug, please open an issue on GitHub."
            )

        if not section.content:
            section.content = content
        elif isinstance(section.content, str):
            section.content = section.content + "\n\n" + content
        else:  # pragma: no cover
            # TODO: Content is a Formattable, no generic way to modify it --
            # should we require each Formattable to have an update method?
            raise ValueError(f"Could not modify content of {section.content}")

    def _parse_header(
        self, item: PandocItem, section_trace: list[str]
    ) -> tuple[str, int]:
        # Headers are the only type of item that needs to be handled
        # differently. This is because we structure the underlying model card
        # data as a tree with nodes corresponding to headers. To assign the
        # right parent or child node, we need to keep track of the level of the
        # headers. This cannot be done on the level of the markdown mapping,
        # since it is not aware of the tree structure.
        level, _, _ = item["c"]
        content = self.mapping(item)
        return content, level

    def _post_process(self, res: str) -> str:
        # replace Latin1 space
        res = res.replace("\xa0", " ")

        # pandoc creates ☒ and ☐ for to do items but GitHub requires [x] and [ ]
        # for an item to be considered a to do item
        res = res.replace("- ☒", "- [x]").replace("- ☐", "- [ ]")
        return res

    def generate(self) -> Card:
        """Generate the model card instance from the parsed card.

        Returns
        -------
        card : :class:`~skops.card.Card`
            The parsed model card instance. If not further modified, the output
            of saving that card should be (almost) identical to the initial
            model card.
        """
        section: Section | None = None
        section_trace: list[str] = []
        card = Card(None, template=None)

        # Parsing the flat structure, not recursively as in pandocfilters. After
        # visiting the parent node, it's not necessary to visit its child nodes,
        # because the mapping class already takes care of visiting the child
        # nodes.
        for item in json.loads(self.source)["blocks"]:
            if item["t"] == "Header":
                content, level = self._parse_header(item, section_trace=section_trace)
                res = self._post_process(content)
                section_trace = section_trace[: level - 1] + [res]
                section = self._add_section(res, card=card, section_trace=section_trace)
            else:
                res = self._post_process(self.mapping(item))
                self._add_content(res, section=section)

        return card


def _get_pandoc_version() -> str:
    """Shell out to retrieve the pandoc version

    Raises
    ------
    RuntimeError
        If pandoc version could not be determined, raise a ``RuntimeError``.

    Returns
    -------
    pandoc_version : str
        The pandoc version as a list of ints.
    """
    proc = subprocess.run(
        ["pandoc", "--version"],
        capture_output=True,
    )
    version_info = str(proc.stdout.decode("utf-8")).split("\n", 1)[0]
    if not version_info.startswith("pandoc "):
        # pandoc is installed but version cannot be determined
        raise RuntimeError("Could not determine version of pandoc.")

    _, _, pandoc_version = version_info.partition(" ")
    return pandoc_version


def check_pandoc_installed(
    min_version: str | None = PANDOC_MIN_VERSION,
) -> None:
    """Check if pandoc is installed on the system

    Parameters
    ----------
    min_version : list[int] or None
        If passed, check that the pandoc version is greater or equal to this one.

    Raises
    ------
    FileNotFoundError
        When the binary is not found, raise this error.

    RuntimeError
        If pandoc version could not be determined, raise a ``RuntimeError``.

    ValueError
        If min version is passed and not matched or exceeded, raise a ``ValueError``.
    """
    try:
        pandoc_version = _get_pandoc_version()
    except FileNotFoundError as exc:
        # pandoc is not installed
        msg = (
            "This feature requires the pandoc library to be installed on your system, "
            "please follow these install instructions: "
            "https://pandoc.org/installing.html."
        )
        raise FileNotFoundError(msg) from exc

    if not min_version:
        return

    if Version(pandoc_version) < Version(min_version):
        raise ValueError(
            f"Pandoc version too low, expected at least {min_version}, "
            f"got {pandoc_version} instead."
        )


def parse_modelcard(path: str | Path) -> Card:
    """Read a model card and return a Card object

    This allows users to load a dumped model card and continue to edit it.

    Using this function requires ``pandoc`` to be installed. Please follow these
    instructions:

    https://pandoc.org/installing.html

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from skops.card import Card
    >>> from skops.card import parse_modelcard
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> regr = LinearRegression().fit(X, y)
    >>> card = Card(regr)
    >>> card.save("README.md")  # doctest: +SKIP
    >>> # later, load the card again
    >>> parsed_card = parse_modelcard("README.md")  # doctest: +SKIP
    >>> # continue editing the card
    >>> parsed_card.add(**{"My new section": "My new content"})  # doctest: +SKIP
    Card(...)
    >>> # overwrite old card with new one
    >>> parsed_card.save("README.md")  # doctest: +SKIP

    Notes
    -----
    There are some **known limitations** to the parser that may result in the
    model card generated from the parsed file not being 100% identical to the
    original model card:

    - In markdown, bold and italic text can be encoded in different fashions,
      e.g. ``_like this_`` or ``*like this*`` for italic text. Pandoc doesn't
      differentiate between the two. Therefore, the output may use one method
      where the original card used the other. When rendered, the two results
      should, however, be the same.
    - Table alignment may be different. At the moment, skops does not make use
      of column alignment information in tables, so that may differ.
    - Quote symbols may differ, e.g. ``it’s`` becoming ``it's``.
    - The number of empty lines may differ, e.g. two empty lines being
      transformed into one empty line.
    - The optional title of links is not preserved, as e.g. in
      `[text](https://example.com "this disappears")`
    - Trailing whitespace is removed.
    - Tab indentation may be removed, e.g. in raw html.
    - The yaml part of the model card can have some non-semantic differences,
      like omitting optional quotation marks.

    For these reasons, please don't expect the output of a parsed card to be
    100% identical to the original input. However, none of the listed changes
    makes any _semantic_ difference. If you find that there is a semantic
    difference in the output, please open an issue on GitHub.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the existing model card.

    Returns
    -------
    card : skops.card.Card
        The model card object.

    """
    check_pandoc_installed()

    proc = subprocess.run(
        ["pandoc", "-t", "json", "-s", str(path)],
        capture_output=True,
    )
    source = str(proc.stdout.decode("utf-8"))

    parser = PandocParser(source)
    card = parser.generate()
    return card
