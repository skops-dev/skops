"""Contains the PandocParser

This class needs to know about the pandoc parse tree but should not have
knowledge of any particular markup syntex; everything related to markup should
be known by the mapping attribute.

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Sequence

import yaml  # type: ignore

from skops.card import Card
from skops.card._model_card import Section

from ._markup import Markdown, PandocItem

PANDOC_MIN_VERSION = (2, 19, 0)


class PandocParser:
    """TODO"""

    def __init__(self, source, mapping="markdown") -> None:
        self.source = source
        if mapping == "markdown":
            self.mapping = Markdown()
        else:
            raise ValueError(f"Markup of type {mapping} is not supported (yet)")

        self.card = Card(None, template=None)
        self._section_trace: list[str] = []
        self._cur_section: Section | None = None

    def get_cur_level(self) -> int:
        # level 0 can be interpreted implictly as the root level
        return len(self._section_trace)

    def get_cur_section(self):
        # including supersections
        return "/".join(self._section_trace)

    def add_section(self, section_name: str) -> None:
        self._cur_section = self.card._add_single(self.get_cur_section(), "")

    def add_content(self, content: str) -> None:
        section = self._cur_section
        if section is None:
            raise ValueError(
                "Ooops, no current section, please open an issue on GitHub"
            )

        if not section.content:
            section.content = content
        elif isinstance(section.content, str):
            section.content = section.content + "\n\n" + content
        else:
            # A Formattable, no generic way to modify it -- should we add an
            # update method?
            raise ValueError(f"Could not modify content of {section.content}")

    def parse_header(self, item: PandocItem) -> str:
        # Headers are the only type of item that needs to be handled
        # differently. This is because we structure the underlying model card
        # data as a tree with nodes corresponding to headers. To assign the
        # right parent or child node, we need to keep track of the level of the
        # headers. This cannot be done solely by the markdown mapping, since it
        # is not aware of the tree structure.
        level, _, _ = item["c"]
        content = self.mapping(item)
        self._section_trace = self._section_trace[: level - 1] + [content]
        return content

    def post_process(self, res: str) -> str:
        # replace Latin1 space
        res = res.replace("\xa0", " ")
        return res

    def generate(self) -> Card:
        # Parsing the flat structure, not recursively as in pandocfilters.
        # After visiting the parent node, it's not necessary to visit its
        # child nodes, because that's already done during parsing.
        for item in json.loads(self.source)["blocks"]:
            if item["t"] == "Header":
                res = self.post_process(self.parse_header(item))
                self.add_section(res)
            else:
                res = self.post_process(self.mapping(item))
                self.add_content(res)

        return self.card


def _get_pandoc_version() -> list[int]:
    """Shell out to retrieve the pandoc version

    Raises
    ------
    RuntimeError
        If pandoc version could not be determined, raise a ``RuntimeError``.

    Returns
    -------
    pandoc_version : list[int]
        The pandoc version as a list of ints.
    """
    proc = subprocess.run(
        ["pandoc", "--version"],
        capture_output=True,
    )
    version_info = str(proc.stdout.decode("utf-8")).split("\n", 1)[0]
    if not version_info.startswith("pandoc "):
        raise RuntimeError("Could not determine version of pandoc")

    _, _, actual_version = version_info.partition(" ")
    pandoc_version = [int(v) for v in actual_version.split(".")]
    return pandoc_version


def _check_version_greater_equal(
    version: Sequence[int], min_version: Sequence[int]
) -> None:
    """Very bad version comparison function to ensure that the first version is
    >= the second."""
    for v1, v2 in zip(version, min_version):
        if v1 > v2:
            return

        if v1 < v2:
            raise ValueError(
                "Pandoc version too low, expected at least "
                f"{'.'.join(map(str, min_version))}"
            )


def check_pandoc_installed(
    min_version: Sequence[int] | None = PANDOC_MIN_VERSION,
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
        msg = (
            "This feature requires the pandoc library to be installed on your system, "
            "please follow these install instructions: "
            "https://pandoc.org/installing.html"
        )
        raise FileNotFoundError(msg) from exc

    if not min_version:
        return

    _check_version_greater_equal(pandoc_version, min_version)


def _card_with_detached_metainfo(path: str | Path) -> tuple[str | Path, dict[str, Any]]:
    """Detach the possibly existing yaml part of the model card

    Model cards always have a markdown part and optionally a yaml part at the
    head, delimited by "---". Obviously, pandoc cannot parse that. Therefore, we
    detach the yaml part and return it as a separate dict, only leaving
    (hopefully) valid markdown.

    path : str or pathlib.Path
        The path to the model card file.

    Returns
    -------
    file : path
        The path to the model card without any yaml metainfo. If the model card
        didn't contain that metainfo to begin with, this is just the path to the
        original model card. If it did contain metainfo, this is a path to a new
        temporary file with the metainfo removed.

    metainfo : dict
        The metainfo from the yaml part as a parsed dict. If no metainfo was
        present, the dict is empty.
    """
    with open(path, "r") as f:
        text = f.read()

    sep_start, sep_end = "---\n", "\n---"

    metainfo: dict[str, Any] = {}
    if not text.startswith(sep_start):  # no metainfo:
        return path, metainfo

    idx_separator = text.find(sep_end)
    if idx_separator < len(sep_start):  # separator shouldn't come earlier than this
        return path, metainfo

    # https://black.readthedocs.io/en/stable/faq.html#why-are-flake8-s-e203-and-w503-violated
    text_clean = text[idx_separator + len(sep_end) :]  # noqa: E203
    metainfo = yaml.safe_load(  # type: ignore
        text[len(sep_start) : idx_separator]  # noqa: E203
    )

    file = Path(mkdtemp()) / "tmp-model-card.md"
    with open(file, "w") as f:
        f.write(text_clean)
    return file, metainfo


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
    >>> from skops.card import parse_card
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> regr = LinearRegression().fit(X, y)
    >>> card = Card(regr)
    >>> card.save("README.md")
    >>> # later, load the card again
    >>> parsed_card = parse_modelcard("README.md")
    >>> # continue editing the card
    >>> parsed_card.add(**{"My new section": "My new content"})
    >>> # overwrite old card with new one
    >>> parsed_card.save("README.md")

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

    path, metainfo = _card_with_detached_metainfo(path)

    proc = subprocess.run(
        ["pandoc", "-t", "json", "-s", str(path)],
        capture_output=True,
    )
    source = str(proc.stdout.decode("utf-8"))

    parser = PandocParser(source)
    card = parser.generate()
    for key, val in metainfo.items():
        setattr(card.metadata, key, val)

    return card
