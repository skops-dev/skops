"""Contains the PandocParser

This class needs to know about the pandoc parse tree but should not have
knowledge of any particular markup syntex; everything related to markup should
be known by the mapping attribute.

"""

import json
import subprocess
from pathlib import Path

from skops.card import Card
from skops.card._model_card import Section

from ._markup import Markdown, PandocItem


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

    def generate(self) -> Card:
        # Parsing the flat structure, not recursively as in pandocfilters.
        # After visiting the parent node, it's not necessary to visit its
        # child nodes, because that's already done during parsing.
        for item in json.loads(self.source)["blocks"]:
            if item["t"] == "Header":
                res = self.parse_header(item)
                self.add_section(res)
            else:
                res = self.mapping(item)
                self.add_content(res)

        return self.card


def check_pandoc_installed() -> None:
    """Check if pandoc is installed on the system

    Raises
    ------
    FileNotFoundError
        When the binary is not found, raise this error.

    """
    try:
        subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
        )
    except FileNotFoundError as exc:
        msg = (
            "This feature requires the pandoc library to be installed on your system, "
            "please follow these install instructions: "
            "https://pandoc.org/installing.html"
        )
        raise FileNotFoundError(msg) from exc


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
