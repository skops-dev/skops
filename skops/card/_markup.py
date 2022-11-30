"""Classes for translating into the syntax of different markup languages"""

from collections.abc import Mapping
from typing import Any, Sequence, TypedDict

from skops.card._model_card import TableSection


class PandocItem(TypedDict):
    t: str
    c: dict


class Markdown:
    """Mapping of pandoc parsed document to Markdown

    This class has a ``mapping`` attribute, which is just a dict. The keys are
    Pandoc types and the values are functions that transform the corresponding
    value into a string with markdown syntax. Those functions are all prefixed
    with ``md_``, e.g. ``md_Image`` for transforming a pandoc ``Image`` into a
    markdown figure.

    From the caller side, only the ``__call__`` method should be used, the rest
    should be considered internals.

    """

    def __init__(self):
        # markdown syntax dispatch table
        self.mapping = {
            "Space": self.md_space,
            "Strong": self.md_strong,
            "Plain": self.md_plain,
            "Str": self.md_str,
            "RawInline": self.md_rawline,
            "RawBlock": self.md_raw_block,
            "SoftBreak": self.md_softbreak,
            "Para": self.md_para,
            "Header": self.md_header,
            "Image": self.md_image,
            "CodeBlock": self.md_code_block,
            "Table": self.md_table,
            "Div": self.md_parse_div,
        }

    @staticmethod
    def md_space(value) -> str:
        return " "

    def md_strong(self, value) -> str:
        parts = ["**"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("**")
        return "".join(parts)

    def md_plain(self, value) -> str:
        parts = [self.__call__(subitem) for subitem in value]
        return "".join(parts)

    @staticmethod
    def md_str(value) -> str:
        return value

    @staticmethod
    def md_rawline(value) -> str:
        _, line = value
        return line

    def md_raw_block(self, item) -> str:
        # throw away the first item, which is just something like 'html'
        # might have to revisit this if output != markdown
        _, line = item
        return line

    @staticmethod
    def md_softbreak(value) -> str:
        return "\n"

    def _make_content(self, content):
        parts = []
        for item in content:
            part = "".join(self.__call__(item))
            parts.append(part)
        return "".join(parts)

    def md_para(self, value: list[dict[str, str]]) -> str:
        content = self._make_content(value)
        return content

    def md_header(self, value: tuple[int, Any, list[dict[str, str]]]) -> str:
        level, _, content_parts = value
        section_name = self._make_content(content_parts)
        return section_name

    def md_image(self, value) -> str:
        (ident, _, keyvals), caption, (dest, typef) = value
        # it seems like ident and keyvals are not relevant for markdown
        assert caption
        assert typef == "fig:"

        caption = "".join([self.__call__(i) for i in caption])
        content = f"![{caption}]({dest})"
        return content

    @staticmethod
    def md_code_block(item: tuple[tuple[int, list[str], list[str]], str]) -> str:
        # a codeblock consists of: (id, classes, namevals) contents
        (_, _, namevals), content = item
        block_start = "```"
        if namevals:  # TODO: check if this makes "```python" etc.
            block_start += namevals[0]
        block_end = "```"
        content = "\n".join((block_start, content, block_end))
        return content

    def md_table(self, item) -> str:
        _, alignments, _, header, rows = item
        fn = self.__call__
        columns = ["".join(fn(part) for part in col) for col in header]
        if not columns:
            raise ValueError("Table with no columns...")

        data = []  # row oriented
        for row in rows:
            data.append(["".join(fn(part) for part in col) for col in row])

        table: Mapping[str, Sequence[Any]]
        if not data:
            table = {key: [] for key in columns}
        else:
            data_transposed = zip(*data)  # column oriented
            table = {key: val for key, val in zip(columns, data_transposed)}

        res = TableSection(table).format()
        return res

    def md_parse_div(self, item) -> str:
        # note that in markdown, we basically just use the raw html
        (ident, classes, kvs), contents = item

        # build diff tag
        tags = ["<div"]
        if ident:
            tags.append(f' id="{ident}"')
        if classes:
            classes = " ".join(classes)
            tags.append(f' class="{classes}"')
        if kvs:
            kvparts = []
            for k, v in kvs:
                if not v:  # e.g. just ['hidden', '']
                    kvparts.append(k)
                else:
                    kvparts.append(f'{k}="{v}"')
            tags.append(f' {" ".join(kvparts)}')
        tags.append(">")

        start = "".join(tags)
        middle = []
        for content in contents:
            middle.append(self.__call__(content))
        end = "</div>"
        return "".join([start] + middle + [end])

    def __call__(self, item: str | PandocItem) -> str:
        if isinstance(item, str):
            return item

        type_, value = item["t"], item.get("c")
        try:
            res = self.mapping[type_](value)
        except KeyError as exc:
            msg = (
                f"The parsed document contains '{type_}', which is not "
                "supported yet, please open an issue on GitHub"
            )
            raise ValueError(msg) from exc

        # recursively call until the value has been resolved into a str
        return self.__call__(res)
