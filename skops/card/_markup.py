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
    with ``_``, e.g. ``_image`` for transforming a pandoc ``Image`` into a
    markdown figure, or ``_raw_block``, to transform a pandoc ``RawBlock``.

    From the caller side, only the ``__call__`` method should be used, the rest
    should be considered internals.

    """

    def __init__(self):
        # markdown syntax dispatch table
        self.mapping = {
            "Space": self._space,
            "Strong": self._strong,
            "Emph": self._emph,
            "Strikeout": self._strikeout,
            "Subscript": self._subscript,
            "Superscript": self._superscript,
            "Plain": self._plain,
            "Str": self._str,
            "RawInline": self._raw_inline,
            "RawBlock": self._raw_block,
            "SoftBreak": self._soft_break,
            "LineBreak": self._line_break,
            "Para": self._para,
            "Header": self._header,
            "Image": self._image,
            "CodeBlock": self._code_block,
            "Code": self._code,
            "Table": self._table,
            "Div": self._parse_div,
            "Link": self._link,
            "BulletList": self._bullet_list,
            "Quoted": self._quoted,
            "BlockQuote": self._block_quote,
        }

    @staticmethod
    def _space(value) -> str:
        return " "

    def _strong(self, value) -> str:
        parts = ["**"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("**")
        return "".join(parts)

    def _emph(self, value) -> str:
        parts = ["_"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("_")
        return "".join(parts)

    def _strikeout(self, value) -> str:
        parts = ["~~"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("~~")
        return "".join(parts)

    def _subscript(self, value) -> str:
        parts = ["<sub>"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("</sub>")
        return "".join(parts)

    def _superscript(self, value) -> str:
        parts = ["<sup>"]
        parts += [self.__call__(subitem) for subitem in value]
        parts.append("</sup>")
        return "".join(parts)

    def _plain(self, value) -> str:
        parts = [self.__call__(subitem) for subitem in value]
        return "".join(parts)

    @staticmethod
    def _str(value) -> str:
        # escape \
        return value.replace("\\", "\\\\")

    @staticmethod
    def _raw_inline(value) -> str:
        _, line = value
        return line

    def _raw_block(self, item) -> str:
        # throw away the first item, which is just something like 'html'
        # might have to revisit this if output != markdown
        _, line = item
        return line

    @staticmethod
    def _soft_break(value) -> str:
        return "\n"

    @staticmethod
    def _line_break(value) -> str:
        return "\n"

    def _make_content(self, content):
        parts = []
        for item in content:
            part = "".join(self.__call__(item))
            parts.append(part)
        return "".join(parts)

    def _para(self, value: list[dict[str, str]]) -> str:
        content = self._make_content(value)
        return content

    def _header(self, value: tuple[int, Any, list[dict[str, str]]]) -> str:
        level, _, content_parts = value
        section_name = self._make_content(content_parts)
        return section_name

    def _image(self, value) -> str:
        (ident, _, keyvals), caption, (dest, typef) = value
        # it seems like ident and keyvals are not relevant for markdown
        assert caption
        assert typef == "fig:"

        caption = "".join([self.__call__(i) for i in caption])
        content = f"![{caption}]({dest})"
        return content

    @staticmethod
    def _code_block(item: tuple[tuple[int, list[str], list[str]], str]) -> str:
        # a codeblock consists of: (id, classes, namevals) contents
        (_, classes, _), content = item
        block_start = "```"
        if classes:
            block_start += ", ".join(classes)
        block_end = "```"
        content = "\n".join((block_start, content, block_end))
        return content

    @staticmethod
    def _code(item: tuple[Any, str]) -> str:
        _, txt = item
        return f"`{txt}`"

    def _table(self, item) -> str:
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

    def _parse_div(self, item) -> str:
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

    def _link(self, item) -> str:
        _, txt, (src, _) = item
        txt_formatted = self._make_content(txt)
        return f"[{txt_formatted}]({src})"

    def _bullet_list(self, item) -> str:
        parts = []
        for subitem in item:
            assert len(subitem) == 1
            content = "".join(self.__call__(i) for i in subitem)
            # indent the lines in lists if they contain line breaks
            content = content.replace("\n", "\n  ")
            parts.append(f"- {content}")
        return "\n".join(parts)

    def _quoted(self, item: tuple[dict[str, str], list[PandocItem]]) -> str:
        quote_type, content = item
        type_ = quote_type["t"]
        try:
            sym = {"DoubleQuote": '"', "SingleQuote": "'"}[type_]
        except KeyError as exc:
            msg = (
                f"The parsed document contains '{type_}', which is not "
                "supported yet, please open an issue on GitHub"
            )
            raise ValueError(msg) from exc

        text = "".join(self.__call__(i) for i in content)
        return f"{sym}{text}{sym}"

    def _block_quote(self, item: list[PandocItem]) -> str:
        parts = []
        for subitem in item:
            content = self.__call__(subitem)
            # add quote symbolx
            content = content.replace("\n", "\n> ")
            parts.append(content)

        # add a quote symbol to the very start
        text = "> " + "\n> ".join(parts)
        return text

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
