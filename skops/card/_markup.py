"""Classes for translating into the syntax of different markup languages"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Sequence

from skops.card._model_card import TableSection

if sys.version_info.minor >= 9:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


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
            "Plain": self._plain,
            "Str": self._str,
            "Strong": self._strong,
            "Emph": self._emph,
            "Strikeout": self._strikeout,
            "RawInline": self._raw_inline,
            "RawBlock": self._raw_block,
            "SoftBreak": self._soft_break,
            "LineBreak": self._line_break,
            "Para": self._para,
            "Header": self._header,
            "Image": self._image,
            "Figure": self._figure,
            "CodeBlock": self._code_block,
            "Code": self._code,
            "Table": self._table,
            "Div": self._parse_div,
            "Link": self._link,
            "BulletList": self._bullet_list,
            "OrderedList": self._ordered_list,
            "Quoted": self._quoted,
            "BlockQuote": self._block_quote,
        }
        # Start indentation level at -1 because we want the first incremented
        # indentation level to be at 0. Otherwise we would need to keep track if
        # it's the first time and then don't increment, which is more
        # complicated.
        self._indent_trace = []

    @contextmanager
    def _indented(self, *, spaces: int):
        """Temporarily increment indentation by one"""
        self._indent_trace.append(spaces)
        yield
        self._indent_trace.pop(-1)

    def _get_indent(self, *, incr: int = 0) -> str:
        """Get current indentation, optionally incremented"""
        # TODO: explain why skipping 1st item
        return " " * (incr + sum(self._indent_trace[:-1]))

    @staticmethod
    def _space(value) -> str:
        return " "

    def _plain(self, value) -> str:
        parts = [self.__call__(subitem) for subitem in value]
        return "".join(parts)

    @staticmethod
    def _str(value) -> str:
        # escape \
        return value.replace("\\", "\\\\")

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

    @staticmethod
    def _raw_inline(value) -> str:
        _, line = value
        return line

    def _raw_block(self, item) -> str:
        # throw away the first item, which is just something like 'html'
        # might have to revisit this if output != markdown
        _, line = item
        return line

    def _soft_break(self, value) -> str:
        incr = 0 if not self._indent_trace else self._indent_trace[-1]
        return "\n" + self._get_indent(incr=incr)

    def _line_break(self, value) -> str:
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

        if not caption:  # pragma: no cover
            # not sure if this can be reached, just to be safe
            raise ValueError("Figure missing a caption")

        if not typef.startswith("fig:"):  # pragma: no cover
            # not sure if this can be reached, just to be safe
            raise ValueError(f"Cannot deal with figure of type '{typef}'")

        caption = "".join(self.__call__(i) for i in caption)
        content = f"![{caption}]({dest})"
        return content

    def _figure(self, value) -> str:  # pragma: no cover
        # Figure type was added in Pandoc v3.0
        (ident, classes, keyvals), caption, (body,) = value

        body_type = body["t"]
        # we can only deal with plain figures for now
        if body_type != "Plain":
            raise ValueError(f"Cannot deal with figure of type '{body_type}'")

        plain_fig = body["c"][0]["c"]
        plain_fig[2][1] = "fig:"
        return self._image(plain_fig)

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

    def _table_cols_old(self, items) -> list[str]:
        columns = []
        for (content,) in items:
            column = self.__call__(content)
            columns.append(column)
        return columns

    def _table_cols_new(self, items) -> list[str]:  # pragma: no cover
        columns = []
        fn = self.__call__
        for item in items:
            _, alignment, _, _, content = item
            column = "".join(fn(part) for part in content)
            columns.append(column)
        return columns

    def _table_body_old(self, items) -> list[list[str]]:
        body = []
        for row_items in items:
            row = []
            for col_row_item in row_items:
                if not col_row_item:
                    content = ""
                else:
                    content = col_row_item[0]
                row.append(self.__call__(content))
            body.append(row)
        return body

    def _table_body_new(self, items) -> list[list[str]]:  # pragma: no cover
        body = []
        fn = self.__call__
        for _, row_items in items:
            row = []
            for col_row_item in row_items:
                _, alignment, _, _, content = col_row_item
                row.append("".join(fn(part) for part in content))
            body.append(row)
        return body

    def _table_old(self, item) -> tuple[list[str], list[list[str]]]:
        # pandoc < 2.10
        _, _, _, thead, tbody = item
        columns = self._table_cols_old(thead)
        body = self._table_body_old(tbody)
        return columns, body

    def _table_new(self, item) -> tuple[list[str], list[list[str]]]:  # pragma: no cover
        # pandoc >= 2.10
        # attr capt specs thead tbody tfoot
        _, _, _, thead, tbody, _ = item
        # header
        (_, thead_bodies) = thead
        (attr, thead_body) = thead_bodies[0]  # multiple headers?
        columns = self._table_cols_new(thead_body)
        # rows
        # attr rhc hd bd
        _, _, _, trows = tbody[0]  # multiple groups of rows?
        body = self._table_body_new(trows)
        return columns, body

    def _table(self, item) -> str:
        if len(item) == 6:  # pragma: no cover
            # pandoc >= 2.5
            columns, body = self._table_new(item)
        else:
            # pandoc < 2.5
            columns, body = self._table_old(item)

        table: Mapping[str, Sequence[Any]]
        if not body:
            table = {key: [] for key in columns}
        else:
            # body is row oriented, transpose to column oriented
            data_transposed = zip(*body)
            table = {key: val for key, val in zip(columns, data_transposed)}

        res = TableSection(title="", content="", table=table).format()
        return res

    def _parse_div(self, item) -> str:
        # note that in markdown, we basically just use the raw html
        (ident, classes, kvs), contents = item

        # build div tag
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
            with self._indented(spaces=2):
                middle.append(self.__call__(content))
        end = "</div>"
        return "".join([start] + middle + [end])

    def _link(self, item) -> str:
        _, txt, (src, _) = item
        txt_formatted = self._make_content(txt)
        return f"[{txt_formatted}]({src})"

    def _make_list_item(self, items: str, list_marker: str):
        # helper function used for bullet and ordered lists
        parts = [self.__call__(subitem) for subitem in items]
        content = "\n".join(parts)
        return f"{self._get_indent()}{list_marker} {content}"

    def _bullet_list(self, item) -> str:
        # we don't differentiate between lists starting with "-", "*", or "+".
        list_marker = "-"
        parts = []
        # bullet lists use 2 spaces for indentation to align "- "
        with self._indented(spaces=2):
            for subitem in item:
                parts.append(self._make_list_item(subitem, list_marker=list_marker))
        return "\n".join(parts)

    def _ordered_list(self, item) -> str:
        # we don't make use of num_type and sep_type, which just indicates that
        # numbers are presented as decimal numbers using a period
        (start, num_type, sep_type), items = item
        parts = []
        # ordered lists use 3 spaces for indentation to align "1. "
        with self._indented(spaces=3):
            for i, subitem in enumerate(items, start=start):
                parts.append(self._make_list_item(subitem, list_marker=f"{i}."))
        return "\n".join(parts)

    def _quoted(self, item: tuple[dict[str, str], list[PandocItem]]) -> str:
        quote_type, content = item
        type_ = quote_type["t"]
        try:
            sym = {"DoubleQuote": '"', "SingleQuote": "'"}[type_]
        except KeyError as exc:  # pragma: no cover
            # can probably not be reached, but let's be sure
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
