from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal
from zipfile import ZipFile

from ._audit import Node, get_tree
from ._general import FunctionNode, JsonNode
from ._numpy import NdArrayNode
from ._scipy import SparseMatrixNode
from ._utils import LoadContext


@dataclass
class PrintConfig:
    # fmt: off
    tag_safe: str =   ""  # noqa: E222
    tag_unsafe: str = "[UNSAFE]"

    line_start: str = "├─"
    line: str =       "──"  # noqa: E222

    use_colors: bool = True
    color_safe: str =         '\033[32m'  # green  # noqa: E222
    color_unsafe: str =       '\033[31m'  # red  # noqa: E222
    color_child_unsafe: str = '\033[33m'  # yellow
    color_end: str =          '\033[0m'   # noqa: E222
    # fmt: on


print_config = PrintConfig()


@dataclass
class FormattedNode:
    level: int
    key: str  # the key to the node
    val: str  # the value of the node
    visible: bool  # whether it should be shown


def pretty_print_tree(
    formatted_nodes: Iterator[FormattedNode], config: PrintConfig
) -> None:
    # TODO: the "tree" lines could be made prettier since all nodes are known
    # here
    for formatted_node in formatted_nodes:
        if not formatted_node.visible:
            continue

        line = print_config.line_start
        line += (formatted_node.level - 1) * print_config.line
        line += f"{formatted_node.key}: {formatted_node.val}"
        print(line)


def _check_visibility(
    node: Node,
    node_is_safe: bool,
    node_and_children_are_safe: bool,
    show: Literal["all", "untrusted", "trusted"],
) -> bool:
    if show == "all":
        should_print = True
    elif show == "untrusted":
        should_print = not node_and_children_are_safe
    else:  # only trusted
        should_print = node_is_safe
    return should_print


def walk_tree(
    node: Node | dict[str, Node] | list[Node],
    show: Literal["all", "untrusted", "trusted"] = "all",
    node_name: str = "root",
    level: int = 0,
    config: PrintConfig = print_config,
) -> Iterator[FormattedNode]:
    # helper function to pretty-print the nodes
    if node_name == "key_types":
        # _check_key_types_schema(node)
        return

    # COMPOSITE TYPES: CHECK ALL ITEMS
    if isinstance(node, dict):
        for key, val in node.items():
            yield from walk_tree(
                val,
                node_name=key,
                level=level,
                show=show,
                config=config,
            )
        return

    if isinstance(node, (list, tuple)):  # shouldn't be tuple, but to be sure
        for val in node:
            yield from walk_tree(
                val,
                node_name=node_name,
                level=level,
                show=show,
                config=config,
            )
        return

    # NO MATCH: RAISE ERROR
    if not isinstance(node, Node):
        raise TypeError(f"{type(node)}")

    # THE ACTUAL FORMATTING HAPPENS HERE
    node_is_safe = node.is_self_safe()
    node_and_children_are_safe = node.is_safe()
    visible = _check_visibility(
        node,
        node_is_safe=node_is_safe,
        node_and_children_are_safe=node_and_children_are_safe,
        show=show,
    )

    node_val = node.format()
    tag = config.tag_safe if node_is_safe else config.tag_unsafe
    if tag:
        node_val += f" {tag}".rstrip(" ")

    if config.use_colors:
        if node_and_children_are_safe:
            color = config.color_safe
        elif node_is_safe:
            color = config.color_child_unsafe
        else:
            color = config.color_unsafe
        node_val = f"{color}{node_val}{config.color_end}"

    yield FormattedNode(level=level, key=node_name, val=node_val, visible=visible)

    # TYPES WHOSE CHILDREN IT MAKES NO SENSE TO VISIT
    if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type != "json"):
        # _check_array_schema(node)
        return

    if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type == "json"):
        # _check_array_json_schema(node)
        return

    if isinstance(node, FunctionNode):
        # _check_function_schema(node)
        return

    if isinstance(node, JsonNode):
        # _check_json_schema(node)
        pass

    # RECURSE
    yield from walk_tree(
        node.children,
        node_name=node_name,
        level=level + 1,
        show=show,
        config=config,
    )


def visualize_tree(
    file: Path | str,  # TODO: from bytes
    show: Literal["all", "untrusted", "trusted"] = "all",
    sink: Callable[[Iterator[FormattedNode], PrintConfig], None] = pretty_print_tree,
    print_config: PrintConfig = print_config,
) -> None:
    """Visualize the contents of a skops file.

    Shows the schema of a skops file as a tree view. In particular, highlights
    untrusted nodes. A node is considered untrusted if at least one of its child
    nodes is untrusted.

    Parameters
    ----------
    file: str or pathlib.Path
        The file name of the object to be loaded.

    show: "all" or "untrusted" or "trusted"
        Whether to print all nodes, only untrusted nodes, or only trusted nodes.

    sink: function
        This function should take two arguments, an iterator of
        ``FormattedNode`` and a ``PrintConfig``. The ``FormattedNode`` contains
        the information about the node, namely:

            - the level of nesting (int)
            - the key of the node (str)
            - the value of the node as a string representation (str)
            - the visibility of the node, depending on the ``show`` argument (bool)

        The second argument is the print config (see description of next argument).

    print_config: :class:`~PrintConfig`
        The ``PrintConfig`` is a simple object with attributes that determine
        how the node should be visualized, e.g. the ``use_colors`` attribute
        determines if colors should be used.

    """
    with ZipFile(file, "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, load_context=LoadContext(src=zip_file))

    nodes = walk_tree(tree, show=show, config=print_config)
    sink(nodes, print_config)
