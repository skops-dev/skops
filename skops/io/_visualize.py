from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from zipfile import ZipFile

from ..utils.importutils import import_or_raise
from ._audit import Node, get_tree
from ._general import FunctionNode, JsonNode
from ._numpy import NdArrayNode
from ._scipy import SparseMatrixNode
from ._utils import LoadContext


@dataclass
class PrintConfig:
    tag_safe: str = ""
    tag_unsafe: str = "[UNSAFE]"

    use_colors: bool = True
    # use rich for coloring
    color_safe: str = "green"
    color_unsafe: str = "red"
    color_child_unsafe: str = "yellow"


print_config = PrintConfig()


def _check_visibility(
    is_self_safe: bool,
    is_safe: bool,
    show: Literal["all", "untrusted", "trusted"],
) -> bool:
    if show == "all":
        return True

    if show == "untrusted":
        return not is_safe

    # case: show only safe node
    return is_self_safe


@dataclass
class NodeInfo:
    level: int
    key: str  # the key to the node
    val: str  # the value of the node
    is_self_safe: bool  # whether this specific node is safe
    is_safe: bool  # whether this node and all of its children are safe


def pretty_print_tree(
    nodes_iter: Iterator[NodeInfo],
    show: Literal["all", "untrusted", "trusted"],
    config: PrintConfig,
) -> None:
    # TODO: print html representation if inside a notebook
    rich = import_or_raise("rich", "pretty printing the object")
    from rich.tree import Tree

    nodes = list(nodes_iter)
    node = nodes.pop(0)
    cur_level = 0
    root = tree = Tree(f"{node.key}: {node.val}")
    trace = [tree]  # trace keeps track of what is the current node to add to

    while nodes:
        node = nodes.pop(0)
        visible = _check_visibility(node.is_self_safe, node.is_safe, show=show)
        if not visible:
            continue

        level_diff = cur_level - node.level
        if level_diff < -1:
            # this would mean it is a "(great-)grandchild" node
            raise ValueError(
                "While constructing the tree of the object, a level difference of "
                f"{level_diff} was encountered, which should not be possible, please "
                "report the issue here: https://github.com/skops-dev/skops/issues"
            )

        for _ in range(level_diff + 1):
            trace.pop(-1)
            # If level_diff == -1, we're fine with not popping, as the code
            # assumes that the current node is a child node of the previous one,
            # which corresponds to a level_diff of -1.

        # add unsafe tag if necessary
        node_val = node.val
        tag = config.tag_safe if node.is_self_safe else config.tag_unsafe
        if tag:
            node_val += f" {tag}"

        # colorize if so desired
        if config.use_colors:
            if node.is_safe:
                color = config.color_safe
            elif node.is_self_safe:
                color = config.color_child_unsafe
            else:
                color = config.color_unsafe
            node_val = f"[{color}]{node_val}"

        text = f"{node.key}: {node_val}"
        tree = trace[-1]
        trace.append(tree.add(text))
        cur_level = node.level

    rich.print(root)


def walk_tree(
    node: Node | dict[str, Node] | list[Node],
    node_name: str = "root",
    level: int = 0,
) -> Iterator[NodeInfo]:
    """Visit all nodes of the tree and yield their important attributes.

    This function visits all nodes of the object tree and determines:

    - level: how nested the node is
    - key: the key of the node, e.g. the key of a dict.
    - val: the value of the node, e.g. builtins.list
    - safety: whether it, and its children, are trusted

    These values are just yielded in a flat manner. This way, the consumer of
    this function doesn't need to know how nodes can be nested and how safety of
    a node is determined.

    Parameters
    ----------
    node: :class:`skops.io._audit.Node`
        The current node to visit. Children are visited recursively.

    show: "all" or "untrusted" or "trusted"
        Whether to print all nodes, only untrusted nodes, or only trusted nodes.

    node_name: str (default="root")
        The key to the current node. If "key_types" is encountered, it is
        skipped.

    level: int (default=0)
        The current level of nesting.

    Yields
    ------
    :class:`~NodeInfo`:
        A dataclass containing the aforementioned information.

    """
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
            )
        return

    if isinstance(node, (list, tuple)):
        # shouldn't be tuple, but check just to be sure
        for val in node:
            yield from walk_tree(
                val,
                node_name=node_name,
                level=level,
            )
        return

    # NO MATCH: RAISE ERROR
    if not isinstance(node, Node):
        raise TypeError(
            f"Cannot deal with {type(node)}, please report the issue here "
            "https://github.com/skops-dev/skops/issues"
        )

    # YIELDING THE ACTUAL NODE INFORMATION HERE

    # Note: calling node.is_safe() on all nodes is potentially wasteful because
    # it is already a recursive call, i.e. child nodes will be checked many
    # times. A solution to this would be to add caching to its call.
    yield NodeInfo(
        level=level,
        key=node_name,
        val=node.format(),
        is_self_safe=node.is_self_safe(),
        is_safe=node.is_safe(),
    )

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
    )


def visualize_tree(
    file: Path | str | bytes,
    show: Literal["all", "untrusted", "trusted"] = "all",
    sink: Callable[
        [Iterator[NodeInfo], Literal["all", "untrusted", "trusted"], PrintConfig], None
    ] = pretty_print_tree,
    **kwargs: Any,
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

        This function should take three arguments, an iterator of
        :class:`~NodeInfo` instances, an indicator of what to show, and a config
        of :class:`~PrintConfig`. The ``NodeInfo`` contains the information
        about the node, namely:

            - the level of nesting (int)
            - the key of the node (str)
            - the value of the node as a string representation (str)
            - the safety of the node and its children

        The ``show`` argument is explained above.

        The last argument is a :class:`~PrintConfig` instance, which is a
        simple dataclass with attributes that determine how the node should be
        visualized, e.g. the ``use_colors`` attribute determines if colors
        should be used.

    kwargs : TODO

    """
    if isinstance(file, bytes):
        zf = ZipFile(io.BytesIO(file), "r")
    else:
        zf = ZipFile(file, "r")

    with zf as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, load_context=LoadContext(src=zip_file))

    if kwargs:
        config = PrintConfig(**kwargs)
    else:
        config = print_config

    nodes = walk_tree(tree)
    sink(nodes, show, config)
