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
class NodeInfo:
    level: int
    key: str  # the key to the node
    val: str  # the value of the node
    is_self_safe: bool  # whether this specific node is safe
    is_safe: bool  # whether this node and all of its children are safe


def _check_visibility(
    is_self_safe: bool,
    is_safe: bool,
    show: Literal["all", "untrusted", "trusted"],
) -> bool:
    """Determine visibility of the node.

    Users can indicate if they want to see all nodes, all trusted nodes, or all
    untrusted nodes.

    """
    if show == "all":
        return True

    if show == "untrusted":
        return not is_safe

    # case: show only safe node
    return is_self_safe


def _get_node_label(
    node: NodeInfo,
    tag_safe: str = "",
    tag_unsafe: str = "[UNSAFE]",
    use_colors: bool = True,
    # use rich for coloring
    color_safe: str = "green",
    color_unsafe: str = "red",
    color_child_unsafe: str = "yellow",
):
    """Determine the label of a node.

    Nodes are labeled differently based on how they're trusted.

    """
    # note: when changing the arguments to this function, also update the
    # docstring of visualize_tree!

    # add tag if necessary
    node_val = node.val
    tag = tag_safe if node.is_self_safe else tag_unsafe
    if tag:
        node_val += f" {tag}"

    # colorize if so desired
    if use_colors:
        if node.is_safe:
            color = color_safe
        elif node.is_self_safe:
            color = color_child_unsafe
        else:
            color = color_unsafe
        node_val = f"[{color}]{node_val}"

    return node_val


def pretty_print_tree(
    nodes_iter: Iterator[NodeInfo],
    show: Literal["all", "untrusted", "trusted"],
    **kwargs,
) -> None:
    # This function loops through the flattened nodes of the tree and creates a
    # rich Tree based on the node information. Rich can then create a pretty
    # visualization of said tree.
    rich = import_or_raise("rich", "pretty printing the object")
    from rich.tree import Tree

    nodes = list(nodes_iter)
    if not nodes:  # empty tree, hmm
        return

    # start with root node, it is always visible
    node = nodes.pop(0)
    node_label = _get_node_label(node, **kwargs)
    cur_level = node.level  # should be 0
    root = tree = Tree(f"{node.key}: {node_label}")
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

        # Level diff of -1 means that this node is a child of the previous node.
        # E.g. if the current level if 4 and the previous level was 3, the
        # current node is a child node of the previous one. Since the previous
        # node is already the last node in the trace, there is nothing more that
        # needs to be done. Therefore, for a diff of -1, we don't pop from the
        # trace.
        for _ in range(level_diff + 1):
            # If the level diff is greater than -1, it means that the current
            # node is not the child of the last node, but of a node higher up.
            # E.g. if the current level is 2 and previous level was 3, it means
            # that we should move up 2 layers of nesting, therefore, we pop
            # 3-2+1 = 2 levels.
            trace.pop(-1)

        # add tag if necessary
        node_label = _get_node_label(node, **kwargs)
        text = f"{node.key}: {node_label}"
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
    # key_types is not helpful, as it is artificially added by skops to
    # circumvent the fact that json only allows keys to be strings. It is not
    # useful to the user and adds a lot of noise, thus skip key_types.
    # TODO: check that no funny business is going on in key types
    if node_name == "key_types":
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
    # TODO: For better security, we should check the schema if we return early,
    # otherwise something nefarious could be hidden inside.
    if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type != "json"):
        return

    if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type == "json"):
        return

    if isinstance(node, FunctionNode):
        return

    if isinstance(node, JsonNode):
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
    sink: Callable[..., None] = pretty_print_tree,
    **kwargs: Any,
) -> None:
    """Visualize the contents of a skops file.

    Shows the schema of a skops file as a tree view. In particular, highlights
    untrusted nodes. A node is considered untrusted if at least one of its child
    nodes is untrusted.

    Visualizing the tree using the default visualization function requires the
    ``rich`` library, which can be installed as:

        python -m pip install rich

    If passing a custom visualization function to ``sink``, ``rich`` is not
    required.

    Parameters
    ----------
    file: str or pathlib.Path
        The file name of the object to be loaded.

    show: "all" or "untrusted" or "trusted"
        Whether to print all nodes, only untrusted nodes, or only trusted nodes.

    sink: function (default=:func:`~pretty_print_tree`)

        This function should take at least two arguments, an iterator of
        :class:`~NodeInfo` instances and an indicator of what to show. The
        ``NodeInfo`` contains the information about the node, namely:

            - the level of nesting (int)
            - the key of the node (str)
            - the value of the node as a string representation (str)
            - the safety of the node and its children

        The ``show`` argument is explained above. Any additional ``kwargs``
        passed to ``visualize_tree`` will also be passed to ``sink``.

        The default sink is :func:`~pretty_print_tree`, which takes these
        additional parameters:

            - tag_safe: The tag used to mark trusted nodes (default="", i.e no
              tag)
            - tag_unsafe: The tag used to mark untrusted nodes
              (default="[UNSAFE]")
            - use_colors: Whether to colorize the nodes (default=True)
            - color_safe: Color to use for trusted nodes (default="green")
            - color_unsafe: Color to use for untrusted nodes (default="red")
            - color_child_unsafe: Color to use for nodes that are trusted but
              that have untrusted child ndoes (default="yellow")

        So if you don't want to have colored output, just pass
        ``use_colors=False`` to ``visualize_tree``. The colors themselves, such
        as "red" and "green", refer to the standard colors used by ``rich``.

    """
    if isinstance(file, bytes):
        zf = ZipFile(io.BytesIO(file), "r")
    else:
        zf = ZipFile(file, "r")

    with zf as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, load_context=LoadContext(src=zip_file))

    nodes = walk_tree(tree)
    # TODO: it would be nice to print html representation if inside a notebook
    sink(nodes, show, **kwargs)
