from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Sequence
from zipfile import ZipFile

from ._audit import VALID_NODE_CHILD_TYPES, Node, get_tree
from ._general import BytearrayNode, BytesNode, FunctionNode, JsonNode, ListNode
from ._numpy import NdArrayNode
from ._scipy import SparseMatrixNode
from ._utils import LoadContext

# The children of these types are not visualized
SKIPPED_TYPES = (
    BytearrayNode,
    BytesNode,
    FunctionNode,
    JsonNode,
    NdArrayNode,
    SparseMatrixNode,
)


@dataclass
class NodeInfo:
    """Information pertinent for visualizatoin, extracted from ``Node``s.

    This class contains all information necessary for visualizing nodes. This
    way, we can have separate functions for:

    - visiting nodes and determining their safety
    - visualizing the nodes

    The visualization function will only receive the ``NodeInfo`` and does not
    have to concern itself with how to discover children or determine safety.

    """

    level: int
    key: str  # the key to the node
    val: str  # the value of the node
    is_self_safe: bool  # whether this specific node is safe
    is_safe: bool  # whether this node and all of its children are safe
    is_last: bool  # whether this is the last child of parent node


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
    color_safe: str = "cyan",
    color_unsafe: str = "orange1",
    color_child_unsafe: str = "magenta",
):
    """Determine the label of a node.

    Nodes are labeled differently based on how they're trusted.

    """
    # note: when changing the arguments to this function, also update the
    # docstring of visualize!

    if use_colors:
        try:
            import rich  # noqa
        except ImportError:
            use_colors = False

    # add tag if necessary
    node_val = node.val
    tag = tag_safe if node.is_self_safe else tag_unsafe
    if tag:
        node_val += f" {tag}"

    # colorize if so desired and if rich is installed
    if use_colors:
        if node.is_safe:
            style = f"{color_safe}"
        elif node.is_self_safe:
            style = f"{color_child_unsafe}"
        else:
            style = f"{color_unsafe}"
        node_val = f"[{style}]{node_val}[/{style}]"

    return node_val


def _traverse_tree(nodes_iter, show, **kwargs):
    """Common tree traversal logic used by both rich and fallback display methods."""
    # start with root node
    node = next(nodes_iter)
    label = _get_node_label(node, **kwargs)
    yield node, label, 0, True  # node, label, level, is_first_node

    prev_level = node.level  # should be 0

    for node in nodes_iter:
        visible = _check_visibility(node.is_self_safe, node.is_safe, show=show)
        if not visible:
            continue

        level_diff = prev_level - node.level
        if level_diff < -1:
            # this would mean it is a "(great-)grandchild" node
            raise ValueError(
                "While constructing the tree of the object, a level difference of "
                f"{level_diff} was encountered, which should not be possible, please "
                "report the issue here: https://github.com/skops-dev/skops/issues"
            )

        label = _get_node_label(node, **kwargs)
        yield node, label, level_diff, False
        prev_level = node.level


def pretty_print_tree(
    nodes_iter: Iterator[NodeInfo],
    show: Literal["all", "untrusted", "trusted"],
    **kwargs: Any,
) -> None:
    try:
        from rich.console import Console
        from rich.tree import Tree

        console = Console()

        for node, label, level_diff, is_first_node in _traverse_tree(
            nodes_iter, show, **kwargs
        ):
            if is_first_node:
                tree = Tree(f"{node.key}: {label}", guide_style="gray50")
                trees = {0: tree}
                continue

            parent_level = node.level - 1
            parent_tree = trees[parent_level]
            current_tree = parent_tree.add(f"{node.key}: {label}")
            trees[node.level] = current_tree

        console.print(tree)

    except ImportError:
        prefix = ""
        for node, label, level_diff, is_first_node in _traverse_tree(
            nodes_iter, show, **kwargs
        ):
            if is_first_node:
                print(f"{node.key}: {label}")
                continue

            # Level diff of -1 means that this node is a child of the previous node.
            # E.g. if the current level if 4 and the previous level was 3, the
            # current node is a child node of the previous one. Since the prefix for
            # a child node was already added, there is nothing more left to do.
            for _ in range(level_diff + 1):
                # This loop is entered if the current node is at the same level as,
                # or higher than, the previous node. This means the prefix has to be
                # truncated according to the level difference. E.g. if the current
                # level is 2 and previous level was 3, it means that we should move
                # up 2 layers of nesting, therefore, we trunce 3-2+1 = 2 times.
                prefix = prefix[:-4]

            print(prefix, end="")
            if node.is_last:
                print("└──", end="")
                prefix += "    "
            else:
                print("├──", end="")
                prefix += "│   "

            print(f" {node.key}: {label}")


def walk_tree(
    node: VALID_NODE_CHILD_TYPES | dict[str, VALID_NODE_CHILD_TYPES],
    node_name: str = "root",
    level: int = 0,
    is_last: bool = False,
) -> Iterator[NodeInfo]:
    """Visit all nodes of the tree and yield their important attributes.

    This function visits all nodes of the object tree and determines:

    - level: how nested the node is
    - key: the key of the node. E.g. if the node is an attribute of an object,
      the key would be the name of the attribute.
    - val: the value of the node, e.g. builtins.list
    - safety: whether it, and its children, are trusted

    These values are just yielded in a flat manner. This way, the consumer of
    this function doesn't need to know how nodes can be nested and how safety of
    a node is determined.

    Parameters
    ----------
    node: :class:`skops.io._audit.Node`
        The current node to visit. Children are visited recursively.

    node_name: str (default="root")
        The key to the current node. If "key_types" is encountered, it is
        skipped.

    level: int (default=0)
        The current level of nesting.

    is_last: bool (default=False)
        Whether this is the last node among its sibling nodes.

    Yields
    ------
    :class:`~NodeInfo`:
        A dataclass containing the aforementioned information.

    """
    # key_types is not helpful, as it is artificially added by skops to
    # circumvent the fact that json only allows keys to be strings. It is not
    # useful to the user and adds a lot of noise, thus skip key_types.
    if node_name == "key_types":
        if isinstance(node, ListNode) and node.is_safe():
            return
        raise ValueError(
            "An invalid 'key_types' node was encountered, please report the issue "
            "here: https://github.com/skops-dev/skops/issues"
        )

    if isinstance(node, dict):
        num_nodes = len(node)
        for i, (key, val) in enumerate(node.items(), start=1):
            yield from walk_tree(
                val,
                node_name=key,
                level=level,
                is_last=i == num_nodes,
            )
        return

    if isinstance(node, (list, tuple)):
        num_nodes = len(node)
        for i, val in enumerate(node, start=1):
            yield from walk_tree(
                val,
                node_name=node_name,
                level=level,
                is_last=i == num_nodes,
            )
        return

    # NO MATCH: RAISE ERROR
    if not isinstance(node, Node):
        raise TypeError(
            f"Cannot deal with {type(node)}, please report the issue here: "
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
        is_last=is_last,
    )

    # TYPES WHOSE CHILDREN IT MAKES NO SENSE TO VISIT
    # TODO: For better security, we should check the schema if we return early,
    # otherwise something nefarious could be hidden inside (however, if there
    # is, the node should be marked as unsafe)
    if isinstance(node, SKIPPED_TYPES):
        return

    yield from walk_tree(
        node.children,
        node_name=node_name,
        level=level + 1,
    )


def visualize(
    file: Path | str | bytes,
    *,
    show: Literal["all", "untrusted", "trusted"] = "all",
    trusted: Optional[Sequence[str]] = None,
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

    trusted: bool, or list of str, default=False
        The object will be loaded only if there are only trusted objects and
        objects of types listed in ``trusted`` in the dumped file.

    sink: function (default=:func:`~pretty_print_tree`)
        This function should take at least two arguments, an iterator of
        :class:`~NodeInfo` instances and an indicator of what to show. The
        ``NodeInfo`` contains the information about the node, namely:

            - the level of nesting (int)
            - the key of the node (str)
            - the value of the node as a string representation (str)
            - the safety of the node and its children

        The ``show`` argument is explained above. Any additional ``kwargs``
        passed to ``visualize`` will also be passed to ``sink``.

        The default sink is :func:`~pretty_print_tree`, which takes these
        additional parameters:

            - tag_safe: The tag used to mark trusted nodes (default="", i.e no
              tag)
            - tag_unsafe: The tag used to mark untrusted nodes
              (default="[UNSAFE]")
            - use_colors: Whether to colorize the nodes (default=True). Colors
              requires the ``rich`` package to be installed.
            - color_safe: Color to use for trusted nodes (default="orange1")
            - color_unsafe: Color to use for untrusted nodes (default="cyan")
            - color_child_unsafe: Color to use for nodes that are trusted but
              that have untrusted child ndoes (default="magenta")

        So if you don't want to have colored output, just pass
        ``use_colors=False`` to ``visualize``. The colors themselves, such
        as "orange1" and "cyan", refer to the standard colors used by ``rich``.

    """
    if isinstance(file, bytes):
        zf = ZipFile(io.BytesIO(file), "r")
    else:
        zf = ZipFile(file, "r")

    with zf as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        load_context = LoadContext(src=zip_file, protocol=schema["protocol"])
        tree = get_tree(schema, load_context=load_context, trusted=trusted)

    nodes = walk_tree(tree)
    # TODO: it would be nice to print html representation if inside a notebook
    sink(nodes, show, **kwargs)
