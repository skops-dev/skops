from __future__ import annotations

import json
from functools import singledispatch
from pathlib import Path
from typing import Callable, Literal, Sequence
from zipfile import ZipFile

from ._audit import Node, get_tree
from ._general import FunctionNode, JsonNode
from ._numpy import NdArrayNode
from ._scipy import SparseMatrixNode
from ._utils import LoadContext

PrintFn = Callable[
    [Node, str, int, bool | Sequence[str], Literal["all", "untrusted", "trusted"]], None
]


class PrintConfig:
    # fmt: off
    print_fn = print
    template = "{prefix}{key}: {name}{tag}"

    tag_safe =   ""  # noqa: E222
    tag_unsafe = " [UNSAFE]"

    line_start = "├─"
    line =       "──"  # noqa: E222

    use_colors = True
    color_safe =         '\033[32m'  # green  # noqa: E222
    color_unsafe =       '\033[31m'  # red  # noqa: E222
    color_child_unsafe = '\033[33m'  # yellow
    color_end =          '\033[0m'   # noqa: E222
    # fmt: on


print_config = PrintConfig()


def _check_should_print(
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


def _check_node_and_children_safe(node: Node, trusted: bool | Sequence[str]) -> bool:
    # Note: this is very inefficient, because get_unsafe_set will be called many
    # times on the same node (since parents recursively call children) but maybe
    # that's acceptable for this context. If not, caching could be an option.
    if trusted is True:
        node_and_children_are_safe = True
    elif trusted is False:
        node_and_children_are_safe = not node.get_unsafe_set()
    else:
        node_and_children_are_safe = not (node.get_unsafe_set() - set(trusted))
    return node_and_children_are_safe


def _print_node(
    node: Node,
    name: str,
    key: str,
    level: int,
    trusted: bool | Sequence[str],
    show: Literal["all", "untrusted", "trusted"],
    config: PrintConfig = print_config,
):
    # determine if node itself and if its children are safe
    node_is_safe = node.is_self_safe()
    node_and_children_are_safe = _check_node_and_children_safe(node, trusted)

    should_print = _check_should_print(
        node,
        node_is_safe=node_is_safe,
        node_and_children_are_safe=node_and_children_are_safe,
        show=show,
    )
    if not should_print:
        return

    tag = config.tag_safe if node_is_safe else config.tag_unsafe

    prefix = "  " * (level - 1) + config.line_start

    # prefix = ""
    # if level > 0:
    #     prefix += config.line_start
    # if level > 1:
    #     prefix += config.line * (level - 1)

    template = config.template
    if config.use_colors:
        if node_and_children_are_safe:
            color = config.color_safe
        elif node_is_safe:
            color = config.color_child_unsafe
        else:
            color = config.color_unsafe
        name = f"{color}{name}{config.color_end}"

    text = template.format(prefix=prefix, key=key, name=name, tag=tag)
    config.print_fn(text)


# use singledispatch so that we can register specialized visualization functions
@singledispatch
def print_node(
    node: Node,
    key: str,
    level: int,
    trusted: bool | Sequence[str],
    show: Literal["all", "untrusted", "trusted"],
):
    name = f"{node.module_name}.{node.class_name}"
    _print_node(node, name=name, key=key, level=level, trusted=trusted, show=show)


@print_node.register
def _print_function_node(
    node: FunctionNode,
    key: str,
    level: int,
    trusted: bool | Sequence[str],
    show: Literal["all", "untrusted", "trusted"],
):
    # if a FunctionNode, children are not visited, but safety should still be checked
    child = node.children["content"]
    fn_name = f"{child['module_path']}.{child['function']}"
    name = f"{node.module_name}.{node.class_name} => {fn_name}"
    _print_node(node, name=name, key=key, level=level, trusted=trusted, show=show)


@print_node.register
def _print_json_node(
    node: JsonNode,
    key: str,
    level: int,
    trusted: bool | Sequence[str],
    show: Literal["all", "untrusted", "trusted"],
):
    name = f"json-type({node.content})"
    return _print_node(
        node, name=name, key=key, level=level, trusted=trusted, show=show
    )


def _visualize_tree(
    node: Node | dict[str, Node] | Sequence[Node],
    trusted: bool | Sequence[str] = False,
    show: Literal["all", "untrusted", "trusted"] = "all",
    node_name: str = "root",
    level: int = 0,
    sink: PrintFn = print_node,
) -> None:
    # helper function to pretty-print the nodes
    if node_name == "key_types":
        # _check_key_types_schema(node)
        return

    # COMPOSITE TYPES: CHECK ALL ITEMS
    if isinstance(node, dict):
        for key, val in node.items():
            _visualize_tree(
                val, node_name=key, level=level, trusted=trusted, show=show, sink=sink
            )
        return

    if isinstance(node, (list, tuple)):
        for val in node:
            _visualize_tree(
                val,
                node_name=node_name,
                level=level,
                trusted=trusted,
                show=show,
                sink=sink,
            )
        return

    # NO MATCH: RAISE ERROR
    if not isinstance(node, Node):
        raise TypeError(f"{type(node)}")

    # TRIGGER SIDE-EFFECT
    sink(node, node_name, level, trusted, show)

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
    _visualize_tree(
        node.children,
        node_name=node_name,
        level=level + 1,
        trusted=trusted,
        show=show,
        sink=sink,
    )


def visualize_tree(
    file: Path | str,  # TODO: from bytes
    trusted: bool | Sequence[str] = False,
    show: Literal["all", "untrusted", "trusted"] = "all",
    sink: PrintFn = print_node,
) -> None:
    """Visualize the contents of a skops file.

    Shows the schema of a skops file as a tree view. In particular, highlights
    untrusted nodes. A node is considered untrusted if at least one of its child
    nodes is untrusted.

    Parameters
    ----------
    file: str or pathlib.Path
        The file name of the object to be loaded.

    trusted: bool, or list of str, default=False
        If ``True``, the object will be loaded without any security checks. If
        ``False``, the object will be loaded only if there are only trusted
        objects in the dumped file. If a list of strings, the object will be
        loaded only if there are only trusted objects and objects of types
        listed in ``trusted`` are in the dumped file.

    show: "all" or "untrusted" or "trusted"
        Whether to print all nodes, only untrusted nodes, or only trusted nodes.

    sink: function

        Function used to print the schema. By default, this generates a tree
        view and prints it to stdout. If you want to do something else with the
        output, e.g. log it to a file, pass a function here that does that. The
        signature of this function should be ``Callable[[Node, str, int, bool |
        Sequence[str], Literal["all", "untrusted", "trusted"]], None]``.

    """
    with ZipFile(file, "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, load_context=LoadContext(src=zip_file))
    _visualize_tree(tree, trusted=trusted, show=show, sink=sink)


# def _walk_tree(
#     node: Node | dict[str, Node] | Sequence[Node],
#     trusted: bool | Sequence[str] = False,
#     show: Literal["all", "untrusted", "trusted"] = "all",
#     node_name: str = "root",
#     level: int = 0,
#     sink: PrintFn = print_node,
# ) -> Iterator[Any]:  # TODO
#     # helper function to pretty-print the nodes
#     if node_name == "key_types":
#         # _check_key_types_schema(node)
#         return

#     # COMPOSITE TYPES: CHECK ALL ITEMS
#     if isinstance(node, dict):
#         for key, val in node.items():
#             yield from _walk_tree(
#                 val, node_name=key, level=level, trusted=trusted, show=show, sink=sink
#             )
#         return

#     if isinstance(node, (list, tuple)):
#         for val in node:
#             yield from _walk_tree(
#                 val,
#                 node_name=node_name,
#                 level=level,
#                 trusted=trusted,
#                 show=show,
#                 sink=sink,
#             )
#         return

#     # NO MATCH: RAISE ERROR
#     if not isinstance(node, Node):
#         raise TypeError(f"{type(node)}")

#     # TRIGGER SIDE-EFFECT
#     sink(node, node_name, level, trusted, show)

#     # TYPES WHOSE CHILDREN IT MAKES NO SENSE TO VISIT
#     if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type != "json"):
#         # _check_array_schema(node)
#         return

#     if isinstance(node, (NdArrayNode, SparseMatrixNode)) and (node.type == "json"):
#         # _check_array_json_schema(node)
#         return

#     if isinstance(node, FunctionNode):
#         # _check_function_schema(node)
#         return

#     if isinstance(node, JsonNode):
#         # _check_json_schema(node)
#         pass

#     # RECURSE
#     yield from _walk_tree(
#         node.children,
#         node_name=node_name,
#         level=level + 1,
#         trusted=trusted,
#         show=show,
#         sink=sink,
#     )


# def visualize_tree(
#     file: Path | str,  # TODO: from bytes
#     trusted: bool | Sequence[str] = False,
#     show: Literal["all", "untrusted", "trusted"] = "all",
#     sink: PrintFn = print_node,
# ) -> None:
#     """Visualize the contents of a skops file.

#     Shows the schema of a skops file as a tree view. In particular, highlights
#     untrusted nodes. A node is considered untrusted if at least one of its child
#     nodes is untrusted.

#     Parameters
#     ----------
#     file: str or pathlib.Path
#         The file name of the object to be loaded.

#     trusted: bool, or list of str, default=False
#         If ``True``, the object will be loaded without any security checks. If
#         ``False``, the object will be loaded only if there are only trusted
#         objects in the dumped file. If a list of strings, the object will be
#         loaded only if there are only trusted objects and objects of types
#         listed in ``trusted`` are in the dumped file.

#     show: "all" or "untrusted" or "trusted"
#         Whether to print all nodes, only untrusted nodes, or only trusted nodes.

#     sink: function

#         Function used to print the schema. By default, this generates a tree
#         view and prints it to stdout. If you want to do something else with the
#         output, e.g. log it to a file, pass a function here that does that. The
#         signature of this function should be ``Callable[[Node, str, int, bool |
#         Sequence[str], Literal["all", "untrusted", "trusted"]], None]``.

#     """
#     with ZipFile(file, "r") as zip_file:
#         schema = json.loads(zip_file.read("schema.json"))
#         tree = get_tree(schema, load_context=LoadContext(src=zip_file))
#     _visualize_tree(tree, trusted=trusted, show=show, sink=sink)
