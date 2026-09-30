from __future__ import annotations

import operator
from typing import Any

from skops.io import _general
from skops.io._audit import Node, get_tree
from skops.io._utils import LoadContext, TrustedTypes, gettype

PROTOCOL = 2


# Protocol 2 reader for ``operator.attrgetter``, ``operator.itemgetter`` and
# ``operator.methodcaller`` objects. Protocol 3 added a ``kwargs`` entry to the
# state so that ``methodcaller`` objects with keyword arguments round trip, and
# the current ``OperatorFuncNode`` in ``skops.io._general`` requires it. Files
# up to protocol 2 have no such entry and are read here as before, with the
# positional arguments only.
class OperatorFuncNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        if self.module_name != "operator":  # pragma: no cover
            raise ValueError(
                f"Expected module 'operator', got {self.module_name}. This is probably"
                " due to a corrupted or a malicious file."
            )
        self.trusted = self._get_trusted(trusted, [])
        self.attrs = get_tree(state["attrs"], load_context, trusted=trusted)
        self.children = {"attrs": self.attrs}

    def _construct(self):
        op = getattr(operator, self.class_name)
        attrs = self.attrs.construct()
        return op(*attrs)


# Protocol 2 readers for dicts, lists and sets. Protocol 3 added an ``attrs``
# entry to their state, holding the instance attributes of a subclass, and the
# current nodes in ``skops.io._general`` create every instance with
# ``__new__``, fill it in place and restore its attributes, as pickle does for
# dicts and lists; this also lets an item refer back to a list or set subclass
# which holds it. Files up to protocol 2 have no ``attrs`` entry and are read
# here as before: a subclass is built through its constructor, a dict subclass
# with no arguments and then filled, a list or set subclass from its items.
# The classes derive from the current nodes only to pass the ``allowed_types``
# check of the nodes which require a ``DictNode``, ``ListNode`` or ``SetNode``
# child, e.g. the key types of a ``DictNode`` or the keyword arguments of a
# ``PartialNode``; they override everything they inherit.
class ListNode(_general.ListNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        Node.__init__(self, state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [list])
        self.content = [
            get_tree(value, load_context, trusted=trusted) for value in state["content"]
        ]
        self.children = {"content": self.content}

    def _construct(self):
        content_type = gettype(self.module_name, self.class_name)
        if content_type is not list:
            return content_type([item.construct() for item in self.content])

        # Fill a plain list in place and make it available to children which
        # refer back to it, see ``Node.construct``.
        content: list[Any] = []
        self._constructed = content
        content.extend(item.construct() for item in self.content)
        return content


class SetNode(_general.SetNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        Node.__init__(self, state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [set])
        self.content = [
            get_tree(value, load_context, trusted=trusted) for value in state["content"]
        ]
        self.children = {"content": self.content}

    def _construct(self):
        content_type = gettype(self.module_name, self.class_name)
        if content_type is not set:
            return content_type([item.construct() for item in self.content])

        # Fill a plain set in place and make it available to children which
        # refer back to it, see ``Node.construct``.
        content: set[Any] = set()
        self._constructed = content
        content.update(item.construct() for item in self.content)
        return content


class DictNode(_general.DictNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        Node.__init__(self, state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [dict, "collections.OrderedDict"])
        self.key_types = get_tree(
            state["key_types"], load_context, trusted=trusted, allowed_types=(ListNode,)
        )
        self.content = {
            key: get_tree(value, load_context, trusted=trusted)
            for key, value in state["content"].items()
        }
        self.children = {"key_types": self.key_types, "content": self.content}

    def _construct(self):
        content = gettype(self.module_name, self.class_name)()
        # Make the dict available to children which refer back to it, see
        # ``Node.construct``.
        self._constructed = content
        key_types = self.key_types.construct()
        for k_type, (key, val) in zip(key_types, self.content.items()):
            content[k_type(key)] = val.construct()
        return content


# ``get_tree`` looks a node up by the protocol of the file, and falls back to
# the current node when nothing is registered for that protocol. The state of
# the objects above did not change between protocol 0 and 2, so their readers
# are registered for every one of these protocols.
NODE_TYPE_MAPPING = {
    (loader, protocol): node
    for loader, node in (
        ("OperatorFuncNode", OperatorFuncNode),
        ("DictNode", DictNode),
        ("ListNode", ListNode),
        ("SetNode", SetNode),
    )
    for protocol in range(PROTOCOL + 1)
}
