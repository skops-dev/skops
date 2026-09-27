from __future__ import annotations

import operator
from typing import Any

from skops.io._audit import Node, get_tree
from skops.io._utils import LoadContext, TrustedTypes

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
        if self.module_name != "operator":
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


# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    ("OperatorFuncNode", PROTOCOL): OperatorFuncNode,
}
