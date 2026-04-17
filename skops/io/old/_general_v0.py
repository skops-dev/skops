from __future__ import annotations

from typing import Any, Optional, Sequence

from skops.io._audit import Node
from skops.io._trusted_types import SCIPY_UFUNC_TYPE_NAMES
from skops.io._utils import LoadContext, _import_obj

PROTOCOL = 0


class FunctionNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, default=SCIPY_UFUNC_TYPE_NAMES)
        self.children: dict = {"content": state["content"]}

    def _construct(self):
        return _import_obj(
            self.children["content"]["module_path"],
            self.children["content"]["function"],
        )

    def _get_function_name(self) -> str:  # pragma: no cover
        return (
            self.children["content"]["module_path"]
            + "."
            + self.children["content"]["function"]
        )

    def get_unsafe_set(self) -> set[str]:  # pragma: no cover
        if (self.trusted is True) or (self._get_function_name() in self.trusted):
            return set()

        return {self._get_function_name()}


NODE_TYPE_MAPPING = {
    ("FunctionNode", PROTOCOL): FunctionNode,
}
