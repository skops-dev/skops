from __future__ import annotations

from typing import Any, Sequence

from ._protocol import PROTOCOL
from ._sklearn import ReduceNode, reduce_get_state
from ._utils import LoadContext, SaveContext


def _lazy_import():
    from quantile_forest._quantile_forest_fast import QuantileForest

    def quantile_forest_get_state(
        obj: Any,
        save_context: SaveContext,
    ) -> dict[str, Any]:
        state = reduce_get_state(obj, save_context)
        state["__loader__"] = "QuantileForestNode"
        return state

    class QuantileForestNode(ReduceNode):
        def __init__(
            self,
            state: dict[str, Any],
            load_context: LoadContext,
            trusted: bool | Sequence[str] = False,
        ) -> None:
            super().__init__(
                state,
                load_context,
                constructor=QuantileForest,
                trusted=trusted,
            )
            self.trusted = self._get_trusted(trusted, [])

    return QuantileForest, QuantileForestNode, quantile_forest_get_state


try:
    QuantileForest, QuantileForestNode, quantile_forest_get_state = _lazy_import()

    # tuples of type and function that gets the state of that type
    GET_STATE_DISPATCH_FUNCTIONS = [(QuantileForest, quantile_forest_get_state)]

    # tuples of type and function that creates the instance of that type
    NODE_TYPE_MAPPING = {("QuantileForestNode", PROTOCOL): QuantileForestNode}
except ModuleNotFoundError:
    GET_STATE_DISPATCH_FUNCTIONS = []
    NODE_TYPE_MAPPING = {}
