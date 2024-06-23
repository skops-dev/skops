from __future__ import annotations

from typing import Any, Optional, Sequence

from ._protocol import PROTOCOL
from ._sklearn import ReduceNode, reduce_get_state
from ._utils import LoadContext, SaveContext

try:
    from quantile_forest._quantile_forest_fast import QuantileForest
except Exception:
    # Mostly ImportError, but in case of older QuantileForest and numpy>=2 it
    # could also be ValueError.
    # In general, this warrants no errors on our side if the import fails.
    QuantileForest = None


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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        if QuantileForest is None:
            raise ImportError(
                "`quantile_forest` is missing and needs to be installed in order to"
                " load this model."
            )

        super().__init__(
            state,
            load_context,
            constructor=QuantileForest,
            trusted=trusted,
        )
        self.trusted = self._get_trusted(trusted, [])


# tuples of type and function that gets the state of that type
if QuantileForest is not None:
    GET_STATE_DISPATCH_FUNCTIONS = [(QuantileForest, quantile_forest_get_state)]

# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {("QuantileForestNode", PROTOCOL): QuantileForestNode}
