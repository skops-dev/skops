from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from skops.io._audit import Node, get_tree
from skops.io._utils import LoadContext, gettype

PROTOCOL = 1


class RandomGeneratorNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {
            "bit_generator_state": get_tree(
                state["content"]["bit_generator"], load_context, trusted=trusted
            )
        }
        self.trusted = self._get_trusted(trusted, [np.random.Generator])

    def _construct(self):
        # first restore the state of the bit generator
        bit_generator_state = self.children["bit_generator_state"].construct()
        bit_generator_cls = gettype(
            "numpy.random", bit_generator_state["bit_generator"]
        )
        bit_generator = bit_generator_cls()
        bit_generator.state = bit_generator_state

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
