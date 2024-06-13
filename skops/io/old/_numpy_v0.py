from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from skops.io._audit import Node
from skops.io._utils import LoadContext, gettype

PROTOCOL = 0


class RandomGeneratorNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {"bit_generator_state": state["content"]["bit_generator"]}
        self.trusted = self._get_trusted(trusted, [np.random.Generator])

    def _construct(self):
        # first restore the state of the bit generator
        bit_generator = gettype(
            "numpy.random", self.children["bit_generator_state"]["bit_generator"]
        )()
        bit_generator.state = self.children["bit_generator_state"]

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


NODE_TYPE_MAPPING = {
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
