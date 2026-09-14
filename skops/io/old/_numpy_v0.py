from __future__ import annotations

from typing import Any

import numpy as np

from skops.io._audit import Node
from skops.io._utils import LoadContext, gettype

PROTOCOL = 0


class RandomGeneratorNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {"bit_generator_state": state["content"]["bit_generator"]}
        self.trusted = self._get_trusted(trusted, [np.random.Generator])

    def _construct(self):
        # first restore the state of the bit generator
        bit_generator_name = self.children["bit_generator_state"]["bit_generator"]
        bit_generator_cls = gettype("numpy.random", bit_generator_name)
        # The bit generator's class name comes from the file and is used to look
        # up and call a ``numpy.random`` attribute (see ``RandomGeneratorNode``
        # in ``skops.io._numpy`` for the full explanation). Retrieving the
        # attribute is harmless, calling it is not, so refuse to call anything
        # that isn't a genuine bit generator.
        if not (
            isinstance(bit_generator_cls, type)
            and issubclass(bit_generator_cls, np.random.BitGenerator)
        ):
            raise ValueError(
                f"Expected a numpy.random bit generator, got {bit_generator_name!r}."
                " This is probably due to a corrupted or a malicious file."
            )
        bit_generator = bit_generator_cls()
        bit_generator.state = self.children["bit_generator_state"]

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


NODE_TYPE_MAPPING = {
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
