from __future__ import annotations

from typing import Any

import numpy as np

from skops.io._audit import Node
from skops.io._utils import LoadContext, TrustedTypes, gettype

PROTOCOL = 0


class RandomGeneratorNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # protocol 0 stored the bit generator state as a plain dict
        self.bit_generator_state = state["content"]["bit_generator"]
        self.children = {"bit_generator_state": self.bit_generator_state}
        self.trusted = self._get_trusted(trusted, [np.random.Generator])

    def _construct(self):
        # NOTE: this reads a class name from the file and calls the matching
        # numpy.random attribute, which would be the same audit-bypass fixed in
        # the current ``RandomGeneratorNode`` (skops.io._numpy). It is safe here
        # only because a protocol-0 Generator can never reach construction: its
        # bit generator state is kept as a raw dict, so ``get_unsafe_set`` raises
        # while auditing it (see ``test_random_generator_v0``) and the object is
        # never built. Protocol 0 is effectively unloadable and kept only for
        # completeness; there is nothing to harden on a path that never runs.
        bit_generator = gettype(
            "numpy.random", self.bit_generator_state["bit_generator"]
        )()
        bit_generator.state = self.bit_generator_state

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


NODE_TYPE_MAPPING = {
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
