from __future__ import annotations

import json
from typing import Any

import numpy as np

from skops.io._audit import Node, get_tree
from skops.io._trusted_types import NUMPY_RANDOM_BIT_GENERATOR_TYPE_NAMES
from skops.io._utils import LoadContext, gettype

PROTOCOL = 1


class RandomGeneratorNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {
            "bit_generator_state": get_tree(
                state["content"]["bit_generator"], load_context, trusted=trusted
            )
        }
        self.trusted = self._get_trusted(trusted, [np.random.Generator])
        # Old-protocol reader for the same file shape as the current
        # ``RandomGeneratorNode`` in ``skops.io._numpy``, and it carries the same
        # risk: the bit generator's class name is a plain string in the file
        # that we later resolve to a ``numpy.random`` attribute and *call*. If we
        # don't declare it, the audit sees only a trusted ``str`` and a crafted
        # file could have an arbitrary ``numpy.random`` attribute called on load.
        # See that class for the full explanation; we apply the same fix here so
        # old files are read just as safely.
        self.bit_generator_name = self._get_bit_generator_name(
            state["content"]["bit_generator"]
        )
        if self.bit_generator_name is None:
            # Corrupted or tampered file: no name where we expect it, nothing to
            # build and nothing meaningful to review, so refuse it outright.
            raise ValueError(
                "Could not find the bit generator name in the "
                "numpy.random.Generator state. This is probably due to a "
                "corrupted or a malicious file."
            )

    @staticmethod
    def _get_bit_generator_name(bit_generator_state: dict[str, Any]) -> str | None:
        try:
            name_state = bit_generator_state["content"]["bit_generator"]
            name = json.loads(name_state["content"])
        except (KeyError, TypeError, ValueError):
            return None
        return name if isinstance(name, str) else None

    def get_unsafe_set(self) -> set[str]:
        res = super().get_unsafe_set()
        full_name = f"numpy.random.{self.bit_generator_name}"
        if full_name not in NUMPY_RANDOM_BIT_GENERATOR_TYPE_NAMES:
            res.add(full_name)
        return res

    def _construct(self):
        # first restore the state of the bit generator
        bit_generator_state = self.children["bit_generator_state"].construct()
        bit_generator_name = bit_generator_state["bit_generator"]
        bit_generator_cls = gettype("numpy.random", bit_generator_name)
        # As in the current node: retrieving the attribute is harmless, calling
        # it is not, so refuse to call anything that isn't a real bit generator.
        if not (
            isinstance(bit_generator_cls, type)
            and issubclass(bit_generator_cls, np.random.BitGenerator)
        ):
            raise ValueError(
                f"Expected a numpy.random bit generator, got {bit_generator_name!r}."
                " This is probably due to a corrupted or a malicious file."
            )
        bit_generator = bit_generator_cls()
        bit_generator.state = bit_generator_state

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
