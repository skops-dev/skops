from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

from ._audit import Node, get_tree
from ._general import function_get_state
from ._protocol import PROTOCOL
from ._trusted_types import (
    NUMPY_DTYPE_TYPE_NAMES,
    NUMPY_RANDOM_BIT_GENERATOR_TYPE_NAMES,
)
from ._utils import LoadContext, SaveContext, get_module, get_state, gettype
from .exceptions import UnsupportedTypeException


def ndarray_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "NdArrayNode",
    }

    try:
        # If the dtype is object, np.save should not work with
        # allow_pickle=False, therefore we convert them to a list and
        # recursively call get_state on it.
        if obj.dtype == object:
            obj_serialized = get_state(obj.tolist(), save_context)
            res["content"] = obj_serialized["content"]
            res["type"] = "json"
            res["shape"] = get_state(obj.shape, save_context)
        else:
            data_buffer = io.BytesIO()
            np.save(data_buffer, obj, allow_pickle=False)
            # Memoize the object and then check if it's file name (containing
            # the object id) already exists. If it does, there is no need to
            # save the object again. Memoizitation is necessary since for
            # ephemeral objects, the same id might otherwise be reused.
            obj_id = save_context.memoize(obj)
            f_name = f"{obj_id}.npy"
            if f_name not in save_context.zip_file.namelist():
                save_context.zip_file.writestr(f_name, data_buffer.getbuffer())
            res.update(type="numpy", file=f_name)
    except ValueError:
        # Couldn't save the numpy array with either method
        raise UnsupportedTypeException(
            f"numpy arrays of dtype {obj.dtype} are not supported yet, please "
            "open an issue at https://github.com/skops-dev/skops/issues and "
            "report your error"
        )

    return res


class NdArrayNode(Node):
    # TODO: NdArrayNode is not only responsible for np.arrays
    #  but also for np.generics, thus the confusion with DTypeNode.
    #  See PR-336

    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.type = state["type"]
        self.trusted = self._get_trusted(trusted, [np.ndarray] + NUMPY_DTYPE_TYPE_NAMES)
        if self.type == "numpy":
            self.children = {
                "content": io.BytesIO(load_context.src.read(state["file"]))
            }
        elif self.type == "json":
            self.children = {
                "content": [
                    get_tree(o, load_context, trusted=trusted) for o in state["content"]
                ],
                "shape": get_tree(state["shape"], load_context, trusted=trusted),
            }
        else:
            raise ValueError(f"Unknown type {self.type}.")

    def _construct(self):
        # Dealing with a regular numpy array, where dtype != object
        if self.type == "numpy":
            content = np.load(self.children["content"], allow_pickle=False)
            if f"{self.module_name}.{self.class_name}" != "numpy.ndarray":
                content = gettype(self.module_name, self.class_name)(content)
            return content

        if self.type == "json":
            # We explicitly set the dtype to "O" since we only save object
            # arrays in json.
            shape = self.children["shape"].construct()
            tmp = [o.construct() for o in self.children["content"]]

            # TODO: this is a hack to get the correct shape of the array. We
            # should find _a better way_ to do this.
            if len(shape) == 1:
                content = np.ndarray(shape=len(tmp), dtype="O")
                for i, v in enumerate(tmp):
                    content[i] = v
            else:
                content = np.array(tmp, dtype="O")

            return content

        raise ValueError(f"Unknown type for a numpy object: {self.type}.")


def maskedarray_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "MaskedArrayNode",
        "content": {
            "data": get_state(obj.data, save_context),
            "mask": get_state(obj.mask, save_context),
        },
    }
    return res


class MaskedArrayNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [np.ma.MaskedArray])
        self.children = {
            "data": get_tree(state["content"]["data"], load_context, trusted=trusted),
            "mask": get_tree(state["content"]["mask"], load_context, trusted=trusted),
        }

    def _construct(self):
        data = self.children["data"].construct()
        mask = self.children["mask"].construct()
        return np.ma.MaskedArray(data, mask)


def random_state_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = get_state(obj.get_state(legacy=False), save_context)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "RandomStateNode",
        "content": content,
    }
    return res


class RandomStateNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO
        self.children = {
            "content": get_tree(state["content"], load_context, trusted=trusted)
        }
        self.trusted = self._get_trusted(trusted, [np.random.RandomState])

    def _construct(self):
        random_state = gettype(self.module_name, self.class_name)()
        random_state.set_state(self.children["content"].construct())
        return random_state


def random_generator_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    bit_generator_state = get_state(obj.bit_generator.state, save_context)
    seed_seq_state = get_state(obj.bit_generator.seed_seq.state, save_context)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "RandomGeneratorNode",
        "content": {"bit_generator": bit_generator_state, "seed_seq": seed_seq_state},
    }
    return res


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
            ),
            "seed_seq_state": get_tree(
                state["content"]["seed_seq"], load_context, trusted=trusted
            ),
        }
        self.trusted = self._get_trusted(trusted, [np.random.Generator])
        # Security note. To rebuild a Generator we need its bit generator, and
        # numpy identifies that by class name: the name (e.g. "PCG64") is stored
        # by numpy as a plain string inside ``bit_generator.state``. At load
        # time ``_construct`` turns that string back into a class via
        # ``getattr(numpy.random, <name>)`` and *calls* it.
        #
        # skops' safety check works by having every node declare, through
        # ``get_unsafe_set``, which types it will materialise; the user reviews
        # that list (``get_untrusted_types``) before trusting a file. The
        # problem is that this class name is not declared anywhere: it rides
        # along as an ordinary ``str`` inside a ``dict``, both trusted by
        # default, so the audit never sees it. A hand-crafted file can put *any*
        # ``numpy.random`` attribute name there (``seed``, ``set_bit_generator``,
        # ...) and have it called on load, while ``get_untrusted_types`` still
        # reports the file as containing nothing untrusted.
        #
        # We close that gap by pulling the name out here and declaring it in
        # ``get_unsafe_set`` as ``numpy.random.<name>``, so it goes through the
        # same trust review as every other type we load.
        self.bit_generator_name = self._get_bit_generator_name(
            state["content"]["bit_generator"]
        )
        if self.bit_generator_name is None:
            # A genuine file always stores the name where we expect it, so not
            # finding it means the file is corrupted or has been tampered with.
            # We can't build the object and there is no meaningful type to put
            # in front of the user, so we refuse the file outright rather than
            # reporting a fake untrusted type. This mirrors how ``MethodNode``
            # and ``OperatorFuncNode`` reject a malformed state.
            raise ValueError(
                "Could not find the bit generator name in the "
                "numpy.random.Generator state. This is probably due to a "
                "corrupted or a malicious file."
            )

    @staticmethod
    def _get_bit_generator_name(bit_generator_state: dict[str, Any]) -> str | None:
        # Read the class name out of the raw (not-yet-constructed) state. numpy
        # stores it under the ``bit_generator`` key of the state dict, which we
        # serialise as a ``DictNode`` whose values are ``JsonNode``s -- hence
        # the ``["content"]["bit_generator"]["content"]`` path and ``json.loads``
        # of the stored literal. Return ``None`` if the file doesn't have that
        # exact shape; the caller turns that into an error.
        try:
            name_state = bit_generator_state["content"]["bit_generator"]
            name = json.loads(name_state["content"])
        except (KeyError, TypeError, ValueError):
            return None
        return name if isinstance(name, str) else None

    def get_unsafe_set(self) -> set[str]:
        res = super().get_unsafe_set()
        # Declare the bit generator as a type to be reviewed. Real bit
        # generators are trusted by default, so genuine files stay clean;
        # anything else shows up in ``get_untrusted_types``.
        full_name = f"numpy.random.{self.bit_generator_name}"
        if full_name not in NUMPY_RANDOM_BIT_GENERATOR_TYPE_NAMES:
            res.add(full_name)
        return res

    def _construct(self):
        # first restore the state of the bit generator
        seed_seq_cls = gettype(
            "numpy.random.bit_generator",
            "SeedSequence",
        )
        seed_seq_state = self.children["seed_seq_state"].construct()
        seed_seq = seed_seq_cls(**seed_seq_state)

        bit_generator_state = self.children["bit_generator_state"].construct()
        bit_generator_name = bit_generator_state["bit_generator"]
        bit_generator_cls = gettype("numpy.random", bit_generator_name)
        # Second line of defence, independent of the audit above. ``gettype`` is
        # just ``getattr(numpy.random, name)``; retrieving the attribute is
        # harmless, but the next line *calls* it, and that is the only dangerous
        # step. So even if the audit was satisfied (for instance the user
        # explicitly trusted the name) we refuse to call anything that isn't an
        # actual bit generator, rather than invoking an arbitrary callable.
        if not (
            isinstance(bit_generator_cls, type)
            and issubclass(bit_generator_cls, np.random.BitGenerator)
        ):
            raise ValueError(
                f"Expected a numpy.random bit generator, got {bit_generator_name!r}."
                " This is probably due to a corrupted or a malicious file."
            )
        bit_generator = bit_generator_cls(seed_seq)
        bit_generator.state = bit_generator_state

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


def dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp: np.typing.NDArray = np.ndarray(0, dtype=obj)
    res = {
        "__class__": "dtype",
        "__module__": "numpy",
        "__loader__": "DTypeNode",
        "content": get_state(tmp, save_context),
    }
    return res


class DTypeNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {
            "content": get_tree(state["content"], load_context, trusted=trusted)
        }
        # TODO: what should we trust?
        self.trusted = self._get_trusted(trusted, [])

    def _construct(self):
        # we use numpy's internal save mechanism to store the dtype by
        # saving/loading an empty array with that dtype.
        return self.children["content"].construct().dtype


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_state),
    (np.ndarray, ndarray_get_state),
    (np.ma.MaskedArray, maskedarray_get_state),
    (np.ufunc, function_get_state),
    (np.dtype, dtype_get_state),
    (np.random.RandomState, random_state_get_state),
    (np.random.Generator, random_generator_get_state),
]

try:
    # From numpy=1.25.0 dispatching for `__array_function__` is done via
    # a C wrapper: https://github.com/numpy/numpy/pull/23020
    try:
        # numpy>=2
        from numpy._core._multiarray_umath import (
            _ArrayFunctionDispatcher,
        )
    except ImportError:
        from numpy.core._multiarray_umath import (
            _ArrayFunctionDispatcher,
        )

    GET_STATE_DISPATCH_FUNCTIONS.append((_ArrayFunctionDispatcher, function_get_state))
except ImportError:
    pass


# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    ("NdArrayNode", PROTOCOL): NdArrayNode,
    ("MaskedArrayNode", PROTOCOL): MaskedArrayNode,
    ("DTypeNode", PROTOCOL): DTypeNode,
    ("RandomStateNode", PROTOCOL): RandomStateNode,
    ("RandomGeneratorNode", PROTOCOL): RandomGeneratorNode,
}
