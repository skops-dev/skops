from __future__ import annotations

import io
from typing import Any, Optional, Sequence

import numpy as np

from ._audit import Node, get_tree
from ._general import function_get_state
from ._protocol import PROTOCOL
from ._trusted_types import NUMPY_DTYPE_TYPE_NAMES
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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.type = state["type"]
        self.trusted = self._get_trusted(
            trusted, [np.ndarray] + NUMPY_DTYPE_TYPE_NAMES  # type: ignore
        )
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
        trusted: Optional[Sequence[str]] = None,
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
        trusted: Optional[Sequence[str]] = None,
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
        trusted: Optional[Sequence[str]] = None,
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

    def _construct(self):
        # first restore the state of the bit generator
        seed_seq_cls = gettype(
            "numpy.random.bit_generator",
            "SeedSequence",
        )
        seed_seq_state = self.children["seed_seq_state"].construct()
        seed_seq = seed_seq_cls(**seed_seq_state)

        bit_generator_state = self.children["bit_generator_state"].construct()
        bit_generator_cls = gettype(
            "numpy.random", bit_generator_state["bit_generator"]
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
        trusted: Optional[Sequence[str]] = None,
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
        from numpy._core._multiarray_umath import (  # type: ignore
            _ArrayFunctionDispatcher,
        )
    except ImportError:
        from numpy.core._multiarray_umath import (  # type: ignore
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
