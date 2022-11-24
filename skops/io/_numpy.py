from __future__ import annotations

import io
from typing import Any

import numpy as np

from ._dispatch import Node, get_tree
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
    def __init__(self, state, load_context: LoadContext, trusted=False):
        super().__init__(state, load_context, trusted)
        self.type = state["type"]
        self.trusted = self._get_trusted(trusted, ["numpy.ndarray"])
        if self.type == "numpy":
            self.children = {
                "content": io.BytesIO(load_context.src.read(state["file"]))
            }
        elif self.type == "json":
            self.children = {
                "content": [  # type: ignore
                    get_tree(o, load_context) for o in state["content"]  # type: ignore
                ],
                "shape": get_tree(state["shape"], load_context),
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
    def __init__(self, state, load_context: LoadContext, trusted=False):
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, ["numpy.ma.MaskedArray"])
        self.children = {
            "data": get_tree(state["content"]["data"], load_context),
            "mask": get_tree(state["content"]["mask"], load_context),
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
    def __init__(self, state, load_context: LoadContext, trusted=False):
        super().__init__(state, load_context, trusted)
        self.children = {"content": get_tree(state["content"], load_context)}
        self.trusted = self._get_trusted(trusted, ["numpy.random.RandomState"])

    def _construct(self):
        random_state = gettype(self.module_name, self.class_name)()
        random_state.set_state(self.children["content"].construct())
        return random_state


def random_generator_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    bit_generator_state = obj.bit_generator.state
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "RandomGeneratorNode",
        "content": {"bit_generator": bit_generator_state},
    }
    return res


class RandomGeneratorNode(Node):
    def __init__(self, state, load_context: LoadContext, trusted=False):
        super().__init__(state, load_context, trusted)
        self.children = {"bit_generator_state": state["content"]["bit_generator"]}
        self.trusted = self._get_trusted(trusted, ["numpy.random.Generator"])

    def _construct(self):
        # first restore the state of the bit generator
        bit_generator = gettype(
            "numpy.random", self.children["bit_generator_state"]["bit_generator"]
        )()
        bit_generator.state = self.children["bit_generator_state"]

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


# For numpy.ufunc we need to get the type from the type's module, but for other
# functions we get it from objet's module directly. Therefore sett a especial
# get_state method for them here. The load is the same as other functions.
def ufunc_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,  # ufunc
        "__module__": get_module(type(obj)),  # numpy
        "__loader__": "FunctionNode",
        "content": {
            "module_path": get_module(obj),
            "function": obj.__name__,
        },
    }
    return res


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
    def __init__(self, state, load_context: LoadContext, trusted=False):
        super().__init__(state, load_context, trusted)
        self.children = {"content": get_tree(state["content"], load_context)}
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
    (np.ufunc, ufunc_get_state),
    (np.dtype, dtype_get_state),
    (np.random.RandomState, random_state_get_state),
    (np.random.Generator, random_generator_get_state),
]
# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    "NdArrayNode": NdArrayNode,
    "MaskedArrayNode": MaskedArrayNode,
    "DTypeNode": DTypeNode,
    "RandomStateNode": RandomStateNode,
    "RandomGeneratorNode": RandomGeneratorNode,
}
