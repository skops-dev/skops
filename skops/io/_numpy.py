from __future__ import annotations

import io
from operator import getitem
from typing import Any

import numpy as np

from ._audit import Node
from ._dispatch import get_tree
from ._general import function_get_instance
from ._utils import SaveState, _import_obj, get_module, get_state, gettype
from .exceptions import UnsupportedTypeException


def ndarray_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
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
            obj_serialized = get_state(obj.tolist(), save_state)
            res["content"] = obj_serialized["content"]
            res["type"] = "json"
            res["shape"] = get_state(obj.shape, save_state)
        else:
            data_buffer = io.BytesIO()
            np.save(data_buffer, obj)
            # Memoize the object and then check if it's file name (containing
            # the object id) already exists. If it does, there is no need to
            # save the object again. Memoizitation is necessary since for
            # ephemeral objects, the same id might otherwise be reused.
            obj_id = save_state.memoize(obj)
            f_name = f"{obj_id}.npy"
            if f_name not in save_state.zip_file.namelist():
                save_state.zip_file.writestr(f_name, data_buffer.getbuffer())
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
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.type = state["type"]
        if self.type == "numpy":
            self.file = state["file"]
            self.children = {}
        elif self.type == "json":
            self.shape = get_tree(state["shape"], src, trusted)
            self.content = [get_tree(o, src, trusted) for o in state["content"]]
            self.children = {"shape": Node, "content": list}
        else:
            raise ValueError(f"Unknown type {self.type}.")

    def construct(self):
        # Dealing with a regular numpy array, where dtype != object
        if self.type == "numpy":
            content = np.load(self.file)
            if f"{self.module_name}.{self.class_name}" != "numpy.ndarray":
                content = _import_obj(self.module_name, self.class_name)(content)
            return content

        elif self.type == "json":
            # We explicitly set the dtype to "O" since we only save object
            # arrays in json.
            shape = self.shape.construct()
            tmp = [o.construct() for o in self.content]

            # TODO: this is a hack to get the correct shape of the array. We
            # should find _a better way_ to do this.
            if len(shape) == 1:
                content = np.ndarray(shape=len(tmp), dtype="O")
                for i, v in enumerate(tmp):
                    content[i] = v
            else:
                content = np.array(tmp, dtype="O")

            return content
            return np.array(self.content, dtype=object).reshape(self.shape)


def maskedarray_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "MaskedArrayNode",
        "content": {
            "data": get_state(obj.data, save_state),
            "mask": get_state(obj.mask, save_state),
        },
    }
    return res


class MaskedArrayNode(Node):
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.data = get_tree(state["content"]["data"], src, trusted)
        self.mask = get_tree(state["content"]["mask"], src, trusted)
        self.children = {"data": Node, "mask": Node}

    def construct(self):
        data = self.content["data"].construct()
        mask = self.content["mask"].construct()
        return np.ma.MaskedArray(data, mask)


def random_state_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    content = get_state(obj.get_state(legacy=False), save_state)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "RandomStateNode",
        "content": content,
    }
    return res


class RandomStateNode(Node):
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.content = get_tree(state["content"], src, trusted)
        self.children = {"content": Node}

    def construct(self):
        random_state = gettype(self.module_name, self.class_name)()
        random_state.set_state(self.content.construct())
        return random_state


def random_generator_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    bit_generator_state = obj.bit_generator.state
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "RandomGeneratorNode",
        "content": {"bit_generator": bit_generator_state},
    }
    return res


class RandomGeneratorNode(Node):
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.content = get_tree(state["content"], src, trusted)
        self.children = {"content": Node}

    def construct(self):
        # first restore the state of the bit generator
        bit_generator_state = self.content.construct()
        bit_generator = gettype("numpy.random", bit_generator_state["bit_generator"])()
        bit_generator.state = bit_generator_state

        # next create the generator instance
        return gettype(self.module_name, self.class_name)(bit_generator=bit_generator)


# For numpy.ufunc we need to get the type from the type's module, but for other
# functions we get it from objet's module directly. Therefore sett a especial
# get_state method for them here. The load is the same as other functions.
def ufunc_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
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


def dtype_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp: np.typing.NDArray = np.ndarray(0, dtype=obj)
    res = {
        "__class__": "dtype",
        "__module__": "numpy",
        "__loader__": "dtype_get_instance",
        "content": ndarray_get_state(tmp, save_state),
    }
    return res


class DTypeNode(Node):
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.content = get_tree(state["content"], src, trusted)
        self.children = {"content": Node}

    def construct(self):
        # we use numpy's internal save mechanism to store the dtype by
        # saving/loading an empty array with that dtype.
        return self.content.construct().dtype


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
GET_INSTANCE_DISPATCH_MAPPING = {
    "ndarray_get_instance": ndarray_get_instance,
    "maskedarray_get_instance": maskedarray_get_instance,
    "function_get_instance": function_get_instance,
    "dtype_get_instance": dtype_get_instance,
    "random_state_get_instance": random_state_get_instance,
    "random_generator_get_instance": random_generator_get_instance,
}
