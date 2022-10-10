from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from ._general import function_get_instance
from ._utils import SaveState, _import_obj, get_instance, get_module, get_state
from .exceptions import UnsupportedTypeException


def ndarray_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
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
        elif save_state.path is None:
            # using skops.io.dumps, don't write to file
            f = io.BytesIO()
            np.save(f, obj)
            b = f.getvalue()
            b64 = base64.b64encode(b)
            s = b64.decode("utf-8")
            res.update(content=s, type="base64")
        else:
            # Memoize the object and then check if it's file name (containing
            # the object id) already exists. If it does, there is no need to
            # save the object again. Memoizitation is necessary since for
            # ephemeral objects, the same id might otherwise be reused.
            obj_id = save_state.memoize(obj)
            f_name = f"{obj_id}.npy"
            path = save_state.path / f_name
            if not path.exists():
                with open(path, "wb") as f:
                    np.save(f, obj, allow_pickle=False)
            res.update(type="numpy", file=f_name)
    except ValueError:
        # Couldn't save the numpy array with either method
        raise UnsupportedTypeException(
            f"numpy arrays of dtype {obj.dtype} are not supported yet, please "
            "open an issue at https://github.com/skops-dev/skops/issues and "
            "report your error"
        )

    return res


def ndarray_get_instance(state, src):
    if state["type"] == "base64":
        # TODO
        b64 = state["content"].encode("utf-8")
        b = io.BytesIO(base64.b64decode(b64))
        val = np.load(b)
        if state["__class__"] != "ndarray":
            cls = _import_obj(state["__module__"], state["__class__"])
            val = cls(val)
        return val

    # Dealing with a regular numpy array, where dtype != object
    if state["type"] == "numpy":
        val = np.load(io.BytesIO(src.read(state["file"])), allow_pickle=False)
        # Coerce type, because it may not be conserved by np.save/load. E.g. a
        # scalar will be loaded as a 0-dim array.
        if state["__class__"] != "ndarray":
            cls = _import_obj(state["__module__"], state["__class__"])
            val = cls(val)
        return val

    # We explicitly set the dtype to "O" since we only save object arrays in
    # json.
    shape = get_instance(state["shape"], src)
    tmp = [get_instance(s, src) for s in state["content"]]
    # TODO: this is a hack to get the correct shape of the array. We should
    # find _a better way_ to do this.
    if len(shape) == 1:
        val = np.ndarray(shape=len(tmp), dtype="O")
        for i, v in enumerate(tmp):
            val[i] = v
    else:
        val = np.array(tmp, dtype="O")
    return val


def maskedarray_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {
            "data": get_state(obj.data, save_state),
            "mask": get_state(obj.mask, save_state),
        },
    }
    return res


def maskedarray_get_instance(state, src):
    data = get_instance(state["content"]["data"], src)
    mask = get_instance(state["content"]["mask"], src)
    return np.ma.MaskedArray(data, mask)


def random_state_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    content = get_state(obj.get_state(legacy=False), save_state)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": content,
    }
    return res


def random_state_get_instance(state, src):
    cls = _import_obj(state["__module__"], state["__class__"])
    random_state = cls()
    content = get_instance(state["content"], src)
    random_state.set_state(content)
    return random_state


def random_generator_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    bit_generator_state = obj.bit_generator.state
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {"bit_generator": bit_generator_state},
    }
    return res


def random_generator_get_instance(state, src):
    # first restore the state of the bit generator
    bit_generator_state = state["content"]["bit_generator"]
    bit_generator = _import_obj("numpy.random", bit_generator_state["bit_generator"])()
    bit_generator.state = bit_generator_state

    # next create the generator instance
    cls = _import_obj(state["__module__"], state["__class__"])
    random_generator = cls(bit_generator=bit_generator)
    return random_generator


# For numpy.ufunc we need to get the type from the type's module, but for other
# functions we get it from objet's module directly. Therefore sett a especial
# get_state method for them here. The load is the same as other functions.
def ufunc_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,  # ufunc
        "__module__": get_module(type(obj)),  # numpy
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
        "content": ndarray_get_state(tmp, save_state),
    }
    return res


def dtype_get_instance(state, src):
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp = ndarray_get_instance(state["content"], src)
    return tmp.dtype


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
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_instance),
    (np.ndarray, ndarray_get_instance),
    (np.ma.MaskedArray, maskedarray_get_instance),
    (np.ufunc, function_get_instance),
    (np.dtype, dtype_get_instance),
    (np.random.RandomState, random_state_get_instance),
    (np.random.Generator, random_generator_get_instance),
]
