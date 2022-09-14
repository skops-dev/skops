import io
from pathlib import Path
from uuid import uuid4

import numpy as np

from ._general import function_get_instance
from ._persist import get_instance, get_state
from ._utils import _import_obj, get_module


def ndarray_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    try:
        f_name = f"{uuid4()}.npy"
        with open(Path(dst) / f_name, "wb") as f:
            np.save(f, obj, allow_pickle=False)
            res["type"] = "numpy"
            res["file"] = f_name
    except ValueError:
        # Object arrays cannot be saved with allow_pickle=False, therefore we
        # convert them to a list and recursively call get_state on it.
        if obj.dtype == object:
            obj_serialized = get_state(obj.tolist(), dst)
            res["content"] = obj_serialized["content"]
            res["type"] = "json"
            res["shape"] = get_state(obj.shape, dst)
        else:
            raise TypeError(f"numpy arrays of dtype {obj.dtype} are not supported yet")

    return res


def ndarray_get_instance(state, src):
    if state["type"] == "numpy":
        val = np.load(io.BytesIO(src.read(state["file"])), allow_pickle=False)
        # Coerce type, because it may not be conserved by np.save/load. E.g. a
        # scalar will be loaded as a 0-dim array.
        if state["__class__"] != "ndarray":
            cls = _import_obj(state["__module__"], state["__class__"])
            val = cls(val)
    else:
        # We explicitly set the dtype to "O" since we only save object arrays
        # in json.
        shape = get_instance(state["shape"], src)
        tmp = [get_instance(s, src) for s in state["content"]]
        # TODO: this is a hack to get the correct shape of the array. We should
        # find a better way to do this.
        if len(shape) == 1:
            val = np.ndarray(shape=len(tmp), dtype="O")
            for i, v in enumerate(tmp):
                val[i] = v
        else:
            val = np.array(tmp, dtype="O")
    return val


def random_state_get_state(obj, dst):
    content = get_state(obj.get_state(legacy=False), dst)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "type": "numpy",
        "content": content,
    }
    return res


def random_state_get_instance(state, src):
    cls = _import_obj(state["__module__"], state["__class__"])
    random_state = cls()
    content = get_instance(state["content"], src)
    random_state.set_state(content)
    return random_state


def random_generator_get_state(obj, dst):
    bit_generator_state = obj.bit_generator.state
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "type": "numpy",
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
def ufunc_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,  # ufunc
        "__module__": get_module(type(obj)),  # numpy
        "content": {
            "module_path": get_module(obj),
            "function": obj.__name__,
        },
    }
    return res


def dtype_get_state(obj, dst):
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp = np.ndarray(0, dtype=obj)
    res = {
        "__class__": "dtype",
        "__module__": "numpy",
        "content": ndarray_get_state(tmp, dst),
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
    (np.ufunc, ufunc_get_state),
    (np.dtype, dtype_get_state),
    (np.random.RandomState, random_state_get_state),
    (np.random.Generator, random_generator_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_instance),
    (np.ndarray, ndarray_get_instance),
    (np.ufunc, function_get_instance),
    (np.dtype, dtype_get_instance),
    (np.random.RandomState, random_state_get_instance),
    (np.random.Generator, random_generator_get_instance),
]
