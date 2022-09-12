import io
import json
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np

from ._general import function_get_instance
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
        # object arrays cannot be saved with allow_pickle=False, therefore we
        # convert them to a list and store them as a json file.
        f_name = f"{uuid4()}.json"
        with open(Path(dst) / f_name, "w") as f:
            f.write(json.dumps(obj.tolist()))
            res["type"] = "json"
            res["file"] = f_name

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
        val = np.array(json.loads(src.read(state["file"])))
    return val


# For numpy.ufunc we need to get the type from the type's module, but for other
# functions we get it from objet's module directly. Therefore sett a especial
# get_state method for them here. The load is the same as other functions.
def ufunc_get_state(obj, dst):
    if isinstance(obj, partial):
        raise TypeError("partial function are not supported yet")
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "content": obj.__name__,
    }
    return res


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_state),
    (np.ndarray, ndarray_get_state),
    (np.ufunc, ufunc_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_instance),
    (np.ndarray, ndarray_get_instance),
    (np.ufunc, function_get_instance),
]
