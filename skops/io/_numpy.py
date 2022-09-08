import inspect
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ._utils import _import_obj


def ndarray_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
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


def get_state_methods():
    from ._general import function_get_state

    return {
        np.ufunc: function_get_state,
        np.ndarray: ndarray_get_state,
        np.generic: ndarray_get_state,
    }


def get_instance_methods():
    from ._general import function_get_instance

    return {
        np.ufunc: function_get_instance,
        np.ndarray: ndarray_get_instance,
        np.generic: ndarray_get_instance,
    }
