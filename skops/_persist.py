import importlib
import inspect
import io
import json
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
from sklearn.base import BaseEstimator


def gettype(state):
    if "__module__" in state and "__class__" in state:
        return getattr(importlib.import_module(state["__module__"]), state["__class__"])
    return None


def BaseEstimator_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    for key, value in obj.__dict__.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        try:
            res[key] = get_state_method(value)(value, dst)
        except TypeError:
            res[key] = json.dumps(value)

    return res


def BaseEstimator_get_instance(state, src):
    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")
    instance = cls()
    for key, value in state.items():
        if isinstance(value, dict):
            setattr(instance, key, get_instance_method(value)(value, src))
        else:
            setattr(instance, key, json.loads(value))
    return instance


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
        return np.load(io.BytesIO(src.read(state["file"])), allow_pickle=False)
    return np.array(json.loads(src.read(state["file"])))


def dict_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = {}
    for key, value in obj.items():
        if np.isscalar(key) and hasattr(key, "item"):
            # convert numpy value to python object
            key = key.item()
        try:
            content[key] = get_state_method(value)(value, dst)
        except TypeError:
            content[key] = json.dumps(value)
    res["content"] = content
    return res


def dict_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = {}
    for key, value in state["content"].items():
        if isinstance(value, dict):
            content[key] = get_instance_method(value)(value, src)
        else:
            content[key] = json.loads(value)
    return content


# A dictionary mapping types to their corresponding persistance method.
GET_STATE_METHODS = {
    BaseEstimator: BaseEstimator_get_state,
    np.ndarray: ndarray_get_state,
    dict: dict_get_state,
}

SET_STATE_METHODS = {
    BaseEstimator: BaseEstimator_get_instance,
    np.ndarray: ndarray_get_instance,
    dict: dict_get_instance,
}


def get_state_method(obj):
    # we go through the MRO and find the first class for which we have a method
    # to save the object. For instance, we might have a function for
    # BaseEstimator used for most classes, but a specialized one for Pipeline.
    for cls in type(obj).mro():
        if cls in GET_STATE_METHODS:
            return GET_STATE_METHODS[cls]

    raise TypeError(f"Can't serialize {type(obj)}")


def get_instance_method(state):
    cls = gettype(state)
    for cls in cls.mro():
        if cls in SET_STATE_METHODS:
            return SET_STATE_METHODS[cls]

    raise TypeError(f"Can't deserialize {type(state)}")


def save(file, obj):
    with tempfile.TemporaryDirectory() as dst:
        with open(Path(dst) / "schema.json", "w") as f:
            json.dump(get_state_method(obj)(obj, dst), f)

        # we use the zip format since tarfile can be exploited to create files
        # outside of the destination directory:
        # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        shutil.make_archive(file, format="zip", root_dir=dst)
        shutil.move(f"{file}.zip", file)


def load(file):
    input_zip = ZipFile(file)
    schema = input_zip.read("schema.json")
    return get_instance_method(json.loads(schema))(json.loads(schema), input_zip)
    # return {name: input_zip.read(name) for name in input_zip.namelist()}
