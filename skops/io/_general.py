import inspect
import json
from functools import partial
from types import FunctionType

import numpy as np

from ._utils import _import_obj, gettype


def dict_get_state(obj, dst):
    from ._persist import get_state_method

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
    from ._persist import get_instance_method

    state.pop("__class__")
    state.pop("__module__")
    content = {}
    for key, value in state["content"].items():
        if isinstance(value, dict):
            content[key] = get_instance_method(value)(value, src)
        else:
            content[key] = json.loads(value)
    return content


def list_get_state(obj, dst):
    from ._persist import get_state_method

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = []
    for value in obj:
        try:
            content.append(get_state_method(value)(value, dst))
        except TypeError:
            content.append(json.dumps(value))
    res["content"] = content
    return res


def list_get_instance(state, src):
    from ._persist import get_instance_method

    state.pop("__class__")
    state.pop("__module__")
    content = []
    for value in state["content"]:
        if gettype(value):
            content.append(get_instance_method(value)(value, src))
        else:
            content.append(json.loads(value))
    return content


def tuple_get_state(obj, dst):
    from ._persist import get_state_method

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = ()
    for value in obj:
        try:
            content += (get_state_method(value)(value, dst),)
        except TypeError:
            content += (json.dumps(value),)
    res["content"] = content
    return res


def tuple_get_instance(state, src):
    from ._persist import get_instance_method

    state.pop("__class__")
    state.pop("__module__")
    content = ()
    for value in state["content"]:
        if gettype(value):
            content += (get_instance_method(value)(value, src),)
        else:
            content += (json.loads(value),)
    return content


def function_get_state(obj, dst):
    if isinstance(obj, partial):
        raise TypeError("partial function are not supported yet")
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
        "__content__": obj.__name__,
    }
    return res


def function_get_instance(obj, src):
    loaded = _import_obj(obj["__module__"], obj["__content__"])
    return loaded


def get_state_methods():
    return {
        FunctionType: function_get_state,
        dict: dict_get_state,
        list: list_get_state,
        tuple: tuple_get_state,
    }


def get_instance_methods():
    return {
        FunctionType: function_get_instance,
        dict: dict_get_instance,
        list: list_get_instance,
        tuple: tuple_get_instance,
    }
