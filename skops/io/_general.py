import inspect
import json
from functools import partial
from types import FunctionType

import numpy as np

from ._utils import _import_obj, get_instance, get_state, gettype


@get_state.register(dict)
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
            content[key] = get_state(value, dst)
        except TypeError:
            content[key] = json.dumps(value)
    res["content"] = content
    return res


@get_instance.register(dict)
def dict_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = {}
    for key, value in state["content"].items():
        if isinstance(value, dict):
            content[key] = get_instance(value, src)
        else:
            content[key] = json.loads(value)
    return content


@get_state.register(list)
def list_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = []
    for value in obj:
        try:
            content.append(get_state(value, dst))
        except TypeError:
            content.append(json.dumps(value))
    res["content"] = content
    return res


@get_instance.register(list)
def list_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = []
    for value in state["content"]:
        if gettype(value):
            content.append(get_instance(value, src))
        else:
            content.append(json.loads(value))
    return content


@get_state.register(tuple)
def tuple_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = ()
    for value in obj:
        try:
            content += (get_state(value, dst),)
        except TypeError:
            content += (json.dumps(value),)
    res["content"] = content
    return res


@get_instance.register(tuple)
def tuple_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = ()
    for value in state["content"]:
        if gettype(value):
            content += (get_instance(value, src),)
        else:
            content += (json.loads(value),)
    return content


@get_state.register(FunctionType)
def function_get_state(obj, dst):
    if isinstance(obj, partial):
        raise TypeError("partial function are not supported yet")
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(obj).__name__,
        "content": obj.__name__,
    }
    return res


@get_instance.register(np.ufunc)
@get_instance.register(FunctionType)
def function_get_instance(obj, src):
    loaded = _import_obj(obj["__module__"], obj["content"])
    return loaded


@get_state.register(type)
def type_get_state(obj, dst):
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
        "content": {
            "__class__": obj.__name__,
            "__module__": inspect.getmodule(obj).__name__,
        },
    }
    return res


@get_instance.register(type)
def type_get_instance(obj, src):
    loaded = _import_obj(obj["content"]["__module__"], obj["content"]["__class__"])
    return loaded
