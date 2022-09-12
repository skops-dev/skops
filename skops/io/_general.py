import inspect
from functools import partial
from types import FunctionType

import numpy as np

from ._utils import (
    _import_obj,
    get_instance,
    get_state,
    try_get_instance,
    try_get_state,
)


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
        content[key] = try_get_state(value, dst)
    res["content"] = content
    return res


def dict_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = {}
    for key, value in state["content"].items():
        content[key] = try_get_instance(value, src)
    return content


def list_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = []
    for value in obj:
        content.append(try_get_state(value, dst))
    res["content"] = content
    return res


def list_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = []
    for value in state["content"]:
        content.append(try_get_instance(value, src))
    return content


def tuple_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = ()
    for value in obj:
        content += (try_get_state(value, dst),)
    res["content"] = content
    return res


def tuple_get_instance(state, src):
    state.pop("__class__")
    state.pop("__module__")
    content = ()
    for value in state["content"]:
        content += (try_get_instance(value, src),)
    return content


def function_get_state(obj, dst):
    if isinstance(obj, partial):
        raise TypeError("partial function are not supported yet")
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(obj).__name__,
        "content": obj.__name__,
    }
    return res


def function_get_instance(obj, src):
    loaded = _import_obj(obj["__module__"], obj["content"])
    return loaded


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


def type_get_instance(obj, src):
    loaded = _import_obj(obj["content"]["__module__"], obj["content"]["__class__"])
    return loaded


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (list, list_get_state),
    (tuple, tuple_get_state),
    (FunctionType, function_get_state),
    (type, type_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_instance),
    (list, list_get_instance),
    (tuple, tuple_get_instance),
    (FunctionType, function_get_instance),
    (type, type_get_instance),
]
