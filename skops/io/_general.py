from __future__ import annotations

import json
from functools import partial
from types import FunctionType, MethodType
from typing import Any

import numpy as np

from ._dispatch import get_instance
from ._utils import (
    LoadContext,
    SaveContext,
    _import_obj,
    get_module,
    get_state,
    gettype,
)
from .exceptions import UnsupportedTypeException


def dict_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "dict_get_instance",
    }

    key_types = get_state([type(key) for key in obj.keys()], save_context)
    content = {}
    for key, value in obj.items():
        if isinstance(value, property):
            continue
        if np.isscalar(key) and hasattr(key, "item"):
            # convert numpy value to python object
            key = key.item()  # type: ignore
        content[key] = get_state(value, save_context)
    res["content"] = content
    res["key_types"] = key_types
    return res


def dict_get_instance(state, load_context: LoadContext):
    content = gettype(state)()
    key_types = get_instance(state["key_types"], load_context)
    for k_type, item in zip(key_types, state["content"].items()):
        content[k_type(item[0])] = get_instance(item[1], load_context)
    return content


def list_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "list_get_instance",
    }
    content = []
    for value in obj:
        content.append(get_state(value, save_context))
    res["content"] = content
    return res


def list_get_instance(state, load_context: LoadContext):
    content = gettype(state)()
    for value in state["content"]:
        content.append(get_instance(value, load_context))
    return content


def tuple_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "tuple_get_instance",
    }
    content = tuple(get_state(value, save_context) for value in obj)
    res["content"] = content
    return res


def tuple_get_instance(state, load_context: LoadContext):
    # Returns a tuple or a namedtuple instance.
    def isnamedtuple(t):
        # This is needed since namedtuples need to have the args when
        # initialized.
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple:
            return False
        f = getattr(t, "_fields", None)
        if not isinstance(f, tuple):
            return False
        return all(type(n) == str for n in f)

    cls = gettype(state)
    content = tuple(get_instance(value, load_context) for value in state["content"])

    if isnamedtuple(cls):
        return cls(*content)
    return content


def function_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "__loader__": "function_get_instance",
        "content": {
            "module_path": get_module(obj),
            "function": obj.__name__,
        },
    }
    return res


def function_get_instance(state, load_context: LoadContext):
    loaded = _import_obj(state["content"]["module_path"], state["content"]["function"])
    return loaded


def partial_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    _, _, (func, args, kwds, namespace) = obj.__reduce__()
    res = {
        "__class__": "partial",  # don't allow any subclass
        "__module__": get_module(type(obj)),
        "__loader__": "partial_get_instance",
        "content": {
            "func": get_state(func, save_context),
            "args": get_state(args, save_context),
            "kwds": get_state(kwds, save_context),
            "namespace": get_state(namespace, save_context),
        },
    }
    return res


def partial_get_instance(state, load_context: LoadContext):
    content = state["content"]
    func = get_instance(content["func"], load_context)
    args = get_instance(content["args"], load_context)
    kwds = get_instance(content["kwds"], load_context)
    namespace = get_instance(content["namespace"], load_context)
    instance = partial(func, *args, **kwds)  # always use partial, not a subclass
    instance.__setstate__((func, args, kwds, namespace))  # type: ignore
    return instance


def type_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "type_get_instance",
        "content": {
            "__class__": obj.__name__,
            "__module__": get_module(obj),
        },
    }
    return res


def type_get_instance(state, load_context: LoadContext):
    loaded = _import_obj(state["content"]["__module__"], state["content"]["__class__"])
    return loaded


def slice_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "slice_get_instance",
        "content": {
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        },
    }
    return res


def slice_get_instance(state, load_context: LoadContext):
    start = state["content"]["start"]
    stop = state["content"]["stop"]
    step = state["content"]["step"]
    return slice(start, stop, step)


def object_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # This method is for objects which can either be persisted with json, or
    # the ones for which we can get/set attributes through
    # __getstate__/__setstate__ or reading/writing to __dict__.
    try:
        # if we can simply use json, then we're done.
        obj_str = json.dumps(obj)
        return {
            "__class__": "str",
            "__module__": "builtins",
            "__loader__": "none",
            "content": obj_str,
            "is_json": True,
        }
    except Exception:
        pass

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "object_get_instance",
    }

    # __getstate__ takes priority over __dict__, and if non exist, we only save
    # the type of the object, and loading would mean instantiating the object.
    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return res

    content = get_state(attrs, save_context)
    # it's sufficient to store the "content" because we know that this dict can
    # only have str type keys
    res["content"] = content
    return res


def object_get_instance(state, load_context: LoadContext):
    if state.get("is_json", False):
        return json.loads(state["content"])

    cls = gettype(state)

    # Instead of simply constructing the instance, we use __new__, which
    # bypasses the __init__, and then we set the attributes. This solves
    # the issue of required init arguments.
    instance = cls.__new__(cls)

    content = state.get("content")
    if not content:  # nothing more to do
        return instance

    attrs = get_instance(content, load_context)
    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def method_get_state(obj: Any, save_context: SaveContext):
    # This method is used to persist bound methods, which are
    # dependent on a specific instance of an object.
    # It stores the state of the object the method is bound to,
    # and prepares both to be persisted.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "__loader__": "method_get_instance",
        "content": {
            "func": obj.__func__.__name__,
            "obj": get_state(obj.__self__, save_context),
        },
    }
    return res


def method_get_instance(state, load_context: LoadContext):
    loaded_obj = get_instance(state["content"]["obj"], load_context)
    method = getattr(loaded_obj, state["content"]["func"])
    return method


def unsupported_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    raise UnsupportedTypeException(obj)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (list, list_get_state),
    (tuple, tuple_get_state),
    (slice, slice_get_state),
    (FunctionType, function_get_state),
    (MethodType, method_get_state),
    (partial, partial_get_state),
    (type, type_get_state),
    (object, object_get_state),
]

GET_INSTANCE_DISPATCH_MAPPING = {
    "dict_get_instance": dict_get_instance,
    "list_get_instance": list_get_instance,
    "tuple_get_instance": tuple_get_instance,
    "slice_get_instance": slice_get_instance,
    "function_get_instance": function_get_instance,
    "method_get_instance": method_get_instance,
    "partial_get_instance": partial_get_instance,
    "type_get_instance": type_get_instance,
    "object_get_instance": object_get_instance,
}
