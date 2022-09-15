from functools import partial
from types import FunctionType

import numpy as np

from ._utils import _get_instance, _get_state, _import_obj, get_module, gettype


def dict_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    key_types = _get_state([type(key) for key in obj.keys()], dst)
    content = {}
    for key, value in obj.items():
        if np.isscalar(key) and hasattr(key, "item"):
            # convert numpy value to python object
            key = key.item()
        content[key] = _get_state(value, dst)
    res["content"] = content
    res["key_types"] = key_types
    return res


def dict_get_instance(state, src):
    content = gettype(state)()
    key_types = _get_instance(state["key_types"], src)
    for k_type, item in zip(key_types, state["content"].items()):
        content[k_type(item[0])] = _get_instance(item[1], src)
    return content


def list_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }
    content = []
    for value in obj:
        content.append(_get_state(value, dst))
    res["content"] = content
    return res


def list_get_instance(state, src):
    content = gettype(state)()
    for value in state["content"]:
        content.append(_get_instance(value, src))
    return content


def tuple_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }
    content = ()
    for value in obj:
        content += (_get_state(value, dst),)
    res["content"] = content
    return res


def tuple_get_instance(state, src):
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

    content = tuple()
    for value in state["content"]:
        content += (_get_instance(value, src),)

    if isnamedtuple(cls):
        return cls(*content)
    return content


def function_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "content": {
            "module_path": get_module(obj),
            "function": obj.__name__,
        },
    }
    return res


def function_get_instance(obj, src):
    loaded = _import_obj(obj["content"]["module_path"], obj["content"]["function"])
    return loaded


def partial_get_state(obj, dst):
    _, _, (func, args, kwds, namespace) = obj.__reduce__()
    res = {
        "__class__": "partial",  # don't allow any subclass
        "__module__": get_module(type(obj)),
        "content": {
            "func": _get_state(func, dst),
            "args": _get_state(args, dst),
            "kwds": _get_state(kwds, dst),
            "namespace": _get_state(namespace, dst),
        },
    }
    return res


def partial_get_instance(obj, src):
    content = obj["content"]
    func = _get_instance(content["func"], src)
    args = _get_instance(content["args"], src)
    kwds = _get_instance(content["kwds"], src)
    namespace = _get_instance(content["namespace"], src)
    instance = partial(func, *args, **kwds)  # always use partial, not a subclass
    instance.__setstate__((func, args, kwds, namespace))
    return instance


def type_get_state(obj, dst):
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {
            "__class__": obj.__name__,
            "__module__": get_module(obj),
        },
    }
    return res


def type_get_instance(obj, src):
    loaded = _import_obj(obj["content"]["__module__"], obj["content"]["__class__"])
    return loaded


def slice_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        },
    }
    return res


def slice_get_instance(obj, src):
    start = obj["content"]["start"]
    stop = obj["content"]["stop"]
    step = obj["content"]["step"]
    return slice(start, stop, step)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (list, list_get_state),
    (tuple, tuple_get_state),
    (slice, slice_get_state),
    (FunctionType, function_get_state),
    (partial, partial_get_state),
    (type, type_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_instance),
    (list, list_get_instance),
    (tuple, tuple_get_instance),
    (slice, slice_get_instance),
    (FunctionType, function_get_instance),
    (partial, partial_get_instance),
    (type, type_get_instance),
]
