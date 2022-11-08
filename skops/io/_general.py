from __future__ import annotations

import json
from functools import partial
from types import FunctionType, MethodType
from typing import Any

import numpy as np

from ._dispatch import Node, get_tree
from ._trusted_types import PRIMITIVE_TYPE_NAMES
from ._utils import SaveState, _import_obj, get_module, get_state, gettype
from .exceptions import UnsupportedTypeException


def dict_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "DictNode",
    }

    key_types = get_state([type(key) for key in obj.keys()], save_state)
    content = {}
    for key, value in obj.items():
        if isinstance(value, property):
            continue
        if np.isscalar(key) and hasattr(key, "item"):
            # convert numpy value to python object
            key = key.item()  # type: ignore
        content[key] = get_state(value, save_state)
    res["content"] = content
    res["key_types"] = key_types
    return res


class DictNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.trusted = self._get_trusted(trusted, ["builtins.dict"])
        self.children = {"key_types": Node, "content": dict}
        self.key_types = get_tree(state["key_types"], src)
        self.content = {
            key: get_tree(value, src) for key, value in state["content"].items()
        }

    def construct(self):
        content = gettype(self.module_name, self.class_name)()
        key_types = self.key_types.construct()
        for k_type, item in zip(key_types, self.content.items()):
            content[k_type(item[0])] = item[1].construct()
        return content


def list_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "ListNode",
    }
    content = [get_state(value, save_state) for value in obj]
    res["content"] = content
    return res


class ListNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.trusted = self._get_trusted(trusted, ["builtins.list"])
        self.children = {"content": list}
        self.content = [get_tree(value, src) for value in state["content"]]

    def construct(self):
        content_type = gettype(self.module_name, self.class_name)
        return content_type([item.construct() for item in self.content])


def set_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "SetNode",
    }
    content = [get_state(value, save_state) for value in obj]
    res["content"] = content
    return res


class SetNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.trusted = self._get_trusted(trusted, ["builtins.set"])
        self.children = {"content": list}
        self.content = [get_tree(value, src) for value in state["content"]]

    def construct(self):
        content_type = gettype(self.module_name, self.class_name)
        return content_type([item.construct() for item in self.content])


def tuple_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "TupleNode",
    }
    content = tuple(get_state(value, save_state) for value in obj)
    res["content"] = content
    return res


class TupleNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.trusted = self._get_trusted(trusted, ["builtins.tuple"])
        self.children = {"content": list}
        self.content = [get_tree(value, src) for value in state["content"]]

    def construct(self):
        # Returns a tuple or a namedtuple instance.

        cls = gettype(self.module_name, self.class_name)
        content = tuple(value.construct() for value in self.content)

        if self.isnamedtuple(cls):
            return cls(*content)
        return content

    def isnamedtuple(self, t):
        # This is needed since namedtuples need to have the args when
        # initialized.
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple:
            return False
        f = getattr(t, "_fields", None)
        if not isinstance(f, tuple):
            return False
        return all(type(n) == str for n in f)


def function_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "__loader__": "FunctionNode",
        "content": {
            "module_path": get_module(obj),
            "function": obj.__name__,
        },
    }
    return res


class FunctionNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])
        self.children = {"content": FunctionType}
        self.content = state["content"]

    def construct(self):
        return _import_obj(self.content["module_path"], self.content["function"])

    @property
    def is_safe(self):
        return False

    def get_safety_tree(self, report_safe=True):
        raise NotImplementedError()

    def get_unsafe_set(self):
        raise NotImplementedError()


def partial_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    _, _, (func, args, kwds, namespace) = obj.__reduce__()
    res = {
        "__class__": "partial",  # don't allow any subclass
        "__module__": get_module(type(obj)),
        "__loader__": "PartialNode",
        "content": {
            "func": get_state(func, save_state),
            "args": get_state(args, save_state),
            "kwds": get_state(kwds, save_state),
            "namespace": get_state(namespace, save_state),
        },
    }
    return res


class PartialNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        # TODO: should we trust anything?
        self.trusted = self._get_trusted(trusted, [])
        self.children = {"func": Node, "args": Node, "kwds": Node, "namespace": Node}
        self.func = get_tree(state["content"]["func"], src)
        self.args = get_tree(state["content"]["args"], src)
        self.kwds = get_tree(state["content"]["kwds"], src)
        self.namespace = get_tree(state["content"]["namespace"], src)

    def construct(self):
        func = self.func.construct()
        args = self.args.construct()
        kwds = self.kwds.construct()
        namespace = self.namespace.construct()
        instance = partial(func, *args, **kwds)  # always use partial, not a subclass
        instance.__setstate__((func, args, kwds, namespace))
        return instance


def type_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__name__,
        "__module__": get_module(obj),
        "__loader__": "TypeNode",
    }
    return res

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "TypeNode",
        "content": {
            "__class__": obj.__name__,
            "__module__": get_module(obj),
        },
    }
    return res


class TypeNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, PRIMITIVE_TYPE_NAMES)
        # We use a bare Node type here since a Node only checks the type in the
        # dict using __class__ and __module__ keys.
        # self.children = {"content": Node}
        self.children = {}
        # self.content = Node(state["content"]

    def construct(self):
        return _import_obj(self.module_name, self.class_name)
        return _import_obj(self.content["__module__"], self.content["__class__"])


def slice_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "SliceNode",
        "content": {
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        },
    }
    return res


class SliceNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.trusted = self._get_trusted(trusted, ["builtins.slice"])
        self.children = {}
        self.start = state["content"]["start"]
        self.stop = state["content"]["stop"]
        self.step = state["content"]["step"]

    def construct(self):
        return slice(self.start, self.stop, self.step)


def object_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
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
        "__loader__": "ObjectNode",
    }

    # __getstate__ takes priority over __dict__, and if non exist, we only save
    # the type of the object, and loading would mean instantiating the object.
    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return res

    content = get_state(attrs, save_state)
    # it's sufficient to store the "content" because we know that this dict can
    # only have str type keys
    res["content"] = content
    return res


class ObjectNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)

        if "content" in state:
            self.attrs = get_tree(state.get("content"), src)
        else:
            self.attrs = None

        self.children = {"attrs": Node}
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])

    def construct(self):
        cls = gettype(self.module_name, self.class_name)

        # Instead of simply constructing the instance, we use __new__, which
        # bypasses the __init__, and then we set the attributes. This solves
        # the issue of required init arguments.
        instance = cls.__new__(cls)

        if not self.attrs:  # nothing more to do
            return instance

        attrs = self.attrs.construct()
        if hasattr(instance, "__setstate__"):
            instance.__setstate__(attrs)
        else:
            instance.__dict__.update(attrs)

        return instance


def method_get_state(obj: Any, save_state: SaveState):
    # This method is used to persist bound methods, which are
    # dependent on a specific instance of an object.
    # It stores the state of the object the method is bound to,
    # and prepares both to be persisted.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(obj),
        "__loader__": "MethodNode",
        "content": {
            "func": obj.__func__.__name__,
            "obj": get_state(obj.__self__, save_state),
        },
    }

    return res


class MethodNode(Node):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.children = {"obj": Node}
        self.func = state["content"]["func"]
        self.obj = get_tree(state["content"]["obj"], src)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])

    def construct(self):
        loaded_obj = self.obj.construct()
        method = getattr(loaded_obj, self.func)
        return method


def unsupported_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    raise UnsupportedTypeException(obj)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (list, list_get_state),
    (set, set_get_state),
    (tuple, tuple_get_state),
    (slice, slice_get_state),
    (FunctionType, function_get_state),
    (MethodType, method_get_state),
    (partial, partial_get_state),
    (type, type_get_state),
    (object, object_get_state),
]

NODE_TYPE_MAPPING = {
    "DictNode": DictNode,
    "ListNode": ListNode,
    "SetNode": SetNode,
    "TupleNode": TupleNode,
    "SliceNode": SliceNode,
    "FunctionNode": FunctionNode,
    "MethodNode": MethodNode,
    "PartialNode": PartialNode,
    "TypeNode": TypeNode,
    "ObjectNode": ObjectNode,
}
