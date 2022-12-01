from __future__ import annotations

import json
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Sequence

import numpy as np

from ._audit import Node, get_tree
from ._trusted_types import PRIMITIVE_TYPE_NAMES
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
        "__loader__": "DictNode",
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


class DictNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [dict])
        self.children = {
            "key_types": get_tree(state["key_types"], load_context),
            "content": {
                key: get_tree(value, load_context)
                for key, value in state["content"].items()
            },
        }

    def _construct(self):
        content = gettype(self.module_name, self.class_name)()
        key_types = self.children["key_types"].construct()
        for k_type, (key, val) in zip(key_types, self.children["content"].items()):
            content[k_type(key)] = val.construct()
        return content


def list_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "ListNode",
    }
    content = [get_state(value, save_context) for value in obj]

    res["content"] = content
    return res


class ListNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [list])
        self.children = {
            "content": [get_tree(value, load_context) for value in state["content"]]
        }

    def _construct(self):
        content_type = gettype(self.module_name, self.class_name)
        return content_type([item.construct() for item in self.children["content"]])


def set_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "SetNode",
    }
    content = [get_state(value, save_context) for value in obj]
    res["content"] = content
    return res


class SetNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [set])
        self.children = {
            "content": [get_tree(value, load_context) for value in state["content"]]
        }

    def _construct(self):
        content_type = gettype(self.module_name, self.class_name)
        return content_type([item.construct() for item in self.children["content"]])


def tuple_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "TupleNode",
    }
    content = tuple(get_state(value, save_context) for value in obj)
    res["content"] = content
    return res


class TupleNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [tuple])
        self.children = {
            "content": [get_tree(value, load_context) for value in state["content"]]
        }

    def _construct(self):
        # Returns a tuple or a namedtuple instance.

        cls = gettype(self.module_name, self.class_name)
        content = tuple(value.construct() for value in self.children["content"])

        if self.isnamedtuple(cls):
            return cls(*content)
        return content

    def isnamedtuple(self, t) -> bool:
        # This is needed since namedtuples need to have the args when
        # initialized.
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple:
            return False
        f = getattr(t, "_fields", None)
        if not isinstance(f, tuple):
            return False
        return all(type(n) == str for n in f)


def function_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
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
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])
        self.children = {"content": state["content"]}

    def _construct(self):
        return _import_obj(
            self.children["content"]["module_path"],
            self.children["content"]["function"],
        )

    def _get_function_name(self) -> str:
        return (
            self.children["content"]["module_path"]
            + "."
            + self.children["content"]["function"]
        )

    def get_unsafe_set(self) -> set[str]:
        if self.trusted is True:
            return set()

        return {self._get_function_name()}


def partial_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    _, _, (func, args, kwds, namespace) = obj.__reduce__()
    res = {
        "__class__": "partial",  # don't allow any subclass
        "__module__": get_module(type(obj)),
        "__loader__": "PartialNode",
        "content": {
            "func": get_state(func, save_context),
            "args": get_state(args, save_context),
            "kwds": get_state(kwds, save_context),
            "namespace": get_state(namespace, save_context),
        },
    }
    return res


class PartialNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: should we trust anything?
        self.trusted = self._get_trusted(trusted, [])
        self.children = {
            "func": get_tree(state["content"]["func"], load_context),
            "args": get_tree(state["content"]["args"], load_context),
            "kwds": get_tree(state["content"]["kwds"], load_context),
            "namespace": get_tree(state["content"]["namespace"], load_context),
        }

    def _construct(self):
        func = self.children["func"].construct()
        args = self.children["args"].construct()
        kwds = self.children["kwds"].construct()
        namespace = self.children["namespace"].construct()
        instance = partial(func, *args, **kwds)  # always use partial, not a subclass
        # partial always has __setstate__
        instance.__setstate__((func, args, kwds, namespace))  # type: ignore
        return instance


def type_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__name__,
        "__module__": get_module(obj),
        "__loader__": "TypeNode",
    }
    return res


class TypeNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, PRIMITIVE_TYPE_NAMES)
        # We use a bare Node type here since a Node only checks the type in the
        # dict using __class__ and __module__ keys.
        self.children = {}

    def _construct(self):
        return _import_obj(self.module_name, self.class_name)


def slice_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
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
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [slice])
        self.children = {
            "start": state["content"]["start"],
            "stop": state["content"]["stop"],
            "step": state["content"]["step"],
        }

    def _construct(self):
        return slice(
            self.children["start"], self.children["stop"], self.children["step"]
        )

    def get_unsafe_set(self):
        return set()


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
            "__loader__": "JsonNode",
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

    content = get_state(attrs, save_context)
    # it's sufficient to store the "content" because we know that this dict can
    # only have str type keys
    res["content"] = content
    return res


class ObjectNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)

        content = state.get("content")
        if content is not None:
            attrs = get_tree(content, load_context)
        else:
            attrs = None

        self.children = {"attrs": attrs}
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])

    def _construct(self):
        cls = gettype(self.module_name, self.class_name)

        # Instead of simply constructing the instance, we use __new__, which
        # bypasses the __init__, and then we set the attributes. This solves the
        # issue of required init arguments. Note that the instance created here
        # might not be valid until all its attributes have been set below.
        instance = cls.__new__(cls)  # type: ignore

        if not self.children["attrs"]:
            # nothing more to do
            return instance

        attrs = self.children["attrs"].construct()
        if hasattr(instance, "__setstate__"):
            instance.__setstate__(attrs)
        else:
            instance.__dict__.update(attrs)

        return instance


def method_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
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
            "obj": get_state(obj.__self__, save_context),
        },
    }
    return res


class MethodNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {
            "obj": get_tree(state["content"]["obj"], load_context),
            "func": state["content"]["func"],
        }
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])

    def _construct(self):
        loaded_obj = self.children["obj"].construct()
        method = getattr(loaded_obj, self.children["func"])
        return method


def unsupported_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    raise UnsupportedTypeException(obj)


class JsonNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.content = state["content"]
        self.children = {}

    def is_safe(self) -> bool:
        # JsonNode is always considered safe.
        # TODO: should we consider a JsonNode always safe?
        return True

    def is_self_safe(self) -> bool:
        return True

    def get_unsafe_set(self) -> set[str]:
        return set()

    def _construct(self):
        return json.loads(self.content)


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
    "JsonNode": JsonNode,
}
