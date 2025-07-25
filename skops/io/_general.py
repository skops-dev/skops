from __future__ import annotations

import io
import json
import operator
import uuid
from collections import defaultdict
from functools import partial
from reprlib import Repr
from types import FunctionType, MethodType
from typing import Any, Optional, Sequence

import numpy as np

from ._audit import Node, get_tree
from ._protocol import PROTOCOL
from ._trusted_types import (
    CONTAINER_TYPE_NAMES,
    NUMPY_DTYPE_TYPE_NAMES,
    NUMPY_UFUNC_TYPE_NAMES,
    PRIMITIVE_TYPE_NAMES,
    SCIPY_UFUNC_TYPE_NAMES,
    SKLEARN_ESTIMATOR_TYPE_NAMES,
)
from ._utils import (
    LoadContext,
    SaveContext,
    _import_obj,
    get_module,
    get_state,
    gettype,
)
from .exceptions import UnsupportedTypeException

arepr = Repr()
arepr.maxstring = 24


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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [dict, "collections.OrderedDict"])
        self.children = {
            "key_types": get_tree(state["key_types"], load_context, trusted=trusted),
            "content": {
                key: get_tree(value, load_context, trusted=trusted)
                for key, value in state["content"].items()
            },
        }

    def _construct(self):
        content = gettype(self.module_name, self.class_name)()
        key_types = self.children["key_types"].construct()
        for k_type, (key, val) in zip(key_types, self.children["content"].items()):
            content[k_type(key)] = val.construct()
        return content


def defaultdict_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "DefaultDictNode",
    }
    content = {}
    # explicitly pass a dict object instead of _DictWithDeprecatedKeys and
    # later construct a _DictWithDeprecatedKeys object.
    content["main"] = get_state(dict(obj), save_context)
    content["default_factory"] = get_state(obj.default_factory, save_context)
    res["content"] = content
    return res


class DefaultDictNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = ["collections.defaultdict"]
        self.children = {
            "main": get_tree(state["content"]["main"], load_context, trusted=trusted),
            "default_factory": get_tree(
                state["content"]["default_factory"],
                load_context,
                trusted=trusted,
            ),
        }

    def _construct(self):
        instance = defaultdict(**self.children["main"].construct())
        instance.default_factory = self.children["default_factory"].construct()
        return instance


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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [list])
        self.children = {
            "content": [
                get_tree(value, load_context, trusted=trusted)
                for value in state["content"]
            ]
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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [set])
        self.children = {
            "content": [
                get_tree(value, load_context, trusted=trusted)
                for value in state["content"]
            ]
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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [tuple])
        self.children = {
            "content": [
                get_tree(value, load_context, trusted=trusted)
                for value in state["content"]
            ]
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
        return all(isinstance(n, str) for n in f)


def function_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__name__,
        "__module__": get_module(obj),
        "__loader__": "FunctionNode",
    }
    return res


class FunctionNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(
            trusted, default=SCIPY_UFUNC_TYPE_NAMES + NUMPY_UFUNC_TYPE_NAMES
        )
        self.children = {}

    def _construct(self):
        return gettype(self.module_name, self.class_name)

    def _get_function_name(self) -> str:
        return f"{self.module_name}.{self.class_name}"

    def get_unsafe_set(self) -> set[str]:
        fn_name = self._get_function_name()
        if (self.trusted is True) or (fn_name in self.trusted):
            return set()

        return {fn_name}


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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: should we trust anything?
        self.trusted = self._get_trusted(trusted, [])
        self.children = {
            "func": get_tree(state["content"]["func"], load_context, trusted=trusted),
            "args": get_tree(state["content"]["args"], load_context, trusted=trusted),
            "kwds": get_tree(state["content"]["kwds"], load_context, trusted=trusted),
            "namespace": get_tree(
                state["content"]["namespace"], load_context, trusted=trusted
            ),
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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        # TODO: what do we trust?
        self.trusted = self._get_trusted(
            trusted,
            PRIMITIVE_TYPE_NAMES + CONTAINER_TYPE_NAMES + NUMPY_DTYPE_TYPE_NAMES,
        )
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
        trusted: Optional[Sequence[str]] = None,
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

    # We first check if the object can be serialized using json.
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

    # Then we check if the output of __reduce__ is of the form
    # (constructor, (constructor_args,))
    # If the constructor is the same as the object's type, then we consider it
    # safe to call it with the specified arguments.

    reduce_output = obj.__reduce__()
    if len(reduce_output) == 2 and reduce_output[0] is type(obj):
        return {
            "__class__": type(obj).__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "ConstructorFromReduceNode",
            "content": get_state(reduce_output[1], save_context),
        }

    # Otherwise we recover the object from the __dict__ or __getstate__
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


class ConstructorFromReduceNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.children = {
            "content": get_tree(state["content"], load_context, trusted=trusted)
        }

    def _construct(self):
        return gettype(self.module_name, self.class_name)(
            *self.children["content"].construct()
        )


class ObjectNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)

        content = state.get("content")
        if content is not None:
            attrs = get_tree(content, load_context, trusted=trusted)
        else:
            attrs = None

        self.children = {"attrs": attrs}
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, default=SKLEARN_ESTIMATOR_TYPE_NAMES)

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
        if attrs is not None:
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
    owner = obj.__self__
    func_name = obj.__func__.__name__
    res = {
        "__class__": owner.__class__.__name__,
        "__module__": get_module(obj),
        "__loader__": "MethodNode",
        "content": {
            "func": func_name,
            "obj": get_state(owner, save_context),
        },
    }
    return res


class MethodNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        obj = get_tree(state["content"]["obj"], load_context, trusted=trusted)
        if self.module_name != obj.module_name or self.class_name != obj.class_name:
            raise ValueError(
                f"Expected object of type {self.module_name}.{self.class_name}, got"
                f" {obj.module_name}.{obj.class_name}. This is probably due to a"
                " corrupted or a malicious file."
            )
        self.children = {
            "obj": obj,
            "func": state["content"]["func"],
        }
        # TODO: what do we trust?
        self.trusted = self._get_trusted(trusted, [])

    def get_unsafe_set(self) -> set[str]:
        res = super().get_unsafe_set()
        obj_node = self.children["obj"]
        res.add(
            obj_node.module_name  # type: ignore
            + "."
            + obj_node.class_name  # type: ignore
            + "."
            + self.children["func"]
        )
        return res

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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.content = state["content"]
        self.children = {}
        self.trusted = self._get_trusted(trusted, PRIMITIVE_TYPE_NAMES)

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

    def format(self) -> str:
        """Representation of the node's content.

        Since no module is used, just show the content.

        """
        return f"json-type({self.content})"


def bytes_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    f_name = f"{uuid.uuid4()}.bin"
    save_context.zip_file.writestr(f_name, obj)
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "BytesNode",
        "file": f_name,
    }
    return res


def bytearray_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = bytes_get_state(obj, save_context)
    res["__loader__"] = "BytearrayNode"
    return res


class BytesNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [bytes])
        self.children = {"content": io.BytesIO(load_context.src.read(state["file"]))}

    def _construct(self):
        content = self.children["content"].getvalue()
        return content

    def format(self):
        content = self.children["content"].getvalue()
        byte_repr = arepr.repr(content)
        return byte_repr


class BytearrayNode(BytesNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [bytearray])

    def _construct(self):
        content_bytes = super()._construct()
        content_bytearray = bytearray(list(content_bytes))
        return content_bytearray

    def format(self):
        return f"bytearray({super().format()})"


def operator_func_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    _, attrs = obj.__reduce__()
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": "operator",
        "__loader__": "OperatorFuncNode",
        "attrs": get_state(attrs, save_context),
    }
    return res


class OperatorFuncNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        if self.module_name != "operator":
            raise ValueError(
                f"Expected module 'operator', got {self.module_name}. This is probably"
                " due to a corrupted or a malicious file."
            )
        self.trusted = self._get_trusted(trusted, [])
        self.children["attrs"] = get_tree(state["attrs"], load_context, trusted=trusted)

    def _construct(self):
        op = getattr(operator, self.class_name)
        attrs = self.children["attrs"].construct()
        return op(*attrs)


# <class 'builtin_function_or_method'>
builtin_function_or_method = type(len)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (defaultdict, defaultdict_get_state),
    (list, list_get_state),
    (set, set_get_state),
    (tuple, tuple_get_state),
    (bytes, bytes_get_state),
    (bytearray, bytearray_get_state),
    (slice, slice_get_state),
    (FunctionType, function_get_state),
    (MethodType, method_get_state),
    (partial, partial_get_state),
    (type, type_get_state),
    (builtin_function_or_method, type_get_state),
    (operator.attrgetter, operator_func_get_state),
    (operator.itemgetter, operator_func_get_state),
    (operator.methodcaller, operator_func_get_state),
    (object, object_get_state),
]

NODE_TYPE_MAPPING = {
    ("DictNode", PROTOCOL): DictNode,
    ("DefaultDictNode", PROTOCOL): DefaultDictNode,
    ("ListNode", PROTOCOL): ListNode,
    ("SetNode", PROTOCOL): SetNode,
    ("TupleNode", PROTOCOL): TupleNode,
    ("BytesNode", PROTOCOL): BytesNode,
    ("BytearrayNode", PROTOCOL): BytearrayNode,
    ("SliceNode", PROTOCOL): SliceNode,
    ("FunctionNode", PROTOCOL): FunctionNode,
    ("MethodNode", PROTOCOL): MethodNode,
    ("PartialNode", PROTOCOL): PartialNode,
    ("TypeNode", PROTOCOL): TypeNode,
    ("ConstructorFromReduceNode", PROTOCOL): ConstructorFromReduceNode,
    ("ObjectNode", PROTOCOL): ObjectNode,
    ("JsonNode", PROTOCOL): JsonNode,
    ("OperatorFuncNode", PROTOCOL): OperatorFuncNode,
}
