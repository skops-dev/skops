from __future__ import annotations

import importlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from functools import singledispatch
from types import ModuleType
from typing import Any, Sequence, Type, Union
from zipfile import ZipFile

from ._protocol import PROTOCOL
from .exceptions import UnsupportedTypeException

# Types the user trusts to be loaded, given either by their fully qualified
# name, e.g. "sklearn.linear_model._logistic.LogisticRegression" as returned by
# ``get_untrusted_types``, or as the type object itself.
TrustedTypes = Sequence[Union[str, Type[Any]]]


# The following two functions are copied from cpython's pickle.py file.
# ---------------------------------------------------------------------
def _getattribute(obj, name):  # pragma: no cover
    parent = obj
    for subpath in name.split("."):
        if subpath == "<locals>":
            raise AttributeError(
                "Can't get local attribute {!r} on {!r}".format(name, obj)
            )
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError(
                "Can't get attribute {!r} on {!r}".format(name, obj)
            ) from None
    return obj, parent


# This function is particularly used to detect the path of functions such as
# ufuncs. It returns the full path, instead of returning the module name.
def whichmodule(obj: Any, name: str) -> str:
    """Find the module an object belong to."""
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        return module_name
    # Objects without ``__module__`` (e.g. scipy ufuncs) are searched for in
    # every loaded module, in import order, until the first match. A ``getattr``
    # with a default avoids raising and catching an exception for every module
    # that lacks the attribute, which keeps this loop cheap in processes with
    # many loaded modules. Dotted names keep going through ``_getattribute``
    # for its ``<locals>`` handling; skops itself only passes ``__name__``.
    with warnings.catch_warnings():
        # this is to silence numpy.core import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        # Protect the iteration by using a list copy of sys.modules against
        # dynamic modules that trigger imports of other modules upon calls to
        # getattr.
        for module_name, module in sys.modules.copy().items():
            if (
                module_name == "__main__"
                or module_name == "__mp_main__"  # bpo-42406
                or module is None
            ):
                continue
            try:
                if "." in name:  # pragma: no cover
                    found = _getattribute(module, name)[0]
                else:
                    found = getattr(module, name, None)
            except (AttributeError, ImportError):
                continue
            if found is obj:
                return module_name
    return "__main__"


# ---------------------------------------------------------------------


def _import_obj(module: str, cls_or_func: str, package: str | None = None) -> Any:
    return getattr(importlib.import_module(module, package=package), cls_or_func)


def gettype(module_name: str, cls_or_func: str) -> Type[Any]:
    if module_name and cls_or_func:
        return _import_obj(module_name, cls_or_func)

    raise ValueError(f"Object {cls_or_func!r} of module {module_name!r} is unknown")


def get_module(obj: Any) -> str:
    """Find module for given object

    If the module cannot be identified, it's assumed to be "__main__".

    Parameters
    ----------
    obj: Any
       Object whose module is requested.

    Returns
    -------
    name: str
        Name of the module.

    """
    return whichmodule(obj, obj.__name__)


@dataclass(frozen=True)
class SaveContext:
    """Context required for saving the objects

    This context is passed to each ``get_state_*`` function.

    Parameters
    ----------
    zip_file: zipfile.ZipFile
        The zip file to write the data to, must be in write mode.

    protocol: int
        The protocol of the persistence format. Right now, there is only
        protocol 0, but this leaves the door open for future changes.

    """

    zip_file: ZipFile
    protocol: int = PROTOCOL
    memo: dict[int, Any] = field(default_factory=dict)
    # The ids of the objects whose state is being computed right now, i.e. the
    # path from the root object to the current one, mapped to whether a
    # reference back to that object was found inside its own state. Used by
    # ``get_state`` to detect circular references.
    in_progress: dict[int, bool] = field(default_factory=dict)

    def memoize(self, obj: Any) -> int:
        # Currently, the only purpose for saving the object id is to make sure
        # that for the length of the context that the main object is being
        # saved, all attributes persist, so that the same id cannot be re-used
        # for different objects.
        obj_id = id(obj)
        if obj_id not in self.memo:
            self.memo[obj_id] = obj
        return obj_id

    def clear_memo(self) -> None:
        self.memo.clear()


@dataclass(frozen=True)
class LoadContext:
    """Context required for loading an object

    This context is passed to each ``*Node`` class when loading an object.

    Parameters
    ----------
    src: zipfile.ZipFile
        The zip file the target object is saved in

    protocol: int
        The protocol the file was saved with. Use :func:`read_schema` to build
        a ``LoadContext`` from a file, which validates this value.
    """

    src: ZipFile
    protocol: int
    memo: dict[int, Any] = field(default_factory=dict)

    def memoize(self, obj: Any, id: int) -> None:
        self.memo[id] = obj

    def get_object(self, id: int) -> Any:
        return self.memo.get(id)


def read_schema(zip_file: ZipFile) -> tuple[dict[str, Any], LoadContext]:
    """Read and validate ``schema.json`` of a skops file.

    The protocol number stored in the file decides which ``Node`` classes are
    used to audit and construct its content, and it is fully under the control
    of whoever produced the file. It is therefore validated here, before any
    node is created: it must be an integer between 0 and the protocol of the
    running skops version.

    Parameters
    ----------
    zip_file: zipfile.ZipFile
        The skops file, opened for reading.

    Returns
    -------
    schema: dict
        The parsed ``schema.json``.

    load_context: LoadContext
        The context to pass to ``get_tree``.
    """
    schema = json.loads(zip_file.read("schema.json"))
    protocol = schema.get("protocol")
    # bool is a subclass of int, and True would silently act as protocol 1
    if isinstance(protocol, bool) or not isinstance(protocol, int):
        raise TypeError(
            f"Invalid skops protocol {protocol!r} in schema.json, expected an integer."
        )
    if protocol > PROTOCOL:
        raise ValueError(
            f"The file was saved with skops protocol {protocol}, but this version "
            f"of skops only supports protocols up to {PROTOCOL}. You might need to "
            "update skops to load this file."
        )
    if protocol < 0:
        raise ValueError(
            f"Invalid skops protocol {protocol} in schema.json, expected a value "
            f"between 0 and {PROTOCOL}."
        )
    return schema, LoadContext(src=zip_file, protocol=protocol)


@singledispatch
def _get_state(obj, save_context: SaveContext):
    # This function should never be called directly. Instead, it is used to
    # dispatch to the correct implementation of get_state for the given type of
    # its first argument.
    raise TypeError(f"Getting the state of type {type(obj)} is not supported yet")


def _supports_circular_reference(value: Any, state: dict[str, Any]) -> bool:
    """Whether a reference to ``value`` from inside its own state can be loaded.

    This mirrors the ``Node`` classes whose ``_construct`` registers the
    instance before constructing its children, see ``Node.construct``.
    ``ListNode`` and ``SetNode`` only do so for plain lists and sets.
    """
    if type(value) in (list, set):
        return True
    return state["__loader__"] in ("DictNode", "ObjectNode")


def get_state(value, save_context: SaveContext) -> dict[str, Any]:
    # This is a helper function to try to get the state of an object. If it
    # fails with `get_state`, we try with json.dumps, if that fails, we raise
    # the original error alongside the json error.
    __id__ = save_context.memoize(obj=value)

    if __id__ in save_context.in_progress:
        # We are already computing the state of ``value``, so it contains a
        # reference to itself, directly or through its children. Instead of
        # recursing forever, save a reference to it. When loading, ``get_tree``
        # resolves the ``__id__`` to the node of the occurrence which holds the
        # actual state, since that node is created before its children.
        save_context.in_progress[__id__] = True
        return {
            "__class__": type(value).__name__,
            "__module__": get_module(type(value)),
            "__loader__": "CachedNode",
            "__id__": __id__,
        }

    save_context.in_progress[__id__] = False
    try:
        res = _get_state(value, save_context)
        if save_context.in_progress[__id__] and not _supports_circular_reference(
            value, res
        ):
            raise UnsupportedTypeException(
                f"Objects of type {type(value).__name__} which contain a"
                " reference to themselves are not supported yet."
            )
    finally:
        del save_context.in_progress[__id__]

    res["__id__"] = __id__
    return res


def get_type_name(t: Any) -> str:
    """Helper function to take in a type, and return its name as a string"""
    return f"{get_module(t)}.{t.__name__}"


def get_type_paths(types: str | type[Any] | TrustedTypes | None) -> list[str]:
    """Helper function that takes in a types,
    and converts any the types found to a list of strings.

    Parameters
    ----------
    types: str, type, list of str and types, or None
        Types to get. Can be either a string, a single type, or a list of strings
        and types.

    Returns
    ----------
    types_list: list of str
        The list of types, all as strings, e.g. ``["builtins.list"]``.

    """
    if not types:
        return []
    items: TrustedTypes = [types] if isinstance(types, (str, type)) else types
    return [t if isinstance(t, str) else get_type_name(t) for t in items]


def get_public_type_names(module: ModuleType, oftype: Type) -> list[str]:
    """
    Helper function that gets the type names of all
    public objects of the given ``oftype`` from the given ``module``,
    which start with the root module name.

    Public objects are those that can be read via ``dir(...)``.

    Parameters
    ----------
    module: ModuleType
        Module under which the public objects are defined.
    oftype: Type
        The type of the objects.

    Returns
    ----------
    type_names_list: list of str
        The sorted list of type names, all as strings,
         e.g. ``["numpy.core._multiarray_umath.absolute"]``.
    """
    module_name, _, _ = module.__name__.rpartition(".")

    return sorted(
        {
            type_name
            for attr in dir(module)
            if issubclass((obj := getattr(module, attr)).__class__, oftype)
            and (type_name := get_type_name(obj)).startswith(module_name)
        }
    )
