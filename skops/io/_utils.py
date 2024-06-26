from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass, field
from functools import singledispatch
from types import ModuleType
from typing import Any, Type
from zipfile import ZipFile

from ._protocol import PROTOCOL


# The following two functions are copied from cpython's pickle.py file.
# ---------------------------------------------------------------------
def _getattribute(obj, name):
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
    # Protect the iteration by using a list copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr.
    for module_name, module in sys.modules.copy().items():
        if (
            module_name == "__main__"
            or module_name == "__mp_main__"  # bpo-42406
            or module is None
        ):
            continue
        try:
            with warnings.catch_warnings():
                # this is to silence numpy.core import warnings
                warnings.simplefilter("ignore", DeprecationWarning)
                if _getattribute(module, name)[0] is obj:
                    return module_name
        except AttributeError:
            pass
    return "__main__"


# ---------------------------------------------------------------------


def _import_obj(module: str, cls_or_func: str, package: str | None = None) -> Any:
    return getattr(importlib.import_module(module, package=package), cls_or_func)


def gettype(module_name: str, cls_or_func: str) -> Type[Any]:
    if module_name and cls_or_func:
        return _import_obj(module_name, cls_or_func)

    raise ValueError(f"Object {cls_or_func} of module {module_name} is unknown")


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
    """

    src: ZipFile
    protocol: int
    memo: dict[int, Any] = field(default_factory=dict)

    def memoize(self, obj: Any, id: int) -> None:
        self.memo[id] = obj

    def get_object(self, id: int) -> Any:
        return self.memo.get(id)


@singledispatch
def _get_state(obj, save_context: SaveContext):
    # This function should never be called directly. Instead, it is used to
    # dispatch to the correct implementation of get_state for the given type of
    # its first argument.
    raise TypeError(f"Getting the state of type {type(obj)} is not supported yet")


def get_state(value, save_context: SaveContext) -> dict[str, Any]:
    # This is a helper function to try to get the state of an object. If it
    # fails with `get_state`, we try with json.dumps, if that fails, we raise
    # the original error alongside the json error.

    # TODO: This should help with fixing recursive references.
    # if id(value) in save_context.memo:
    #     return {
    #         "__module__": None,
    #         "__class__": None,
    #         "__id__": id(value),
    #         "__loader__": "CachedNode",
    #     }

    __id__ = save_context.memoize(obj=value)

    res = _get_state(value, save_context)

    res["__id__"] = __id__
    return res


def get_type_name(t: Any) -> str:
    """Helper function to take in a type, and return its name as a string"""
    return f"{get_module(t)}.{t.__name__}"


def get_type_paths(types: Any) -> list[str]:
    """Helper function that takes in a types,
    and converts any the types found to a list of strings.

    Parameters
    ----------
    types: Any
        Types to get. Can be either a string, a single type, or a list of strings
        and types.

    Returns
    ----------
    types_list: list of str
        The list of types, all as strings, e.g. ``["builtins.list"]``.

    """
    if not types:
        return []
    if not isinstance(types, (list, tuple)):
        types = [types]

    return [get_type_name(t) if not isinstance(t, str) else t for t in types]


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
