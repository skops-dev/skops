from __future__ import annotations

import importlib
import json  # type: ignore
import sys
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any
from zipfile import ZipFile


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
def whichmodule(obj, name):
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
            if _getattribute(module, name)[0] is obj:
                return module_name
        except AttributeError:
            pass
    return "__main__"


# ---------------------------------------------------------------------


def _import_obj(module, cls_or_func, package=None):
    return getattr(importlib.import_module(module, package=package), cls_or_func)


def gettype(state):
    if "__module__" in state and "__class__" in state:
        return _import_obj(state["__module__"], state["__class__"])
    return None


def get_module(obj):
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


# For now, there is just one protocol version
DEFAULT_PROTOCOL = 0


@dataclass(frozen=True)
class SaveState:
    """State required for saving the objects

    This state is passed to each ``get_state_*`` function.

    Parameters
    ----------
    zip_file: zipfile.ZipFile
        The zip file to write the data to, must be in write mode.

    path: pathlib.Path
        The path to the directory to store the object in.

    protocol: int
        The protocol of the persistence format. Right now, there is only
        protocol 0, but this leaves the door open for future changes.

    """

    zip_file: ZipFile
    protocol: int = DEFAULT_PROTOCOL
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
class LoadState:
    """State required for loading an object

    This state is passed to each ``get_instance_*`` function.

    Parameters
    ----------
    src: zipfile.ZipFile
        The zip file the target object is saved in

    """

    src: ZipFile
    memo: dict[int, Any] = field(default_factory=dict)

    def memoize(self, obj: Any, id: int) -> None:
        self.memo[id] = obj

    def get_instance(self, id: int) -> Any:
        return self.memo.get(id)


@singledispatch
def _get_state(obj, save_state):
    # This function should never be called directly. Instead, it is used to
    # dispatch to the correct implementation of get_state for the given type of
    # its first argument.
    raise TypeError(f"Getting the state of type {type(obj)} is not supported yet")


def get_state(value, save_state):
    # This is a helper function to try to get the state of an object. If it
    # fails with `get_state`, we try with json.dumps, if that fails, we raise
    # the original error alongside the json error.
    try:
        return _get_state(value, save_state)
    except TypeError as e1:
        try:
            return json.dumps(value)
        except Exception as e2:
            raise e1 from e2
