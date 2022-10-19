from __future__ import annotations

import importlib
import json  # type: ignore
import sys
from dataclasses import dataclass, field
from functools import _find_impl, get_cache_token, update_wrapper  # type: ignore
from types import FunctionType
from typing import Any
from zipfile import ZipFile

from skops.utils.fixes import GenericAlias


# This is an almost 1:1 copy of functools.singledispatch. There is one crucial
# difference, however. Usually, we want to dispatch on the class of the object.
# However, when we call get_instance, the object is *always* a dict, which
# invalidates the dispatch. Therefore, we change the dispatcher to dispatch on
# the instance, not the class. By default, we just use the class of the instance
# being passed, i.e. we do exactly the same as in the original implementation.
# However, if we encounter a state dict, we resolve the actual class from the
# state dict first and then dispatch on that class. The changed lines are marked
# as "# CHANGED".
# fmt: off
def singledispatch(func):
    """Single-dispatch generic function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.
    """
    # There are many programs that use functools without singledispatch, so we
    # trade-off making singledispatch marginally slower for the benefit of
    # making start-up of such applications slightly faster.
    import types
    import weakref

    registry = {}
    dispatch_cache = weakref.WeakKeyDictionary()
    cache_token = None

    def dispatch(instance):  # CHANGED: variable name cls->instance
        """generic_func.dispatch(cls) -> <function implementation>

        Runs the dispatch algorithm to return the best available implementation
        for the given *cls* registered on *generic_func*.

        """
        # CHANGED: check if we deal with a state dict, in which case we use it
        # to resolve the correct class. Otherwise, just use the class of the
        # instance.
        if (
            isinstance(instance, dict)
            and "__module__" in instance
            and "__class__" in instance
        ):
            cls = gettype(instance)
        else:
            cls = instance.__class__

        nonlocal cache_token
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token
        try:
            impl = dispatch_cache[cls]
        except KeyError:
            try:
                impl = registry[cls]
            except KeyError:
                impl = _find_impl(cls, registry)
            dispatch_cache[cls] = impl
        return impl

    def _is_valid_dispatch_type(cls):
        return isinstance(cls, type) and not isinstance(cls, GenericAlias)

    def register(cls, func=None):
        """generic_func.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_func*.

        """
        nonlocal cache_token
        if _is_valid_dispatch_type(cls):
            if func is None:
                return lambda f: register(cls, f)
        else:
            if func is not None:
                raise TypeError(
                    f"Invalid first argument to `register()`. "
                    f"{cls!r} is not a class."
                )
            ann = getattr(cls, '__annotations__', {})
            if not ann:
                raise TypeError(
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
            func = cls
            # only import typing if annotation parsing is necessary
            from typing import get_type_hints
            argname, cls = next(iter(get_type_hints(func).items()))
            if not _is_valid_dispatch_type(cls):
                raise TypeError(
                    f"Invalid annotation for {argname!r}. "
                    f"{cls!r} is not a class."
                )

        registry[cls] = func
        if cache_token is None and hasattr(cls, '__abstractmethods__'):
            cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        if not args:
            raise TypeError(f'{funcname} requires at least '
                            '1 positional argument')

        # CHANGED: dispatch on instance, not class
        return dispatch(args[0])(*args, **kw)

    funcname = getattr(func, '__name__', 'singledispatch function')
    registry[object] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper
# fmt: on


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
        if state["__class__"] == "function":
            # This special case is due to how functions are serialized. We
            # could try to change it.
            return FunctionType
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
        # Currenlty, the only purpose for saving the object id is to make sure
        # that for the length of the context that the main object is being
        # saved, all attributes persist, so that the same id cannot be re-used
        # for different objects.
        obj_id = id(obj)
        if obj_id not in self.memo:
            self.memo[obj_id] = obj
        return obj_id

    def clear_memo(self) -> None:
        self.memo.clear()


@singledispatch
def _get_state(obj, dst):
    # This function should never be called directly. Instead, it is used to
    # dispatch to the correct implementation of get_state for the given type of
    # its first argument.
    raise TypeError(f"Getting the state of type {type(obj)} is not supported yet")


@singledispatch
def _get_instance(obj, src):
    # This function should never be called directly. Instead, it is used to
    # dispatch to the correct implementation of get_instance for the given type
    # of its first argument.
    raise TypeError(f"Creating an instance of type {type(obj)} is not supported yet")


def get_state(value, dst):
    # This is a helper function to try to get the state of an object. If it
    # fails with `get_state`, we try with json.dumps, if that fails, we raise
    # the original error alongside the json error.
    try:
        return _get_state(value, dst)
    except TypeError as e1:
        try:
            return json.dumps(value)
        except Exception as e2:
            raise e1 from e2


def get_instance(value, src):
    # This is a helper function to try to get the state of an object. If
    # `gettype` fails, we load with `json`.
    if value is None:
        return None

    if gettype(value):
        return _get_instance(value, src)

    return json.loads(value)
