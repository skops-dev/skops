import importlib
import json  # type: ignore
from functools import _find_impl, get_cache_token, update_wrapper  # type: ignore
from types import FunctionType

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


@singledispatch
def get_state(obj, dst):
    raise TypeError(f"Getting the state of type {type(obj)} is not supported yet")


@singledispatch
def get_instance(obj):
    raise TypeError(f"Creating an instance of type {type(obj)} is not supported yet")


def try_get_state(value, dst):
    # This is a helper function to try to get the state of an object. If it
    # fails with `get_state`, we try with json.dumps, if that fails, we raise
    # the original error alongside the json error.
    try:
        return get_state(value, dst)
    except TypeError as e1:
        try:
            return json.dumps(value)
        except Exception as e2:
            raise e1 from e2


def try_get_instance(value, src):
    # This is a helper function to try to get the state of an object. If
    # `gettype` fails, we load with `json`.
    if gettype(value):
        return get_instance(value, src)

    return json.loads(value)
