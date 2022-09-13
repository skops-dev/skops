import inspect
import json

from sklearn.tree._tree import Tree
from sklearn.utils import Bunch

from ._general import dict_get_instance
from ._utils import (
    _get_instance,
    _get_state,
    get_instance,
    get_module,
    get_state,
    gettype,
)


def generic_get_state(obj, dst):
    try:
        return json.dumps(obj)
    except Exception:
        pass

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return res

    content = {}
    for key, value in attrs.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        content[key] = _get_state(value, dst)

    res["content"] = content

    return res


def generic_get_instance(state, src):
    try:
        return json.loads(state)
    except Exception:
        pass

    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")

    # Instead of simply constructing the instance, we use __new__, which
    # bypasses the __init__, and then we set the attributes. This solves
    # the issue of required init arguments.
    instance = cls.__new__(cls)

    content = state.get("content", {})
    if not len(content):
        return instance

    attrs = {}
    for key, value in content.items():
        attrs[key] = _get_instance(value, src)

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def reduce_get_state(obj, dst):
    # This method is for objects for which we have to use the __reduce__
    # method to get the state.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }

    # We get the output of __reduce__ and use it to reconstruct the object.
    # For security reasons, we don't save the constructor object returned by
    # __reduce__, and instead use the pre-defined constructor for the object
    # that we know. This avoids having a function such as `eval()` as the
    # "constructor", abused by attackers.
    #
    # We can/should also look into removing __reduce__ from scikit-learn,
    # and that is not impossible. Most objects which use this don't really
    # need it.
    #
    # More info on __reduce__:
    # https://docs.python.org/3/library/pickle.html#object.__reduce__
    #
    # As a good example, this makes Tree object to be serializable.
    reduce = obj.__reduce__()
    res["__reduce__"] = {}
    res["__reduce__"]["args"] = get_state(reduce[1], dst)

    if len(reduce) == 3:
        # reduce includes what's needed for __getstate__ and we don't need to
        # call __getstate__ directly.
        attrs = reduce[2]
    elif hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    else:
        attrs = obj.__dict__

    content = {}
    for key, value in attrs.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        content[key] = _get_state(value, dst)

    res["content"] = content

    return res


def reduce_get_instance(state, src, constructor):
    state.pop("__class__")
    state.pop("__module__")

    reduce = state.pop("__reduce__")
    args = get_instance(reduce["args"], src)
    instance = constructor(*args)

    content = state["content"]
    attrs = {}
    for key, value in content.items():
        attrs[key] = _get_instance(value, src)

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def Tree_get_instance(state, src):
    return reduce_get_instance(state, src, Tree)


def bunch_get_instance(state, src):
    # Bunch is just a wrapper for dict
    content = dict_get_instance(state, src)
    return Bunch(**content)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (Tree, reduce_get_state),
    (object, generic_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (Tree, Tree_get_instance),
    (Bunch, bunch_get_instance),
    (object, generic_get_instance),
]
