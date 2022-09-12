import inspect
import json

from sklearn.base import BaseEstimator
from sklearn.calibration import _CalibratedClassifier
from sklearn.tree._tree import Tree
from sklearn.utils import Bunch

from ._general import dict_get_instance
from ._utils import get_instance, get_module, get_state, gettype, try_get_state


def BaseEstimator_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    else:
        attrs = obj.__dict__

    reduce = False
    if hasattr(obj, "__reduce__"):
        # We get the output of __reduce__ and use it to reconstruct the object.
        # This is very insecure by itself, since the constructor can be `eval`
        # and the args can be an arbitrary code. Therefore we should never load
        # it unless the user explicitly allows it.
        #
        # We can/should also look into removing __reduce__ from scikit-learn,
        # and that is not impossible. Most objects which use this don't really
        # need it.
        #
        # More info on __reduce__:
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        #
        # Crucially, this makes Tree object to be serializable.
        reduce = obj.__reduce__()
        if inspect.getmodule(reduce[0]) != inspect.getmodule(type(obj)):
            # only use reduce if the constructor is in the same module as the
            # object.
            reduce = False

    if reduce is not False:
        res["__reduce__"] = {}
        res["__reduce__"]["constructor"] = get_state(reduce[0], dst)
        res["__reduce__"]["args"] = get_state(reduce[1], dst)

        if len(reduce) == 3:
            # reduce includes what's needed for __getstate__ and we overwrite
            # what we had before
            attrs = reduce[2]

    content = {}
    for key, value in attrs.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        content[key] = try_get_state(value, dst)

    res["content"] = content

    return res


def BaseEstimator_get_instance(state, src):
    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")

    if "__reduce__" in state:
        # If the state has a "__reduce__" key, it includes the method which
        # creates the object, and the args which should be passed to it.
        reduce = state.pop("__reduce__")
        constructor = get_instance(reduce["constructor"], src)
        args = get_instance(reduce["args"], src)
        instance = constructor(*args)
    else:
        # Instead of simply constructing the instance, we use __new__, which
        # bypasses the __init__, and then we set the attributes. This solves
        # the issue of required init arguments.
        instance = cls.__new__(cls)

    content = state["content"]
    attrs = {}
    for key, value in content.items():
        if value is None:
            attrs[key] = None
        elif isinstance(value, dict):
            attrs[key] = get_instance(value, src)
        else:
            attrs[key] = json.loads(value)

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def bunch_get_instance(state, src):
    # Bunch is just a wrapper for dict
    content = dict_get_instance(state, src)
    return Bunch(**content)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (Tree, BaseEstimator_get_state),
    (_CalibratedClassifier, BaseEstimator_get_state),
    (BaseEstimator, BaseEstimator_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (Tree, BaseEstimator_get_instance),
    (Bunch, bunch_get_instance),
    (_CalibratedClassifier, BaseEstimator_get_instance),
    (BaseEstimator, BaseEstimator_get_instance),
]
