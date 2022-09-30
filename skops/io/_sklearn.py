from sklearn.cluster import Birch
from sklearn.covariance._graph_lasso import _DictWithDeprecatedKeys
from sklearn.linear_model._sgd_fast import (
    EpsilonInsensitive,
    Hinge,
    Huber,
    Log,
    LossFunction,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
    SquaredLoss,
)
from sklearn.tree._tree import Tree
from sklearn.utils import Bunch

from ._general import dict_get_instance, dict_get_state, unsupported_get_state
from ._utils import get_instance, get_module, get_state, gettype
from .exceptions import UnsupportedTypeException

ALLOWED_SGD_LOSSES = {
    ModifiedHuber,
    Hinge,
    SquaredHinge,
    Log,
    SquaredLoss,
    Huber,
    EpsilonInsensitive,
    SquaredEpsilonInsensitive,
}

UNSUPPORTED_TYPES = {Birch}


def reduce_get_state(obj, dst):
    # This method is for objects for which we have to use the __reduce__
    # method to get the state.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
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
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        attrs = {}

    if not isinstance(attrs, dict):
        raise UnsupportedTypeException(
            f"Objects of type {res['__class__']} not supported yet"
        )

    res["content"] = get_state(attrs, dst)
    return res


def reduce_get_instance(state, src, constructor):
    reduce = state["__reduce__"]
    args = get_instance(reduce["args"], src)
    instance = constructor(*args)

    attrs = get_instance(state["content"], src)
    if not attrs:
        # nothing more to do
        return instance

    if isinstance(args, tuple) and not hasattr(instance, "__setstate__"):
        raise UnsupportedTypeException(
            f"Objects of type {constructor} are not supported yet"
        )

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def Tree_get_instance(state, src):
    return reduce_get_instance(state, src, constructor=Tree)


def sgd_loss_get_instance(state, src):
    cls = gettype(state)
    if cls not in ALLOWED_SGD_LOSSES:
        raise UnsupportedTypeException(f"Expected LossFunction, got {cls}")
    return reduce_get_instance(state, src, constructor=cls)


def bunch_get_instance(state, src):
    # Bunch is just a wrapper for dict
    content = dict_get_instance(state, src)
    return Bunch(**content)


def _DictWithDeprecatedKeys_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }
    content = {}
    content["main"] = dict_get_state(obj, dst)
    content["_deprecated_key_to_new_key"] = dict_get_state(
        obj._deprecated_key_to_new_key, dst
    )
    res["content"] = content
    return res


def _DictWithDeprecatedKeys_get_instance(state, src):
    # _DictWithDeprecatedKeys is just a wrapper for dict
    content = dict_get_instance(state["content"]["main"], src)
    deprecated_key_to_new_key = dict_get_instance(
        state["content"]["_deprecated_key_to_new_key"], src
    )
    res = _DictWithDeprecatedKeys(**content)
    res._deprecated_key_to_new_key = deprecated_key_to_new_key
    return res


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (LossFunction, reduce_get_state),
    (Tree, reduce_get_state),
    (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_state),
]
for type_ in UNSUPPORTED_TYPES:
    GET_STATE_DISPATCH_FUNCTIONS.append((type_, unsupported_get_state))

# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (LossFunction, sgd_loss_get_instance),
    (Tree, Tree_get_instance),
    (Bunch, bunch_get_instance),
    (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_instance),
]
