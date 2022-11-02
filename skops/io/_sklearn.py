from __future__ import annotations

from typing import Any

from sklearn.cluster import Birch

try:
    # TODO: remove once support for sklearn<1.2 is dropped. See #187
    from sklearn.covariance._graph_lasso import _DictWithDeprecatedKeys
except ImportError:
    _DictWithDeprecatedKeys = None
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

from ._dispatch import Node, get_tree
from ._general import dict_get_state, unsupported_get_state
from ._utils import SaveState, get_module, get_state, gettype
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


def reduce_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
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
    res["__reduce__"]["args"] = get_state(reduce[1], save_state)

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

    res["content"] = get_state(attrs, save_state)
    return res


class ReduceNode(Node):
    def __init__(self, state, src, constructor=None, trusted=False):
        super().__init__(state, src, trusted)
        reduce = state["__reduce__"]
        self.args = get_tree(reduce["args"], src)
        self.constructor = constructor
        self.attrs = get_tree(state["content"], src)

    def construct(self):
        args = self.args.construct()
        instance = self.constructor(*args)
        attrs = self.attrs.construct()
        if not attrs:
            # nothing more to do
            return instance

        if isinstance(args, tuple) and not hasattr(instance, "__setstate__"):
            raise UnsupportedTypeException(
                f"Objects of type {self.constructor} are not supported yet"
            )

        if hasattr(instance, "__setstate__"):
            instance.__setstate__(attrs)
        else:
            instance.__dict__.update(attrs)

        return instance


def tree_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    state = reduce_get_state(obj, save_state)
    state["__loader__"] = "TreeNode"
    return state


class TreeNode(ReduceNode):
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, constructor=Tree, trusted=trusted)


def sgd_loss_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    state = reduce_get_state(obj, save_state)
    state["__loader__"] = "SGDNode"
    return state


class SGDNode(ReduceNode):
    def __init__(self, state, src, trusted=False):
        # TODO: make sure trusted here makes sense and used.
        super().__init__(
            state,
            src,
            constructor=gettype(state.get("__module__"), state.get("__class__")),
            trusted=ALLOWED_SGD_LOSSES,
        )


# TODO: remove once support for sklearn<1.2 is dropped.
def _DictWithDeprecatedKeys_get_state(
    obj: Any, save_state: SaveState
) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "_DictWithDeprecatedKeysNode",
    }
    content = {}
    # explicitly pass a dict object instead of _DictWithDeprecatedKeys and
    # later construct a _DictWithDeprecatedKeys object.
    content["main"] = dict_get_state(dict(obj), save_state)
    content["_deprecated_key_to_new_key"] = dict_get_state(
        obj._deprecated_key_to_new_key, save_state
    )
    res["content"] = content
    return res


# TODO: remove once support for sklearn<1.2 is dropped.
class _DictWithDeprecatedKeysNode(Node):
    # _DictWithDeprecatedKeys is just a wrapper for dict
    def __init__(self, state, src, trusted=False):
        super().__init__(state, src, trusted)
        self.main = get_tree(state["content"]["main"], src)
        self._deprecated_key_to_new_key = get_tree(
            state["content"]["_deprecated_key_to_new_key"], src
        )

    def construct(self):
        instance = _DictWithDeprecatedKeys(**self.main.construct())
        instance._deprecated_key_to_new_key = (
            self._deprecated_key_to_new_key.construct()
        )
        return instance


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (LossFunction, sgd_loss_get_state),
    (Tree, tree_get_state),
]
for type_ in UNSUPPORTED_TYPES:
    GET_STATE_DISPATCH_FUNCTIONS.append((type_, unsupported_get_state))

# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    "SGDNode": SGDNode,
    "TreeNode": TreeNode,
}

# TODO: remove once support for sklearn<1.2 is dropped.
# Starting from sklearn 1.2, _DictWithDeprecatedKeys is removed as it's no
# longer needed for GraphicalLassoCV, see #187.
if _DictWithDeprecatedKeys is not None:
    GET_STATE_DISPATCH_FUNCTIONS.append(
        (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_state)
    )
    NODE_TYPE_MAPPING[
        "_DictWithDeprecatedKeysNode"
    ] = _DictWithDeprecatedKeysNode  # type: ignore
