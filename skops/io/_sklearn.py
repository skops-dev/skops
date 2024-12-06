from __future__ import annotations

from typing import Any, Optional, Sequence, Type

from sklearn.cluster import Birch
from sklearn.tree._tree import Tree

from ._audit import Node, get_tree
from ._general import TypeNode, unsupported_get_state
from ._protocol import PROTOCOL
from ._utils import LoadContext, SaveContext, get_module, get_state, gettype
from .exceptions import UnsupportedTypeException

try:
    # TODO: remove once support for sklearn<1.2 is dropped. See #187
    from sklearn.covariance._graph_lasso import _DictWithDeprecatedKeys
except ImportError:
    _DictWithDeprecatedKeys = None

from sklearn.linear_model._sgd_fast import (
    EpsilonInsensitive,
    Hinge,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
)

ALLOWED_LOSSES = {
    EpsilonInsensitive,
    Hinge,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
}

try:
    # TODO: remove once support for sklearn<1.6 is dropped.
    from sklearn.linear_model._sgd_fast import (
        Huber,
        Log,
        SquaredLoss,
    )

    ALLOWED_LOSSES |= {
        Huber,
        Log,
        SquaredLoss,
    }
except ImportError:
    pass

try:
    # sklearn>=1.6
    from sklearn._loss._loss import (
        CyAbsoluteError,
        CyExponentialLoss,
        CyHalfBinomialLoss,
        CyHalfGammaLoss,
        CyHalfMultinomialLoss,
        CyHalfPoissonLoss,
        CyHalfSquaredError,
        CyHalfTweedieLoss,
        CyHalfTweedieLossIdentity,
        CyHuberLoss,
        CyPinballLoss,
    )

    ALLOWED_LOSSES |= {
        CyAbsoluteError,
        CyExponentialLoss,
        CyHalfBinomialLoss,
        CyHalfGammaLoss,
        CyHalfMultinomialLoss,
        CyHalfPoissonLoss,
        CyHalfSquaredError,
        CyHalfTweedieLoss,
        CyHalfTweedieLossIdentity,
        CyHuberLoss,
        CyPinballLoss,
    }
except ImportError:
    pass

# This import is for the parent class of all loss functions, which is used to
# set the dispatch function for all loss functions.
try:
    # From sklearn>=1.5
    from sklearn._loss._loss import CyLossFunction
except ImportError:
    CyLossFunction = None


try:
    # sklearn<1.6
    from sklearn.linear_model._sgd_fast import LossFunction
except ImportError:
    LossFunction = None


UNSUPPORTED_TYPES = {Birch}


def reduce_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
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
    reduced = obj.__reduce__()
    res["__reduce__"] = {}
    res["__reduce__"]["args"] = get_state(reduced[1], save_context)

    if len(reduced) == 3:
        # reduce includes what's needed for __getstate__ and we don't need to
        # call __getstate__ directly.
        attrs = reduced[2]
    elif hasattr(obj, "__getstate__"):
        # since python311 __getstate__ is defined for `object` and might return
        # None
        attrs = obj.__getstate__() or {}
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        attrs = {}

    if not isinstance(attrs, (dict, tuple)):
        raise UnsupportedTypeException(
            f"Objects of type {res['__class__']} not supported yet"
        )

    res["content"] = get_state(attrs, save_context)
    return res


class ReduceNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        constructor: Type[Any],
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        reduce = state["__reduce__"]
        self.children = {
            "attrs": get_tree(state["content"], load_context, trusted=trusted),
            "args": get_tree(reduce["args"], load_context, trusted=trusted),
            "constructor": TypeNode(
                {
                    "__class__": constructor.__name__,
                    "__module__": get_module(constructor),
                    "__id__": id(constructor),
                },
                load_context,
                trusted=trusted,
            ),
        }

    def _construct(self):
        args = self.children["args"].construct()
        constructor = gettype(
            self.children["constructor"].module_name,
            self.children["constructor"].class_name,
        )
        instance = constructor(*args)
        attrs = self.children["attrs"].construct()
        if not attrs:
            # nothing more to do
            return instance

        if isinstance(args, tuple) and not hasattr(instance, "__setstate__"):
            raise UnsupportedTypeException(
                f"Objects of type {constructor} are not supported yet"
            )

        if hasattr(instance, "__setstate__"):
            instance.__setstate__(attrs)
        elif isinstance(attrs, dict):
            instance.__dict__.update(attrs)
        else:
            # we (probably) got tuple attrs but cannot setstate with them
            raise UnsupportedTypeException(
                f"Objects of type {constructor} are not supported yet"
            )

        return instance


def tree_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    state = reduce_get_state(obj, save_context)
    state["__loader__"] = "TreeNode"
    return state


class TreeNode(ReduceNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        self.trusted = self._get_trusted(trusted, [get_module(Tree) + ".Tree"])
        super().__init__(state, load_context, constructor=Tree, trusted=self.trusted)


def loss_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    reduce = obj.__reduce__()
    if type(obj) == reduce[0]:
        state = reduce_get_state(obj, save_context)
        state["__loader__"] = "LossNode"
    elif type(obj) == reduce[1][0]:
        # the output is of the form:
        # >>> CyPinballLoss(1).__reduce__()
        # (<cyfunction __pyx_unpickle_CyPinballLoss at 0x7b1d00099ff0>,
        #             (<class '_loss.CyPinballLoss'>, 232784418, (1.0,)))
        state = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "LossNode",
        }
        state["__reduce__"] = {}
        state["__reduce__"]["args"] = get_state(reduce[1][2], save_context)
        state["content"] = get_state({}, save_context)

    return state


class LossNode(ReduceNode):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        # TODO: make sure trusted here makes sense and used.
        self.trusted = self._get_trusted(
            trusted, [get_module(x) + "." + x.__name__ for x in ALLOWED_LOSSES]
        )
        super().__init__(
            state,
            load_context,
            constructor=gettype(state["__module__"], state["__class__"]),
            trusted=self.trusted,
        )


# TODO: remove once support for sklearn<1.2 is dropped.
def _DictWithDeprecatedKeys_get_state(
    obj: Any, save_context: SaveContext
) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "_DictWithDeprecatedKeysNode",
    }
    content = {}
    # explicitly pass a dict object instead of _DictWithDeprecatedKeys and
    # later construct a _DictWithDeprecatedKeys object.
    content["main"] = get_state(dict(obj), save_context)
    content["_deprecated_key_to_new_key"] = get_state(
        obj._deprecated_key_to_new_key, save_context
    )
    res["content"] = content
    return res


# TODO: remove once support for sklearn<1.2 is dropped.
class _DictWithDeprecatedKeysNode(Node):
    # _DictWithDeprecatedKeys is just a wrapper for dict
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = [
            get_module(_DictWithDeprecatedKeysNode) + "._DictWithDeprecatedKeys"
        ]
        self.children = {
            "main": get_tree(state["content"]["main"], load_context, trusted=trusted),
            "_deprecated_key_to_new_key": get_tree(
                state["content"]["_deprecated_key_to_new_key"],
                load_context,
                trusted=trusted,
            ),
        }

    def _construct(self):
        instance = _DictWithDeprecatedKeys(**self.children["main"].construct())
        instance._deprecated_key_to_new_key = self.children[
            "_deprecated_key_to_new_key"
        ].construct()
        return instance


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (Tree, tree_get_state),
]

if LossFunction is not None:
    GET_STATE_DISPATCH_FUNCTIONS.append((LossFunction, loss_get_state))

if CyLossFunction is not None:
    GET_STATE_DISPATCH_FUNCTIONS.append((CyLossFunction, loss_get_state))

for type_ in UNSUPPORTED_TYPES:
    GET_STATE_DISPATCH_FUNCTIONS.append((type_, unsupported_get_state))

# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING: dict[tuple[str, int], Any] = {
    ("LossNode", PROTOCOL): LossNode,
    ("TreeNode", PROTOCOL): TreeNode,
}

# TODO: remove once support for sklearn<1.2 is dropped.
# Starting from sklearn 1.2, _DictWithDeprecatedKeys is removed as it's no
# longer needed for GraphicalLassoCV, see #187.
if _DictWithDeprecatedKeys is not None:
    GET_STATE_DISPATCH_FUNCTIONS.append(
        (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_state)
    )
    NODE_TYPE_MAPPING[
        ("_DictWithDeprecatedKeysNode", PROTOCOL)
    ] = _DictWithDeprecatedKeysNode  # type: ignore
