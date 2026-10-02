"""Persistence tests for old versions of the protocol"""

from __future__ import annotations

import io
import json
import operator
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import special
from sklearn.preprocessing import FunctionTransformer

from skops.io import dumps, get_untrusted_types, loads
from skops.io._audit import get_tree
from skops.io._general import (
    DictNode,
    ListNode,
    OperatorFuncNode,
    SetNode,
    dict_get_state,
    list_get_state,
    operator_func_get_state,
    set_get_state,
)
from skops.io._utils import SaveContext, get_module, get_state, read_schema
from skops.io.exceptions import UntrustedTypesFoundException
from skops.io.old._general_v2 import DictNode as DictNodeV2
from skops.io.old._general_v2 import ListNode as ListNodeV2
from skops.io.old._general_v2 import OperatorFuncNode as OperatorFuncNodeV2
from skops.io.old._general_v2 import SetNode as SetNodeV2
from skops.io.tests._utils import (
    assert_method_outputs_equal,
    assert_params_equal,
    downgrade_state,
    make_load_context,
)


def dummy_func(X):
    return X


def test_downgrade_state_assigns_unused_id():
    # The __id__ given to the downgraded node must not be in use by another
    # node of the file, otherwise both are loaded as the same object. Using the
    # id() of a new object is not enough: the ids in the file belong to objects
    # of dump time, some of which have been freed since.
    dumped = dumps(FunctionTransformer(func=np.sqrt))
    old_state = {
        "__class__": "ufunc",
        "__module__": "numpy",
        "__loader__": "FunctionNode",
        "content": {"module_path": "numpy", "function": "sqrt"},
    }
    downgraded = downgrade_state(
        data=dumped,
        keys=["content", "content", "func"],
        old_state=old_state,
        protocol=0,
    )
    with ZipFile(io.BytesIO(downgraded), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    func_state = schema["content"]["content"]["func"]

    # the ids of all other nodes; shared objects like None legitimately repeat
    other_ids: set[int] = set()

    def collect(state):
        if state is func_state:
            return
        if isinstance(state, dict):
            if "__id__" in state:
                other_ids.add(state["__id__"])
            for value in state.values():
                collect(value)
        elif isinstance(state, list):
            for value in state:
                collect(value)

    collect(schema)
    assert func_state["__id__"] not in other_ids


@pytest.fixture
def save_context():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        yield SaveContext(zip_file=zip_file)


#############
# VERSION 0 #
#############


@pytest.mark.parametrize("func", [np.sqrt, len, special.exp10, dummy_func])
def test_persist_function_v0(func):
    call_count = 0

    # function_get_state as it was for protocol 0
    def old_function_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(obj),
            "__loader__": "FunctionNode",
            "content": {
                "module_path": get_module(obj),
                "function": obj.__name__,
            },
        }
        return res

    estimator = FunctionTransformer(func=func)
    X, y = [0, 1], [2, 3]
    estimator.fit(X, y)

    dumped = dumps(estimator)
    # importent: downgrade the state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=["content", "content", "func"],
        old_state=old_function_get_state(func, None),
        protocol=0,
    )
    loaded = loads(downgraded, trusted=get_untrusted_types(data=downgraded))

    # sanity check: ensure that the old get_state function was really called
    assert call_count == 1

    # check that loaded estimator is identical
    assert_params_equal(estimator.__dict__, loaded.__dict__)
    assert_method_outputs_equal(estimator, loaded, X)


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
    ids=["default_rng", "Generator"],
)
def test_random_generator_v0(rng):
    call_count = 0

    # random_generator_get_state as it was for protocol 0
    def old_random_generator_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        bit_generator_state = obj.bit_generator.state
        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {"bit_generator": bit_generator_state},
        }
        return res

    rng.random(123)  # move RNG forwards
    dumped = dumps(rng)
    # importent: downgrade the whole state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=None,
        old_state=old_random_generator_get_state(rng, None),
        protocol=0,
    )

    # old loader only worked with trusted=True, see #329
    # update: we have removed trusted=True, so this doesn't work anymore.
    with pytest.raises(AttributeError):
        loads(downgraded, trusted=[])


#############
# VERSION 1 #
#############


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
    ids=["default_rng", "Generator"],
)
def test_random_generator_v1(save_context, rng):
    call_count = 0

    # random_generator_get_state as it was for protocol 0
    def old_random_generator_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        bit_generator_state = get_state(obj.bit_generator.state, save_context)
        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {"bit_generator": bit_generator_state},
        }
        return res

    rng.random(123)  # move RNG forwards
    dumped = dumps(rng)
    # importent: downgrade the whole state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=None,
        old_state=old_random_generator_get_state(rng, save_context),
        protocol=1,
    )

    loads(downgraded, trusted=[])


def _dump_v1_generator(save_context, rng) -> bytes:
    """Produce a protocol-1 ``RandomGeneratorNode`` file for a Generator."""

    def old_random_generator_get_state(obj, save_context):
        return {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {
                "bit_generator": get_state(obj.bit_generator.state, save_context)
            },
        }

    return downgrade_state(
        data=dumps(rng),
        keys=None,
        old_state=old_random_generator_get_state(rng, save_context),
        protocol=1,
    )


def _tamper_v1_bit_generator_name(
    data: bytes, new_name: str | None = None, drop: bool = False
) -> bytes:
    src = ZipFile(io.BytesIO(data))
    schema = json.loads(src.read("schema.json"))
    entry = schema["content"]["bit_generator"]["content"]
    if drop:
        del entry["bit_generator"]
    else:
        entry["bit_generator"]["content"] = json.dumps(new_name)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as out:
        for name in src.namelist():
            if name == "schema.json":
                out.writestr(name, json.dumps(schema))
            else:
                out.writestr(name, src.read(name))
    return buffer.getvalue()


# The current RandomGeneratorNode is hardened against a bit generator name read
# from the file being resolved and called as a numpy.random attribute (see
# test_audit.py). The protocol-1 reader shares that shape and the same fix, so
# it gets the same checks.
def test_random_generator_v1_smuggled_bit_generator_is_surfaced(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    evil = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), "set_bit_generator"
    )
    assert get_untrusted_types(data=evil) == ["numpy.random.set_bit_generator"]
    with pytest.raises(UntrustedTypesFoundException):
        loads(evil)


def test_random_generator_v1_construct_rejects_non_bit_generator(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    evil = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), "set_bit_generator"
    )
    with pytest.raises(ValueError, match="Expected a numpy.random bit generator"):
        loads(evil, trusted=["numpy.random.set_bit_generator"])


def test_random_generator_v1_missing_name_is_rejected(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    broken = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), drop=True
    )
    with pytest.raises(ValueError, match="Could not find the bit generator name"):
        get_untrusted_types(data=broken)


def test_random_generator_v1_wrong_child_type_is_rejected(save_context):
    # As for the current node (see test_audit.py), the bit generator state has
    # to be a DictNode, and a file with any other node there is refused before
    # the audit.
    rng = np.random.default_rng(42)
    slice_state = {
        "__class__": "slice",
        "__module__": "builtins",
        "__loader__": "SliceNode",
        "content": {"start": None, "stop": None, "step": None},
    }
    broken = downgrade_state(
        data=_dump_v1_generator(save_context, rng),
        keys=["content", "bit_generator"],
        old_state=slice_state,
        protocol=1,
    )
    msg = "Expected a node of type DictNode, got SliceNode"
    with pytest.raises(ValueError, match=msg):
        get_untrusted_types(data=broken)
    with pytest.raises(ValueError, match=msg):
        loads(broken)


#############
# VERSION 2 #
#############


@pytest.mark.parametrize(
    "func, arg",
    [
        (operator.attrgetter("real"), 3.5),
        (operator.itemgetter(1), "banana"),
        (operator.methodcaller("replace", "a", "b"), "banana"),
    ],
    ids=["attrgetter", "itemgetter", "methodcaller"],
)
@pytest.mark.parametrize("protocol", [0, 1, 2])
def test_operator_func_v2(save_context, func, arg, protocol):
    # Up to protocol 2 an OperatorFuncNode state had no "kwargs" entry. Such
    # files, whichever of these protocols they were written with, are read by
    # the protocol-2 node in skops.io.old, and load and behave as before.

    # operator_func_get_state as it was for protocol 2
    def old_operator_func_get_state(obj, save_context):
        _, attrs = obj.__reduce__()
        return {
            "__class__": obj.__class__.__name__,
            "__module__": "operator",
            "__loader__": "OperatorFuncNode",
            "attrs": get_state(attrs, save_context),
        }

    downgraded = downgrade_state(
        data=dumps(func),
        keys=None,
        old_state=old_operator_func_get_state(func, save_context),
        protocol=protocol,
    )
    with ZipFile(io.BytesIO(downgraded)) as zip_file:
        schema, load_context = read_schema(zip_file)
        node = get_tree(schema, load_context, trusted=None)
    assert isinstance(node, OperatorFuncNodeV2)

    type_name = f"operator.{type(func).__name__}"
    assert get_untrusted_types(data=downgraded) == [type_name]
    loaded = loads(downgraded, trusted=[type_name])
    assert loaded(arg) == func(arg)


def test_operator_func_current_requires_kwargs(save_context):
    # The current OperatorFuncNode requires the "kwargs" entry added in
    # protocol 3, so it cannot read a protocol-2 state; those go through the
    # old node instead, see test_operator_func_v2.
    state = operator_func_get_state(operator.methodcaller("upper"), save_context)
    del state["kwargs"]
    with pytest.raises(KeyError, match="kwargs"):
        OperatorFuncNode(state, make_load_context(), trusted=None)


class TaggedDict(dict):
    """Dict subclass whose constructor sets an attribute."""

    def __init__(self, items=()):
        super().__init__(items)
        self.tagged = True


class TaggedList(list):
    """List subclass whose constructor sets an attribute."""

    def __init__(self, items):
        super().__init__(items)
        self.tagged = True


class TaggedSet(set):
    """Set subclass whose constructor sets an attribute."""

    def __init__(self, items):
        super().__init__(items)
        self.tagged = True


CONTAINER_SUBCLASS_CASES = [
    pytest.param(
        TaggedDict, {"a": 1, "b": 2}, dict_get_state, DictNode, DictNodeV2, id="dict"
    ),
    pytest.param(
        TaggedList, [1, 2, 3], list_get_state, ListNode, ListNodeV2, id="list"
    ),
    pytest.param(TaggedSet, {1, 2, 3}, set_get_state, SetNode, SetNodeV2, id="set"),
]


@pytest.mark.parametrize(
    "container_type, items, get_state_func, node_cls, old_node_cls",
    CONTAINER_SUBCLASS_CASES,
)
@pytest.mark.parametrize("protocol", [0, 1, 2])
def test_container_subclass_v2(
    save_context,
    container_type,
    items,
    get_state_func,
    node_cls,
    old_node_cls,
    protocol,
):
    # Up to protocol 2 the state of a dict, list or set had no "attrs" entry,
    # and an instance of a subclass was built through its constructor. Such
    # files, whichever of these protocols they were written with, are read by
    # the protocol-2 nodes in skops.io.old and load as before, here with the
    # attribute the constructor sets.
    obj = container_type(items)
    # the state as it was for protocol 2
    old_state = get_state_func(obj, save_context)
    del old_state["attrs"]
    downgraded = downgrade_state(
        data=dumps(obj), keys=None, old_state=old_state, protocol=protocol
    )
    with ZipFile(io.BytesIO(downgraded)) as zip_file:
        schema, load_context = read_schema(zip_file)
        node = get_tree(schema, load_context, trusted=None)
    assert isinstance(node, old_node_cls)

    type_name = f"{get_module(container_type)}.{container_type.__name__}"
    assert get_untrusted_types(data=downgraded) == [type_name]
    loaded = loads(downgraded, trusted=[type_name])
    assert type(loaded) is container_type
    assert loaded == obj
    assert loaded.tagged is True


@pytest.mark.parametrize(
    "obj, get_state_func, old_node_cls",
    [
        pytest.param({"a": 1, "b": 2}, dict_get_state, DictNodeV2, id="dict"),
        pytest.param([1, 2, 3], list_get_state, ListNodeV2, id="list"),
        pytest.param({1, 2, 3}, set_get_state, SetNodeV2, id="set"),
    ],
)
@pytest.mark.parametrize("protocol", [0, 1, 2])
def test_plain_container_v2(save_context, obj, get_state_func, old_node_cls, protocol):
    # A plain dict, list or set has no "attrs" entry in any protocol. Files up
    # to protocol 2 are read through the old nodes, which build it as before.
    old_state = get_state_func(obj, save_context)
    assert "attrs" not in old_state
    downgraded = downgrade_state(
        data=dumps(obj), keys=None, old_state=old_state, protocol=protocol
    )
    with ZipFile(io.BytesIO(downgraded)) as zip_file:
        schema, load_context = read_schema(zip_file)
        node = get_tree(schema, load_context, trusted=None)
    assert isinstance(node, old_node_cls)
    loaded = loads(downgraded)
    assert type(loaded) is type(obj)
    assert loaded == obj


@pytest.mark.parametrize(
    "container_type, items, get_state_func, node_cls, old_node_cls",
    CONTAINER_SUBCLASS_CASES,
)
def test_container_subclass_current_does_not_call_constructor(
    save_context, container_type, items, get_state_func, node_cls, old_node_cls
):
    # The current nodes create the instance with __new__ and restore its
    # attributes from the "attrs" entry added in protocol 3. Without the entry
    # the constructor is not called either, which is why a protocol-2 state
    # goes through the old node instead, see test_container_subclass_v2.
    obj = container_type(items)
    state = get_state_func(obj, save_context)
    del state["attrs"]
    loaded = node_cls(state, make_load_context(), trusted=None).construct()
    assert type(loaded) is container_type
    assert loaded == obj
    assert not hasattr(loaded, "tagged")
