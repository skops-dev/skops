import io
import json
import operator
import re
from contextlib import suppress
from zipfile import ZipFile

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from skops.io import dumps, get_untrusted_types
from skops.io._audit import (
    CachedNode,
    Node,
    audit_tree,
    check_type,
    get_tree,
    temp_setattr,
)
from skops.io._general import (
    DictNode,
    JsonNode,
    MethodNode,
    ObjectNode,
    OperatorFuncNode,
    dict_get_state,
    method_get_state,
    operator_func_get_state,
)
from skops.io._utils import LoadContext, SaveContext, get_state, gettype


class CustomType:
    """A custom untrusted class."""

    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize(
    "module_name, type_name, trusted, expected",
    [
        ("sklearn", "Pipeline", ["sklearn.Pipeline"], True),
        ("sklearn", "Pipeline", ["sklearn.preprocessing.StandardScaler"], False),
        ("builtins", "int", ["builtins.int"], True),
        ("builtins", "int", [], False),
    ],
    ids=["list-True", "list-False", "int-True", "int-False"],
)
def test_check_type(module_name, type_name, trusted, expected):
    assert check_type(module_name, type_name, trusted) == expected


def test_audit_tree_untrusted():
    var = {"a": CustomType(1), 2: CustomType(2)}
    state = dict_get_state(var, SaveContext(None, 0, {}))
    load_context = LoadContext(None, -1)

    node = DictNode(state, load_context, trusted=None)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Untrusted types found in the file: ['test_audit.CustomType']."
        ),
    ):
        audit_tree(node, None)

    # there shouldn't be an error with trusted=everything
    node = DictNode(state, LoadContext(None, -1), trusted=["test_audit.CustomType"])
    audit_tree(node, None)

    untrusted_list = get_untrusted_types(data=dumps(var))
    assert untrusted_list == ["test_audit.CustomType"]

    # passing the type would fix it.
    node = DictNode(state, LoadContext(None, -1), trusted=untrusted_list)
    audit_tree(node, None)


def test_audit_tree_defaults():
    # test that the default types are trusted
    var = {"a": 1, 2: "b"}
    state = dict_get_state(var, SaveContext(None, 0, {}))
    node = DictNode(state, LoadContext(None, -1), trusted=None)
    audit_tree(node, None)


@pytest.mark.parametrize(
    "trusted, defaults, expected",
    [
        (None, int, ["builtins.int"]),
        ([int], None, ["builtins.int"]),
    ],
    ids=["untrusted", "untrusted_list"],
)
def test_Node_get_trusted(trusted, defaults, expected):
    assert Node._get_trusted(trusted, defaults) == expected


@pytest.mark.parametrize(
    "values, is_safe",
    [
        ([1, 2], True),
        ([1, {1: 2}], True),
        ([1, {1: CustomType(1)}], False),
        (eval, False),
        (pytest.mark.parametrize, False),
    ],
    ids=["int", "dict", "untrusted", "eval", "parametrize"],
)
def test_list_safety(values, is_safe):
    content = dumps(values)

    with ZipFile(io.BytesIO(content), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(
            schema, load_context=LoadContext(src=zip_file, protocol=-1), trusted=False
        )
        assert tree.is_safe() == is_safe


def test_gettype_error():
    msg = "Object None of module test is unknown"
    with pytest.raises(ValueError, match=msg):
        gettype(module_name="test", cls_or_func=None)

    msg = "Object test of module None is unknown"
    with pytest.raises(ValueError, match=msg):
        gettype(module_name=None, cls_or_func="test")

    # ImportError if the module cannot be imported
    with pytest.raises(ImportError):
        gettype(module_name="invalid-module", cls_or_func="invalid-type")


@pytest.mark.parametrize(
    "data, file, exception, message",
    [
        ("not-none", "not-none", ValueError, "Only one of data or file"),
        (None, None, ValueError, "Exactly one of data or file should be passed"),
        ("string", None, TypeError, "a bytes-like object is required, not 'str'"),
    ],
    ids=["both", "neither", "string-data"],
)
def test_get_untrusted_types_validation(data, file, exception, message):
    with pytest.raises(exception, match=message):
        get_untrusted_types(data=data, file=file)


def test_temp_setattr():
    # Test that temp_setattr works as expected
    class A:
        def __init__(self):
            self.a = 1

    temp = A()
    with suppress(ValueError):
        with temp_setattr(temp, a=2, b=3):
            assert temp.a == 2
            assert temp.b == 3
            raise ValueError  # to make sure context manager handles exceptions

    assert temp.a == 1
    assert not hasattr(temp, "b")


def test_format_object_node():
    estimator = LogisticRegression(random_state=0, solver="liblinear")
    state = get_state(estimator, SaveContext(None))
    node = ObjectNode(state, LoadContext(None, -1))
    expected = "sklearn.linear_model._logistic.LogisticRegression"
    assert node.format() == expected


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("hello", 'json-type("hello")'),
        (123, "json-type(123)"),
        (0.456, "json-type(0.456)"),
        (True, "json-type(true)"),
        (False, "json-type(false)"),
        (None, "json-type(null)"),
    ],
)
def test_format_json_node(inp, expected):
    state = get_state(inp, SaveContext(None))
    node = JsonNode(state, LoadContext(None, -1))
    assert node.format() == expected


def test_method_node_invalid_state():
    # Test that MethodNode raises a ValueError if the state is invalid.
    # The __class__ and __module__ should match what's inside the content.
    var = FunctionTransformer().fit
    state = method_get_state(var, SaveContext(None, 0, {}))
    state["content"]["obj"]["__class__"] = "foo"
    load_context = LoadContext(None, -1)

    with pytest.raises(ValueError, match="Expected object of type"):
        MethodNode(state, load_context, trusted=None)


def test_operator_func_node_invalid_state():
    var = operator.methodcaller("fit")
    state = operator_func_get_state(var, SaveContext(None, 0, {}))
    state["__module__"] = "foo"
    load_context = LoadContext(None, -1)

    with pytest.raises(ValueError, match="Expected module 'operator'"):
        OperatorFuncNode(state, load_context, trusted=None)


def _tamper_bit_generator_name(data: bytes, new_name: str) -> bytes:
    """Rewrite the bit generator class name inside a dumped Generator file."""
    src = ZipFile(io.BytesIO(data))
    schema = json.loads(src.read("schema.json"))
    schema["content"]["bit_generator"]["content"]["bit_generator"]["content"] = (
        json.dumps(new_name)
    )
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as out:
        for name in src.namelist():
            if name == "schema.json":
                out.writestr(name, json.dumps(schema))
            else:
                out.writestr(name, src.read(name))
    return buffer.getvalue()


def test_random_generator_smuggled_bit_generator_is_surfaced():
    # Non-regression test: a Generator's bit generator is identified in the file
    # by a class name that gets resolved to a numpy.random attribute and called
    # at load time. That name must be surfaced to the audit, otherwise
    # get_untrusted_types() reports [] for a file that would have skops call an
    # attacker-chosen numpy.random attribute.
    np = pytest.importorskip("numpy")

    data = dumps(np.random.default_rng(42))
    # sanity check: a genuine Generator is fully trusted by default
    assert get_untrusted_types(data=data) == []

    evil = _tamper_bit_generator_name(data, "set_bit_generator")
    assert get_untrusted_types(data=evil) == ["numpy.random.set_bit_generator"]

    # loading without trusting it must fail before the callable is invoked, so
    # the process-global numpy RNG must be left untouched.
    state_before = np.random.get_state()
    from skops.io import loads
    from skops.io.exceptions import UntrustedTypesFoundException

    with pytest.raises(UntrustedTypesFoundException):
        loads(evil)
    assert np.random.get_state()[0] == state_before[0]


def test_random_generator_missing_name_is_rejected():
    # If the bit generator name can't be found where a genuine file always puts
    # it, the file is corrupted/tampered: we refuse it with a clear error rather
    # than reporting a non-actionable "unknown" untrusted type.
    np = pytest.importorskip("numpy")

    from skops.io import loads

    data = dumps(np.random.default_rng(42))
    # drop the bit generator name from the serialized state
    src = ZipFile(io.BytesIO(data))
    schema = json.loads(src.read("schema.json"))
    del schema["content"]["bit_generator"]["content"]["bit_generator"]
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as out:
        for name in src.namelist():
            if name == "schema.json":
                out.writestr(name, json.dumps(schema))
            else:
                out.writestr(name, src.read(name))
    broken = buffer.getvalue()

    with pytest.raises(ValueError, match="Could not find the bit generator name"):
        get_untrusted_types(data=broken)
    with pytest.raises(ValueError, match="Could not find the bit generator name"):
        loads(broken)


def test_random_generator_construct_rejects_non_bit_generator():
    # Defense in depth: even if the smuggled name is explicitly trusted, the
    # node must refuse to instantiate anything that is not a real bit generator
    # instead of calling an arbitrary numpy.random attribute.
    np = pytest.importorskip("numpy")

    from skops.io import loads

    data = dumps(np.random.default_rng(42))
    evil = _tamper_bit_generator_name(data, "set_bit_generator")

    state_before = np.random.get_state()
    with pytest.raises(ValueError, match="Expected a numpy.random bit generator"):
        loads(evil, trusted=["numpy.random.set_bit_generator"])
    assert np.random.get_state()[0] == state_before[0]


def test_cached_node_resolves_to_memoized_node():
    # A "CachedNode" state refers, through __id__, to a node that was already
    # loaded in the same LoadContext. Constructing it yields the cached object.
    load_context = LoadContext(None, -1)
    state = get_state([1, 2], SaveContext(None))
    node = get_tree(state, load_context, trusted=None)

    cached_state = {
        "__class__": "list",
        "__module__": "builtins",
        "__loader__": "CachedNode",
        "__id__": state["__id__"],
    }
    cached_node = CachedNode(cached_state, load_context, trusted=None)
    assert cached_node.cached is node
    assert cached_node.construct() == [1, 2]

    # get_tree short-circuits on an already memoized __id__ and hands back the
    # original node instead of building a CachedNode.
    assert get_tree(cached_state, load_context, trusted=None) is node
