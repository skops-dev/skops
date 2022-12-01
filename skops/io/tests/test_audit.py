import io
import json
import re
from contextlib import suppress
from zipfile import ZipFile

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from skops.io import dumps, get_untrusted_types
from skops.io._audit import audit_tree, check_type
from skops.io._dispatch import Node, get_tree, temp_setattr
from skops.io._general import DictNode, dict_get_state
from skops.io._utils import LoadContext, SaveContext, gettype


class CustomType:
    """A custom untrusted class."""

    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize(
    "module_name, type_name, trusted, expected",
    [
        ("sklearn", "Pipeline", ["sklearn.Pipeline"], True),
        ("sklearn", "Pipeline", ["sklearn.preprocessing.StandardScaler"], False),
        ("sklearn", "Pipeline", True, True),
        ("builtins", "int", ["builtins.int"], True),
        ("builtins", "int", [], False),
    ],
    ids=["list-True", "list-False", "True", "int-True", "int-False"],
)
def test_check_type(module_name, type_name, trusted, expected):
    assert check_type(module_name, type_name, trusted) == expected


def test_audit_tree_untrusted():
    var = {"a": CustomType(1), 2: CustomType(2)}
    state = dict_get_state(var, SaveContext(None, 0, {}))
    node = DictNode(state, LoadContext(None), trusted=False)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Untrusted types found in the file: ['test_audit.CustomType']."
        ),
    ):
        audit_tree(node, trusted=False)

    # there shouldn't be an error with trusted=True
    audit_tree(node, trusted=True)

    untrusted_list = get_untrusted_types(data=dumps(var))
    assert untrusted_list == ["test_audit.CustomType"]

    # passing the type would fix it.
    audit_tree(node, trusted=untrusted_list)


def test_audit_tree_defaults():
    # test that the default types are trusted
    var = {"a": 1, 2: "b"}
    state = dict_get_state(var, SaveContext(None, 0, {}))
    node = DictNode(state, LoadContext(None), trusted=False)
    audit_tree(node, trusted=[])


@pytest.mark.parametrize(
    "trusted, defaults, expected",
    [
        (True, None, True),
        (False, int, ["builtins.int"]),
        ([int], None, ["builtins.int"]),
    ],
    ids=["trusted", "untrusted", "untrusted_list"],
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
        tree = get_tree(schema, load_context=LoadContext(src=zip_file))
        assert tree.is_safe() == is_safe


def test_gettype():
    # return None if one argument is None
    assert gettype(module_name="test", cls_or_func=None) is None

    # ImportError if the module cannot be imported
    with pytest.raises(ImportError):
        gettype(module_name="invalid-module", cls_or_func="invalid-type")


@pytest.mark.parametrize(
    "data, file, exception, message",
    [
        ("not-none", "not-none", ValueError, "Only one of data or file"),
        ("string", None, TypeError, "a bytes-like object is required, not 'str'"),
    ],
    ids=["both", "string-data"],
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


def test_complex_pipeline_untrusted_set():
    # fmt: off
    clf = Pipeline([
        ("features", FeatureUnion([
            ("scaler", StandardScaler()),
            ("sqrt", FunctionTransformer(
                    func=np.sqrt,
                    inverse_func=np.square,
                )),
        ])),
        ("clf", LogisticRegression(random_state=0, solver="liblinear")),
    ])
    # fmt: on

    untrusted = get_untrusted_types(data=dumps(clf))
    type_names = [x.split(".")[-1] for x in untrusted]
    assert type_names == [
        "sqrt",
        "square",
        "LogisticRegression",
        "FeatureUnion",
        "Pipeline",
        "StandardScaler",
        "FunctionTransformer",
    ]
