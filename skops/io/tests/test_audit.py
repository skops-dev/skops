import io
import json
from zipfile import ZipFile

import pytest

from skops.io import dumps, get_untrusted_types
from skops.io._audit import audit_tree, check_type
from skops.io._dispatch import Node, get_tree
from skops.io._general import DictNode, dict_get_state
from skops.io._utils import SaveState


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
    ],
    ids=[True, False],
)
def test_check_type(module_name, type_name, trusted, expected):
    assert check_type(module_name, type_name, trusted) == expected


def test_audit_tree_untrusted():
    var = {"a": CustomType(1), 2: CustomType(2)}
    state = dict_get_state(var, SaveState(None, 0, {}))
    node = DictNode(state, None, trusted=False)
    with pytest.raises(
        TypeError, match="Untrusted types found in the file: {'test_audit.CustomType'}."
    ):
        audit_tree(node, trusted=False)

    # there shouldn't be an error with trusted=True
    audit_tree(node, trusted=True)

    untrusted_list = get_untrusted_types(data=dumps(var))
    assert untrusted_list == ["test_audit.CustomType"]

    # passing the type would fix it.
    audit_tree(node, trusted=untrusted_list)


def test_audit_tree_defaults():
    var = {"a": 1, 2: "b"}
    state = dict_get_state(var, SaveState(None, 0, {}))
    node = DictNode(state, None, trusted=False)
    audit_tree(node, trusted=[])


@pytest.mark.parametrize(
    "trusted, defaults, expected",
    [
        (True, None, True),
        (False, 1, 1),
        ([1], None, [1]),
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
    ],
    ids=["int", "dict", "untrusted"],
)
def test_list_safety(values, is_safe):
    content = dumps(values)

    with ZipFile(io.BytesIO(content), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, src=zip_file)
        assert tree.is_safe == is_safe
