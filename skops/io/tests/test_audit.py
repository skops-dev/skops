import pytest

from skops.io._audit import audit_tree, check_type
from skops.io._general import DictNode, dict_get_state
from skops.io._utils import SaveState


@pytest.mark.parametrize(
    "module_name, type_name, trusted, expected",
    [
        ("sklearn", "Pipeline", ["sklearn.Pipeline"], True),
        ("sklearn", "Pipeline", ["sklearn.preprocessing.StandardScaler"], False),
    ],
    ids=[True, False],
)
def test_check_type(module_name, type_name, trusted, expected):
    assert check_type(module_name, type_name, trusted) == expected


def test_audit_tree_untrusted():
    class Test:
        def __init__(self, value):
            self.value = value

    var = {"a": Test(1), 2: Test(2)}
    state = dict_get_state(var, SaveState(None, 0, {}))
    node = DictNode(state, None, trusted=False)
    with pytest.raises(
        TypeError, match="Untrusted types found in the file: {'test_audit.Test'}."
    ):
        audit_tree(node, trusted=False)

    # passing the type would fix it.
    audit_tree(node, trusted=["test_audit.Test"])


def test_audit_tree_defaults():
    var = {"a": 1, 2: "b"}
    # breakpoint()
    state = dict_get_state(var, SaveState(None, 0, {}))
    node = DictNode(state, None, trusted=False)
    audit_tree(node, trusted=[])
