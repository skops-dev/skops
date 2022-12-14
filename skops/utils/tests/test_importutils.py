import importlib
from unittest import mock

import pytest

from skops.utils.importutils import import_or_raise


def test_import_or_raise():
    # orig_import = __import__

    def mock_import(name, *args):
        if name == "matplotlib":
            raise ImportError("No module named 'matplotlib'")
        else:
            return importlib.import_module(name)

    with mock.patch("importlib.import_module", side_effect=mock_import):
        with pytest.raises(
            ModuleNotFoundError,
            match=(
                "Permutation importance requires matplotlib to be installed. In order"
                " to use permutation importance, you need to install the package in"
                " your current python environment."
            ),
        ):
            import_or_raise("matplotlib", "permutation importance")
