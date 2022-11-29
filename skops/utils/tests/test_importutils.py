import pytest
from unittest import mock
from skops.utils.importutils import import_or_raise

def test_import_or_raise():
    orig_import = __import__

    def mock_import(name, *args):
        if name == "matplotlib":
            pass
        else:
            return orig_import(name, *args)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="cannot import name*"):
            with pytest.raises(
                ModuleNotFoundError,
                match="This feature requires matplotlib to be installed.",
            ):
                import_or_raise("matplotlib")