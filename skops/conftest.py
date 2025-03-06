import builtins
from unittest.mock import patch

import pytest


@pytest.fixture
def pandas_not_installed():
    # patch import so that it raises an ImportError when trying to import
    # pandas. This works because pandas is only imported lazily.

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield


@pytest.fixture
def matplotlib_not_installed():
    # patch import so that it raises an ImportError when trying to import
    # matplotlib. This works because matplotlib is only imported lazily.

    # ugly way of removing matplotlib from cached imports
    import sys

    for key in list(sys.modules.keys()):
        if key.startswith("matplotlib"):
            del sys.modules[key]

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield

    import matplotlib  # noqa


@pytest.fixture
def rich_not_installed():
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield
