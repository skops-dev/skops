from unittest.mock import patch

import pytest


@pytest.fixture
def pandas_not_installed():
    # patch import so that it raises an ImportError when trying to import
    # pandas. This works because pandas is only imported lazily.
    orig_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        yield
