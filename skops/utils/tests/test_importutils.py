import sys
from unittest.mock import MagicMock

import pytest

from skops.utils.importutils import import_or_raise


def test_import_or_raise():
    sys.modules["matplotlib"] = MagicMock()
    with pytest.raises(
        ModuleNotFoundError,
        match=(
            "Permutation importance requires matplotlib to be installed. In order"
            " to use permutation importance, you need to install the package in"
            " your current python environment."
        ),
    ):
        import_or_raise("matplotlib.pyplot", "permutation importance")
