import pytest

from skops.utils.importutils import import_or_raise


@pytest.mark.usefixtures("matplotlib_not_installed")
def test_import_or_raise():
    with pytest.raises(
        ModuleNotFoundError,
        match=(
            "Permutation importance requires matplotlib to be installed. In order"
            " to use permutation importance, you need to install the package in"
            " your current python environment."
        ),
    ):
        import_or_raise("matplotlib", "permutation importance")
