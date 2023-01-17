import builtins

import pytest

from skops.utils.importutils import import_or_raise


@pytest.fixture
def hide_available_matplotlib(monkeypatch):
    import_orig = builtins.__import__

    # ugly way of removing matplotlib from cached imports
    import sys

    for key in list(sys.modules.keys()):
        if key.startswith("matplotlib"):
            del sys.modules[key]

    def mocked_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


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
