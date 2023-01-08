import builtins

import pytest

from skops.utils.importutils import import_or_raise


@pytest.fixture
def hide_available_matplotlib(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "matplotlib":
            print("*" * 50, "INTERCEPT MATPLOTLIB IMPORT")
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    print("*" * 50, "THE FIXTURE IS BEING USED")


@pytest.mark.usefixtures("hide_available_matplotlib")
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
