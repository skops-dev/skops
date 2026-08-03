import sys
from types import SimpleNamespace

from skops.utils._fixes import boxplot


def test_boxplot_does_not_pass_vert_on_new_matplotlib(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "matplotlib",
        SimpleNamespace(__version__="3.11.0"),
    )

    class Ax:
        def boxplot(self, **kwargs):
            assert "vert" not in kwargs
            assert kwargs["orientation"] == "horizontal"
            return kwargs

    boxplot(
        Ax(),
        x=[[1.0, 2.0]],
        tick_labels=["feature"],
        orientation="horizontal",
        vert=False,
    )


def test_boxplot_passes_vert_on_old_matplotlib(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "matplotlib",
        SimpleNamespace(__version__="3.9.0"),
    )

    class Ax:
        def boxplot(self, **kwargs):
            assert kwargs["vert"] is False
            return kwargs

    boxplot(
        Ax(),
        x=[[1.0, 2.0]],
        tick_labels=["feature"],
        orientation="horizontal",
    )
