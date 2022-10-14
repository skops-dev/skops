from unittest.mock import patch

orig_import = __import__


def test_dictwithdeprecatedkeys_cannot_be_imported(tmp_path):
    # _DictWithDeprecatedKeys is removed in sklearn 1.2.0
    # see bug reported in #187

    # mock the loading of
    # sklearn.covariance._graph_lasso._DictWithDeprecatedKeys to raise an
    # ImportError
    def import_mock(name, *args, **kwargs):
        if name == "sklearn.covariance._graph_lasso":
            if args and ("_DictWithDeprecatedKeys" in args[2]):
                raise ImportError("mock import error")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_mock):
        # important: skops.io has to be loaded _after_ mocking the import,
        # otherwise it's too late, as the dispatch methods are added to their
        # respective lists on root level of their respective modules, so
        # patching after that is too late.
        from sklearn.covariance import GraphicalLassoCV

        from skops.io import load, save

        f_name = tmp_path / "file.skops"
        estimator = GraphicalLassoCV()
        save(file=f_name, obj=estimator)
        load(file=f_name)

        # sanity check: make sure that the import really raised an error and
        # thus there is no dispatch for _DictWithDeprecatedKeys, or else this
        # test would pass trivially
        from skops.io._sklearn import (
            GET_INSTANCE_DISPATCH_FUNCTIONS,
            GET_STATE_DISPATCH_FUNCTIONS,
        )

        assert not any(
            t.__name__ == "_DictWithDeprecatedKeys"
            for (t, _) in GET_STATE_DISPATCH_FUNCTIONS
        )
        assert not any(
            t.__name__ == "_DictWithDeprecatedKeys"
            for (t, _) in GET_INSTANCE_DISPATCH_FUNCTIONS
        )
