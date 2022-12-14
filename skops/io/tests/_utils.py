import sys
import warnings

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils._testing import assert_allclose_dense_sparse

# TODO: Investigate why that seems to be an issue on MacOS (only observed with
# Python 3.8)
ATOL = 1e-6 if sys.platform == "darwin" else 1e-7


def _is_steps_like(obj):
    # helper function to check if an object is something like Pipeline.steps,
    # i.e. a list of tuples of names and estimators
    if not isinstance(obj, list):  # must be a list
        return False

    if not obj:  # must not be empty
        return False

    if not isinstance(obj[0], tuple):  # must be list of tuples
        return False

    lens = set(map(len, obj))
    if not lens == {2}:  # all elements must be length 2 tuples
        return False

    keys, vals = list(zip(*obj))

    if len(keys) != len(set(keys)):  # keys must be unique
        return False

    if not all(map(lambda x: isinstance(x, (type(None), BaseEstimator)), vals)):
        # values must be BaseEstimators or None
        return False

    return True


def _assert_generic_objects_equal(val1, val2):
    def _is_builtin(val):
        # Check if value is a builtin type
        return getattr(getattr(val, "__class__", {}), "__module__", None) == "builtins"

    if isinstance(val1, (list, tuple, np.ndarray)):
        assert len(val1) == len(val2)
        for subval1, subval2 in zip(val1, val2):
            _assert_generic_objects_equal(subval1, subval2)
            return

    assert type(val1) == type(val2)
    if hasattr(val1, "__dict__"):
        assert_params_equal(val1.__dict__, val2.__dict__)
    elif _is_builtin(val1):
        assert val1 == val2
    else:
        # not a normal Python class, could be e.g. a Cython class
        assert val1.__reduce__() == val2.__reduce__()


def _assert_tuples_equal(val1, val2):
    assert len(val1) == len(val2)
    for subval1, subval2 in zip(val1, val2):
        _assert_vals_equal(subval1, subval2)


def _assert_vals_equal(val1, val2):
    if hasattr(val1, "__getstate__"):
        # This includes BaseEstimator since they implement __getstate__ and
        # that returns the parameters as well.
        #
        # Some objects return a tuple of parameters, others a dict.
        state1 = val1.__getstate__()
        state2 = val2.__getstate__()
        assert type(state1) == type(state2)
        if isinstance(state1, tuple):
            _assert_tuples_equal(state1, state2)
        else:
            assert_params_equal(val1.__getstate__(), val2.__getstate__())
    elif sparse.issparse(val1):
        assert sparse.issparse(val2) and ((val1 - val2).nnz == 0)
    elif isinstance(val1, (np.ndarray, np.generic)):
        if len(val1.dtype) == 0:
            # for arrays with at least 2 dimensions, check that contiguity is
            # preserved
            if val1.squeeze().ndim > 1:
                assert val1.flags["C_CONTIGUOUS"] is val2.flags["C_CONTIGUOUS"]
                assert val1.flags["F_CONTIGUOUS"] is val2.flags["F_CONTIGUOUS"]
            if val1.dtype == object:
                assert val2.dtype == object
                assert val1.shape == val2.shape
                for subval1, subval2 in zip(val1, val2):
                    _assert_generic_objects_equal(subval1, subval2)
            else:
                # simple comparison of arrays with simple dtypes, almost all
                # arrays are of this sort.
                np.testing.assert_array_equal(val1, val2)
        elif len(val1.shape) == 1:
            # comparing arrays with structured dtypes, but they have to be 1D
            # arrays. This is what we get from the Tree's state.
            assert np.all([x == y for x, y in zip(val1, val2)])
        else:
            # we don't know what to do with these values, for now.
            assert False
    elif isinstance(val1, (tuple, list)):
        assert len(val1) == len(val2)
        for subval1, subval2 in zip(val1, val2):
            _assert_vals_equal(subval1, subval2)
    elif isinstance(val1, float) and np.isnan(val1):
        assert np.isnan(val2)
    elif isinstance(val1, dict):
        # dictionaries are compared by comparing their values recursively.
        assert set(val1.keys()) == set(val2.keys())
        for key in val1:
            _assert_vals_equal(val1[key], val2[key])
    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        _assert_vals_equal(val1.__dict__, val2.__dict__)
    elif isinstance(val1, np.ufunc):
        assert val1 == val2
    elif val1.__class__.__module__ == "builtins":
        assert val1 == val2
    else:
        _assert_generic_objects_equal(val1, val2)


def assert_params_equal(params1, params2):
    # helper function to compare estimator dictionaries of parameters
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        with warnings.catch_warnings():
            # this is to silence the deprecation warning from _DictWithDeprecatedKeys
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)

        if _is_steps_like(val1):
            # Deal with Pipeline.steps, FeatureUnion.transformer_list, etc.
            assert _is_steps_like(val2)
            val1, val2 = dict(val1), dict(val2)

        if isinstance(val1, (tuple, list)):
            assert len(val1) == len(val2)
            for subval1, subval2 in zip(val1, val2):
                _assert_vals_equal(subval1, subval2)
        elif isinstance(val1, dict):
            assert_params_equal(val1, val2)
        else:
            _assert_vals_equal(val1, val2)


def assert_method_outputs_equal(estimator, loaded, X):
    # helper function that checks the output of all supported methods
    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
        "predict_log_proba",
    ]:
        err_msg = (
            f"{estimator.__class__.__name__}.{method}() doesn't produce the same"
            " results after loading the persisted model."
        )
        if hasattr(estimator, method):
            X_out1 = getattr(estimator, method)(X)
            X_out2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_out1, X_out2, err_msg=err_msg, atol=ATOL)
