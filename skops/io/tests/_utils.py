from __future__ import annotations

import io
import json
import sys
import warnings
from zipfile import ZipFile

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


def _assert_generic_objects_equal(val1, val2, path=""):
    def _is_builtin(val):
        # Check if value is a builtin type
        return getattr(getattr(val, "__class__", {}), "__module__", None) == "builtins"

    if isinstance(val1, (list, tuple, np.ndarray)):
        assert len(val1) == len(val2), f"Path: len({path})"
        for subval1, subval2 in zip(val1, val2):
            _assert_generic_objects_equal(subval1, subval2, path=f"{path}[]")
            return

    assert type(val1) == type(val2), f"Path: type({path})"
    if hasattr(val1, "__dict__"):
        assert_params_equal(val1.__dict__, val2.__dict__, path=f"{path}.__dict__")
    elif _is_builtin(val1):
        assert val1 == val2, f"Path: {path}"
    else:
        # not a normal Python class, could be e.g. a Cython class
        _assert_tuples_equal(
            val1.__reduce__(), val2.__reduce__(), path=f"{path}.__reduce__"
        )


def _assert_tuples_equal(val1, val2, path=""):
    assert len(val1) == len(val2), f"Path: len({path})"
    for subval1, subval2 in zip(val1, val2):
        _assert_vals_equal(subval1, subval2, path=f"{path}[]")


def _assert_vals_equal(val1, val2, path=""):
    if isinstance(val1, type):  # e.g. could be np.int64
        assert val1 is val2, f"Path: {path}"
    elif hasattr(val1, "__getstate__") and (val1.__getstate__() is not None):
        # This includes BaseEstimator since they implement __getstate__ and
        # that returns the parameters as well.
        # Since Python 3.11, all objects have a __getstate__ but they return
        # None by default, in which case this check is not performed.
        # Some objects return a tuple of parameters, others a dict.
        state1 = val1.__getstate__()
        state2 = val2.__getstate__()
        assert type(state1) == type(state2), f"Path: {path}"
        if isinstance(state1, tuple):
            _assert_tuples_equal(state1, state2, path=path)
        else:
            assert_params_equal(
                val1.__getstate__(), val2.__getstate__(), path=f"{path}.__getstate__()"
            )
    elif sparse.issparse(val1):
        assert sparse.issparse(val2) and ((val1 - val2).nnz == 0), f"Path: {path}"
    elif isinstance(val1, (np.ndarray, np.generic)):
        if len(val1.dtype) == 0:
            # for arrays with at least 2 dimensions, check that contiguity is
            # preserved, but only if the array is not a view
            if val1.squeeze().ndim > 1 and val1.flags["OWNDATA"]:
                assert (
                    val1.flags["C_CONTIGUOUS"] is val2.flags["C_CONTIGUOUS"]
                ), f"Path: {path}.flags"
                assert (
                    val1.flags["F_CONTIGUOUS"] is val2.flags["F_CONTIGUOUS"]
                ), f"Path: {path}.flags"
            if val1.dtype == object:
                assert val2.dtype == object, f"Path: {path}.dtype"
                assert val1.shape == val2.shape, f"Path: {path}.shape"
                for subval1, subval2 in zip(val1, val2):
                    _assert_generic_objects_equal(subval1, subval2, path=f"{path}[]")
            else:
                # simple comparison of arrays with simple dtypes, almost all
                # arrays are of this sort.
                np.testing.assert_array_equal(val1, val2, err_msg=f"Path: {path}")
        elif len(val1.shape) == 1:
            # comparing arrays with structured dtypes, but they have to be 1D
            # arrays. This is what we get from the Tree's state.
            assert np.all([x == y for x, y in zip(val1, val2)]), f"Path: {path}"
        else:
            # we don't know what to do with these values, for now.
            assert False, f"Path: {path}"
    elif isinstance(val1, (tuple, list)):
        _assert_tuples_equal(val1, val2, path=path)
    elif isinstance(val1, float) and np.isnan(val1):
        assert np.isnan(val2), f"Path: {path}"
    elif isinstance(val1, dict):
        # dictionaries are compared by comparing their values recursively.
        assert set(val1.keys()) == set(val2.keys()), f"Path: {path}.keys()"
        for key in val1:
            _assert_vals_equal(val1[key], val2[key], path=f"{path}[{key}]")
    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        _assert_vals_equal(val1.__dict__, val2.__dict__, path=f"{path}.__dict__")
    elif isinstance(val1, np.ufunc):
        assert val1 == val2, f"Path: {path}"
    elif val1.__class__.__module__ == "builtins":
        assert val1 == val2, f"Path: {path}"
    else:
        _assert_generic_objects_equal(val1, val2, path=path)


def _clean_params(params):
    # this function deals with cleaning special parameters that for one reason
    # or another should be removed or modified.
    params = params.copy()

    # see #375
    keys_to_remove = ["_has_canonical_format", "_has_sorted_indices"]
    for key in keys_to_remove:
        params.pop(key, None)

    return params


def assert_params_equal(params1, params2, path=""):
    # helper function to compare estimator dictionaries of parameters
    if params1 is None and params2 is None:
        return

    params1, params2 = _clean_params(params1), _clean_params(params2)
    assert len(params1) == len(params2), f"Path: len({path})"
    assert set(params1.keys()) == set(params2.keys()), f"Path: {path}.keys()"
    for key in params1:
        with warnings.catch_warnings():
            # this is to silence the deprecation warning from _DictWithDeprecatedKeys
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            val1, val2 = params1[key], params2[key]
            subpath = f"{path}[{key}]"
        assert type(val1) == type(val2), f"Path: type({subpath})"

        if _is_steps_like(val1):
            # Deal with Pipeline.steps, FeatureUnion.transformer_list, etc.
            assert _is_steps_like(val2), f"Path: {subpath}"
            val1, val2 = dict(val1), dict(val2)

        if isinstance(val1, (tuple, list)):
            assert len(val1) == len(val2)
            for subval1, subval2 in zip(val1, val2):
                _assert_vals_equal(subval1, subval2, path=f"{subpath}[]")
        elif isinstance(val1, dict):
            assert_params_equal(val1, val2, path=subpath)
        else:
            _assert_vals_equal(val1, val2, path=subpath)


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


def downgrade_state(*, data: bytes, keys: list[str], old_state: dict, protocol: int):
    """Function to downgrade the persisted state of a skops object.

    This function is important for testing upgrades to the skops persistence
    protocol. When an upgrade is made, we add a test to ensure that the old
    state can still be loaded successfully. For this, we need to generate a
    state that looks like it came from the previous protocol. This function
    helps doing that.

    The caller should pass the new state, the path to the sub-state that needs
    to be downgraded, the actual old state to be downgraded to, and the protocol
    of that old version. Then this function will replace the new state with the
    old state and insert the old protocol number. It also adds an ``__id__``
    field, which is expected for memoization.

    Here is an example of how to use it:

    .. code:: python

        estimator = ...
        # get the state of the object using current protocol
        dumped = sio.dumps(estimator)
        # let's assume that estimator.foo.bar was changed
        keys = ["foo", "bar"]
        old_state = old_get_state_function(bar)
        downgraded = downgrad_state(
            data=dumped,
            keys=keys,
            old_state=old_state
            protocol=current_protocol - 1,
        )
        # check that this does not raise an error:
        sio.loads(downgrade, trusted=...)

    Parameters
    ----------
    data : bytes
        The old state, as generated by ``skops.io.dumps``.

    keys : list of str, or None
        The keys that lead to the old state. E.g. if we want to replace
        ``state["foo"]["bar"]``, then keys should be ``["foo", "bar"]``. If
        ``keys=None``, the whole schema is instead replaced by the old state
        being passed.

    old_state : dict
        The old state, as would be produced by the old ``get_state`` function.

    protocol : int
        The protocol number corresponding to the old state.

    Returns
    -------
    bytes
        The old state, as would have been dumped by ``skops.io.dumps``.

    """
    # load from bytes
    with ZipFile(io.BytesIO(data), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))

    # replace state in schema using old state
    if keys is None:
        # replace all fields
        schema = old_state
        schema["__id__"] = id(schema)
    else:
        # replace specific field
        state = schema
        for key in keys[:-1]:
            state = state[key]
        state[keys[-1]] = old_state

        # there has to be an __id__ field for memoization
        state[keys[-1]]["__id__"] = id(schema)

    schema["protocol"] = protocol

    # dump into bytes
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        zip_file.writestr("schema.json", json.dumps(schema, indent=2))
    return buffer.getbuffer().tobytes()
