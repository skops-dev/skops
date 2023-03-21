"""Persistence tests for old versions of the protocol"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special
from sklearn.preprocessing import FunctionTransformer

from skops.io import dumps, loads
from skops.io._utils import get_module
from skops.io.tests._utils import (
    assert_method_outputs_equal,
    assert_params_equal,
    downgrade_state,
)

#############
# VERSION 0 #
#############


def dummy_func(X):
    return X


@pytest.mark.parametrize("func", [np.sqrt, len, special.exp10, dummy_func])
def test_function_v0(func):
    call_count = 0

    # function_get_state as it was for protocol 0
    def old_function_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(obj),
            "__loader__": "FunctionNode",
            "content": {
                "module_path": get_module(obj),
                "function": obj.__name__,
            },
        }
        return res

    estimator = FunctionTransformer(func=func)
    X, y = [0, 1], [2, 3]
    estimator.fit(X, y)

    dumped = dumps(estimator)
    # importent: downgrade the state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=["content", "content", "func"],
        old_state=old_function_get_state(func, None),
        protocol=0,
    )
    loaded = loads(downgraded, trusted=True)

    # sanity check: ensure that the old get_state function was really called
    assert call_count == 1

    # check that loaded estimator is identical
    assert_params_equal(estimator.__dict__, loaded.__dict__)
    assert_method_outputs_equal(estimator, loaded, X)
