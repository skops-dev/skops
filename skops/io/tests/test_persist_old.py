"""Persistence tests for old versions of the protocol"""

from __future__ import annotations

import io
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import special
from sklearn.preprocessing import FunctionTransformer

from skops.io import dumps, get_untrusted_types, loads
from skops.io._utils import SaveContext, get_module, get_state
from skops.io.tests._utils import (
    assert_method_outputs_equal,
    assert_params_equal,
    downgrade_state,
)


def dummy_func(X):
    return X


@pytest.fixture
def save_context():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        yield SaveContext(zip_file=zip_file)


#############
# VERSION 0 #
#############


@pytest.mark.parametrize("func", [np.sqrt, len, special.exp10, dummy_func])
def test_persist_function_v0(func):
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
    loaded = loads(downgraded, trusted=get_untrusted_types(data=downgraded))

    # sanity check: ensure that the old get_state function was really called
    assert call_count == 1

    # check that loaded estimator is identical
    assert_params_equal(estimator.__dict__, loaded.__dict__)
    assert_method_outputs_equal(estimator, loaded, X)


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
    ids=["default_rng", "Generator"],
)
def test_random_generator_v0(rng):
    call_count = 0

    # random_generator_get_state as it was for protocol 0
    def old_random_generator_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        bit_generator_state = obj.bit_generator.state
        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {"bit_generator": bit_generator_state},
        }
        return res

    rng.random(123)  # move RNG forwards
    dumped = dumps(rng)
    # importent: downgrade the whole state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=None,
        old_state=old_random_generator_get_state(rng, None),
        protocol=0,
    )

    # old loader only worked with trusted=True, see #329
    # update: we have removed trusted=True, so this doesn't work anymore.
    with pytest.raises(AttributeError):
        loads(downgraded, trusted=[])


#############
# VERSION 1 #
#############


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
    ids=["default_rng", "Generator"],
)
def test_random_generator_v1(save_context, rng):
    call_count = 0

    # random_generator_get_state as it was for protocol 0
    def old_random_generator_get_state(obj, save_context):
        # added for testing
        nonlocal call_count
        call_count += 1
        # end

        bit_generator_state = get_state(obj.bit_generator.state, save_context)
        res = {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {"bit_generator": bit_generator_state},
        }
        return res

    rng.random(123)  # move RNG forwards
    dumped = dumps(rng)
    # importent: downgrade the whole state to mimic older version
    downgraded = downgrade_state(
        data=dumped,
        keys=None,
        old_state=old_random_generator_get_state(rng, save_context),
        protocol=1,
    )

    loads(downgraded, trusted=[])
