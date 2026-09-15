"""Persistence tests for old versions of the protocol"""

from __future__ import annotations

import io
import json
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import special
from sklearn.preprocessing import FunctionTransformer

from skops.io import dumps, get_untrusted_types, loads
from skops.io._utils import SaveContext, get_module, get_state
from skops.io.exceptions import UntrustedTypesFoundException
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


def _dump_v1_generator(save_context, rng) -> bytes:
    """Produce a protocol-1 ``RandomGeneratorNode`` file for a Generator."""

    def old_random_generator_get_state(obj, save_context):
        return {
            "__class__": obj.__class__.__name__,
            "__module__": get_module(type(obj)),
            "__loader__": "RandomGeneratorNode",
            "content": {
                "bit_generator": get_state(obj.bit_generator.state, save_context)
            },
        }

    return downgrade_state(
        data=dumps(rng),
        keys=None,
        old_state=old_random_generator_get_state(rng, save_context),
        protocol=1,
    )


def _tamper_v1_bit_generator_name(
    data: bytes, new_name: str | None = None, drop: bool = False
) -> bytes:
    src = ZipFile(io.BytesIO(data))
    schema = json.loads(src.read("schema.json"))
    entry = schema["content"]["bit_generator"]["content"]
    if drop:
        del entry["bit_generator"]
    else:
        entry["bit_generator"]["content"] = json.dumps(new_name)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as out:
        for name in src.namelist():
            if name == "schema.json":
                out.writestr(name, json.dumps(schema))
            else:
                out.writestr(name, src.read(name))
    return buffer.getvalue()


# The current RandomGeneratorNode is hardened against a bit generator name read
# from the file being resolved and called as a numpy.random attribute (see
# test_audit.py). The protocol-1 reader shares that shape and the same fix, so
# it gets the same checks.
def test_random_generator_v1_smuggled_bit_generator_is_surfaced(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    evil = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), "set_bit_generator"
    )
    assert get_untrusted_types(data=evil) == ["numpy.random.set_bit_generator"]
    with pytest.raises(UntrustedTypesFoundException):
        loads(evil)


def test_random_generator_v1_construct_rejects_non_bit_generator(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    evil = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), "set_bit_generator"
    )
    with pytest.raises(ValueError, match="Expected a numpy.random bit generator"):
        loads(evil, trusted=["numpy.random.set_bit_generator"])


def test_random_generator_v1_missing_name_is_rejected(save_context):
    rng = np.random.default_rng(42)
    rng.random(5)
    broken = _tamper_v1_bit_generator_name(
        _dump_v1_generator(save_context, rng), drop=True
    )
    with pytest.raises(ValueError, match="Could not find the bit generator name"):
        get_untrusted_types(data=broken)
