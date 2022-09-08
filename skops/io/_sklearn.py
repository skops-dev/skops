import inspect
import json

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier

from ._utils import gettype

# Fixes to the existing _required_params attribute.
FIXED_REQUIRED_PARAMS = {
    CalibratedClassifierCV: ["base_estimator", "estimator"],
    _CalibratedClassifier: ["base_estimator", "calibrators", "classes"],
}


def BaseEstimator_get_state(obj, dst):
    from ._persist import get_state_method

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    for key, value in obj.__dict__.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        try:
            res[key] = get_state_method(value)(value, dst)
        except TypeError:
            res[key] = json.dumps(value)

    return res


def BaseEstimator_get_instance(state, src):
    from ._persist import get_instance_method

    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")

    if cls in FIXED_REQUIRED_PARAMS:
        required_parameters = FIXED_REQUIRED_PARAMS[cls]
    else:
        required_parameters = getattr(cls, "_required_parameters", [])
    params = {}
    for param in required_parameters:
        # sometimes required params have alternatives and only one needs to be
        # provided.
        if param not in state:
            continue
        param_ = state.pop(param)
        params[param] = get_instance_method(param_)(param_, src)

    instance = cls(**params)

    for key, value in state.items():
        if isinstance(value, dict):
            setattr(instance, key, get_instance_method(value)(value, src))
        else:
            setattr(instance, key, json.loads(value))
    return instance


def get_state_methods():
    return {
        BaseEstimator: BaseEstimator_get_state,
        # This class doesn't inherit from BaseEstimator
        _CalibratedClassifier: BaseEstimator_get_state,
    }


def get_instance_methods():
    return {
        BaseEstimator: BaseEstimator_get_instance,
        # This class doesn't inherit from BaseEstimator
        _CalibratedClassifier: BaseEstimator_get_instance,
    }
