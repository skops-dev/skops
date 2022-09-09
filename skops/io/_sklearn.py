import inspect
import json

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier

from ._utils import get_instance, get_state, gettype

# Fixes to the existing _required_params attribute.
FIXED_REQUIRED_PARAMS = {
    CalibratedClassifierCV: ["base_estimator", "estimator"],
    _CalibratedClassifier: ["base_estimator", "calibrators", "classes"],
}


@get_state.register(_CalibratedClassifier)
@get_state.register(BaseEstimator)
def BaseEstimator_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    for key, value in obj.__dict__.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        try:
            res[key] = get_state(value, dst)
        except TypeError:
            res[key] = json.dumps(value)

    return res


@get_instance.register(_CalibratedClassifier)
@get_instance.register(BaseEstimator)
def BaseEstimator_get_instance(state, src):
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
        params[param] = get_instance(param_, src)

    instance = cls(**params)

    for key, value in state.items():
        if isinstance(value, dict):
            setattr(instance, key, get_instance(value, src))
        else:
            setattr(instance, key, json.loads(value))
    return instance
