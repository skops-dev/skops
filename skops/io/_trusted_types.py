import warnings

import numpy as np
import scipy
from sklearn.utils import all_estimators

from ._utils import get_public_type_names, get_type_name

PRIMITIVES_TYPES = [int, float, str, bool]

PRIMITIVE_TYPE_NAMES = ["builtins." + t.__name__ for t in PRIMITIVES_TYPES]

CONTAINER_TYPES = [list, set, map, tuple]

CONTAINER_TYPE_NAMES = ["builtins." + t.__name__ for t in CONTAINER_TYPES]

SKLEARN_ESTIMATOR_TYPE_NAMES = [
    get_type_name(estimator_class)
    for _, estimator_class in all_estimators()
    if get_type_name(estimator_class).startswith("sklearn.")
]

# Internal sklearn types used by GradientBoosting and HistGradientBoosting models.
# These are not public estimators but are safe internal types (loss functions, link
# functions, binning, and predictor objects) needed for serialization of fitted models.
_SKLEARN_INTERNAL_TYPES: list[type] = []

try:
    from sklearn._loss.link import (
        HalfLogitLink,
        IdentityLink,
        Interval,
        LogitLink,
        LogLink,
        MultinomialLogit,
    )

    _SKLEARN_INTERNAL_TYPES.extend(
        [HalfLogitLink, IdentityLink, Interval, LogitLink, LogLink, MultinomialLogit]
    )
except ImportError:
    pass

try:
    from sklearn._loss.loss import (
        AbsoluteError,
        ExponentialLoss,
        HalfBinomialLoss,
        HalfGammaLoss,
        HalfMultinomialLoss,
        HalfPoissonLoss,
        HalfSquaredError,
        HuberLoss,
        PinballLoss,
    )

    _SKLEARN_INTERNAL_TYPES.extend(
        [
            AbsoluteError,
            ExponentialLoss,
            HalfBinomialLoss,
            HalfGammaLoss,
            HalfMultinomialLoss,
            HalfPoissonLoss,
            HalfSquaredError,
            HuberLoss,
            PinballLoss,
        ]
    )
except ImportError:
    pass

try:
    from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

    _SKLEARN_INTERNAL_TYPES.extend([_BinMapper, TreePredictor])
except ImportError:
    pass

SKLEARN_INTERNAL_TYPE_NAMES = [
    get_type_name(t)
    for t in _SKLEARN_INTERNAL_TYPES
    if get_type_name(t).startswith("sklearn.")
]

with warnings.catch_warnings():
    # This is to suppress deprecation warning coming from the fact that scipy reports
    # numpy.core for ufuncs, and numpy.core is deprecated and renamed to numpy._core
    warnings.simplefilter("ignore", category=DeprecationWarning)
    SCIPY_UFUNC_TYPE_NAMES = get_public_type_names(
        module=scipy.special, oftype=np.ufunc
    )

NUMPY_UFUNC_TYPE_NAMES = get_public_type_names(module=np, oftype=np.ufunc)

NUMPY_DTYPE_TYPE_NAMES = sorted(
    {
        type_name
        for dtype in np.sctypeDict.values()
        if (type_name := get_type_name(dtype)).startswith("numpy")
    }
)
