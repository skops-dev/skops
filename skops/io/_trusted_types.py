import warnings

import numpy as np
import scipy
from sklearn.utils import all_estimators

from skops.utils._fixes import get_scipy_ufunc_wrapper_type

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

    _SKLEARN_INTERNAL_TYPES.extend([_BinMapper])
except ImportError:
    pass

# NOTE: sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor is
# deliberately *not* added to `_SKLEARN_INTERNAL_TYPES`. skops can only check
# that a loaded object is of a trusted type, not that its content is safe.
# TreePredictor's ``nodes`` array holds raw child/feature indices that
# scikit-learn indexes into without bounds checks, so a crafted file can
# crash the process the first time the model runs inference.
# See https://github.com/scikit-learn/scikit-learn/pull/34558 and
# `UNTRUSTED_TYPE_REASONS` in `exceptions.py`.

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
    # As of scipy 2.0, some scipy.special ufuncs are wrapper objects that are no
    # longer numpy.ufunc instances, so they need to be discovered separately.
    _scipy_ufunc_wrapper = get_scipy_ufunc_wrapper_type()
    if _scipy_ufunc_wrapper is not None:
        SCIPY_UFUNC_TYPE_NAMES = sorted(
            set(SCIPY_UFUNC_TYPE_NAMES)
            | set(
                get_public_type_names(module=scipy.special, oftype=_scipy_ufunc_wrapper)
            )
        )

NUMPY_UFUNC_TYPE_NAMES = get_public_type_names(module=np, oftype=np.ufunc)

# Concrete bit generators exposed under ``numpy.random`` (e.g. PCG64, MT19937,
# Philox, ...). These are the only types ``RandomGeneratorNode`` is allowed to
# resolve and instantiate from a file, so they are trusted by default. Discovered
# dynamically so that bit generators added by future numpy versions are covered.
# The names use the ``numpy.random.<name>`` access path (rather than the private
# defining module) because that is how they are resolved when loading.
NUMPY_RANDOM_BIT_GENERATOR_TYPE_NAMES = sorted(
    f"numpy.random.{attr}"
    for attr in dir(np.random)
    if isinstance(obj := getattr(np.random, attr), type)
    and issubclass(obj, np.random.BitGenerator)
    and obj is not np.random.BitGenerator
)

NUMPY_DTYPE_TYPE_NAMES = sorted(
    {
        type_name
        for dtype in np.sctypeDict.values()
        if (type_name := get_type_name(dtype)).startswith("numpy")
    }
)
