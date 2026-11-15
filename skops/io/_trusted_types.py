from __future__ import annotations

import warnings

import numpy as np
import scipy
import sklearn
from sklearn.utils import all_estimators
from sklearn.utils.fixes import parse_version

from skops.utils._fixes import get_scipy_ufunc_wrapper_type

from ._utils import get_public_type_names, get_type_name

PRIMITIVES_TYPES = [int, float, str, bool]

PRIMITIVE_TYPE_NAMES = ["builtins." + t.__name__ for t in PRIMITIVES_TYPES]

CONTAINER_TYPES = [list, set, map, tuple]

CONTAINER_TYPE_NAMES = ["builtins." + t.__name__ for t in CONTAINER_TYPES]

# libsvm-backed estimators are deliberately *not* trusted by default, even
# though they are public scikit-learn estimators. skops can only check that a
# loaded object is of a trusted type, not that its content is safe. These
# estimators hand their saved arrays (``support_vectors_``, ``_n_support``,
# ``_dual_coef_``, ``_intercept_``) straight to the libsvm C code, which
# indexes into them without bounds checks. A crafted file can keep each array's
# dtype and shape valid but make them inconsistent with one another, so skops
# loads the object successfully and the first ``.predict()`` reads or writes
# out of bounds, crashing the process (and, in one case, overflowing the heap).
# The names are derived from the classes so they stay in sync with what the
# audit sees via ``all_estimators()``. See
# https://github.com/scikit-learn/scikit-learn/pull/34558 and
# ``UNTRUSTED_TYPE_REASONS`` in ``exceptions.py``.
#
# This is an "until" gate. skops reconstructs estimators through
# ``__setstate__`` (see ``_set_attrs`` in ``_general.py``), so once
# scikit-learn validates these arrays at load time -- e.g. via a
# ``__sklearn_validate_model__`` called from ``__setstate__`` -- a loaded
# object is safe and the estimators can be trusted again. No released
# scikit-learn validates the libsvm models yet (scikit-learn#34558 currently
# covers only the tree types), so ``_SKLEARN_LIBSVM_VALIDATED_VERSION`` is
# ``None``, meaning "no fixed version yet", and the estimators are untrusted on
# every version. When a fixed scikit-learn ships, set it to that version string
# (e.g. ``"1.11"``) to trust the estimators again from that version on.
_SKLEARN_LIBSVM_VALIDATED_VERSION: str | None = None


def _sklearn_validates_libsvm_load(
    sklearn_version: str, validated_version: str | None
) -> bool:
    """Whether the installed scikit-learn validates libsvm estimators' arrays
    at load time, which would make them safe for skops to trust by default.

    ``validated_version`` is the first scikit-learn version that does so, or
    ``None`` when no such version exists yet -- the arrays are then never
    considered validated, so the estimators stay untrusted on every version.
    """
    if validated_version is None:
        return False
    return parse_version(sklearn_version) >= parse_version(validated_version)


_SKLEARN_UNTRUSTED_TYPES: list[type] = []

if not _sklearn_validates_libsvm_load(
    sklearn.__version__, _SKLEARN_LIBSVM_VALIDATED_VERSION
):
    try:
        from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM

        _SKLEARN_UNTRUSTED_TYPES.extend([SVC, SVR, NuSVC, NuSVR, OneClassSVM])
    except ImportError:
        pass

SKLEARN_UNTRUSTED_ESTIMATOR_TYPE_NAMES = [
    get_type_name(t) for t in _SKLEARN_UNTRUSTED_TYPES
]

SKLEARN_ESTIMATOR_TYPE_NAMES = [
    name
    for _, estimator_class in all_estimators()
    if (name := get_type_name(estimator_class)).startswith("sklearn.")
    and name not in SKLEARN_UNTRUSTED_ESTIMATOR_TYPE_NAMES
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
