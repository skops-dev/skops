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
