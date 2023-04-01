import numpy as np
import scipy
from sklearn.utils import all_estimators

from ._utils import get_type_name

PRIMITIVES_TYPES = [int, float, str, bool]

PRIMITIVE_TYPE_NAMES = ["builtins." + t.__name__ for t in PRIMITIVES_TYPES]

SKLEARN_ESTIMATOR_TYPE_NAMES = [
    get_type_name(estimator_class)
    for _, estimator_class in all_estimators()
    if get_type_name(estimator_class).startswith("sklearn.")
]

SCIPY_UFUNC_TYPE_NAMES = sorted(
    set(
        [
            get_type_name(getattr(scipy.special, attr))
            for attr in dir(scipy.special)
            if isinstance(getattr(scipy.special, attr), np.ufunc)
            and get_type_name(getattr(scipy.special, attr)).startswith("scipy")
        ]
    )
)

NUMPY_UFUNC_TYPE_NAMES = sorted(
    set(
        [
            get_type_name(getattr(np, attr))
            for attr in dir(np)
            if isinstance(getattr(np, attr), np.ufunc)
            and get_type_name(getattr(np, attr)).startswith("numpy")
        ]
    )
)

NUMPY_DTYPE_TYPE_NAMES = sorted(
    set(
        [
            get_type_name(dtype)
            for dtypes in np.sctypes.values()
            for dtype in dtypes  # type: ignore
            if get_type_name(dtype).startswith("numpy")
        ]
    )
)
