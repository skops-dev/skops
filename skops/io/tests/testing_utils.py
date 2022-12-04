import warnings
from functools import partial

import numpy as np
from scipy import special
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import SparseCoder
from sklearn.exceptions import SkipTestWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.utils import all_estimators
from sklearn.utils._testing import SkipTest
from sklearn.utils.estimator_checks import _construct_instance

from skops.io._sklearn import UNSUPPORTED_TYPES

# Default settings for X
N_SAMPLES = 50
N_FEATURES = 20


def get_tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        if Estimator in UNSUPPORTED_TYPES:
            continue
        try:
            # suppress warnings here for skipped estimators.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=SkipTestWarning,
                    message="Can't instantiate estimator",
                )
                estimator = _construct_instance(Estimator)
                # with the kind of data we pass, it needs to be 1 for the few
                # estimators which have this.
                if "n_components" in estimator.get_params():
                    estimator.set_params(n_components=1)
                    # Then n_best needs to be <= n_components
                    if "n_best" in estimator.get_params():
                        estimator.set_params(n_best=1)
                if "patch_size" in estimator.get_params():
                    # set patch size to fix PatchExtractor test.
                    estimator.set_params(patch_size=(3, 3))
        except SkipTest:
            continue

        yield estimator

    # nested Pipeline & FeatureUnion
    # fmt: off
    yield Pipeline([
        ("features", FeatureUnion([
            ("scaler", StandardScaler()),
            ("scaled-poly", Pipeline([
                ("polys", FeatureUnion([
                    ("poly1", PolynomialFeatures()),
                    ("poly2", PolynomialFeatures(degree=3, include_bias=False))
                ])),
                ("scale", MinMaxScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(random_state=0, solver="liblinear")),
    ])
    # fmt: on

    # FunctionTransformer with numpy functions
    yield FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.square,
    )

    # FunctionTransformer with scipy functions - problem is that they look like
    # numpy ufuncs
    yield FunctionTransformer(
        func=special.erf,
        inverse_func=special.erfinv,
    )

    # partial functions should be supported
    yield FunctionTransformer(
        func=partial(np.add, 10),
        inverse_func=partial(np.add, -10),
    )

    yield KNeighborsClassifier(algorithm="kd_tree")
    yield KNeighborsRegressor(algorithm="ball_tree")

    yield ColumnTransformer(
        [
            ("norm1", Normalizer(norm="l1"), [0]),
            ("norm2", Normalizer(norm="l1"), [1, 2]),
            ("norm3", Normalizer(norm="l1"), [True] + (N_FEATURES - 1) * [False]),
            ("norm4", Normalizer(norm="l1"), np.array([1, 2])),
            ("norm5", Normalizer(norm="l1"), slice(3)),
            ("norm6", Normalizer(norm="l1"), slice(-10, -3, 2)),
        ],
    )

    yield GridSearchCV(
        LogisticRegression(random_state=0, solver="liblinear"),
        {"C": [1, 2, 3, 4, 5]},
    )

    yield HalvingGridSearchCV(
        LogisticRegression(random_state=0, solver="liblinear"),
        {"C": [1, 2, 3, 4, 5]},
    )

    yield HalvingRandomSearchCV(
        LogisticRegression(random_state=0, solver="liblinear"),
        {"C": [1, 2, 3, 4, 5]},
    )

    yield RandomizedSearchCV(
        LogisticRegression(random_state=0, solver="liblinear"),
        {"C": [1, 2, 3, 4, 5]},
        n_iter=3,
    )

    dictionary = np.random.randint(-2, 3, size=(5, N_FEATURES)).astype(float)
    yield SparseCoder(
        dictionary=dictionary,
        transform_algorithm="lasso_lars",
    )
