import tempfile
import warnings

import numpy as np
import pytest
from scipy import special
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.utils import all_estimators
from sklearn.utils._testing import (
    SkipTest,
    assert_allclose_dense_sparse,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _construct_instance,
    _enforce_estimator_tags_y,
    _get_check_estimator_ids,
)

from skops import load, save

# list of estimators for which we need to write tests since we can't
# automatically create an instance of them.
EXPLICIT_TESTS = [
    "ColumnTransformer",
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "RandomizedSearchCV",
    "SparseCoder",
]

# These estimators fail in our tests, we should fix them one by one, by
# removing them from this list, and fixing the error.
ESTIMATORS_TO_IGNORE = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "Birch",
    "BisectingKMeans",
    # CalibratedClassifierCV: It's not technically broken but the equality check
    # of the sklearn.calibration._CalibratedClassifier attribute fails. If it
    # were a BaseEstimator, we would be good. Alternatively, we have to special
    # case this type when equality checking
    "CalibratedClassifierCV",
    "CCA",
    "ClassifierChain",
    "CountVectorizer",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "DictVectorizer",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "FastICA",
    "FeatureAgglomeration",
    "FeatureHasher",
    "GammaRegressor",
    "GaussianProcessClassifier",
    "GaussianProcessRegressor",
    "GaussianRandomProjection",
    "GenericUnivariateSelect",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    # GraphicalLassoCV fails because it has an
    # sklearn.covariance._graph_lasso._DictWithDeprecatedKeys attribute that is
    # converted into a regular dict
    "GraphicalLassoCV",
    "HashingVectorizer",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "IsolationForest",
    "Isomap",
    "IsotonicRegression",
    "IterativeImputer",
    "KBinsDiscretizer",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KNeighborsTransformer",
    "KernelCenterer",
    "KernelDensity",
    "KernelRidge",
    "LabelBinarizer",
    "LabelEncoder",
    "Lars",
    "LarsCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LatentDirichletAllocation",
    "LocallyLinearEmbedding",
    # LogisticRegressionCV fails because it stores a dict with ints as keys, but
    # the json roundtrip results in the keys being strings
    "LogisticRegressionCV",
    "MLPClassifier",
    "MLPRegressor",
    "MiniBatchDictionaryLearning",
    "MiniBatchSparsePCA",
    "MultiLabelBinarizer",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "NearestNeighbors",
    "NeighborhoodComponentsAnalysis",
    "OneHotEncoder",
    "OrdinalEncoder",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "PLSSVD",
    "PassiveAggressiveClassifier",
    "PatchExtractor",
    "Perceptron",
    "PoissonRegressor",
    "QuantileRegressor",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "RadiusNeighborsTransformer",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreesEmbedding",
    "RegressorChain",
    "Ridge",
    "RidgeClassifier",
    "SGDClassifier",
    "SGDOneClassSVM",
    "SelectFdr",
    "SelectFpr",
    "SelectFwe",
    "SelectKBest",
    "SelectPercentile",
    "SelfTrainingClassifier",
    "SequentialFeatureSelector",
    "SparsePCA",
    "SparseRandomProjection",
    "SpectralBiclustering",
    "SpectralEmbedding",
    "SimpleImputer",
    "SplineTransformer",
    "StackingClassifier",
    "StackingRegressor",
    "TSNE",
    "TfidfTransformer",
    "TfidfVectorizer",
    "TweedieRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "FunctionTransformer",
]


def save_load_round(estimator):
    # save and then load the model, and return the loaded model.
    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)
    return loaded


def _tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        try:
            # suppress warnings here for skipped estimators.
            with warnings.catch_warnings():
                estimator = _construct_instance(Estimator)
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


def _assert_vals_equal(val1, val2):
    if isinstance(val1, BaseEstimator):
        assert_params_equal(val1.get_params(), val2.get_params())
    elif isinstance(val1, (np.ndarray, np.generic)):
        assert np.allclose(val1, val2)
    elif isinstance(val1, float) and np.isnan(val1):
        assert np.isnan(val2)
    else:
        assert val1 == val2


def assert_params_equal(params1, params2):
    # helper function to compare estimator dictionaries of parameters
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        if key.endswith("steps") or key.endswith("transformer_list"):
            # TODO: anything smarter?
            continue

        val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)

        if isinstance(val1, (tuple, list)):
            assert len(val1) == len(val2)
            for subval1, subval2 in zip(val1, val2):
                _assert_vals_equal(subval1, subval2)
        elif isinstance(val1, dict):
            assert_params_equal(val1, val2)
        else:
            _assert_vals_equal(val1, val2)


def _get_learned_attrs(estimator):
    # Find the learned attributes like "coefs_"
    attrs = {}
    for key in estimator.__dict__:
        if key.startswith("_") or not key.endswith("_"):
            continue

        val = getattr(estimator, key)
        if isinstance(val, property):
            continue
        attrs[key] = val
    return attrs


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_non_fitted(estimator, request):
    """Check that non-fitted estimators can be persisted."""
    if estimator.__class__.__name__ in ESTIMATORS_TO_IGNORE:
        pytest.skip()

    loaded = save_load_round(estimator)
    assert_params_equal(estimator.get_params(), loaded.get_params())


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator, request):
    """Check that fitted estimators can be persisted and return the right results."""
    if estimator.__class__.__name__ in ESTIMATORS_TO_IGNORE:
        pytest.skip()

    set_random_state(estimator, random_state=0)

    # TODO: make this a parameter and test with sparse data
    X = np.array(
        [
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    y = _enforce_estimator_tags_y(estimator, y)

    with warnings.catch_warnings():
        estimator.fit(X, y=y)

    loaded = save_load_round(estimator)
    # check that params and learned attributes are equal
    assert_params_equal(estimator.get_params(), loaded.get_params())
    attrs_est = _get_learned_attrs(estimator)
    attrs_loaded = _get_learned_attrs(loaded)
    assert_params_equal(attrs_est, attrs_loaded)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
        "predict_log_proba",
    ]:
        err_msg = (
            f"{estimator.__class__.__name__}.{method}() doesn't produce the same"
            " results after loading the persisted model."
        )
        if hasattr(estimator, method):
            X_pred1 = getattr(estimator, method)(X)
            X_pred2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)
