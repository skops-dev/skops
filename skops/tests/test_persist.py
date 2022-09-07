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
from sklearn.utils.validation import has_fit_parameter

from skops import load, save

# list of estimators for which we need to write tests since we can't
# automatically create an instance of them.
EXPLICIT_TESTS = [
    "ColumnTransformer",
    "FeatureUnion",
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "Pipeline",
    "RandomizedSearchCV",
    "SparseCoder",
]


ESTIMATORS_EXPECTED_TO_FAIL_NON_FITTED = {
    "ClassifierChain(base_estimator=LogisticRegression(C=1))",
    "CountVectorizer()",
    "DictVectorizer()",
    "FeatureAgglomeration()",
    "FeatureHasher()",
    "FunctionTransformer(func=<ufunc'erf'>,inverse_func=<ufunc'erfinv'>)",
    "GenericUnivariateSelect()",
    "HashingVectorizer()",
    "Lars()",
    "LarsCV()",
    "LassoLars()",
    "LassoLarsCV()",
    "LassoLarsIC()",
    "MultiOutputClassifier(estimator=LogisticRegression(C=1))",
    "MultiOutputRegressor(estimator=Ridge())",
    "OneHotEncoder()",
    "OneVsOneClassifier(estimator=LogisticRegression(C=1))",
    "OneVsRestClassifier(estimator=LogisticRegression(C=1))",
    "OrdinalEncoder()",
    "OutputCodeClassifier(estimator=LogisticRegression(C=1))",
    "RFE(estimator=LogisticRegression(C=1))",
    "RFECV(estimator=LogisticRegression(C=1))",
    "RegressorChain(base_estimator=Ridge())",
    "SelectFdr()",
    "SelectFpr()",
    "SelectFromModel(estimator=SGDRegressor(random_state=0))",
    "SelectFwe()",
    "SelectKBest()",
    "SelectPercentile()",
    "SelfTrainingClassifier(base_estimator=LogisticRegression(C=1))",
    "SequentialFeatureSelector(estimator=LogisticRegression(C=1))",
    "StackingClassifier(estimators=[('est1',LogisticRegression(C=0.1)),('est2',LogisticRegression(C=1))])",  # noqa: E501
    "StackingRegressor(estimators=[('est1',Ridge(alpha=0.1)),('est2',Ridge(alpha=1))])",
    "TfidfVectorizer()",
    "VotingClassifier(estimators=[('est1',LogisticRegression(C=0.1)),('est2',LogisticRegression(C=1))])",  # noqa: E501
    "VotingRegressor(estimators=[('est1',Ridge(alpha=0.1)),('est2',Ridge(alpha=1))])",
}

ESTIMATORS_EXPECTED_TO_FAIL_FITTED = {
    "AdaBoostClassifier()",
    "AdaBoostRegressor()",
    "BaggingClassifier()",
    "BaggingRegressor()",
    "Birch()",
    "BisectingKMeans()",
    "CCA()",
    "CalibratedClassifierCV(base_estimator=LogisticRegression(C=1))",
    "ClassifierChain(base_estimator=LogisticRegression(C=1))",
    "CountVectorizer()",
    "DecisionTreeClassifier()",
    "DecisionTreeRegressor()",
    "DictVectorizer()",
    "ExtraTreeClassifier()",
    "ExtraTreeRegressor()",
    "ExtraTreesClassifier()",
    "ExtraTreesRegressor()",
    "FastICA()",
    "FeatureAgglomeration()",
    "FeatureHasher()",
    "FunctionTransformer(func=<ufunc'erf'>,inverse_func=<ufunc'erfinv'>)",
    "GammaRegressor()",
    "GaussianProcessClassifier()",
    "GaussianProcessRegressor()",
    "GaussianRandomProjection()",
    "GenericUnivariateSelect()",
    "GradientBoostingClassifier()",
    "GradientBoostingRegressor()",
    "HashingVectorizer()",
    "HistGradientBoostingClassifier()",
    "HistGradientBoostingRegressor()",
    "IsolationForest()",
    "Isomap()",
    "IsotonicRegression()",
    "IterativeImputer()",
    "KBinsDiscretizer()",
    "KNeighborsClassifier()",
    "KNeighborsRegressor()",
    "KNeighborsTransformer()",
    "KernelCenterer()",
    "KernelDensity()",
    "LabelBinarizer()",
    "LabelEncoder()",
    "Lars()",
    "LarsCV()",
    "LassoLars()",
    "LassoLarsCV()",
    "LassoLarsIC()",
    "LatentDirichletAllocation()",
    "LocallyLinearEmbedding()",
    "MLPClassifier()",
    "MLPRegressor()",
    "MiniBatchDictionaryLearning()",
    "MultiLabelBinarizer()",
    "MultiOutputClassifier(estimator=LogisticRegression(C=1))",
    "MultiOutputRegressor(estimator=Ridge())",
    "NearestNeighbors()",
    "NeighborhoodComponentsAnalysis()",
    "OneHotEncoder()",
    "OneVsOneClassifier(estimator=LogisticRegression(C=1))",
    "OneVsRestClassifier(estimator=LogisticRegression(C=1))",
    "OrdinalEncoder()",
    "OrthogonalMatchingPursuit()",
    "OrthogonalMatchingPursuitCV()",
    "OutputCodeClassifier(estimator=LogisticRegression(C=1))",
    "PLSCanonical()",
    "PLSRegression()",
    "PLSSVD()",
    "PassiveAggressiveClassifier()",
    "PatchExtractor()",
    "Perceptron()",
    "PoissonRegressor()",
    "RFE(estimator=LogisticRegression(C=1))",
    "RFECV(estimator=LogisticRegression(C=1))",
    "RadiusNeighborsClassifier()",
    "RadiusNeighborsRegressor()",
    "RadiusNeighborsTransformer()",
    "RandomForestClassifier()",
    "RandomForestRegressor()",
    "RandomTreesEmbedding()",
    "RegressorChain(base_estimator=Ridge())",
    "SGDClassifier()",
    "SGDOneClassSVM()",
    "SelectFdr()",
    "SelectFpr()",
    "SelectFromModel(estimator=SGDRegressor(random_state=0))",
    "SelectFwe()",
    "SelectKBest()",
    "SelectPercentile()",
    "SelfTrainingClassifier(base_estimator=LogisticRegression(C=1))",
    "SequentialFeatureSelector(estimator=LogisticRegression(C=1))",
    "SparseRandomProjection()",
    "SpectralBiclustering()",
    "SpectralEmbedding()",
    "SplineTransformer()",
    "StackingClassifier(estimators=[('est1',LogisticRegression(C=0.1)),('est2',LogisticRegression(C=1))])",  # noqa: E501
    "StackingRegressor(estimators=[('est1',Ridge(alpha=0.1)),('est2',Ridge(alpha=1))])",
    "TSNE()",
    "TfidfTransformer()",
    "TfidfVectorizer()",
    "TweedieRegressor()",
    "VotingClassifier(estimators=[('est1',LogisticRegression(C=0.1)),('est2',LogisticRegression(C=1))])",  # noqa: E501
    "VotingRegressor(estimators=[('est1',Ridge(alpha=0.1)),('est2',Ridge(alpha=1))])",
}


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


def assert_params_equal(est1, est2):
    # helper function to compare estimator params
    params1 = est1.get_params()
    params2 = est2.get_params()
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        if key.endswith("steps") or key.endswith("transformer_list"):
            # TODO: anything smarter?
            continue

        val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)
        if isinstance(val1, BaseEstimator):
            assert_params_equal(val1, val2)
        elif isinstance(val1, (np.ndarray, np.generic)):
            assert np.all_close(val1, val2)
        elif isinstance(val1, float) and np.isnan(val1):
            assert np.isnan(val2)
        else:
            assert val1 == val2


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_non_fitted(estimator, request):
    """Check that non-fitted estimators can be persisted."""
    if _get_check_estimator_ids(estimator) in ESTIMATORS_EXPECTED_TO_FAIL_NON_FITTED:
        request.applymarker(
            pytest.mark.xfail(
                run=False, strict=True, reason="TODO this estimator does not pass yet"
            )
        )

    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)
    assert_params_equal(estimator, loaded)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator, request):
    """Check that fitted estimators can be persisted and return the right results."""
    if _get_check_estimator_ids(estimator) in ESTIMATORS_EXPECTED_TO_FAIL_FITTED:
        request.applymarker(
            pytest.mark.xfail(
                run=False, strict=True, reason="TODO this estimator does not pass yet"
            )
        )

    set_random_state(estimator, random_state=0)

    # TODO: make this a parameter and test with sparse data
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
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
        if has_fit_parameter(estimator, "sample_weight"):
            estimator.fit(X, y=y, sample_weight=None)
        else:
            estimator.fit(X, y=y)

    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)

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
