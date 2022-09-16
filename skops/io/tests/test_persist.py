import json
import warnings
from collections import Counter
from functools import partial
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import sparse, special
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_sample_images, make_classification
from sklearn.decomposition import SparseCoder
from sklearn.exceptions import SkipTestWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedGroupKFold,
    check_cv,
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
from sklearn.utils import all_estimators, check_random_state
from sklearn.utils._tags import _safe_tags
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

import skops
from skops.io import load, save
from skops.io._sklearn import UNSUPPORTED_TYPES
from skops.io.exceptions import UnsupportedTypeException

# Default settings for X
N_SAMPLES = 50
N_FEATURES = 20


def save_load_round(estimator, f_name):
    # save and then load the model, and return the loaded model.
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)
    return loaded


def _tested_estimators(type_filter=None):
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


def _unsupported_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        if Estimator not in UNSUPPORTED_TYPES:
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


def _is_steps_like(obj):
    # helper function to check if an object is something like Pipeline.steps,
    # i.e. a list of tuples of names and estimators
    if not isinstance(obj, list):  # must be a list
        return False

    if not obj:  # must not be empty
        return False

    if not isinstance(obj[0], tuple):  # must be list of tuples
        return False

    lens = set(map(len, obj))
    if not lens == {2}:  # all elements must be length 2 tuples
        return False

    keys, vals = list(zip(*obj))

    if len(keys) != len(set(keys)):  # keys must be unique
        return False

    if not all(map(lambda x: isinstance(x, (type(None), BaseEstimator)), vals)):
        # values must be BaseEstimators or None
        return False

    return True


def _assert_generic_objects_equal(val1, val2):
    def _is_builtin(val):
        # Check if value is a builtin type
        return getattr(getattr(val, "__class__", {}), "__module__", None) == "builtins"

    if isinstance(val1, (list, tuple, np.ndarray)):
        assert len(val1) == len(val2)
        for subval1, subval2 in zip(val1, val2):
            _assert_generic_objects_equal(subval1, subval2)
            return

    assert type(val1) == type(val2)
    if hasattr(val1, "__dict__"):
        assert_params_equal(val1.__dict__, val2.__dict__)
    elif _is_builtin(val1):
        assert val1 == val2
    else:
        # not a normal Python class, could be e.g. a Cython class
        assert val1.__reduce__() == val2.__reduce__()


def _assert_tuples_equal(val1, val2):
    assert len(val1) == len(val2)
    for subval1, subval2 in zip(val1, val2):
        _assert_vals_equal(subval1, subval2)


def _assert_vals_equal(val1, val2):
    if hasattr(val1, "__getstate__"):
        # This includes BaseEstimator since they implement __getstate__ and
        # that returns the parameters as well.
        #
        # Some objects return a tuple of parameters, others a dict.
        state1 = val1.__getstate__()
        state2 = val2.__getstate__()
        assert type(state1) == type(state2)
        if isinstance(state1, tuple):
            _assert_tuples_equal(state1, state2)
        else:
            assert_params_equal(val1.__getstate__(), val2.__getstate__())
    elif sparse.issparse(val1):
        assert sparse.issparse(val2) and ((val1 - val2).nnz == 0)
    elif isinstance(val1, (np.ndarray, np.generic)):
        if len(val1.dtype) == 0:
            if val1.dtype == object:
                assert val2.dtype == object
                assert val1.shape == val2.shape
                for subval1, subval2 in zip(val1, val2):
                    _assert_generic_objects_equal(subval1, subval2)
            else:
                # simple comparison of arrays with simple dtypes, almost all
                # arrays are of this sort.
                np.testing.assert_array_equal(val1, val2)
        elif len(val1.shape) == 1:
            # comparing arrays with structured dtypes, but they have to be 1D
            # arrays. This is what we get from the Tree's state.
            assert np.all([x == y for x, y in zip(val1, val2)])
        else:
            # we don't know what to do with these values, for now.
            assert False
    elif isinstance(val1, (tuple, list)):
        assert len(val1) == len(val2)
        for subval1, subval2 in zip(val1, val2):
            _assert_vals_equal(subval1, subval2)
    elif isinstance(val1, float) and np.isnan(val1):
        assert np.isnan(val2)
    elif isinstance(val1, dict):
        # dictionaries are compared by comparing their values recursively.
        assert set(val1.keys()) == set(val2.keys())
        for key in val1:
            _assert_vals_equal(val1[key], val2[key])
    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        _assert_vals_equal(val1.__dict__, val2.__dict__)
    elif isinstance(val1, np.ufunc):
        assert val1 == val2
    elif val1.__class__.__module__ == "builtins":
        assert val1 == val2
    else:
        _assert_generic_objects_equal(val1, val2)


def assert_params_equal(params1, params2):
    # helper function to compare estimator dictionaries of parameters
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        with warnings.catch_warnings():
            # this is to silence the deprecation warning from _DictWithDeprecatedKeys
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)

        if _is_steps_like(val1):
            # Deal with Pipeline.steps, FeatureUnion.transformer_list, etc.
            assert _is_steps_like(val2)
            val1, val2 = dict(val1), dict(val2)

        if isinstance(val1, (tuple, list)):
            assert len(val1) == len(val2)
            for subval1, subval2 in zip(val1, val2):
                _assert_vals_equal(subval1, subval2)
        elif isinstance(val1, dict):
            assert_params_equal(val1, val2)
        else:
            _assert_vals_equal(val1, val2)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_non_fitted(estimator, tmp_path):
    """Check that non-fitted estimators can be persisted."""
    f_name = tmp_path / "file.skops"
    loaded = save_load_round(estimator, f_name)
    assert_params_equal(estimator.get_params(), loaded.get_params())


def get_input(estimator):
    # Return a valid input for estimator.fit

    # TODO: make this a parameter and test with sparse data
    # TODO: try with pandas.DataFrame as well
    # This data can be used for a regression model as well.
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0
    )
    y = _enforce_estimator_tags_y(estimator, y)
    tags = _safe_tags(estimator)

    if tags["pairwise"] is True:
        return np.random.rand(N_FEATURES, N_FEATURES), None

    if "2darray" in tags["X_types"]:
        # Some models require positive X
        return np.abs(X), y

    if "1darray" in tags["X_types"]:
        return X[:, 0], y

    if "3darray" in tags["X_types"]:
        return load_sample_images().images[1], None

    if "1dlabels" in tags["X_types"]:
        # model only expects y
        return y, None

    if "2dlabels" in tags["X_types"]:
        return [(1, 2), (3,)], None

    if "categorical" in tags["X_types"]:
        return [["Male", 1], ["Female", 3], ["Female", 2]], None

    if "dict" in tags["X_types"]:
        return [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}], None

    if "string" in tags["X_types"]:
        return [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ], None

    raise ValueError(f"Unsupported X type for estimator: {tags['X_types']}")


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator, request, tmp_path):
    """Check that fitted estimators can be persisted and return the right results."""
    set_random_state(estimator, random_state=0)

    X, y = get_input(estimator)
    tags = _safe_tags(estimator)
    if tags.get("requires_fit", True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn")
            if y is not None:
                estimator.fit(X, y)
            else:
                estimator.fit(X)

    f_name = tmp_path / "file.skops"
    loaded = save_load_round(estimator, f_name)
    assert_params_equal(estimator.__dict__, loaded.__dict__)

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
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg, atol=1e-7)


@pytest.mark.parametrize(
    "estimator", _unsupported_estimators(), ids=_get_check_estimator_ids
)
def test_unsupported_type_raises(estimator, tmp_path):
    """Estimators that are known to fail should raise an error"""
    set_random_state(estimator, random_state=0)

    X, y = get_input(estimator)
    tags = _safe_tags(estimator)
    if tags.get("requires_fit", True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn")
            if y is not None:
                estimator.fit(X, y)
            else:
                estimator.fit(X)

    msg = f"Objects of type {estimator.__class__.__name__} are not supported yet"
    with pytest.raises(UnsupportedTypeException, match=msg):
        f_name = tmp_path / "file.skops"
        save_load_round(estimator, f_name)


class RandomStateEstimator(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        if isinstance(self.random_state, np.random.Generator):
            # forwards compatibility with np.random.Generator
            self.random_state_ = self.random_state
        else:
            self.random_state_ = check_random_state(self.random_state)
        return self


@pytest.mark.parametrize(
    "random_state",
    [
        None,
        0,
        np.random.RandomState(42),
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
)
def test_random_state(random_state, tmp_path):
    # Numpy random Generators
    # (https://numpy.org/doc/stable/reference/random/generator.html) are not
    # supported by sklearn yet but will be in the future, thus they're tested
    # here
    est = RandomStateEstimator(random_state=random_state).fit(None, None)
    est.random_state_.random(123)  # move RNG forwards

    f_name = tmp_path / "file.skops"
    loaded = save_load_round(est, f_name)
    rand_floats_expected = est.random_state_.random(100)
    rand_floats_loaded = loaded.random_state_.random(100)
    np.testing.assert_equal(rand_floats_loaded, rand_floats_expected)


class CVEstimator(BaseEstimator):
    def __init__(self, cv=None):
        self.cv = cv

    def fit(self, X, y, **fit_params):
        self.cv_ = check_cv(self.cv)
        return self

    def split(self, X, **kwargs):
        return list(self.cv_.split(X, **kwargs))


@pytest.mark.parametrize(
    "cv",
    [
        None,
        3,
        KFold(4),
        StratifiedGroupKFold(5, shuffle=True, random_state=42),
        ShuffleSplit(6, random_state=np.random.RandomState(123)),
    ],
)
def test_cross_validator(cv, tmp_path):
    est = CVEstimator(cv=cv).fit(None, None)
    f_name = tmp_path / "file.skops"
    loaded = save_load_round(est, f_name)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0
    )

    kwargs = {}
    name = est.cv_.__class__.__name__.lower()
    if "stratified" in name:
        kwargs["y"] = y
    if "group" in name:
        kwargs["groups"] = np.random.randint(0, 5, size=len(y))

    splits_est = est.split(X, **kwargs)
    splits_loaded = loaded.split(X, **kwargs)
    assert len(splits_est) == len(splits_loaded)
    for split_est, split_loaded in zip(splits_est, splits_loaded):
        np.testing.assert_equal(split_est, split_loaded)


def test_metainfo(tmp_path):
    class MyEstimator(BaseEstimator):
        """Estimator with attributes of different supported types"""

        def fit(self, X, y=None, **fit_params):
            self.builtin_ = [1, 2, 3]
            self.stdlib_ = Counter([10, 20, 20, 30, 30, 30])
            self.numpy_ = np.arange(5)
            self.sparse_ = sparse.csr_matrix([[0, 1], [1, 0]])
            self.sklearn_ = LogisticRegression()
            # create a nested data structure to check if that works too
            self.nested_ = {
                "builtin_": self.builtin_,
                "stdlib_": self.stdlib_,
                "numpy_": self.numpy_,
                "sparse_": self.sparse_,
                "sklearn_": self.sklearn_,
            }
            return self

    # safe and load the schema
    estimator = MyEstimator().fit(None)
    f_name = tmp_path / "file.skops"
    save(file=f_name, obj=estimator)
    schema = json.loads(ZipFile(f_name).read("schema.json"))

    # check some schema metainfo
    assert schema["protocol"] == skops.io._persist.PROTOCOL
    assert schema["_skops_version"] == skops.__version__

    # additionally, check following metainfo: class, module, and version
    expected = {
        "builtin_": {
            "__class__": "list",
            "__module__": "builtins",
        },
        "stdlib_": {
            "__class__": "Counter",
            "__module__": "collections",
        },
        "numpy_": {
            "__class__": "ndarray",
            "__module__": "numpy",
        },
        "sparse_": {
            "__class__": "csr_matrix",
            "__module__": "scipy.sparse",
        },
        "sklearn_": {
            "__class__": "LogisticRegression",
            "__module__": "sklearn.linear_model",
        },
    }
    # check both the top level state and the nested state
    states = (
        schema["content"]["content"],
        schema["content"]["content"]["nested_"]["content"],
    )
    for key, val_expected in expected.items():
        for state in states:
            val_state = state[key]
            # check presence of "content"/"file" but not exact values
            assert ("content" in val_state) or ("file" in val_state)
            assert val_state["__class__"] == val_expected["__class__"]
            # We don't want to compare full module structures, because they can
            # change across versions, e.g. 'scipy.sparse.csr' moving to
            # 'scipy.sparse._csr'.
            assert val_state["__module__"].startswith(val_expected["__module__"])
