import importlib
import inspect
import io
import json
import sys
import warnings
from collections import Counter
from functools import partial, wraps
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pytest
from scipy import sparse, special
from sklearn.base import BaseEstimator, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_sample_images, make_classification, make_regression
from sklearn.decomposition import SparseCoder
from sklearn.exceptions import SkipTestWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
    ShuffleSplit,
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
from skops.io import dump, dumps, get_untrusted_types, load, loads
from skops.io._dispatch import NODE_TYPE_MAPPING, get_tree
from skops.io._sklearn import UNSUPPORTED_TYPES
from skops.io._utils import LoadContext, SaveContext, _get_state, get_state
from skops.io.exceptions import UnsupportedTypeException

# Default settings for X
N_SAMPLES = 50
N_FEATURES = 20

# TODO: Investigate why that seems to be an issue on MacOS (only observed with
# Python 3.8)
ATOL = 1e-6 if sys.platform == "darwin" else 1e-7


@pytest.fixture(autouse=True, scope="module")
def debug_dispatch_functions():
    # Patch the get_state and get_tree methods to add some sanity checks on
    # them. Specifically, we test that the arguments of the functions all follow
    # the same pattern to enforce consistency and that the "state" is either a
    # dict with specified keys or a primitive type.

    def debug_get_state(func):
        # Check consistency of argument names, output type, and that the output,
        # if a dict, has certain keys, or if not a dict, is a primitive type.
        signature = inspect.signature(func)
        assert list(signature.parameters.keys()) == ["obj", "save_context"]

        @wraps(func)
        def wrapper(obj, save_context):
            # NB: __id__ set in main 'get_state' func, so no check here
            result = func(obj, save_context)

            assert "__class__" in result
            assert "__module__" in result
            assert "__loader__" in result

            return result

        return wrapper

    def debug_get_tree(func):
        # check consistency of argument names and input type
        signature = inspect.signature(func)
        assert list(signature.parameters.keys()) == ["state", "load_context", "trusted"]

        @wraps(func)
        def wrapper(state, load_context, trusted):
            assert "__class__" in state
            assert "__module__" in state
            assert "__loader__" in state
            assert "__id__" in state
            assert isinstance(load_context, LoadContext)

            result = func(state, load_context, trusted)
            return result

        return wrapper

    modules = ["._general", "._numpy", "._scipy", "._sklearn"]
    for module_name in modules:
        # overwrite exposed functions for get_state and get_tree
        module = importlib.import_module(module_name, package="skops.io")
        for cls, method in getattr(module, "GET_STATE_DISPATCH_FUNCTIONS", []):
            _get_state.register(cls)(debug_get_state(method))
        for key, method in NODE_TYPE_MAPPING.copy().items():
            NODE_TYPE_MAPPING[key] = debug_get_tree(method)


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
            # for arrays with at least 2 dimensions, check that contiguity is
            # preserved
            if val1.squeeze().ndim > 1:
                assert val1.flags["C_CONTIGUOUS"] is val2.flags["C_CONTIGUOUS"]
                assert val1.flags["F_CONTIGUOUS"] is val2.flags["F_CONTIGUOUS"]
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
def test_can_persist_non_fitted(estimator):
    """Check that non-fitted estimators can be persisted."""
    loaded = loads(dumps(estimator), trusted=True)
    assert_params_equal(estimator.get_params(), loaded.get_params())


def get_input(estimator):
    # Return a valid input for estimator.fit

    # TODO: make this a parameter and test with sparse data
    # TODO: try with pandas.DataFrame as well
    if is_regressor(estimator):
        # classifier data can lead to failure of certain regressors to fit, e.g.
        # RANSAC in sklearn 0.24, so regression data is needed
        X, y = make_regression(
            n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0
        )
    else:
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

    if tags["X_types"] == "sparse":
        # TfidfTransformer in sklearn 0.24 needs this
        return sparse.csr_matrix(X), y

    raise ValueError(f"Unsupported X type for estimator: {tags['X_types']}")


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator, request):
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

    # test that we can get a list of untrusted types. This is a smoke test
    # to make sure there are no errors running this method.
    # it is in this test to save time, as it requires a fitted estimator.
    dumped = dumps(estimator)
    untrusted_types = get_untrusted_types(data=dumped)

    loaded = loads(dumped, trusted=untrusted_types)
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
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg, atol=ATOL)


@pytest.mark.parametrize(
    "estimator", _unsupported_estimators(), ids=_get_check_estimator_ids
)
def test_unsupported_type_raises(estimator):
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
        dumps(estimator)


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
    ids=["None", "int", "RandomState", "default_rng", "Generator"],
)
def test_random_state(random_state):
    # Numpy random Generators
    # (https://numpy.org/doc/stable/reference/random/generator.html) are not
    # supported by sklearn yet but will be in the future, thus they're tested
    # here
    est = RandomStateEstimator(random_state=random_state).fit(None, None)
    est.random_state_.random(123)  # move RNG forwards

    loaded = loads(dumps(est), trusted=True)
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
        KFold(4, shuffle=True, random_state=42),
        GroupKFold(5),
        ShuffleSplit(6, random_state=np.random.RandomState(123)),
    ],
)
def test_cross_validator(cv):
    est = CVEstimator(cv=cv).fit(None, None)
    loaded = loads(dumps(est), trusted=True)
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


class EstimatorWith2dObjectArray(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        self.obj_array_ = np.array([[1, "2"], [3.0, None]])
        return self


@pytest.mark.parametrize(
    "transpose",
    [
        False,
        pytest.param(True, marks=pytest.mark.xfail(raises=AssertionError)),
    ],
)
def test_numpy_object_dtype_2d_array(transpose):
    # Explicitly test multi-dimensional (i.e. more than 1) object arrays, since
    # those use json instead of numpy.save/load and some errors may only occur
    # with multi-dimensional arrays (e.g. mismatched contiguity). For
    # F-contiguous object arrays, this test currently fails, as the array is
    # loaded as C-contiguous.
    est = EstimatorWith2dObjectArray().fit(None)
    if transpose:
        est.obj_array_ = est.obj_array_.T

    loaded = loads(dumps(est), trusted=True)
    assert_params_equal(est.__dict__, loaded.__dict__)


def test_metainfo():
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
    dumped = dumps(estimator)
    schema = json.loads(ZipFile(io.BytesIO(dumped)).read("schema.json"))

    # check some schema metainfo
    assert schema["protocol"] == skops.io._utils.DEFAULT_PROTOCOL
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


class EstimatorIdenticalArrays(BaseEstimator):
    """Estimator that stores multiple references to the same array"""

    def fit(self, X, y=None, **fit_params):
        # each block below should reference the same file
        self.X = X
        self.X_2 = X
        self.X_list = [X, X]
        self.X_dict = {"a": X, 2: X}

        # copies are not deduplicated
        X_copy = X.copy()
        self.X_copy = X_copy
        self.X_copy2 = X_copy

        # transposed matrices are not the same
        X_T = X.T
        self.X_T = X_T
        self.X_T2 = X_T

        # slices are not the same
        self.vector = X[0]

        self.vector_2 = X[0]

        self.scalar = X[0, 0]

        self.scalar_2 = X[0, 0]

        # deduplication should work on sparse matrices
        X_sparse = sparse.csr_matrix(X)
        self.X_sparse = X_sparse
        self.X_sparse2 = X_sparse

        return self


def test_identical_numpy_arrays_not_duplicated():
    # Test that identical numpy arrays are not stored multiple times
    X = np.random.random((10, 5))
    estimator = EstimatorIdenticalArrays().fit(X)
    dumped = dumps(estimator)
    loaded = loads(dumped, trusted=True)
    assert_params_equal(estimator.__dict__, loaded.__dict__)

    # check number of numpy arrays stored on disk
    with ZipFile(io.BytesIO(dumped), "r") as input_zip:
        files = input_zip.namelist()
    # expected number of files are:
    # schema, X, X_copy, X_t, 2 vectors, 2 scalars, X_sparse = 9
    expected_files = 9
    num_files = len(files)
    assert num_files == expected_files


class NumpyDtypeObjectEstimator(BaseEstimator):
    """An estimator with a numpy array of dtype object"""

    def fit(self, X, y=None, **fit_params):
        self.obj_ = np.zeros(3, dtype=object)
        return self


def test_numpy_dtype_object_does_not_store_broken_file():
    # This addresses a specific bug where trying to store an object numpy array
    # resulted in the creation of a broken .npy file being left over. This is
    # because numpy tries to write to the file until it encounters an error and
    # raises, but then doesn't clean up said file. Before the bugfix in #150, we
    # would include that broken file in the zip archive, although we wouldn't do
    # anything with it. Here we test that no such file exists.
    estimator = NumpyDtypeObjectEstimator().fit(None)
    dumped = dumps(estimator)
    with ZipFile(io.BytesIO(dumped), "r") as input_zip:
        files = input_zip.namelist()

    # this estimator should not have any numpy file
    assert not any(file.endswith(".npy") for file in files)


def test_loads_from_str():
    # loads expects bytes, not str
    msg = "Can't load skops format from string, pass bytes"
    with pytest.raises(TypeError, match=msg):
        loads("this is a string")


def test_get_tree_unknown_type_error_msg():
    state = get_state(("hi", [123]), SaveContext(None))
    state["__loader__"] = "this_get_tree_does_not_exist"
    msg = "Can't find loader this_get_tree_does_not_exist for type builtins.tuple."
    with pytest.raises(TypeError, match=msg):
        get_tree(state, LoadContext(None))


class _BoundMethodHolder:
    """Used to test the ability to serialize and deserialize bound methods"""

    def __init__(self, object_state: str):
        # Initialize with some state to make sure state is persisted
        self.object_state = object_state
        # bind some method to this object, could be any persistable function
        self.chosen_function = np.log

    def bound_method(self, x):
        return self.chosen_function(x)

    def other_bound_method(self, x):
        # arbitrary other function, used for checking single instance loaded
        return self.chosen_function(x)


class TestPersistingBoundMethods:
    @staticmethod
    def assert_transformer_persisted_correctly(
        loaded_transformer: FunctionTransformer,
        original_transformer: FunctionTransformer,
    ):
        """Checks that a persisted and original transformer are equivalent, including
        the func passed to it
        """
        assert loaded_transformer.func.__name__ == original_transformer.func.__name__

        assert_params_equal(
            loaded_transformer.func.__self__.__dict__,
            original_transformer.func.__self__.__dict__,
        )
        assert_params_equal(loaded_transformer.__dict__, original_transformer.__dict__)

    @staticmethod
    def assert_bound_method_holder_persisted_correctly(
        original_obj: _BoundMethodHolder, loaded_obj: _BoundMethodHolder
    ):
        """Checks that the persisted and original instances of _BoundMethodHolder are
        equivalent
        """
        assert original_obj.bound_method.__name__ == loaded_obj.bound_method.__name__
        assert original_obj.chosen_function == loaded_obj.chosen_function

        assert_params_equal(original_obj.__dict__, loaded_obj.__dict__)

    def test_for_base_case_returns_as_expected(self):
        initial_state = "This is an arbitrary state"
        obj = _BoundMethodHolder(object_state=initial_state)
        bound_function = obj.bound_method
        transformer = FunctionTransformer(func=bound_function)

        loaded_transformer = loads(dumps(transformer), trusted=True)
        loaded_obj = loaded_transformer.func.__self__

        self.assert_transformer_persisted_correctly(loaded_transformer, transformer)
        self.assert_bound_method_holder_persisted_correctly(obj, loaded_obj)

    def test_when_object_is_changed_after_init_works_as_expected(self):
        # given change to object with bound method after initialisation,
        # make sure still persists correctly

        initial_state = "This is an arbitrary state"
        obj = _BoundMethodHolder(object_state=initial_state)
        obj.chosen_function = np.sqrt
        bound_function = obj.bound_method

        transformer = FunctionTransformer(func=bound_function)

        loaded_transformer = loads(dumps(transformer), trusted=True)
        loaded_obj = loaded_transformer.func.__self__

        self.assert_transformer_persisted_correctly(loaded_transformer, transformer)
        self.assert_bound_method_holder_persisted_correctly(obj, loaded_obj)

    def test_works_when_given_multiple_bound_methods_attached_to_single_instance(self):
        obj = _BoundMethodHolder(object_state="")

        transformer = FunctionTransformer(
            func=obj.bound_method, inverse_func=obj.other_bound_method
        )

        loaded_transformer = loads(dumps(transformer), trusted=True)

        # check that both func and inverse_func are from the same object instance
        loaded_0 = loaded_transformer.func.__self__
        loaded_1 = loaded_transformer.inverse_func.__self__
        assert loaded_0 is loaded_1

    @pytest.mark.xfail(reason="Failing due to circular self reference")
    def test_scipy_stats(self, tmp_path):
        from scipy import stats

        estimator = FunctionTransformer(func=stats.zipf)
        loads(dumps(estimator), trusted=True)


class CustomEstimator(BaseEstimator):
    """Estimator with np array, np scalar, and sparse matrix attribute"""

    def fit(self, X, y=None):
        self.numpy_array = np.zeros(3)
        self.numpy_scalar = np.ones(1)[0]
        self.sparse_matrix = sparse.csr_matrix(np.arange(3))
        return self


def test_dump_to_and_load_from_disk(tmp_path):
    # Test saving to and loading from disk. Functionality-wise, this is almost
    # identical to saving to and loading from memory using dumps and loads.
    # Therefore, only test functionality that is specific to dump and load.

    estimator = CustomEstimator().fit(None)
    f_name = tmp_path / "estimator.skops"
    dump(estimator, f_name)
    file = Path(f_name)
    assert file.exists()

    with ZipFile(f_name, "r") as input_zip:
        files = input_zip.namelist()

    # there should be 4 files in total, schema.json, 2 np arrays, and 1 sparse matrix
    assert len(files) == 4
    assert "schema.json" in files

    num_array_files = sum(1 for file in files if file.endswith(".npy"))
    assert num_array_files == 2
    num_sparse_files = sum(1 for file in files if file.endswith(".npz"))
    assert num_sparse_files == 1

    # check that schema is valid json by loading it
    json.loads(ZipFile(f_name).read("schema.json"))

    # load and compare the actual estimator
    loaded = load(f_name, trusted=True)
    assert_params_equal(loaded.__dict__, estimator.__dict__)


def test_disk_and_memory_are_identical(tmp_path):
    # Test that model hashes are the same for models stored on disk and in
    # memory.
    # Use a somewhat complex model.
    # fmt: off
    estimator = Pipeline([
        ("features", FeatureUnion([
            ("scaler", StandardScaler()),
            ("scaled-poly", Pipeline([
                ("polys", FeatureUnion([
                    ("poly1", PolynomialFeatures()),
                    ("poly2", PolynomialFeatures(degree=3, include_bias=False))
                ])),
                ("square-root", FunctionTransformer(np.sqrt)),
                ("scale", MinMaxScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(random_state=0, solver="liblinear")),
    ]).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
    # fmt: on

    f_name = tmp_path / "estimator.skops"
    dump(estimator, f_name)
    loaded_disk = load(f_name, trusted=True)
    loaded_memory = loads(dumps(estimator), trusted=True)

    assert joblib.hash(loaded_disk) == joblib.hash(loaded_memory)


@pytest.mark.parametrize(
    "obj",
    [
        np.array([1, 2]),
        [1, 2, 3],
        {1: 1, 2: 2},
        {1, 2, 3},
        "A string",
        np.random.RandomState(42),
    ],
)
def test_when_given_object_referenced_twice_loads_as_one_object(obj):
    an_object = {"obj_1": obj, "obj_2": obj}
    persisted_object = loads(dumps(an_object), trusted=True)

    assert persisted_object["obj_1"] is persisted_object["obj_2"]
