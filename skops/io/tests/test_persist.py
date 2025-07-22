import importlib
import inspect
import io
import json
import operator
import sys
import warnings
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

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
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.utils._testing import SkipTest, set_random_state
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
    _get_check_estimator_ids,
)
from sklearn.utils.fixes import parse_version, sp_version

import skops
from skops.io import dump, dumps, get_untrusted_types, load, loads
from skops.io._audit import NODE_TYPE_MAPPING, get_tree
from skops.io._sklearn import UNSUPPORTED_TYPES
from skops.io._trusted_types import (
    CONTAINER_TYPE_NAMES,
    NUMPY_DTYPE_TYPE_NAMES,
    NUMPY_UFUNC_TYPE_NAMES,
    PRIMITIVE_TYPE_NAMES,
    SCIPY_UFUNC_TYPE_NAMES,
    SKLEARN_ESTIMATOR_TYPE_NAMES,
)
from skops.io._utils import LoadContext, SaveContext, _get_state, get_state, gettype
from skops.io.exceptions import UnsupportedTypeException, UntrustedTypesFoundException
from skops.io.tests._utils import assert_method_outputs_equal, assert_params_equal
from skops.utils._fixes import construct_instances, get_tags

# Default settings for X
N_SAMPLES = 120
N_FEATURES = 20


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
                if name == "QuantileRegressor" and sp_version >= parse_version(
                    "1.11.0"
                ):
                    # The solver "interior-point" (the default solver in
                    # scikit-learn < 1.4.0) is not available in scipy >= 1.11.0. The
                    # default solver will be "highs" from scikit-learn >= 1.4.0.
                    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
                    estimators = construct_instances(partial(Estimator, solver="highs"))
                else:
                    estimators = construct_instances(Estimator)

                for estimator in estimators:
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
                    if "skewedness" in estimator.get_params():
                        # prevent data generation errors for SkewedChi2Sampler
                        estimator.set_params(skewedness=20)
                    if estimator.__class__.__name__ == "GraphicalLasso":
                        # prevent data generation errors
                        estimator.set_params(alpha=1)
                    if estimator.__class__.__name__ == "GraphicalLassoCV":
                        # prevent data generation errors
                        estimator.set_params(alphas=[1, 2])
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


def _tested_ufuncs():
    for full_name in SCIPY_UFUNC_TYPE_NAMES + NUMPY_UFUNC_TYPE_NAMES:
        module_name, _, ufunc_name = full_name.rpartition(".")
        yield gettype(module_name=module_name, cls_or_func=ufunc_name)


def _tested_types():
    for full_name in (
        PRIMITIVE_TYPE_NAMES + NUMPY_DTYPE_TYPE_NAMES + CONTAINER_TYPE_NAMES
    ):
        module_name, _, type_name = full_name.rpartition(".")
        yield gettype(module_name=module_name, cls_or_func=type_name)


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
                # Get the first instance directly from the generator
                estimators = construct_instances(Estimator)
                # with the kind of data we pass, it needs to be 1 for the few
                # estimators which have this.
                for estimator in estimators:
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


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_non_fitted(estimator):
    """Check that non-fitted estimators can be persisted."""
    dumped = dumps(estimator)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
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
    X = _enforce_estimator_tags_X(estimator, X)

    tags = get_tags(estimator)

    if tags.input_tags.pairwise:
        # return a square matrix of size N_FEATURES x N_FEATURES and positive values
        return np.abs(X[:N_FEATURES, :N_FEATURES]), y[:N_FEATURES]

    if tags.input_tags.positive_only:
        # Some models require positive X
        return np.abs(X), y

    if tags.input_tags.two_d_array:
        return X, y

    if tags.input_tags.one_d_array:
        if X.ndim == 1:
            return X, y
        return X[:, 0], y

    if tags.input_tags.three_d_array:
        return load_sample_images().images[1], None

    if tags.target_tags.one_d_labels:
        # model only expects y
        return y, None

    if tags.target_tags.two_d_labels:
        return [(1, 2), (3,)], None

    if tags.input_tags.categorical:
        X = [["Male", 1], ["Female", 3], ["Female", 2]]
        y = y[: len(X)] if tags.target_tags.required else None
        return X, y

    if tags.input_tags.dict:
        return [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}], None

    if tags.input_tags.string:
        return [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ], None

    if tags.input_tags.sparse:
        # TfidfTransformer in sklearn 0.24 needs this
        return sparse.csr_matrix(X), y

    raise ValueError(f"Unsupported X type for estimator: {tags.input_tags}")


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator):
    """Check that fitted estimators can be persisted and return the right results."""
    set_random_state(estimator, random_state=0)

    X, y = get_input(estimator)
    if get_tags(estimator).requires_fit:
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

    assert not any(type_ in SKLEARN_ESTIMATOR_TYPE_NAMES for type_ in untrusted_types)
    assert not any(type_ in SCIPY_UFUNC_TYPE_NAMES for type_ in untrusted_types)
    assert not any(type_ in NUMPY_UFUNC_TYPE_NAMES for type_ in untrusted_types)
    assert not any(type_ in NUMPY_DTYPE_TYPE_NAMES for type_ in untrusted_types)
    assert_method_outputs_equal(estimator, loaded, X)


@pytest.mark.parametrize(
    "ufunc", _tested_ufuncs(), ids=SCIPY_UFUNC_TYPE_NAMES + NUMPY_UFUNC_TYPE_NAMES
)
def test_can_trust_ufuncs(ufunc):
    dumped = dumps(ufunc)
    untrusted_types = get_untrusted_types(data=dumped)
    assert len(untrusted_types) == 0


@pytest.mark.parametrize(
    "type_",
    _tested_types(),
    ids=PRIMITIVE_TYPE_NAMES + NUMPY_DTYPE_TYPE_NAMES + CONTAINER_TYPE_NAMES,
)
def test_can_trust_types(type_):
    dumped = dumps(type_)
    untrusted_types = get_untrusted_types(data=dumped)
    assert len(untrusted_types) == 0


@pytest.mark.parametrize(
    "estimator", _unsupported_estimators(), ids=_get_check_estimator_ids
)
def test_unsupported_type_raises(estimator):
    """Estimators that are known to fail should raise an error"""
    set_random_state(estimator, random_state=0)
    X, y = get_input(estimator)
    if get_tags(estimator).requires_fit:
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

    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)

    if hasattr(est, "__dict__"):
        # what to do if object has no __dict__, like Generator?
        assert_params_equal(est.__dict__, loaded.__dict__)

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
    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
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

    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
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
    assert schema["protocol"] == skops.io._protocol.PROTOCOL
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
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
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
        get_tree(state, LoadContext(None, -1), trusted=False)


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

        dumped = dumps(transformer)
        untrusted_types = get_untrusted_types(data=dumped)
        loaded_transformer = loads(dumped, trusted=untrusted_types)
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

        dumped = dumps(transformer)
        untrusted_types = get_untrusted_types(data=dumped)
        loaded_transformer = loads(dumped, trusted=untrusted_types)
        loaded_obj = loaded_transformer.func.__self__

        self.assert_transformer_persisted_correctly(loaded_transformer, transformer)
        self.assert_bound_method_holder_persisted_correctly(obj, loaded_obj)

    def test_works_when_given_multiple_bound_methods_attached_to_single_instance(self):
        obj = _BoundMethodHolder(object_state="")

        transformer = FunctionTransformer(
            func=obj.bound_method, inverse_func=obj.other_bound_method
        )

        dumped = dumps(transformer)
        untrusted_types = get_untrusted_types(data=dumped)
        loaded_transformer = loads(dumped, trusted=untrusted_types)

        # check that both func and inverse_func are from the same object instance
        loaded_0 = loaded_transformer.func.__self__
        loaded_1 = loaded_transformer.inverse_func.__self__
        assert loaded_0 is loaded_1

    @pytest.mark.xfail(reason="Failing due to circular self reference", strict=True)
    def test_scipy_stats(self, tmp_path):
        from scipy import stats

        estimator = FunctionTransformer(func=stats.zipf)
        dumped = dumps(estimator)
        untrusted_types = get_untrusted_types(data=dumped)
        loads(dumped, trusted=untrusted_types)


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
    loaded = load(f_name, trusted=get_untrusted_types(file=f_name))
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
        ("clf", LogisticRegression(random_state=0, solver="saga")),
    ]).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
    # fmt: on

    f_name = tmp_path / "estimator.skops"
    dump(estimator, f_name)
    loaded_disk = load(f_name, trusted=get_untrusted_types(file=f_name))
    loaded_memory = loads(
        dumps(estimator), trusted=get_untrusted_types(data=dumps(estimator))
    )

    assert joblib.hash(loaded_disk) == joblib.hash(loaded_memory)


def test_dump_and_load_with_file_wrapper(tmp_path):
    # The idea here is to make it possible to use dump and load with a file
    # wrapper, i.e. using 'with open(...)'. This makes it easier to search and
    # replace pickle dump and load by skops dump and load.
    estimator = LogisticRegression().fit([[0, 1], [2, 3], [4, 5]], [0, 1, 1])
    f_name = tmp_path / "estimator.skops"

    with open(f_name, "wb") as f:
        dump(estimator, f)
    with open(f_name, "rb") as f:
        loaded = load(f, trusted=get_untrusted_types(file=f_name))

    assert_params_equal(loaded.__dict__, estimator.__dict__)


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
    dumped = dumps(an_object)
    untrusted_types = get_untrusted_types(data=dumped)
    persisted_object = loads(dumped, trusted=untrusted_types)

    assert persisted_object["obj_1"] is persisted_object["obj_2"]


class EstimatorWithBytes(BaseEstimator):
    def fit(self, X, y, **fit_params):
        self.bytes_ = b"hello"
        self.bytearray_ = bytearray([0, 1, 2, 253, 254, 255])
        return self


def test_estimator_with_bytes():
    est = EstimatorWithBytes().fit(None, None)
    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
    assert_params_equal(est.__dict__, loaded.__dict__)


def test_estimator_with_bytes_files_created(tmp_path):
    est = EstimatorWithBytes().fit(None, None)
    f_name = tmp_path / "estimator.skops"
    dump(est, f_name)
    file = Path(f_name)
    assert file.exists()

    with ZipFile(f_name, "r") as input_zip:
        files = input_zip.namelist()
    bin_files = [file for file in files if file.endswith(".bin")]
    assert len(bin_files) == 2


OPERATORS = [
    ("add", partial(operator.add, 0)),
    ("sub", partial(operator.sub, 0)),
    ("mul", partial(operator.mul, 1)),
    ("truediv", partial(operator.truediv, 1)),
    ("pow", partial(operator.pow, 1)),
    ("matmul", partial(operator.matmul, np.eye(N_SAMPLES))),
    ("iadd", partial(operator.iadd, 1)),
    ("isub", partial(operator.isub, 1)),
    ("imul", partial(operator.imul, 1)),
    ("itruediv", partial(operator.itruediv, 1)),
    ("ipow", partial(operator.ipow, 1)),
    # note: inplace matmul is not supported by numpy
    ("ge", partial(operator.ge, 0)),
    ("gt", partial(operator.gt, 0)),
    ("le", partial(operator.le, 0)),
    ("lt", partial(operator.lt, 0)),
    ("eq", partial(operator.eq, 0)),
    ("neg", operator.neg),
    ("attrgetter", operator.attrgetter("real")),
    ("attrgetter", operator.attrgetter("real", "real")),
    ("itemgetter", operator.itemgetter(None)),
    ("itemgetter", operator.itemgetter(None, None)),
    ("methodcaller", operator.methodcaller("round")),
    ("methodcaller", operator.methodcaller("round", 2)),
]

if sys.version_info >= (3, 11):
    OPERATORS.append(("call", partial(operator.call, len)))


@pytest.mark.parametrize("op", OPERATORS)
def test_persist_operator(op):
    # Test a couple of functions from the operator module. This is not an
    # exhaustive list but rather the most plausible functions. To check all
    # operators would require specific tests, not a generic one like this.
    # Fixes #283

    _, func = op
    # unfitted
    est = FunctionTransformer(func)
    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
    assert_params_equal(est.__dict__, loaded.__dict__)

    # fitted
    X, y = get_input(est)
    est.fit(X, y)
    dumped = dumps(est)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)
    assert_params_equal(est.__dict__, loaded.__dict__)

    # Technically, we don't need to call transform. However, if this is skipped,
    # there is a danger to not define the function sufficiently, which could
    # lead to the test passing but being useless in practice -- e.g. for
    # func=operator.methodcaller, the test would pass without the fix for #283
    # but it would be useless in practice, since methodcaller is not properly
    # instantiated.
    est.transform(X)


@pytest.mark.parametrize("op", OPERATORS)
def test_persist_operator_raises_untrusted(op):
    # check that operators are not trusted by default, because at least some of
    # them could perform unsafe operations
    name, func = op
    est = FunctionTransformer(func)
    with pytest.raises(UntrustedTypesFoundException, match=name):
        loads(dumps(est), trusted=None)


def dummy_func(X):
    return X


@pytest.mark.parametrize("func", [np.sqrt, len, special.exp10, dummy_func])
def test_persist_function(func):
    estimator = FunctionTransformer(func=func)
    X, y = [0, 1], [2, 3]
    estimator.fit(X, y)

    dumped = dumps(estimator)
    untrusted_types = get_untrusted_types(data=dumped)
    loaded = loads(dumped, trusted=untrusted_types)

    # check that loaded estimator is identical
    assert_params_equal(estimator.__dict__, loaded.__dict__)
    assert_method_outputs_equal(estimator, loaded, X)


def test_compression_level():
    # Test that setting the compression to zlib and specifying a
    # compressionlevel reduces the dumped size.
    model = TfidfVectorizer().fit([np.__doc__])
    dumped_raw = dumps(model)
    dumped_compressed = dumps(model, compression=ZIP_DEFLATED, compresslevel=9)
    # This reduces the size substantially
    assert len(dumped_raw) > 5 * len(dumped_compressed)


@pytest.mark.parametrize("call_has_canonical_format", [False, True])
def test_sparse_matrix(call_has_canonical_format):
    # see https://github.com/skops-dev/skops/pull/375

    # note: this behavior is already implicitly tested by sklearn estimators
    # that use sparse matrices under the hood (tfidf) but it is better to check
    # the behavior explicitly
    x = sparse.csr_matrix((3, 4))
    if call_has_canonical_format:
        x.has_canonical_format

    dumped = dumps(x)
    untrusted_types = get_untrusted_types(data=dumped)
    y = loads(dumped, trusted=untrusted_types)

    assert_params_equal(x.__dict__, y.__dict__)


def test_trusted_bool_raises(tmp_path):
    """Make sure trusted=True is no longer accepted."""
    f_name = tmp_path / "file.skops"
    dump(10, f_name)
    with pytest.raises(TypeError, match="trusted must be a list of strings"):
        load(f_name, trusted=True)  # type: ignore

    with pytest.raises(TypeError, match="trusted must be a list of strings"):
        loads(dumps(10), trusted=True)  # type: ignore


def test_defaultdict():
    """Test that we correctly restore a defaultdict."""
    obj = defaultdict(set)
    obj["foo"] = "bar"
    obj_loaded = loads(dumps(obj))
    assert obj_loaded == obj
    assert obj_loaded.default_factory == obj.default_factory


@pytest.mark.parametrize("cls", [dict, OrderedDict])
def test_dictionary(cls):
    obj = cls({1: 5, 6: 3, 2: 4})
    loaded_obj = loads(dumps(obj))
    assert obj == loaded_obj
    assert type(obj) is cls


def test_datetime():
    obj = datetime.now()
    loaded_obj = loads(dumps(obj), trusted=[datetime])
    assert obj == loaded_obj
    assert type(obj) is datetime


def test_slice():
    obj = slice(1, 2, 3)
    loaded_obj = loads(dumps(obj))
    assert obj == loaded_obj
    assert type(obj) is slice


# This class is here as opposed to inside the test because it needs to be importable.
reduce_calls = 0


class CustomReduce:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        global reduce_calls
        reduce_calls += 1
        return (type(self), (self.value,))


def test_custom_reduce():
    obj = CustomReduce(10)
    dumped = dumps(obj)

    # make sure __reduce__ is called, once.
    assert reduce_calls == 1

    with pytest.raises(TypeError, match="Untrusted types found"):
        loads(dumped)

    loaded_obj = loads(dumps(obj), trusted=[CustomReduce])
    assert obj.value == loaded_obj.value
