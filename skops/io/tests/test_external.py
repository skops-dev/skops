"""Test persistence of "external" packages

Packages that are not builtins, standard lib, numpy, scipy, or scikit-learn.

Testing:

- persistence of unfitted models
- persistence of fitted models
- visualization of dumped models

with a range of hyperparameters.

"""

from unittest.mock import Mock, patch

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline

from skops.io import dumps, loads, visualize
from skops.io.tests._utils import assert_method_outputs_equal, assert_params_equal

# Default settings for generated data
N_SAMPLES = 30
N_FEATURES = 10
N_CLASSES = 4  # for classification only


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_classes=N_CLASSES,
        n_features=N_FEATURES,
        random_state=0,
        n_redundant=1,
        n_informative=N_FEATURES - 1,
    )
    return X, y


@pytest.fixture(scope="module")
def regr_data():
    X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def rank_data(clf_data):
    X, y = clf_data
    group = [10 for _ in range(N_SAMPLES // 10)]
    n = sum(group)
    if N_SAMPLES > n:
        group[-1] += N_SAMPLES - n
    assert sum(group) == N_SAMPLES
    return X, y, group


class TestLightGBM:
    """Tests for LGBMClassifier, LGBMRegressor, LGBMRanker"""

    @pytest.fixture(autouse=True)
    def capture_stdout(self):
        # Mock print and rich.print so that running these tests with pytest -s
        # does not spam stdout. Other, more common methods of suppressing
        # printing to stdout don't seem to work, perhaps because of pytest.
        with patch("builtins.print", Mock()), patch("rich.print", Mock()):
            yield

    @pytest.fixture(autouse=True)
    def lgbm(self):
        lgbm = pytest.importorskip("lightgbm")
        return lgbm

    @pytest.fixture
    def trusted(self):
        # TODO: adjust once more types are trusted by default
        return [
            "collections.defaultdict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "lightgbm.sklearn.LGBMRegressor",
            "lightgbm.sklearn.LGBMRanker",
            "sklearn.preprocessing._label.LabelEncoder",
        ]

    boosting_types = ["gbdt", "dart", "goss", "rf"]

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_classifier(self, lgbm, clf_data, trusted, boosting_type):
        kw = {}
        if boosting_type == "rf":
            kw["bagging_fraction"] = 0.5
            kw["bagging_freq"] = 2

        estimator = lgbm.LGBMClassifier(boosting_type=boosting_type, **kw)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = clf_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_regressor(self, lgbm, regr_data, trusted, boosting_type):
        kw = {}
        if boosting_type == "rf":
            kw["bagging_fraction"] = 0.5
            kw["bagging_freq"] = 2

        estimator = lgbm.LGBMRegressor(boosting_type=boosting_type, **kw)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = regr_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_ranker(self, lgbm, rank_data, trusted, boosting_type):
        kw = {}
        if boosting_type == "rf":
            kw["bagging_fraction"] = 0.5
            kw["bagging_freq"] = 2

        estimator = lgbm.LGBMRanker(boosting_type=boosting_type, **kw)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y, group = rank_data
        estimator.fit(X, y, group=group)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)


class TestXGBoost:
    """Tests for XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor, XGBRanker

    Known bugs:

    - When initialzing with tree_method=None, its value resolves to "exact", but
      after loading, it resolves to "auto" when calling get_params().
    - When initializing with tree_method='gpu_hist' and gpu_id=None, the
      latter's value resolves to 0, but after loading, it resolves to -1, when
      calling get_params()

    These discrepancies occur regardless of skops, so they're a problem in
    xgboost itself. We assume that this has no practical consequences and thus
    avoid testing these cases. See https://github.com/dmlc/xgboost/issues/8596

    """

    @pytest.fixture(autouse=True)
    def capture_stdout(self):
        # Mock print and rich.print so that running these tests with pytest -s
        # does not spam stdout. Other, more common methods of suppressing
        # printing to stdout don't seem to work, perhaps because of pytest.
        with patch("builtins.print", Mock()), patch("rich.print", Mock()):
            yield

    @pytest.fixture(autouse=True)
    def xgboost(self):
        xgboost = pytest.importorskip("xgboost")
        return xgboost

    @pytest.fixture
    def trusted(self):
        # TODO: adjust once more types are trusted by default
        return [
            "xgboost.sklearn.XGBClassifier",
            "xgboost.sklearn.XGBRegressor",
            "xgboost.sklearn.XGBRFClassifier",
            "xgboost.sklearn.XGBRFRegressor",
            "xgboost.sklearn.XGBRanker",
            "builtins.bytearray",
            "xgboost.core.Booster",
        ]

    boosters = ["gbtree", "gblinear", "dart"]
    tree_methods = ["approx", "hist", "auto"]

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_classifier(self, xgboost, clf_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBClassifier(booster=booster, tree_method=tree_method)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = clf_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_regressor(self, xgboost, regr_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRegressor(booster=booster, tree_method=tree_method)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = regr_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_rf_classifier(self, xgboost, clf_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRFClassifier(booster=booster, tree_method=tree_method)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = clf_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_rf_regressor(self, xgboost, regr_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRFRegressor(booster=booster, tree_method=tree_method)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = regr_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_ranker(self, xgboost, rank_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRanker(booster=booster, tree_method=tree_method)
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y, group = rank_data
        estimator.fit(X, y, group=group)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)


class TestCatboost:
    """Tests for CatBoostClassifier, CatBoostRegressor, and CatBoostRanker"""

    @pytest.fixture(autouse=True)
    def capture_stdout(self):
        # Mock print and rich.print so that running these tests with pytest -s
        # does not spam stdout. Other, more common methods of suppressing
        # printing to stdout don't seem to work, perhaps because of pytest.
        with patch("builtins.print", Mock()), patch("rich.print", Mock()):
            yield

    # CatBoost data is a little different so that it works as categorical data
    @pytest.fixture(scope="module")
    def cb_clf_data(self, clf_data):
        X, y = clf_data
        X = (X - X.min()).astype(int)
        return X, y

    @pytest.fixture(scope="module")
    def cb_regr_data(self, regr_data):
        X, y = regr_data
        X = (X - X.min()).astype(int)
        return X, y

    @pytest.fixture(scope="module")
    def cb_rank_data(self, rank_data):
        X, y, group = rank_data
        X = (X - X.min()).astype(int)
        group_id = sum([[i] * n for i, n in enumerate(group)], [])
        return X, y, group_id

    @pytest.fixture(autouse=True)
    def catboost(self):
        catboost = pytest.importorskip("catboost")
        return catboost

    @pytest.fixture
    def trusted(self):
        # TODO: adjust once more types are trusted by default
        return [
            "builtins.bytes",
            "catboost.core.CatBoostClassifier",
            "catboost.core.CatBoostRegressor",
            "catboost.core.CatBoostRanker",
        ]

    boosting_types = ["Ordered", "Plain"]

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_classifier(self, catboost, cb_clf_data, trusted, boosting_type):
        estimator = catboost.CatBoostClassifier(
            verbose=False, boosting_type=boosting_type, allow_writing_files=False
        )
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = cb_clf_data
        estimator.fit(X, y, cat_features=[0, 1])
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_regressor(self, catboost, cb_regr_data, trusted, boosting_type):
        estimator = catboost.CatBoostRegressor(
            verbose=False, boosting_type=boosting_type, allow_writing_files=False
        )
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = cb_regr_data
        estimator.fit(X, y, cat_features=[0, 1])
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_ranker(self, catboost, cb_rank_data, trusted, boosting_type):
        estimator = catboost.CatBoostRanker(
            verbose=False, boosting_type=boosting_type, allow_writing_files=False
        )
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y, group_id = cb_rank_data
        estimator.fit(X, y, cat_features=[0, 1], group_id=group_id)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)


class TestQuantileForest:
    """Tests for RandomForestQuantileRegressor and ExtraTreesQuantileRegressor"""

    @pytest.fixture(autouse=True)
    def capture_stdout(self):
        # Mock print and rich.print so that running these tests with pytest -s
        # does not spam stdout. Other, more common methods of suppressing
        # printing to stdout don't seem to work, perhaps because of pytest.
        with patch("builtins.print", Mock()), patch("rich.print", Mock()):
            yield

    @pytest.fixture(autouse=True)
    def quantile_forest(self):
        quantile_forest = pytest.importorskip("quantile_forest")
        return quantile_forest

    @pytest.fixture
    def trusted(self):
        # TODO: adjust once more types are trusted by default
        return [
            "quantile_forest._quantile_forest.RandomForestQuantileRegressor",
            "quantile_forest._quantile_forest.ExtraTreesQuantileRegressor",
            "quantile_forest._quantile_forest_fast.QuantileForest",
        ]

    tree_methods = [
        "RandomForestQuantileRegressor",
        "ExtraTreesQuantileRegressor",
    ]

    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_quantile_forest(self, quantile_forest, regr_data, trusted, tree_method):
        cls = getattr(quantile_forest, tree_method)
        estimator = cls()
        loaded = loads(dumps(estimator), trusted=trusted)
        assert_params_equal(estimator.get_params(), loaded.get_params())

        X, y = regr_data
        estimator.fit(X, y)
        dumped = dumps(estimator)
        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(estimator, loaded, X)

        visualize(dumped, trusted=trusted)


class TestSciKeras:
    """Tests for SciKerasRegressor and SciKerasClassifier"""

    @pytest.fixture(autouse=True)
    def capture_stdout(self):
        # Mock print and rich.print so that running these tests with pytest -s
        # does not spam stdout. Other, more common methods of suppressing
        # printing to stdout don't seem to work, perhaps because of pytest.
        with patch("builtins.print", Mock()), patch("rich.print", Mock()):
            yield

    @pytest.fixture(autouse=True)
    def tensorflow(self):
        tensorflow = pytest.importorskip("tensorflow")
        return tensorflow

    @pytest.fixture
    def trusted(self):
        return [
            "_thread.RLock",
            "builtins.weakref",
            "collections.Counter",
            "collections.OrderedDict",
            "collections.defaultdict",
            "inspect.FullArgSpec",
            "inspect.Signature",
            "keras.src.activations.sigmoid",
            "keras.src.backend.RandomGenerator",
            "keras.src.callbacks.History",
            "keras.src.engine.compile_utils.LossesContainer",
            "keras.src.engine.compile_utils.MetricsContainer",
            "keras.src.engine.input_layer.InputLayer",
            "keras.src.engine.input_spec.InputSpec",
            "keras.src.engine.keras_tensor.KerasTensor",
            "keras.src.engine.node.KerasHistory",
            "keras.src.engine.node.Node",
            "keras.src.engine.sequential.Sequential",
            "keras.src.engine.training.train_function",
            "keras.src.initializers.initializers.GlorotUniform",
            "keras.src.initializers.initializers.Zeros",
            "keras.src.layers.core.dense.Dense",
            "keras.src.losses.LossFunctionWrapper",
            "keras.src.losses.binary_crossentropy",
            "keras.src.metrics.base_metric.Mean",
            "keras.src.metrics.base_metric.method",
            "keras.src.mixed_precision.policy.Policy",
            "keras.src.optimizers.legacy.rmsprop.RMSprop",
            "keras.src.optimizers.utils.<lambda>",
            "keras.src.optimizers.utils.all_reduce_sum_gradients",
            "keras.src.saving.serialization_lib.Config",
            "keras.src.utils.layer_utils.CallFunctionSpec",
            "keras.src.utils.metrics_utils.Reduction",
            "keras.src.utils.object_identity.ObjectIdentityDictionary",
            "numpy.dtype",
            "scikeras.utils.transformers.ClassifierLabelEncoder",
            "scikeras.utils.transformers.TargetReshaper",
            "scikeras.wrappers.KerasClassifier",
            "tensorflow.core.function.capture.capture_container.FunctionCaptures",
            "tensorflow.core.function.capture.capture_container.MutationAwareDict",
            "tensorflow.core.function.polymorphism.function_cache.FunctionCache",
            "tensorflow.core.function.polymorphism.function_type.FunctionType",
            "tensorflow.python.checkpoint.checkpoint.Checkpoint",
            "tensorflow.python.checkpoint.checkpoint.TrackableSaver",
            "tensorflow.python.checkpoint.graph_view.ObjectGraphView",
            "tensorflow.python.data.ops.iterator_ops.IteratorSpec",
            "tensorflow.python.eager.polymorphic_function.atomic_function.AtomicFunction",
            "tensorflow.python.eager.polymorphic_function.function_spec.FunctionSpec",
            "tensorflow.python.eager.polymorphic_function.monomorphic_function.ConcreteFunction",
            "tensorflow.python.eager.polymorphic_function.monomorphic_function.ConcreteFunctionGarbageCollector",
            "tensorflow.python.eager.polymorphic_function.monomorphic_function._DelayedRewriteGradientFunctions",
            "tensorflow.python.eager.polymorphic_function.polymorphic_function.Function",
            "tensorflow.python.eager.polymorphic_function.polymorphic_function.UnliftedInitializerVariable",
            "tensorflow.python.eager.polymorphic_function.tracing_compiler.TracingCompiler",
            "tensorflow.python.framework.dtypes.DType",
            "tensorflow.python.framework.func_graph.FuncGraph",
            "tensorflow.python.framework.ops.EagerTensor",
            "tensorflow.python.framework.tensor.TensorSpec",
            "tensorflow.python.framework.tensor_shape.TensorShape",
            "tensorflow.python.ops.resource_variable_ops.ResourceVariable",
            "tensorflow.python.ops.variables.VariableAggregation",
            "tensorflow.python.ops.variables.VariableAggregationV2",
            "tensorflow.python.ops.variables.VariableSynchronization",
            "tensorflow.python.trackable.base.TrackableReference",
            "tensorflow.python.trackable.base.WeakTrackableReference",
            "tensorflow.python.trackable.data_structures.dict",
            "tensorflow.python.util.object_identity.ObjectIdentitySet",
            "tensorflow.python.util.tf_decorator.TFDecorator",
            "weakref.WeakKeyDictionary",
            "weakref.remove",
        ]

    def test_dumping_model(self, tensorflow, trusted):
        # This simplifies the basic usage tutorial from https://adriangb.com/scikeras/stable/notebooks/Basic_Usage.html

        n_features_in_ = 20
        model = tensorflow.keras.models.Sequential()
        model.add(tensorflow.keras.layers.Input(shape=(n_features_in_,)))
        model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))

        from scikeras.wrappers import KerasClassifier

        clf = KerasClassifier(model=model, loss="binary_crossentropy")

        pipeline = Pipeline([("classifier", clf)])

        dumps(clf)
        dumps(pipeline)

        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        clf.fit(X, y)
        dumped = dumps(clf)

        loaded = loads(dumped, trusted=trusted)
        assert_method_outputs_equal(clf, loaded, X)
