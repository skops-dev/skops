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
            "collections.OrderedDict",
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
    def catboost(self):
        """Skip all tests in this class if catboost is not available."""
        try:
            catboost = pytest.importorskip("catboost")
        except (ImportError, ValueError):  # ValueError for numpy2 incompatibility
            pytest.skip("Catboost not available or incompatible")
        return catboost

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
