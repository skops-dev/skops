import sys

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils._testing import assert_allclose_dense_sparse

from skops.io import dumps, loads
from skops.io.tests._utils import assert_params_equal

ATOL = 1e-6 if sys.platform == "darwin" else 1e-7
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


def check_estimator(estimator, trusted):
    loaded = loads(dumps(estimator), trusted=trusted)
    assert_params_equal(estimator.get_params(), loaded.get_params())


def check_method_outputs(estimator, X, trusted):
    loaded = loads(dumps(estimator), trusted=trusted)
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
            X_out1 = getattr(estimator, method)(X)
            X_out2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_out1, X_out2, err_msg=err_msg, atol=ATOL)


class TestLightGBM:
    """Tests for LGBMClassifier, LGBMRegressor, LGBMRanker"""

    @pytest.fixture(autouse=True)
    def lgbm(self):
        lgbm = pytest.importorskip("lightgbm")
        return lgbm

    @pytest.fixture
    def trusted(self):
        return [
            "collections.defaultdict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "lightgbm.sklearn.LGBMRegressor",
            "lightgbm.sklearn.LGBMRanker",
            "numpy.int64",
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
        check_estimator(estimator, trusted=trusted)

        X, y = clf_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_regressor(self, lgbm, regr_data, trusted, boosting_type):
        kw = {}
        if boosting_type == "rf":
            kw["bagging_fraction"] = 0.5
            kw["bagging_freq"] = 2

        estimator = lgbm.LGBMRegressor(boosting_type=boosting_type, **kw)
        check_estimator(estimator, trusted=trusted)

        X, y = regr_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("boosting_type", boosting_types)
    def test_ranker(self, lgbm, rank_data, trusted, boosting_type):
        kw = {}
        if boosting_type == "rf":
            kw["bagging_fraction"] = 0.5
            kw["bagging_freq"] = 2

        estimator = lgbm.LGBMRanker(boosting_type=boosting_type, **kw)
        check_estimator(estimator, trusted=trusted)

        X, y, group = rank_data
        estimator.fit(X, y, group=group)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)


class TestXGBoost:
    """Tests for XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor, XGBRanker

    Known bugs:

    - When initialzing with tree_method=None, its value resolves to "exact", but
      after loading, it resolves to "auto".
    - When initializing with gpu_id=None, its value resolves to 0, but after
      loading, it resolves to -1.

    This can be verified like this:

    >>> import xgboost
    >>> estimator = xgboost.XGBClassifier(tree_method=None)
    >>> X, y = [[0, 1], [2, 3]], [0, 1]
    >>> estimator.fit(X, y)
    XGBClassifier(...)
    >>> print(estimator.tree_method)
    None
    >>> print(estimator.get_params()["tree_method"])
    exact
    >>> # after save/load roundtrip, values of get_params change
    >>> import tempfile
    >>> tmp_file = f"{tempfile.mkdtemp()}.ubj"
    >>> estimator.save_model(tmp_file)
    >>> estimator.load_model(tmp_file)
    >>> print(estimator.tree_method)
    None
    >>> print(estimator.get_params()["tree_method"])
    auto

    >>> estimator = xgboost.XGBClassifier(tree_method='gpu_hist', booster='gbtree')
    >>> estimator.fit(X, y)
    XGBClassifier(...)
    >>> print(estimator.gpu_id)
    None
    >>> print(estimator.get_params()["gpu_id"])
    0
    >>> # after save/load roundtrip, values of get_params change
    >>> estimator.save_model(tmp_file)
    >>> # for gpu_id, the estimator needs to be re-initialized for the effect to occur
    >>> estimator = xgboost.XGBClassifier()
    >>> estimator.load_model(tmp_file)
    >>> print(estimator.gpu_id)
    None
    >>> print(estimator.get_params()["gpu_id"])
    -1

    As can be seen, this has nothing to do with skops but is a bug/feature of
    xgboost. We assume that this has no practical consequences and thus avoid
    testing these cases.

    """

    @pytest.fixture(autouse=True)
    def xgboost(self):
        xgboost = pytest.importorskip("xgboost")
        return xgboost

    @pytest.fixture
    def trusted(self):
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
    tree_methods = ["approx", "hist", "gpu_hist", "auto"]

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_classifier(self, xgboost, clf_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBClassifier(
            booster=booster, tree_method=tree_method, gpu_id=-1
        )
        check_estimator(estimator, trusted=trusted)

        X, y = clf_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_regressor(self, xgboost, regr_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRegressor(
            booster=booster, tree_method=tree_method, gpu_id=-1
        )
        check_estimator(estimator, trusted=trusted)

        X, y = regr_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_rf_classifier(self, xgboost, clf_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRFClassifier(
            booster=booster, tree_method=tree_method, gpu_id=-1
        )
        check_estimator(estimator, trusted=trusted)

        X, y = clf_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_rf_regressor(self, xgboost, regr_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRFRegressor(
            booster=booster, tree_method=tree_method, gpu_id=-1
        )
        check_estimator(estimator, trusted=trusted)

        X, y = regr_data
        estimator.fit(X, y)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)

    @pytest.mark.parametrize("booster", boosters)
    @pytest.mark.parametrize("tree_method", tree_methods)
    def test_ranker(self, xgboost, rank_data, trusted, booster, tree_method):
        if (booster == "gblinear") and (tree_method != "approx"):
            # This parameter combination is not supported in XGBoost
            return

        estimator = xgboost.XGBRanker(
            booster=booster, tree_method=tree_method, gpu_id=-1
        )
        check_estimator(estimator, trusted=trusted)

        X, y, group = rank_data
        estimator.fit(X, y, group=group)
        check_estimator(estimator, trusted=trusted)
        check_method_outputs(estimator, X, trusted=trusted)
