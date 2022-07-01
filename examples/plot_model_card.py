"""
scikit-learn model cards
--------------------------------------

This guide demonstrates how you can use this package to create a model card on a
scikit-learn compatible model and save it.
"""

# %% Imports
# =======
# First we will import everything required for the rest of this document.

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, train_test_split

from skops import card

# %% Data
# ====
# Then we create some random data to train and evaluate our model.

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("X's summary: ", X.describe())
print("y's summary: ", y.describe())

# %% Train a Model
# =============
# Using the above data, we train a model. To select the model, we use
# :class:`~sklearn.model_selection.HalvingGridSearchCV` with a parameter grid
# over :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.

param_grid = {
    "max_leaf_nodes": [5, 10, 15],
    "max_depth": [2, 5, 10],
}

model = HalvingGridSearchCV(
    estimator=HistGradientBoostingClassifier(),
    param_grid=param_grid,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)
model.score(X_test, y_test)

# %% Create a model card
# =======================
# We now create a model card, set couple of attributes and save it.

model_card = card.create_model_card(model)
model_card.limitations = "This model is not ready to be used in production."
model_card.license = "afl3.0"
model_card.save("./README.md")
