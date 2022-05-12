"""
scikit-learn models on HuggingFace Hub
--------------------------------------

This guide demonstrates how you can use this package to create a HuggingFace
Hub model repository based on a scikit-learn compatible model, and how to
fetch scikit-learn compatible models from the Hub and run them locally.
"""

# %%
# Imports
# =======
# First we will import everything required for the rest of this document.

import os
import pickle
from tempfile import mkdtemp, mkstemp
from uuid import uuid4

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, train_test_split

from skops import hf_hub

# %%
# Data
# ====
# Then we create some random data to train and evaluate our model.

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("X's summary: ", X.describe())
print("y's summary: ", y.describe())

# %%
# Train a Model
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

# %%
# Initialize a Model Repo
# =======================
# We now initialize a model repository locally, and push it to the hub. For
# that, we need to first store the model as a pickle file and pass it to the
# hub tools.

_, pkl_name = mkstemp(prefix="skops")
with open(pkl_name, mode="bw") as f:
    pickle.dump(model, file=f)

local_repo = mkdtemp(prefix="skops")
hf_hub.init(model=pkl_name, requirements=["scikit-learn"], destination=local_repo)

# %%
# We can no see what the contents of the created local repo are:
print(os.listdir(local_repo))


# %%
# Push to Hub
# ===========
# And finally, we can push the model to the hub.
hf_hub.push(repo_id=f"hf_hub_example-{uuid4()}", source=local_repo)
