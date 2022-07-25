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

import json
import os
import pickle
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from uuid import uuid4

import sklearn
from huggingface_hub import HfApi
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, train_test_split

from skops import card, hub_utils

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

# The file name is not significant, here we choose to save it with a `pkl`
# extension.
_, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
with open(pkl_name, mode="bw") as f:
    pickle.dump(model, file=f)

local_repo = mkdtemp(prefix="skops-")
hub_utils.init(
    model=pkl_name, requirements=[f"scikit-learn={sklearn.__version__}"], dst=local_repo
)

# %%
# We can no see what the contents of the created local repo are:
print(os.listdir(local_repo))

# %%
# Model Card
# ==========
# We will now create a model card and save it
model_card = card.Card(model)
model_card.save((Path(local_repo) / "README.md"))

# %%
# Push to Hub
# ===========
# And finally, we can push the model to the hub. This requires a user access
# token which you can get under https://huggingface.co/settings/tokens

# you can put your own token here, or set it as an environment variable before
# running this script.
token = os.environ["HF_HUB_TOKEN"]

repo_name = f"hf_hub_example-{uuid4()}"
user_name = HfApi().whoami(token=token)["name"]
repo_id = f"{user_name}/{repo_name}"

# Now we can push our files to the repo. The following function creates the
# remote repository if it doesn't exist; this is controlled via the
# ``create_remote`` argument.
hub_utils.push(
    repo_id=repo_id,
    source=local_repo,
    token=token,
    commit_message="pushing files to the repo from the example!",
    create_remote=True,
)

# %%
# Once uploaded, other users can download and use it, unless you make the repo
# private. Given a repository's name, here's how one can download it:
repo_copy = mkdtemp(prefix="skops")
hub_utils.download(repo_id=repo_id, dst=repo_copy)
print(os.listdir(repo_copy))


# %%
# You can also get the requirements of this repository:
print(hub_utils.get_requirements(path=repo_copy))

# %%
# As well as the complete configuration of the project:
print(json.dumps(hub_utils.get_config(path=repo_copy), indent=2))

# %%
# Now you can check the contents of the repository under your user.
#
# Update Requirements
# ===================
# If you update your environment and the versions of your requirements are
# changed, you can update the requirement in your repo by calling
# ``update_env``, which automatically detects the existing installation of the
# current environment and updates the requirements accordingly.

hub_utils.update_env(path=local_repo, requirements=["scikit-learn"])

# %%
# Delete Repository
# =================
# At the end, you can also delete the repository you created using
# ``HfApi().delete_repo``. For more information please refer to the
# documentation of ``huggingface_hub`` library.

HfApi().delete_repo(repo_id=repo_id, token=token)
