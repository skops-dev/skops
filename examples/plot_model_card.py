"""
scikit-learn model cards
--------------------------------------

This guide demonstrates how you can use this package to create a model card on a
scikit-learn compatible model and save it.
"""

# %%
# Imports
# =======
# First we will import everything required for the rest of this document.


import os
import pickle
from tempfile import mkdtemp, mkstemp

import matplotlib.pyplot as plt
import sklearn
from modelcards import CardData
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import HalvingGridSearchCV, train_test_split

from skops import card, hub_utils

# %%
# Data
# ====
# We load breast cancer dataset from sklearn.

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
# Create a model card
# ====================
# We now create a model card, set couple of attributes and save it.
# We first set the metadata with CardData and we'll later pass it to create_model_card.

limitations = "This model is not ready to be used in production."
model_description = (
    "This is a HistGradientBoostingClassifier model trained on breast cancer dataset."
    " It's trained with Halving Grid Search Cross Validation, with parameter grids on"
    " max_leaf_nodes and max_depth."
)
license = "mit"

eval_results = card.evaluate(
    model, X_test, y_test, "r2", "random_type", "dummy_dataset", "tabular-regression"
)

card_data = CardData(
    license=license,
    tags=["tabular-classification"],
    datasets="breast-cancer",
    eval_results=eval_results,
    model_name="my-cool-model",
    metrics=["acc"],
)

# %% Adding metrics
# ====================
# We'll pass permutation importances, confusion matrix and classification report
# to our model card template. Skops includes a util to calculate and parse
# permutation importances, we'll use that. For confusion matrix and
# classification report, we'll use tools from scikit-learn. Additionally, model
# card template has an extra section for images, so we will use
# ConfusionMatrixDisplay and put the created plot in that section.


predictions = model.predict(X_test)
permutation_importances = card.permutation_importances(model, X_test, y_test)
confusion_matrix_arr = confusion_matrix(y_test, predictions, labels=model.classes_)
clf_report = classification_report(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_arr, display_labels=model.classes_
)
plt.savefig("./confusion_matrix.png")


# %% Additional sections
# ======================
# We can introduce introductions on how to use the model to our model card. This
# section will be formatted as a code. We will also put citation info and name
# of the author of the model card.

model_card_authors = "skops_user"
get_started_code = (
    "import pickle\nwith open(dtc_pkl_filename, 'rb') as file:\nclf = pickle.load(file)"
)
citation = "bibtex\n@inproceedings{...,year={2020}}"


# %% Create and save the card!
# ============================
# We will now create the model card using model, card_data and rest of the
# information. We'll initialize a repository and save the card along with the
# model.

model_card = card.create_model_card(
    model,
    card_data=card_data,
    limitations=limitations,
    model_description=model_description,
    citation_bibtex=citation,
    model_card_authors=model_card_authors,
    get_started_code=get_started_code,
    permutation_importances=permutation_importances,
    classification_report=clf_report,
    confusion_matrix=confusion_matrix_arr,
    metric_plot="./confusion_matrix.png",
)

_, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")

with open(pkl_name, mode="bw") as f:
    pickle.dump(model, file=f)

local_repo = mkdtemp(prefix="skops-")
hub_utils.init(
    model=pkl_name, requirements=[f"scikit-learn={sklearn.__version__}"], dst=local_repo
)

model_card.save(os.path.join(f"{local_repo}", "README.md"))
