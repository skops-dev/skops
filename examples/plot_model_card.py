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

import pickle
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
# Initialize a repository to save our files in
# ============================================
# We will now initialize a repository and save our model
_, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")

with open(pkl_name, mode="bw") as f:
    pickle.dump(model, file=f)

local_repo = mkdtemp(prefix="skops-")

hub_utils.init(
    model=pkl_name,
    requirements=[f"scikit-learn={sklearn.__version__}"],
    dst=local_repo,
    task="tabular-classification",
    data=X_test,
)

# %%
# Create a model card
# ====================
# We now create a model card, and populate its metadata with information which
# is already provided in ``config.json``, which itself is created by the call
# to ``init`` above. Then, we pass information using ``add()`` and plots using
# ``add_plot()``. We'll then save the card as `README.md`.

model_card = card.Card(model, metadata=card.metadata_from_config(Path(local_repo)))


# %%
# Pass information and plots to our model card
# ============================================
# We will pass information to fill our model card.
# We will add plots to our card, note that these plots don't necessarily
# have to have a section in our template.
# We will save the plots, and then pass plot name with path to ``add_inspection``.

model_card.metadata.license = "mit"
limitations = "This model is not ready to be used in production."
model_description = (
    "This is a HistGradientBoostingClassifier model trained on breast cancer dataset."
    " It's trained with Halving Grid Search Cross Validation, with parameter grids on"
    " max_leaf_nodes and max_depth."
)
model_card_authors = "skops_user"
get_started_code = (
    "import pickle\nwith open(pkl_filename, 'rb') as file:\n    clf = pickle.load(file)"
)
citation_bibtex = "bibtex\n@inproceedings{...,year={2020}}"
model_card.add(
    citation_bibtex=citation_bibtex,
    get_started_code=get_started_code,
    model_card_authors=model_card_authors,
    limitations=limitations,
    model_description=model_description,
)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

disp.figure_.savefig(Path(local_repo) / "confusion_matrix.png")

model_card.add_plot(**{"confusion matrix": "confusion_matrix.png"})

# %%
# Save model card
# ===============
# We can simply save our model card by providing a path to ``save()``

model_card.save(Path(local_repo) / "README.md")
