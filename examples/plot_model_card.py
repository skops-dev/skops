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

import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
# is already provided in ``config.json``, which itself is created by the call to
# :func:`.hub_utils.init` above. We will see below how we can populate the model
# card with useful information.

model_card = card.Card(model, metadata=card.metadata_from_config(Path(local_repo)))

# %%
# Add more information
# ====================
# So far, the model card does not tell viewers a lot about the model. Therefore,
# we add more information about the model, like a description and what its
# license is.

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

# %%
# Add plots, metrics, and tables to our model card
# ================================================
# Furthermore, to better understand the model performance, we should evaluate it
# on certain metrics and add those evaluations to the model card. In this
# particular example, we want to calculate the accuracy and the F1 score. We
# calculate those using sklearn and then add them to the model card by calling
# :meth:`.Card.add_metrics`. But this is not all, we can also add matplotlib
# figures to the model card, e.g. a plot of the confusion matrix. To achieve
# this, we create the plot using sklearn, save it locally, and then add it using
# :meth:`.Card.add_plot` method. Finally, we can also add some useful tables to
# the model card, e.g. the results from the grid search and the classification
# report. Those can be added using :meth:`.Card.add_table`

y_pred = model.predict(X_test)
eval_descr = (
    "The model is evaluated on test data using accuracy and F1-score with macro"
    " average."
)
model_card.add(eval_method=eval_descr)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="micro")
model_card.add_metrics(**{"accuracy": accuracy, "f1 score": f1})

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

disp.figure_.savefig(Path(local_repo) / "confusion_matrix.png")
model_card.add_plot(**{"Confusion matrix": "confusion_matrix.png"})

cv_results = model.cv_results_
clf_report = classification_report(
    y_test, y_pred, output_dict=True, target_names=["malignant", "benign"]
)
# The classification report has to be transformed into a DataFrame first to have
# the correct format. This requires removing the "accuracy", which was added
# above anyway.
del clf_report["accuracy"]
clf_report = pd.DataFrame(clf_report).T.reset_index()
model_card.add_table(
    details_tag=True,
    **{
        "Hyperparameter search results": cv_results,
        "Classification report": clf_report,
    },
)

# %%
# Save model card
# ===============
# We can simply save our model card by providing a path to :meth:`.Card.save`.

model_card.save(Path(local_repo) / "README.md")
