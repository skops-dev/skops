"""
Tabular Regression with scikit-learn
-------------------------------------

This example shows how you can create a Hugging Face Hub compatible repo for a
tabular regression task using scikit-learn. We also show how you can generate
a model card for the model and the task at hand.
"""

# %%
# Imports
# =======
# First we will import everything required for the rest of this document.

import pickle
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import sklearn
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from skops import card, hub_utils

# %%
# Data
# ====
# We will use diabetes dataset from sklearn.

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train a Model
# =============
# To train a model, we need to convert our data first to vectors. We will use
# StandardScalar in our pipeline. We will fit a Linear Regression model with the outputs of the scalar.
model = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression()),
])

model.fit(X_train, y_train)

# %%
# Inference
# =========
# Let's see if the model works.
prediction_data = [[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]]
prediction = model.predict(prediction_data)
print(prediction)

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
    task="tabular-regression",
    data=X_test,
)

# %%
# Create a model card
# ===================
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
    "This is a Linear Regression model trained on diabetes dataset."
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
# We will now evaluate our model and add our findings to the model card.

y_pred = model.predict(X_test)
eval_descr = (
    "The model is evaluated on validation data from 20 news group's test split,"
    " using accuracy and F1-score with micro average."
)
model_card.add(eval_method=eval_descr)


# plot the predicted values against the true values
plt.scatter(y_test, y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig(Path(local_repo) / "prediction_scatter.png")
model_card.add_plot(**{"Confusion matrix": "prediction_scatter.png"})

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
model_card.add_metrics(**{"mean absolute error": mae, "mean squared error": mse, "r2 score": r2})

# %%
# Save model card
# ================
# We can simply save our model card by providing a path to :meth:`.Card.save`.
# The model hasn't been pushed to Hugging Face Hub yet, if you want to see how
# to push your models please refer to
# :ref:`this example <sphx_glr_auto_examples_plot_hf_hub.py>`.

# model_card.save(Path(local_repo) / "README.md")
model_card.save("./README.md")