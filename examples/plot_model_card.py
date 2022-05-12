"""
scikit-learn model cards
--------------------------------------

This guide demonstrates how you can create a model card for scikit-learn models
for evaluating the models and reproducibility of the experiment.
"""

# %%
# Imports
# =======
# First we will import everything required for the rest of this document.

from tempfile import mkdtemp

from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skops import model_card

# %%
# Data
# ====
# We will import a dataset.
X, y = load_diabetes(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("X's summary: ", X.head())
print("y's summary: ", y.head())

# %%
# Train a Model
# =======
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

# %%
# Create a model card
# =======
# We will create and save a model card.
local_repo = mkdtemp(prefix="skops")
model_card.create_model_card(f"{local_repo}/README.md", clf)
