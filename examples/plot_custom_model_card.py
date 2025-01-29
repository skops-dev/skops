"""
========================================
Skops model cards with a custom template
========================================

By default, when using :class:`skops.card.Card`, the skops model card template
is being used, which adds a couple of useful sections and a general document
structure to the model card. In some cases, it might, however, be desired to
use a different template because the skops template doesn't fit our needs. This
document shows how to use a custom template.

"""

# %%
# Imports
# =======

from tempfile import mkstemp

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from skops import card
from skops import io as sio

# %%
# Loading California Housing dataset
# ==================================

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# Training the RandomForestRegressor
# ==================================

est = RandomForestRegressor(n_estimators=10, random_state=0)
est.fit(X_train, y_train)

# %%
# Save the model
# --------------

_, model_file_name = mkstemp(prefix="skops-", suffix=".skops")
sio.dump(est, model_file_name)

# %%
# Create the model card
# =====================

# %%
# Initialize the empty card and add a few sections
# ------------------------------------------------
# Instead of using the default skops template, let's define our own custom
# template to use in this example. A template is a simple dictionary whose keys
# are the section names and whose values are the contents of the sections. Both
# keys and values should be strings, and values may be empty strings if you want
# to only add a section without any content (yet). As always, subsections are
# delimited using a ``"/"``.
#
# Once such a template is defined, it can be passed to the
# :class:`skops.card.Card` class using the ``template`` argument. You may also
# pass ``template=None`` to start with a completely empty model card.

link = (
    "https://scikit-learn.org/stable"
    "/auto_examples/release_highlights/plot_release_highlights_0_24_0.html"
    "#individual-conditional-expectation-plots"
)
dataset_name = "California Housing"
custom_template = {
    "Regression on California Housing dataset": (
        f"This example is adopted from [here]({link})."
    ),
    f"Regression on {dataset_name} dataset/Model Description": (
        f"Use the `{est.__class__.__name__}` from sklearn."
    ),
    f"Regression on {dataset_name} dataset/Usage": "This is only for demo purposes.",
    f"Regression on {dataset_name} dataset/Results": "",
}
model_card = card.Card(est, template=custom_template)

# %%
# Adding more content to the model card
# -------------------------------------
# From here on out, feel free to edit the model card as you would always do by
# calling the different methods starting with ``add``, such as
# :meth:`skops.card.Card.add`. and :meth:`skops.card.Card.add_metrics`. Below,
# we add more content to the model card to demonstrate this.

# %%
# Add metrics
# -----------

y_pred = est.predict(X_test)
metrics_dict = {
    "R-squared": r2_score(y_test, y_pred),
    "Mean absolute error": mean_absolute_error(y_test, y_pred),
    "Mean squared error": mean_squared_error(y_test, y_pred),
}
model_card.add_metrics(
    section="Regression on California Housing dataset/Results/Metrics",
    description=(
        "Metrics are determined on a random split consisting of 25% of the data"
    ),
    **metrics_dict,
)

# %%
# Add hyperparameter table
# ------------------------

model_card.add_hyperparams(
    section=(
        "Regression on California Housing dataset/Model Description/Hyperparameters"
    ),
    description="The model was trained with the hyperparameters listed below",
)

# %%
# Add the model diagram
# ---------------------

model_card.add_model_plot(
    section="Regression on California Housing dataset/Model Description/Model Diagram",
)

# %%
# Add partial dependence plot
# ---------------------------

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
n_samples = 500  # make plot a bit faster
display = PartialDependenceDisplay.from_estimator(
    est,
    X[:n_samples],
    features,
    kind="individual",
    subsample=50,
    grid_resolution=20,
    random_state=0,
)
display.figure_.suptitle(
    "Partial dependence of house value on non-location features\n"
    "for the California housing dataset, with BayesianRidge"
)
display.figure_.subplots_adjust(hspace=0.3)
_, plot_file_name = mkstemp(prefix="skops-", suffix=".png")
display.figure_.savefig(plot_file_name)
model_card.add_plot(
    **{
        "Regression on California Housing dataset/Results/Partial Dependence Plots": (
            plot_file_name
        )
    },
)

# %%
# Save the finished model card
# ----------------------------

_, card_file_name = mkstemp(prefix="skops-", suffix=".md")
model_card.save(card_file_name)

# %%
# Further information
# ===================
#
# The model card created in this example is very bare bones, missing a lot of
# important information. It should only be used as a starting point and not be
# considered a complete example. If you would like to know more about model
# cards, `the model card documentation
# <https://huggingface.co/docs/hub/model-cards>`_ on the Hugging Face Hub could
# be a good starting point.
#
# Furthermore, this model card lacks metadata, which can be very useful if you
# plan to upload the model on Hugging Face Hub. If you want to add metadata,
# instantiate it using :class:`huggingface_hub.ModelCardData` and pass it to the
# :class:`skops.card.Card` class.
