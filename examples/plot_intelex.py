"""
Creating models that are accelerated by Intel(R) Extension for scikit-learn
---------------------------------------------------------------------------

Introduction
============

This guide demonstrates how under certain conditions, Intel(R) Extension for
Scikit-learn (also ``scikit-learn-intelex``, or ``sklearnex``) can be used to
speed up inference of Scikit-learn models.

The extension supports most of Scikit-learn's classical machine learning
algorithms, like k-nearest neighbors, support vector machines, linear/logistic
regression, and more. Stock Scikit-learn implementations are used where no
optimized version is available, making this package 100% compatible with
existing code. Note while compatibility is assured by continuous testing,
equivalence of results between the two packages is not guaranteed. In fact, due
to independent implementations, intermediate results differ in many cases. An
up-to-date list of supported algorithms can be found in the `official
documentation <https://intel.github.io/scikit-learn-intelex/algorithms.html>`_.

Intel(R) Extension for Scikit-learn accelerates Scikit-learn algorithms by using
the latest hardware features and optimized caching and threading strategies.
Find more details in Intel's blog posts on Medium (`1
<https://medium.com/intel-analytics-software/save-time-and-money-with-intel-extension-for-scikit-learn-33627425ae4>`_,
`2
<https://medium.com/intel-analytics-software/accelerate-your-scikit-learn-applications-a06cacf44912>`_).
In many cases, optimizations translate to hardware from other vendors, albeit
with smaller performance gains.

For this example, we train two simple
:class:`sklearn.neighbors.KNeighborsClassifier` instances, one with and one
without using ``sklearnex``, and compare inference times. Afterward, we upload
both models to the Hugging Face Model Hub. Hugging Face Hub supports
``sklearnex``-optimized models, meaning the achieved speedup will translate for
Inference API users.
"""

# %%
# Imports
# =======
# First, we import everything required for the rest of this document.

import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from time import perf_counter
from uuid import uuid4

from huggingface_hub import delete_repo, whoami
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearnex import patch_sklearn
from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifierOptimized

from skops import card, hub_utils

# %%
# Data
# ====
# Next, we create some generic data. A dataset of 50k rows x 15 columns is big enough
# to showcase a performance gain from using ``sklearnex``. Generally speaking,
# larger datasets will benefit more from the ``sklearnex`` optimizations. More
# details can be found in the official
# `README <https://github.com/intel/scikit-learn-intelex/blob/master/README.md>`_.
X, y = make_classification(
    n_samples=50_000,
    n_features=15,
    n_informative=15,
    n_redundant=0,
    n_clusters_per_class=1,
    shuffle=False,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
# Training the stock model
# ========================
# Now we can train a stock Scikit-learn
# :class:`sklearn.neighbors.KNeighborsClassifier`

clf = KNeighborsClassifier(3, n_jobs=-1)
start = perf_counter()
clf.fit(X_train, y_train)
print(f"Training finished in {perf_counter() - start:.2f}s")

# %%
# Training the optimized model
# ============================
# Now we fit the optimized algorithm. Note, that rather than loading the model
# from ``sklearnex``, we could also load and call ``patch_sklearn()``. Find more
# details in the `documentation
# <https://intel.github.io/scikit-learn-intelex/#usage>`_.

clf_opt = KNeighborsClassifierOptimized(3, n_jobs=-1)
start = perf_counter()
clf_opt.fit(X_train, y_train)
print(f"Training finished in {perf_counter() - start:.2f}s")

# %%
# We are not comparing the k-NN fit times, since this is not a compute-intensive
# task and both are typically very fast.

# %%
# Comparing inference times
# =========================
# Now to the interesting part: We measure the execution time of
# ``predict_proba()`` for the two models.

start = perf_counter()
y_proba = clf.predict_proba(X_test)
t_stock = perf_counter() - start

log_loss_score = log_loss(y_test, y_proba)
print(
    f"[stock scikit-learn] Inference took t_stock = {t_stock:.2f}s with a "
    f"log-loss score of {log_loss_score:.3f}"
)

start = perf_counter()
y_proba = clf_opt.predict_proba(X_test)
t_opt = perf_counter() - start

log_loss_score = log_loss(y_test, y_proba)
print(
    f"[sklearnex] Inference took t_opt = {t_opt:.2f}s with a log-loss score of"
    f" {log_loss_score:.3f}"
)

print(f"t_stock / t_opt = {t_stock/t_opt:.1f}")

# %%
# We see that inference using ``sklearnex`` is a lot faster while achieving the
# same log-loss score.

# %%
# Save and upload the models
# ==========================
# Let's save all required files to disk and initialize Hugging Face Model Hub
# repositories.

# replace with your own token or set it as an environment variable before
# running the script
token = os.environ["HF_HUB_TOKEN"]

with NamedTemporaryFile(mode="bw", prefix="stock-", suffix=".pkl") as fp:
    pickle.dump(clf, file=fp)

    stock_repo = mkdtemp(prefix="stock-")
    hub_utils.init(
        model=fp.name,
        requirements=["scikit-learn=1.2.1"],
        dst=stock_repo,
        task="tabular-classification",
        data=X_test,
    )


with NamedTemporaryFile(mode="bw", prefix="opt-", suffix=".pkl") as fp:
    pickle.dump(clf_opt, file=fp)

    opt_repo = mkdtemp(prefix="opt-")
    hub_utils.init(
        model=fp.name,
        requirements=["scikit-learn=1.2.1", "scikit-learn-intelex=2023.0.1"],
        dst=opt_repo,
        task="tabular-classification",
        data=X_test,
        use_intelex=True,
    )

# Create Model cards with the most basic information
clf_card = card.Card(clf, metadata=card.metadata_from_config(Path(stock_repo)))
clf_card.metadata.license = "mit"
limitations = "This model is not ready to be used in production."
model_description = (
    "This is a `KNeighborsClassifier` model trained on synthetic data. It is "
    "trained with the stock scikit-learn algorithm and part of a "
    "demonstration, showing how Intel(R) Extension for scikit-learn can be "
    "used to speed up model inference times."
)
model_card_authors = "skops_user"
citation_bibtex = "**BibTeX**\n\n```\n@inproceedings{...,year={2020}}\n```"
clf_card.add(
    **{
        "Citation": citation_bibtex,
        "Model Card Authors": model_card_authors,
        "Model description": model_description,
        "Model description/Intended uses & limitations": limitations,
    }
)
clf_card.save(Path(stock_repo) / "README.md")

clf_opt_card = card.Card(clf_opt, metadata=card.metadata_from_config(Path(opt_repo)))
model_description = (
    "This is a `KNeighborsClassifier` model trained on synthetic data. It is "
    "trained with the Intel(R) extension for scikit-learn optimized version of "
    "the algorithm, and part of a demonstration, showing how Intel(R) "
    "Extension for scikit-learn can be used to speed up model inference times."
)
clf_card.add(
    **{
        "Citation": citation_bibtex,
        "Model Card Authors": model_card_authors,
        "Model description": model_description,
        "Model description/Intended uses & limitations": limitations,
    }
)
clf_opt_card.save(Path(opt_repo) / "README.md")

# Push everything to the Model hub
user_name = whoami(token=token)["name"]
uuid = uuid4()
repo_id_stock = f"{user_name}/knn-example-stock-{uuid}"
repo_id_opt = f"{user_name}/knn-example-intelex-{uuid}"

print(f"Pushing skl model to: {repo_id_stock}")
hub_utils.push(
    repo_id=repo_id_stock,
    source=stock_repo,
    token=token,
    commit_message="Add scikit-learn KNN model example",
    create_remote=True,
    private=False,
)
print(f"Pushing sklearnex model to: {repo_id_opt}")
hub_utils.push(
    repo_id=repo_id_opt,
    source=opt_repo,
    token=token,
    commit_message="Add scikit-learn-intelex KNN model example",
    create_remote=True,
    private=False,
)


# %%
# Loading non-optimized models
# ============================
# It is possible to load non-optimized models even after Intel(R) optimizations
# were loaded with ``patch_sklearn()``. Note, however, that this will not result
# in faster inference times, since loading a persisted model will always load
# the objects exactly as they were saved.

patch_sklearn()

with NamedTemporaryFile(mode="bw+") as fp:
    pickle.dump(clf, file=fp)
    fp.seek(0)
    clf = pickle.load(fp)

start = perf_counter()
clf.predict_proba(X_test)
t_stock = perf_counter() - start

log_loss_score = log_loss(y_test, y_proba)
print(
    f"[stock scikit-learn] Inference took t_stock = {t_stock:.2f}s with a "
    f"log-loss score of {log_loss_score:.3f}"
)


# %%
# Delete Repository
# =================
# At the end, we can delete the created repositories again using
# ``delete_repo``. For more information please refer to the
# documentation of ``huggingface_hub`` library.

delete_repo(repo_id=repo_id_stock, token=token)
delete_repo(repo_id=repo_id_opt, token=token)
