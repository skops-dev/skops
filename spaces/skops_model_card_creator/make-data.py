# companion script to the space creator
# generates the logreg.pkl and logreg.skops file, as well as data.csv

import pickle

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import skops.io as sio

X, y = make_classification()
df = pd.DataFrame(X)

clf = Pipeline(
    [
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(random_state=0)),
    ]
)
clf.fit(X, y)

with open("logreg.pkl", "wb") as f:
    pickle.dump(clf, f)
sio.dump(clf, "logreg.skops")


df.to_csv("data.csv", index=False)
