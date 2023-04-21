"""Check that the file size of skops files is not too large.

Load each (fitted) estimator and persist it with pickle and with skops. Measure
the file size of the resulting files. Report the results but in contrast to the
runtime check, don't raise any errors if the file size differences is too big.

For skops, zip compression is applied. This is because we can assume that if a
user really cares about file size, they will compress the file.

"""

from __future__ import annotations

import io
import os
import pickle
import warnings
from tempfile import mkstemp
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from sklearn.utils._tags import _safe_tags
from sklearn.utils._testing import set_random_state

import skops.io as sio
from skops.io.tests.test_persist import (
    _get_check_estimator_ids,
    _tested_estimators,
    get_input,
)

TOPK = 10  # number of largest estimators reported


def check_file_size() -> None:
    """Run all file size checks on all estimators and report the results.

    Print the results twice, once sorted by absolute differences, once sorted by
    relative differences.

    """
    results: dict[str, list[Any]] = {"name": [], "pickle (kb)": [], "skops (kb)": []}
    for estimator in _tested_estimators():
        set_random_state(estimator, random_state=0)

        X, y = get_input(estimator)
        tags = _safe_tags(estimator)
        if tags.get("requires_fit", True):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="sklearn")
                if y is not None:
                    estimator.fit(X, y)
                else:
                    estimator.fit(X)

        name = _get_check_estimator_ids(estimator)
        cls_name, _, _ = name.partition("(")
        size_pickle, size_skops = run_check(estimator)

        results["name"].append(cls_name)
        results["pickle (kb)"].append(size_pickle)
        results["skops (kb)"].append(size_skops)

    format_result(results, topk=TOPK)


def run_check(estimator) -> tuple[float, float]:
    """Run file size check with the given estimator for pickle and skops."""
    _, name = mkstemp(prefix="skops")

    def run_pickle():
        fname = name + ".pickle"
        buffer = io.BytesIO()
        pickle.dump(estimator, buffer)
        with ZipFile(
            fname + ".zip", mode="w", compression=ZIP_DEFLATED, compresslevel=9
        ) as zipf:
            zipf.writestr(fname, buffer.getvalue())

        # return size in kb
        return os.stat(fname + ".zip").st_size / 1024

    def run_skops():
        fname = name + ".skops"
        sio.dump(estimator, fname, compression=ZIP_DEFLATED, compresslevel=9)
        # return size in kb
        return os.stat(fname).st_size / 1024

    size_pickle = run_pickle()
    size_skops = run_skops()
    return size_pickle, size_skops


def format_result(results: dict[str, list[Any]], topk: int) -> None:
    """Report results from performance checks.

    Print the largest file size differences between pickle and skops, once for
    absolute, once for relative differences.

    """
    df = pd.DataFrame(results)
    df = df.assign(
        abs_diff=df["skops (kb)"] - df["pickle (kb)"],
        rel_diff=df["skops (kb)"] / df["pickle (kb)"],
    )

    dfs = df.sort_values(["abs_diff"], ascending=False).reset_index(drop=True)
    print(f"{topk} largest absolute differences:")
    print(dfs[["name", "pickle (kb)", "skops (kb)", "abs_diff"]].head(10))

    print(f"{topk} largest relative differences:")
    dfs = df.sort_values(["rel_diff"], ascending=False).reset_index(drop=True)
    print(dfs[["name", "pickle (kb)", "skops (kb)", "rel_diff"]].head(10))


if __name__ == "__main__":
    check_file_size()
