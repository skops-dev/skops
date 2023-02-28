"""Check that the performance of skops persistence is not too slow

Load each (fitted) estimator and persist it with pickle and with skops. Measure
the time it takes and record it. Report the estimators that were slowest. If
skops is much slower than pickle (in absolute terms), raise an error to make the
GH action fail.

"""

from __future__ import annotations

import pickle
import timeit
import warnings
from typing import Any

import pandas as pd
from sklearn.utils._tags import _safe_tags
from sklearn.utils._testing import set_random_state

import skops.io as sio
from skops.io.tests.test_persist import (
    _get_check_estimator_ids,
    _tested_estimators,
    get_input,
)

ATOL = 1  # seconds absolute difference allowed at max
NUM_REPS = 10  # number of times the check is repeated
TOPK = 10  # number of slowest estimators reported


def check_persist_performance() -> None:
    """Run all performance checks on all estimators and print results.

    For each estimator, record how long it takes to dump+load with pickle and
    with skops. If any estimator takes much longer (in absolute time) with skops
    than pickle, raise a RuntimeError. Print the worst results to sdtout.

    """
    results: dict[str, list[Any]] = {"name": [], "pickle (s)": [], "skops (s)": []}
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
        time_pickle, time_skops = run_check(estimator, number=NUM_REPS)

        results["name"].append(name)
        results["pickle (s)"].append(time_pickle)
        results["skops (s)"].append(time_skops)

    format_result(results, topk=TOPK)


def run_check(estimator, number: int) -> tuple[float, float]:
    """Run performance check with the given estimator for pickle and skops.

    The test is run multiple times to get more robust results, ``number``
    indicates how often the it is run.

    """

    def run_pickle():
        pickle.loads(pickle.dumps(estimator))

    def run_skops():
        sio.loads(sio.dumps(estimator), trusted=True)

    time_pickle = timeit.timeit(run_pickle, number=number) / number
    time_skops = timeit.timeit(run_skops, number=number) / number

    return time_pickle, time_skops


def format_result(results: dict[str, list[Any]], topk: int) -> None:
    """Report results from performance checks.

    Print the ``topk`` slowest results. If any estimator takes much longer (in
    absolute time) with skops than pickle, raise a RuntimeError.

    """
    df = pd.DataFrame(results)
    df = df.assign(
        abs_diff=df["skops (s)"] - df["pickle (s)"],
        rel_diff=df["skops (s)"] / df["pickle (s)"],
        too_slow=lambda d: d["abs_diff"] > ATOL,
    )

    df = df.sort_values(["abs_diff"], ascending=False).reset_index(drop=True)
    print(f"{topk} largest differences:")
    print(df.head(10))

    df_slow = df.query("too_slow")
    if df_slow.empty:
        print("No estimator was found to be unacceptably slow")
        return

    print(
        f"Found {len(df_slow)} estimator(s) that are at least {ATOL:.1f} sec slower "
        "with skops:"
    )
    print(", ".join(df_slow["name"].tolist()))
    raise RuntimeError("Skops persistence too slow")


if __name__ == "__main__":
    check_persist_performance()
