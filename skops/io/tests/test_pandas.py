"""Tests for persisting pandas objects."""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from skops.io import dump, dumps, get_untrusted_types, load, loads, visualize
from skops.io._pandas import _public_module
from skops.io._trusted_types import PANDAS_TYPE_NAMES
from skops.io._utils import gettype
from skops.io.exceptions import UnsupportedTypeException
from skops.io.tests._utils import _assert_vals_equal

pd = pytest.importorskip("pandas")


INDEXES = [
    pd.Index([1, 2, 3], name="ints"),
    pd.Index([1.5, np.nan, 3.0]),
    pd.Index([True, False]),
    pd.Index(["a", None, "c"], name="strings"),
    pd.Index([1, "a", None], dtype=object),
    pd.Index([(1, 2), (3, 4)], dtype=object, tupleize_cols=False),
    pd.Index([], dtype=object),
    pd.Index([1, None, 3], dtype="Int64"),
    pd.RangeIndex(5),
    pd.RangeIndex(2, 20, 3, name="range"),
    pd.date_range("2024-01-01", periods=3, name="dates"),
    pd.date_range("2024-01-01", periods=3, tz="Europe/Berlin"),
    pd.date_range("2024-01-01", periods=2, tz="UTC"),
    # a fixed offset parsed from the strings
    pd.to_datetime(["2024-01-01T00:00:00+01:00", "2024-01-02T00:00:00+01:00"]),
    pd.DatetimeIndex(["2024-01-01", None]),
    pd.timedelta_range("1D", periods=2),
    pd.period_range("2024-01", periods=2, freq="M"),
    pd.interval_range(0, 3),
    pd.CategoricalIndex(["a", "b", "a"], categories=["b", "a"], ordered=True),
    pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["letters", None]),
    pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], sortorder=0),
]

SERIES = [
    pd.Series([1, 2, 3]),
    pd.Series([1.5, np.nan], index=["a", "b"], name="floats"),
    pd.Series(["x", "y", None]),
    pd.Series(["x", 1, None], dtype=object),
    pd.Series(np.array(["x", "y"], dtype=object), dtype=object),
    pd.Series([1, None, 3], dtype="Int64"),
    pd.Series([True, None], dtype="boolean"),
    pd.Series([1.5, None], dtype="Float64"),
    pd.Series(["a", "b", "a"], dtype="category"),
    pd.Series(pd.Categorical(["a", "b"], categories=["b", "a", "c"], ordered=True)),
    pd.Series(pd.to_datetime(["2024-01-01", None])),
    pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC")),
    pd.Series(pd.to_timedelta([1, 2], unit="D")),
    pd.Series(pd.period_range("2024-01", periods=2, freq="M")),
    pd.Series(pd.interval_range(0, 2)),
    pd.Series(pd.arrays.SparseArray([0, 0, 1.5])),
    pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])),
    pd.Series([1, 2, 3], index=[1, 1, 2]),
    pd.Series([1], name=("a", "b")),
    pd.Series([], dtype=float),
    # the shape of category_encoders' TargetEncoder.mapping values
    pd.Series([0.49, 0.66, 0.6], index=pd.Index([1, 2, -1]), name="category"),
]

FRAMES = [
    pd.DataFrame(
        {"i": [1, 2], "f": [1.5, np.nan], "s": ["a", None], "b": [True, False]}
    ),
    pd.DataFrame(
        {
            "o": pd.Series(["a", "b"], dtype=object),
            "n": pd.array([1, None], dtype="Int64"),
            "c": pd.Categorical(["x", "y"]),
            "t": pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin"),
        }
    ),
    pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]),
    pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "x"], name="dups")),
    pd.DataFrame(index=pd.RangeIndex(3)),
    pd.DataFrame(columns=["a", "b"]),
    pd.DataFrame(),
    pd.DataFrame(
        np.arange(6).reshape(2, 3),
        columns=pd.MultiIndex.from_tuples([("x", 1), ("x", 2), ("y", 1)]),
    ),
    pd.DataFrame(
        {"a": [1]}, index=pd.MultiIndex.from_tuples([("k", 0)], names=["l", "n"])
    ),
]

ARRAYS = [
    pd.array([1, None], dtype="Int64"),
    pd.array([1.5, None], dtype="Float64"),
    pd.array([True, None], dtype="boolean"),
    pd.array(["a", None], dtype="string"),
    pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True),
    pd.array(pd.to_datetime(["2024-01-01", None])),
    pd.array(
        pd.to_datetime(["2024-01-01"]).tz_localize(dt.timezone(dt.timedelta(hours=1)))
    ),
    pd.array(pd.to_timedelta([1], unit="s")),
    pd.array(pd.period_range("2024-01", periods=1, freq="M")),
    pd.array(pd.interval_range(0, 2)),
    pd.arrays.SparseArray([0, 1]),
    pd.Series([1, 2]).array,
]

DTYPES = [
    pd.Int64Dtype(),
    pd.BooleanDtype(),
    pd.StringDtype(),
    pd.CategoricalDtype(["b", "a"], ordered=True),
    pd.CategoricalDtype(),
    pd.DatetimeTZDtype("ns", "UTC"),
    pd.PeriodDtype("M"),
    pd.IntervalDtype("int64", closed="left"),
    pd.SparseDtype(float, 0.0),
]


def _id(obj):
    return f"{type(obj).__name__}-{getattr(obj, 'dtype', '')}"


@pytest.mark.parametrize("obj", INDEXES + SERIES + FRAMES + ARRAYS + DTYPES, ids=_id)
def test_roundtrip(obj):
    # pandas types are trusted by default, so no trusted list is needed
    loaded = loads(dumps(obj))
    # the strict pandas comparison shared with the estimator tests
    _assert_vals_equal(obj, loaded)


def test_pandas_types_are_trusted_by_default():
    assert get_untrusted_types(data=dumps(FRAMES[1])) == []


def _with_edited_schema(dumped, edit):
    # ``dumped`` with ``edit`` applied to its schema, to mimic a crafted file
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        files = {
            name: zip_file.read(name)
            for name in zip_file.namelist()
            if name != "schema.json"
        }
    edit(schema)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        zip_file.writestr("schema.json", json.dumps(schema))
        for name, data in files.items():
            zip_file.writestr(name, data)
    return buffer.getvalue()


def test_dtype_node_only_builds_the_declared_class():
    # the name in the file is parsed by the declared, trusted dtype class and
    # not looked up in pandas' registry of extension dtypes, where it could
    # name the dtype of another library, whose code would then run
    dumped = dumps(pd.Int64Dtype())

    def edit(schema):
        schema["content"]["name"]["content"] = json.dumps("period[M]")

    with pytest.raises(TypeError, match="Cannot construct"):
        loads(_with_edited_schema(dumped, edit))


def test_child_of_wrong_kind_is_refused():
    # the values of an Index are an array; a file holding something else there
    # is refused while it is read, before anything is constructed
    dumped = dumps(pd.Index([1, 2]))

    def edit(schema):
        schema["content"]["values"] = {
            "__class__": "str",
            "__module__": "builtins",
            "__loader__": "JsonNode",
            "content": json.dumps("x"),
            "is_json": True,
            "__id__": 1,
        }

    with pytest.raises(ValueError, match="Expected a node of type"):
        loads(_with_edited_schema(dumped, edit))


def test_missing_entry_is_refused():
    dumped = dumps(pd.Index([1, 2]))

    def edit(schema):
        del schema["content"]["name"]

    with pytest.raises(ValueError, match="Expected the entries"):
        loads(_with_edited_schema(dumped, edit))


def test_subclass_from_other_library_is_unsupported():
    class MySeries(pd.Series):
        pass

    with pytest.raises(UnsupportedTypeException, match="subclass of a pandas type"):
        dumps(MySeries([1, 2]))


def test_visualize(capsys):
    visualize(dumps(FRAMES[0]))
    assert "pandas.DataFrame" in capsys.readouterr().out


def test_file_uses_public_type_names():
    # older pandas versions report the defining module of a class, e.g.
    # pandas.core.series.Series, and pandas 3 reports pandas.Series; the file
    # always holds the public name, which is the one trusted by default
    dumped = dumps(pd.Series([1], index=pd.Index([1])))
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    assert (schema["__module__"], schema["__class__"]) == ("pandas", "Series")
    index = schema["content"]["index"]
    assert (index["__module__"], index["__class__"]) == ("pandas", "Index")


def test_trusted_type_names_are_valid():
    # every name is the public path of a pandas class, as written by dumps, so
    # that a rename in pandas does not silently leave a type untrusted
    for name in PANDAS_TYPE_NAMES:
        module, _, class_name = name.rpartition(".")
        try:
            cls = gettype(module, class_name)
        except AttributeError:
            # types that only exist in some pandas versions, e.g. PandasArray
            continue
        assert f"{_public_module(cls)}.{cls.__name__}" == name


FIXTURE_DIR = Path(__file__).parent / "data"


def cross_version_objects():
    """Objects whose files, written by one pandas version, load on every other.

    :func:`write_pandas_fixture_file` dumps them with the running pandas version
    into ``data/``, and :func:`test_load_file_of_other_pandas_version` loads
    every file there. Add a file for a new pandas minor version when it changes
    how any of these is represented, and never regenerate an existing one.
    """
    return {
        "str_index": pd.Index(["a", None, "c"], name="strings"),
        "str_series": pd.Series(["x", "y", None], index=["a", "b", "c"]),
        "mixed_frame": pd.DataFrame(
            {
                "i": [1, 2],
                "f": [1.5, np.nan],
                "s": ["a", None],
                "o": pd.Series(["a", 1], dtype=object),
                "c": pd.Categorical(["x", "y"], categories=["y", "x"], ordered=True),
                "t": pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin"),
            }
        ),
        "datetime_index": pd.date_range("2024-01-01", periods=3, name="dates"),
        "datetime_index_tz": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "datetime_fixed_offset": pd.to_datetime(["2024-01-01T00:00:00+01:00"]),
        "timedelta_index": pd.timedelta_range("1D", periods=2),
        "period_series": pd.Series(pd.period_range("2024-01", periods=2, freq="M")),
        "interval_index": pd.interval_range(0, 3),
        "categorical_index": pd.CategoricalIndex(
            ["a", "b", "a"], categories=["b", "a"], ordered=True
        ),
        "multi_index": pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2)], names=["letters", None]
        ),
        "range_index": pd.RangeIndex(2, 20, 3, name="range"),
        "nullable_frame": pd.DataFrame(
            {
                "i": pd.array([1, None], dtype="Int64"),
                "b": pd.array([True, None], dtype="boolean"),
                "f": pd.array([1.5, None], dtype="Float64"),
            }
        ),
        "sparse_series": pd.Series(pd.arrays.SparseArray([0, 0, 1.5])),
        "duplicate_columns": pd.DataFrame([[1, 2]], columns=["a", "a"]),
        # the shape of category_encoders' TargetEncoder.mapping
        "encoder_mapping": {"col": pd.Series([0.49, 0.66], index=pd.Index([1, 2]))},
        "dtypes": [
            pd.Int64Dtype(),
            pd.CategoricalDtype(["b", "a"], ordered=True),
            pd.StringDtype(),
        ],
    }


def write_pandas_fixture_file():
    """Dump the cross-version objects with the running pandas version."""
    path = FIXTURE_DIR / f"pandas-{pd.__version__}.skops"
    dump(cross_version_objects(), path)
    return path


def _as_objects(values):
    # the values as Python objects, with ``None`` for every missing value
    return values.to_numpy(dtype=object, na_value=None)


def assert_same_data(expected, actual):
    """Equality up to the dtype differences between pandas versions.

    Strings are ``str`` in pandas 3 and ``object`` before, datetimes have a
    microsecond resolution in pandas 3 and nanoseconds before, and a file keeps
    the dtypes of the version that wrote it. The values are therefore compared
    as Python objects, with a single marker for missing values.
    """
    assert type(actual) is type(expected)
    if isinstance(expected, pd.DataFrame):
        assert expected.shape == actual.shape
        assert_same_data(expected.columns, actual.columns)
        assert_same_data(expected.index, actual.index)
        for i in range(expected.shape[1]):
            np.testing.assert_array_equal(
                _as_objects(expected.iloc[:, i]), _as_objects(actual.iloc[:, i])
            )
    elif isinstance(expected, pd.Series):
        assert expected.name == actual.name
        assert_same_data(expected.index, actual.index)
        np.testing.assert_array_equal(_as_objects(expected), _as_objects(actual))
    elif isinstance(expected, pd.MultiIndex):
        assert list(expected.names) == list(actual.names)
        assert_same_data(expected.to_frame(index=False), actual.to_frame(index=False))
    elif isinstance(expected, pd.Index):
        assert expected.name == actual.name
        np.testing.assert_array_equal(_as_objects(expected), _as_objects(actual))
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_same_data(expected[key], actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            assert_same_data(expected_item, actual_item)
    else:
        assert expected == actual


@pytest.mark.parametrize(
    "path", sorted(FIXTURE_DIR.glob("pandas-*.skops")), ids=lambda path: path.stem
)
def test_load_file_of_other_pandas_version(path):
    # pandas types are trusted by default, so no trusted list is needed
    loaded = load(path)
    assert_same_data(cross_version_objects(), loaded)
