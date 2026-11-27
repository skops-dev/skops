"""Tests for persisting pandas objects."""

from __future__ import annotations

import datetime as dt
import io
import json
import struct
import sys
import warnings
import zoneinfo
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from packaging.version import Version

from skops.io import (
    PANDAS_TYPES,
    dump,
    dumps,
    get_untrusted_types,
    load,
    loads,
    visualize,
)
from skops.io._pandas import _RENAMED_CLASSES
from skops.io._protocol import PROTOCOL
from skops.io._utils import get_type_name
from skops.io.exceptions import (
    UnsupportedTypeException,
    UntrustedTypesFoundException,
)
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
    pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])),
    pd.Series([1, 2, 3], index=[1, 1, 2]),
    pd.Series([1], name=("a", "b")),
    pd.Series([], dtype=float),
    # the shape of category_encoders' TargetEncoder.mapping values
    pd.Series([0.49, 0.66, 0.6], index=pd.Index([1, 2, -1]), name="category"),
    # pandas scalars in an object column
    pd.Series(
        [
            pd.Timestamp("2024-01-01 00:00:00.000000001", tz="Europe/Berlin"),
            pd.Timedelta(1, unit="ns"),
            pd.NA,
            pd.NaT,
        ],
        dtype=object,
    ),
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

# the plain wrapper of a numpy array, under its old name on pandas 2.0
_NumpyBackedArray = (
    getattr(pd.arrays, "NumpyExtensionArray", None) or pd.arrays.PandasArray
)

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
    # fixed offsets with seconds, as in the local mean time of old data
    pd.array(
        pd.to_datetime(["1850-01-01"]).tz_localize(
            dt.timezone(dt.timedelta(minutes=19, seconds=32))
        )
    ),
    pd.array(
        pd.to_datetime(["2024-01-01"]).tz_localize(
            dt.timezone(-dt.timedelta(hours=3, minutes=30, seconds=45))
        )
    ),
    pd.array(pd.to_timedelta([1], unit="s")),
]

DTYPES = [
    pd.Int64Dtype(),
    pd.BooleanDtype(),
    pd.StringDtype(),
    pd.CategoricalDtype(["b", "a"], ordered=True),
    pd.CategoricalDtype(),
    pd.DatetimeTZDtype("ns", "UTC"),
    pd.DatetimeTZDtype("us", "Europe/Berlin"),
    pd.DatetimeTZDtype("ns", dt.timezone(dt.timedelta(seconds=30))),
]


SCALARS = [
    pd.Timestamp("2024-01-01 00:00:00.000000001"),
    pd.Timestamp("2024-01-01", tz="Europe/Berlin"),
    pd.Timestamp("2024-01-01", tz=dt.timezone(dt.timedelta(hours=1))),
    pd.Timestamp("2024-01-01").as_unit("s"),
    pd.Timedelta(1, unit="ns"),
    pd.Timedelta(days=1).as_unit("ms"),
    pd.NA,
    pd.NaT,
]


def _id(obj):
    return f"{type(obj).__name__}-{getattr(obj, 'dtype', '')}"


def _loads(dumped):
    # pandas types are not trusted by default: a user passes PANDAS_TYPES,
    # which hold every pandas name a file can hold
    untrusted = get_untrusted_types(data=dumped)
    assert untrusted and set(untrusted) <= set(PANDAS_TYPES)
    return loads(dumped, trusted=PANDAS_TYPES)


@pytest.mark.parametrize("obj", INDEXES + SERIES + FRAMES + ARRAYS + DTYPES, ids=_id)
def test_roundtrip(obj):
    loaded = _loads(dumps(obj))
    # the strict pandas comparison shared with the estimator tests
    _assert_vals_equal(obj, loaded)


@pytest.mark.parametrize("obj", SCALARS, ids=repr)
def test_scalar_roundtrip(obj):
    loaded = _loads(dumps(obj))
    assert type(loaded) is type(obj)
    # the repr shows the unit, the nanoseconds and the zone
    assert repr(loaded) == repr(obj)


@pytest.mark.parametrize("singleton", [pd.NA, pd.NaT], ids=repr)
def test_missing_value_singletons_keep_their_identity(singleton):
    # pandas recognises these by identity: a copy of NaT is not missing to
    # pd.isna, and acts as the year 1 in arithmetic
    loaded = _loads(dumps({"fill_value_": singleton, "values_": [singleton]}))
    assert loaded["fill_value_"] is singleton
    assert loaded["values_"][0] is singleton
    assert pd.isna(loaded["fill_value_"])
    series = _loads(dumps(pd.Series([singleton, 1], dtype=object)))
    assert series.isna().tolist() == [True, False]


@pytest.mark.parametrize(
    "freq",
    [
        pd.offsets.Day(2),
        pd.offsets.Hour(),
        pd.offsets.MonthEnd(),
        pd.offsets.Week(weekday=6),
        pd.offsets.QuarterEnd(startingMonth=3),
        pd.offsets.BusinessDay(),
        pd.offsets.MonthBegin(normalize=True),
    ],
    ids=repr,
)
def test_index_freq_is_kept(freq):
    # Stored as the offset's class and arguments: the aliases changed between
    # pandas versions, e.g. the month end is "M" on pandas 2.0 and "ME" on 3.
    # A time series index without its freq cannot be shifted.
    index = pd.date_range("2024-01-01", periods=3, freq=freq)
    loaded = _loads(dumps(pd.Series([1, 2, 3], index=index)))
    assert loaded.index.freq == freq
    assert loaded.index.shift(1).equals(index.shift(1))
    timedeltas = pd.timedelta_range("1D", periods=2, freq="6h")
    assert _loads(dumps(timedeltas)).freq == timedeltas.freq


@pytest.mark.parametrize(
    "freq",
    [pd.offsets.CustomBusinessDay(holidays=["2024-01-02"]), pd.offsets.BusinessHour()],
    ids=repr,
)
def test_index_freq_with_other_arguments_is_unsupported(freq):
    # their arguments are a holiday calendar and times of day, not JSON
    index = pd.date_range("2024-01-01", periods=3, freq=freq)
    with pytest.raises(UnsupportedTypeException, match="freq=None"):
        dumps(index)
    # the hint in the message works
    assert _loads(dumps(pd.DatetimeIndex(index, freq=None))).freq is None


@pytest.mark.parametrize(
    "freq, match",
    [
        # not an offset, or not one of pandas.offsets
        ({"name": "BaseOffset", "n": 1, "normalize": False, "kwds": {"x": 1}}, "x"),
        ({"name": "__class__", "n": 1, "normalize": False, "kwds": {}}, "offset"),
        ({"name": "Day", "n": "1", "normalize": False, "kwds": {}}, "offset"),
        ({"name": "Week", "n": 1, "normalize": False, "kwds": {"weekday": [6]}}, None),
        # values that do not conform to the freq
        ({"name": "Day", "n": 2, "normalize": False, "kwds": {}}, "conform"),
    ],
)
def test_index_freq_is_validated(freq, match):
    dumped = dumps(pd.date_range("2024-01-01", periods=3, freq="D"))

    def edit(schema):
        schema["content"]["freq"]["content"] = json.dumps(json.dumps(freq))

    with pytest.raises((ValueError, TypeError), match=match):
        _loads(_with_edited_schema(dumped, edit))


def test_freq_only_on_datetime_like_indexes():
    dumped = dumps(pd.Index([1, 2, 3]))

    def edit(schema):
        parts = {"name": "Day", "n": 1, "normalize": False, "kwds": {}}
        freq = schema["content"]["freq"]
        freq["content"] = json.dumps(json.dumps(parts))
        # the name is None too, and shares the node of the same object
        freq["__id__"] = -1

    with pytest.raises(ValueError, match="Only a DatetimeIndex"):
        _loads(_with_edited_schema(dumped, edit))


def test_attrs_and_flags_are_kept():
    frame = pd.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)
    frame.attrs = {"unit": "m", "source": [1, 2]}
    loaded = _loads(dumps(frame))
    assert loaded.attrs == frame.attrs
    assert loaded.flags.allows_duplicate_labels is False
    series = pd.Series([1], name="s")
    series.attrs = {"unit": "s"}
    assert _loads(dumps(series)).attrs == {"unit": "s"}


def test_json_entries_of_the_wrong_type_are_refused():
    dumped = dumps(pd.DataFrame({"a": [1]}))

    def edit(schema):
        schema["content"]["allows_duplicate_labels"]["content"] = json.dumps(1)
        schema["content"]["allows_duplicate_labels"]["__id__"] = -1

    with pytest.raises(ValueError, match="must be a"):
        _loads(_with_edited_schema(dumped, edit))


def test_single_pandas_class_as_trusted():
    # a single class rather than a list, matched by its public name
    loads(dumps(pd.Int64Dtype()), trusted=pd.Int64Dtype)


def test_duplicate_labels_flag_is_validated():
    # a file cannot claim unique labels the data does not have
    dumped = dumps(pd.DataFrame({"a": [1, 2]}, index=["x", "x"]))

    def edit(schema):
        schema["content"]["allows_duplicate_labels"]["content"] = "false"

    with pytest.raises(pd.errors.DuplicateLabelError):
        _loads(_with_edited_schema(dumped, edit))


def test_pandas_types():
    # the public names of the classes the loaders build, which a user passes
    # as trusted; a tuple so that it cannot be changed by accident
    from skops.io._pandas import NODE_TYPE_MAPPING

    assert isinstance(PANDAS_TYPES, tuple)
    expected = {
        f"{node._module}.{name}"
        for node in NODE_TYPE_MAPPING.values()
        for name in node._classes
    }
    assert set(PANDAS_TYPES) == expected
    assert all(name.startswith("pandas.") for name in PANDAS_TYPES)
    assert "pandas.DataFrame" in PANDAS_TYPES


def test_pandas_types_are_not_trusted_by_default():
    dumped = dumps(FRAMES[1])
    untrusted = get_untrusted_types(data=dumped)
    assert "pandas.DataFrame" in untrusted
    with pytest.raises(UntrustedTypesFoundException, match="pandas.DataFrame"):
        loads(dumped)
    loads(dumped, trusted=untrusted)


def test_class_without_public_name_is_unsupported():
    # the dtype of a numpy-backed extension array has no public name; a file
    # never names a private class, whose module moves between versions
    with pytest.raises(UnsupportedTypeException, match="not supported"):
        dumps(pd.Series([1, 2]).array.dtype)


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
    # the name in the file is parsed by the declared dtype class, which the
    # user trusts, and not looked up in pandas' registry of extension dtypes,
    # where it could name the dtype of another library, whose code would run
    dumped = dumps(pd.Int64Dtype())

    def edit(schema):
        schema["content"]["name"]["content"] = json.dumps("period[M]")

    with pytest.raises(TypeError, match="Cannot construct"):
        loads(_with_edited_schema(dumped, edit), trusted=["pandas.Int64Dtype"])


@pytest.mark.parametrize(
    "module, class_name",
    [
        # a pandas class that is not a dtype
        ("pandas", "Series"),
        # dtypes whose parsers consult pandas' registry of extension dtypes or
        # the file system, which have loaders of their own
        ("pandas", "DatetimeTZDtype"),
        ("pandas", "IntervalDtype"),
        # its names would be parsed by pyarrow
        ("pandas", "ArrowDtype"),
        # a private module, and pandas' own test suite, which ships with it
        ("pandas.core.dtypes.dtypes", "NumpyEADtype"),
        ("pandas.tests.extension.decimal.array", "DecimalDtype"),
        ("pandas.conftest", "Anything"),
    ],
)
def test_dtype_node_refuses_other_classes(module, class_name):
    # the name based dtype loader only accepts the public dtype classes it
    # builds; anything else is refused while the file is read, before any
    # module the file names could be imported
    dumped = dumps(pd.Int64Dtype())

    def edit(schema):
        schema["__module__"], schema["__class__"] = module, class_name
        schema["content"]["name"]["content"] = json.dumps(
            "datetime64[ns, dateutil//etc/localtime]"
        )

    imported = set(sys.modules)
    with pytest.raises(
        ValueError, match="not a class PandasExtensionDtypeNode|only be loaded by"
    ):
        get_untrusted_types(data=_with_edited_schema(dumped, edit))
    assert module == "pandas" or module in imported or module not in sys.modules


# The skops loaders that build an object from the class a file names, or hand
# out the class itself, which a file could then call through other objects.
# the loaders that build an instance from the file; TypeNode, which yields
# the class object itself, may carry a pandas name
GENERIC_LOADERS = [
    "ObjectNode",
    "ConstructorFromReduceNode",
    "FunctionNode",
    "MethodNode",
    "PartialNode",
]


@pytest.mark.parametrize("loader", GENERIC_LOADERS)
def test_pandas_names_only_on_pandas_loaders(loader):
    # Trusting PANDAS_TYPES must not let a generic loader build those classes:
    # it would call their constructors with arguments from the file, e.g. a
    # DatetimeTZDtype whose time zone pandas reads from any file on disk.
    for name in PANDAS_TYPES:
        module, _, class_name = name.rpartition(".")
        state = {
            "__class__": class_name,
            "__module__": module,
            "__loader__": loader,
            "__id__": 1,
            "content": {},
        }
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as zip_file:
            zip_file.writestr(
                "schema.json", json.dumps({**state, "protocol": PROTOCOL})
            )
        with pytest.raises(ValueError, match="can only be loaded by Pandas"):
            get_untrusted_types(data=buffer.getvalue())


def test_datetime_dtype_through_a_generic_loader_is_refused():
    # the case of the review: pandas.DatetimeTZDtype("ns", "dateutil/<path>")
    # through ConstructorFromReduceNode, whose time zone would be read from a
    # file outside the zoneinfo directories
    dumped = dumps(pd.Interval(0, 1))

    def edit(schema):
        schema["__module__"], schema["__class__"] = "pandas", "DatetimeTZDtype"
        args = schema["content"]["content"]
        args[0]["content"] = json.dumps("ns")
        args[1]["content"] = json.dumps("dateutil//etc/localtime")
        del args[2:]

    with pytest.raises(ValueError, match="not by ConstructorFromReduceNode"):
        loads(_with_edited_schema(dumped, edit), trusted=PANDAS_TYPES)


@pytest.mark.parametrize("cls", [pd.DataFrame, pd.DatetimeTZDtype])
def test_pandas_class_itself_round_trips(cls):
    # The class itself, rather than an instance, e.g. a parameter set to
    # pandas.DataFrame, is saved by TypeNode like any other class, under the
    # name get_type_name gives it: the public one on pandas 3, which is one of
    # PANDAS_TYPES, the defining module before. TypeNode yields the class
    # object and builds nothing, so it may carry a pandas name.
    dumped = dumps({"cls": cls})
    assert get_untrusted_types(data=dumped) == [get_type_name(cls)]
    assert loads(dumped, trusted=[get_type_name(cls)])["cls"] is cls


def test_array_node_refuses_other_classes():
    # a trusted name cannot label an object of another class: an array
    # declared as a DataFrame is refused while the file is read
    dumped = dumps(pd.array(["a", None], dtype="string"))

    def edit(schema):
        schema["__module__"], schema["__class__"] = "pandas", "DataFrame"

    with pytest.raises(
        ValueError, match="not a class PandasExtensionArrayNode|only be loaded by"
    ):
        get_untrusted_types(data=_with_edited_schema(dumped, edit))


def test_na_node_refuses_other_modules():
    dumped = dumps(pd.NA)
    assert get_untrusted_types(data=dumped) == ["pandas.api.typing.NAType"]

    def edit(schema):
        schema["__module__"] = "pandas.io.anything"

    with pytest.raises(ValueError, match="not a class PandasNANode"):
        get_untrusted_types(data=_with_edited_schema(dumped, edit))


def test_arrow_backed_types_are_unsupported():
    pa = pytest.importorskip("pyarrow")
    arrow = pd.ArrowDtype(pa.int64())
    for obj in [arrow, pd.array([1, None], dtype=arrow), pd.Series([1], dtype=arrow)]:
        with pytest.raises(UnsupportedTypeException, match="not supported"):
            dumps(obj)
    # the string dtype with pyarrow storage is a string dtype, and supported
    strings = pd.array(["a", None], dtype=pd.StringDtype("pyarrow"))
    _assert_vals_equal(strings, _loads(dumps(strings)))


def _string_dtypes():
    # every string dtype this pandas version has: both storages, if pyarrow is
    # installed, with pandas.NA and, from pandas 2.1 on, with NaN as missing
    # value, which pandas 2.1 and 2.2 call the "pyarrow_numpy" storage
    dtypes = [pd.StringDtype("python")]
    candidates = [
        lambda: pd.StringDtype("pyarrow"),
        lambda: pd.StringDtype("python", na_value=np.nan),
        lambda: pd.StringDtype("pyarrow", na_value=np.nan),
    ]
    has_na_value = True
    for candidate in candidates:
        try:
            dtypes.append(candidate())
        except ImportError:
            pass
        except TypeError:
            has_na_value = False
    if not has_na_value:
        # pandas 2.1 and 2.2; deprecated from 2.3 on, which has na_value
        try:
            dtypes.append(pd.StringDtype("pyarrow_numpy"))
        except (ImportError, ValueError):
            pass
    return dtypes


@pytest.mark.parametrize("dtype", _string_dtypes(), ids=repr)
def test_string_dtype_keeps_missing_value_and_storage(dtype):
    # the name of a string dtype leaves out its missing value: on pandas 2.1
    # and 2.2, "string" is also the name of the dtype with NaN, which would
    # then load as the one with pandas.NA, whose comparisons return NA
    series = pd.Series(["a", None], dtype=dtype)
    loaded = _loads(dumps(series))
    assert loaded.dtype == series.dtype
    assert type(loaded.array) is type(series.array)
    pd.testing.assert_series_equal(loaded == "b", series == "b")


def test_string_dtype_fallbacks(monkeypatch):
    # without pyarrow the python storage is used, and without a string dtype
    # with NaN as missing value, object arrays of strings
    from skops.io._pandas import _string_dtype

    real = pd.StringDtype

    def no_pyarrow(storage=None, *args, **kwargs):
        if storage is not None and storage.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real(storage, *args, **kwargs)

    monkeypatch.setattr(pd, "StringDtype", no_pyarrow)
    assert _string_dtype("pyarrow", nan=False) == real("python")
    nan_strings = _string_dtype("pyarrow", nan=True)
    assert nan_strings == np.dtype(object) or nan_strings == real(
        "python", na_value=np.nan
    )

    def old_pandas(storage=None, **kwargs):
        if kwargs or storage != "python":
            raise ImportError("no pyarrow") if kwargs == {} else TypeError("old")
        return real(storage)

    monkeypatch.setattr(pd, "StringDtype", old_pandas)
    assert _string_dtype("python", nan=True) == np.dtype(object)


@pytest.mark.parametrize(
    "storage, na_value", [("pyarrow_numpy", "NaN"), ("python", "nan"), (1, "NA")]
)
def test_string_dtype_node_refuses_unknown_parts(storage, na_value):
    dumped = dumps(pd.StringDtype("python"))

    def edit(schema):
        schema["content"]["storage"]["content"] = json.dumps(storage)
        schema["content"]["na_value"]["content"] = json.dumps(na_value)

    with pytest.raises(ValueError, match="Unknown string dtype"):
        loads(_with_edited_schema(dumped, edit), trusted=["pandas.StringDtype"])


@pytest.mark.parametrize(
    "name", ["dateutil//etc/localtime", "tzlocal()", "../../etc/localtime", "/x"]
)
def test_timezone_from_file_is_restricted(name):
    # pandas' own parser would open any file on disk for "dateutil/<path>";
    # only "UTC", fixed offsets and valid zoneinfo keys are accepted
    dumped = dumps(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    trusted = get_untrusted_types(data=dumped)

    def edit(schema):
        schema["content"]["values"]["content"]["tz"]["content"] = json.dumps(name)

    with pytest.raises(ValueError, match="not a known time zone"):
        loads(_with_edited_schema(dumped, edit), trusted=trusted)


def test_file_backed_zone_is_unsupported():
    # a zone created with ZoneInfo.from_file may carry a key that does not
    # describe its data, and the key is all the file holds, so it is refused
    # like skops.io refuses such zones on their own. This one is a minimal
    # version 1 TZif file with a single offset of one hour, under the key UTC.
    tzif = (
        b"TZif\x00"
        + bytes(15)
        # number of isut, isstd, leap, transition, type and abbreviation entries
        + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
        # the single type entry: utoff, isdst, abbreviation index
        + struct.pack(">lbb", 3600, 0, 0)
        + b"UTC\x00"
    )
    tz = zoneinfo.ZoneInfo.from_file(io.BytesIO(tzif), key="UTC")
    with pytest.raises(UnsupportedTypeException, match="ZoneInfo.from_file"):
        dumps(pd.DatetimeTZDtype("ns", tz))
    with pytest.raises(UnsupportedTypeException, match="ZoneInfo.from_file"):
        dumps(pd.array(pd.to_datetime(["2024-01-01"]).tz_localize(tz)))


def test_declared_class_must_be_one_the_loader_builds():
    # the audit trusts the class a file declares, so a loader only accepts the
    # classes it builds: a trusted name cannot make it build another class
    dumped = dumps(pd.RangeIndex(3))

    def edit(schema):
        schema["__class__"] = "Series"

    with pytest.raises(
        ValueError, match="not a class PandasRangeIndexNode builds|only be loaded by"
    ):
        get_untrusted_types(data=_with_edited_schema(dumped, edit))

    def edit_module(schema):
        schema["__module__"] = "pandas.core.indexes.range"

    with pytest.raises(
        ValueError, match="not a class PandasRangeIndexNode builds|only be loaded by"
    ):
        get_untrusted_types(data=_with_edited_schema(dumped, edit_module))


def test_built_object_must_be_of_the_declared_class():
    # the declared class is one the loader builds, but the data builds another
    dumped = dumps(pd.Index(pd.to_datetime(["2024-01-01"])))

    def edit(schema):
        schema["__class__"] = "Index"

    with pytest.raises(ValueError, match="builds pandas.DatetimeIndex"):
        loads(_with_edited_schema(dumped, edit), trusted=["pandas.Index"])


def test_numpy_backed_array_follows_the_declared_class():
    # the declared class, not the dtype of the values, decides between the
    # plain wrapper and the datetime-like arrays, and must fit the values
    timedeltas = dumps(pd.array(pd.to_timedelta([1], unit="s")))
    aware = dumps(pd.array(pd.to_datetime(["2024-01-01"]).tz_localize("UTC")))

    def to_datetime_array(schema):
        schema["__class__"] = "DatetimeArray"

    with pytest.raises(ValueError, match="must have a datetime64 dtype"):
        loads(
            _with_edited_schema(timedeltas, to_datetime_array),
            trusted=["pandas.arrays.DatetimeArray"],
        )

    def to_wrapper(schema):
        schema["__class__"] = "NumpyExtensionArray"

    # the plain wrapper is not supported yet
    with pytest.raises(ValueError, match="not a class PandasNumpyBackedArrayNode"):
        get_untrusted_types(data=_with_edited_schema(aware, to_wrapper))

    def with_zone(schema):
        schema["content"]["tz"]["content"] = json.dumps("UTC")

    with pytest.raises(ValueError, match="TimedeltaArray has no time zone"):
        loads(
            _with_edited_schema(timedeltas, with_zone),
            trusted=["pandas.arrays.TimedeltaArray"],
        )


def test_pandas_classes_as_trusted_entries():
    # a pandas class passed as trusted is matched by the public name the file
    # holds, which on pandas 2.x differs from its defining module
    loads(dumps(pd.Int64Dtype()), trusted=[pd.Int64Dtype])
    loads(dumps(pd.Series([1], index=pd.Index([1]))), trusted=[pd.Series, pd.Index])


def test_sub_second_offset_is_unsupported():
    # the name of a fixed offset holds whole seconds; a finer offset cannot be
    # written, and must not be rounded silently
    tz = dt.timezone(dt.timedelta(microseconds=1))
    with pytest.raises(UnsupportedTypeException, match="whole seconds"):
        dumps(pd.DatetimeTZDtype("ns", tz))


def test_child_of_wrong_kind_is_refused():
    # the values of an Index are an array; a file holding something else there
    # is refused while it is read, before anything is constructed
    dumped = dumps(pd.Index([1, 2]))

    def edit(schema):
        schema["content"]["values"] = {
            "__class__": "list",
            "__module__": "builtins",
            "__loader__": "ListNode",
            "content": [],
            "__id__": 1,
        }

    with pytest.raises(ValueError, match="Expected a node of type"):
        get_untrusted_types(data=_with_edited_schema(dumped, edit))


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _with_edited_files(dumped, edit):
    # like _with_edited_schema, with the other files of the archive editable
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        files = {n: zip_file.read(n) for n in zip_file.namelist() if n != "schema.json"}
    edit(schema, files)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        zip_file.writestr("schema.json", json.dumps(schema))
        for name, data in files.items():
            zip_file.writestr(name, data)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "obj, values_path",
    [
        (pd.Series([1, 2]), ("values",)),
        (pd.Index([1, 2]), ("values",)),
        (pd.DataFrame({"a": [1, 2]}), ("data", "content", 0)),
        (pd.array(pd.to_timedelta([1, 2], unit="s")), ("values",)),
    ],
    ids=["Series", "Index", "DataFrame", "TimedeltaArray"],
)
@pytest.mark.parametrize("kind", ["0-d", "scalar", "2-d"])
def test_values_must_be_one_dimensional(obj, values_path, kind):
    # A numpy scalar as the values of a Series is repeated for every label of
    # its index, and a RangeIndex of 10**9 labels takes no space in a file:
    # pandas would allocate gigabytes. Values must be 1-d arrays.
    dumped = dumps(obj)
    values = np.zeros((2, 2)) if kind == "2-d" else np.array(1)

    def edit(schema, files):
        node = schema["content"]
        for key in values_path:
            node = node[key]
        files[node["file"]] = _npy(values.astype(np.dtype(node.get("dtype", "f8"))))
        if kind == "scalar":
            # the class of a numpy scalar, which NdArrayNode builds
            node["__module__"], node["__class__"] = "numpy", "float64"
        if "index" in schema["content"] and kind != "2-d":
            stop = schema["content"]["index"]["content"]["stop"]
            stop["content"] = json.dumps(10**9)

    with pytest.raises(ValueError, match="one-dimensional"):
        _loads(_with_edited_files(dumped, edit))


def test_dataframe_columns_must_fit_its_axes():
    # pd.concat pads columns of different lengths, so one long column and many
    # short ones would take rows times columns of memory: refused before that
    frame = pd.DataFrame({f"c{i}": [1.0] for i in range(3)})
    dumped = dumps(frame)

    def long_column(schema, files):
        files[schema["content"]["data"]["content"][0]["file"]] = _npy(np.zeros(10**5))

    with pytest.raises(ValueError, match="cannot hold 3 columns"):
        _loads(_with_edited_files(dumped, long_column))

    def fewer_columns(schema, files):
        del schema["content"]["data"]["content"][0]

    with pytest.raises(ValueError, match="cannot hold 2 columns"):
        _loads(_with_edited_files(dumped, fewer_columns))


def test_series_values_must_fit_its_index():
    dumped = dumps(pd.Series([1, 2]))

    def long_index(schema, files):
        schema["content"]["index"]["content"]["stop"]["content"] = json.dumps(10**9)

    with pytest.raises(ValueError, match="cannot have an index of 1000000000"):
        _loads(_with_edited_files(dumped, long_index))


@pytest.mark.parametrize("values", ["x", 1, {"a": "b"}, ["a", 1], ["a", [None]]])
def test_json_values_must_be_strings(values):
    # values may be a JSON list of strings and missing values, see _Strings;
    # any other JSON is refused
    dumped = dumps(pd.Index(["a", "b"], dtype=object))

    def edit(schema):
        schema["content"]["values"]["content"] = json.dumps(values)

    with pytest.raises(ValueError, match="list of strings and missing values"):
        loads(_with_edited_schema(dumped, edit), trusted=["pandas.Index"])


def test_strings_are_one_json_list():
    # one node for the whole column, not one per string: about 15 times
    # smaller than before for short strings
    values = [f"category {i % 5000}" for i in range(20_000)] + [None, np.nan]
    series = pd.Series(values, dtype=object)
    dumped = dumps(series)
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    assert schema["content"]["values"]["__loader__"] == "JsonNode"
    assert len(dumped) < 30 * len(values)
    loaded = _loads(dumped)
    pd.testing.assert_series_equal(loaded, series)
    # the missing values keep their kind
    assert loaded[20_000] is None
    assert isinstance(loaded[20_001], float)


def test_other_object_values_keep_their_types():
    # numpy strings, or a mix with other objects, are saved item by item
    for values in [np.array(["a", "b"]).astype(object), ["a", b"b"], ["a", 1]]:
        series = pd.Series(values, dtype=object)
        loaded = _loads(dumps(series))
        assert [type(v) for v in loaded] == [type(v) for v in series]


def test_missing_entry_is_refused():
    dumped = dumps(pd.Index([1, 2]))

    def edit(schema):
        del schema["content"]["name"]

    with pytest.raises(ValueError, match="Expected the entries"):
        loads(_with_edited_schema(dumped, edit))


@pytest.mark.parametrize(
    "obj",
    [
        pd.PeriodDtype("M"),
        pd.period_range("2024-01", periods=2, freq="M"),
        pd.array(pd.period_range("2024-01", periods=1, freq="M")),
        pd.IntervalDtype("int64"),
        pd.interval_range(0, 2),
        pd.SparseDtype(float, 0.0),
        pd.arrays.SparseArray([0, 1]),
        pd.Series(pd.arrays.SparseArray([0, 1])),
        pd.Series([1, 2]).array,
    ],
    ids=_id,
)
def test_types_not_supported_yet(obj):
    # refused when saving, to keep the first version small
    with pytest.raises(UnsupportedTypeException, match="not supported yet"):
        dumps(obj)


def test_subclass_from_other_library_is_unsupported():
    class MySeries(pd.Series):
        pass

    with pytest.raises(UnsupportedTypeException, match="subclass of a pandas type"):
        dumps(MySeries([1, 2]))


class ThirdPartyDtype(pd.api.extensions.ExtensionDtype):
    """The dtype of an extension array defined outside pandas."""

    name = "third_party"
    type = object

    @classmethod
    def construct_array_type(cls):
        return ThirdPartyArray


class ThirdPartyArray(pd.api.extensions.ExtensionArray):
    """An extension array defined outside pandas, as pint-pandas does."""

    dtype = ThirdPartyDtype()

    def __init__(self, values):
        self.values = np.asarray(values, dtype=object)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(list(scalars))

    @classmethod
    def _from_factorized(cls, values, original):  # pragma: no cover
        return cls(values)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self.values[item]
        return type(self)(self.values[item])

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):  # pragma: no cover
        return np.asarray(self.values == np.asarray(other, dtype=object), dtype=bool)

    @property
    def nbytes(self):  # pragma: no cover
        return self.values.nbytes

    def isna(self):
        return np.array([value is None for value in self.values], dtype=bool)

    def take(self, indices, allow_fill=False, fill_value=None):
        values = pd.api.extensions.take(
            self.values, indices, allow_fill=allow_fill, fill_value=fill_value
        )
        return type(self)(values)

    def copy(self):
        return type(self)(self.values.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([array.values for array in to_concat]))


def test_extension_types_from_other_libraries_are_saved_generically():
    # They derive from pandas' extension base classes, but are saved through
    # their state under their own class, as before pandas was supported, and
    # not rebuilt as a pandas class.
    array = ThirdPartyArray(["a", 1])
    dumped = dumps({"fitted_": array, "dtype_": array.dtype})
    untrusted = get_untrusted_types(data=dumped)
    assert {get_type_name(ThirdPartyArray), get_type_name(ThirdPartyDtype)} <= set(
        untrusted
    )
    loaded = loads(dumped, trusted=untrusted)
    assert type(loaded["fitted_"]) is ThirdPartyArray
    assert loaded["fitted_"].values.tolist() == ["a", 1]
    assert type(loaded["dtype_"]) is ThirdPartyDtype


@pytest.mark.parametrize(
    "container",
    [pd.Series, pd.Index, lambda array: pd.DataFrame({"a": array})],
    ids=["Series", "Index", "DataFrame"],
)
def test_containers_of_extension_arrays_from_other_libraries(container):
    # The array is saved the generic way, under its own class, and the
    # container loads once the user trusts that class too.
    obj = container(ThirdPartyArray(["a", None, "b"]))
    dumped = dumps(obj)
    untrusted = set(get_untrusted_types(data=dumped))
    assert untrusted - set(PANDAS_TYPES) == {get_type_name(ThirdPartyArray)}
    with pytest.raises(UntrustedTypesFoundException, match="ThirdPartyArray"):
        loads(dumped, trusted=PANDAS_TYPES)
    loaded = loads(dumped, trusted=[*PANDAS_TYPES, ThirdPartyArray])
    values = loaded.iloc[:, 0] if isinstance(loaded, pd.DataFrame) else loaded
    assert type(values.array) is ThirdPartyArray
    assert list(values.array.values) == ["a", None, "b"]


def test_visualize(capsys):
    visualize(dumps(FRAMES[0]))
    assert "pandas.DataFrame" in capsys.readouterr().out


def test_file_uses_public_type_names():
    # older pandas versions report the defining module of a class, e.g.
    # pandas.core.series.Series, and pandas 3 reports pandas.Series; the file
    # always holds the public name, which is the one a user passes as trusted
    dumped = dumps(pd.Series([1], index=pd.Index([1])))
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    assert (schema["__module__"], schema["__class__"]) == ("pandas", "Series")
    index = schema["content"]["index"]
    assert (index["__module__"], index["__class__"]) == ("pandas", "Index")


def test_dtype_not_rebuildable_from_its_name_is_unsupported(monkeypatch):
    # A dtype stored by its name is refused when saving, rather than failing
    # or changing when loading, if its name does not rebuild it: a guard for
    # future pandas versions. ArrowDtype(pyarrow.string()) is such a dtype, a
    # name "string[pyarrow]" that pandas reserves for its StringDtype.
    monkeypatch.setattr(
        pd.Int64Dtype, "construct_from_string", classmethod(lambda cls, name: None)
    )
    with pytest.raises(UnsupportedTypeException, match="cannot be rebuilt"):
        dumps(pd.Int64Dtype())


def test_renamed_classes_match_pandas_version():
    # each rename is a fact about pandas: before the version that introduced
    # the new name only the old one exists, from then on the new one does
    def exists(name):
        with warnings.catch_warnings():
            # the old name may live on as a deprecated alias that warns
            warnings.simplefilter("ignore")
            return any(hasattr(module, name) for module in (pd, pd.arrays))

    for old, (new, since) in _RENAMED_CLASSES.items():
        if Version(pd.__version__) < Version(since):
            assert exists(old) and not exists(new), old
        else:
            assert exists(new), new


# Small objects which together hold every pandas node, for the tampering test.
TAMPER_OBJECTS = [
    pd.Series(["a", None], index=pd.RangeIndex(2), name="s"),
    pd.DataFrame({"i": [1, 2]}, index=pd.Index(["x", "y"])),
    pd.MultiIndex.from_tuples([("a", 1)]),
    pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    pd.array([1, None], dtype="Int64"),
    pd.array(["a", None], dtype="string"),
    pd.Categorical(["a"]),
    pd.Timestamp("2024-01-01"),
    pd.Timedelta(1),
    pd.NA,
    pd.NaT,
    pd.array(pd.to_timedelta([1], unit="s")),
]

# Names no loader builds: pandas' test suite and configuration, which ship
# with it, a private class, an invented module, and no pandas at all.
HOSTILE_NAMES = [
    "pandas.tests.extension.decimal.array.DecimalDtype",
    "pandas.conftest.Anything",
    "pandas.core.dtypes.dtypes.NumpyEADtype",
    "pandas.io.anything.NAType",
    "builtins.eval",
    "os.system",
]


def _pandas_node_paths(state, path=()):
    # the paths to the states of the pandas nodes in a schema
    from skops.io._pandas import NODE_TYPE_MAPPING

    if isinstance(state, dict):
        if (state.get("__loader__"), PROTOCOL) in NODE_TYPE_MAPPING:
            yield path
        for key, value in state.items():
            yield from _pandas_node_paths(value, path + (key,))
    elif isinstance(state, list):
        for i, value in enumerate(state):
            yield from _pandas_node_paths(value, path + (i,))


def _at(state, path):
    for key in path:
        state = state[key]
    return state


def _builds_declared(module, name, obj):
    # whether ``obj`` is of the class a file declares, decided independently
    # of the check the loaders do themselves
    if module == "pandas.api.typing":
        return obj is {"NAType": pd.NA, "NaTType": pd.NaT}[name]
    if name in ("StringArray", "ArrowStringArray"):
        # the class follows the storage of the environment, or is an object
        # array where pandas has no string dtype with the saved missing value
        strings = (pd.arrays.StringArray, pd.arrays.ArrowStringArray)
        return isinstance(obj, strings) or (
            type(obj) is _NumpyBackedArray and obj.to_numpy().dtype == object
        )
    if name == "StringDtype":
        return isinstance(obj, pd.StringDtype) or obj == np.dtype(object)
    namespace = pd.arrays if module == "pandas.arrays" else pd
    return type(obj) is getattr(namespace, name)


@pytest.mark.parametrize("obj", TAMPER_OBJECTS, ids=_id)
def test_tampered_class_names(obj, monkeypatch):
    # Every pandas node of a file, declared as every other pandas class and as
    # hostile names, while the user trusts PANDAS_TYPES and the hostile names.
    # A name the node does not build is refused while the file is read; one it
    # does build either fails to construct or builds exactly that class.
    from skops.io._pandas import NODE_TYPE_MAPPING, _PandasNode

    built = []
    construct = _PandasNode._construct

    def recording_construct(node):
        result = construct(node)
        built.append((node.module_name, node.class_name, result))
        return result

    monkeypatch.setattr(_PandasNode, "_construct", recording_construct)
    dumped = dumps(obj)
    with ZipFile(io.BytesIO(dumped)) as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    imported = set(sys.modules)
    trusted = [*PANDAS_TYPES, *HOSTILE_NAMES]
    # the classes one loader builds from the same data: the string arrays of
    # either storage
    interchangeable = [{"pandas.arrays.StringArray", "pandas.arrays.ArrowStringArray"}]
    for path in _pandas_node_paths(schema):
        state = _at(schema, path)
        node = NODE_TYPE_MAPPING[(state["__loader__"], PROTOCOL)]
        original = f"{state['__module__']}.{state['__class__']}"
        for name in [*PANDAS_TYPES, *HOSTILE_NAMES]:
            if name == original:
                continue
            module, _, class_name = name.rpartition(".")

            def edit(schema):
                tampered = _at(schema, path)
                tampered["__module__"], tampered["__class__"] = module, class_name

            tampered = _with_edited_schema(dumped, edit)
            if module != node._module or class_name not in node._classes:
                # refused by the node, or before it is built, if another
                # pandas loader builds that name
                with pytest.raises(ValueError, match="is not a class|only be loaded"):
                    get_untrusted_types(data=tampered)
                continue
            built.clear()
            try:
                loads(tampered, trusted=trusted)
            except (ValueError, TypeError):
                continue
            assert any({original, name} <= names for names in interchangeable), name
            for built_module, built_name, result in built:
                assert _builds_declared(built_module, built_name, result), name
    # no module the file names was imported
    assert {m for m in set(sys.modules) - imported if m.startswith("pandas.")} == set()


FIXTURE_DIR = Path(__file__).parent / "data"


def cross_version_objects():
    """Objects whose files, written by one pandas version, load on every other.

    :func:`write_pandas_fixture_file` dumps them with the running pandas version
    into ``data/``, and :func:`test_load_file_of_other_pandas_version` loads
    every file there. Add a file for a new pandas minor version when it changes
    how any of these is represented, and never regenerate an existing one.

    They hold every pandas node, and are built the same way on every pandas
    version: frequencies from offsets, not from their aliases, which changed.
    The string dtypes, which depend on the version and on pyarrow, are in
    :func:`string_fixture_objects`.
    """
    frame = pd.DataFrame(
        {
            "i": [1, 2],
            "f": [1.5, np.nan],
            "s": ["a", None],
            "o": pd.Series(["a", 1], dtype=object),
            "c": pd.Categorical(["x", "y"], categories=["y", "x"], ordered=True),
            "t": pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin"),
        }
    )
    frame.attrs = {"unit": "m"}
    return {
        "str_index": pd.Index(["a", None, "c"], name="strings"),
        "str_series": pd.Series(["x", "y", None], index=["a", "b", "c"]),
        "mixed_frame": frame,
        "unique_labels": pd.DataFrame({"a": [1]}).set_flags(
            allows_duplicate_labels=False
        ),
        "datetime_index": pd.date_range("2024-01-01", periods=3, name="dates"),
        "datetime_index_tz": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "month_end_index": pd.date_range(
            "2024-01-31", periods=3, freq=pd.offsets.MonthEnd()
        ),
        "hourly_index": pd.date_range("2024-01-01", periods=3, freq=pd.offsets.Hour()),
        "datetime_fixed_offset": pd.to_datetime(["2024-01-01T00:00:00+01:00"]),
        "timedelta_index": pd.timedelta_range("1D", periods=2),
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
        "duplicate_columns": pd.DataFrame([[1, 2]], columns=["a", "a"]),
        # the shape of category_encoders' TargetEncoder.mapping
        "encoder_mapping": {"col": pd.Series([0.49, 0.66], index=pd.Index([1, 2]))},
        "dtypes": [
            pd.Int64Dtype(),
            pd.CategoricalDtype(["b", "a"], ordered=True),
            pd.DatetimeTZDtype("ns", "Europe/Berlin"),
        ],
        "scalars": [
            pd.NA,
            pd.NaT,
            pd.Timestamp("2024-01-01 00:00:00.000000001", tz="Europe/Berlin"),
            pd.Timedelta(1, unit="ns"),
        ],
    }


def _string_key(dtype):
    # the storage and the missing value of a string dtype, e.g. "pyarrow-NaN"
    storage = "python" if dtype.storage == "python" else "pyarrow"
    return f"{storage}-{'NaN' if isinstance(dtype.na_value, float) else 'NA'}"


def string_fixture_objects():
    """A Series of every string dtype the running pandas version has.

    Which those are depends on the pandas version and on pyarrow, so they are
    checked by how they behave rather than against objects built when loading.
    """
    return {
        _string_key(dtype): pd.Series(["a", None, "c"], dtype=dtype)
        for dtype in _string_dtypes()
    }


def write_pandas_fixture_file():
    """Dump the cross-version objects with the running pandas version.

    Run it with pyarrow installed, so that the file has the pyarrow string
    dtypes too.
    """
    import pyarrow  # noqa: F401

    path = FIXTURE_DIR / f"pandas-{pd.__version__}.skops"
    objects = {"common": cross_version_objects(), "strings": string_fixture_objects()}
    dump(objects, path, compression=ZIP_DEFLATED, compresslevel=9)
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
    if expected is pd.NA or expected is pd.NaT:
        assert actual is expected
        return
    assert type(actual) is type(expected)
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        assert expected.attrs == actual.attrs
        assert (
            expected.flags.allows_duplicate_labels
            == actual.flags.allows_duplicate_labels
        )
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
        if isinstance(expected, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            assert expected.freq == actual.freq
        np.testing.assert_array_equal(_as_objects(expected), _as_objects(actual))
    elif isinstance(expected, pd.CategoricalDtype):
        # the categories keep the string dtype of the version that wrote them
        assert expected.ordered == actual.ordered
        assert_same_data(expected.categories, actual.categories)
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


def _has_string_dtype(key):
    # whether this pandas version has the string dtype of ``key``
    return key in {_string_key(dtype) for dtype in _string_dtypes()}


def assert_same_strings(key, actual):
    # A Series of strings from string_fixture_objects, loaded with this pandas
    # version: same values, same missing value semantics, and the same storage
    # if pyarrow is installed.
    assert _as_objects(actual).tolist() == ["a", None, "c"]
    storage, na_value = key.split("-")
    equal = actual == "b"
    if na_value == "NA":
        # "boolean", or "bool[pyarrow]" for the pyarrow storage
        assert equal.isna().tolist() == [False, True, False]
    else:
        assert equal.dtype == bool and equal.tolist() == [False, False, False]
    if isinstance(actual.dtype, pd.StringDtype):
        # the storage is kept when this version has it with the same missing
        # value: pandas 2.1 and 2.2 have NaN strings with pyarrow storage only
        if _has_string_dtype(key):
            assert _string_key(actual.dtype) == key
    else:
        # pandas versions without a string dtype with NaN keep object strings
        assert na_value == "NaN" and actual.dtype == object


@pytest.mark.parametrize(
    "path", sorted(FIXTURE_DIR.glob("pandas-*.skops")), ids=lambda path: path.stem
)
def test_load_file_of_other_pandas_version(path):
    # the names in the file are the public ones, which every version resolves
    assert set(get_untrusted_types(file=path)) <= set(PANDAS_TYPES)
    loaded = load(path, trusted=PANDAS_TYPES)
    assert_same_data(cross_version_objects(), loaded["common"])
    assert loaded["strings"]
    for key, strings in loaded["strings"].items():
        assert_same_strings(key, strings)
