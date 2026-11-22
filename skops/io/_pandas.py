"""Persistence of pandas objects.

pandas objects are not persisted through ``__reduce__`` or ``__getstate__``:
those expose internals such as block managers, index engines and reference
trackers, which change between pandas versions and cannot be rebuilt from data
alone. Instead, every object is taken apart into the public pieces its
constructor accepts, and rebuilt by calling that constructor with an explicit
dtype, so that no type inference happens on load:

- an :class:`~pandas.Index` is stored as its values and name,
- a :class:`~pandas.Series` as its values, index and name,
- a :class:`~pandas.DataFrame` as its columns, index and one array per column,
- extension arrays as the numpy arrays or scalars they are made of, plus their
  dtype, and extension dtypes as their string representation,
- a :class:`~pandas.Timestamp` or :class:`~pandas.Timedelta` as the numpy
  scalar of its value, plus the zone of an aware Timestamp, and ``pandas.NA``
  and ``pandas.NaT`` by their names alone.

Values with a numpy dtype are stored as numpy arrays, everything else as lists
of scalars.

pandas is optional and slow to import, so ``skops.io`` does not import it. The
``get_state`` handlers are registered on the first dump after the user has
imported pandas, see :func:`register_if_imported`, and the nodes only import
pandas when they construct an object.

Not preserved: the ``freq`` of datetime-like indexes and arrays, and the
``attrs`` and ``flags`` of a Series or DataFrame. A string dtype keeps its
missing value, ``pandas.NA`` or ``NaN``, and its storage, python or pyarrow,
when the loading environment has them, see :func:`_string_dtype`.

Not supported: the pyarrow backed arrays and dtypes, those with an
``ArrowDtype``, whose names pyarrow parses. The string dtype with pyarrow
storage is a string dtype, and supported as one.

The pandas types are not trusted by default, since this support is new; a user
passes the names a file holds, see :func:`skops.io.get_untrusted_types`.
Trusting them should mean trusting pandas and nothing else, so loading stays
within pandas' constructors and the data from the file. Every node lists the
public pandas classes it builds, by their name in ``pandas`` or
``pandas.arrays``; a file declaring any other class is refused while it is
read, before anything is constructed, and the built object is checked to be of
the declared class. Classes are looked up in those two namespaces rather than
imported from a module the file names, so a file cannot make the loader import
any other part of pandas, such as its test suite, which ships with it. In
particular, no name from the file is looked up in pandas' registry of extension
dtypes, where any imported library can register a dtype whose parser would
then run, and time zones are parsed here rather than by pandas, whose parser
can open any file on disk.
"""

from __future__ import annotations

import datetime
import pickle
import re
import sys
import warnings
import zoneinfo
from typing import Any

import numpy as np

from ._audit import Node, get_tree
from ._general import JsonNode, ListNode
from ._numpy import NdArrayNode
from ._protocol import PROTOCOL
from ._utils import (
    LoadContext,
    SaveContext,
    TrustedTypes,
    _get_state,
    get_module,
    get_state,
)
from .exceptions import UnsupportedTypeException

# Classes that later pandas versions renamed, mapped to their current name and
# the pandas version that introduced it, so that a file never names a class
# the loading version may not have. An entry can go once the versions before
# the rename are no longer supported. The test suite checks each entry against
# the running pandas version.
_RENAMED_CLASSES = {
    "PandasArray": ("NumpyExtensionArray", "2.1"),
}

_ARROW_UNSUPPORTED = (
    "The pyarrow backed arrays and dtypes of pandas, those with an ArrowDtype,"
    " are not supported yet."
)


def _public_name(cls: type) -> str | None:
    """The name a file holds for a pandas class, or ``None`` if it has none.

    This is also the name a user passes as ``trusted``: the name of the class
    in ``pandas.arrays`` for the array classes and in ``pandas`` for everything
    else, under its current name if it was renamed. pandas 3 reports these as
    ``__module__`` already, but older versions report the defining module, e.g.
    ``pandas.core.series``, and a file must not depend on that, since those
    modules are private and move between versions. The types of ``pandas.NA``
    and ``pandas.NaT`` are named by ``pandas.api.typing``, where pandas 3
    defines them, on every version. Deprecated aliases such as
    ``pandas.arrays.PandasArray`` warn when accessed, hence the suppression.
    """
    import pandas as pd

    if cls is type(pd.NA) or cls is type(pd.NaT):
        return f"pandas.api.typing.{cls.__name__}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in (pd.arrays, pd):
            if getattr(module, cls.__name__, None) is cls:
                class_name = _RENAMED_CLASSES.get(cls.__name__, (cls.__name__,))[0]
                return f"{module.__name__}.{class_name}"
    return None


def _nearest_public_name(cls: type) -> str | None:
    # The public name of ``cls`` or of its closest base class that has one,
    # e.g. ``pandas.arrays.ArrowStringArray`` for the private string array
    # classes of some pandas versions.
    for base in cls.__mro__:
        if (name := _public_name(base)) is not None:
            return name
    return None  # pragma: no cover


def _public_trusted(trusted: Any) -> Any:
    # pandas classes passed as ``trusted`` are matched by the public names the
    # file holds, rather than by their defining module, which pandas before 3.0
    # reports, or by an old name.
    if isinstance(trusted, (str, type)):
        trusted = [trusted]
    if not isinstance(trusted, (list, tuple)):
        return trusted
    return [
        (_public_name(entry) or entry)
        if isinstance(entry, type) and entry.__module__.partition(".")[0] == "pandas"
        else entry
        for entry in trusted
    ]


def _pandas_state(
    obj: Any,
    loader: str,
    content: dict[str, Any],
    save_context: SaveContext,
    name: str | None = None,
) -> dict[str, Any]:
    # ``name`` is the public name to write, if it is not the one of the class.
    cls = type(obj)
    # The nodes below rebuild objects with the pandas constructors, so an
    # instance of a subclass defined by another library would silently be
    # loaded as its pandas base class. Refuse those instead.
    if cls.__module__.partition(".")[0] != "pandas":
        raise UnsupportedTypeException(
            f"{get_module(cls)}.{cls.__name__} is a subclass of a pandas type"
            " defined outside pandas, which is not supported: it would be loaded"
            " as its pandas base class."
        )

    # Only write a file the loader accepts: a class that is not among the
    # ones the node builds, e.g. a private one, is refused now rather than
    # when the file is loaded.
    node = NODE_TYPE_MAPPING[(loader, PROTOCOL)]
    name = name or _public_name(cls)
    module, _, class_name = (name or "").rpartition(".")
    if module != node._module or class_name not in node._classes:
        raise UnsupportedTypeException(
            f"Objects of type {get_module(cls)}.{cls.__name__} are not supported yet."
        )
    return {
        "__class__": class_name,
        "__module__": module,
        "__loader__": loader,
        "content": {
            key: get_state(value, save_context) for key, value in content.items()
        },
    }


def _values(obj: Any) -> Any:
    # The array behind an Index or Series: a numpy array for numpy dtypes, the
    # extension array otherwise.
    if isinstance(obj.dtype, np.dtype):
        return obj.to_numpy()
    return obj.array


def index_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"values": _values(obj), "name": obj.name}
    return _pandas_state(obj, "PandasIndexNode", content, save_context)


def range_index_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"start": obj.start, "stop": obj.stop, "step": obj.step, "name": obj.name}
    return _pandas_state(obj, "PandasRangeIndexNode", content, save_context)


def multi_index_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {
        "levels": list(obj.levels),
        "codes": list(obj.codes),
        "sortorder": obj.sortorder,
        "names": list(obj.names),
    }
    return _pandas_state(obj, "PandasMultiIndexNode", content, save_context)


def series_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"values": _values(obj), "index": obj.index, "name": obj.name}
    return _pandas_state(obj, "PandasSeriesNode", content, save_context)


def dataframe_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {
        "columns": obj.columns,
        "index": obj.index,
        "data": [_values(obj.iloc[:, i]) for i in range(obj.shape[1])],
    }
    return _pandas_state(obj, "PandasDataFrameNode", content, save_context)


_FIXED_OFFSET = re.compile(r"^UTC([+-])(\d{2}):(\d{2})(?::(\d{2}))?$")


def _timezone_name(tz: Any) -> str:
    # The name of a time zone as written to the file: the key of a zoneinfo
    # or pytz zone, "UTC", or a fixed offset as "UTC+01:00", with its seconds
    # as "UTC+00:19:32" if it has any. These are the only names ``_timezone``
    # accepts when loading.
    if isinstance(tz, zoneinfo.ZoneInfo):
        try:
            tz.__reduce__()
        except pickle.PicklingError as err:
            # Created with ``ZoneInfo.from_file``: its key, if it has one,
            # need not describe its data, and the key is all the file holds.
            # Refused like ``skops.io`` refuses such objects on their own.
            raise UnsupportedTypeException(
                "ZoneInfo objects created with ZoneInfo.from_file are not"
                " supported, since their time zone data cannot be loaded from a"
                " key; create the zone with ZoneInfo(key) instead."
            ) from err
    name = getattr(tz, "key", None) or getattr(tz, "zone", None)
    if name is not None:
        return name
    offset = tz.utcoffset(None)
    if offset is None or offset.microseconds:
        raise UnsupportedTypeException(
            f"The time zone {tz!r} has neither a name nor a fixed offset of whole"
            " seconds, so it cannot be saved."
        )
    seconds = int(offset.total_seconds())
    if seconds == 0:
        return "UTC"
    sign, seconds = ("-", -seconds) if seconds < 0 else ("+", seconds)
    name = f"UTC{sign}{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"
    if seconds % 60:
        name += f":{seconds % 60:02d}"
    return name


def _timezone(name: str) -> str | datetime.tzinfo:
    # The time zone for a name from the file, parsed here rather than by
    # pandas: pandas' own parser would also accept "tzlocal()" and
    # "dateutil/<path>", the latter opening any file on disk, and it drops the
    # seconds of a fixed offset. "UTC" and fixed offsets become the
    # ``datetime.timezone`` objects pandas builds for them too. A zone key is
    # checked with zoneinfo, which only looks in its own directories, and then
    # handed to pandas as a name, so that pandas builds the zone of its
    # default implementation and the loaded zone equals the saved one.
    if name == "UTC":
        return datetime.timezone.utc
    match = _FIXED_OFFSET.match(name)
    if match:
        sign, hours, minutes, seconds = match.groups()
        offset = datetime.timedelta(
            hours=int(hours), minutes=int(minutes), seconds=int(seconds or 0)
        )
        return datetime.timezone(-offset if sign == "-" else offset)
    try:
        zoneinfo.ZoneInfo(name)
    except (ValueError, KeyError, OSError) as err:
        raise ValueError(
            f"{name!r} is not a known time zone. This is probably due to a"
            " corrupted or a malicious file."
        ) from err
    return name


def numpy_backed_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # NumpyExtensionArray, DatetimeArray and TimedeltaArray wrap a numpy array.
    # Time zone aware datetimes are stored as naive UTC values plus the zone,
    # since ``to_numpy`` would otherwise give an array of Timestamp objects.
    tz = getattr(obj, "tz", None)
    values = obj if tz is None else obj.tz_convert("UTC").tz_localize(None)
    content = {
        "values": values.to_numpy(),
        "tz": None if tz is None else _timezone_name(tz),
    }
    return _pandas_state(obj, "PandasNumpyBackedArrayNode", content, save_context)


def timestamp_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # A Timestamp as the numpy datetime64 of its value, which keeps its unit
    # and its nanoseconds, in UTC when it is aware, plus the name of its zone.
    tz = obj.tz
    value = obj if tz is None else obj.tz_convert("UTC").tz_localize(None)
    content = {
        "value": value.to_datetime64(),
        "tz": None if tz is None else _timezone_name(tz),
    }
    return _pandas_state(obj, "PandasTimestampNode", content, save_context)


def timedelta_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"value": obj.to_timedelta64()}
    return _pandas_state(obj, "PandasTimedeltaNode", content, save_context)


def na_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # ``pandas.NA`` is a singleton, so nothing but its name is stored.
    return _pandas_state(obj, "PandasNANode", {}, save_context)


def nat_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # ``pandas.NaT`` is a singleton too, which pandas recognises as missing by
    # identity: a copy built from its state, as the generic path would, is not
    # missing to ``pandas.isna`` and acts as the year 1 in arithmetic.
    return _pandas_state(obj, "PandasNaTNode", {}, save_context)


def masked_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # IntegerArray, FloatingArray and BooleanArray: a numpy array of values and
    # a boolean mask of the missing entries, which is what their constructor
    # takes. Masked positions hold arbitrary values, so they are zeroed.
    numpy_dtype = obj.dtype.numpy_dtype
    content = {
        "values": obj.to_numpy(dtype=numpy_dtype, na_value=numpy_dtype.type(0)),
        "mask": obj.isna(),
    }
    return _pandas_state(obj, "PandasMaskedArrayNode", content, save_context)


def categorical_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"codes": obj.codes, "dtype": obj.dtype}
    return _pandas_state(obj, "PandasCategoricalNode", content, save_context)


def period_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"ordinals": obj.asi8, "dtype": obj.dtype}
    return _pandas_state(obj, "PandasPeriodArrayNode", content, save_context)


def interval_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"left": obj.left, "right": obj.right, "closed": obj.closed}
    return _pandas_state(obj, "PandasIntervalArrayNode", content, save_context)


def extension_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # Any other extension array, of which only the string arrays are
    # supported: stored as their scalars, with ``None`` for missing values,
    # plus their dtype. They are named by their storage, since some pandas
    # versions have private string array classes.
    import pandas as pd

    if isinstance(obj.dtype, pd.ArrowDtype):
        raise UnsupportedTypeException(_ARROW_UNSUPPORTED)
    name = None
    if isinstance(obj.dtype, pd.StringDtype):
        storage = obj.dtype.storage
        name = "pandas.arrays." + (
            "StringArray" if storage == "python" else "ArrowStringArray"
        )
    content = {"values": obj.to_numpy(dtype=object, na_value=None), "dtype": obj.dtype}
    return _pandas_state(
        obj, "PandasExtensionArrayNode", content, save_context, name=name
    )


def sparse_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # Stored dense, as the numpy array of its subtype, rather than as scalars:
    # the array keeps its subtype, e.g. a complex one which scalars would not,
    # and the fill value is part of the dtype, which rebuilds the sparse array.
    content = {"values": obj.to_numpy(), "dtype": obj.dtype}
    return _pandas_state(obj, "PandasExtensionArrayNode", content, save_context)


def extension_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # Extension dtypes are rebuilt from their string form, e.g. "Int64" or
    # "period[M]", except those whose name does not hold all of their state,
    # or whose parser consults pandas' registry of extension dtypes or the
    # file system, which are stored as parts below.
    import pandas as pd

    if isinstance(obj, pd.ArrowDtype):
        raise UnsupportedTypeException(_ARROW_UNSUPPORTED)
    name = str(obj)
    try:
        rebuilt = type(obj).construct_from_string(name)
    except TypeError:
        rebuilt = None
    if rebuilt != obj:
        # e.g. a dtype whose name another dtype claims, or whose name leaves
        # out some of its state
        raise UnsupportedTypeException(
            f"The dtype {obj!r} cannot be rebuilt from its name {name!r}, so it"
            " cannot be saved."
        )
    content = {"name": name}
    return _pandas_state(obj, "PandasExtensionDtypeNode", content, save_context)


def string_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The name of a string dtype leaves out how it behaves: "string" is the
    # name of the dtype with pandas.NA as missing value on every version, but
    # also of the one with NaN on pandas 2.1 and 2.2, whose storage is
    # "pyarrow_numpy". The storage and the missing value are stored instead.
    na_value = obj.na_value
    content = {
        "storage": "python" if obj.storage == "python" else "pyarrow",
        "na_value": "NaN" if isinstance(na_value, float) else "NA",
    }
    return _pandas_state(obj, "PandasStringDtypeNode", content, save_context)


def _string_dtype(storage: str, nan: bool) -> Any:
    """The string dtype closest to a saved one that this pandas version has.

    The missing value, ``pandas.NA`` or ``NaN``, decides how the strings
    behave, e.g. whether comparisons return booleans or ``NA``, and is always
    kept. The storage, python or pyarrow, holds the same values, and is kept
    when pyarrow is installed.

    pandas 2.3 and later have ``NaN`` strings of both storages, 2.1 and 2.2 only
    with pyarrow, under the storage name "pyarrow_numpy", and 2.0 not at all.
    Without such a dtype, strings with ``NaN`` as missing value are object
    arrays of strings, which is how pandas keeps them without a string dtype.
    """
    import pandas as pd

    candidates = [storage, "python"] if storage == "pyarrow" else ["python"]
    for candidate in candidates:
        try:
            if not nan:
                return pd.StringDtype(candidate)
            return pd.StringDtype(candidate, na_value=np.nan)
        except ImportError:
            # pyarrow is not installed
            continue
        except TypeError:
            # pandas before 2.3, which has no ``na_value``
            break
    try:
        return pd.StringDtype("pyarrow_numpy")
    except (ImportError, ValueError):
        return np.dtype(object)


def datetime_tz_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    content = {"unit": obj.unit, "tz": _timezone_name(obj.tz)}
    return _pandas_state(obj, "PandasDatetimeTZDtypeNode", content, save_context)


def interval_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The subtype is a numpy dtype, stored by its name like the subtype of a
    # sparse dtype, a tz-aware datetime dtype, or None.
    subtype = obj.subtype
    if isinstance(subtype, np.dtype):
        subtype = str(subtype)
    content = {"subtype": subtype, "closed": obj.closed}
    return _pandas_state(obj, "PandasIntervalDtypeNode", content, save_context)


def categorical_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The string form of a categorical dtype does not include its categories.
    content = {"categories": obj.categories, "ordered": obj.ordered}
    return _pandas_state(obj, "PandasCategoricalDtypeNode", content, save_context)


def _sparse_fill_value(dtype: Any) -> Any:
    """The fill value of a sparse dtype in a form that can be stored.

    JSON scalars and numpy scalars are stored as they are. Any other scalar is
    converted to the numpy scalar of the subtype, e.g. a complex number, a
    Timestamp or a Timedelta, which JSON cannot hold.
    """
    import pandas as pd

    fill_value = dtype.fill_value
    if fill_value is None or isinstance(
        fill_value, (str, bool, int, float, np.generic)
    ):
        return fill_value
    kind = dtype.subtype.kind
    if kind in "biufcMm" and not pd.isna(fill_value):
        try:
            if kind == "M":
                # keeps the nanoseconds, which np.datetime64(datetime) drops
                return pd.Timestamp(fill_value).to_datetime64()
            if kind == "m":
                return pd.Timedelta(fill_value).to_timedelta64()
            return dtype.subtype.type(fill_value)
        except (TypeError, ValueError):  # pragma: no cover
            pass
    raise UnsupportedTypeException(
        f"The fill value {fill_value!r} of the sparse dtype {dtype} cannot be saved."
    )


def sparse_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The string form of a sparse dtype can only be parsed back when the fill
    # value is the default one for its subtype, so the parts are stored.
    content = {"subtype": str(obj.subtype), "fill_value": _sparse_fill_value(obj)}
    return _pandas_state(obj, "PandasSparseDtypeNode", content, save_context)


class _PandasNode(Node):
    """Base class of the pandas nodes.

    The children are the entries of ``state["content"]``. ``_allowed_types``
    of each subclass names every entry and the node types it may hold, which
    is checked while the file is read, and ``_build`` builds the object from
    the constructed children.

    The audit trusts the class a file declares, so the declared class has to
    be one the loader builds: ``_module`` and ``_classes`` of each subclass
    name them by their public names, see :func:`_public_name`, the declaration
    is checked while the file is read, and the built object is checked to be
    of the declared class, see ``_built_name``. The dump functions only write
    these names, see :func:`_pandas_state`.
    """

    _module: str = "pandas"
    _classes: tuple[str, ...] = ()

    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        trusted = _public_trusted(trusted)
        self.trusted = self._get_trusted(trusted, [])
        declared_ok = (
            self.module_name == self._module and self.class_name in self._classes
        )
        if not declared_ok:
            raise ValueError(
                f"{self.module_name}.{self.class_name} is not a class"
                f" {type(self).__name__} builds. This is probably due to a"
                " corrupted or a malicious file."
            )
        allowed_types = self._allowed_types()
        if set(state["content"]) != set(allowed_types):
            raise ValueError(
                f"Expected the entries {sorted(allowed_types)}, got"
                f" {sorted(state['content'])}. This is probably due to a corrupted"
                " or a malicious file."
            )
        self.content = {
            key: get_tree(
                value, load_context, trusted=trusted, allowed_types=allowed_types[key]
            )
            for key, value in state["content"].items()
        }
        self.children = dict(self.content)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        # The node types each entry may hold, ``None`` for any: names for
        # instance can be any hashable.
        raise NotImplementedError  # pragma: no cover

    def _construct_content(self) -> dict[str, Any]:
        return {key: node.construct() for key, node in self.content.items()}

    def _build(self):
        raise NotImplementedError  # pragma: no cover

    def _built_name(self, obj: Any) -> str | None:
        # The public name of the built object, which has to be the declared
        # one. Loaders whose class follows the environment override this.
        return _public_name(type(obj))

    def _construct(self):
        obj = self._build()
        declared = f"{self.module_name}.{self.class_name}"
        built = self._built_name(obj)
        if built != declared:
            raise ValueError(
                f"The file declares {declared}, but its data builds"
                f" {built or get_module(type(obj)) + '.' + type(obj).__name__}."
                " This is probably due to a corrupted or a malicious file."
            )
        return obj


class PandasIndexNode(_PandasNode):
    # ``pd.Index`` returns the subclass for the dtype of the values
    _classes = (
        "Index",
        "DatetimeIndex",
        "TimedeltaIndex",
        "PeriodIndex",
        "CategoricalIndex",
        "IntervalIndex",
    )

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"values": _ARRAY_NODES, "name": None}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        values = content["values"]
        # The dtype is passed to prevent inference, and ``tupleize_cols`` keeps
        # an object Index of tuples from becoming a MultiIndex.
        return pd.Index(
            values, dtype=values.dtype, name=content["name"], tupleize_cols=False
        )


class PandasRangeIndexNode(_PandasNode):
    _classes = ("RangeIndex",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {
            "start": (JsonNode,),
            "stop": (JsonNode,),
            "step": (JsonNode,),
            "name": None,
        }

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.RangeIndex(
            content["start"], content["stop"], content["step"], name=content["name"]
        )


class PandasMultiIndexNode(_PandasNode):
    _classes = ("MultiIndex",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {
            "levels": (ListNode,),
            "codes": (ListNode,),
            "sortorder": (JsonNode,),
            "names": (ListNode,),
        }

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.MultiIndex(
            levels=content["levels"],
            codes=content["codes"],
            sortorder=content["sortorder"],
            names=content["names"],
        )


class PandasSeriesNode(_PandasNode):
    _classes = ("Series",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"values": _ARRAY_NODES, "index": _INDEX_NODES, "name": None}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        values = content["values"]
        return pd.Series(
            values, index=content["index"], dtype=values.dtype, name=content["name"]
        )


class PandasDataFrameNode(_PandasNode):
    _classes = ("DataFrame",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"columns": _INDEX_NODES, "index": _INDEX_NODES, "data": (ListNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        columns = [pd.Series(values, dtype=values.dtype) for values in content["data"]]
        if columns:
            frame = pd.concat(columns, axis=1, ignore_index=True)
            frame.index = content["index"]
        else:
            frame = pd.DataFrame(index=content["index"])
        frame.columns = content["columns"]
        return frame


class PandasNumpyBackedArrayNode(_PandasNode):
    _module = "pandas.arrays"
    _classes = ("NumpyExtensionArray", "DatetimeArray", "TimedeltaArray")

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"values": (NdArrayNode,), "tz": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        values, tz = content["values"], content["tz"]
        if self.class_name == "NumpyExtensionArray":
            # The plain wrapper of a numpy array of any dtype, datetime64 and
            # timedelta64 included, which pandas 2.0 has under its old name.
            if tz is not None:
                raise ValueError(
                    "A NumpyExtensionArray has no time zone. This is probably due"
                    " to a corrupted or a malicious file."
                )
            cls = getattr(pd.arrays, "NumpyExtensionArray", None)
            if cls is None:
                cls = pd.arrays.PandasArray
            return cls(values)
        # A DatetimeArray or TimedeltaArray, which ``pd.array`` infers from the
        # dtype of the values on every pandas version.
        kind = "M" if self.class_name == "DatetimeArray" else "m"
        if values.dtype.kind != kind:
            raise ValueError(
                f"The values of a {self.class_name} must have a {np.dtype(kind)}"
                " dtype. This is probably due to a corrupted or a malicious file."
            )
        if tz is not None and kind == "m":
            raise ValueError(
                "A TimedeltaArray has no time zone. This is probably due to a"
                " corrupted or a malicious file."
            )
        array = pd.array(values)
        if tz is not None:
            array = array.tz_localize("UTC").tz_convert(_timezone(tz))
        return array


class PandasTimestampNode(_PandasNode):
    _classes = ("Timestamp",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"value": (NdArrayNode,), "tz": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        stamp = pd.Timestamp(content["value"])
        if content["tz"] is not None:
            stamp = stamp.tz_localize("UTC").tz_convert(_timezone(content["tz"]))
        return stamp


class PandasTimedeltaNode(_PandasNode):
    _classes = ("Timedelta",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"value": (NdArrayNode,)}

    def _build(self):
        import pandas as pd

        return pd.Timedelta(self._construct_content()["value"])


class PandasNANode(_PandasNode):
    # named by pandas.api.typing on every version, see _public_name
    _module = "pandas.api.typing"
    _classes = ("NAType",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {}

    def _build(self):
        import pandas as pd

        return pd.NA


class PandasNaTNode(_PandasNode):
    # named by pandas.api.typing on every version, see _public_name
    _module = "pandas.api.typing"
    _classes = ("NaTType",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {}

    def _build(self):
        import pandas as pd

        return pd.NaT


class PandasMaskedArrayNode(_PandasNode):
    _module = "pandas.arrays"
    _classes = ("IntegerArray", "FloatingArray", "BooleanArray")

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"values": (NdArrayNode,), "mask": (NdArrayNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        cls = getattr(pd.arrays, self.class_name)
        return cls(content["values"], content["mask"])


class PandasCategoricalNode(_PandasNode):
    _module = "pandas.arrays"
    _classes = ("Categorical",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"codes": (NdArrayNode,), "dtype": (PandasCategoricalDtypeNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.Categorical.from_codes(content["codes"], dtype=content["dtype"])


class PandasPeriodArrayNode(_PandasNode):
    _module = "pandas.arrays"
    _classes = ("PeriodArray",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"ordinals": (NdArrayNode,), "dtype": (PandasExtensionDtypeNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.arrays.PeriodArray(content["ordinals"], dtype=content["dtype"])


class PandasIntervalArrayNode(_PandasNode):
    _module = "pandas.arrays"
    _classes = ("IntervalArray",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"left": _INDEX_NODES, "right": _INDEX_NODES, "closed": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.arrays.IntervalArray.from_arrays(
            content["left"], content["right"], closed=content["closed"]
        )


_STRING_ARRAYS = ("StringArray", "ArrowStringArray")


class PandasExtensionArrayNode(_PandasNode):
    # The string arrays, and the sparse array whose values are its dense
    # numpy array. The array is built by ``pd.array`` from its dtype, whose
    # node is audited itself.
    _module = "pandas.arrays"
    _classes = (*_STRING_ARRAYS, "SparseArray")

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"values": (NdArrayNode,), "dtype": _DTYPE_NODES}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.array(content["values"], dtype=content["dtype"])

    def _built_name(self, obj: Any) -> str | None:
        # The class of a string array follows the string storage of the
        # environment, and some pandas versions have private subclasses of
        # the public ones, e.g. for the strings with ``NaN`` as missing value.
        # Versions without the string dtype of the file keep the strings in an
        # object array, see PandasExtensionDtypeNode.
        built = _nearest_public_name(type(obj))
        string_arrays = ["pandas.arrays." + name for name in _STRING_ARRAYS]
        if self.class_name in _STRING_ARRAYS and (
            built in string_arrays
            or (
                built == "pandas.arrays.NumpyExtensionArray"
                and obj.to_numpy().dtype == object
            )
        ):
            return f"{self.module_name}.{self.class_name}"
        return built


class PandasExtensionDtypeNode(_PandasNode):
    # The dtypes that are rebuilt from their string form.
    _classes = (
        "Int8Dtype",
        "Int16Dtype",
        "Int32Dtype",
        "Int64Dtype",
        "UInt8Dtype",
        "UInt16Dtype",
        "UInt32Dtype",
        "UInt64Dtype",
        "Float32Dtype",
        "Float64Dtype",
        "BooleanDtype",
        "PeriodDtype",
    )

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"name": (JsonNode,)}

    def _build(self):
        import pandas as pd

        name = self._construct_content()["name"]
        # The declared, trusted dtype class parses the name itself.
        # ``pandas.api.types.pandas_dtype`` would look the name up in pandas'
        # registry of extension dtypes instead, where any imported library can
        # register one, and run that library's code for a name from the file.
        # The parsers of the dtypes above compare the name with their own or
        # hand it to the offset parser. The dtypes whose parsers consult the
        # registry (the subtype of an interval or sparse dtype) or the file
        # system (the time zone of a datetime dtype) have loaders of their own.
        return getattr(pd, self.class_name).construct_from_string(name)


class PandasStringDtypeNode(_PandasNode):
    _classes = ("StringDtype",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"storage": (JsonNode,), "na_value": (JsonNode,)}

    def _build(self):
        content = self._construct_content()
        storage, na_value = content["storage"], content["na_value"]
        if storage not in ("python", "pyarrow") or na_value not in ("NA", "NaN"):
            raise ValueError(
                f"Unknown string dtype with storage {storage!r} and missing value"
                f" {na_value!r}. This is probably due to a corrupted or a malicious"
                " file."
            )
        return _string_dtype(storage, nan=na_value == "NaN")

    def _built_name(self, obj: Any) -> str | None:
        if obj == np.dtype(object):
            # strings with NaN as missing value where pandas has no such dtype
            return f"{self.module_name}.{self.class_name}"
        return super()._built_name(obj)


class PandasCategoricalDtypeNode(_PandasNode):
    _classes = ("CategoricalDtype",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"categories": _INDEX_NODES + (JsonNode,), "ordered": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.CategoricalDtype(content["categories"], ordered=content["ordered"])


class PandasSparseDtypeNode(_PandasNode):
    _classes = ("SparseDtype",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"subtype": (JsonNode,), "fill_value": (JsonNode, NdArrayNode)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        # numpy parses the subtype, so that the name from the file is not
        # looked up in pandas' registry of extension dtypes
        return pd.SparseDtype(np.dtype(content["subtype"]), content["fill_value"])


class PandasDatetimeTZDtypeNode(_PandasNode):
    _classes = ("DatetimeTZDtype",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"unit": (JsonNode,), "tz": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        return pd.DatetimeTZDtype(unit=content["unit"], tz=_timezone(content["tz"]))


class PandasIntervalDtypeNode(_PandasNode):
    _classes = ("IntervalDtype",)

    def _allowed_types(self) -> dict[str, tuple[type[Node], ...] | None]:
        return {"subtype": (JsonNode, PandasDatetimeTZDtypeNode), "closed": (JsonNode,)}

    def _build(self):
        import pandas as pd

        content = self._construct_content()
        subtype = content["subtype"]
        # A name is parsed by numpy, so that it is not looked up in pandas'
        # registry of extension dtypes; otherwise the subtype is a tz-aware
        # datetime dtype, built by its own node, or None.
        message = (
            "The subtype of an interval dtype must be a numpy dtype, a datetime"
            " dtype or None. This is probably due to a corrupted or a malicious"
            " file."
        )
        if isinstance(subtype, str):
            try:
                subtype = np.dtype(subtype)
            except TypeError as err:
                raise ValueError(message) from err
        elif subtype is not None and not isinstance(subtype, pd.DatetimeTZDtype):
            raise ValueError(message)
        return pd.IntervalDtype(subtype, closed=content["closed"])


# The node types the values, the index and the dtype of a pandas object may be
# stored as.
_ARRAY_NODES = (
    NdArrayNode,
    PandasNumpyBackedArrayNode,
    PandasMaskedArrayNode,
    PandasCategoricalNode,
    PandasPeriodArrayNode,
    PandasIntervalArrayNode,
    PandasExtensionArrayNode,
)
_INDEX_NODES = (PandasIndexNode, PandasRangeIndexNode, PandasMultiIndexNode)
_DTYPE_NODES = (
    PandasExtensionDtypeNode,
    PandasStringDtypeNode,
    PandasCategoricalDtypeNode,
    PandasSparseDtypeNode,
    PandasDatetimeTZDtypeNode,
    PandasIntervalDtypeNode,
)


_registered = False


def register_if_imported() -> None:
    """Register the ``get_state`` handlers of pandas types, if pandas is imported.

    This is called before every dump. An object can only contain pandas
    objects if pandas has been imported, so checking ``sys.modules`` is
    enough, and skops itself never imports pandas.
    """
    global _registered
    if _registered or "pandas" not in sys.modules:
        return

    import pandas as pd

    dispatch_functions = [
        (pd.Index, index_get_state),
        (pd.RangeIndex, range_index_get_state),
        (pd.MultiIndex, multi_index_get_state),
        (pd.Series, series_get_state),
        (pd.DataFrame, dataframe_get_state),
        (pd.api.extensions.ExtensionArray, extension_array_get_state),
        (pd.arrays.DatetimeArray, numpy_backed_array_get_state),
        (pd.arrays.TimedeltaArray, numpy_backed_array_get_state),
        (pd.Timestamp, timestamp_get_state),
        (pd.Timedelta, timedelta_get_state),
        (type(pd.NA), na_get_state),
        (type(pd.NaT), nat_get_state),
        (pd.arrays.IntegerArray, masked_array_get_state),
        (pd.arrays.FloatingArray, masked_array_get_state),
        (pd.arrays.BooleanArray, masked_array_get_state),
        (pd.Categorical, categorical_get_state),
        (pd.arrays.PeriodArray, period_array_get_state),
        (pd.arrays.IntervalArray, interval_array_get_state),
        (pd.arrays.SparseArray, sparse_array_get_state),
        (pd.api.extensions.ExtensionDtype, extension_dtype_get_state),
        (pd.StringDtype, string_dtype_get_state),
        (pd.CategoricalDtype, categorical_dtype_get_state),
        (pd.SparseDtype, sparse_dtype_get_state),
        (pd.DatetimeTZDtype, datetime_tz_dtype_get_state),
        (pd.IntervalDtype, interval_dtype_get_state),
    ]
    # pandas.arrays.PandasArray was renamed to NumpyExtensionArray in pandas 2.1
    numpy_backed = getattr(pd.arrays, "NumpyExtensionArray", None)
    if numpy_backed is None:
        numpy_backed = pd.arrays.PandasArray
    dispatch_functions.append((numpy_backed, numpy_backed_array_get_state))
    # StringArray subclasses NumpyExtensionArray, but its numpy form is an
    # object array which loses the string dtype, so it takes the generic path.
    dispatch_functions.append((pd.arrays.StringArray, extension_array_get_state))

    for cls, func in dispatch_functions:
        _get_state.register(cls)(func)
    _registered = True


NODE_TYPE_MAPPING = {
    ("PandasIndexNode", PROTOCOL): PandasIndexNode,
    ("PandasRangeIndexNode", PROTOCOL): PandasRangeIndexNode,
    ("PandasMultiIndexNode", PROTOCOL): PandasMultiIndexNode,
    ("PandasSeriesNode", PROTOCOL): PandasSeriesNode,
    ("PandasDataFrameNode", PROTOCOL): PandasDataFrameNode,
    ("PandasNumpyBackedArrayNode", PROTOCOL): PandasNumpyBackedArrayNode,
    ("PandasTimestampNode", PROTOCOL): PandasTimestampNode,
    ("PandasTimedeltaNode", PROTOCOL): PandasTimedeltaNode,
    ("PandasNANode", PROTOCOL): PandasNANode,
    ("PandasNaTNode", PROTOCOL): PandasNaTNode,
    ("PandasMaskedArrayNode", PROTOCOL): PandasMaskedArrayNode,
    ("PandasCategoricalNode", PROTOCOL): PandasCategoricalNode,
    ("PandasPeriodArrayNode", PROTOCOL): PandasPeriodArrayNode,
    ("PandasIntervalArrayNode", PROTOCOL): PandasIntervalArrayNode,
    ("PandasExtensionArrayNode", PROTOCOL): PandasExtensionArrayNode,
    ("PandasExtensionDtypeNode", PROTOCOL): PandasExtensionDtypeNode,
    ("PandasStringDtypeNode", PROTOCOL): PandasStringDtypeNode,
    ("PandasCategoricalDtypeNode", PROTOCOL): PandasCategoricalDtypeNode,
    ("PandasSparseDtypeNode", PROTOCOL): PandasSparseDtypeNode,
    ("PandasDatetimeTZDtypeNode", PROTOCOL): PandasDatetimeTZDtypeNode,
    ("PandasIntervalDtypeNode", PROTOCOL): PandasIntervalDtypeNode,
}
