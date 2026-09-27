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
  dtype, and extension dtypes as their string representation.

Values with a numpy dtype are stored as numpy arrays, everything else as lists
of scalars.

pandas is optional and slow to import, so ``skops.io`` does not import it. The
``get_state`` handlers are registered on the first dump after the user has
imported pandas, see :func:`register_if_imported`, and the nodes only import
pandas when they construct an object.

Not preserved: the ``freq`` of datetime-like indexes and arrays, and the
``attrs`` and ``flags`` of a Series or DataFrame.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np

from ._audit import Node, get_tree
from ._protocol import PROTOCOL
from ._trusted_types import PANDAS_TYPE_NAMES
from ._utils import (
    LoadContext,
    SaveContext,
    TrustedTypes,
    _get_state,
    get_module,
    get_state,
    gettype,
)
from .exceptions import UnsupportedTypeException


def _public_module(cls: type) -> str:
    # The public module of a pandas class: "pandas.arrays" for the array
    # classes and "pandas" for everything else. pandas 3 reports these as
    # ``__module__`` already, but older versions report the defining module,
    # e.g. ``pandas.core.series``, and a file must not depend on that: the
    # public name is what the default trusted list knows, and what stays
    # importable across versions. Deprecated aliases such as
    # ``pandas.arrays.PandasArray`` warn when accessed, hence the suppression.
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in (pd.arrays, pd):
            if getattr(module, cls.__name__, None) is cls:
                return module.__name__
    return get_module(cls)


# Classes that later pandas versions renamed, mapped to their current name, so
# that a file never names a class the loading version may not have.
_RENAMED_CLASSES = {
    # renamed in pandas 2.1
    "PandasArray": "NumpyExtensionArray",
}


def _pandas_state(
    obj: Any, loader: str, content: dict[str, Any], save_context: SaveContext
) -> dict[str, Any]:
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

    return {
        "__class__": _RENAMED_CLASSES.get(cls.__name__, cls.__name__),
        "__module__": _public_module(cls),
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


def numpy_backed_array_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # NumpyExtensionArray, DatetimeArray and TimedeltaArray wrap a numpy array.
    # Time zone aware datetimes are stored as naive UTC values plus the zone,
    # since ``to_numpy`` would otherwise give an array of Timestamp objects.
    tz = getattr(obj, "tz", None)
    values = obj if tz is None else obj.tz_convert("UTC").tz_localize(None)
    content = {"values": values.to_numpy(), "tz": None if tz is None else str(tz)}
    return _pandas_state(obj, "PandasNumpyBackedArrayNode", content, save_context)


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
    # Any other extension array, e.g. string, sparse or pyarrow backed arrays:
    # stored as its scalars, with ``None`` for missing values, plus its dtype.
    content = {"values": obj.to_numpy(dtype=object, na_value=None), "dtype": obj.dtype}
    return _pandas_state(obj, "PandasExtensionArrayNode", content, save_context)


def extension_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # Extension dtypes are rebuilt from their string form, e.g. "Int64",
    # "datetime64[ns, UTC]" or "period[M]".
    content = {"name": str(obj)}
    return _pandas_state(obj, "PandasExtensionDtypeNode", content, save_context)


def categorical_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The string form of a categorical dtype does not include its categories.
    content = {"categories": obj.categories, "ordered": obj.ordered}
    return _pandas_state(obj, "PandasCategoricalDtypeNode", content, save_context)


def sparse_dtype_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    # The string form of a sparse dtype can only be parsed back when the fill
    # value is the default one for its subtype.
    content = {"subtype": str(obj.subtype), "fill_value": obj.fill_value}
    return _pandas_state(obj, "PandasSparseDtypeNode", content, save_context)


class _PandasNode(Node):
    """Base class of the pandas nodes.

    The children are the entries of ``state["content"]``, and ``_construct``
    of each subclass builds the object from the constructed children.
    """

    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: TrustedTypes | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, PANDAS_TYPE_NAMES)
        self.content = {
            key: get_tree(value, load_context, trusted=trusted)
            for key, value in state["content"].items()
        }
        self.children = dict(self.content)

    def _construct_content(self) -> dict[str, Any]:
        return {key: node.construct() for key, node in self.content.items()}


class PandasIndexNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        values = content["values"]
        # The dtype is passed to prevent inference, and ``tupleize_cols`` keeps
        # an object Index of tuples from becoming a MultiIndex.
        return pd.Index(
            values, dtype=values.dtype, name=content["name"], tupleize_cols=False
        )


class PandasRangeIndexNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.RangeIndex(
            content["start"], content["stop"], content["step"], name=content["name"]
        )


class PandasMultiIndexNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.MultiIndex(
            levels=content["levels"], codes=content["codes"], names=content["names"]
        )


class PandasSeriesNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        values = content["values"]
        return pd.Series(
            values, index=content["index"], dtype=values.dtype, name=content["name"]
        )


class PandasDataFrameNode(_PandasNode):
    def _construct(self):
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
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        values = content["values"]
        # A numpy dtype makes ``pd.array`` return a NumpyExtensionArray, which
        # is what is wanted for all but datetime64 and timedelta64 arrays: for
        # those pandas 2.0 also returns one when the dtype is given with a unit
        # other than nanoseconds, while without a dtype every version infers
        # a DatetimeArray or TimedeltaArray.
        dtype = None if values.dtype.kind in "Mm" else values.dtype
        array = pd.array(values, dtype=dtype)
        if content["tz"] is not None:
            array = array.tz_localize("UTC").tz_convert(content["tz"])
        return array


class PandasMaskedArrayNode(_PandasNode):
    def _construct(self):
        content = self._construct_content()
        cls = gettype(self.module_name, self.class_name)
        return cls(content["values"], content["mask"])


class PandasCategoricalNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.Categorical.from_codes(content["codes"], dtype=content["dtype"])


class PandasPeriodArrayNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.arrays.PeriodArray(content["ordinals"], dtype=content["dtype"])


class PandasIntervalArrayNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.arrays.IntervalArray.from_arrays(
            content["left"], content["right"], closed=content["closed"]
        )


class PandasExtensionArrayNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.array(content["values"], dtype=content["dtype"])


class PandasExtensionDtypeNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        name = self._construct_content()["name"]
        dtype = pd.api.types.pandas_dtype(name)
        if name == "str" and not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            # "str" is the default string dtype of pandas 3. Older versions
            # parse it as a numpy unicode dtype, and keep strings in object
            # arrays instead, which is what the values are stored as.
            return np.dtype(object)
        return dtype


class PandasCategoricalDtypeNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.CategoricalDtype(content["categories"], ordered=content["ordered"])


class PandasSparseDtypeNode(_PandasNode):
    def _construct(self):
        import pandas as pd

        content = self._construct_content()
        return pd.SparseDtype(content["subtype"], content["fill_value"])


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
        (pd.arrays.IntegerArray, masked_array_get_state),
        (pd.arrays.FloatingArray, masked_array_get_state),
        (pd.arrays.BooleanArray, masked_array_get_state),
        (pd.Categorical, categorical_get_state),
        (pd.arrays.PeriodArray, period_array_get_state),
        (pd.arrays.IntervalArray, interval_array_get_state),
        (pd.api.extensions.ExtensionDtype, extension_dtype_get_state),
        (pd.CategoricalDtype, categorical_dtype_get_state),
        (pd.SparseDtype, sparse_dtype_get_state),
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
    ("PandasMaskedArrayNode", PROTOCOL): PandasMaskedArrayNode,
    ("PandasCategoricalNode", PROTOCOL): PandasCategoricalNode,
    ("PandasPeriodArrayNode", PROTOCOL): PandasPeriodArrayNode,
    ("PandasIntervalArrayNode", PROTOCOL): PandasIntervalArrayNode,
    ("PandasExtensionArrayNode", PROTOCOL): PandasExtensionArrayNode,
    ("PandasExtensionDtypeNode", PROTOCOL): PandasExtensionDtypeNode,
    ("PandasCategoricalDtypeNode", PROTOCOL): PandasCategoricalDtypeNode,
    ("PandasSparseDtypeNode", PROTOCOL): PandasSparseDtypeNode,
}
