from __future__ import annotations

import io
from typing import Any

from scipy.sparse import load_npz, save_npz, spmatrix

from skops.utils._fixes import get_scipy_ufunc_wrapper_type, get_sparray_type

from ._audit import Node
from ._general import function_get_state
from ._protocol import PROTOCOL
from ._utils import LoadContext, SaveContext, get_module

_SPARRAY = get_sparray_type()
_SCIPY_UFUNC_WRAPPER = get_scipy_ufunc_wrapper_type()


def sparse_matrix_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "SparseMatrixNode",
    }

    data_buffer = io.BytesIO()
    save_npz(data_buffer, obj)
    # Memoize the object and then check if it's file name (containing
    # the object id) already exists. If it does, there is no need to
    # save the object again. Memoizitation is necessary since for
    # ephemeral objects, the same id might otherwise be reused.
    obj_id = save_context.memoize(obj)
    f_name = f"{obj_id}.npz"
    if f_name not in save_context.zip_file.namelist():
        save_context.zip_file.writestr(f_name, data_buffer.getbuffer())

    res["type"] = "scipy"
    res["file"] = f_name
    return res


class SparseMatrixNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: list[str] | None = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.type = state["type"]
        sparse_types: list[Any] = [spmatrix]
        if _SPARRAY is not None:
            sparse_types.append(_SPARRAY)
        self.trusted = self._get_trusted(trusted, sparse_types)
        if self.type != "scipy":
            raise TypeError(
                f"Cannot load object of type {self.module_name}.{self.class_name}"
            )

        self.children = {"content": io.BytesIO(load_context.src.read(state["file"]))}

    def _construct(self):
        # scipy load_npz uses numpy.save with allow_pickle=False under the
        # hood, so we're safe using it
        return load_npz(self.children["content"])


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    (spmatrix, sparse_matrix_get_state),
]
if _SPARRAY is not None:
    # scipy sparse *arrays* (e.g. csr_array) are the modern, NumPy-compatible
    # replacement for sparse matrices; they round-trip through the same npz node.
    GET_STATE_DISPATCH_FUNCTIONS.append((_SPARRAY, sparse_matrix_get_state))
if _SCIPY_UFUNC_WRAPPER is not None:
    # as of scipy 2.0, some scipy.special ufuncs are wrapper objects that are no
    # longer numpy.ufunc instances; persist them like any other ufunc/function.
    GET_STATE_DISPATCH_FUNCTIONS.append((_SCIPY_UFUNC_WRAPPER, function_get_state))
# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    ("SparseMatrixNode", PROTOCOL): SparseMatrixNode,
}
