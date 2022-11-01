from __future__ import annotations

import io
from typing import Any

from scipy.sparse import load_npz, save_npz, spmatrix

from ._audit import Node
from ._utils import SaveState, get_module


def sparse_matrix_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "sparse_matrix_get_instance",
    }

    data_buffer = io.BytesIO()
    save_npz(data_buffer, obj)
    # Memoize the object and then check if it's file name (containing
    # the object id) already exists. If it does, there is no need to
    # save the object again. Memoizitation is necessary since for
    # ephemeral objects, the same id might otherwise be reused.
    obj_id = save_state.memoize(obj)
    f_name = f"{obj_id}.npz"
    if f_name not in save_state.zip_file.namelist():
        save_state.zip_file.writestr(f_name, data_buffer.getbuffer())

    res["type"] = "scipy"
    res["file"] = f_name
    return res


class SparseMatrixNode(Node):
    def __init__(self, state, src, trusted=None):
        super().__init__(state, src, trusted)
        self.type = state["type"]
        if self.type != "scipy":
            raise TypeError(
                f"Cannot load object of type {self.module_name}.{self.class_name}"
            )

        self.content = io.BytesIO(src.read(state["file"]))

    def construct(self):
        # scipy load_npz uses numpy.save with allow_pickle=False under the
        # hood, so we're safe using it
        return load_npz(self.content)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    (spmatrix, sparse_matrix_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_MAPPING = {
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    "SparseMatrixNode": SparseMatrixNode,
}
