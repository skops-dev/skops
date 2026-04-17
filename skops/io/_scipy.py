from __future__ import annotations

import io
from typing import Any, Optional, Sequence

from scipy.sparse import load_npz, save_npz, spmatrix

from ._audit import Node
from ._protocol import PROTOCOL
from ._utils import LoadContext, SaveContext, get_module


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
        trusted: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.type = state["type"]
        self.trusted = self._get_trusted(trusted, [spmatrix])
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
# tuples of type and function that creates the instance of that type
NODE_TYPE_MAPPING = {
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    ("SparseMatrixNode", PROTOCOL): SparseMatrixNode,
}
