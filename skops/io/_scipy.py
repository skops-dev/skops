from __future__ import annotations

import io
from typing import Any

from scipy.sparse import load_npz, save_npz, spmatrix

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


def sparse_matrix_get_instance(state, src):
    if state["type"] != "scipy":
        raise TypeError(
            f"Cannot load object of type {state['__module__']}.{state['__class__']}"
        )

    # scipy load_npz uses numpy.save with allow_pickle=False under the hood, so
    # we're safe using it
    val = load_npz(io.BytesIO(src.read(state["file"])))
    return val


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
    "sparse_matrix_get_instance": sparse_matrix_get_instance,
}
