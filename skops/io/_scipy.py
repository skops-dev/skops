from __future__ import annotations

import base64
import io
from typing import Any

from scipy.sparse import load_npz, save_npz, spmatrix

from ._utils import SaveState, get_module


def sparse_matrix_get_state(obj: Any, save_state: SaveState) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    if save_state.path is None:
        # using skops.io.dumps, don't write to file
        f = io.BytesIO()
        save_npz(f, obj)
        b = f.getvalue()
        b64 = base64.b64encode(b)
        s = b64.decode("utf-8")
        res.update(content=s, type="base64")
        return res

    # Memoize the object and then check if it's file name (containing the object
    # id) already exists. If it does, there is no need to save the object again.
    # Memoizitation is necessary since for ephemeral objects, the same id might
    # otherwise be reused.
    obj_id = save_state.memoize(obj)
    f_name = f"{obj_id}.npz"
    path = save_state.path / f_name
    if not path.exists():
        save_npz(path, obj)

    res["type"] = "scipy"
    res["file"] = f_name
    return res


def sparse_matrix_get_instance(state, src):
    if state["type"] == "base64":
        # TODO
        b64 = state["content"].encode("utf-8")
        b = io.BytesIO(base64.b64decode(b64))
        val = load_npz(b)
        return val

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
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    (spmatrix, sparse_matrix_get_instance),
]
