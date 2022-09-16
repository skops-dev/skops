import io
from pathlib import Path
from uuid import uuid4

from scipy.sparse import load_npz, save_npz, spmatrix

from ._utils import get_module


def sparse_matrix_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    f_name = f"{uuid4()}.npz"
    save_npz(Path(dst) / f_name, obj)
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
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    # use 'spmatrix' to check if a matrix is a sparse matrix because that is
    # what scipy.sparse.issparse checks
    (spmatrix, sparse_matrix_get_instance),
]
