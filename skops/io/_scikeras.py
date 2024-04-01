import os
import tempfile
from typing import Type

from scikeras.wrappers import KerasClassifier, KerasRegressor
from torch import Node

from ._utils import Any, SaveContext, get_module


def scikeras_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "ScikerasNode",
    }

    obj_id = save_context.memoize(obj)
    f_name = f"{obj_id}.keras"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = os.path.join(temp_dir, "model.keras")
        obj.model.save(file_name)
        save_context.zip_file.write(file_name, f_name)

    res.update(type="scikeras", file=f_name)
    return res


GET_STATE_DISPATCH_FUNCTIONS = [
    (KerasClassifier, scikeras_get_state),
    (KerasRegressor, scikeras_get_state),
]

NODE_TYPE_MAPPING: dict[tuple[str, int], Type[Node]] = {}
