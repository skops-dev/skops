import io
import os
import tempfile
from typing import Sequence, Type

import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor

from ._audit import Node
from ._protocol import PROTOCOL
from ._utils import Any, LoadContext, SaveContext, get_module


def scikeras_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "SciKerasNode",
    }

    obj_id = save_context.memoize(obj)
    f_name = f"{obj_id}.keras"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = os.path.join(temp_dir, "model.keras")
        obj.model.save(file_name)
        save_context.zip_file.write(file_name, f_name)

    res.update(type="scikeras", file=f_name)
    return res


class SciKerasNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: bool | Sequence[str] = False,
    ) -> None:
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, [KerasClassifier, KerasRegressor])

        self.children = {"content": io.BytesIO(load_context.src.read(state["file"]))}

    def _construct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "model.keras")
            with open(file_path, "wb") as f:
                f.write(self.children["content"].getbuffer())
            model = tf.keras.models.load_model(file_path, compile=False)
        return model


GET_STATE_DISPATCH_FUNCTIONS = [
    (KerasClassifier, scikeras_get_state),
    (KerasRegressor, scikeras_get_state),
]

NODE_TYPE_MAPPING: dict[tuple[str, int], Type[Node]] = {
    ("SciKerasNode", PROTOCOL): SciKerasNode
}
