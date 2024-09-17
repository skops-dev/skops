from __future__ import annotations

import io
import os
import tempfile
from typing import Sequence, Type

from ._audit import Node
from ._protocol import PROTOCOL
from ._utils import Any, LoadContext, SaveContext, get_module

try:
    from tensorflow.keras.models import Model, Sequential, load_model, save_model

    tf_present = True
except ImportError:
    tf_present = False


def keras_get_state(obj: Any, save_context: SaveContext) -> dict[str, Any]:
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "__loader__": "KerasNode",
    }

    # Memoize the object and then check if it's file name (containing
    # the object id) already exists. If it does, there is no need to
    # save the object again. Memoizitation is necessary since for
    # ephemeral objects, the same id might otherwise be reused.
    obj_id = save_context.memoize(obj)
    f_name = f"{obj_id}.keras"
    if f_name in save_context.zip_file.namelist():
        return res

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = os.path.join(temp_dir, "model.keras")
        save_model(obj, file_name)
        save_context.zip_file.write(file_name, f_name)
        res.update(file=f_name)
    return res


class KerasNode(Node):
    def __init__(
        self,
        state: dict[str, Any],
        load_context: LoadContext,
        trusted: Sequence[str] | None = None,
    ) -> None:
        if not tf_present:
            raise ImportError(
                "`tf.keras` is missing and needs to be installed in order to load this"
                " object."
            )
        super().__init__(state, load_context, trusted)
        self.trusted = self._get_trusted(trusted, default=[])

        self.children = {"content": io.BytesIO(load_context.src.read(state["file"]))}

    def _construct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "model.keras")
            with open(file_path, "wb") as f:
                f.write(self.children["content"].getbuffer())
            model = load_model(file_path, compile=False, safe_mode=True)
        return model


if tf_present:
    GET_STATE_DISPATCH_FUNCTIONS = [
        (Model, keras_get_state),
        (Sequential, keras_get_state),
    ]

NODE_TYPE_MAPPING: dict[tuple[str, int], Type[Node]] = {
    ("KerasNode", PROTOCOL): KerasNode
}
