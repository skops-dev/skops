from __future__ import annotations

import json

from skops.io._utils import LoadContext

GET_INSTANCE_MAPPING = {}  # type: ignore


def get_instance(state, load_context: LoadContext):
    """Create instance based on the state, using json if possible"""

    saved_id = state.get("__id__")
    if saved_id in load_context.memo:
        # an instance has already been loaded, just return the loaded instance
        return load_context.get_instance(saved_id)

    if state.get("is_json"):
        loaded_obj = json.loads(state["content"])
    else:
        try:
            get_instance_func = GET_INSTANCE_MAPPING[state["__loader__"]]
        except KeyError:
            type_name = f"{state['__module__']}.{state['__class__']}"
            raise TypeError(
                f" Can't find loader {state['__loader__']} for type {type_name}."
            )

        loaded_obj = get_instance_func(state, load_context)

    # hold reference to obj in case same instance encountered again in save state
    if saved_id:
        load_context.memoize(loaded_obj, saved_id)

    return loaded_obj
