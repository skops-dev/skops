from __future__ import annotations

import json

from skops.io._utils import LoadState

GET_INSTANCE_MAPPING = {}  # type: ignore


def get_instance(state, load_state: LoadState):
    """Create instance based on the state, using json if possible"""
    if state.get("is_json"):
        return json.loads(state["content"])

    saved_id = state.get("__id__")
    if saved_id and saved_id in load_state.memo:
        # an instance has already been loaded, just return the loaded instance
        return load_state.get_instance(saved_id)

    try:
        get_instance_func = GET_INSTANCE_MAPPING[state["__loader__"]]
    except KeyError:
        type_name = f"{state['__module__']}.{state['__class__']}"
        raise TypeError(
            f" Can't find loader {state['__loader__']} for type {type_name}."
        )

    loaded_obj = get_instance_func(state, load_state)

    # hold reference to obj in case same instance encountered again in save state
    if saved_id:
        load_state.memoize(loaded_obj, saved_id)

    return loaded_obj
