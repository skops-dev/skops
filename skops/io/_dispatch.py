from __future__ import annotations

import json

GET_INSTANCE_MAPPING = {}  # type: ignore


def get_instance(state, src):
    """Create instance based on the state, using json if possible"""
    if state.get("is_json"):
        return json.loads(state["content"])

    try:
        get_instance_func = GET_INSTANCE_MAPPING[state["__loader__"]]
    except KeyError:
        type_name = f"{state['__module__']}.{state['__class__']}"
        raise TypeError(
            f" Can't find loader {state['__loader__']} for type {type_name}."
        )
    return get_instance_func(state, src)
