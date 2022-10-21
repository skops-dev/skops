#!/usr/bin/env python3

from __future__ import annotations

import json
from typing import Any, Callable
from zipfile import ZipFile

GET_INSTANCE_MAPPING: dict[str, Callable[[dict[str, Any], ZipFile], Any]] = {}


def get_instance(state, src):
    """Create instance based on the state, using json if possible"""
    if state.get("is_json"):
        return json.loads(state["content"])

    try:
        get_instance_func = GET_INSTANCE_MAPPING[state["__loader__"]]
    except KeyError:
        raise TypeError(
            f"Creating an instance of type {type(state)} is not supported yet"
        )
    return get_instance_func(state, src)
