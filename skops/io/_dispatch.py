from __future__ import annotations

import json

from ._audit import check_type

NODE_TYPE_MAPPING = {}  # type: ignore


class Node:
    def __init__(self, state, src, trusted=None):
        self.class_name, self.module_name = state["__class__"], state["__module__"]
        self.trusted = trusted
        self._is_safe = None

    def _get_iterable_safety(self, values):
        is_safe = True
        for item in values:
            is_safe = is_safe and item.is_safe
        return is_safe

    @property
    def is_safe(self):
        if self._is_safe is not None:
            return self._is_safe

        is_safe = check_type(self.module_name, self.class_name, trusted=self.trusted)

        for child, _type in self.children.items():
            if _type is list:
                is_safe = is_safe and self._get_iterable_safety(child)
            elif _type is dict:
                is_safe = is_safe and self._get_iterable_safety(child.values())
            elif _type is Node:
                is_safe = is_safe and getattr(self, child).is_safe
            else:
                raise ValueError(f"Unknown type {_type}.")

        self._is_safe = is_safe
        return is_safe

    def get_safety_tree(self, report_safe=True):
        if not report_safe and self.is_safe:
            return None

        res = dict()
        res["self"] = self.module_name + "." + self.class_name

        if report_safe or not self.is_safe:
            res["safe"] = self.is_safe

        res["children"] = {}

        for child, type in self.children.items():
            if type is list:
                res["children"][child] = self._get_list_safety_tree(child, report_safe)
                if not res["children"][child]:
                    del res["children"][child]
            elif type is dict:
                res["children"][child] = self._get_dict_safety_tree(child, report_safe)
                if not res["children"][child]:
                    del res["children"][child]
            else:
                raise ValueError(f"Unknown type {type}.")

    def _get_list_safety_tree(self, child, report_safe):
        res = []
        for value in getattr(self, child):
            if report_safe or not value.is_safe:
                res.append(value.get_safety_tree())
        return res

    def _get_dict_safety_tree(self, child, report_safe):
        res = {}
        for key, value in getattr(self, child):
            if report_safe or not value.is_safe:
                res[key] = value.get_safety_tree()
        return res

    def get_unsafe_set(self):
        res = set()
        if not self.is_safe:
            res.add(self.module_name + "." + self.class_name)

        for child, type in self.children.items():
            if type is list:
                for value in getattr(self, child):
                    res.update(value.get_unsafe_set())
            elif type is dict:
                for value in getattr(self, child).values():
                    res.update(value.get_unsafe_set())
        return res


class JsonNode:
    def __init__(self, state):
        self.value = json.loads(state["content"])
        self.is_safe = True

    def get_unsafe_set(self):
        return set()

    def get_safety_tree(self, report_safe=True):
        return None

    def construct(self):
        return self.value


def get_tree(state, src):
    """Create instance based on the state, using json if possible"""
    if state.get("is_json"):
        return JsonNode(state)

    try:
        node_cls = NODE_TYPE_MAPPING[state["__loader__"]]
    except KeyError:
        type_name = f"{state['__module__']}.{state['__class__']}"
        raise TypeError(
            f" Can't find loader {state['__loader__']} for type {type_name}."
        )
    return node_cls(state, src)
