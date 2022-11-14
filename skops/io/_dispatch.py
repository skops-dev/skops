from __future__ import annotations

import json

from ._audit import check_type
from ._trusted_types import PRIMITIVES_TYPES

NODE_TYPE_MAPPING = {}  # type: ignore


class Node:
    def __init__(self, state, src, trusted=False):
        self.class_name, self.module_name = state["__class__"], state["__module__"]
        self.trusted = trusted
        self._is_safe = None

    @classmethod
    def _get_trusted(cls, trusted, default):
        """Return a trusted list, or True.

        If `trusted` is `False`, we return the `defaults`, otherwise the
        `trusted` value is used.
        """
        if trusted is True:
            # if trusted is True, we trust the node
            return True
        elif trusted is False:
            # if trusted is False, we only trust the defaults
            return default
        # otherwise we trust the given list
        return trusted

    def _get_iterable_safety(self, values):
        """Check if members of an iterable are all safe."""
        for item in values:
            # primitive types are always trusted
            if type(item) in PRIMITIVES_TYPES:
                continue

            if not item.is_safe:
                return False
        return True

    @property
    def is_self_safe(self):
        """Check if the node's self value is safe.

        A node is considered safe if it and its children are safe.
        """
        return check_type(self.module_name, self.class_name, self.trusted)

    @property
    def is_safe(self):
        """Check if the node and all its children are safe."""
        if self._is_safe is not None:
            return self._is_safe

        if self.trusted is True:
            self._is_safe = True
            return True

        is_safe = self.is_self_safe

        for child, _type in self.children.items():
            if _type is list:
                is_safe = is_safe and self._get_iterable_safety(getattr(self, child))
            elif _type is dict:
                is_safe = is_safe and self._get_iterable_safety(
                    getattr(self, child).values()
                )
            elif _type is Node:
                is_safe = is_safe and getattr(self, child).is_safe
            else:
                raise ValueError(f"Unknown type {_type}.")

        self._is_safe = is_safe
        return is_safe

    def get_safety_tree(self, report_safe=True):
        """Get the safety tree of the node.

        Parameters
        ----------
        report_safe : bool, default=True
            If True, the safety tree will contain all nodes, even the safe ones.
            Otherwise, only unsafe nodes will be reported.

        Returns
        -------
        safety_tree : dict
            A dictionary containing the safety tree of the node.
        """
        if not report_safe and self.is_safe:
            return None

        res = dict()
        res["self"] = self.module_name + "." + self.class_name

        if report_safe or not self.is_safe:
            res["safe"] = self.is_safe

        res["children"] = {}

        for child, ch_type in self.children.items():
            if ch_type is list:
                res["children"][child] = self._get_list_safety_tree(child, report_safe)
                if not res["children"][child]:
                    del res["children"][child]
            elif ch_type is dict:
                res["children"][child] = self._get_dict_safety_tree(child, report_safe)
                if not res["children"][child]:
                    del res["children"][child]
            elif ch_type is Node:
                res["children"][child] = getattr(self, child).get_safety_tree(
                    report_safe
                )
                if not res["children"][child]:
                    del res["children"][child]
            else:
                raise ValueError(f"Unknown type {ch_type}.")

    def _get_list_safety_tree(self, child, report_safe):
        """Get the safety tree of a list."""
        res = []
        for value in getattr(self, child):
            if report_safe or not value.is_safe:
                res.append(value.get_safety_tree())
        return res

    def _get_dict_safety_tree(self, child, report_safe):
        """Get the safety tree of a dict."""
        res = {}
        for key, value in getattr(self, child):
            if report_safe or not value.is_safe:
                res[key] = value.get_safety_tree()
        return res

    def get_unsafe_set(self):
        """Get the set of unsafe types.

        This method returns all types which are not trusted.

        Returns
        -------
        unsafe_set : set
            A set of unsafe types.
        """
        res = set()
        if not self.is_self_safe:
            res.add(self.module_name + "." + self.class_name)

        for child, ch_type in self.children.items():
            if getattr(self, child) is None:
                continue
            if ch_type is list:
                for value in getattr(self, child):
                    res.update(value.get_unsafe_set())
            elif ch_type is dict:
                for value in getattr(self, child).values():
                    res.update(value.get_unsafe_set())
            elif issubclass(ch_type, Node):
                res.update(getattr(self, child).get_unsafe_set())
            else:
                raise ValueError(f"Unknown type {ch_type}.")
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
    return node_cls(state, src, trusted=False)
