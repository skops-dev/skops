from __future__ import annotations

import json

from ._audit import check_type
from ._utils import LoadContext

NODE_TYPE_MAPPING = {}  # type: ignore


class Node:
    def __init__(self, state, load_context: LoadContext, trusted=False):
        self.class_name, self.module_name = state["__class__"], state["__module__"]
        self.trusted = trusted
        self._is_safe = None
        self._constructed = None

    def construct(self):
        """Construct the object.

        We only construct the object once, and then cache the result.
        """
        if self._constructed is not None:
            return self._constructed
        self._constructed = self._construct()
        return self._constructed

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


class JsonNode(Node):
    def __init__(self, state):
        self.value = json.loads(state["content"])
        self._constructed = None

    @property
    def is_safe(self):
        return True

    @property
    def is_self_safe(self):
        return True

    def get_unsafe_set(self):
        return set()

    def _construct(self):
        return self.value


def get_tree(state, load_context: LoadContext):
    """Create instance based on the state, using json if possible"""
    saved_id = state.get("__id__")
    if saved_id in load_context.memo:
        # an instance has already been loaded, just return the loaded instance
        return load_context.get_object(saved_id)

    if state.get("is_json"):
        loaded_tree = JsonNode(state)
    else:
        try:
            node_cls = NODE_TYPE_MAPPING[state["__loader__"]]
        except KeyError:
            type_name = f"{state['__module__']}.{state['__class__']}"
            raise TypeError(
                f" Can't find loader {state['__loader__']} for type {type_name}."
            )

        loaded_tree = node_cls(state, load_context, trusted=False)

    # hold reference to obj in case same instance encountered again in save state
    if saved_id:
        load_context.memoize(loaded_tree, saved_id)

    return loaded_tree
