from __future__ import annotations

import json

from ._audit import check_type
from ._utils import LoadContext

NODE_TYPE_MAPPING = {}  # type: ignore


class Node:
    """A node in the tree of objects.

    This class is a parent class for all nodes in the tree of objects. Each
    type of object (e.g. dict, list, etc.) has its own subclass of Node.

    Each child class has to implement two methods: ``__init__`` and
    ``_construct``.

    ``__init__`` takes care of traversing the state tree and to create the
    corresponding ``Node`` objects. It has access to the ``load_context`` which
    in turn has access to the source zip file. The child class's ``__init__``
    must also set the ``children`` attribute, which is a dictionary of
    ``{child_name: child_type}``. ``child_name`` is the name of the attribute
    which can be checked for safety, and ``child_type`` is the type of the
    attribute. ``child_type`` can be ``list``, ``dict``, or ``Node``. Note that
    primitives are persisted as a ``JsonNode``.

    ``_construct`` takes care of constructing the object. It is only called
    once and the result is cached in ``construct`` which is implemented in this
    class. All required data to construct an instance should be loaded during
    ``__init__``.

    The separation of ``__init__`` and ``_construct`` is necessary because
    audit methods are called after ``__init__`` and before ``construct``.
    Therefore ``__init__`` should avoid creating any instances or importing
    any modules, to avoid running unwanted code.

    Parameters
    ----------
    state : dict
        A dict representing the state of the dumped object.

    load_context : LoadContext
        The context of the loading process.

    trusted : bool or list, default=False
        If ``True``, the object will be loaded without any security checks. If
        ``False``, the object will be loaded only if there are only trusted
        objects in the dumped file. If a list of strings, the object will be
        loaded only if there are only trusted objects and objects of types
        listed in ``trusted`` are in the dumped file.
    """

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

        This is a convenience method called by child classes.
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
        """True only if the node's type is considered safe.

        This property only checks the type of the node, not its children.
        """
        return check_type(self.module_name, self.class_name, self.trusted)

    @property
    def is_safe(self):
        """Trie only if the node and all its children are safe."""
        # the safety value is cached.
        if self._is_safe is not None:
            return self._is_safe

        # if trusted is set to True, we don't do any safety checks.
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

        This method returns all types which are not trusted, including this
        node and all its children.

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

            # Get the safety set based on the type of the child. In most cases
            # other than ListNode and DictNode, children are all of type Node.
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
        # JsonNode is always considered safe.
        # TODO: should we consider a JsonNode always safe?
        return True

    @property
    def is_self_safe(self):
        return True

    def get_unsafe_set(self):
        return set()

    def _construct(self):
        return self.value


def get_tree(state, load_context: LoadContext):
    """Get the tree of nodes.

    This function returns the root node of the tree of nodes. The tree is
    constructed recursively by traversing the state tree. No instances are
    created during this process. One would need to call ``construct`` on the
    root node to create the instances.

    This function also handles memoization of the nodes. If a node has already
    been created, it is returned instead of creating a new one.

    Parameters
    ----------
    state : dict
        The state of the dumped object.

    load_context : LoadContext
        The context of the loading process.
    """
    saved_id = state.get("__id__")
    if saved_id in load_context.memo:
        # The Node is already loaded. We return the node. Note that the node is
        # not constructed at this point. It will be constructed when the parent
        # node's ``construct`` method is called, and for this node it'll be
        # called more than once. But that's not an issue since the node's
        # ``construct`` method caches the instance.
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
