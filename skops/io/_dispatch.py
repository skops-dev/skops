from __future__ import annotations

from contextlib import contextmanager

from ._audit import check_type
from ._utils import LoadContext

NODE_TYPE_MAPPING = {}  # type: ignore


class UNINITIALIZED:
    """Sentinel value to indicate that a value has not been initialized yet."""

    ...


@contextmanager
def temp_setattr(obj, **kwargs):
    """Context manager to temporarily set attributes on an object."""
    has_attr = {k: hasattr(obj, k) for k in kwargs.keys()}
    previous_values = {k: getattr(obj, k, None) for k in kwargs}
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in previous_values.items():
            if has_attr[k]:
                setattr(obj, k, v)
            else:
                delattr(obj, k)


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
    any modules, to avoid running potentially untrusted code.

    Parameters
    ----------
    state : dict
        A dict representing the state of the dumped object.

    load_context : LoadContext
        The context of the loading process.

    trusted : bool or list of str, default=False
        If ``True``, the object will be loaded without any security checks. If
        ``False``, the object will be loaded only if there are only trusted
        objects in the dumped file. If a list of strings, the object will be
        loaded only if all of its required types are listed in ``trusted``
        or are trusted by default.
    """

    def __init__(self, state, load_context: LoadContext, trusted=False, memoize=True):
        self.class_name, self.module_name = state["__class__"], state["__module__"]
        self.trusted = trusted
        self._is_safe = None
        self._constructed = UNINITIALIZED()
        saved_id = state.get("__id__")
        if saved_id and memoize:
            # hold reference to obj in case same instance encountered again in
            # save state
            load_context.memoize(self, saved_id)

    def construct(self):
        """Construct the object.

        We only construct the object once, and then cache the result.
        """
        if not isinstance(self._constructed, UNINITIALIZED):
            return self._constructed
        self._constructed = self._construct()
        return self._constructed

    @staticmethod
    def _get_trusted(trusted, default):
        """Return a trusted list, or True.

        If ``trusted`` is ``False``, we return the ``default``, otherwise the
        ``trusted`` value is used.

        This is a convenience method called by child classes.
        """
        if trusted is True:
            # if trusted is True, we trust the node
            return True

        if trusted is False:
            # if trusted is False, we only trust the defaults
            return default

        # otherwise we trust the given list
        return trusted

    def is_self_safe(self):
        """True only if the node's type is considered safe.

        This property only checks the type of the node, not its children.
        """
        return check_type(self.module_name, self.class_name, self.trusted)

    def is_safe(self):
        """True only if the node and all its children are safe."""
        # if trusted is set to True, we don't do any safety checks.
        if self.trusted is True:
            return True

        return len(self.get_unsafe_set()) == 0

    def get_unsafe_set(self):
        """Get the set of unsafe types.

        This method returns all types which are not trusted, including this
        node and all its children.

        Returns
        -------
        unsafe_set : set
            A set of unsafe types.
        """
        if hasattr(self, "_computing_unsafe_set"):
            # this means we're already computing this node's unsafe set, so we
            # return an empty set and let the computation of the parent node
            # continue. This is to avoid infinite recursion.
            return set()

        with temp_setattr(self, _computing_unsafe_set=True):
            res = set()
            if not self.is_self_safe():
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


class CachedNode(Node):
    def __init__(self, state, load_context: LoadContext, trusted=False):
        # we pass memoize as False because we don't want to memoize the cached
        # node.
        super().__init__(state, load_context, trusted, memoize=False)
        self.trusted = True
        self.cached = load_context.get_object(state.get("__id__"))
        self.children = {}  # type: ignore

    def _construct(self):
        # TODO: FIXME This causes a recursion error when loading a cached
        # object if we call the cached object's `construct``. Some refactoring
        # is needed to fix this.
        return self.cached.construct()


NODE_TYPE_MAPPING.update({"CachedNode": CachedNode})


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

    try:
        node_cls = NODE_TYPE_MAPPING[state["__loader__"]]
    except KeyError:
        type_name = f"{state['__module__']}.{state['__class__']}"
        raise TypeError(
            f" Can't find loader {state['__loader__']} for type {type_name}."
        )

    loaded_tree = node_cls(state, load_context, trusted=False)  # type: ignore

    return loaded_tree
