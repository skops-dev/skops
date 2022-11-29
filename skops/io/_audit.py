from skops.io.exceptions import UntrustedTypesFoundException


def check_type(module_name, type_name, trusted):
    """Check if a type is safe to load.

    A type is safe to load only if it's present in the trusted list.

    Parameters
    ----------
    module_name : str
        The module name of the type.

    type_name : str
        The class name of the type.

    trusted : bool, or list of str
        If ``True``, the tree is considered safe. Otherwise trusted has to be
        a list of trusted types.

    Returns
    -------
    is_safe : bool
        True if the type is safe, False otherwise.
    """
    if trusted is True:
        return True
    return module_name + "." + type_name in trusted


def audit_tree(tree, trusted):
    """Audit a tree of nodes.

    A tree is safe if it only contains trusted types. Audit is skipped if
    trusted is ``True``.

    Parameters
    ----------
    tree : skops.io._dispatch.Node
        The tree to audit.

    trusted : bool, or list of str
        If ``True``, the tree is considered safe. Otherwise trusted has to be
        a list of trusted types names.

        An entry in the list is typically of the form
        ``skops.io._utils.get_module(obj) + "." + obj.__class__.__name__``.

    Raises
    ------
    UntrustedTypesFoundException
        If the tree contains an untrusted type.
    """
    if trusted is True:
        return

    unsafe = tree.get_unsafe_set()
    if isinstance(trusted, (list, set)):
        unsafe -= set(trusted)
    if unsafe:
        raise UntrustedTypesFoundException(unsafe)
