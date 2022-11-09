def check_type(module_name, type_name, trusted):
    """Check if a type is safe to load.

    A type is safe to load only if it's present in the trusted list.

    Parameters
    ----------
    module_name : str
        The module name of the type.

    type_name : str
        The class name of the type.

    trusted : list of str
        A list of trusted types. If the type is in this list, it is considered safe.

    Returns
    -------
    is_safe : bool
        True if the type is safe, False otherwise.
    """
    return module_name + "." + type_name in trusted


def audit_tree(tree, trusted):
    """Audit a tree of nodes.

    A tree is safe only if it contains trusted types. Audit is skipped if
    trusted is ``True``.

    Parameters
    ----------
    tree : Node
        The tree to audit.

    trusted : bool, or list of str
        If ``True``, the tree is considered safe. Otherwise trusted has to be
        a list of trusted types.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the tree contains an untrusted type.
    """
    if trusted is True:
        return

    # breakpoint()
    unsafe = tree.get_unsafe_set()
    if isinstance(trusted, (list, set)):
        unsafe -= set(trusted)
    if unsafe:
        raise TypeError(f"Untrusted types found in the file: {unsafe}.")
