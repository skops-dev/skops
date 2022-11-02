def check_type(module_name, class_name, trusted):
    """Check if a type is safe to load.

    Parameters
    ----------
    module_name : str
        The module name of the type.

    class_name : str
        The class name of the type.

    trusted : list of str
        A list of trusted types. If the type is in this list, it is considered safe.

    Returns
    -------
    is_safe : bool
        True if the type is safe, False otherwise.
    """
    return module_name + class_name in trusted


def audit_tree(tree, trusted):
    if trusted is True:
        return

    unsafe = tree.get_unsafe_set()
    if isinstance(trusted, (list, set)):
        unsafe -= set(trusted)
    if unsafe:
        raise TypeError(f"Untrusted types found in the file: {unsafe}.")
