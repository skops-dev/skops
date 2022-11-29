from importlib import import_module

from skops.utils.fixes import metadata


def import_or_raise(package):
    """Raise error

    Parameters
    ----------
    package: str
        Name of the package.

    Raises
    ------
    ModuleNotFoundError
        Is raised if a given module is not present in the environment
    """
    try:
        import_module(package)
    except metadata.PackageNotFoundError:
        raise ModuleNotFoundError(f"This feature requires {package} to be installed.")
