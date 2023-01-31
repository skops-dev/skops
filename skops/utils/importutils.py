from importlib import import_module


def import_or_raise(module, feature_name):
    """Raise error if a given library is not present in the environment.

    Parameters
    ----------
    module: str
        Name of the module.

    feature_name: str
        Name of the feature module is required for.

    Raises
    ------
    ModuleNotFoundError
        Is raised if a given module is not present in the environment
    """
    try:
        module = import_module(module)
    except ImportError as e:
        package = module.split(".")[0]
        raise ModuleNotFoundError(
            f"{feature_name.capitalize()} requires {package} to be installed. In order"
            f" to use {feature_name}, you need to install the package in your current"
            " python environment."
        ) from e
    return module
