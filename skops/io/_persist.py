from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import skops

from ._utils import _get_instance, _get_state

# For now, there is just one protocol version
PROTOCOL = 0


# We load the dispatch functions from the corresponding modules and register
# them.
modules = ["._general", "._numpy", "._scipy", "._sklearn"]
for module_name in modules:
    # register exposed functions for get_state and get_instance
    module = importlib.import_module(module_name, package="skops.io")
    for cls, method in getattr(module, "GET_STATE_DISPATCH_FUNCTIONS", []):
        _get_state.register(cls)(method)
    for cls, method in getattr(module, "GET_INSTANCE_DISPATCH_FUNCTIONS", []):
        _get_instance.register(cls)(method)


def save(obj, file):
    """Save an object using the skops persistence format.

    Skops aims at providing a secure persistence feature that does not rely on
    :mod:`pickle`, which is inherently insecure. For more information, please
    visit the :ref:`persistence` documentation.

    .. warning::

       This feature is very early in development, which means the API is
       unstable and it is **not secure** at the moment. Therefore, use the same
       caution as you would for ``pickle``: Don't load from sources that you
       don't trust. In the future, more security will be added.

    Parameters
    ----------
    obj: object
        The object to be saved. Usually a scikit-learn compatible model.

    file: str
        The file name. A zip archive will automatically created. As a matter of
        convention, we recommend to use the ".skops" file extension, e.g.
        ``save(model, "my-model.skops")``.

    """
    with tempfile.TemporaryDirectory() as dst:
        with open(Path(dst) / "schema.json", "w") as f:
            state = _get_state(obj, dst)
            state["protocol"] = PROTOCOL
            state["_skops_version"] = skops.__version__
            json.dump(state, f, indent=2)

        # we use the zip format since tarfile can be exploited to create files
        # outside of the destination directory:
        # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        shutil.make_archive(file, format="zip", root_dir=dst)
        shutil.move(f"{file}.zip", file)


def load(file):
    """Load an object saved with the skops persistence format.

    Skops aims at providing a secure persistence feature that does not rely on
    :mod:`pickle`, which is inherently insecure. For more information, please
    visit the :ref:`persistence` documentation.

    .. warning::

       This feature is very early in development, which means the API is
       unstable and it is **not secure** at the moment. Therefore, use the same
       caution as you would for ``pickle``: Don't load from sources that you
       don't trust. In the future, more security will be added.

    Parameters
    ----------
    file: str
        The file name of the object to be loaded.

    Returns
    -------
    instance: object
        The loaded object.

    """
    with ZipFile(file, "r") as input_zip:
        schema = input_zip.read("schema.json")
        instance = _get_instance(json.loads(schema), input_zip)
    return instance
