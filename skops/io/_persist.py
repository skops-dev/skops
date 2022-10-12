from __future__ import annotations

import base64
import importlib
import io
import json
from zipfile import ZipFile

import skops

from ._utils import SaveState, _get_instance, _get_state, get_instance, get_state

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


def _save(obj):
    buffer = io.BytesIO()

    with ZipFile(buffer, "w") as zip_file:
        save_state = SaveState(zip_file=zip_file)
        state = get_state(obj, save_state)
        save_state.clear_memo()

        state["protocol"] = save_state.protocol
        state["_skops_version"] = skops.__version__

        zip_file.writestr("schema.json", json.dumps(state))

    return buffer


def dump(obj, file):
    buffer = _save(obj)
    with open(file, "wb") as f:
        f.write(buffer.getbuffer())


# TODO
save = dump


def dumps(obj):
    """TODO"""
    buffer = _save(obj)
    return base64.b64encode(buffer.getbuffer())


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
        instance = get_instance(json.loads(schema), input_zip)
    return instance


def loads(s):
    """TODO"""
    if isinstance(s, str):
        raise TypeError("Can't load skops format from string, pass bytes")

    with ZipFile(io.BytesIO(base64.b64decode(s)), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        instance = get_instance(schema, src=zip_file)
    return instance
