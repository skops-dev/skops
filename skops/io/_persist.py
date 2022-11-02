from __future__ import annotations

import importlib
import io
import json
from zipfile import ZipFile

import skops

from ._audit import audit_tree
from ._dispatch import NODE_TYPE_MAPPING, get_tree
from ._utils import SaveState, _get_state, get_state

# We load the dispatch functions from the corresponding modules and register
# them.
modules = ["._general", "._numpy", "._scipy", "._sklearn"]
for module_name in modules:
    # register exposed functions for get_state and get_tree
    module = importlib.import_module(module_name, package="skops.io")
    for cls, method in getattr(module, "GET_STATE_DISPATCH_FUNCTIONS", []):
        _get_state.register(cls)(method)
    # populate the the dict used for dispatching get_tree functions
    NODE_TYPE_MAPPING.update(module.NODE_TYPE_MAPPING)


def _save(obj):
    buffer = io.BytesIO()

    with ZipFile(buffer, "w") as zip_file:
        save_state = SaveState(zip_file=zip_file)
        state = get_state(obj, save_state)
        save_state.clear_memo()

        state["protocol"] = save_state.protocol
        state["_skops_version"] = skops.__version__

        zip_file.writestr("schema.json", json.dumps(state, indent=2))

    return buffer


def dump(obj, file):
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
    buffer = _save(obj)
    with open(file, "wb") as f:
        f.write(buffer.getbuffer())


def dumps(obj):
    """Save an object using the skops persistence format as a bytes object.

    .. warning::

       This feature is very early in development, which means the API is
       unstable and it is **not secure** at the moment. Therefore, use the same
       caution as you would for ``pickle``: Don't load from sources that you
       don't trust. In the future, more security will be added.

    Parameters
    ----------
    obj: object
        The object to be saved. Usually a scikit-learn compatible model.

    """
    buffer = _save(obj)
    return buffer.getbuffer().tobytes()


def load(file, trusted=False):
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

    trusted: bool, or list of str, default=False
        If ``True``, the object will be loaded without any security checks. If
        ``False``, the object will be loaded only if there are only trusted
        objects in the dumped file. If a list of strings, the object will be
        loaded only if there are only trusted objects and objects of types
        listed in ``trusted`` are in the dumped file.

    Returns
    -------
    instance: object
        The loaded object.

    """
    with ZipFile(file, "r") as input_zip:
        schema = input_zip.read("schema.json")
        tree = get_tree(json.loads(schema), input_zip)
        audit_tree(tree, trusted)
        instance = tree.construct()

    return instance


def loads(data, trusted=False):
    """Load an object saved with the skops persistence format from a bytes
    object.

    .. warning::

       This feature is very early in development, which means the API is
       unstable and it is **not secure** at the moment. Therefore, use the same
       caution as you would for ``pickle``: Don't load from sources that you
       don't trust. In the future, more security will be added.

    Parameters
    ----------
    data: bytes
        The file name of the object to be loaded.

    trusted: bool, or list of str, default=False
        If ``True``, the object will be loaded without any security checks. If
        ``False``, the object will be loaded only if there are only trusted
        objects in the dumped file. If a list of strings, the object will be
        loaded only if there are only trusted objects and objects of types
        listed in ``trusted`` are in the dumped file.

    Returns
    -------
    instance: object
        The loaded object.
    """
    if isinstance(data, str):
        raise TypeError("Can't load skops format from string, pass bytes")

    with ZipFile(io.BytesIO(data), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        tree = get_tree(schema, src=zip_file)
        audit_tree(tree, trusted)
        instance = tree.construct()

    return instance
