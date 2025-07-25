from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
from typing import Any, BinaryIO, Optional, Sequence
from zipfile import ZIP_STORED, ZipFile

import skops

from ._audit import NODE_TYPE_MAPPING, audit_tree, get_tree
from ._utils import LoadContext, SaveContext, _get_state, get_state

# We load the dispatch functions from the corresponding modules and register
# them. Old protocols are found in the 'old/' directory, with the protocol
# version appended to the corresponding module name.
modules = ["._general", "._numpy", "._scipy", "._sklearn", "._quantile_forest"]
modules.extend([".old._general_v0", ".old._numpy_v0", ".old._numpy_v1"])
for module_name in modules:
    # register exposed functions for get_state and get_tree
    module = importlib.import_module(module_name, package="skops.io")
    for cls, method in getattr(module, "GET_STATE_DISPATCH_FUNCTIONS", []):
        _get_state.register(cls)(method)
    # populate the the dict used for dispatching get_tree functions
    NODE_TYPE_MAPPING.update(module.NODE_TYPE_MAPPING)


def _save(obj: Any, compression: int, compresslevel: int | None) -> io.BytesIO:
    buffer = io.BytesIO()

    with ZipFile(
        buffer, "w", compression=compression, compresslevel=compresslevel
    ) as zip_file:
        save_context = SaveContext(zip_file=zip_file)
        state = get_state(obj, save_context)
        save_context.clear_memo()

        state["protocol"] = save_context.protocol
        state["_skops_version"] = skops.__version__
        zip_file.writestr("schema.json", json.dumps(state, indent=2))

    return buffer


def dump(
    obj: Any,
    file: str | Path | BinaryIO,
    *,
    compression: int = ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Save an object using the skops persistence format.

    Skops aims at providing a secure persistence feature that does not rely on
    :mod:`pickle`, which is inherently insecure. For more information, please
    visit the :ref:`persistence` documentation.

    Parameters
    ----------
    obj: object
        The object to be saved. Usually a scikit-learn compatible model.

    file: str, path, or file-like object
        The file name. A zip archive will automatically created. As a matter of
        convention, we recommend to use the ".skops" file extension, e.g.
        ``save(model, "my-model.skops")``.

    compression: int, default=zipfile.ZIP_STORED
        The compression method to use. See :class:`zipfile.ZipFile` for more
        information.

        .. versionadded:: 0.7

    compresslevel: int, default=None
        The compression level to use. See :class:`zipfile.ZipFile` for more
        information.

        .. versionadded:: 0.7
    """
    buffer = _save(obj, compression=compression, compresslevel=compresslevel)

    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            f.write(buffer.getbuffer())
    else:
        file.write(buffer.getbuffer())


def dumps(
    obj: Any, *, compression: int = ZIP_STORED, compresslevel: int | None = None
) -> bytes:
    """Save an object using the skops persistence format as a bytes object.

    Parameters
    ----------
    obj: object
        The object to be saved. Usually a scikit-learn compatible model.

    compression: int, default=zipfile.ZIP_STORED
        The compression method to use. See :class:`zipfile.ZipFile` for more
        information.

        .. versionadded:: 0.7

    compresslevel: int, default=None
        The compression level to use. See :class:`zipfile.ZipFile` for more
        information.

        .. versionadded:: 0.7
    """
    buffer = _save(obj, compression=compression, compresslevel=compresslevel)
    return buffer.getbuffer().tobytes()


def load(file: str | Path, trusted: Optional[Sequence[str]] = None) -> Any:
    """Load an object saved with the skops persistence format.

    Skops aims at providing a secure persistence feature that does not rely on
    :mod:`pickle`, which is inherently insecure. For more information, please
    visit the :ref:`persistence` documentation.

    Parameters
    ----------
    file: str or pathlib.Path
        The file name of the object to be loaded.

    trusted: list of str, default=None
        The object will be loaded only if there are only trusted objects and
        objects of types listed in ``trusted`` in the dumped file.

    Returns
    -------
    instance: object
        The loaded object.

    """
    if trusted is True:
        raise TypeError(
            "trusted must be a list of strings. Before version 0.10 trusted could "
            "be a boolean, but this is no longer supported, due to a reported "
            "CVE-2024-37065. You can pass the output of `get_untrusted_types` as "
            "trusted to load the data. Be sure to review the output of the function "
            "before passing it as trusted."
        )

    with ZipFile(file, "r") as input_zip:
        schema = json.loads(input_zip.read("schema.json"))
        load_context = LoadContext(src=input_zip, protocol=schema["protocol"])
        tree = get_tree(schema, load_context, trusted=trusted)
        audit_tree(tree, trusted=trusted)
        instance = tree.construct()

    return instance


def loads(data: bytes, trusted: Optional[Sequence[str]] = None) -> Any:
    """Load an object saved with the skops persistence format from a bytes
    object.

    Parameters
    ----------
    data: bytes
        The dumped data to be loaded in bytes format.

    trusted: bool, or list of str, default=False
        The object will be loaded only if there are only trusted objects and
        objects of types listed in ``trusted`` in the dumped file.

    Returns
    -------
    instance: object
        The loaded object.
    """
    if isinstance(data, str):
        raise TypeError("Can't load skops format from string, pass bytes")

    if trusted is True:
        raise TypeError(
            "trusted must be a list of strings. Before version 0.10 trusted could "
            "be a boolean, but this is no longer supported, due to a reported "
            "CVE-2024-37065. You can pass the output of `get_untrusted_types` as "
            "trusted to load the data. Be sure to review the output of the function "
            "before passing it as trusted."
        )

    with ZipFile(io.BytesIO(data), "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        load_context = LoadContext(src=zip_file, protocol=schema["protocol"])
        tree = get_tree(schema, load_context, trusted=trusted)
        audit_tree(tree, trusted=trusted)
        instance = tree.construct()

    return instance


def get_untrusted_types(
    *, data: bytes | None = None, file: str | Path | None = None
) -> list[str]:
    """Get a list of untrusted types in a skops dump.

    Parameters
    ----------
    data: bytes
        The data to be checked, in bytes format.

    file: str or Path
        The file to be checked.

    Returns
    -------
    untrusted_types: list of str
        The list of untrusted types in the dump.

    Notes
    -----
    Only one of data or file should be passed.
    """
    if data and file:
        raise ValueError("Only one of data or file should be passed.")
    if not data and not file:
        raise ValueError("Exactly one of data or file should be passed.")

    content: io.BytesIO | str | Path
    if data:
        content = io.BytesIO(data)
    else:
        # mypy doesn't understand that file cannot be None here, thus ignore
        content = file  # type: ignore

    with ZipFile(content, "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
        load_context = LoadContext(src=zip_file, protocol=schema["protocol"])
        tree = get_tree(schema, load_context=load_context, trusted=None)
        untrusted_types = tree.get_unsafe_set()

    return sorted(untrusted_types)
