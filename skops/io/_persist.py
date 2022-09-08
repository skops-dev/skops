from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from ._utils import _import_obj, gettype

# We load these dictionaries from corresponding modules and merge them.
modules = ["._general", "._numpy", "._sklearn"]
GET_STATE_METHODS = {}
GET_INSTANCE_METHODS = {}
for module in modules:
    GET_STATE_METHODS.update(
        _import_obj(module, "get_state_methods", package="skops.io")()
    )
    GET_INSTANCE_METHODS.update(
        _import_obj(module, "get_instance_methods", package="skops.io")()
    )


def get_state_method(obj):
    # we go through the MRO and find the first class for which we have a method
    # to save the object. For instance, we might have a function for
    # BaseEstimator used for most classes, but a specialized one for Pipeline.
    for cls in type(obj).mro():
        if cls in GET_STATE_METHODS:
            return GET_STATE_METHODS[cls]

    raise TypeError(f"Can't serialize {type(obj)}")


def get_instance_method(state):
    cls_ = gettype(state)
    for cls in cls_.mro():
        if cls in GET_INSTANCE_METHODS:
            return GET_INSTANCE_METHODS[cls]

    raise TypeError(f"Can't deserialize {type(state)}")


def save(obj, file):
    with tempfile.TemporaryDirectory() as dst:
        with open(Path(dst) / "schema.json", "w") as f:
            json.dump(get_state_method(obj)(obj, dst), f, indent=2)

        # we use the zip format since tarfile can be exploited to create files
        # outside of the destination directory:
        # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        shutil.make_archive(file, format="zip", root_dir=dst)
        shutil.move(f"{file}.zip", file)


def load(file):
    input_zip = ZipFile(file)
    schema = input_zip.read("schema.json")
    return get_instance_method(json.loads(schema))(json.loads(schema), input_zip)
