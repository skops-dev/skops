from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import skops

from ._utils import get_instance, get_state

# For now, there is just one protocol version
PROTOCOL = 0


# We load the dispatch functions from the corresponding modules and register
# them.
modules = ["._general", "._numpy", "._scipy", "._sklearn"]
for module_name in modules:
    # register exposed functions for get_state and get_instance
    module = importlib.import_module(module_name, package="skops.io")
    for cls, method in getattr(module, "GET_STATE_DISPATCH_FUNCTIONS", []):
        get_state.register(cls)(method)
    for cls, method in getattr(module, "GET_INSTANCE_DISPATCH_FUNCTIONS", []):
        get_instance.register(cls)(method)


def save(obj, file):
    with tempfile.TemporaryDirectory() as dst:
        with open(Path(dst) / "schema.json", "w") as f:
            state = get_state(obj, dst)
            state["protocol"] = PROTOCOL
            state["_skops_version"] = skops.__version__
            json.dump(state, f, indent=2)

        # we use the zip format since tarfile can be exploited to create files
        # outside of the destination directory:
        # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        shutil.make_archive(file, format="zip", root_dir=dst)
        shutil.move(f"{file}.zip", file)


def load(file):
    with ZipFile(file, "r") as input_zip:
        schema = input_zip.read("schema.json")
        instance = get_instance(json.loads(schema), input_zip)
    input_zip.close()
    return instance
