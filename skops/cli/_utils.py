import json
import logging
import pathlib
import zipfile
from typing import Any, Union


def get_log_level(level: int = 0) -> int:
    """Takes in verbosity from a CLI entrypoint (number of times -v specified),
    and sets the logger to the required log level"""

    all_levels = [logging.WARNING, logging.INFO, logging.DEBUG]

    if level > len(all_levels):
        level = len(all_levels) - 1
    elif level < 0:
        level = 0

    return all_levels[level]


def load_schema(skops_file_path: Union[str, pathlib.Path]) -> dict[str, Any]:
    with zipfile.ZipFile(skops_file_path, "r") as zip_file:
        schema = json.loads(zip_file.read("schema.json"))
    return schema
