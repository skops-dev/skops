from __future__ import annotations

import argparse
import logging
import os
import pathlib
import pickle
from typing import Optional

from skops.io import dumps, get_untrusted_types


def _convert_file(input_file: os.PathLike, output_file: os.PathLike):
    """
    Function that is called by ``skops convert`` entrypoint.

    Loads a pickle model from the input path, converts to skops format, and saves to
    output file.

    Parameters
    ----------
    input_file : os.PathLike
        Path of input .pkl model to load.

    output_file : os.PathLike
        Path to save .skops model to.

    """
    model_name = pathlib.Path(input_file).stem

    logging.info(f"Converting {model_name}")

    with open(input_file, "rb") as f:
        obj = pickle.load(f)
    skops_dump = dumps(obj)

    untrusted_types = get_untrusted_types(data=skops_dump)

    if not untrusted_types:
        logging.debug(f"No unknown types found in {model_name}.")
    else:
        untrusted_str = "\n".join(untrusted_types)

        logging.warning(
            "Unknown Types Detected!\n"
            f"While converting {model_name}, "
            "the following unknown types were found: \n"
            f"{untrusted_str}\n\n"
            f"When loading {output_file}, add ``trusted=True`` to the skops.load call."
        )

    with open(output_file, "wb") as out_file:
        logging.info(f"Writing to {output_file}")
        out_file.write(skops_dump)


def main_convert(
    command_line_args: Optional[list[str]] = None,
    parent: Optional[argparse.ArgumentParser] = None,
):
    parents = [parent] if parent else []

    parser = argparse.ArgumentParser(
        description="Convert input Pickle files to .skops files", parents=parents
    )

    parser.add_argument("inputs", nargs="+", help="Input files to convert.")
    parser.add_argument(
        "-o",
        "--output-files",
        nargs="+",
        help=(
            "Specify output file names for the converted skops files."
            "If not provided, will default to using the same name as the input file, "
            "and saving to the current working directory."
        ),
        default=None,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Enable debug logging.",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose logging.",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args(command_line_args)
    output_files = args.output_files
    input_files = args.inputs
    logging.basicConfig(format="%(message)s", level=args.loglevel)

    for input_file_index in range(len(args.inputs)):
        input_file = input_files[input_file_index]
        if output_files and len(output_files) > input_file_index:
            output_file = output_files[input_file_index]
        else:
            # No filename provided, defaulting to base file path
            file_name = pathlib.Path(input_file).stem
            output_file = pathlib.Path.cwd() / f"{file_name}.skops"

        _convert_file(
            input_file=input_file,
            output_file=output_file,
        )
