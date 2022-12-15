from __future__ import annotations

import argparse
import logging
import os
import pathlib
import pickle as pkl
from typing import Optional

from skops.io import dumps, get_untrusted_types


def _convert(
    input_file: os.PathLike, output_dir: pathlib.Path, is_trusted: bool = False
):
    model_name = pathlib.Path(input_file).stem

    logging.info(f"Converting {model_name}")

    with open(input_file, "rb") as f:
        obj = pkl.load(f)
    skops_dump = dumps(obj)

    untrusted_types = get_untrusted_types(data=skops_dump)

    if not untrusted_types:
        logging.debug(f"No unsafe types found in {model_name}.")
    else:
        untrusted_str = "\n".join(untrusted_types)

        logging.warning(
            "Unknown Types Detected!\n"
            f"While converting {model_name}, "
            "the following unsafe types were found: \n"
            f"{untrusted_str}\n"
        )

        if not is_trusted:
            logging.warning(
                f"Model {model_name} will not be converted due to unsafe types.\n"
                "To convert this anyway, add `-t` to this command."
            )
            return

    with open(output_dir / f"{model_name}.skops", "wb") as out_file:
        logging.info(f"Writing to {output_dir / model_name}.skops")
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
        "-t",
        "--trusted",
        help=(
            "Automatically trust all files, "
            "and convert all inputs, "
            "even if an untrusted type is detected."
        ),
        action="store_true",
        default=False,
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
    parser.add_argument(
        "--output-dir",
        help=(
            "Specify a directory to save converted files to. "
            "Default will save files to the current working directory."
        ),
        type=str,
        default=None,
    )
    args = parser.parse_args(command_line_args)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = pathlib.Path.cwd()
    else:
        output_dir = pathlib.Path(output_dir)

    logging.basicConfig(format="%(message)s", level=args.loglevel)

    for input_file in args.inputs:
        _convert(
            input_file=input_file,
            output_dir=output_dir,
            is_trusted=args.trusted,
        )
