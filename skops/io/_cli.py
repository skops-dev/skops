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

    if not is_trusted:
        untrusted_types = get_untrusted_types(data=skops_dump)
        if len(untrusted_types) > 0:
            # Print out untrusted types to command line
            untrusted_str = "\n".join(untrusted_types)
            print(untrusted_str)
            logging.warning(
                "Unsafe Types Detected!\n"
                f"While converting {model_name}, "
                "the following unsafe types were found: \n"
                f"{untrusted_str}\n"
                "To convert this anyway, add `-t` to this command."
            )
            return

        logging.debug(f"No unsafe types found in {model_name}.")

    with open(output_dir / f"{model_name}.skops", "wb") as out_file:
        print(f"Writing to {output_dir/model_name}.skops")
        out_file.write(skops_dump)


def main_convert(command_line_args: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert input Pickle files to .skops files"
    )
    parser.add_argument("input-files", type=argparse.FileType("r"), nargs="+")
    parser.add_argument("-t", "--trusted", action="store_true", default=False)
    parser.add_argument(
        "-d",
        "--debug",
        help="Enable debug logging",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose logging",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()

    for input_file in args.input_files:
        _convert(
            input_file=input_file,
            output_dir=pathlib.Path.cwd(),
            is_trusted=args.trusted,
        )
