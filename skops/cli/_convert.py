from __future__ import annotations

import argparse
import logging
import os
import pathlib
import pickle
from typing import Optional

from skops.cli._utils import get_log_level
from skops.io import dumps, get_untrusted_types


def _convert_file(
    input_file: os.PathLike,
    output_file: os.PathLike,
    logger: logging.Logger = logging.getLogger(),
) -> None:
    """Function that is called by ``skops convert`` entrypoint.

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

    logger.debug(f"Converting {model_name}")

    with open(input_file, "rb") as f:
        obj = pickle.load(f)
    skops_dump = dumps(obj)

    untrusted_types = get_untrusted_types(data=skops_dump)

    if not untrusted_types:
        logger.info(f"No unknown types found in {model_name}.")
    else:
        untrusted_str = ", ".join(untrusted_types)

        logger.warning(
            f"While converting {input_file}, "
            "the following unknown types were found: "
            f"{untrusted_str}. "
            f"When loading {output_file} with skops.load, these types must be "
            "specified as 'trusted'"
        )

    with open(output_file, "wb") as out_file:
        logger.debug(f"Writing to {output_file}")
        out_file.write(skops_dump)


def format_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Adds arguments and help to parent CLI parser for the convert method."""

    if not parser:  # used in tests
        parser = argparse.ArgumentParser()

    parser_subgroup = parser.add_argument_group("convert")
    parser_subgroup.add_argument("input", help="Path to an input file to convert. ")

    parser_subgroup.add_argument(
        "-o",
        "--output-file",
        help=(
            "Specify the output file name for the converted skops file. "
            "If not provided, will default to using the same name as the input file, "
            "and saving to the current working directory with the suffix '.skops'."
        ),
        default=None,
    )
    parser_subgroup.add_argument(
        "-v",
        "--verbose",
        help=(
            "Increases verbosity of logging. Can be used multiple times to increase "
            "verbosity further."
        ),
        action="count",
        dest="loglevel",
        default=0,
    )
    return parser


def main(
    parsed_args: argparse.Namespace,
) -> None:
    output_file = parsed_args.output_file
    input_file = parsed_args.input

    logging.basicConfig(
        format="%(levelname)-8s: %(message)s", level=get_log_level(parsed_args.loglevel)
    )

    if not output_file:
        # No filename provided, defaulting to base file path
        file_name = pathlib.Path(input_file).stem
        output_file = pathlib.Path.cwd() / f"{file_name}.skops"

    _convert_file(
        input_file=input_file,
        output_file=output_file,
    )
