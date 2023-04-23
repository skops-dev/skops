from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

from skops.cli._utils import get_log_level, load_schema
from skops.io import dump, load
from skops.io._protocol import PROTOCOL


def _update_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    logger: logging.Logger = logging.getLogger(),
) -> None:
    """Function that is called by ``skops update`` entrypoint.

    Loads a skops model from the input path, converts to the current skops format, and
    saves to output file.

    Parameters
    ----------
    input_file : Union[str, Path]
        Path of input skops model to load.

    output_file : Union[str, Path]
        Path to save the updated skops model to.

    """
    input_model = load(input_file, trusted=True)
    input_file_schema = load_schema(input_file)

    if input_file_schema["protocol"] != PROTOCOL:
        dump(input_model, output_file)
        logger.info(f"Updated skops file written to {output_file}")

    logger.info(f"Input file is already up to date to the current protocol: {PROTOCOL}")


def format_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Adds arguments and help to parent CLI parser for the `update` method."""

    if not parser:  # used in tests
        parser = argparse.ArgumentParser()

    parser_subgroup = parser.add_argument_group("update")
    parser_subgroup.add_argument("input", help="Path to an input file to update.")

    parser_subgroup.add_argument(
        "-o",
        "--output-file",
        help="Specify the output file name for the updated skops file.",
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
    logger: logging.Logger = logging.getLogger(),
) -> None:
    output_file = Path(parsed_args.output_file)
    input_file = Path(parsed_args.input)

    logging.basicConfig(format="%(levelname)-8s: %(message)s")
    logger.setLevel(level=get_log_level(parsed_args.loglevel))

    _update_file(
        input_file=input_file,
        output_file=output_file,
        logger=logger,
    )
