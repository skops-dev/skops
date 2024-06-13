from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from skops.cli._utils import get_log_level
from skops.io import dump, get_untrusted_types, load
from skops.io._protocol import PROTOCOL


def _update_file(
    input_file: str | Path,
    output_file: str | Path | None = None,
    inplace: bool = False,
    logger: logging.Logger = logging.getLogger(),
) -> None:
    """Function that is called by ``skops update`` entrypoint.

    Loads a skops model from the input path, updates it to the current skops format, and
    saves to an output file. It will overwrite the input file if `inplace` is True.

    Parameters
    ----------
    input_file : str, or Path
        Path of input skops model to load.

    output_file : str, or Path, default=None
        Path to save the updated skops model to.

    inplace : bool, default=False
        Whether to update and overwrite the input file in place.

    logger : logging.Logger, default=logging.getLogger()
        Logger to use for logging.
    """
    if inplace:
        if output_file is None:
            output_file = input_file
        else:
            raise ValueError(
                "Cannot specify both an output file path and the inplace flag. Please"
                " choose whether you want to create a new file or overwrite the input"
                " file."
            )

    input_model = load(input_file, trusted=get_untrusted_types(file=input_file))
    with zipfile.ZipFile(input_file, "r") as zip_file:
        input_file_schema = json.loads(zip_file.read("schema.json"))

    if input_file_schema["protocol"] == PROTOCOL:
        logger.warning(
            "File was not updated because already up to date with the current protocol:"
            f" {PROTOCOL}"
        )
        return None

    if input_file_schema["protocol"] > PROTOCOL:
        logger.warning(
            "File cannot be updated because its protocol is more recent than the "
            f"current protocol: {PROTOCOL}"
        )
        return None

    if output_file is None:
        logger.warning(
            f"File can be updated to the current protocol: {PROTOCOL}. Please"
            " specify an output file path or use the `inplace` flag to create the"
            " updated Skops file."
        )
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_output_file = Path(tmp_dir) / f"{output_file}.tmp"
        dump(input_model, tmp_output_file)
        shutil.move(str(tmp_output_file), str(output_file))
    logger.info(f"Updated skops file written to {output_file}")


def format_parser(
    parser: argparse.ArgumentParser | None = None,
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
        default=None,
    )
    parser_subgroup.add_argument(
        "--inplace",
        help="Update and overwrite the input file in place.",
        action="store_true",
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
    output_file = Path(parsed_args.output_file) if parsed_args.output_file else None
    input_file = Path(parsed_args.input)
    inplace = parsed_args.inplace

    logging.basicConfig(format="%(levelname)-8s: %(message)s")
    logger.setLevel(level=get_log_level(parsed_args.loglevel))

    _update_file(
        input_file=input_file,
        output_file=output_file,
        inplace=inplace,
        logger=logger,
    )
