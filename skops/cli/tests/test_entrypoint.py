import logging
import pathlib
import sys
from unittest import mock

import pytest

from skops.cli.entrypoint import main_cli


class TestEntrypoint:
    """Integration tests that check that entrypoint calls pass through correctly.
    Full coverage of individual entrypoint calls should be done in their own classes.
    """

    @pytest.fixture(autouse=True)
    def clear_argv(self):
        # Required to clear argv in case Pytest is called on this specific function.
        # Otherwise, clogs parser.parse_known_args() in argparse
        sys.argv = [""]

    @mock.patch("skops.cli._convert._convert_file")
    def test_convert_works_as_expected(
        self,
        convert_file_mock: mock.MagicMock,
        caplog,
    ):
        """
        Intended as a unit test to make sure,
        given 'convert' as the first argument,
        the parser is configured correctly
        """

        args = ["convert", "abc.def"]

        main_cli(args)
        convert_file_mock.assert_called_once_with(
            input_file="abc.def", output_file=pathlib.Path.cwd() / "abc.skops"
        )

        assert caplog.at_level(logging.WARNING)

    @mock.patch("skops.cli._update._update_file")
    def test_update_works_as_expected(
        self,
        update_file_mock: mock.MagicMock,
    ):
        """
        To make sure the parser is configured correctly, when 'update'
        is the first argument.
        """

        args = ["update", "abc.skops", "-o", "abc-new.skops"]

        main_cli(args)
        update_file_mock.assert_called_once_with(
            input_file=pathlib.Path("abc.skops"),
            output_file=pathlib.Path("abc-new.skops"),
            inplace=False,
            logger=mock.ANY,
        )
