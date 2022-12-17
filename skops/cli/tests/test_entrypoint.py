import pathlib
import sys
from unittest import mock

import pytest

from skops.cli.entrypoint import main_entrypoint


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
    def test_convert_works_as_expected(self, mocked_convert: mock.MagicMock):
        args = ["convert", "abc.def"]

        main_entrypoint(args)

        mocked_convert.assert_called_once_with(
            input_file="abc.def",
            output_file=pathlib.Path.cwd() / "abc.skops",
        )
