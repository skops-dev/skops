import importlib.metadata
import logging
import pathlib
import subprocess
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
            trusted=[],
            logger=mock.ANY,
        )


def test_console_script_is_registered():
    """The ``skops`` command must be declared in the package metadata.

    The console script was declared in ``setup.py`` and silently lost when the
    packaging moved to ``pyproject.toml`` in :pr:`451`.
    """
    dist = importlib.metadata.distribution("skops")
    scripts = {
        ep.name: ep.value for ep in dist.entry_points if ep.group == "console_scripts"
    }
    assert scripts == {"skops": "skops.cli.entrypoint:main_cli"}


def test_python_m_skops():
    """``python -m skops`` runs the same CLI as the ``skops`` command."""
    result = subprocess.run(
        [sys.executable, "-m", "skops", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "convert" in result.stdout
    assert "update" in result.stdout
