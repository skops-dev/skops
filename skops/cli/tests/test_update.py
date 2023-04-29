import logging
import pathlib
from functools import partial
from unittest import mock

import numpy as np
import pytest

from skops.cli import _update
from skops.io import _persist, _protocol, dump, load


class TestUpdate:
    model_name = "some_model_name"

    @pytest.fixture
    def safe_obj(self) -> np.ndarray:
        return np.ndarray([1, 2, 3, 4])

    @pytest.fixture
    def skops_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / f"{self.model_name}.skops"

    @pytest.fixture
    def new_skops_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / f"{self.model_name}-new.skops"

    @pytest.fixture
    def dump_file(self, skops_path: pathlib.Path, safe_obj: np.ndarray):
        """Dump an object using the current protocol version."""
        dump(safe_obj, skops_path)

    @pytest.fixture
    @mock.patch(
        "skops.io._persist.SaveContext", partial(_persist.SaveContext, protocol=0)
    )
    def dump_old_file(self, skops_path: pathlib.Path, safe_obj: np.ndarray):
        """Dump an object using an old protocol version so that the file needs
        updating.
        """
        dump(safe_obj, skops_path)

    def test_base_case_works_as_expected(
        self,
        skops_path: pathlib.Path,
        new_skops_path: pathlib.Path,
        dump_old_file,
        safe_obj,
    ):
        mock_logger = mock.MagicMock()
        _update._update_file(
            input_file=skops_path, output_file=new_skops_path, logger=mock_logger
        )
        updated_obj = load(new_skops_path)

        assert np.array_equal(updated_obj, safe_obj)

        # Check logging messages
        mock_logger.info.assert_called_once_with(
            f"Updated skops file written to {new_skops_path}"
        )
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.debug.assert_not_called()

    def test_no_update(
        self,
        skops_path: pathlib.Path,
        new_skops_path: pathlib.Path,
        dump_file,
    ):
        mock_logger = mock.MagicMock()
        _update._update_file(
            input_file=skops_path, output_file=new_skops_path, logger=mock_logger
        )
        mock_logger.info.assert_called_once_with(
            "File was not updated because already up to date with the current protocol:"
            f" {_protocol.PROTOCOL}"
        )
        assert not new_skops_path.exists()


class TestMain:
    @pytest.fixture
    def tmp_logger(self) -> logging.Logger:
        return logging.getLogger()

    @mock.patch("skops.cli._update._update_file")
    def test_base_works_as_expected(
        self, mock_update: mock.MagicMock, tmp_logger: logging.Logger
    ):
        input_path = "abc.skops"
        output_path = "abc-new.skops"
        namespace, _ = _update.format_parser().parse_known_args(
            [input_path, "-o", output_path]
        )

        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=pathlib.Path(output_path),
            logger=tmp_logger,
        )

    @mock.patch("skops.cli._update._update_file")
    @pytest.mark.parametrize(
        "verbosity, expected_level",
        [
            ("", logging.WARNING),
            ("-v", logging.INFO),
            ("--verbose", logging.INFO),
            ("-vv", logging.DEBUG),
            ("-v -v", logging.DEBUG),
            ("-vvvvv", logging.DEBUG),
            ("--verbose --verbose", logging.DEBUG),
        ],
    )
    def test_given_log_levels_works_as_expected(
        self,
        mock_update: mock.MagicMock,
        verbosity: str,
        expected_level: int,
        tmp_logger: logging.Logger,
    ):
        input_path = "abc.skops"
        output_path = "def.skops"
        args = [input_path, "--output", output_path, *verbosity.split()]

        namespace, _ = _update.format_parser().parse_known_args(args)
        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=pathlib.Path(output_path),
            logger=tmp_logger,
        )
        assert tmp_logger.getEffectiveLevel() == expected_level
