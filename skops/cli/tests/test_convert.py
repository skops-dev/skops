import logging
import pathlib
import pickle
from unittest import mock

import numpy as np
import pytest

from skops.cli import _convert
from skops.io import get_untrusted_types, load


class MockUnsafeType:
    def __init__(self):
        pass


class TestConvert:
    model_name = "some_model_name"

    @pytest.fixture
    def safe_obj(self):
        return np.ndarray([1, 2, 3, 4])

    @pytest.fixture
    def unsafe_obj(self):
        return MockUnsafeType()

    @pytest.fixture
    def pkl_path(self, tmp_path):
        return tmp_path / f"{self.model_name}.pkl"

    @pytest.fixture
    def skops_path(self, tmp_path):
        return tmp_path / f"{self.model_name}.skops"

    @pytest.fixture
    def write_safe_file(self, pkl_path, safe_obj):
        with open(pkl_path, "wb") as f:
            pickle.dump(safe_obj, f)

    @pytest.fixture
    def write_unsafe_file(self, pkl_path, unsafe_obj):
        with open(pkl_path, "wb") as f:
            pickle.dump(unsafe_obj, f)

    def test_base_case_works_as_expected(
        self, pkl_path, tmp_path, skops_path, write_safe_file, safe_obj, caplog
    ):
        mock_logger = mock.MagicMock()
        _convert._convert_file(pkl_path, skops_path, logger=mock_logger)
        persisted_obj = load(skops_path)
        assert np.array_equal(persisted_obj, safe_obj)

        # Check no warnings or errors raised
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_unsafe_case_works_as_expected(
        self, pkl_path, tmp_path, skops_path, write_unsafe_file, caplog
    ):
        caplog.set_level(logging.WARNING)
        _convert._convert_file(pkl_path, skops_path)
        persisted_obj = load(skops_path, trusted=get_untrusted_types(file=skops_path))

        assert isinstance(persisted_obj, MockUnsafeType)

        # check logging has warned that an unsafe type was found
        assert MockUnsafeType.__name__ in caplog.text


class TestMain:
    @staticmethod
    def assert_called_correctly(
        mock_convert: mock.MagicMock,
        path,
        output_file=None,
    ):
        if not output_file:
            output_file = pathlib.Path.cwd() / f"{pathlib.Path(path).stem}.skops"
        mock_convert.assert_called_once_with(input_file=path, output_file=output_file)

    @mock.patch("skops.cli._convert._convert_file")
    def test_base_works_as_expected(self, mock_convert: mock.MagicMock):
        path = "123.pkl"
        namespace, _ = _convert.format_parser().parse_known_args([path])

        _convert.main(namespace)
        self.assert_called_correctly(mock_convert, path)

    @mock.patch("skops.cli._convert._convert_file")
    @pytest.mark.parametrize(
        "input_path, output_file, expected_path",
        [
            ("abc.123", "some/file/path.out", "some/file/path.out"),
            ("abc.123", None, pathlib.Path.cwd() / "abc.skops"),
        ],
        ids=["Given an output path", "No output path"],
    )
    def test_with_output_dir_works_as_expected(
        self, mock_convert: mock.MagicMock, input_path, output_file, expected_path
    ):
        if output_file is not None:
            args = [input_path, "--output", output_file]
        else:
            args = [input_path]

        namespace, _ = _convert.format_parser().parse_known_args(args)

        _convert.main(namespace)
        self.assert_called_correctly(
            mock_convert, path=input_path, output_file=expected_path
        )

    @mock.patch("skops.cli._convert._convert_file")
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
        self, mock_convert: mock.MagicMock, verbosity, expected_level, caplog
    ):
        input_path = "abc.def"
        output_path = "bde.skops"
        args = [input_path, "--output", output_path, verbosity.split()]

        namespace, _ = _convert.format_parser().parse_known_args(args)

        _convert.main(namespace)
        self.assert_called_correctly(
            mock_convert, path=input_path, output_file=output_path
        )

        assert caplog.at_level(expected_level)
