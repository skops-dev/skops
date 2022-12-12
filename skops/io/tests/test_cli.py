import logging
import os.path
import pathlib
import pickle as pkl
from typing import Optional
from unittest import mock

import numpy as np
import pytest

import skops
from skops.io._cli import _convert


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
            pkl.dump(safe_obj, f)

    @pytest.fixture
    def write_unsafe_file(self, pkl_path, unsafe_obj):
        with open(pkl_path, "wb") as f:
            pkl.dump(unsafe_obj, f)

    def test_base_case_works_as_expected(
        self, pkl_path, tmp_path, skops_path, write_safe_file, safe_obj
    ):
        _convert(pkl_path, tmp_path)
        persisted_obj = skops.io.load(skops_path)
        assert np.array_equal(persisted_obj, safe_obj)

    def test_unsafe_case_works_as_expected(
        self, pkl_path, tmp_path, skops_path, write_unsafe_file, caplog
    ):
        caplog.set_level(logging.WARNING)
        _convert(pkl_path, tmp_path)
        assert not os.path.isfile(skops_path)

        # check logging has warned that an unsafe type was found
        assert MockUnsafeType.__name__ in caplog.text


class TestMainConvert:
    @staticmethod
    def assert_called_correctly(
        mock_convert: mock.MagicMock,
        paths: list,
        output_dir: Optional[pathlib.Path] = pathlib.Path.cwd(),
        trusted: Optional[bool] = False,
    ):
        assert mock_convert.call_count == len(paths)

        mock_convert.assert_has_calls(
            [
                mock.call(input_file=p, output_dir=output_dir, is_trusted=trusted)
                for p in paths
            ]
        )

    @mock.patch("skops.io._cli._convert")
    def test_base_works_as_expected(self, mock_convert: mock.MagicMock):
        args = [
            "123.pkl",
            "abc.pkl",
        ]

        skops.io._cli.main_convert(command_line_args=args)
        self.assert_called_correctly(mock_convert, args)

    @mock.patch("skops.io._cli._convert")
    @pytest.mark.parametrize("trusted_flag", ["-t", "--trusted"])
    def test_with_trusted_works_as_expected(
        self, mock_convert: mock.MagicMock, trusted_flag
    ):
        paths = ["abc.123", "234.567", "d/object_1.pkl"]
        args = paths + [trusted_flag]
        skops.io._cli.main_convert(command_line_args=args)
        self.assert_called_correctly(mock_convert, paths=paths, trusted=True)

    @mock.patch("skops.io._cli._convert")
    @pytest.mark.parametrize(
        "output_dir, expected_dir",
        [("a/b/c", pathlib.Path("a/b/c")), (None, pathlib.Path.cwd())],
    )
    def test_with_output_dir_works_as_expected(
        self, mock_convert: mock.MagicMock, output_dir, expected_dir
    ):
        paths = ["abc.123", "234.567", "d/object_1.pkl"]

        if output_dir is not None:
            args = paths + ["--output-dir", output_dir]
        else:
            args = paths

        skops.io._cli.main_convert(command_line_args=args)
        self.assert_called_correctly(mock_convert, paths=paths, output_dir=expected_dir)
