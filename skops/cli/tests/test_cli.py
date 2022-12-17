import logging
import pathlib
import pickle as pkl
from typing import Optional
from unittest import mock

import numpy as np
import pytest

from skops import cli
from skops.io import load


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
        cli._convert_file(pkl_path, skops_path)
        persisted_obj = load(skops_path)
        assert np.array_equal(persisted_obj, safe_obj)

    def test_unsafe_case_works_as_expected(
        self, pkl_path, tmp_path, skops_path, write_unsafe_file, caplog
    ):
        caplog.set_level(logging.WARNING)
        cli._convert_file(pkl_path, skops_path)
        persisted_obj = load(skops_path, trusted=True)

        assert isinstance(persisted_obj, MockUnsafeType)

        # check logging has warned that an unsafe type was found
        assert MockUnsafeType.__name__ in caplog.text


class TestMainConvert:
    @staticmethod
    def assert_called_correctly(
        mock_convert: mock.MagicMock,
        paths: list,
        output_files: Optional[list] = None,
    ):
        if not output_files:
            output_files = [
                pathlib.Path.cwd() / f"{pathlib.Path(p).stem}.skops" for p in paths
            ]
        assert mock_convert.call_count == len(paths)
        mock_convert.assert_has_calls(
            [
                mock.call(input_file=paths[i], output_file=output_files[i])
                for i in range(len(paths))
            ]
        )

    @mock.patch("skops.cli._convert._convert_file")
    def test_base_works_as_expected(self, mock_convert: mock.MagicMock):
        args = [
            "123.pkl",
            "abc.pkl",
        ]

        cli.main_convert(command_line_args=args)
        self.assert_called_correctly(mock_convert, args)

    @mock.patch("skops.cli._convert._convert_file")
    @pytest.mark.parametrize(
        "input_paths, output_files, expected_paths",
        [
            (["abc.123"], ["a/b/c"], ["a/b/c"]),
            (["abc.123"], None, [pathlib.Path.cwd() / "abc.skops"]),
        ],
        ids=["Given an output path", "No output path"],
    )
    def test_with_output_dir_works_as_expected(
        self, mock_convert: mock.MagicMock, input_paths, output_files, expected_paths
    ):
        if output_files is not None:
            args = input_paths + ["--output-files"] + output_files
        else:
            args = input_paths

        cli.main_convert(command_line_args=args)
        self.assert_called_correctly(
            mock_convert, paths=input_paths, output_files=expected_paths
        )

    @mock.patch("skops.cli._convert._convert_file")
    @pytest.mark.parametrize(
        "input_paths, output_files, expected_paths",
        [
            (
                ["model_a.pkl", "model_b.pkl"],
                ["a.skops", "b.skops"],
                ["a.skops", "b.skops"],
            ),
            (
                ["model_a.pkl", "model_b.pkl", "model_c.pkl"],
                ["a.skops", "b.skops"],
                ["a.skops", "b.skops", pathlib.Path.cwd() / "model_c.skops"],
            ),
        ],
        ids=["With enough output paths", "With only some output paths"],
    )
    def test_for_multiple_inputs_and_outputs_works_as_expected(
        self, mock_convert: mock.MagicMock, input_paths, output_files, expected_paths
    ):
        args = input_paths + ["--output-files"] + output_files
        cli.main_convert(args)

        self.assert_called_correctly(mock_convert, input_paths, expected_paths)
