import pickle as pkl

import numpy as np
import pytest

from skops.io._cli import _convert


class MockUnsafeType:
    def __init__(self):
        pass


class TestConvert:
    model_name = "a_model"

    @pytest.fixture
    def pkl_path(self, tmp_path):
        return tmp_path / f"{self.model_name}.pkl"

    @pytest.fixture
    def skops_path(self, tmp_path):
        return tmp_path / f"{self.model_name}.skops"

    @pytest.fixture
    def write_safe_file(self, pkl_path):
        obj = np.ndarray([1, 2, 3, 4])
        with open(pkl_path, "wb") as f:
            pkl.dump(obj, f)

    @pytest.fixture
    def write_unsafe_file(self, pkl_path):
        obj = MockUnsafeType()
        with open(pkl_path, "wb") as f:
            pkl.dump(obj, f)

    def test_base_case_works_as_expected(
        self, write_safe_file, pkl_path, skops_path, tmp_path
    ):
        _convert(pkl_path, tmp_path)
