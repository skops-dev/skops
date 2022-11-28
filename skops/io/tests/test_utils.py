import numpy as np
import pytest
import scipy
import sklearn.tree

from skops.io._utils import get_type_name, get_type_paths


class UserDefinedClass:
    pass


class TestGetTypeName:
    @pytest.mark.parametrize(
        "input_type, expected_output",
        [
            # Built-In types
            (list, "builtins.list"),
            (set, "builtins.set"),
            (dict, "builtins.dict"),
            (str, "builtins.str"),
            # Numpy types
            (np.ndarray, "numpy.ndarray"),
            (np.ma.MaskedArray, "numpy.ma.core.MaskedArray"),
            # SciPy types
            (scipy.fft.fft, "scipy.fft._basic.fft"),
            # SKlearn types
            (
                sklearn.linear_model.HuberRegressor,
                "sklearn.linear_model._huber.HuberRegressor",
            ),
            # User defined types
            (UserDefinedClass, "test_utils.UserDefinedClass"),
        ],
    )
    def test_for_input_types_returns_as_expected(self, input_type, expected_output):
        assert get_type_name(input_type) == expected_output


class TestConvertTypesToStrings:
    @pytest.mark.parametrize(
        "input_list, output_list",
        [
            (["builtins.str", "builtins.list"], ["builtins.str", "builtins.list"]),
            ([str, list], ["builtins.str", "builtins.list"]),
            ([np.ndarray, "builtins.str"], ["numpy.ndarray", "builtins.str"]),
        ],
        ids=["as strings", "as types", "mixed"],
    )
    def test_for_normal_input_lists_returns_as_expected(self, input_list, output_list):
        assert get_type_paths(input_list) == output_list

    @pytest.mark.parametrize(
        "input, output",
        [(None, []), (int, ["builtins.int"]), ((list,), ["builtins.list"]), ([], [])],
        ids=["None", "single int type", "list in tuple", "empty list"],
    )
    def test_for_edge_cases_handles_as_expected(self, input, output):
        assert get_type_paths(input) == output
