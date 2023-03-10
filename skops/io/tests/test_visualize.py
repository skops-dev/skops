import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)

import skops.io as sio
from skops.io._visualize import _check_should_print, visualize_tree


class TestVisualizeTree:
    @pytest.fixture
    def simple(self):
        return MinMaxScaler(feature_range=(-555, 123))

    @pytest.fixture
    def simple_file(self, simple, tmp_path):
        f_name = tmp_path / "estimator.skops"
        sio.dump(simple, f_name)
        return f_name

    @pytest.fixture
    def pipeline(self):
        # fmt: off
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("scaler", StandardScaler()),
                ("scaled-poly", Pipeline([
                    ("polys", FeatureUnion([
                        ("poly1", PolynomialFeatures()),
                        ("poly2", PolynomialFeatures(degree=3, include_bias=False))
                    ])),
                    ("square-root", FunctionTransformer(np.sqrt)),
                    ("scale", MinMaxScaler()),
                ])),
            ])),
            ("clf", LogisticRegression(random_state=0, solver="liblinear")),
        ]).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
        # fmt: on
        return pipeline

    @pytest.fixture
    def pipeline_file(self, pipeline, tmp_path):
        f_name = tmp_path / "estimator.skops"
        sio.dump(pipeline, f_name)
        return f_name

    @pytest.fixture
    def side_effect_and_contents(self):
        # This side effect collects the contents of what would normally be
        # printed. That way, we can test more precisely than just capturing
        # stdout and inspecting strings.
        contents = []

        def side_effect(node, key, level, trusted, show):
            should_print, _ = _check_should_print(node, trusted, show)
            if should_print:
                contents.append((node, key, level, trusted, show))

        return side_effect, contents

    @pytest.mark.parametrize("show", ["all", "trusted", "untrusted"])
    def test_print_simple(self, simple_file, show):
        visualize_tree(simple_file, show=show)

    @pytest.mark.parametrize(
        "show_tell", [("all", 8), ("trusted", 8), ("untrusted", 0)]
    )
    def test_inspect_simple(self, simple_file, side_effect_and_contents, show_tell):
        side_effect, contents = side_effect_and_contents
        show, expected_elements = show_tell
        visualize_tree(simple_file, sink=side_effect, show=show)
        assert len(contents) == expected_elements

    @pytest.mark.parametrize("show", ["all", "trusted", "untrusted"])
    def test_print_pipeline(self, pipeline_file, show):
        visualize_tree(pipeline_file, show=show)

    @pytest.mark.parametrize(
        "show_tell", [("all", 129), ("trusted", 110), ("untrusted", 19)]
    )
    def test_inspect_pipeline(self, pipeline_file, side_effect_and_contents, show_tell):
        side_effect, contents = side_effect_and_contents
        show, expected_elements = show_tell
        visualize_tree(pipeline_file, sink=side_effect, show=show)
        assert len(contents) == expected_elements
