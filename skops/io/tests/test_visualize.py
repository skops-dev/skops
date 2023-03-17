"""Tests for skops.io.visualize"""

from unittest.mock import Mock, patch

import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)

import skops.io as sio


class TestVisualizeTree:
    @pytest.fixture
    def simple(self):
        return MinMaxScaler(feature_range=(-555, 123))

    @pytest.fixture
    def pipeline(self):
        def unsafe_function(x):
            return x

        # fmt: off
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("scaler", StandardScaler()),
                ("scaled-poly", Pipeline([
                    ("polys", FeatureUnion([
                        ("poly1", PolynomialFeatures()),
                        ("poly2", PolynomialFeatures(degree=3, include_bias=False))
                    ])),
                    ("square-root", FunctionTransformer(unsafe_function)),
                    ("scale", MinMaxScaler()),
                ])),
            ])),
            ("clf", LogisticRegression(random_state=0, solver="liblinear")),
        ]).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
        # fmt: on
        return pipeline

    @pytest.mark.parametrize("show", ["all", "trusted", "untrusted"])
    def test_print_simple(self, simple, show, capsys):
        file = sio.dumps(simple)
        sio.visualize_tree(file, show=show)

        # Output is the same for "all" and "trusted" because all nodes are
        # trusted. Colors are not recorded by capsys.
        expected = [
            "root: sklearn.preprocessing._data.MinMaxScaler",
            "└── attrs: builtins.dict",
            "    ├── feature_range: builtins.tuple",
            "    │   ├── content: json-type(-555)",
            "    │   └── content: json-type(123)",
            "    ├── copy: json-type(true)",
            "    ├── clip: json-type(false)",
            '    └── _sklearn_version: json-type("{}")'.format(sklearn.__version__),
        ]
        if show == "untrusted":
            # since no untrusted, only show root
            expected = expected[:1]

        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)

    def test_print_pipeline(self, pipeline, capsys):
        file = sio.dumps(pipeline)
        sio.visualize_tree(file)

        # no point in checking the whole output with > 120 lines
        expected_start = [
            "root: sklearn.pipeline.Pipeline",
            "└── attrs: builtins.dict",
            "    ├── steps: builtins.list",
            "    │   ├── content: builtins.tuple",
            '    │   │   ├── content: json-type("features")',
        ]
        expected_end = [
            "    ├── memory: json-type(null)",
            "    ├── verbose: json-type(false)",
            '    └── _sklearn_version: json-type("{}")'.format(sklearn.__version__),
        ]

        stdout, _ = capsys.readouterr()
        assert stdout.startswith("\n".join(expected_start))
        assert stdout.rstrip().endswith("\n".join(expected_end))

    def test_unsafe_nodes(self, pipeline):
        file = sio.dumps(pipeline)
        nodes = []

        def sink(nodes_iter, *args, **kwargs):
            nodes.extend(nodes_iter)

        sio.visualize_tree(file, sink=sink)
        nodes_self_unsafe = [node for node in nodes if not node.is_self_safe]
        nodes_unsafe = [node for node in nodes if not node.is_safe]

        # there are currently 2 unsafe nodes, a numpy int and the custom
        # functions. The former might be considered safe in the future, in which
        # case this test needs to be changed.
        assert len(nodes_self_unsafe) == 2
        assert nodes_self_unsafe[0].val == "numpy.int64"
        assert nodes_self_unsafe[1].val == "test_visualize.function"

        # it's not easy to test the number of indirectly unsafe nodes, because
        # it will depend on the nesting structure; we can only be sure that it's
        # more than 2, and one of them should be the FunctionTransformer
        assert len(nodes_unsafe) > 2
        assert any("FunctionTransformer" in node.val for node in nodes_unsafe)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"use_colors": False},
            {"tag_unsafe": "<careful>", "color_unsafe": "blue"},
        ],
    )
    def test_custom_print_config_passed_to_sink(self, simple, kwargs):
        # check that arguments are passed to sink
        def my_sink(nodes_iter, show, **sink_kwargs):
            for key, val in kwargs.items():
                assert sink_kwargs[key] == val

        file = sio.dumps(simple)
        sio.visualize_tree(file, sink=my_sink, **kwargs)

    def test_custom_tags(self, simple, capsys):
        class UnsafeType:
            pass

        simple.copy = UnsafeType

        file = sio.dumps(simple)
        sio.visualize_tree(file, tag_safe="NICE", tag_unsafe="OHNO")
        expected = [
            "root: sklearn.preprocessing._data.MinMaxScaler NICE",
            "└── attrs: builtins.dict NICE",
            "    ├── feature_range: builtins.tuple NICE",
            "    │   ├── content: json-type(-555) NICE",
            "    │   └── content: json-type(123) NICE",
            "    ├── copy: test_visualize.UnsafeType OHNO",
            "    ├── clip: json-type(false) NICE",
            '    └── _sklearn_version: json-type("{}") NICE'.format(
                sklearn.__version__
            ),
        ]

        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)

    def test_custom_colors(self, simple):
        # Colors are not recorded by capsys, so we cannot use it
        class UnsafeType:
            pass

        simple.copy = UnsafeType

        file = sio.dumps(simple)
        mock_print = Mock()
        with patch("rich.print", mock_print):
            sio.visualize_tree(
                file,
                color_safe="black",
                color_unsafe="cyan",
                color_child_unsafe="orange3",
            )

        assert mock_print.call_count == 1

        tree = mock_print.call_args_list[0].args[0]
        # The root node is indirectly unsafe through child
        assert "[orange3]" in tree.label
        # feature_range is safe
        assert "[black]" in tree.children[0].children[0].label
        # copy is unsafe
        assert "[cyan]" in tree.children[0].children[1].label

    def test_from_file(self, simple, tmp_path, capsys):
        f_name = tmp_path / "estimator.skops"
        sio.dump(simple, f_name)
        sio.visualize_tree(f_name)

        expected = [
            "root: sklearn.preprocessing._data.MinMaxScaler",
            "└── attrs: builtins.dict",
            "    ├── feature_range: builtins.tuple",
            "    │   ├── content: json-type(-555)",
            "    │   └── content: json-type(123)",
            "    ├── copy: json-type(true)",
            "    ├── clip: json-type(false)",
            '    └── _sklearn_version: json-type("{}")'.format(sklearn.__version__),
        ]
        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)
