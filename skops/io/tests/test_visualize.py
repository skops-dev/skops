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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.fixes import parse_version

import skops.io as sio
from skops.io import get_untrusted_types


def get_Tree_str(tree):
    """Get the string representation of a tree in Rich's markup syntax."""
    from rich.console import Console
    from rich.text import Text

    # force the color system to check that we have the right colors across
    # platforms and terminals
    console = Console(force_terminal=True, color_system="truecolor")
    with console.capture() as capture:
        console.print(tree)
    text = Text.from_ansi(capture.get())
    return text.markup


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
            ("clf", LogisticRegression(random_state=0, solver="saga")),
        ]).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
        # fmt: on
        return pipeline

    @pytest.mark.parametrize("show", ["all", "trusted", "untrusted"])
    def test_print_simple(self, simple, show, capsys):
        file = sio.dumps(simple)
        sio.visualize(file, show=show)

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
        sio.visualize(file)

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

        sio.visualize(file, sink=sink)
        nodes_self_unsafe = [node for node in nodes if not node.is_self_safe]
        nodes_unsafe = [node for node in nodes if not node.is_safe]

        assert len(nodes_self_unsafe) == 1
        assert nodes_self_unsafe[0].val == "test_visualize.unsafe_function"

        # it's not easy to test the number of indirectly unsafe nodes, because
        # it will depend on the nesting structure; we can only be sure that it's
        # more than 2, and one of them should be the FunctionTransformer
        assert len(nodes_unsafe) > 2
        assert any("FunctionTransformer" in node.val for node in nodes_unsafe)

    def test_all_nodes_trusted(self, pipeline, capsys):
        # The pipeline contains untrusted type(s), so we have to pass extra
        # trusted types
        file = sio.dumps(pipeline)
        sio.visualize(file, show="untrusted", trusted=get_untrusted_types(data=file))
        expected = "root: sklearn.pipeline.Pipeline"
        stdout, _ = capsys.readouterr()
        assert stdout.strip() == expected

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
        sio.visualize(file, sink=my_sink, **kwargs)

    def test_custom_tags(self, simple, capsys):
        class UnsafeType:
            pass

        simple.copy = UnsafeType

        file = sio.dumps(simple)
        sio.visualize(file, tag_safe="NICE", tag_unsafe="OHNO")
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
        # test that custom colors are used in node representation, requires rich
        # to work
        pytest.importorskip("rich")

        class UnsafeType:
            pass

        simple.copy = UnsafeType
        file = sio.dumps(simple)

        # Colors are not recorded by capsys, so we cannot use it and must mock
        # printing
        mock_print = Mock()
        with patch("rich.console.Console.print", mock_print):
            sio.visualize(
                file,
                color_safe="black",
                color_unsafe="cyan",
                color_child_unsafe="orange3",
            )

        mock_print.assert_called()

        calls = mock_print.call_args_list
        tree_repr = get_Tree_str(calls[0].args[0])
        # The root node is indirectly unsafe through child
        # orange3 is color(172)
        assert "root: [color(172)]sklearn.preprocessing._data.MinMaxScaler" in tree_repr
        # 'feature_range' is safe
        # black is color(0)
        assert "feature_range: [color(0)]builtins.tuple" in tree_repr
        # 'copy' is unsafe
        # cyan is color(6)
        assert "copy: [color(6)]test_visualize.UnsafeType [UNSAFE]" in tree_repr

    @pytest.mark.usefixtures("rich_not_installed")
    def test_no_colors_if_rich_not_installed(self, simple):
        # this test is similar to the previous one, except that we test that the
        # colors are *not* used if rich is not installed
        file = sio.dumps(simple)

        # don't use capsys, because it wouldn't capture the colors, thus need to
        # use mock
        mock_print = Mock()
        with patch("builtins.print", mock_print):
            sio.visualize(
                file,
                color_safe="black",
                color_unsafe="cyan",
                color_child_unsafe="orange3",
            )
        mock_print.assert_called()

        # check that none of the colors are being used
        for call in mock_print.call_args_list:
            # check for the color markers
            assert "[color(" not in call.args[0]

    def test_no_colors_if_use_colors_false(self, simple):
        # this test is similar to the previous one, except that we test that the
        # colors are *not* used, even if rich is installed, when passing
        # use_colors=False
        file = sio.dumps(simple)

        # don't use capsys, because it wouldn't capture the colors, thus need to
        # use mock
        mock_print = Mock()
        with patch("rich.console.Console.print", mock_print):
            sio.visualize(
                file,
                color_safe="black",
                color_unsafe="cyan",
                color_child_unsafe="orange3",
                use_colors=False,
            )
        mock_print.assert_called()

        # check that none of the colors are being used
        colors = ("black", "cyan", "orange3")
        for call in mock_print.call_args_list:
            for color in colors:
                assert color not in get_Tree_str(call.args[0])

    def test_from_file(self, simple, tmp_path, capsys):
        f_name = tmp_path / "estimator.skops"
        sio.dump(simple, f_name)
        sio.visualize(f_name)

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

    def test_long_bytes(self, capsys):
        obj = {
            "short_byte": b"abc",
            "long_byte": b"010203040506070809101112131415",
            "short_bytearray": bytearray(b"abc"),
            "long_bytearray": bytearray(b"010203040506070809101112131415"),
        }
        dumped = sio.dumps(obj)
        sio.visualize(dumped)

        expected = [
            "root: builtins.dict",
            "├── short_byte: b'abc'",
            "├── long_byte: b'01020304050...9101112131415'",
            "├── short_bytearray: bytearray(b'abc')",
            "└── long_bytearray: bytearray(b'01020304050...9101112131415')",
        ]
        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)

    @pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
    def test_decision_tree(self, cls, capsys):
        model = cls(random_state=0).fit([[0, 1], [2, 3], [4, 5]], [0, 1, 2])
        dumped = sio.dumps(model)
        sio.visualize(dumped)

        if isinstance(model, DecisionTreeClassifier):
            dt_criterion = "gini"
            dt_classes = [
                "    ├── classes_: numpy.ndarray",
                "    ├── n_classes_: numpy.int64",
            ]
        elif isinstance(model, DecisionTreeRegressor):
            dt_criterion = "squared_error"
            dt_classes = []

        expected = [
            "root: sklearn.tree._classes.{}".format(cls.__name__),
            "└── attrs: builtins.dict",
            '    ├── criterion: json-type("{}")'.format(dt_criterion),
            '    ├── splitter: json-type("best")',
            "    ├── max_depth: json-type(null)",
            "    ├── min_samples_split: json-type(2)",
            "    ├── min_samples_leaf: json-type(1)",
            "    ├── min_weight_fraction_leaf: json-type(0.0)",
            "    ├── max_features: json-type(null)",
            "    ├── max_leaf_nodes: json-type(null)",
            "    ├── random_state: json-type(0)",
            "    ├── min_impurity_decrease: json-type(0.0)",
            "    ├── class_weight: json-type(null)",
            "    ├── ccp_alpha: json-type(0.0)",
        ]
        if parse_version(sklearn.__version__) >= parse_version("1.4.0dev"):
            expected += ["    ├── monotonic_cst: json-type(null)"]
        expected += [
            "    ├── n_features_in_: json-type(2)",
            "    ├── n_outputs_: json-type(1)",
        ]
        expected += dt_classes
        expected += [
            "    ├── max_features_: json-type(2)",
            "    ├── tree_: sklearn.tree._tree.Tree",
            "    │   ├── attrs: builtins.dict",
            "    │   │   ├── max_depth: json-type(2)",
            "    │   │   ├── node_count: json-type(5)",
            "    │   │   ├── nodes: numpy.ndarray",
            "    │   │   └── values: numpy.ndarray",
            "    │   ├── args: builtins.tuple",
            "    │   │   ├── content: json-type(2)",
            "    │   │   ├── content: numpy.ndarray",
            "    │   │   └── content: json-type(1)",
            "    │   └── constructor: sklearn.tree._tree.Tree",
            '    └── _sklearn_version: json-type("{}")'.format(sklearn.__version__),
        ]
        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)
