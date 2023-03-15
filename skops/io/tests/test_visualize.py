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
from skops.io._visualize import visualize_tree


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
        visualize_tree(file, show=show)

        # output is always the same for "all" and "trusted" because all nodes
        # are trusted
        expected = [
            "root: sklearn.preprocessing._data.MinMaxScaler",
            "└── attrs: builtins.dict",
            "    ├── feature_range: builtins.tuple",
            "    │   ├── content: json-type(-555)",
            "    │   └── content: json-type(123)",
            "    ├── copy: json-type(true)",
            "    ├── clip: json-type(false)",
            f'    └── _sklearn_version: json-type("{sklearn.__version__}")',
        ]
        if show == "untrusted":
            # since no untrusted, only show root
            expected = expected[:1]

        stdout, _ = capsys.readouterr()
        assert stdout.strip() == "\n".join(expected)

    def test_print_pipelien(self, pipeline, capsys):
        file = sio.dumps(pipeline)
        visualize_tree(file)

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
            f'    └── _sklearn_version: json-type("{sklearn.__version__}")',
        ]

        stdout, _ = capsys.readouterr()
        assert stdout.startswith("\n".join(expected_start))
        assert stdout.rstrip().endswith("\n".join(expected_end))

    def test_unsafe_nodes(self, pipeline):
        file = sio.dumps(pipeline)
        nodes = []

        def sink(nodes_iter, *args, **kwargs):
            nodes.extend(nodes_iter)

        visualize_tree(file, sink=sink)
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

    def test_custom_print_config(self):
        pass

    def test_from_file(self):
        pass

    def test_rich_not_installed(self):
        pass
