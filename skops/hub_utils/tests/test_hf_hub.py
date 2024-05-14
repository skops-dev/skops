import json
import os
import pickle
import re
import shutil
import tempfile
import warnings
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import sklearn
from flaky import flaky
from huggingface_hub import HfApi
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from skops import card
from skops.hub_utils import (
    add_files,
    get_config,
    get_model_output,
    get_requirements,
    init,
    push,
    update_env,
)
from skops.hub_utils._hf_hub import (
    _create_config,
    _get_column_names,
    _get_example_input_from_tabular_data,
    _get_example_input_from_text_data,
    _validate_folder,
)
from skops.hub_utils.tests.common import HF_HUB_TOKEN
from skops.io import dump

iris = load_iris(as_frame=True, return_X_y=False)
diabetes = load_diabetes(as_frame=True, return_X_y=False)

IS_SKLEARN_DEV_BUILD = "dev" in sklearn.__version__


@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory(prefix="skops-test-temp-path") as temp_path:
        yield temp_path


@pytest.fixture(scope="session")
def repo_path():
    with tempfile.TemporaryDirectory(prefix="skops-test-sample-repo") as repo_path:
        yield Path(repo_path)


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


def get_classifier():
    X, y = iris.data, iris.target
    clf = LogisticRegression(solver="newton-cg").fit(X, y)
    return clf


def get_regressor():
    X, y = diabetes.data, diabetes.target
    model = LinearRegression().fit(X, y)
    return model


@pytest.fixture(scope="session")
def classifier(repo_path, config_json):
    # Create a simple model file for the purpose of testing
    clf = get_classifier()
    config_path, file_format = config_json
    model_file = CONFIG[file_format]["sklearn"]["model"]["file"]
    path = repo_path / model_file

    try:
        if file_format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(clf, f)
        elif file_format == "skops":
            dump(clf, path)
        yield path
    finally:
        path.unlink(missing_ok=True)


CONFIG = {
    "pickle": {
        "sklearn": {
            "environment": ['scikit-learn="1.1.1"'],
            "model": {"file": "model.pickle"},
        }
    },
    "skops": {
        "sklearn": {
            "environment": ['scikit-learn="1.1.1"'],
            "model": {"file": "model.skops"},
        }
    },
}


@pytest.fixture(scope="session", params=["skops", "pickle"])
def config_json(repo_path, request):
    path = repo_path / "config.json"
    try:
        with open(path, "w") as f:
            json.dump(CONFIG[request.param], f)
        yield path, request.param
    finally:
        path.unlink(missing_ok=True)


def test_validate_format(classifier):
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)
    with pytest.raises(ValueError, match="Cannot determine the input file*"):
        init(
            model=classifier,
            requirements=["scikit-learn"],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
            model_format="dummy",
        )


def test_validate_folder(config_json):
    config_path, file_format = config_json
    _, file_path = tempfile.mkstemp()
    dir_path = tempfile.mkdtemp()
    with pytest.raises(TypeError, match="The given path is not a directory."):
        _validate_folder(path=file_path)

    with pytest.raises(TypeError, match="Configuration file `config.json` missing."):
        _validate_folder(path=dir_path)

    with open(Path(dir_path) / "config.json", "w") as f:
        json.dump(dict(), f)

    with pytest.raises(
        TypeError, match="Model file not configured in the configuration file."
    ):
        _validate_folder(path=dir_path)

    shutil.copy2(config_path, dir_path)
    model_file = CONFIG[file_format]["sklearn"]["model"]["file"]
    with pytest.raises(TypeError, match=f"Model file {model_file} does not exist."):
        _validate_folder(path=dir_path)

    (Path(dir_path) / model_file).touch()

    # this should now work w/o an error
    _validate_folder(path=dir_path)


@pytest.mark.parametrize(
    "data, task, expected_config",
    [
        (
            iris.data,
            "tabular-classification",
            {
                "sklearn": {
                    "columns": [
                        "petal length (cm)",
                        "petal width (cm)",
                        "sepal length (cm)",
                        "sepal width (cm)",
                    ],
                    "environment": ['scikit-learn="1.1.1"', "numpy"],
                    "example_input": {
                        "petal length (cm)": [1.4, 1.4, 1.3],
                        "petal width (cm)": [0.2, 0.2, 0.2],
                        "sepal length (cm)": [5.1, 4.9, 4.7],
                        "sepal width (cm)": [3.5, 3.0, 3.2],
                    },
                    "model": {"file": "model.pkl"},
                    "task": "tabular-classification",
                }
            },
        ),
        (
            ["test", "text", "problem", "random"],
            "text-classification",
            {
                "sklearn": {
                    "environment": ['scikit-learn="1.1.1"', "numpy"],
                    "example_input": {"data": ["test", "text", "problem"]},
                    "model": {"file": "model.pkl"},
                    "task": "text-classification",
                }
            },
        ),
    ],
)
def test_create_config(data, task, expected_config):
    dir_path = tempfile.mkdtemp()
    _create_config(
        model_path="model.pkl",
        requirements=['scikit-learn="1.1.1"', "numpy"],
        dst=dir_path,
        task=task,
        data=data,
    )

    with open(Path(dir_path) / "config.json") as f:
        config = json.load(f)
        for key in ["environment", "model", "task"]:
            assert config["sklearn"][key] == expected_config["sklearn"][key]

        keys = ["example_input"]
        if "tabular" in task:
            # text data doesn't introduce any "columns" in the configuration
            keys += ["columns"]
        for key in keys:
            assert sorted(config["sklearn"][key]) == sorted(
                expected_config["sklearn"][key]
            )


def test_create_config_invalid_text_data(temp_path):
    with pytest.raises(ValueError, match="The data needs to be a sequence of strings."):
        _create_config(
            model_path="model.pkl",
            requirements=['scikit-learn="1.1.1"', "numpy"],
            task="text-classification",
            data=[1, 2, 3],
            dst=temp_path,
        )


def test_atomic_init(classifier, temp_path):
    with pytest.raises(ValueError):
        # this fails since we're passing an invalid task.
        init(
            model=classifier,
            requirements=["scikit-learn"],
            dst=temp_path,
            task="tabular-classification",
            data="invalid",
        )

    # this passes even though the above init has failed once, on the same
    # destination path.
    init(
        model=classifier,
        requirements=["scikit-learn"],
        dst=temp_path,
        task="tabular-classification",
        data=iris.data,
    )


def test_init_invalid_task(classifier, temp_path):
    with pytest.raises(
        ValueError, match="Task invalid not supported. Supported tasks are"
    ):
        init(
            model=classifier,
            requirements=["scikit-learn"],
            dst=temp_path,
            task="invalid",
            data=iris.data,
        )


def test_init(classifier, config_json):
    config_path, file_format = config_json
    # create a temp directory and delete it, we just need a unique name.
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)

    version = metadata.version("scikit-learn")
    init(
        model=classifier,
        requirements=[f'scikit-learn="{version}"'],
        dst=dir_path,
        task="tabular-classification",
        data=iris.data,
    )
    _validate_folder(path=dir_path)

    # it should fail a second time since the folder is no longer empty.
    with pytest.raises(OSError, match="None-empty dst path already exists!"):
        init(
            model=classifier,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )


def test_init_no_warning_or_error(classifier, config_json):
    config_path, file_format = config_json
    # for the happy path, there should be no warning
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)
    version = metadata.version("scikit-learn")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        init(
            model=classifier,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )


def test_model_file_does_not_exist_raises(repo_path, config_json):
    config_path, file_format = config_json
    # when the model file does not exist, raise an OSError
    model_path = repo_path / "foobar.pickle"
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)
    version = metadata.version("scikit-learn")

    msg = re.escape(f"Model file '{model_path}' does not exist.")
    with pytest.raises(OSError, match=msg):
        init(
            model=model_path,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )
    model_path.unlink(missing_ok=True)


def test_init_empty_model_file_errors(repo_path, config_json):
    config_path, file_format = config_json
    # when model file is empty, warn users
    model_path = Path(repo_path / "foobar.pickle")
    model_path.touch()

    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)
    version = metadata.version("scikit-learn")

    with pytest.raises(
        RuntimeError, match=re.escape(f"Model file '{model_path}' is empty.")
    ):
        init(
            model=model_path,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )
    model_path.unlink(missing_ok=True)


def test_push_deprecation():
    with pytest.raises(Exception):
        with pytest.warns(FutureWarning, match="Creating repos on hf.co is subject"):
            push(repo_id="dummy", source=".")


@pytest.fixture
def repo_path_for_inference():
    # Create a separate path for test_inference so that the test does not have
    # any side-effect on existing tests
    with tempfile.TemporaryDirectory(prefix="skops-test-sample-repo") as repo_path:
        yield Path(repo_path)


@pytest.mark.network
@pytest.mark.inference
@pytest.mark.skipif(
    IS_SKLEARN_DEV_BUILD, reason="Inference tests cannot run with sklearn dev build"
)
@flaky(max_runs=3)
@pytest.mark.parametrize(
    "model_func, data, task",
    [
        (get_classifier, iris, "tabular-classification"),
        (get_regressor, diabetes, "tabular-regression"),
    ],
    ids=["classifier", "regressor"],
)
def test_inference(
    config_json,
    model_func,
    data,
    task,
    repo_path_for_inference,
    destination_path,
):
    # test inference backend for classifier and regressor models. This test can
    # take a lot of time and be flaky.
    config_path, file_format = config_json
    if file_format != "pickle":
        pytest.skip(
            f"Inference only supports pickle at the moment. Given format: {file_format}"
        )

    client = HfApi()
    repo_path = repo_path_for_inference
    model_file = CONFIG[file_format]["sklearn"]["model"]["file"]
    model = model_func()
    model_path = repo_path / model_file

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    version = metadata.version("scikit-learn")
    init(
        model=model_path,
        requirements=[f'scikit-learn="{version}"'],
        dst=destination_path,
        task=task,
        data=data.data,
    )

    # TODO: remove when card init at repo init is merged
    model_card = card.Card(
        model, metadata=card.metadata_from_config(Path(destination_path))
    )
    model_card.save(Path(destination_path) / "README.md")

    user = client.whoami(token=HF_HUB_TOKEN)["name"]
    repo_id = f"{user}/test-{uuid4()}"

    with pytest.warns(FutureWarning, match="Creating repos on hf.co is subject"):
        push(
            repo_id=repo_id,
            source=destination_path,
            token=HF_HUB_TOKEN,
            commit_message="test message",
            create_remote=True,
            # api-inference doesn't support private repos for community projects.
            private=False,
        )

    X_test = data.data.head(5)
    y_pred = model.predict(X_test)
    with pytest.warns(FutureWarning):
        output = get_model_output(repo_id, data=X_test, token=HF_HUB_TOKEN)

    # cleanup
    client.delete_repo(repo_id=repo_id, token=HF_HUB_TOKEN)
    model_path.unlink(missing_ok=True)

    assert np.allclose(output, y_pred)


def test_get_model_output_deprecated():
    with pytest.raises(Exception):
        with pytest.warns(FutureWarning, match="This feature is no longer free"):
            get_model_output("dummy", data=iris.data)


def test_get_config(repo_path, config_json):
    config_path, file_format = config_json
    config = get_config(repo_path)

    assert config == CONFIG[file_format]
    assert get_requirements(repo_path) == ['scikit-learn="1.1.1"']


def test_update_env(repo_path, config_json):
    # sanity check
    assert get_requirements(repo_path) == ['scikit-learn="1.1.1"']
    update_env(path=repo_path, requirements=['scikit-learn="1.1.2"'])
    assert get_requirements(repo_path) == ['scikit-learn="1.1.2"']


def test_get_example_input_from_tabular_data():
    with pytest.raises(
        ValueError,
        match=(
            "The data is not a pandas.DataFrame, a 2D numpy.ndarray or a "
            "list/tuple that can be converted to a 2D numpy.ndarray."
        ),
    ):
        _get_example_input_from_tabular_data("random")
    with pytest.raises(ValueError):
        _get_example_input_from_tabular_data(["a", "b", "c"])

    examples = _get_example_input_from_tabular_data(np.ones((5, 10)))
    # the result is a dictionary of column name: list of values
    assert len(examples) == 10
    assert len(examples["x0"]) == 3

    examples = _get_example_input_from_tabular_data(np.ones((5, 10)).tolist())
    # the result is a dictionary of column name: list of values
    assert len(examples) == 10
    assert len(examples["x0"]) == 3

    examples = _get_example_input_from_tabular_data(
        pd.DataFrame(np.ones((5, 10)), columns=[f"column{x}" for x in range(10)])
    )
    # the result is a dictionary of column name: list of values
    assert len(examples) == 10
    assert len(examples["column0"]) == 3


@pytest.mark.parametrize(
    "data, expected_length",
    [
        (["a", "b", "c", "d"], 3),
        (np.array(["a", "b", "c", "d"]), 3),
        (set(["a", "b", "c", "d"]), 3),
        (tuple(["a", "b", "c", "d"]), 3),
        (["a"], 1),
        ([], 0),
    ],
)
def test_get_example_input_from_text_data(data, expected_length):
    example_input = _get_example_input_from_text_data(data)
    assert len(example_input["data"]) == expected_length


@pytest.mark.parametrize("data", ["random", [1, 2, 3], 420])
def test_get_example_input_from_text_data_invalid_text_data(data):
    with pytest.raises(ValueError, match="The data needs to be a sequence of strings."):
        _get_example_input_from_text_data(data)


def test_get_column_names():
    with pytest.raises(
        ValueError,
        match=(
            "The data is not a pandas.DataFrame, a 2D numpy.ndarray or a "
            "list/tuple that can be converted to a 2D numpy.ndarray."
        ),
    ):
        _get_column_names("random")
    with pytest.raises(ValueError):
        _get_column_names(["a", "b", "c"])

    X_array = np.ones((5, 10), dtype=np.float32)
    expected_columns = [f"x{x}" for x in range(10)]
    assert _get_column_names(X_array) == expected_columns

    expected_columns = [f"column{x}" for x in range(10)]
    X_df = pd.DataFrame(X_array, columns=expected_columns)
    assert _get_column_names(X_df) == expected_columns


def test_get_example_input_from_tabular_data_pandas_not_installed(pandas_not_installed):
    # use pandas_not_installed fixture from conftest.py to pretend that pandas
    # is not installed and check that the function does not raise when pandas
    # import fails
    _get_example_input_from_tabular_data(np.ones((5, 10)))


def test_get_column_names_pandas_not_installed(pandas_not_installed):
    # use pandas_not_installed fixture from conftest.py to pretend that pandas
    # is not installed and check that the function does not raise when pandas
    # import fails
    _get_column_names(np.ones((5, 10)))


class TestAddFiles:
    @pytest.fixture
    def init_path(self, classifier, config_json):
        # create temporary directory
        dir_path = tempfile.mkdtemp()
        shutil.rmtree(dir_path)

        version = metadata.version("scikit-learn")
        init(
            model=classifier,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )
        yield dir_path

    @pytest.fixture
    def some_file_0(self, temp_path):
        filename = Path(temp_path) / "file0.txt"
        with open(filename, "w") as f:
            f.write("")
        yield filename

    @pytest.fixture
    def some_file_1(self, temp_path):
        filename = Path(temp_path) / "file1.txt"
        with open(filename, "w") as f:
            f.write("")
        yield filename

    def test_adding_one_file_path(self, init_path, some_file_0):
        add_files(some_file_0, dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)

    def test_adding_two_file_paths(self, init_path, some_file_0, some_file_1):
        add_files(some_file_0, some_file_1, dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)
        assert os.path.exists(Path(init_path) / some_file_1.name)

    def test_adding_one_file_str(self, init_path, some_file_0):
        add_files(str(some_file_0), dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)

    def test_adding_two_files_str(self, init_path, some_file_0, some_file_1):
        add_files(str(some_file_0), str(some_file_1), dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)
        assert os.path.exists(Path(init_path) / some_file_1.name)

    def test_adding_str_and_path(self, init_path, some_file_0, some_file_1):
        add_files(str(some_file_0), some_file_1, dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)
        assert os.path.exists(Path(init_path) / some_file_1.name)

    def test_dst_does_not_exist_raises(self, some_file_0):
        dst = tempfile.mkdtemp()
        shutil.rmtree(dst)
        msg = (
            rf"Could not find \'{re.escape(dst)}\', did you run "
            r"\'skops.hub_utils.init\' first\?"
        )
        with pytest.raises(FileNotFoundError, match=msg):
            add_files(some_file_0, dst=dst)

    def test_file_does_not_exist_raises(self, init_path, some_file_0):
        non_existing_file = "foobar.baz"
        msg = r"File \'foobar.baz\' could not be found."
        with pytest.raises(FileNotFoundError, match=msg):
            add_files(some_file_0, non_existing_file, dst=init_path)

    def test_adding_existing_file_works_if_exist_ok(self, init_path, some_file_0):
        add_files(some_file_0, dst=init_path)
        assert os.path.exists(Path(init_path) / some_file_0.name)
        add_files(some_file_0, dst=init_path, exist_ok=True)
        assert os.path.exists(Path(init_path) / some_file_0.name)

    def test_adding_existing_file_raises(self, init_path, some_file_0):
        # first time around no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            add_files(some_file_0, dst=init_path, exist_ok=False)

        msg = (
            f"File '{re.escape(some_file_0.name)}' already found "
            f"at '{re.escape(init_path)}'."
        )
        with pytest.raises(FileExistsError, match=msg):
            add_files(some_file_0, dst=init_path)
