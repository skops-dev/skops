import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from huggingface_hub import HfApi
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from skops.hub_utils import download, get_config, get_requirements, init, push
from skops.hub_utils._hf_hub import (
    _create_config,
    _get_column_names,
    _get_example_input,
    _validate_folder,
)
from skops.hub_utils.tests.common import HF_HUB_TOKEN
from skops.utils.fixes import metadata, path_unlink

iris = load_iris(as_frame=True, return_X_y=False)


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


@pytest.fixture(scope="session")
def classifier_pickle(repo_path):
    # Create a simple pickle file for the purpose of testing
    X, y = iris.data, iris.target
    clf = LogisticRegression(solver="newton-cg")
    clf.fit(X, y)
    path = repo_path / "model.pickle"

    try:
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        yield path
    finally:
        path_unlink(path, missing_ok=True)


CONFIG = {
    "sklearn": {
        "environment": ['scikit-learn="1.1.1"'],
        "model": {"file": "model.pickle"},
    }
}


@pytest.fixture(scope="session")
def config_json(repo_path):
    path = repo_path / "config.json"
    try:
        with open(path, "w") as f:
            json.dump(CONFIG, f)
        yield path
    finally:
        path_unlink(path, missing_ok=True)


def test_validate_folder(config_json):
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

    shutil.copy2(config_json, dir_path)
    with pytest.raises(TypeError, match="Model file model.pickle does not exist."):
        _validate_folder(path=dir_path)

    (Path(dir_path) / "model.pickle").touch()

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
    with pytest.raises(ValueError, match="The data needs to be a list of strings."):
        _create_config(
            model_path="model.pkl",
            requirements=['scikit-learn="1.1.1"', "numpy"],
            task="text-classification",
            data=[1, 2, 3],
            dst=temp_path,
        )


def test_init_invalid_task(classifier_pickle, temp_path):
    with pytest.raises(
        ValueError, match="Task invalid not supported. Supported tasks are"
    ):
        init(
            model=classifier_pickle,
            requirements=["scikit-learn"],
            dst=temp_path,
            task="invalid",
            data=iris.data,
        )


def test_init(classifier_pickle, config_json):
    # create a temp directory and delete it, we just need a unique name.
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)

    version = metadata.version("scikit-learn")
    init(
        model=classifier_pickle,
        requirements=[f'scikit-learn="{version}"'],
        dst=dir_path,
        task="tabular-classification",
        data=iris.data,
    )
    _validate_folder(path=dir_path)

    # it should fail a second time since the folder is no longer empty.
    with pytest.raises(OSError, match="None-empty dst path already exists!"):
        init(
            model=classifier_pickle,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
            task="tabular-classification",
            data=iris.data,
        )


@pytest.mark.network
@flaky(max_runs=3)
@pytest.mark.parametrize("explicit_create", [True, False])
def test_push_download(
    explicit_create,
    repo_path,
    destination_path,
    classifier_pickle,
    config_json,
):
    client = HfApi()

    version = metadata.version("scikit-learn")
    init(
        model=classifier_pickle,
        requirements=[f'scikit-learn="{version}"'],
        dst=destination_path,
        task="tabular-classification",
        data=iris.data,
    )

    user = client.whoami(token=HF_HUB_TOKEN)["name"]
    repo_id = f"{user}/test-{uuid4()}"
    if explicit_create:
        client.create_repo(repo_id=repo_id, token=HF_HUB_TOKEN, repo_type="model")
    push(
        repo_id=repo_id,
        source=repo_path,
        token=HF_HUB_TOKEN,
        commit_message="test message",
        create_remote=True,
    )

    with pytest.raises(OSError, match="None-empty dst path already exists!"):
        download(repo_id=repo_id, dst=destination_path)

    files = client.list_repo_files(repo_id=repo_id, token=HF_HUB_TOKEN)
    for f_name in [classifier_pickle.name, config_json.name]:
        assert f_name in files

    try:
        with tempfile.TemporaryDirectory(prefix="skops-test") as dst:
            download(repo_id=repo_id, dst=dst, token=HF_HUB_TOKEN, keep_cache=False)
            copy_files = os.listdir(dst)
            assert set(copy_files) == set(files)
    finally:
        client.delete_repo(repo_id=repo_id, token=HF_HUB_TOKEN)


def test_get_config(repo_path):
    config = get_config(repo_path)
    assert config == CONFIG
    assert get_requirements(repo_path) == ['scikit-learn="1.1.1"']


def test_get_example_input():
    """Test the _get_example_input function."""
    with pytest.raises(
        ValueError, match="The data is not a pandas.DataFrame or a numpy.ndarray."
    ):
        _get_example_input(["a", "b", "c"])

    examples = _get_example_input(np.ones((5, 10)))
    # the result if a dictionary of column name: list of values
    assert len(examples) == 10
    assert len(examples["x0"]) == 3

    examples = _get_example_input(
        pd.DataFrame(np.ones((5, 10)), columns=[f"column{x}" for x in range(10)])
    )
    # the result if a dictionary of column name: list of values
    assert len(examples) == 10
    assert len(examples["column0"]) == 3


def test_get_column_names():
    with pytest.raises(
        ValueError, match="The data is not a pandas.DataFrame or a numpy.ndarray."
    ):
        _get_column_names(["a", "b", "c"])

    X_array = np.ones((5, 10), dtype=np.float32)
    expected_columns = [f"x{x}" for x in range(10)]
    assert _get_column_names(X_array) == expected_columns

    expected_columns = [f"column{x}" for x in range(10)]
    X_df = pd.DataFrame(X_array, columns=expected_columns)
    assert _get_column_names(X_df) == expected_columns
