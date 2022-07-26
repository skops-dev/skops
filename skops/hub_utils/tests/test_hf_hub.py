import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from flaky import flaky
from huggingface_hub import HfApi
from sklearn.linear_model import LogisticRegression

from skops.hub_utils import download, get_config, get_requirements, init, push
from skops.hub_utils._hf_hub import _create_config, _validate_folder
from skops.hub_utils.tests.common import HF_HUB_TOKEN
from skops.utils.fixes import metadata, path_unlink


@pytest.fixture(scope="session")
def repo_path():
    with tempfile.TemporaryDirectory(prefix="skops-test-sample-repo") as repo_path:
        yield Path(repo_path)


@pytest.fixture
def destination_path():
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        yield Path(dir_path)


@pytest.fixture(scope="session")
def model_pickle(repo_path):
    # Create a simple pickle file for the purpose of testing
    clf = LogisticRegression()
    clf.fit([[0, 1], [1, 0]], [0, 1])
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


def test_create_config():
    dir_path = tempfile.mkdtemp()
    _create_config(
        model_path="model.pkl",
        requirements=['scikit-learn="1.1.1"', "numpy"],
        dst=dir_path,
    )

    config_content = {
        "sklearn": {
            "environment": ['scikit-learn="1.1.1"', "numpy"],
            "model": {"file": "model.pkl"},
        }
    }

    with open(Path(dir_path) / "config.json") as f:
        config = json.load(f)
        assert config == config_content


def test_init(model_pickle, config_json):
    # create a temp directory and delete it, we just need a unique name.
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)

    version = metadata.version("scikit-learn")
    init(
        model=model_pickle,
        requirements=[f'scikit-learn="{version}"'],
        dst=dir_path,
    )
    _validate_folder(path=dir_path)

    # it should fail a second time since the folder is no longer empty.
    with pytest.raises(OSError, match="None-empty dst path already exists!"):
        init(
            model=model_pickle,
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
        )


@flaky(max_runs=3)
@pytest.mark.parametrize("explicit_create", [True, False])
def test_push_download(
    explicit_create, repo_path, destination_path, model_pickle, config_json
):
    client = HfApi()

    version = metadata.version("scikit-learn")
    init(
        model=model_pickle,
        requirements=[f'scikit-learn="{version}"'],
        dst=destination_path,
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
    for f_name in [model_pickle.name, config_json.name]:
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
