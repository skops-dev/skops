import json
import os
import shutil
import tempfile
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import pytest
from huggingface_hub import HfApi

from skops.hf_hub import init, push
from skops.hf_hub._hf_hub import _create_config, _validate_folder

HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", None)


def _get_cwd():
    """Return the current working directory.

    Only works if we're using pytest.
    """
    return Path(os.getenv("PYTEST_CURRENT_TEST").split("::")[0]).parent


def test_validate_folder():
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

    example_file = _get_cwd() / "sample_repo/config.json"

    shutil.copy2(example_file, dir_path)
    with pytest.raises(TypeError, match="Model file model.pkl does not exist."):
        _validate_folder(path=dir_path)

    (Path(dir_path) / "model.pkl").touch()

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


def test_init():
    # create a temp directory and delete it, we just need a unique name.
    dir_path = tempfile.mkdtemp()
    shutil.rmtree(dir_path)

    version = metadata.version("scikit-learn")
    init(
        model=_get_cwd() / "sample_repo/model.pkl",
        requirements=[f'scikit-learn="{version}"'],
        dst=dir_path,
    )
    _validate_folder(path=dir_path)

    # it should fail a second time since the folder is no longer empty.
    with pytest.raises(OSError, match="None-empty dst path already exists!"):
        init(
            model=_get_cwd() / "sample_repo/model.pkl",
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
        )


def test_push():
    client = HfApi()
    with tempfile.TemporaryDirectory(prefix="skops-test") as dir_path:
        version = metadata.version("scikit-learn")
        init(
            model=_get_cwd() / "sample_repo/model.pkl",
            requirements=[f'scikit-learn="{version}"'],
            dst=dir_path,
        )

        user = client.whoami(token=HF_HUB_TOKEN)["name"]
        repo_id = f"{user}/test-{uuid4()}"
        client.create_repo(repo_id=repo_id, token=HF_HUB_TOKEN, repo_type="model")
        push(
            repo_id=repo_id,
            source=dir_path,
            token=HF_HUB_TOKEN,
            commit_message="test message",
        )

    files = client.list_repo_files(repo_id=repo_id, token=HF_HUB_TOKEN)
    for f_name in ["model.pkl", "config.json"]:
        assert f_name in files
