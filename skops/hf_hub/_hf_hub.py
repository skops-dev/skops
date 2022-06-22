"""
This module contains utilities to push a model to the hub and pull from the
hub.
"""

import shutil
import sys
from pathlib import Path
from typing import List, Union

from huggingface_hub import HfApi

if sys.version_info >= (3, 11):
    try:
        import tomllib
    except ImportError:
        # Help users on older alphas
        import tomli as tomllib
else:
    import tomli as tomllib


def _validate_folder(path: Union[str, Path]):
    """Validate the contents of a folder.

    This function checks if the contents of a folder make a valid repo for a
    scikit-learn based repo on the HuggingFace Hub.

    A valid repository is one which is understood by the Hub as well as this
    library to run and use the model. Otherwise anything can be put as a model
    repository on the Hub and use it as a `git` and `git lfs` server.

    Raises a ``TypeError`` if invalid.

    Parameters
    ----------
    path: str or Path
        The location of the repo.

    Returns
    -------
    None
    """
    path = Path(path)
    if not path.is_dir():
        raise TypeError("The given path is not a directory.")

    config_path = path / "pyproject.toml"
    if not config_path.exists():
        raise TypeError("Configuration file `pyproject.toml` missing.")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    model_path = (
        config.get("hf_hub", {}).get("sklearn", {}).get("model", {}).get("file", None)
    )
    if not model_path:
        raise TypeError(
            "Model file not configured in the configuration file. It should be stored"
            " in the hf_hub.sklearn.model key."
        )

    if not (path / model_path).exists():
        raise TypeError(f"Model file {model_path} does not exist.")


def _create_config(*, model_path: str, requirements: List[str], dst: str):
    """Write the configuration into a `pyproject.toml` file.

    Parameters
    ----------
    model_path : str
        The relative path (from the repo root) to the model file.

    requirements : list of str
        A list of required packages. The versions are then extracted from the
        current environment.

    dst : str, or Path
        The path to an existing folder where the config file should be created.

    Returns
    -------
    None
    """
    with open(Path(dst) / "pyproject.toml", mode="w") as f:
        f.write("[hf_hub.sklearn.model]\n")
        f.write(f'file="{model_path}"\n')

        f.write("\n")

        f.write("[hf_hub.sklearn.environment]\n")

        for package in requirements:
            f.write(f"{package}\n")


def init(*, model: Union[str, Path], requirements: List[str], dst: Union[str, Path]):
    """Initialize a scikit-learn based HuggingFace repo.

    Given a model pickle and a set of required packages, this function
    initializes a folder to be a valid HuggingFace scikit-learn based repo.

    Parameters
    ----------
    model: str, or Path
        The path to a model pickle file.

    requirements: list of str
        A list of required packages. The versions are then extracted from the
        current environment.

    dst: str, or Path
        The path to a non-existing or empty folder which is to be initialized.

    Returns
    -------
    None
    """
    dst = Path(dst)
    if dst.exists() and next(dst.iterdir(), None):
        raise OSError("None-empty dst path already exists!")
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src=model, dst=dst)

    model_name = Path(model).name
    _create_config(model_path=model_name, requirements=requirements, dst=dst)


def update_env(*, path: Union[str, Path], requirements: List[str] = None):
    """Update the environment requirements of a repo.

    This function takes the path to the repo, and updates the requirements of
    running the scikit-learn based model in the repo.

    Parameters
    ----------
    path: str, or Path
        The path to an existing local repo.

    requirements: list of str, optional
        The list of required packages for the model. If none is passed, the
        list of existing requirements is used and their versions are updated.

    Returns
    -------
    None
    """
    pass


def push(
    *,
    repo_id: str,
    source: Union[str, Path],
    token: str = None,
    commit_message: str = None,
):
    """Pushes the contents of a model repo to HuggingFace Hub.

    This function validates the contents of the folder before pushing it to the
    Hub.

    Parameters
    ----------
    repo_id: str
        The ID of the destination repository in the form of ``OWNER/REPO_NAME``.

    source: str or Path
        A folder where the contents of the model repo are located.

    token: str, optional
        A token to push to the hub. If not provided, the user should be already
        logged in using ``huggingface-cli login``.

    commit_message: str, optional
        The commit message to be used when pushing to the repo.

    Returns
    -------
    None

    Notes
    -----
    This function raises a ``TypeError`` if the contents of the source folder
    do not make a valid HuggingFace Hub scikit-learn based repo.
    """
    client = HfApi()
    client.upload_folder(
        repo_id=repo_id,
        path_in_repo=".",
        folder_path=source,
        commit_message=commit_message,
        commit_description=None,
        token=token,
        repo_type=None,
        revision=None,
        create_pr=False,
    )
