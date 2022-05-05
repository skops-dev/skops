"""
This module contains utilities to push a model to the hub and pull from the
hub.
"""

from pathlib import Path
from typing import List, Union


def _validate_folder(path: Union[str, Path]):
    """Validate the contents of a folder.

    This function checks if the contents of a folder make a valid repo for a
    scikit-learn based repo on the HuggingFace Hub.

    Raises a ``TypeError`` if invalid.

    Parameters
    ----------
    path: str or Path
        The location of the repo.

    Returns
    -------
    None
    """
    pass


def init(
    *, model: Union[str, Path], requirements: List[str], destination: Union[str, Path]
):
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

    destination: str, or Path
        The path to a non-existing folder which is to be initializes.

    Returns
    -------
    None
    """
    pass


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


def push(*, repo_id: str, source: Union[str, Path], token: str = None):
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

    Returns
    -------
    None

    Notes
    -----
    This function raises a ``TypeError`` if the contents of the source folder
    do not make a valid HuggingFace Hub scikit-learn based repo.
    """
    pass
