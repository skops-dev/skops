"""
This module contains utilities to push a model to the hub and pull from the
hub.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, List, Literal, MutableMapping, Optional, Sequence, Union

import numpy as np
from huggingface_hub import HfApi, InferenceClient, snapshot_download
from sklearn.utils import check_array

SUPPORTED_TASKS = [
    "tabular-classification",
    "tabular-regression",
    "text-classification",
    "text-regression",
]


def _validate_folder(path: Union[str, Path]) -> None:
    """Validate the contents of a folder.

    This function checks if the contents of a folder make a valid repo for a
    scikit-learn based repo on the Hugging Face Hub.

    A valid repository is one which is understood by the Hub as well as this
    library to run and use the model. Otherwise anything can be put as a model
    repository on the Hub and use it as a `git` and `git lfs` server.

    Raises a ``TypeError`` if invalid.

    Parameters
    ----------
    path: str or Path
        The location of the repo.

    Raises
    ------
    TypeError
        Raised when the passed path is invalid.

    Returns
    -------
    None
    """
    path = Path(path)
    if not path.is_dir():
        raise TypeError("The given path is not a directory.")

    config_path = path / "config.json"
    if not config_path.exists():
        raise TypeError("Configuration file `config.json` missing.")

    with open(config_path, "r") as f:
        config = json.load(f)

    model_path = config.get("sklearn", {}).get("model", {}).get("file", None)
    if not model_path:
        raise TypeError(
            "Model file not configured in the configuration file. It should be stored"
            " in the hf_hub.sklearn.model key."
        )

    if not (path / model_path).exists():
        raise TypeError(f"Model file {model_path} does not exist.")


def _get_example_input_from_tabular_data(data):
    """Returns the example input of a model for a tabular task.

    The input is converted into a dictionary which is then stored in the config
    file.

    Parameters
    ----------
    data: array-like
        The input needs to be either a ``pandas.DataFrame``, a 2D
        ``numpy.ndarray`` or a list/tuple that can be converted to a 2D
        ``numpy.ndarray``. The first 3 rows are used as example input.

    Returns
    -------
    example_input: dict of lists
        The example input of the model as accepted by Hugging Face's backend.
    """
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return {x: data[x][:3].to_list() for x in data.columns}
    except ImportError:
        # pandas is not installed, the data cannot be a pandas DataFrame
        pass

    # here we convert the first three rows of `data` to a dict of lists
    # to be stored in the config file
    if isinstance(data, (np.ndarray, list, tuple)):
        data_slice = data[:3]
        # This will raise a ValueError if the array is not 2D
        data_slice_array = check_array(data_slice, ensure_2d=True)
        return {
            f"x{x}": data_slice_array[:, x].tolist()
            for x in range(data_slice_array.shape[1])
        }

    raise ValueError(
        "The data is not a pandas.DataFrame, a 2D numpy.ndarray or a "
        "list/tuple that can be converted to a 2D numpy.ndarray."
    )


def _get_example_input_from_text_data(data: Sequence[str]):
    """Returns the example input of a model for a text task.

    The input is converted into a dictionary which is then stored in the config
    file.

    Parameters
    ----------
    data: Sequence[str]
        A sequence of strings. The first 3 elements are used as example input.

    Returns
    -------
    example_input: dict of lists
        The example input of the model as accepted by Hugging Face's backend.
    """

    def _head(data, n):
        is_data_subscriptable = hasattr(data, "__getitem__")
        if is_data_subscriptable:
            return data[:n]

        return list(itertools.islice(data, n))

    def _is_sequence_of_strings(data):
        return not isinstance(data, str) and all(isinstance(x, str) for x in data)

    error_message = "The data needs to be a sequence of strings."
    try:
        data_head = _head(data, n=3)
        if _is_sequence_of_strings(data_head):
            return {"data": data_head}
        else:
            raise ValueError(error_message)
    except TypeError as e:
        raise ValueError(error_message) from e


def _get_column_names(data):
    """Returns the column names of the input.

    If data is not a ``pandas.DataFrame``, column names are assumed to be
    ``x0`` to ``xn-1``, where ``n`` is the number of columns.

    Parameters
    ----------
    data: array-like
        The data whose columns names are to be returned. Must be a
        ``pandas.DataFrame``, a 2D ``numpy.ndarray`` or a list/tuple that can
        be converted to a 2D ``numpy.ndarray``.

    Returns
    -------
    columns: list of strings
        A list of strings. Each string is a column name.
    """
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return list(data.columns)
    except ImportError:
        # pandas is not installed, the data cannot be a pandas DataFrame
        pass

    # TODO: this is going to fail for Structured Arrays. We can add support for
    # them later if we see need for it.
    if isinstance(data, (np.ndarray, list, tuple)):
        # This will raise a ValueError if the array is not 2D
        data_array = check_array(data, ensure_2d=True)
        return [f"x{x}" for x in range(data_array.shape[1])]

    raise ValueError(
        "The data is not a pandas.DataFrame, a 2D numpy.ndarray or a "
        "list/tuple that can be converted to a 2D numpy.ndarray."
    )


def _create_config(
    *,
    model_path: Union[str, Path],
    requirements: List[str],
    dst: Union[str, Path],
    task: Literal[
        "tabular-classification",
        "tabular-regression",
        "text-classification",
        "text-regression",
    ],
    data,
    model_format: Literal[
        "skops",
        "pickle",
        "auto",
    ] = "auto",
) -> None:
    """Write the configuration into a ``config.json`` file.

    Parameters
    ----------
    model_path : str, or Path
        The relative path (from the repo root) to the model file.

    requirements : list of str
        A list of required packages. The versions are then extracted from the
        current environment.

    dst : str, or Path
        The path to an existing folder where the config file should be created.

    task: "tabular-classification", "tabular-regression",
    "text-classification", /
            or "text-regression"
        The task of the model, which determines the input and output type of
        the model. It can be one of: ``tabular-classification``,
        ``tabular-regression``, ``text-classification``, ``text-regression``.

    data: array-like or sequence
        The input to the model. This is used for two purposes:

            1. Save an example input to the model, which is used by
               HuggingFace's backend and shown in the widget of the model's
               page.
            2. Store the columns and their order of the input, which is used by
               HuggingFace's backend to pass the data in the right form to the
               model.

        The first 3 input values are used as example inputs. If the task is
        ``tabular-classification`` or ``tabular-regression``, then data is
        expected to be an array-like. Otherwise, it is expected to be an
        sequence of strings.

    model_format: str (default="auto")
        The format used to persist the model. Can be ``"auto"``, ``"skops"``
        or ``"pickle"``. Defaults to ``"auto"``, which would mean:

        - ``"pickle"`` if the extension is one of ``{".pickle", ".pkl", ".joblib"}``
        - ``"skops"`` if the extension is ``".skops"``
    """

    # so that we don't have to explicitly add keys and they're added as a
    # dictionary if they are not found
    # see: https://stackoverflow.com/a/13151294/2536294
    def recursively_default_dict() -> MutableMapping:
        return collections.defaultdict(recursively_default_dict)

    if model_format == "auto":
        extension = Path(model_path).suffix
        if extension in [".pkl", ".pickle", ".joblib"]:
            model_format = "pickle"
        elif extension == ".skops":
            model_format = "skops"
    if model_format not in ["skops", "pickle"]:
        raise ValueError(
            "Cannot determine the input file format. Please indicate the format using"
            " the `model_format` argument."
        )

    config = recursively_default_dict()
    config["sklearn"]["model"]["file"] = str(model_path)
    config["sklearn"]["environment"] = requirements
    config["sklearn"]["task"] = task
    config["sklearn"]["model_format"] = model_format

    if "tabular" in task:
        config["sklearn"]["example_input"] = _get_example_input_from_tabular_data(data)
        config["sklearn"]["columns"] = _get_column_names(data)
    elif "text" in task:
        config["sklearn"]["example_input"] = _get_example_input_from_text_data(data)

    dump_json(Path(dst) / "config.json", config)


def _check_model_file(path: str | Path) -> Path:
    """Perform sanity checks on the model file

    Parameters
    ----------
    path : str or Path
      The model path

    Returns
    -------
    path : Path
      The model path as a  ``pathlib.Path``.

    Raises
    ------
    OSError
      If the model file does not exist.
    """
    if not os.path.exists(path):
        raise OSError(f"Model file '{path}' does not exist.")

    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Model file '{path}' is empty.")

    return Path(path)


def init(
    *,
    model: Union[str, Path],
    requirements: List[str],
    dst: Union[str, Path],
    task: Literal[
        "tabular-classification",
        "tabular-regression",
        "text-classification",
        "text-regression",
    ],
    data,
    model_format: Literal[
        "skops",
        "pickle",
        "auto",
    ] = "auto",
) -> None:
    """Initialize a scikit-learn based Hugging Face repo.

    Given a pickled model and a set of required packages, this function
    initializes a folder to be a valid Hugging Face scikit-learn based repo.

    Parameters
    ----------
    model: str, or Path
        The path to a model pickle file.

    requirements: list of str
        A list of required packages. The versions are then extracted from the
        current environment.

    dst: str, or Path
        The path to a non-existing or empty folder which is to be initialized.

    task: str
        The task of the model, which determines the input and output type of
        the model. It can be one of: ``tabular-classification``,
        ``tabular-regression``, ``text-classification``, ``text-regression``.

    data: array-like
        The input to the model. This is used for two purposes:

            1. Save an example input to the model, which is used by
               HuggingFace's backend and shown in the widget of the model's
               page.
            2. Store the columns and their order of the input, which is used by
               HuggingFace's backend to pass the data in the right form to the
               model.

        The first 3 input values are used as example inputs.

        If ``task`` is ``"tabular-classification"`` or ``"tabular-regression"``,
        the data needs to be a :class:`pandas.DataFrame` or a
        :class:`numpy.ndarray`. If ``task`` is ``"text-classification"`` or
        ``"text-regression"``, the data needs to be a ``list`` of strings.

    model_format: str (default="auto")
        The format the model was persisted in. Can be ``"auto"``, ``"skops"``
        or ``"pickle"``. Defaults to ``"auto"`` that relies on file extension.
    """
    dst = Path(dst)
    if dst.exists() and bool(next(dst.iterdir(), None)):
        raise OSError("None-empty dst path already exists!")

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Task {task} not supported. Supported tasks are: {SUPPORTED_TASKS}"
        )

    model = _check_model_file(model)

    dst.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(src=model, dst=dst)

        model_name = model.name
        _create_config(
            model_path=model_name,
            requirements=requirements,
            dst=dst,
            task=task,
            data=data,
            model_format=model_format,
        )
    except Exception:
        shutil.rmtree(dst)
        raise


def add_files(*files: str | Path, dst: str | Path, exist_ok: bool = False) -> None:
    """Add files to initialized repo.

    After having called :func:`.hub_utils.init`, use this function to add
    arbitrary files to be uploaded in addition to the model and model card.

    In particular, it can be useful to upload the script itself that produces
    those artifacts by calling ``hub_utils.add_files([__file__], dst=...)``.

    Parameters
    ----------
    *files : str or Path
        The files to be added.

    dst : str or Path
        Path to the initialized repo, same as used during
        :func:`.hub_utils.init`.

    exist_ok : bool (default=False)
        Whether it's okay or not to add a file that already exists. If
        ``True``, override the files, otherwise raise a ``FileExistsError``.

    Raises
    ------
    FileNotFoundError
        When the target folder or the files to be added are not found.

    FileExistsError
        When a file is added that already exists at the target location and
        ``exist_ok=False``.

    """
    dst = Path(dst)
    # check dst exists
    if not dst.exists():
        msg = f"Could not find '{dst}', did you run 'skops.hub_utils.init' first?"
        raise FileNotFoundError(msg)

    src_files = [Path(file) for file in files]
    # check that source files exist
    for file in src_files:
        if not file.exists():
            msg = f"File '{file}' could not be found."
            raise FileNotFoundError(msg)

    dst_files = [dst / Path(file).name for file in files]
    for src_file, dst_file in zip(src_files, dst_files):
        # check if target file already exists
        if dst_file.exists() and not exist_ok:
            msg = f"File '{src_file.name}' already found at '{dst}'."
            raise FileExistsError(msg)

        shutil.copy2(src_file, dst_file)


def dump_json(path, content):
    with open(Path(path), mode="w") as f:
        json.dump(content, f, sort_keys=True, indent=4)


def update_env(
    *, path: Union[str, Path], requirements: Union[List[str], None] = None
) -> None:
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

    """

    with open(Path(path) / "config.json") as f:
        config = json.load(f)

    config["sklearn"]["environment"] = requirements

    dump_json(Path(path) / "config.json", config)


def push(
    *,
    repo_id: str,
    source: Union[str, Path],
    token: str | None = None,
    commit_message: str | None = None,
    create_remote: bool = False,
    private: bool | None = None,
) -> None:
    """Pushes the contents of a model repo to Hugging Face Hub.

    This function validates the contents of the folder before pushing it to the
    Hub.

    Parameters
    ----------
    repo_id: str
        The ID of the destination repository in the form of ``OWNER/REPO_NAME``.

    source: str or Path
        A folder where the contents of the model repo are located.

    token: str, optional
        A token to push to the Hub. If not provided, the user should be already
        logged in using ``huggingface-cli login``.

    commit_message: str, optional
        The commit message to be used when pushing to the repo.

    create_remote: bool, default=False
        Whether to create the remote repository if it doesn't exist. If the
        remote repository doesn't exist and this parameter is ``False``, it
        raises an error. Otherwise it checks if the remote repository exists,
        and would create it if it doesn't.

    private: bool, default=None
        Whether the remote repository should be public or private. If ``True``
        or ``False`` is passed, this method will set the private/public status
        of the remote repository, regardless of it already existing or not. If
        ``None``, no change is applied.

        .. versionadded:: 0.3

    Returns
    -------
    None

    Raises
    ------
    TypeError
        This function raises a ``TypeError`` if the contents of the source
        folder do not make a valid Hugging Face Hub scikit-learn based repo.
    """
    warnings.warn(
        "Creating repos on hf.co is subject to strict rate limits now and therefore"
        " this feature is to be removed from this library in version 0.10. You can"
        " use tools directly available in the huggingface_hub library instead to"
        " create and push files.",
        FutureWarning,
    )
    _validate_folder(path=source)
    client = HfApi()

    if create_remote:
        client.create_repo(
            repo_id=repo_id, token=token, repo_type="model", exist_ok=True
        )

    if private is not None:
        client.update_repo_visibility(repo_id=repo_id, private=private, token=token)

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


def get_config(path: Union[str, Path]) -> dict[str, Any]:
    """Returns the configuration of a project.

    Parameters
    ----------
    path: str
        The path to the directory holding the project and its ``config.json``
        configuration file.

    Returns
    -------
    config: dict
        A dictionary which holds the configs of the project.
    """
    with open(Path(path) / "config.json", "r") as f:
        config = json.load(f)
    return config


def get_requirements(path: Union[str, Path]) -> List[str]:
    """Returns the requirements of a project.

    Parameters
    ----------
    path: str
        The path to the director holding the project and its ``config.json``
        configuration file.

    Returns
    -------
    requirements: list of str
        The list of requirements which can be passed to the package manager to
        be installed.
    """
    config = get_config(path)
    return config.get("sklearn", dict()).get("environment", list())


def download(
    *,
    repo_id: str,
    dst: Union[str, Path],
    revision: str | None = None,
    token: str | None = None,
    keep_cache: bool = True,
    **kwargs: Any,
) -> None:
    """Download a repository into a directory.

    The directory needs to be an empty or a non-existing one.

    Parameters
    ----------
    repo_id: str
        The ID of the Hugging Face Hub repository in the form of
        ``OWNER/REPO_NAME``.

    dst: str, or Path
        The directory to which the files are downloaded.

    revision: str, optional
        The revision of the project to download. This can be a git tag, branch,
        or a git commit hash. By default the latest revision of the default
        branch is downloaded.

    token: str, optional
        The token to be used to download the files. Only required if the
        repository is private.

    keep_cache: bool, default=True
        Whether the cached data should be kept or removed after download. By
        default a copy of the cached files will be created in the ``dst``
        folder. If ``False``, the cache will be removed after the contents are
        copied. Note that the cache is git based and by default new files are
        only downloaded if there is a new revision of them on the hub. If you
        keep the cache, the old files are not removed after downloading the
        newer versions of them.

    kwargs: dict
        Other parameters to be passed to
        :func:`huggingface_hub.snapshot_download`.

    Returns
    -------
    None
    """
    dst = Path(dst)
    if dst.exists() and bool(next(dst.iterdir(), None)):
        raise OSError("None-empty dst path already exists!")

    # remove the folder only if it's empty and it exists
    if dst.exists():
        dst.rmdir()

    # TODO: Switch from use_auth_token to token once huggingface_hub<0.11 is
    # dropped. Until then, we ignore the mypy type check, because mypy doesn't
    # see that use_auth_token is handled by the decorator of snapshot_download.
    cached_folder = snapshot_download(
        repo_id=repo_id, revision=revision, use_auth_token=token, **kwargs  # type: ignore
    )
    shutil.copytree(cached_folder, dst)
    if not keep_cache:
        shutil.rmtree(path=cached_folder)


# TODO(v0.10): remove this function
def get_model_output(repo_id: str, data: Any, token: Optional[str] = None) -> Any:
    """Returns the output of the model using Hugging Face Hub's inference API.

    See the :ref:`User Guide <hf_hub_inference>` for more details.

    .. deprecated:: 0.9
        Will be removed in version 0.10. Use ``huggingface_hub.InferenceClient``
        instead.

    Parameters
    ----------
    repo_id: str
        The ID of the Hugging Face Hub repository in the form of
        ``OWNER/REPO_NAME``.

    data: Any
        The input to be given to the model. This can be a
        :class:`pandas.DataFrame` or a :class:`numpy.ndarray`. If possible, you
        should always pass a :class:`pandas.DataFrame` with correct column
        names.

    token: str, optional
        The token to be used to call the inference API. Only required if the
        repository is private.

    Returns
    -------
    output: numpy.ndarray
        The output of the model.

    Notes
    -----
    If there are warnings or exceptions during inference, this function raises
    a :class:`RuntimeError` including the original errors and warnings
    returned from the server.

    Also note that if the model repo is private, the inference API would not be
    available.
    """
    warnings.warn(
        "This feature is no longer free on hf.co and therefore this function will"
        " be removed in the next release. Use `huggingface_hub.InferenceClient`"
        " instead.",
        FutureWarning,
    )
    model_info = HfApi().model_info(repo_id=repo_id, use_auth_token=token)  # type: ignore
    if not model_info.pipeline_tag:
        raise ValueError(
            f"Repo {repo_id} has no pipeline tag. You should set a valid 'task' in"
            " config.json and README.md files. This is automatically done for you if"
            " you pass a valid task to the skops.hub_utils.init() and"
            " skops.card.metadata_from_util() functions to generate those files."
        )

    try:
        inputs = {"data": data.to_dict(orient="list")}
    except AttributeError:
        # the input is not a pandas DataFrame
        inputs = {f"x{i}": data[:, i] for i in range(data.shape[1])}
        inputs = {"data": inputs}

    client = InferenceClient(token=token)
    res_bytes = client.post(json={"inputs": inputs}, model=repo_id)
    res = json.loads(res_bytes.decode("utf-8"))

    if isinstance(res, list):
        return np.array(res)
    else:
        raise RuntimeError(f"There were errors or warnings during inference: {res}")
