"""Start page of the app

This page is used to initialize a model card that is either:

1. based on the skops template
2. empty
3. loads an existing model card

Optionally, users can add a model file, data, requirements, and choose a task.

"""

import glob
import io
import os
import pickle
import shutil
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import sklearn
import streamlit as st
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

import skops.io as sio
from skops import card, hub_utils

tmp_path = Path(mkdtemp(prefix="skops-"))  # temporary files
description = """Create a Hugging Face model repository for scikit learn models

This page aims to provide a simple interface to use the
[`skops`](https://skops.readthedocs.io/) model card and HF Hub creation
utilities.

"""


def load_model() -> None:
    if st.session_state.get("model_file") is None:
        st.session_state.model = DummyClassifier()
        return

    bytes_data = st.session_state.model_file.getvalue()
    if st.session_state.model_file.name.endswith("skops"):
        model = sio.loads(bytes_data, trusted=True)
    else:
        model = pickle.loads(bytes_data)
    assert isinstance(model, BaseEstimator), "model must be an sklearn model"

    st.session_state.model = model


def load_data() -> None:
    if st.session_state.get("data_file"):
        bytes_data = io.BytesIO(st.session_state.data_file.getvalue())
        df = pd.read_csv(bytes_data)
    else:
        df = pd.DataFrame([])

    st.session_state.data = df


def _clear_repo(path: str) -> None:
    for file_path in glob.glob(str(Path(path) / "*")):
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def init_repo() -> None:
    path = st.session_state.hf_path
    _clear_repo(path)
    requirements = []
    task = "tabular-classification"
    data = pd.DataFrame([])

    if "requirements" in st.session_state:
        requirements = st.session_state.requirements.splitlines()
    if "task" in st.session_state:
        task = st.session_state.task
    if "data_file" in st.session_state:
        load_data()
        data = st.session_state.data

    if task.startswith("text") and isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    try:
        file_name = tmp_path / "model.skops"
        sio.dump(st.session_state.model, file_name)

        hub_utils.init(
            model=file_name,
            dst=path,
            task=task,
            data=data,
            requirements=requirements,
        )
    except Exception as exc:
        print("Uh oh, something went wrong when initializing the repo:", exc)


def create_skops_model_card() -> None:
    init_repo()
    metadata = card.metadata_from_config(st.session_state.hf_path)
    model_card = card.Card(model=st.session_state.model, metadata=metadata)
    st.session_state.model_card = model_card
    st.session_state.model_card_type = "skops"
    st.session_state.screen.state = "edit"


def create_empty_model_card() -> None:
    init_repo()
    metadata = card.metadata_from_config(st.session_state.hf_path)
    model_card = card.Card(
        model=st.session_state.model, metadata=metadata, template=None
    )
    model_card.add(**{"Untitled": "[More Information Needed]"})
    st.session_state.model_card = model_card
    st.session_state.model_card_type = "empty"
    st.session_state.screen.state = "edit"


def create_hf_model_card() -> None:
    repo_id = st.session_state.get("hf_repo_id", "").strip().strip("'").strip('"')
    if not repo_id:
        return

    try:
        allow_patterns = [
            "*.md",
            ".txt",
            "*.png",
            "*.gif",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.webp",
        ]
        path = snapshot_download(repo_id, allow_patterns=allow_patterns)
    except (HFValidationError, RepositoryNotFoundError):
        st.error(
            f"Repository '{repo_id}' could not be found on HF Hub, "
            "please check that the repo ID is correct."
        )
        return

    # move everything to the hf_path and working dir
    hf_path = st.session_state.hf_path
    shutil.copytree(path, hf_path, dirs_exist_ok=True)
    shutil.copytree(path, ".", dirs_exist_ok=True)

    model_card = card.parse_modelcard(hf_path / "README.md")
    st.session_state.model_card = model_card
    st.session_state.model_card_type = "loaded"
    st.session_state.screen.state = "edit"


def add_help_button():
    def fn():
        st.session_state.screen.state = "help"

    st.button(
        "Instructions",
        on_click=fn,
        help="Detailed explanation of this space",
        key="get_help",
    )


def start_input_form():
    if "model" not in st.session_state:
        st.session_state.model = DummyClassifier()

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame([])

    if "model_card" not in st.session_state:
        st.session_state.model_card = None

    st.markdown(description)

    add_help_button()

    st.markdown("---")

    st.text(
        "Upload an sklearn model (strongly recommended)\n"
        "The model can be used to automatically populate fields in the model card."
    )

    if not st.session_state.get("model_file"):
        st.file_uploader(
            "Upload an sklearn model (pickle or skops format)",
            on_change=load_model,
            key="model_file",
        )

    st.markdown("---")

    st.text(
        "Upload samples from your data (in csv format)\n"
        "This sample data can be attached to the metadata of the model card"
    )
    st.file_uploader(
        "Upload input data (csv)", type=["csv"], on_change=load_data, key="data_file"
    )
    st.markdown("---")

    st.selectbox(
        label="Choose the task type",
        options=[
            "tabular-classification",
            "tabular-regression",
            "text-classification",
            "text-regression",
        ],
        key="task",
        on_change=init_repo,
    )
    st.markdown("---")

    st.text_area(
        label="Requirements",
        value=f"scikit-learn=={sklearn.__version__}\n",
        key="requirements",
        on_change=init_repo,
    )
    st.markdown("---")

    st.markdown("Choose one of the options below to get started:")
    col_0, col_1, col_2 = st.columns([2, 2, 2])
    with col_0:
        st.button("Create a new skops model card", on_click=create_skops_model_card)

    with col_1:
        st.button("Create a new empty model card", on_click=create_empty_model_card)

    with col_2:
        with st.form("Load existing model card from HF Hub", clear_on_submit=False):
            st.markdown("Load existing model card from HF Hub")
            st.text_input("Repo name (e.g. 'gpt2')", key="hf_repo_id")
            st.form_submit_button("Load", on_click=create_hf_model_card)


start_input_form()
