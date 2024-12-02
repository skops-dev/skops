import os
from pathlib import Path

import streamlit as st

from skops import hub_utils

from .utils import get_rendered_model_card


def _add_back_button():
    def fn():
        st.session_state.screen.state = "edit"

    st.button("Back", help="continue editing the model card", on_click=fn)


def _add_delete_button():
    def fn():
        if "hf_path" in st.session_state:
            del st.session_state["hf_path"]
        if "model_card" in st.session_state:
            del st.session_state["model_card"]
        if "task_state" in st.session_state:
            st.session_state.task_state.reset()
        if "create_repo_name" in st.session_state:
            del st.session_state["create_repo_name"]
        if "hf_token" in st.session_state:
            del st.session_state["hf_token"]
        st.session_state.screen.state = "start"

    st.button("Delete", on_click=fn, help="Start over from scratch (lose all progress)")


def _save_model_card(path: Path) -> None:
    model_card = st.session_state.get("model_card")
    if model_card:
        # do not use model_card.save, see doc of get_rendered_model_card
        rendered = get_rendered_model_card(
            model_card, hf_path=str(st.session_state.hf_path)
        )
        with open(path / "README.md", "w") as f:
            f.write(rendered)


def _display_repo_overview(path: Path) -> None:
    text = "Files included in the repository:\n"
    for file in os.listdir(path):
        size = os.path.getsize(path / file)
        text += f"- `{file} ({size:,} bytes)`\n"
    st.markdown(text)


def _display_private_box():
    tip = (
        "Private repositories can only seen by you or members of the same "
        "organization, see https://huggingface.co/docs/hub/repositories-settings"
    )
    st.checkbox(
        "Make repository private", value=True, help=tip, key="create_repo_private"
    )


def _repo_id_field():
    st.text_input("Name of the repository (e.g. 'User/MyRepo')", key="create_repo_name")


def _hf_token_field():
    tip = "The Hugging Face token can be found at https://hf.co/settings/token"
    st.text_input("Enter your Hugging Face token ('hf_***')", key="hf_token", help=tip)


def _create_hf_repo(path, repo_name, hf_token, private):
    try:
        hub_utils.push(
            repo_id=repo_name,
            source=path,
            token=hf_token,
            private=private,
            create_remote=True,
        )
    except Exception as exc:
        st.error(
            "Oops, something went wrong, please create an issue. "
            f"The error message is:\n\n{exc}"
        )
        return

    st.success(f"Successfully created the repo 'https://huggingface.co/{repo_name}'")


def _add_create_repo_button():
    private = bool(st.session_state.get("create_repo_private"))
    repo_name = st.session_state.get("create_repo_name")
    hf_token = st.session_state.get("hf_token")
    disabled = (not repo_name) or (not hf_token)

    button_text = "Create a new repository"
    tip = "Creating a repo requires a name and a token"
    path = st.session_state.get("hf_path")
    st.button(
        button_text,
        help=tip,
        disabled=disabled,
        on_click=_create_hf_repo,
        args=(path, repo_name, hf_token, private),
    )

    if not repo_name:
        st.info("Repository name is required")
    if not hf_token:
        st.info("Token is required")


def create_repo_input_form():
    if not st.session_state.screen.state == "create_repo":
        return

    col_0, col_1, *_ = st.columns([2, 2, 2, 2])
    with col_0:
        _add_back_button()
    with col_1:
        _add_delete_button()

    hf_path = st.session_state.hf_path
    _save_model_card(hf_path)
    _display_repo_overview(hf_path)
    _display_private_box()
    st.markdown("---")
    _repo_id_field()
    _hf_token_field()
    _add_create_repo_button()
