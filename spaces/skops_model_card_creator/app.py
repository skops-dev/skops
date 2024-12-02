"""The app.py used with streamlit

This ties together the different parts of the app.

"""

import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

import streamlit as st

from .create import create_repo_input_form
from .edit import edit_input_form
from .help import help_page
from .start import start_input_form

# Change cwd to a temporary path
if "work_dir" not in st.session_state:
    work_dir = Path(mkdtemp(prefix="skops-"))
    shutil.copy2("cat.png", work_dir / "cat.png")
    os.chdir(work_dir)
    st.session_state.work_dir = work_dir

# Create a hf_path, which is where the repo will be created locally. When the
# session is created, copy the dummy cat.png file there and make it the cwd
if "hf_path" not in st.session_state:
    hf_path = Path(mkdtemp(prefix="skops-"))
    st.session_state.hf_path = hf_path


st.header("Skops model card creator")


class Screen:
    state: Literal["start", "edit", "create_repo"] = "start"


if "screen" not in st.session_state:
    st.session_state.screen = Screen()


if st.session_state.screen.state == "start":
    start_input_form()
elif st.session_state.screen.state == "help":
    help_page()
elif st.session_state.screen.state == "edit":
    edit_input_form()
elif st.session_state.screen.state == "create_repo":
    create_repo_input_form()
else:
    st.write("Something went wrong, please open an issue")
