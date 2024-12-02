"""The editing page of the app

This is the meat of the application. On the sidebar, the content of the model
card is displayed in the form of editable fields. On the right side, the
rendered model card is shown.

In the side bar, users can:

- edit the title and content of existing sections
- delete sections
- add new sections below the current section
- add new figures below the current section

Moreover, each action results in a "task" that is tracked in the task state. A
task has a "do" and an "undo" method. This allows us to provide "undo" and
"redo" features to the app, making it easier for users to experiment and deal
with errors. The "reset" button undoes all the tasks, leading back to the
initial model card.

When the user is finished, there is a "save" button that downloads the model
card. They can also click "delete" to start over again, leading them to the
start page.

"""


from __future__ import annotations

import reprlib
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
from huggingface_hub import hf_hub_download

from skops import card
from skops.card._model_card import PlotSection, split_subsection_names

from .tasks import (
    AddFigureTask,
    AddMetricsTask,
    AddSectionTask,
    DeleteSectionTask,
    TaskState,
    UpdateFigureTask,
    UpdateFigureTitleTask,
    UpdateSectionTask,
)
from .utils import (
    get_rendered_model_card,
    iterate_key_section_content,
    process_card_for_rendering,
)

arepr = reprlib.Repr()
arepr.maxstring = 24
tmp_path = Path(mkdtemp(prefix="skops-"))  # temporary files


def load_model_card_from_repo(repo_id: str) -> card.Card:
    print("downloading model card")
    path = hf_hub_download(repo_id, "README.md")
    model_card = card.parse_modelcard(path)
    return model_card


def _update_model_card(
    model_card: card.Card,
    key: str,
    section_name: str,
    content: str,
) -> None:
    # This is a very roundabout way to update the model card but it's necessary
    # because of how streamlit handles session state. Basically, there have to
    # be "key" arguments, which have to be retrieved from the session_state, as
    # they are up-to-date. Just getting the Python variables is not enough, as
    # they can be out of date.

    # key names must match with those used in form
    new_title = st.session_state[f"{key}.title"]
    new_content = st.session_state[f"{key}.content"]

    # determine if title is the same
    old_title_split = split_subsection_names(section_name)
    new_title_split = old_title_split[:-1] + [new_title]
    is_title_same = old_title_split == new_title_split

    # determine if content is the same
    is_content_same = (content == new_content) or (not content and not new_content)
    if is_title_same and is_content_same:
        return

    section = model_card.select(key)
    if not isinstance(section, PlotSection):
        # a normal section
        task = UpdateSectionTask(
            model_card,
            key=key,
            old_name=section_name,
            new_name=new_title,
            old_content=content,
            new_content=new_content,
        )
    else:
        # a plot sectoin
        if not new_content:  # only title changed
            task = UpdateFigureTitleTask(
                model_card, key=key, old_name=section_name, new_name=new_title
            )
        else:  # new figure uploaded
            fname = new_content.name.replace(" ", "_")
            fpath = st.session_state.hf_path / fname
            old_path = fpath.parent / Path(section.path).name
            task = UpdateFigureTask(
                model_card,
                key=key,
                old_name=section_name,
                new_name=new_title,
                data=new_content,
                new_path=fpath,
                old_path=old_path,
            )
    st.session_state.task_state.add(task)


def _add_section(model_card: card.Card, key: str) -> None:
    section_name = f"{key}/Untitled"
    task = AddSectionTask(
        model_card, title=section_name, content="[More Information Needed]"
    )
    st.session_state.task_state.add(task)


def _add_figure(model_card: card.Card, key: str) -> None:
    section_name = f"{key}/Untitled"
    hf_path = st.session_state.hf_path
    task = AddFigureTask(
        model_card, path=hf_path, title=section_name, content="cat.png"
    )
    st.session_state.task_state.add(task)


def _delete_section(model_card: card.Card, key: str, path: Path) -> None:
    task = DeleteSectionTask(model_card, key=key, path=path)
    st.session_state.task_state.add(task)


def _add_section_form(
    model_card: card.Card, key: str, section_name: str, old_title: str, content: str
) -> None:
    with st.form(key, clear_on_submit=False):
        st.header(section_name)
        # setting the 'key' argument below to update the session_state
        st.text_input("Section name", value=old_title, key=f"{key}.title")
        st.text_area("Content", value=content, key=f"{key}.content")
        st.form_submit_button(
            "Update",
            on_click=_update_model_card,
            args=(model_card, key, section_name, content),
        )


def _add_fig_form(
    model_card: card.Card, key: str, section_name: str, old_title: str, content: str
) -> None:
    with st.form(key, clear_on_submit=False):
        st.header(section_name)
        # setting the 'key' argument below to update the session_state
        st.text_input("Section name", value=old_title, key=f"{key}.title")
        st.file_uploader("Upload image", key=f"{key}.content")
        st.form_submit_button(
            "Update",
            on_click=_update_model_card,
            args=(model_card, key, section_name, content),
        )


def create_form_from_section(
    model_card: card.Card,
    key: str,
    section_name: str,
) -> None:
    # Code for creating a single section, plot or text
    section = model_card.select(key)
    content = section.content
    split_sections = split_subsection_names(section_name)
    old_title = split_sections[-1]

    if isinstance(section, PlotSection):
        _add_fig_form(
            model_card=model_card,
            key=key,
            section_name=section_name,
            old_title=old_title,
            content=content,
        )
        path = st.session_state.hf_path / Path(section.path).name
    else:
        _add_section_form(
            model_card=model_card,
            key=key,
            section_name=section_name,
            old_title=old_title,
            content=content,
        )
        path = None

    col_0, col_1, col_2 = st.columns([4, 2, 2])
    with col_0:
        st.button(
            f"Delete '{arepr.repr(old_title)}'",
            on_click=_delete_section,
            args=(model_card, key, path),
            key=f"{key}.delete",
            help="Delete this section, including all its subsections",
        )
    with col_1:
        st.button(
            "add section below",
            on_click=_add_section,
            args=(model_card, key),
            key=f"{key}.add",
            help="Add a new subsection below this section",
        )
    with col_2:
        st.button(
            "add figure below",
            on_click=_add_figure,
            args=(model_card, key),
            key=f"{key}.fig",
            help="Add a new figure below this section",
        )


def display_sections(model_card: card.Card) -> None:
    # display all sections, looping through them recursively
    for key, title in iterate_key_section_content(model_card._data):
        create_form_from_section(model_card, key=key, section_name=title)


def display_toc(model_card: card.Card) -> None:
    toc = model_card.get_toc()
    st.markdown(toc)


def display_model_card(model_card: card.Card) -> None:
    rendered = model_card.render()
    metadata, rendered = process_card_for_rendering(rendered)

    # strip metadata
    with st.expander("show metadata"):
        st.text(metadata)

    with st.expander("Table of Contents"):
        display_toc(model_card)

    st.markdown(rendered, unsafe_allow_html=True)


def reset_model_card() -> None:
    if "task_state" not in st.session_state:
        return
    if "model_card" not in st.session_state:
        del st.session_state["model_card"]

    while st.session_state.task_state.done_list:
        st.session_state.task_state.undo()


def delete_model_card() -> None:
    if "hf_path" in st.session_state:
        del st.session_state["hf_path"]
    if "model_card" in st.session_state:
        del st.session_state["model_card"]
    if "task_state" in st.session_state:
        st.session_state.task_state.reset()
    st.session_state.screen.state = "start"


def undo_last():
    st.session_state.task_state.undo()
    display_model_card(st.session_state.model_card)


def redo_last():
    st.session_state.task_state.redo()
    display_model_card(st.session_state.model_card)


def add_download_model_card_button():
    model_card = st.session_state.model_card
    data = get_rendered_model_card(model_card, hf_path=str(st.session_state.hf_path))
    tip = "Download the generated model card as markdown file"
    st.download_button(
        "Save (md)",
        data=data,
        help=tip,
        file_name="README.md",
    )


def add_create_repo_button():
    def fn():
        st.session_state.screen.state = "create_repo"

    button_disabled = not bool(st.session_state.get("model_card"))
    st.button(
        "Create Repo",
        help="Create a model repository on Hugging Face Hub",
        on_click=fn,
        disabled=button_disabled,
    )


def display_edit_buttons():
    # first row: undo + redo + reset
    col_0, col_1, col_2, *_ = st.columns([2, 2, 2, 2])
    undo_disabled = not bool(st.session_state.task_state.done_list)
    redo_disabled = not bool(st.session_state.task_state.undone_list)
    with col_0:
        name = f"UNDO ({len(st.session_state.task_state.done_list)})"
        tip = "Undo the last edit"
        st.button(name, on_click=undo_last, disabled=undo_disabled, help=tip)
    with col_1:
        name = f"REDO ({len(st.session_state.task_state.undone_list)})"
        tip = "Redo the last undone edit"
        st.button(name, on_click=redo_last, disabled=redo_disabled, help=tip)
    with col_2:
        tip = "Undo all edits"
        st.button("Reset", on_click=reset_model_card, help=tip)

    # second row: download + create repo + delete
    col_0, col_1, col_2, *_ = st.columns([2, 2, 2, 2])
    with col_0:
        add_download_model_card_button()
    with col_1:
        add_create_repo_button()
    with col_2:
        tip = "Start over from scratch (lose all progress)"
        st.button("Delete", on_click=delete_model_card, help=tip)


def _update_model_diagram():
    val = st.session_state.get("special_model_diagram", True)
    model_card = st.session_state.model_card
    model_card.model_diagram = val

    # TODO: this may no longer be necesssary once this issue is solved:
    # https://github.com/skops-dev/skops/issues/292
    if val:
        model_card.add_model_plot()
    else:
        model_card.delete("Model description/Training Procedure/Model Plot")


def _parse_metrics(metrics: str) -> dict[str, str | float]:
    # parse metrics from text area, one per line, into a dict
    metrics_table = {}
    for line in metrics.splitlines():
        line = line.strip()
        val: str | float
        name, _, val = line.partition("=")
        try:
            # try to coerce to float but don't error if it fails
            val = float(val.strip())
        except ValueError:
            pass
        metrics_table[name.strip()] = val
    return metrics_table


def _update_metrics():
    metrics = st.session_state.get("special_metrics_text", {})
    model_card = st.session_state.model_card
    metrics_table = _parse_metrics(metrics)

    # check if any change
    if metrics_table == model_card._metrics:
        return

    task = AddMetricsTask(model_card, metrics_table)
    st.session_state.task_state.add(task)


def display_skops_special_fields():
    st.checkbox(
        "Show model diagram",
        value=True,
        on_change=_update_model_diagram,
        key="special_model_diagram",
    )

    with st.expander("Add metrics"):
        with st.form("special_metrics", clear_on_submit=False):
            st.text_area(
                "Add one metric per line, e.g. 'accuracy = 0.9'",
                key="special_metrics_text",
            )
            st.form_submit_button(
                "Update",
                on_click=_update_metrics,
            )


def edit_input_form():
    if "task_state" not in st.session_state:
        st.session_state.task_state = TaskState()

    with st.sidebar:
        # TOP ROW BUTTONS
        display_edit_buttons()

        # SHOW SPECIAL FIELDS IF SKOPS TEMPLATE WAS USED
        if st.session_state.get("model_card_type", "") == "skops":
            display_skops_special_fields()

        # SHOW EDITABLE SECTIONS
        if "model_card" in st.session_state:
            display_sections(st.session_state.model_card)

    if "model_card" in st.session_state:
        display_model_card(st.session_state.model_card)
