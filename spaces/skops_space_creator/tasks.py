"""Functionality around tasks

Tasks are used to implement "undo" and "redo" functionality.

"""
from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from streamlit.runtime.uploaded_file_manager import UploadedFile

from skops import card
from skops.card._model_card import PlotSection, split_subsection_names


class Task:
    """(Abstract) base class for tasks"""

    def do(self) -> None:
        raise NotImplementedError

    def undo(self) -> None:
        raise NotImplementedError


class TaskState:
    """Tracking the state of tasks"""

    def __init__(self) -> None:
        self.done_list: list[Task] = []
        self.undone_list: list[Task] = []

    def undo(self) -> None:
        if not self.done_list:
            return

        task = self.done_list.pop(-1)
        task.undo()
        self.undone_list.append(task)

    def redo(self) -> None:
        if not self.undone_list:
            return

        task = self.undone_list.pop(-1)
        task.do()
        self.done_list.append(task)

    def add(self, task: Task) -> None:
        task.do()
        self.done_list.append(task)
        self.undone_list.clear()

    def reset(self) -> None:
        self.done_list.clear()
        self.undone_list.clear()


class AddSectionTask(Task):
    """Add a new text section"""

    def __init__(
        self,
        model_card: card.Card,
        title: str,
        content: str,
    ) -> None:
        self.model_card = model_card
        self.title = title
        self.key = title + " " + str(uuid4())[:6]
        self.content = content

    def do(self) -> None:
        self.model_card.add(**{self.key: self.content})
        section = self.model_card.select(self.key)
        section.title = split_subsection_names(self.title)[-1]

    def undo(self) -> None:
        self.model_card.delete(self.key)


class AddFigureTask(Task):
    """Add a new figure section

    Figure always starts out with dummy image cat.png.

    """

    def __init__(
        self,
        model_card: card.Card,
        path: Path,
        title: str,
        content: str,
    ) -> None:
        self.model_card = model_card
        self.title = title

        # Create a unique file name, since the same image can exist more than
        # once per model card.
        fname = Path(content)
        stem = fname.stem
        suffix = fname.suffix
        uniq = str(uuid4())[:6]
        new_fname = str(path / stem) + "_" + uniq + suffix

        self.key = title + " " + uniq
        self.content = Path(new_fname)

    def do(self) -> None:
        shutil.copy("cat.png", self.content)
        self.model_card.add_plot(**{self.key: self.content})
        section = self.model_card.select(self.key)
        section.title = split_subsection_names(self.title)[-1]

    def undo(self) -> None:
        self.content.unlink(missing_ok=True)
        self.model_card.delete(self.key)


class DeleteSectionTask(Task):
    """Delete a section

    The section is not completely removed from the underlying data structure,
    but only turned invisible.

    """

    def __init__(
        self,
        model_card: card.Card,
        key: str,
        path: Path | None,
    ) -> None:
        self.model_card = model_card
        self.key = key
        # when 'deleting' a file, move it to a temp file
        self.path = path
        self.tmp_path = Path(mkdtemp(prefix="skops-")) / str(uuid4())

    def do(self) -> None:
        self.model_card.select(self.key).visible = False
        if self.path:
            shutil.move(self.path, self.tmp_path)

    def undo(self) -> None:
        self.model_card.select(self.key).visible = True
        if self.path:
            shutil.move(self.tmp_path, self.path)


class UpdateSectionTask(Task):
    """Change the title or content of a text section"""

    def __init__(
        self,
        model_card: card.Card,
        key: str,
        old_name: str,
        new_name: str,
        old_content: str,
        new_content: str,
    ) -> None:
        self.model_card = model_card
        self.key = key
        self.old_name = old_name
        self.new_name = new_name
        self.old_content = old_content
        self.new_content = new_content

    def do(self) -> None:
        section = self.model_card.select(self.key)
        new_title = split_subsection_names(self.new_name)[-1]
        section.title = new_title
        section.content = self.new_content

    def undo(self) -> None:
        section = self.model_card.select(self.key)
        old_title = split_subsection_names(self.old_name)[-1]
        section.title = old_title
        section.content = self.old_content


class UpdateFigureTitleTask(Task):
    """Change the title a plot section

    Changing the title is easy, just replace it and be done.

    """

    def __init__(
        self,
        model_card: card.Card,
        key: str,
        old_name: str,
        new_name: str,
    ) -> None:
        self.model_card = model_card
        self.key = key
        self.old_name = old_name
        self.new_name = new_name

    def do(self) -> None:
        section = self.model_card.select(self.key)
        new_title = split_subsection_names(self.new_name)[-1]
        section.title = self.title = new_title

    def undo(self) -> None:
        section = self.model_card.select(self.key)
        old_title = split_subsection_names(self.old_name)[-1]
        section.title = old_title


class UpdateFigureTask(Task):
    """Change the title or image of a figure section

    Changing the title is easy, just replace it and be done.

    Changing the figure is a bit more tricky. The old figure is in the hf_path
    under its old name. The new figure is an UploadFile object. For the DO
    operation, move the old figure to a temporary file and store the UploadFile
    content to a new file (which may have a different name).

    For the UNDO operation, delete the new figure (its content is still stored
    in the UploadFile) and move back the old figure from its temporary file to
    the original location (with its original name).

    """

    def __init__(
        self,
        model_card: card.Card,
        key: str,
        old_name: str,
        new_name: str,
        data: UploadedFile,
        new_path: Path,
        old_path: Path,
    ) -> None:
        self.model_card = model_card
        self.key = key
        self.old_name = old_name
        self.new_name = new_name
        self.new_path = new_path
        self.old_path = old_path
        self.new_data = data
        # when 'deleting' the old image, move to temp path
        self.tmp_path = Path(mkdtemp(prefix="skops-")) / str(uuid4())

    def do(self) -> None:
        section = self.model_card.select(self.key)
        assert isinstance(section, PlotSection), "has to be a PlotSection"
        new_title = split_subsection_names(self.new_name)[-1]
        section.title = self.title = new_title

        # write figure
        # note: this can still be the same image if the image is a file, there
        # is no test to check, e.g., the hash of the image
        shutil.move(self.old_path, self.tmp_path)

        with open(self.new_path, "wb") as f:
            f.write(self.new_data.getvalue())

        section.path = self.new_path

    def undo(self) -> None:
        section = self.model_card.select(self.key)
        assert isinstance(section, PlotSection), "has to be a PlotSection"
        old_title = split_subsection_names(self.old_name)[-1]
        section.title = old_title

        self.new_path.unlink(missing_ok=True)
        shutil.move(self.tmp_path, self.old_path)
        section.path = self.old_path


class AddMetricsTask(Task):
    """Add new metrics"""

    def __init__(
        self,
        model_card: card.Card,
        metrics: dict[str, str | int | float],
    ) -> None:
        self.model_card = model_card
        self.old_metrics = model_card._metrics.copy()
        self.new_metrics = metrics

    def do(self) -> None:
        self.model_card._metrics.clear()
        self.model_card.add_metrics(**self.new_metrics)

    def undo(self) -> None:
        self.model_card._metrics.clear()
        self.model_card.add_metrics(**self.old_metrics)
