from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from reprlib import Repr
from typing import Any, Iterator, Protocol

from huggingface_hub import CardData
from sklearn.utils import estimator_html_repr
from tabulate import tabulate  # type: ignore

from skops.card._model_card import PlotSection, TableSection

aRepr = Repr()
aRepr.maxother = 79
aRepr.maxstring = 79

CONTENT_PLACEHOLDER = "[More Information Needed]"
"""When there is a section but no content, show this"""

DEFAULT_TEMPLATE = {
    "Model description": CONTENT_PLACEHOLDER,
    "Model description/Intended uses & limitations": CONTENT_PLACEHOLDER,
    "Model description/Training Procedure": "",
    "Model description/Training Procedure/Hyperparameters": CONTENT_PLACEHOLDER,
    "Model description/Training Procedure/Model Plot": CONTENT_PLACEHOLDER,
    "Model description/Evaluation Results": CONTENT_PLACEHOLDER,
    "How to Get Started with the Model": CONTENT_PLACEHOLDER,
    "Model Card Authors": (
        f"This model card is written by following authors:\n\n{CONTENT_PLACEHOLDER}"
    ),
    "Model Card Contact": (
        "You can contact the model card authors through following channels:\n"
        f"{CONTENT_PLACEHOLDER}"
    ),
    "Citation": (
        "Below you can find information related to citation.\n\n**BibTeX:**\n```\n"
        f"{CONTENT_PLACEHOLDER}\n```"
    ),
}


def split_subsection_names(key: str) -> list[str]:
    return key.split("/")


def _clean_table(table: str) -> str:
    # replace line breaks "\n" with html tag <br />, however, leave end-of-line
    # line breaks (eol_lb) intact
    eol_lb = "|\n"
    placeholder = "$%!?"  # arbitrary sting that never appears naturally
    table = (
        table.replace(eol_lb, placeholder)
        .replace("\n", "<br />")
        .replace(placeholder, eol_lb)
    )
    return table


@dataclass
class Section:
    title: str
    content: Formattable | str
    subsections: dict[str, Section] = field(default_factory=dict)


class Formattable(Protocol):
    def format(self) -> str:
        ...


class Card:
    def __init__(
        self,
        model,
        model_diagram: bool = True,
        metadata: CardData | None = None,
        prefill: bool = True,
    ):
        self.model = model
        self.model_diagram = model_diagram
        self.metadata = metadata or CardData()

        self._data: dict[str, Section] = {}
        if prefill:
            self._fill_default_sections()
        self._metrics: dict[str, str | float | int] = {}
        # TODO: This is for compatibility with old model card but having an
        # empty table by default is kinda pointless
        self.add_metrics()
        self._reset()

    def _reset(self) -> None:
        model_file = self.metadata.to_dict().get("model_file")
        if model_file:
            self._add_get_started_code(model_file)

        self._add_model_section()
        self._add_hyperparams()

    def _fill_default_sections(self) -> None:
        self.add(**DEFAULT_TEMPLATE)

    def add(self, **kwargs: str | Formattable) -> "Card":
        for key, val in kwargs.items():
            self._add_single(key, val)
        return self

    def _select(
        self, subsection_names: list[str], create: bool = True
    ) -> dict[str, Section]:
        """TODO"""
        section = self._data
        if not subsection_names:
            return section

        for subsection_name in subsection_names:
            section_maybe = section.get(subsection_name)

            # there are already subsections
            if section_maybe is not None:
                section = section_maybe.subsections
                continue

            if create:
                # no subsection, create
                entry = Section(title=subsection_name, content="")
                section[subsection_name] = entry
                section = entry.subsections
            else:
                raise KeyError(f"Section titles {subsection_name} does not exist")

        return section

    def select(self, key: str | list[str]) -> Section:
        assert key  # TODO

        if isinstance(key, str):
            subsection_names = split_subsection_names(key)
        else:
            subsection_names = key

        parent_section = self._select(subsection_names[:-1], create=False)
        return parent_section[subsection_names[-1]]

    def _add_single(self, key: str, val: Formattable | str) -> None:
        section = self._data
        *subsection_names, leaf_node_name = split_subsection_names(key)
        section = self._select(subsection_names)

        if leaf_node_name in section:
            # entry exists, only overwrite content
            section[leaf_node_name].content = val
        else:
            # entry does not exist, create a new one
            section[leaf_node_name] = Section(title=leaf_node_name, content=val)

    def _add_model(self, model) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return

        model_str = self._strip_blank(repr(model))
        model_repr = aRepr.repr(f"model={model_str},").strip('"').strip("'")
        self._add_single("Model description", model_repr)

    def _add_model_section(self) -> None:
        section_title = "Model description/Training Procedure/Model Plot"
        default_content = "The model plot is below."

        if not self.model_diagram:
            self._add_single(section_title, default_content)
            return

        model_plot_div = re.sub(r"\n\s+", "", str(estimator_html_repr(self.model)))
        if model_plot_div.count("sk-top-container") == 1:
            model_plot_div = model_plot_div.replace(
                "sk-top-container", 'sk-top-container" style="overflow: auto;'
            )
        content = f"{default_content}\n\n{model_plot_div}"
        self._add_single(section_title, content)

    def _add_hyperparams(self) -> None:
        hyperparameter_dict = self.model.get_params(deep=True)
        table = _clean_table(
            tabulate(
                list(hyperparameter_dict.items()),
                headers=["Hyperparameter", "Value"],
                tablefmt="github",
            )
        )
        template = textwrap.dedent(
            """        The model is trained with below hyperparameters.

        <details>
        <summary> Click to expand </summary>

        {}

        </details>"""
        )
        self._add_single(
            "Model description/Training Procedure/Hyperparameters",
            template.format(table),
        )

    def add_plot(self, folded=False, **kwargs: str) -> "Card":
        for section_name, plot_path in kwargs.items():
            plot_name = split_subsection_names(section_name)[-1]
            section = PlotSection(alt_text=plot_name, path=plot_path, folded=folded)
            self._add_single(section_name, section)
        return self

    def add_table(self, folded: bool = False, **kwargs: dict["str", list[Any]]) -> Card:
        for key, val in kwargs.items():
            section = TableSection(table=val, folded=folded)
            self._add_single(key, section)
        return self

    def add_metrics(self, **kwargs: str | int | float) -> "Card":
        self._metrics.update(kwargs)
        self._add_metrics(self._metrics)
        return self

    def _add_metrics(self, metrics: dict[str, str | float | int]) -> None:
        table = tabulate(
            list(metrics.items()),
            headers=["Metric", "Value"],
            tablefmt="github",
        )
        template = textwrap.dedent(
            """        You can find the details about evaluation process and the evaluation results.



        {}"""
        )
        self._add_single("Model description/Evaluation Results", template.format(table))

    def _generate_metadata(self, metadata: CardData) -> Iterator[str]:
        for key, val in metadata.to_dict().items() if metadata else {}:
            if key == "widget":
                yield "metadata.widget={...},"
                continue

            yield aRepr.repr(f"metadata.{key}={val},").strip('"').strip("'")

    def _generate_content(
        self, data: dict[str, Section], depth: int = 1
    ) -> Iterator[str]:
        for val in data.values():
            title = f"{depth * '#'} {val.title}"
            yield title

            if isinstance(val.content, str):
                yield val.content
            elif val.content is not None:  # is Formattable
                yield val.content.format()

            if val.subsections:
                yield from self._generate_content(val.subsections, depth=depth + 1)

    def _iterate_content(
        self, data: dict[str, Section], parent_section: str = ""
    ) -> Iterator[tuple[str, Formattable | str]]:
        for val in data.values():
            if parent_section:
                title = "/".join((parent_section, val.title))
            else:
                title = val.title

            yield title, val.content

            if val.subsections:
                yield from self._iterate_content(val.subsections, parent_section=title)

    @staticmethod
    def _strip_blank(text: str) -> str:
        # remove new lines and multiple spaces
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", r" ", text)
        return text

    def _format_repr(self, text: str) -> str:
        # Remove new lines, multiple spaces, quotation marks, and cap line length
        text = self._strip_blank(text)
        return aRepr.repr(text).strip('"').strip("'")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # repr for the model
        model = getattr(self, "model", None)
        if model:
            model_repr = self._format_repr(f"model={repr(model)}")
        else:
            model_repr = None

        # repr for metadata
        metadata_reprs = []
        for key, val in self.metadata.to_dict().items() if self.metadata else {}:
            if key == "widget":
                metadata_reprs.append("metadata.widget={...},")
                continue

            metadata_reprs.append(self._format_repr(f"metadata.{key}={val},"))
        metadata_repr = "\n".join(metadata_reprs)

        # repr for contents
        content_reprs = []
        for title, content in self._iterate_content(self._data):
            if not content:
                continue
            if isinstance(content, str) and content.rstrip().endswith(
                CONTENT_PLACEHOLDER
            ):
                # if content is just some default text, no need to show it
                continue
            content_reprs.append(self._format_repr(f"{title}={content},"))
        content_repr = "\n".join(content_reprs)

        # combine all parts
        complete_repr = "Card(\n"
        if model_repr:
            complete_repr += textwrap.indent(model_repr, "  ") + "\n"
        if metadata_reprs:
            complete_repr += textwrap.indent(metadata_repr, "  ") + "\n"
        if content_reprs:
            complete_repr += textwrap.indent(content_repr, "  ") + "\n"
        complete_repr += ")"
        return complete_repr

    def _add_get_started_code(self, file_name: str, indent: str = "    ") -> None:
        is_skops_format = file_name.endswith(".skops")  # else, assume pickle

        lines = ["```python"]
        if is_skops_format:
            lines += ["from skops.io import load"]
        else:
            lines += ["import joblib"]

        lines += [
            "import json",
            "import pandas as pd",
        ]
        if is_skops_format:
            lines += [
                "from skops.io import load",
                f'model = load("{file_name}")',
            ]
        else:  # pickle
            lines += [f"model = joblib.load({file_name})"]

        lines += [
            'with open("config.json") as f:',
            indent + "config = json.load(f)",
            'model.predict(pd.DataFrame.from_dict(config["sklearn"]["example_input"]))',
            "```",
        ]
        template = textwrap.dedent(
            """        Use the code below to get started with the model.

        {}
        """
        )
        self._add_single(
            "How to Get Started with the Model", template.format("\n".join(lines))
        )

    def _generate_card(self) -> Iterator[str]:
        if self.metadata:
            yield f"---\n{self.metadata.to_yaml()}\n---"

        for line in self._generate_content(self._data):
            if line:
                yield "\n" + line

    def save(self, path: str | Path) -> None:
        """Save the model card.

        This method renders the model card in markdown format and then saves it
        as the specified file.

        Parameters
        ----------
        path: str, or Path
            Filepath to save your card.

        Notes
        -----
        The keys in model card metadata can be seen `here
        <https://huggingface.co/docs/hub/models-cards#model-card-metadata>`__.
        """
        with open(path, "w") as f:
            f.write("\n".join(self._generate_card()))

    def render(self) -> str:
        """Render the final model card as a string.

        Returns
        -------
        result : str
            The rendered model card with all placeholders filled and all extra
            sections inserted.
        """
        return "\n".join(self._generate_card())


def main():
    import os
    import pickle
    import tempfile
    from uuid import uuid4

    import matplotlib.pyplot as plt
    import sklearn
    from huggingface_hub import HfApi
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from skops import hub_utils
    from skops.card import metadata_from_config

    X, y = load_iris(return_X_y=True, as_frame=True)

    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=123))]
    ).fit(X, y)

    pkl_file = tempfile.mkstemp(suffix=".pkl", prefix="skops-test")[1]
    with open(pkl_file, "wb") as f:
        pickle.dump(model, f)

    with tempfile.TemporaryDirectory(prefix="skops-test") as destination_path:
        hub_utils.init(
            model=pkl_file,
            requirements=[f"scikit-learn=={sklearn.__version__}"],
            dst=destination_path,
            task="tabular-classification",
            data=X,
        )
        card = Card(model, metadata=metadata_from_config(destination_path))

        # add a placeholder for figures
        card.add(Plots="")

        # add arbitrary sections, overwrite them, etc.
        card.add(hi="howdy")
        card.add(**{"parent section/child section": "child content"})
        card.add(**{"foo": "bar", "spam": "eggs"})
        # change content of "hi" section
        card.add(**{"hi/german": "guten tag", "hi/french": "salut"})
        card.add(**{"very/deeply/nested/section": "but why?"})

        # add metrics
        card.add_metrics(**{"acc": 0.1})

        # insert the plot in the "Plot" section we inserted above
        plt.plot([4, 5, 6, 7])
        plt.savefig(Path(destination_path) / "fig1.png")
        card.add_plot(**{"Plots/A beautiful plot": "fig1.png"})

        # add table
        table = {"split": [1, 2, 3], "score": [4, 5, 6]}
        card.add_table(
            folded=True,
            **{"Model description/Training Procedure/Yet another table": table},
        )

        # more metrics
        card.add_metrics(**{"f1": 0.2, "roc": 123})

        # add content for "Model description" section, which has subsections but
        # otherwise no content
        card.add(**{"Model description": "This is a fantastic model"})

        card.save(Path(destination_path) / "README.md")
        print(destination_path)

        # pushing to Hub
        token = os.environ["HF_HUB_TOKEN"]
        repo_name = f"hf_hub_example-{uuid4()}"
        user_name = HfApi().whoami(token=token)["name"]
        repo_id = f"{user_name}/{repo_name}"
        print(f"Creating and pushing to repo: {repo_id}")
        hub_utils.push(
            repo_id=repo_id,
            source=destination_path,
            token=token,
            commit_message="testing model cards",
            create_remote=True,
            private=False,
        )


if __name__ == "__main__":
    main()
