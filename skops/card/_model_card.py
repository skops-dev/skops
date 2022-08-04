from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from modelcards import CardData, ModelCard
from sklearn.utils import estimator_html_repr

import skops


def metadata_from_config(config_path: Union[str, Path]) -> CardData:
    """Construct a ``CardData`` object from a ``config.json`` file.

    Most information needed for the metadata section of a ``README.md``
    file on Hugging Face Hub is included in the ``config.json`` file. This
    utility function constructs a ``CardData`` object which can then be
    passed to the :class:`~skops.card.Card` object.

    This method populates the following attributes of the instance:

    - ``library_name``: It needs to be ``sklearn`` for scikit-learn
        compatible models.
    - ``tags``: Set to a list, containing ``"sklearn"`` and the task of the
        model. You can then add more tags to this list.
    - ``widget``: It is populated with the example data to be used by the
        widget component of the Hugging Face Hub widget, on the model's
        repository page.

    Parameters
    ----------
    config_path: str, or Path
        Filepath to the ``config.json`` file, or the folder including that
        file.

    Returns
    -------
    card_data: ``modelcards.CardData``
        ``CardData`` object.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    card_data = CardData()
    card_data.library_name = "sklearn"
    card_data.tags = ["sklearn"]
    task = config.get("sklearn", {}).get("task", None)
    if task:
        card_data.tags += [task]

    example_input = config.get("sklearn", {}).get("example_input", None)
    # Documentation on what the widget expects:
    # https://huggingface.co/docs/hub/models-widgets-examples
    if example_input:
        if "tabular" in task:
            card_data.widget = {"structuredData": example_input}
        # TODO: add text data example here.

    return card_data


class Card:
    """Model card class that will be used to generate model card.

    This class can be used to write information and plots to model card and
    save it. This class by default generates an interactive plot of the model
    and a table of hyperparameters. The slots to be filled are defined in the
    markdown template.

    Parameters
    ----------
    model: estimator object
        Model that will be documented.

    model_diagram: bool, default=True
        Set to True if model diagram should be plotted in the card.

    metadata: CardData, optional
        ``CardData`` object. The contents of this object are saved as metadata
        at the beginning of the output file, and used by Hugging Face Hub.

        You can use :func:`~skops.card.metadata_from_config` to create an
        instance pre-populated with necessary information based on the contents
        of the ``config.json`` file, which itself is created by
        :func:`skops.hub_utils.init`.

    Attributes
    ----------
    model: estimator object
        The scikit-learn compatible model that will be documented.

    metadata: CardData
        Metadata to be stored at the beginning of the saved model card, as
        metadata to be understood by the Hugging Face Hub.

    Notes
    -----
    The contents of the sections of the template can be set using
    :meth:`Card.add` method. Plots can be added to the model card using
    :meth:`Card.add_plot`. The key you pass to :meth:`Card.add_plot` will be
    used as the header of the plot.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skops import card
    >>> X, y = load_iris(return_X_y=True)
    >>> model = LogisticRegression(random_state=0).fit(X, y)
    >>> model_card = card.Card(model)
    >>> model_card.metadata.license = "mit"
    >>> y_pred = model.predict(X)
    >>> cm = confusion_matrix(y, y_pred,labels=model.classes_)
    >>> disp = ConfusionMatrixDisplay(
    ...     confusion_matrix=cm,
    ...     display_labels=model.classes_
    ... )
    >>> disp.plot()  # doctest: +ELLIPSIS
    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at ...>
    >>> disp.figure_.savefig("confusion_matrix.png")
    ...
    >>> model_card.add_plot(confusion_matrix="confusion_matrix.png") # doctest: +ELLIPSIS
    <skops.card._model_card.Card object at ...>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     model_card.save((Path(tmpdir) / "README.md")) # doctest: +ELLIPSIS
    ...
    """

    def __init__(
        self,
        model: Any,
        model_diagram: bool = True,
        metadata: Optional[CardData] = None,
    ) -> None:
        self.model = model
        self._hyperparameter_table = self._extract_estimator_config()
        # the spaces in the pipeline breaks markdown, so we replace them
        if model_diagram is True:
            self._model_plot: str | None = re.sub(
                r"\n\s+", "", str(estimator_html_repr(model))
            )
        else:
            self._model_plot = None
        self._template_sections: dict[str, str] = {}
        self._figure_paths: dict[str, str] = {}
        self.metadata = metadata or CardData()

    def add(self, **kwargs: str) -> "Card":
        """Takes values to fill model card template.

        Parameters
        ----------
        **kwargs : dict
            Parameters to be set for the model card. These parameters
            need to be sections of the underlying `jinja` template used.

        Returns
        -------
        self : object
            Card object.
        """
        for section, value in kwargs.items():
            self._template_sections[section] = value
        return self

    def add_plot(self, **kwargs: str) -> "Card":
        """Add plots to the model card.

        Parameters
        ----------
        **kwargs : dict
            The arguments should be of the form `name=plot_path`, where `name`
            is the name of the plot and `plot_path` is the path to the plot,
            relative to the root of the project. The plots should have already
            been saved under the project's folder.

        Returns
        -------
        self : object
            Card object.
        """
        for plot_name, plot_path in kwargs.items():
            self._figure_paths[plot_name] = plot_path
        return self

    def save(self, path: str | Path) -> None:
        """Save the model card.

        This method renders the model card in markdown format and then saves it
        as the specified file.

        Parameters
        ----------
        path: str, or Path
            filepath to save your card.

        Notes
        -----
        The keys in model card metadata can be seen `here
        <https://huggingface.co/docs/hub/models-cards#model-card-metadata>`__.
        """
        root = skops.__path__

        template_sections = copy.deepcopy(self._template_sections)

        # if template path is not given, use default
        if template_sections.get("template_path") is None:
            template_sections["template_path"] = str(
                Path(root[0]) / "card" / "default_template.md"
            )

        # copying the template so that the original template is not touched/changed
        # append plot_name if any plots are provided, at the end of the template
        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copyfile(
                template_sections["template_path"],
                f"{tmpdirname}/temporary_template.md",
            )
            #  create a temporary template with the additional plots
            template_sections["template_path"] = f"{tmpdirname}/temporary_template.md"
            # add plots at the end of the template
            with open(template_sections["template_path"], "a") as template:
                for plot in self._figure_paths:
                    template.write(
                        f"\n\n{plot}\n" + f"![{plot}]({self._figure_paths[plot]})\n\n"
                    )

            card = ModelCard.from_template(
                card_data=self.metadata,
                hyperparameter_table=self._hyperparameter_table,
                model_plot=self._model_plot,
                **template_sections,
            )

        card.save(path)

    def _extract_estimator_config(self) -> str:
        """Extracts estimator hyperparameters and renders them into a vertical table.

        Returns
        -------
        str:
            Markdown table of hyperparameters.
        """

        hyperparameter_dict = self.model.get_params(deep=True)
        table = "| Hyperparameters | Value |\n| :-- | :-- |\n"
        for hyperparameter, value in hyperparameter_dict.items():
            table += f"| {hyperparameter} | {value} |\n"
        return table
