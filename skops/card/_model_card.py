import re
import shutil
import tempfile
from pathlib import Path

from modelcards import CardData, ModelCard
from sklearn.utils import estimator_html_repr

import skops


class Card:
    """Model card class that will be used to generate model card.
    This class can be used to write information and plots to model card and save
    it. This class by default generates an interactive plot of the model and a
    table of hyperparameters. The slots to be filled are defined in the markdown
    template.
    Parameters:
    ----------
    model: estimator object
        Model that will be documented.
    model_diagram: bool, optional
        Set to True if model diagram should be plotted in the card.
    Notes
    -----
    You can pass your own custom template using ``add`` method. You can add
    plots to the model card template using ``add_plot``. The key you pass to
    ``add_plot`` will be used for header of the plot.

    Examples
    --------
    >>> from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> import matplotlib.pyplot as plt
    >>> from skops import card
    >>> X, y = load_iris(return_X_y=True)
    >>> model = LogisticRegression(random_state=0).fit(X, y)
    >>> model_card = card.Card(model)
    >>> model_card.add(license="mit") # doctest: +ELLIPSIS
    <skops.card._model_card.Card object at ...>
    >>> y_pred = model.predict(X)
    >>> cm = confusion_matrix(y, y_pred,labels=model.classes_)
    >>> disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    ... display_labels=model.classes_)
    >>> disp.plot() # doctest: +ELLIPSIS
    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at ...>
    >>> plt.savefig("confusion_matrix.png") # doctest: +ELLIPSIS
    ...
    >>> model_card.add_plot(confusion_matrix="confusion_matrix.png") # doctest: +ELLIPSIS
    <skops.card._model_card.Card object at ...>
    >>> model_card.save((Path("save_dir") / "README.md")) # doctest: +ELLIPSIS
    ...
    """

    def __init__(self, model, model_diagram=True):
        self.model = model
        self.hyperparameter_table = self._extract_estimator_config()
        # the spaces in the pipeline breaks markdown, so we replace them
        if model_diagram is True:
            self.model_plot = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
        else:
            self.model_plot = None
        self.template_sections = {}
        self._figure_paths = {}

    def add(self, **kwargs):
        """Takes values to fill model card template.
        Parameters:
        ----------
        **kwargs : dict
            Parameters to be set for the model card. These parameters
            need to be sections of the underlying `jinja` template used.

        """
        for section, value in kwargs.items():
            self.template_sections[section] = value
        return self

    def add_plot(self, **kwargs):
        """Add plots to the model card.

        Parameters:
        ----------
        **kwargs : dict
            The arguments should be of the form `name=plot_path`, where `name`
            is the name of the plot and `plot_path` is the path to the plot,
            relative to the root of the project. The plots should have already
            been saved under the project's folder.
        """
        for plot_name, plot_path in kwargs.items():
            self._figure_paths[plot_name] = plot_path
        return self

    def save(self, path):
        """Save the model card.

        This method renders the model card in mardown format and then saves it
        as the specified file.

        Parameters:
        ----------
        path: str, or Path
            filepath to save your card.

        Notes
        -----
        The keys in model card metadata can be seen
        [here](https://huggingface.co/docs/hub/models-cards#model-card-metadata).
        """
        root = skops.__path__

        metadata_keys = [
            "language",
            "license",
            "library_name",
            "tags",
            "datasets",
            "model_name",
            "metrics",
            "model-index",
        ]
        card_data_keys = {}

        # if key is supposed to be in metadata and is provided by user, write it to card_data_keys
        for key in self.template_sections.keys() & metadata_keys:
            card_data_keys[key] = self.template_sections.pop(key, "")

        # construct CardData
        card_data = CardData(**card_data_keys)
        card_data.library_name = "sklearn"

        # if template path is not given, use default
        if self.template_sections.get("template_path") is None:
            self.template_sections["template_path"] = (
                Path(root[0]) / "card" / "default_template.md"
            )

        # copying the template so that the original template is not touched/changed
        # append plot_name if any plots are provided, at the end of the template
        # if any plot is given, copy the template to a different path
        if self._figure_paths:
            with tempfile.NamedTemporaryFile() as fp:
                shutil.copyfile(
                    self.template_sections["template_path"],
                    fp.name,
                )
                #  create a temporary template with the additional plots
                self.template_sections["template_path"] = fp.name
                # add plots at the end of the template
                with open(self.template_sections["template_path"], "a") as template:
                    for plot in self._figure_paths:
                        template.write(
                            f"\n\n{plot}\n"
                            + f"![{plot}]({self._figure_paths[plot]})\n\n"
                        )
                card = ModelCard.from_template(
                    card_data=card_data,
                    hyperparameter_table=self.hyperparameter_table,
                    model_plot=self.model_plot,
                    **self.template_sections,
                )

        else:
            card = ModelCard.from_template(
                card_data=card_data,
                hyperparameter_table=self.hyperparameter_table,
                model_plot=self.model_plot,
                **self.template_sections,
            )

        card.save(path)

    def _extract_estimator_config(self):
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
