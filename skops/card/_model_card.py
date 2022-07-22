import os
import re
import shutil

from modelcards import CardData, ModelCard
from sklearn.utils import estimator_html_repr

import skops


class Card:
    def __init__(self, model):
        self.model = model
        self.hyperparameter_table = self._extract_estimator_config()
        # the spaces in the pipeline breaks markdown, so we replace them
        self.model_plot = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
        self.template_sections = {}
        self.figure_paths = {}

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
            self.figure_paths[plot_name] = plot_path
        return self

    def save(self, path):
         """Save the model card.

        This method renders the model card in mardown format and then saves it
        as the specified file.

        Parameters:
        ----------
        path: Path
              filepath to save your card.
        """
        ROOT = skops.__path__

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

        if self.template_sections.get("template_path") is None:
            self.template_sections["template_path"] = os.path.join(
                f"{ROOT[0]}", "card", "default_template.md"
            )

        # append plot_name if any plots are provided, at the end of the template
        if self.figure_paths:
            shutil.copyfile(
                self.template_sections["template_path"],
                f"{ROOT[0]}/temporary_template.md",
            )
            self.template_sections["template_path"] = f"{ROOT[0]}/temporary_template.md"
            template = open(self.template_sections["template_path"], "a")

            for plot in self.figure_paths:
                template.write(
                    f"\n\n{plot}\n" + f"![{plot}]({self.figure_paths[plot]})\n\n"
                )
            template.close()

        card = ModelCard.from_template(
            card_data=card_data,
            hyperparameter_table=self.hyperparameter_table,
            model_plot=self.model_plot,
            **self.template_sections,
        )
        card.save(path)

    def _extract_estimator_config(self):
        """Extracts estimator hyperparameters and renders them into a vertical table.

        Parameters
        ----------
            model (estimator): scikit-learn pipeline or model.

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
