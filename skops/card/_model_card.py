import os
import re

from modelcards import ModelCard
from sklearn.utils import estimator_html_repr

import skops


class Card:
    def __init__(self, model):
        self.model = model
        self.hyperparameters_table = self.extract_estimator_config(self.model)
        self.model_plot = re.sub(r"\n\s+", "", str(estimator_html_repr(model)))
        self.template_sections = {}

    def add(self, section, value):
        """Takes values to fill model card template.
        Parameters:
        ----------
        section: str
                Section in the template.
        value: str
                Value to fill in the section.
        """
        self.template_sections[section] = value
        return self

    def add_inspection(self):
        pass

    def save(self, path, card_data):
        """Fills model card template, renders and saves it as markdown file to the target directory.

        Parameters:
        ----------
        card_data: CardData
            CardData object.
        """
        ROOT = skops.__path__
        card_data.library_name = "sklearn"
        template_path = self.template_sections.get("template_path")
        if template_path is None:
            template_path = os.path.join(f"{ROOT[0]}", "card", "default_template.md")
        self.template_sections["template_path"] = template_path
        card = ModelCard.from_template(
            card_data=card_data,
            hyperparameter_table=self.hyperparameter_table,
            model_plot=self.model_plot,
            **self.template_sections,
        )
        card.save(path)

    def extract_estimator_config(model):
        """Extracts estimator configuration and renders them into a vertical table.

        Parameters
        ----------
            model (estimator): scikit-learn pipeline or model.

        Returns
        -------
        str:
            Markdown table of hyperparameters.
        """

        hyperparameter_dict = model.get_params(deep=True)
        table = "| Hyperparameters | Value |\n| :-- | :-- |\n"
        for hyperparameter, value in hyperparameter_dict.items():
            table += f"| {hyperparameter} | {value} |\n"
        return table
