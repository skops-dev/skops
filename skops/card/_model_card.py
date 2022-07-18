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

    def add(self):
        pass

    def add_inspection(self):
        pass

    def save(self, path, card_data, **card_kwargs):
        """Fills model card template, renders and saves it as markdown file to the target directory.

        Parameters:
        ----------
        card_data: CardData
            CardData object.
        card_kwargs:
            Card kwargs are information you can pass to fill in the sections of the
            card template, e.g. model_description, citation_bibtex, get_started_code.
        """
        ROOT = skops.__path__
        card_data.library_name = "sklearn"
        template_path = card_kwargs.get("template_path")
        if template_path is None:
            template_path = os.path.join(f"{ROOT[0]}", "card", "default_template.md")
        card_kwargs["template_path"] = template_path
        card = ModelCard.from_template(
            card_data=card_data,
            hyperparameter_table=self.hyperparameter_table,
            model_plot=self.model_plot,
            **card_kwargs,
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
