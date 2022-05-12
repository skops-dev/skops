from pathlib import Path
from typing import Union


def create_model_card(*, path: Union[str, Path], model):
    """Write a model card for the model and save it.

    This function takes the path to the repo and a model, generate
    a model card and saves it in the repository.

    Parameters
    ----------
    path: str, or Path
        Path of the model card.

    model: a scikit-learn model.

    Returns
    -------
    None
    """

    pass
