import os
from pathlib import Path

HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", None)


def _get_cwd():
    """Return the current working directory.

    Only works if we're using pytest.
    """
    return Path(os.getenv("PYTEST_CURRENT_TEST").split("::")[0]).parent


def test_write_model_card():
    # Check if the model card is written
    pass


def test_hyperparameter_table():
    # Check if table is in model card
    pass


def test_hyperparameter_table_search():
    # Check if hyperparameters are extracted for non-trivial hyperparameters are there
    pass


def test_plot_model():
    # Check if model plot is in model card
    pass
