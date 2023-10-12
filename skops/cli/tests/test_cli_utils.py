import logging

import pytest

from skops.cli._utils import get_log_level


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (3, logging.DEBUG),
        (4, logging.DEBUG),
        (-1, logging.WARNING),
    ],
)
def test_get_log_level(level: int, expected: int):
    assert get_log_level(level) == expected
