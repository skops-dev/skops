# This file includes fixes which are usually required to handle multiple
# versions of a dependency.

import sys
from contextlib import suppress
from pathlib import Path

PYTHON_VERSION = sys.version_info


try:
    # py>=3.8
    from importlib import metadata  # noqa
except ImportError:
    # older pythons
    import importlib_metadata as metadata  # noqa

try:
    # py>=3.8
    from typing import Literal  # noqa
except ImportError:
    from typing_extensions import Literal  # noqa


def path_unlink(path: Path, missing_ok=False) -> None:
    """Remove this file or symbolic link

    Parameters
    ----------
    path : pathlib.Path
      Path to the file to be removed

    missing_ok : bool (default=False)
      If False, ``FileNotFoundError`` is raised if the path does not exist. If
      True, ``FileNotFoundError`` exceptions will be ignored (same behavior as
      the POSIX ``rm -f`` command).

    Raises
    ------
    FileNotFoundError
      Is raised if ``missing_ok`` is False and the file is missing.

    """
    # Python 3.7 does not support the missing_ok argument.
    # One we move to Python >= 3.8, this function can just call
    # Path.unlink(missing_ok)
    if not missing_ok:  # default behavior
        path.unlink()
        return

    if PYTHON_VERSION >= (3, 8):
        path.unlink(missing_ok=missing_ok)
        return

    # for Python 3.7, just catch the error
    with suppress(FileNotFoundError):
        path.unlink()
