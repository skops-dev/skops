# This file includes fixes which are usually required to handle multiple
# versions of a dependency.

import sys

if sys.version_info >= (3, 8):
    # py>=3.8
    from importlib import metadata  # noqa
else:
    # older pythons
    import importlib_metadata as metadata  # noqa

if sys.version_info >= (3, 8):
    # py>=3.8
    from typing import Literal  # noqa
else:
    # older pythons, this requires typing_extensions to be installed.
    # if you're removing this, you should also remove the dependency from
    # _min_dependencies.py
    from typing_extensions import Literal  # noqa
