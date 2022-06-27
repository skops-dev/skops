# This file includes fixes which are usually required to handle multiple
# versions of a dependency.

try:
    # py>=3.8
    from importlib import metadata  # noqa
except ImportError:
    # older pythons
    import importlib_metadata as metadata  # noqa
