import sys

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.10.0"

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of skops when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SKOPS_SETUP__'
    __SKOPS_SETUP__  # type: ignore
except NameError:
    __SKOPS_SETUP__ = False

if __SKOPS_SETUP__:
    sys.stderr.write("Partial import of the library during the build process.\n")
    # We are not importing the rest of the library during the build
    # process, as it may not be compiled yet or cause immature import
