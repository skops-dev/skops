import argparse

import skops.io._cli


def main_entrypoint(command_line_args=None):
    """Main entrypoint for all command line Skops methods."""
    parser = argparse.ArgumentParser(
        prog="Skops",
        description="Main entrypoint for all command line Skops methods.",
        add_help=False,
    )

    function_map = {"convert": skops.io._cli.main_convert}

    parser.add_argument(
        "function", help="Function to call.", choices=function_map.keys()
    )

    arg, _ = parser.parse_known_args()
    function_map.get(arg.function)(parent=parser)
