import argparse

import skops.cli._convert


def main_entrypoint(command_line_args=None):
    """Main entrypoint for all command line Skops methods."""
    parser = argparse.ArgumentParser(
        prog="Skops",
        description="Main entrypoint for all command line Skops methods.",
        add_help=False,
    )

    function_map = {"convert": skops.cli._convert.main_convert}
    parser.add_argument(
        "function", help="Function to call.", choices=function_map.keys()
    )
    args, _ = parser.parse_known_args(command_line_args)
    function_map.get(args.function)(parent=parser, command_line_args=command_line_args)
