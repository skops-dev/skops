import argparse

import skops.cli._convert


def main_cli(command_line_args=None):
    """Main command line interface entrypoint for all command line Skops methods."""
    parser = argparse.ArgumentParser(
        prog="Skops",
        description="Main entrypoint for all command line Skops methods.",
        add_help=True,
    )

    # NB: methods should be functions, parsers should be objects
    function_map = {
        "convert": {
            "method": skops.cli._convert.main,
            "parser": skops.cli._convert.get_parser,
        },
    }

    parser.add_argument(
        "function", help="Function to call.", choices=function_map.keys()
    )
    # subparsers = parser.add_subparsers(help="sub-command help")

    args, _ = parser.parse_known_args(command_line_args)
