import argparse

import skops.cli._convert


def main_cli(command_line_args=None):
    """Main command line interface entrypoint for all command line Skops methods.

    To add a new entrypoint:
        1. Create a new method to call that accepts a namespace
        2. Create a new subparser formatter to define the expected CL arguments
        3. Add those to the function map.
    """
    entry_parser = argparse.ArgumentParser(
        prog="Skops",
        description="Main entrypoint for all command line Skops methods.",
        add_help=True,
    )

    subparsers = entry_parser.add_subparsers(
        title="Commands",
        description="Skops command to call",
        dest="cmd",
        help="Sub-commands help",
    )

    # function_map should map a command to
    #   method: the command to call
    #   format_parser: the function used to create a subparser for that command
    function_map = {
        "convert": {
            "method": skops.cli._convert.main,
            "format_parser": skops.cli._convert.format_parser,
        },
    }

    for func_name, values in function_map.items():
        subparser = subparsers.add_parser(func_name)
        subparser.set_defaults(func=values["method"])
        values["format_parser"](subparser)

    args = entry_parser.parse_args(command_line_args)
    args.func(args)
