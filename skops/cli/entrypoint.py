import argparse

import skops.cli._convert
import skops.cli._update


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
    #   method: the command to call (gets set to default 'func')
    #   format_parser: the function used to create a subparser for that command
    function_map = {
        "convert": {
            "method": skops.cli._convert.main,
            "format_parser": skops.cli._convert.format_parser,
        },
        "update": {
            "method": skops.cli._update.main,
            "format_parser": skops.cli._update.format_parser,
        },
    }

    for func_name, values in function_map.items():
        # Add subparser for each function in func map,
        # and assigns default func to be "method" from function_map
        subparser = subparsers.add_parser(func_name)
        subparser.set_defaults(func=values["method"])
        values["format_parser"](subparser)

    # Parse arguments with arg parser for given function in function map,
    # Then call the matching method in the function_map with the argument namespace
    args = entry_parser.parse_args(command_line_args)
    args.func(args)
