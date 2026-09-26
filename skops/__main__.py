"""Allow ``python -m skops`` as an alternative to the ``skops`` command."""

from skops.cli.entrypoint import main_cli

if __name__ == "__main__":
    main_cli()
