import io
import json
import logging
import pathlib
import zipfile
from functools import partial
from unittest import mock

import numpy as np
import pytest

from skops.cli import _update
from skops.io import _persist, _protocol, dump, load
from skops.io.exceptions import UntrustedTypesFoundException


class TestUpdate:
    model_name = "some_model_name"

    @pytest.fixture
    def safe_obj(self) -> np.ndarray:
        return np.ndarray([1, 2, 3, 4])

    @pytest.fixture
    def skops_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / f"{self.model_name}.skops"

    @pytest.fixture
    def new_skops_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / f"{self.model_name}-new.skops"

    @pytest.fixture
    def dump_file(self, skops_path: pathlib.Path, safe_obj: np.ndarray):
        """Dump an object using the current protocol version."""
        dump(safe_obj, skops_path)

    @pytest.fixture
    @mock.patch(
        "skops.io._persist.SaveContext", partial(_persist.SaveContext, protocol=0)
    )
    def dump_old_file(self, skops_path: pathlib.Path, safe_obj: np.ndarray):
        """Dump an object using an old protocol version so that the file needs
        updating.
        """
        dump(safe_obj, skops_path)

    @pytest.fixture
    @mock.patch(
        "skops.io._persist.SaveContext",
        partial(_persist.SaveContext, protocol=_protocol.PROTOCOL + 1),
    )
    def dump_new_file(self, skops_path: pathlib.Path, safe_obj: np.ndarray):
        """Dump an object using an new protocol version so that the file cannot be
        updated.
        """
        dump(safe_obj, skops_path)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_base_case_works_as_expected(
        self,
        skops_path: pathlib.Path,
        new_skops_path: pathlib.Path,
        inplace: bool,
        dump_old_file,
        safe_obj,
    ):
        mock_logger = mock.MagicMock()
        expected_output_file = new_skops_path if not inplace else skops_path

        _update._update_file(
            input_file=skops_path,
            output_file=new_skops_path if not inplace else None,
            inplace=inplace,
            logger=mock_logger,
        )
        updated_obj = load(expected_output_file)

        assert np.array_equal(updated_obj, safe_obj)

        # Check logging messages
        mock_logger.info.assert_called_once_with(
            f"Updated skops file written to {expected_output_file}"
        )
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.debug.assert_not_called()

    @mock.patch("skops.cli._update.dump")
    def test_only_diff(
        self, mock_dump: mock.MagicMock, skops_path: pathlib.Path, dump_old_file
    ):
        mock_logger = mock.MagicMock()
        _update._update_file(
            input_file=skops_path,
            output_file=None,
            inplace=False,
            logger=mock_logger,
        )
        mock_dump.assert_not_called()

        # Check logging messages
        mock_logger.warning.assert_called_once_with(
            f"File can be updated to the current protocol: {_protocol.PROTOCOL}. Please"
            " specify an output file path or use the `inplace` flag to create the"
            " updated Skops file."
        )
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.debug.assert_not_called()

    def test_no_update_same_protocol(
        self,
        skops_path: pathlib.Path,
        new_skops_path: pathlib.Path,
        dump_file,
    ):
        mock_logger = mock.MagicMock()
        _update._update_file(
            input_file=skops_path,
            output_file=new_skops_path,
            inplace=False,
            logger=mock_logger,
        )
        mock_logger.warning.assert_called_once_with(
            "File was not updated because already up to date with the current protocol:"
            f" {_protocol.PROTOCOL}"
        )
        assert not new_skops_path.exists()

    def test_no_update_newer_protocol(
        self,
        skops_path: pathlib.Path,
        new_skops_path: pathlib.Path,
        dump_new_file,
    ):
        mock_logger = mock.MagicMock()
        _update._update_file(
            input_file=skops_path,
            output_file=new_skops_path,
            inplace=False,
            logger=mock_logger,
        )
        mock_logger.warning.assert_called_once_with(
            "File cannot be updated because its protocol is more recent than the "
            f"current protocol: {_protocol.PROTOCOL}"
        )
        assert not new_skops_path.exists()

    def test_error_with_output_file_and_inplace(
        self, skops_path: pathlib.Path, new_skops_path: pathlib.Path, dump_file
    ):
        with pytest.raises(ValueError):
            _update._update_file(
                input_file=skops_path, output_file=new_skops_path, inplace=True
            )

    @pytest.fixture
    def malicious_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """A crafted skops file that tries to run ``os.system`` via a LossNode.

        See the CVE-style report: an attacker can declare an arbitrary callable
        as a LossNode constructor. ``get_untrusted_types`` surfaces it (here as
        ``os.system``), and blindly trusting that output would execute it.
        """
        schema = {
            "__class__": "system",
            "__module__": "os",
            "__loader__": "LossNode",
            "__reduce__": {
                "args": {
                    "__class__": "tuple",
                    "__module__": "builtins",
                    "__loader__": "TupleNode",
                    "content": [
                        {
                            "__class__": "str",
                            "__module__": "builtins",
                            "__loader__": "JsonNode",
                            "content": '"echo pwned"',
                            "is_json": True,
                        }
                    ],
                }
            },
            "content": {
                "__class__": "dict",
                "__module__": "builtins",
                "__loader__": "DictNode",
                "content": {},
                "key_types": {
                    "__class__": "list",
                    "__module__": "builtins",
                    "__loader__": "ListNode",
                    "content": [],
                },
            },
            "protocol": _protocol.PROTOCOL,
            "_skops_version": "0.14.0",
        }
        path = tmp_path / "malicious.skops"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("schema.json", json.dumps(schema))
        path.write_bytes(buf.getvalue())
        return path

    def test_refuses_untrusted_types_by_default(
        self,
        malicious_path: pathlib.Path,
        new_skops_path: pathlib.Path,
    ):
        """``skops update`` must not auto-trust the types declared in the file.

        This is the core of the RCE fix: previously the CLI fed
        ``get_untrusted_types()`` straight back into ``load()``, so a file could
        smuggle any type (e.g. ``os.system``) into the trusted set with no human
        review. Now untrusted types are refused unless the operator lists them
        explicitly via ``--trusted``.
        """
        with pytest.raises(UntrustedTypesFoundException, match="os.system"):
            _update._update_file(
                input_file=malicious_path,
                output_file=new_skops_path,
            )
        assert not new_skops_path.exists()


class TestMain:
    @pytest.fixture
    def tmp_logger(self) -> logging.Logger:
        return logging.getLogger()

    @pytest.mark.parametrize("output_flag", ["-o", "--output"])
    @mock.patch("skops.cli._update._update_file")
    def test_output_argument(
        self, mock_update: mock.MagicMock, output_flag: str, tmp_logger: logging.Logger
    ):
        input_path = "abc.skops"
        output_path = "abc-new.skops"
        namespace, _ = _update.format_parser().parse_known_args(
            [input_path, output_flag, output_path]
        )

        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=pathlib.Path(output_path),
            inplace=False,
            trusted=[],
            logger=tmp_logger,
        )

    @mock.patch("skops.cli._update._update_file")
    def test_inplace_argument(
        self, mock_update: mock.MagicMock, tmp_logger: logging.Logger
    ):
        input_path = "abc.skops"
        namespace, _ = _update.format_parser().parse_known_args(
            [input_path, "--inplace"]
        )

        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=None,
            inplace=True,
            trusted=[],
            logger=tmp_logger,
        )

    @mock.patch("skops.cli._update._update_file")
    def test_trusted_argument(
        self, mock_update: mock.MagicMock, tmp_logger: logging.Logger
    ):
        input_path = "abc.skops"
        output_path = "abc-new.skops"
        namespace, _ = _update.format_parser().parse_known_args(
            [input_path, "-o", output_path, "--trusted", "foo.Bar", "baz.Qux"]
        )

        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=pathlib.Path(output_path),
            inplace=False,
            trusted=["foo.Bar", "baz.Qux"],
            logger=tmp_logger,
        )

    @mock.patch("skops.cli._update._update_file")
    @pytest.mark.parametrize(
        "verbosity, expected_level",
        [
            ("", logging.WARNING),
            ("-v", logging.INFO),
            ("--verbose", logging.INFO),
            ("-vv", logging.DEBUG),
            ("-v -v", logging.DEBUG),
            ("-vvvvv", logging.DEBUG),
            ("--verbose --verbose", logging.DEBUG),
        ],
    )
    def test_given_log_levels_works_as_expected(
        self,
        mock_update: mock.MagicMock,
        verbosity: str,
        expected_level: int,
        tmp_logger: logging.Logger,
    ):
        input_path = "abc.skops"
        output_path = "def.skops"
        args = [input_path, "--output", output_path, *verbosity.split()]

        namespace, _ = _update.format_parser().parse_known_args(args)
        _update.main(namespace, tmp_logger)
        mock_update.assert_called_once_with(
            input_file=pathlib.Path(input_path),
            output_file=pathlib.Path(output_path),
            inplace=False,
            trusted=[],
            logger=tmp_logger,
        )
        assert tmp_logger.getEffectiveLevel() == expected_level
