"""Tests for skops.utils.fixes.py"""

import tempfile
from pathlib import Path

import pytest


class TestPathUnlink:
    @pytest.fixture(scope="class")
    def path_unlink(self):
        from skops.utils.fixes import path_unlink

        return path_unlink

    @pytest.fixture(scope="class")
    def tempdir(self):
        with tempfile.TemporaryDirectory(prefix="skops-test") as directory:
            yield Path(directory)

    def test_path_unlink_file_exists(self, path_unlink, tempdir):
        path = tempdir / "some-file"
        path.touch()
        assert path.exists()

        path_unlink(path)
        assert not path.exists()

    def test_path_unlink_file_does_not_exist_raises(self, path_unlink, tempdir):
        path = tempdir / "some-file"
        assert not path.exists()

        with pytest.raises(FileNotFoundError):
            path_unlink(path)

    def test_path_unlink_file_does_not_missing_ok(self, path_unlink, tempdir):
        path = tempdir / "some-file"
        assert not path.exists()
        # does not raise an error
        path_unlink(path, missing_ok=True)
