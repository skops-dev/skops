"""Tests for ``scripts/release.py``, the helper of the release workflow."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest
import release
from release import Release

CHANGELOG = """\
.. _changelog:

skops Changelog
===============

.. contents:: Table of Contents
    :depth: 1
    :local:

v0.16
-----
- Something new. :pr:`541` by `Adrin Jalali`_.

v0.15
-----
- Something old. :pr:`537` by `Adrin Jalali`_.
"""

INIT = '# The version.\n__version__ = "0.16.dev0"\n'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal repository with the two files the script edits."""
    release.init_file(tmp_path).parent.mkdir()
    release.init_file(tmp_path).write_text(INIT)
    release.changelog(tmp_path).parent.mkdir()
    release.changelog(tmp_path).write_text(CHANGELOG)
    return tmp_path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_release_minor() -> None:
    rel = Release.parse("0.16.0")
    assert rel.version == "0.16.0"
    assert not rel.is_patch
    assert rel.branch == "0.16.X"
    assert rel.tag == "v0.16"
    assert rel.section == "v0.16"
    assert rel.next_minor == "0.17"


def test_release_patch() -> None:
    rel = Release.parse("0.16.1")
    assert rel.version == "0.16.1"
    assert rel.is_patch
    assert rel.branch == "0.16.X"
    assert rel.tag == "v0.16.1"
    assert rel.section == "v0.16.1"
    assert rel.next_minor == "0.17"


def test_release_major() -> None:
    rel = Release.parse("1.0.0")
    assert rel.branch == "1.0.X"
    assert rel.tag == "v1.0"
    assert rel.next_minor == "1.1"


@pytest.mark.parametrize(
    "version",
    [
        "0.16",
        "v0.16.0",
        "0.16.0rc1",
        "0.16.dev0",
        "0.16.0.1",
        " 0.16.0",
        "0.16.0\n",
        # Not canonical: the workflow uses the version as typed.
        "0.016.0",
        "00.16.0",
        "0.16.00",
    ],
)
def test_release_rejects_invalid_versions(version: str) -> None:
    with pytest.raises(SystemExit, match="Invalid release version"):
        Release.parse(version)


def test_check_changelog(root: Path) -> None:
    release.check_changelog(root, Release.parse("0.16.0"))
    with pytest.raises(SystemExit, match="has no 'v0.17' section"):
        release.check_changelog(root, Release.parse("0.17.0"))
    with pytest.raises(SystemExit, match="has no 'v0.16.1' section"):
        release.check_changelog(root, Release.parse("0.16.1"))


def test_check_changelog_empty_section(root: Path) -> None:
    release.start_dev(root, "0.17")
    with pytest.raises(SystemExit, match="'v0.17' section of .* is empty"):
        release.check_changelog(root, Release.parse("0.17.0"))


def test_write_version(root: Path) -> None:
    assert release.read_version(root) == "0.16.dev0"
    assert release.write_version(root, "0.16.0") is True
    assert release.read_version(root) == "0.16.0"
    assert release.write_version(root, "0.16.0") is False
    # Only the version line changes.
    assert release.init_file(root).read_text() == INIT.replace("0.16.dev0", "0.16.0")


def test_start_dev(root: Path) -> None:
    outputs = release.start_dev(root, "0.17")
    assert outputs == {"version": "0.17.dev0", "section": "v0.17", "changed": "true"}
    assert release.read_version(root) == "0.17.dev0"
    text = release.changelog(root).read_text()
    assert "    :local:\n\nv0.17\n-----\n\nv0.16\n-----\n" in text
    assert list(release.changelog_sections(text)) == ["v0.17", "v0.16", "v0.15"]

    # Running it again changes nothing.
    outputs = release.start_dev(root, "0.17")
    assert outputs["changed"] == "false"
    assert release.read_version(root) == "0.17.dev0"
    assert release.changelog(root).read_text() == text


def test_start_dev_rejects_invalid_versions(root: Path) -> None:
    for next_minor in ["0.17.0", "0.017", "v0.17", "17"]:
        with pytest.raises(SystemExit, match="Invalid development version"):
            release.start_dev(root, next_minor)
    assert release.read_version(root) == "0.16.dev0"


def test_repo_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(root)
    assert release.repo_root().resolve() == root.resolve()
    monkeypatch.chdir(root / "docs")
    with pytest.raises(SystemExit, match="not the root of the skops repository"):
        release.repo_root()


@pytest.fixture
def dist(tmp_path: Path) -> list[Path]:
    wheel = tmp_path / "skops-0.16.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    sdist = tmp_path / "skops-0.16.0.tar.gz"
    sdist.write_bytes(b"sdist")
    return [wheel, sdist]


def test_verify_upload(dist: list[Path]) -> None:
    calls: list[tuple[str, str]] = []

    def fetch(index_url: str, version: str) -> dict[str, str]:
        calls.append((index_url, version))
        return {path.name: sha256(path.read_bytes()) for path in dist}

    release.verify_upload(
        "https://test.pypi.org", "0.16.0", dist, attempts=1, fetch=fetch
    )
    assert calls == [("https://test.pypi.org", "0.16.0")]


def test_verify_upload_waits_for_the_index(dist: list[Path]) -> None:
    responses: list[dict[str, str] | None] = [
        None,
        None,
        {path.name: sha256(path.read_bytes()) for path in dist},
    ]

    def fetch(index_url: str, version: str) -> dict[str, str] | None:
        return responses.pop(0)

    release.verify_upload(
        "https://pypi.org", "0.16.0", dist, attempts=3, wait=0, fetch=fetch
    )
    assert responses == []


def test_verify_upload_gives_up(dist: list[Path]) -> None:
    def fetch(index_url: str, version: str) -> None:
        return None

    with pytest.raises(SystemExit, match="did not show up .* after 2 attempts"):
        release.verify_upload(
            "https://pypi.org", "0.16.0", dist, attempts=2, wait=0, fetch=fetch
        )


def test_verify_upload_rejects_different_files(dist: list[Path]) -> None:
    def fetch(index_url: str, version: str) -> dict[str, str]:
        return {dist[0].name: sha256(b"an earlier build"), dist[1].name: sha256(b"")}

    with pytest.raises(SystemExit, match="is not the file built by this run"):
        release.verify_upload("https://pypi.org", "0.16.0", dist, fetch=fetch)


def test_verify_upload_rejects_missing_files(dist: list[Path]) -> None:
    def fetch(index_url: str, version: str) -> dict[str, str]:
        return {dist[0].name: sha256(dist[0].read_bytes())}

    with pytest.raises(SystemExit, match="skops-0.16.0.tar.gz is missing from"):
        release.verify_upload("https://pypi.org", "0.16.0", dist, fetch=fetch)


def test_verify_upload_rejects_extra_files(dist: list[Path]) -> None:
    def fetch(index_url: str, version: str) -> dict[str, str]:
        digests = {path.name: sha256(path.read_bytes()) for path in dist}
        digests["skops-0.16.0-py3-none-win_amd64.whl"] = sha256(b"an earlier build")
        return digests

    with pytest.raises(
        SystemExit,
        match="has files this run did not build: skops-0.16.0-py3-none-win_amd64.whl",
    ):
        release.verify_upload("https://pypi.org", "0.16.0", dist, fetch=fetch)


def test_fetch_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "urls": [
            {"filename": "skops-0.16.0.tar.gz", "digests": {"sha256": "abc"}},
            {"filename": "skops-0.16.0-py3-none-any.whl", "digests": {"sha256": "def"}},
        ]
    }
    urls: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    digests = release.fetch_digests("https://test.pypi.org", "0.16.0")
    assert digests == {
        "skops-0.16.0.tar.gz": "abc",
        "skops-0.16.0-py3-none-any.whl": "def",
    }
    assert urls == ["https://test.pypi.org/pypi/skops/0.16.0/json"]


def test_fetch_digests_unknown_release(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(
            url, 404, "Not Found", email.message.Message(), None
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert release.fetch_digests("https://pypi.org", "0.16.0") is None


def test_fetch_digests_other_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(
            url, 503, "Service Unavailable", email.message.Message(), None
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        release.fetch_digests("https://pypi.org", "0.16.0")


def test_main_meta(capsys: pytest.CaptureFixture[str]) -> None:
    release.main(["meta", "0.16.1"])
    assert capsys.readouterr().out == (
        "version=0.16.1\nbranch=0.16.X\ntag=v0.16.1\nsection=v0.16.1\n"
        "is_patch=true\nnext_minor=0.17\n"
    )


def test_main_edits_the_current_directory(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(root)
    release.main(["check-changelog", "0.16.0"])
    release.main(["set-version", "0.16.0"])
    assert capsys.readouterr().out == "changed=true\n"
    assert release.read_version(root) == "0.16.0"
    release.main(["start-dev", "0.17"])
    assert capsys.readouterr().out == "version=0.17.dev0\nsection=v0.17\nchanged=true\n"
    assert release.read_version(root) == "0.17.dev0"
