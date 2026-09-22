"""Helpers for the release workflow in ``.github/workflows/publish-pypi.yml``.

The workflow drives a release end to end. This script holds the pieces that are
easier to get right in Python than in shell: parsing the version, deriving the
branch, tag and changelog section names from it, checking the changelog, editing
``skops/__init__.py`` and ``docs/changes.rst``, and checking that the files on
(Test)PyPI are the ones built by the workflow.

Every command prints ``key=value`` lines, which the workflow appends to
``$GITHUB_OUTPUT``. Commands that read or edit files find them relative to the
current working directory, so run the script from the repository root; the
script itself can live anywhere::

    python scripts/release.py meta 0.16.0
    python scripts/release.py check-changelog 0.16.0
    python scripts/release.py set-version 0.16.0
    python scripts/release.py start-dev 0.17
    python scripts/release.py verify-upload 0.16.0 dist/*
    python scripts/release.py verify-upload 0.16.0 --index-url https://test.pypi.org \\
        dist/*
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

VERSION_LINE_RE = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
# A changelog section heading: the tag name on one line, underlined with dashes.
SECTION_RE = re.compile(r"^(?P<name>v\d+\.\d+(?:\.\d+)?)\n-+\n", re.MULTILINE)
# A version component in its canonical form: no leading zeros.
COMPONENT = r"(0|[1-9]\d*)"


@dataclass(frozen=True)
class Release:
    """A final release version and the names derived from it."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version: str) -> Release:
        match = re.fullmatch(rf"{COMPONENT}\.{COMPONENT}\.{COMPONENT}", version)
        if match is None:
            raise SystemExit(
                f"Invalid release version {version!r}: expected MAJOR.MINOR.PATCH"
                " without leading zeros, e.g. 0.16.0 for a new minor release or"
                " 0.16.1 for a bug fix release."
            )
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major, minor, patch)

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def is_patch(self) -> bool:
        """Whether this is a bug fix release on an existing release branch."""
        return self.patch != 0

    @property
    def branch(self) -> str:
        """The release branch, shared by all releases of a minor version."""
        return f"{self.major}.{self.minor}.X"

    @property
    def tag(self) -> str:
        """The git tag: ``v0.16`` for ``0.16.0`` and ``v0.16.1`` for ``0.16.1``."""
        if self.is_patch:
            return f"v{self.version}"
        return f"v{self.major}.{self.minor}"

    @property
    def section(self) -> str:
        """The changelog section heading, which is the same as the tag."""
        return self.tag

    @property
    def next_minor(self) -> str:
        """The minor version that development on ``main`` moves on to."""
        return f"{self.major}.{self.minor + 1}"


def repo_root() -> Path:
    """The current directory, which must be the root of the repository."""
    root = Path.cwd()
    if not (init_file(root).is_file() and changelog(root).is_file()):
        raise SystemExit(
            f"{root} is not the root of the skops repository; run this script from"
            " the repository root."
        )
    return root


def init_file(root: Path) -> Path:
    return root / "skops" / "__init__.py"


def changelog(root: Path) -> Path:
    return root / "docs" / "changes.rst"


def read_version(root: Path) -> str:
    match = VERSION_LINE_RE.search(init_file(root).read_text())
    if match is None:
        raise SystemExit(f"Could not find the __version__ line in {init_file(root)}")
    return match["version"]


def write_version(root: Path, version: str) -> bool:
    """Set ``__version__`` in ``skops/__init__.py``; return whether it changed."""
    if read_version(root) == version:
        return False
    path = init_file(root)
    text = path.read_text()
    path.write_text(VERSION_LINE_RE.sub(f'__version__ = "{version}"', text))
    return True


def changelog_sections(text: str) -> dict[str, str]:
    """Map each section heading of the changelog to the text below it."""
    matches = list(SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[match["name"]] = text[match.end() : end]
    return sections


def check_changelog(root: Path, release: Release) -> None:
    """Fail unless the changelog has a non-empty section for the release."""
    path = changelog(root)
    sections = changelog_sections(path.read_text())
    body = sections.get(release.section)
    if body is None:
        newest = ", ".join(list(sections)[:3]) or "none"
        raise SystemExit(
            f"{path} has no '{release.section}' section for release"
            f" {release.version}; the newest sections are: {newest}."
        )
    if not body.strip():
        raise SystemExit(
            f"The '{release.section}' section of {path} is empty; describe the"
            f" changes in release {release.version} before releasing it."
        )


def start_dev(root: Path, next_minor: str) -> dict[str, str]:
    """Move ``main`` on to the next development version.

    Sets ``__version__`` to ``MAJOR.MINOR.dev0`` and adds an empty changelog
    section for it, unless either is already in place.
    """
    if re.fullmatch(rf"{COMPONENT}\.{COMPONENT}", next_minor) is None:
        raise SystemExit(
            f"Invalid development version {next_minor!r}: expected MAJOR.MINOR"
            " without leading zeros, e.g. 0.17."
        )
    version = f"{next_minor}.dev0"
    section = f"v{next_minor}"
    version_changed = write_version(root, version)

    path = changelog(root)
    text = path.read_text()
    changelog_changed = section not in changelog_sections(text)
    if changelog_changed:
        first = SECTION_RE.search(text)
        if first is None:
            raise SystemExit(f"Could not find any release section in {path}")
        heading = f"{section}\n{'-' * len(section)}\n\n"
        path.write_text(text[: first.start()] + heading + text[first.start() :])

    return {
        "version": version,
        "section": section,
        "changed": str(version_changed or changelog_changed).lower(),
    }


def fetch_digests(index_url: str, version: str) -> dict[str, str] | None:
    """Map file names of a release on a PyPI-like index to their sha256 digests.

    Returns ``None`` while the index does not know the release yet.
    """
    url = f"{index_url}/pypi/skops/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return {file["filename"]: file["digests"]["sha256"] for file in data["urls"]}


def verify_upload(
    index_url: str,
    version: str,
    files: list[Path],
    attempts: int = 10,
    wait: float = 30,
    fetch: Callable[[str, str], dict[str, str] | None] = fetch_digests,
) -> None:
    """Fail unless ``files`` are exactly the files of the release on the index.

    The index can take a moment to show a release, so it is polled ``attempts``
    times, ``wait`` seconds apart. A file that differs from its counterpart on
    the index means an earlier run uploaded this version already. File names
    can never be reused on a PyPI index, so that version cannot be re-released
    from different sources.
    """
    for attempt in range(1, attempts + 1):
        digests = fetch(index_url, version)
        if digests is not None:
            break
        if attempt == attempts:
            raise SystemExit(
                f"skops {version} did not show up on {index_url} after {attempts}"
                " attempts."
            )
        print(f"skops {version} is not on {index_url} yet, retrying in {wait:g}s")
        time.sleep(wait)
    else:  # pragma: no cover - attempts < 1
        raise SystemExit("attempts must be at least 1")

    for path in files:
        local = hashlib.sha256(path.read_bytes()).hexdigest()
        remote = digests.get(path.name)
        if remote is None:
            raise SystemExit(
                f"{path.name} is missing from skops {version} on {index_url}; the"
                f" index has: {', '.join(sorted(digests)) or 'nothing'}."
            )
        if remote != local:
            raise SystemExit(
                f"{path.name} on {index_url} is not the file built by this run"
                f" (sha256 {remote} there, {local} here). An earlier run uploaded a"
                f" different skops {version}. File names cannot be reused on a PyPI"
                " index, so release a new version instead."
            )
        print(f"{path.name} matches the file on {index_url}")

    unexpected = sorted(set(digests) - {path.name for path in files})
    if unexpected:
        raise SystemExit(
            f"skops {version} on {index_url} has files this run did not build:"
            f" {', '.join(unexpected)}. An earlier run uploaded a different skops"
            f" {version}. File names cannot be reused on a PyPI index, so release a"
            " new version instead."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    meta = subparsers.add_parser(
        "meta", help="print the branch, tag and changelog section of a release"
    )
    meta.add_argument("version")

    check = subparsers.add_parser(
        "check-changelog",
        help="fail unless the changelog has a non-empty section for the release",
    )
    check.add_argument("version")

    set_version = subparsers.add_parser(
        "set-version", help="set __version__ to the given release version"
    )
    set_version.add_argument("version")

    dev = subparsers.add_parser(
        "start-dev",
        help="set __version__ to MAJOR.MINOR.dev0 and add a changelog section for it",
    )
    dev.add_argument("next_minor", metavar="MAJOR.MINOR")

    verify = subparsers.add_parser(
        "verify-upload",
        help="fail unless the given files are the files of the release on an index",
    )
    verify.add_argument("version")
    verify.add_argument("files", nargs="+", type=Path)
    verify.add_argument(
        "--index-url",
        default="https://pypi.org",
        help="https://pypi.org (default) or https://test.pypi.org",
    )
    verify.add_argument("--attempts", type=int, default=10)
    verify.add_argument("--wait", type=float, default=30, help="seconds")

    args = parser.parse_args(argv)
    outputs: dict[str, str]
    if args.command == "meta":
        release = Release.parse(args.version)
        outputs = {
            "version": release.version,
            "branch": release.branch,
            "tag": release.tag,
            "section": release.section,
            "is_patch": str(release.is_patch).lower(),
            "next_minor": release.next_minor,
        }
    elif args.command == "check-changelog":
        check_changelog(repo_root(), Release.parse(args.version))
        outputs = {}
    elif args.command == "set-version":
        release = Release.parse(args.version)
        outputs = {"changed": str(write_version(repo_root(), release.version)).lower()}
    elif args.command == "start-dev":
        outputs = start_dev(repo_root(), args.next_minor)
    else:
        release = Release.parse(args.version)
        verify_upload(
            args.index_url.rstrip("/"),
            release.version,
            args.files,
            attempts=args.attempts,
            wait=args.wait,
        )
        outputs = {}

    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
