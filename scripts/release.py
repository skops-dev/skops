"""Helpers for the release workflow in ``.github/workflows/publish-pypi.yml``.

The workflow drives a release end to end. This script holds the pieces that are
easier to get right in Python than in shell: parsing the version, deriving the
branch, tag and changelog section names from it, checking the changelog, and
editing ``skops/__init__.py`` and ``docs/changes.rst``.

Every command prints ``key=value`` lines, which the workflow appends to
``$GITHUB_OUTPUT``. Run it from the repository root::

    python scripts/release.py meta 0.16.0
    python scripts/release.py check-changelog 0.16.0
    python scripts/release.py set-version 0.16.0
    python scripts/release.py start-dev 0.17
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_FILE = ROOT / "skops" / "__init__.py"
CHANGELOG = ROOT / "docs" / "changes.rst"

VERSION_LINE_RE = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
# A changelog section heading: the tag name on one line, underlined with dashes.
SECTION_RE = re.compile(r"^(?P<name>v\d+\.\d+(?:\.\d+)?)\n-+\n", re.MULTILINE)


@dataclass(frozen=True)
class Release:
    """A final release version and the names derived from it."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version: str) -> Release:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
        if match is None:
            raise SystemExit(
                f"Invalid release version {version!r}: expected MAJOR.MINOR.PATCH,"
                " e.g. 0.16.0 for a new minor release or 0.16.1 for a bug fix"
                " release."
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


def read_version() -> str:
    match = VERSION_LINE_RE.search(INIT_FILE.read_text())
    if match is None:
        raise SystemExit(f"Could not find the __version__ line in {INIT_FILE}")
    return match["version"]


def write_version(version: str) -> bool:
    """Set ``__version__`` in ``skops/__init__.py``; return whether it changed."""
    if read_version() == version:
        return False
    text = INIT_FILE.read_text()
    INIT_FILE.write_text(VERSION_LINE_RE.sub(f'__version__ = "{version}"', text))
    return True


def changelog_sections(text: str) -> dict[str, str]:
    """Map each section heading of the changelog to the text below it."""
    matches = list(SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[match["name"]] = text[match.end() : end]
    return sections


def check_changelog(release: Release) -> None:
    """Fail unless the changelog has a non-empty section for the release."""
    sections = changelog_sections(CHANGELOG.read_text())
    path = CHANGELOG.relative_to(ROOT)
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


def start_dev(next_minor: str) -> dict[str, str]:
    """Move ``main`` on to the next development version.

    Sets ``__version__`` to ``MAJOR.MINOR.dev0`` and adds an empty changelog
    section for it, unless either is already in place.
    """
    if re.fullmatch(r"\d+\.\d+", next_minor) is None:
        raise SystemExit(
            f"Invalid development version {next_minor!r}: expected MAJOR.MINOR,"
            " e.g. 0.17."
        )
    version = f"{next_minor}.dev0"
    section = f"v{next_minor}"
    version_changed = write_version(version)

    text = CHANGELOG.read_text()
    changelog_changed = section not in changelog_sections(text)
    if changelog_changed:
        first = SECTION_RE.search(text)
        if first is None:
            raise SystemExit(f"Could not find any release section in {CHANGELOG}")
        heading = f"{section}\n{'-' * len(section)}\n\n"
        CHANGELOG.write_text(text[: first.start()] + heading + text[first.start() :])

    return {
        "version": version,
        "section": section,
        "changed": str(version_changed or changelog_changed).lower(),
    }


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
        check_changelog(Release.parse(args.version))
        outputs = {}
    elif args.command == "set-version":
        release = Release.parse(args.version)
        outputs = {"changed": str(write_version(release.version)).lower()}
    else:
        outputs = start_dev(args.next_minor)

    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
