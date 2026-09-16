import difflib
import json
import os
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from skops.card import parse_modelcard
from skops.card._parser import PandocParser, check_pandoc_installed

try:
    # the expected outputs in ./examples are generated with a recent pandoc
    check_pandoc_installed(min_version="3.6.4")
except (FileNotFoundError, ValueError):
    # not installed or too old, skip
    pytest.skip(reason="These tests require a recent pandoc", allow_module_level=True)


EXAMPLE_CARDS = [
    # actual model cards from HF hub
    "bert-base-uncased.md",
    "clip-vit-large-patch14.md",
    "gpt2.md",
    "specter.md",
    "vit-base-patch32-224-in21k.md",
    # not a model card
    "toy-example.md",
]


def assert_readme_files_almost_equal(file0, file1, diff):
    """Check that the two model cards are identical, but allow differences as
    defined in the ``diff`` file
    """
    with open(file0, encoding="utf-8") as f:
        readme0 = f.readlines()

    with open(file1, encoding="utf-8") as f:
        readme1 = f.readlines()

    # exclude trivial case of both being empty
    assert readme0
    assert readme1

    diff_actual = list(difflib.unified_diff(readme0, readme1, n=0))

    with open(diff, encoding="utf-8") as f:
        diff_expected = f.readlines()

    assert diff_actual == diff_expected


@pytest.mark.parametrize("file_name", EXAMPLE_CARDS, ids=EXAMPLE_CARDS)
def test_example_model_cards(tmp_path, file_name):
    """Test that the difference between original and parsed model card is
    acceptable

    For this test, model cards for some of the most popular models on HF Hub
    were retrieved and stored in the ./examples folder. This test checks that
    these model cards can be successfully parsed and that the output is *almost*
    the same.

    We don't expect the output to be 100% identical, see the limitations listed
    in ``parse_modelcard``. Instead, we assert that the diff corresponds to the
    expected diff, which is also checked in.

    So e.g. for "specter.md", we expect that the diff will be the same diff as
    in "specter.md.diff".

    If the output changes on purpose (e.g. because model cards are rendered
    differently or because of a new pandoc version), regenerate the
    ``.md.diff`` files the same way they are compared here, i.e. as a
    ``difflib.unified_diff`` with ``n=0`` between the original card and the
    saved parsed card.

    """
    path = Path(os.getcwd()) / "skops" / "card" / "tests" / "examples"
    file0 = path / file_name
    diff = (path / file_name).with_suffix(".md.diff")
    parsed_card = parse_modelcard(file0)
    file1 = tmp_path / "readme-parsed.md"
    parsed_card.save(file1)

    assert_readme_files_almost_equal(file0, file1, diff)


def test_unknown_pandoc_item_raises():
    source = json.dumps(
        {
            "pandoc-api-version": [1, 22, 2, 1],
            "meta": {},
            "blocks": [
                {
                    "t": "Header",
                    "c": [1, ["section", [], []], [{"t": "Str", "c": "section"}]],
                },
                {"c": "valid", "t": "Str"},
                {"t": "does-not-exist", "c": []},
                {"c": "okay", "t": "Str"},
            ],
        }
    )
    parser = PandocParser(source)
    msg = (
        "The parsed document contains 'does-not-exist', which is not "
        "supported yet, please open an issue on GitHub"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        parser.generate()


def test_content_without_section_raises():
    source = json.dumps(
        {
            "pandoc-api-version": [1, 22, 2, 1],
            "meta": {},
            "blocks": [
                {"c": "whoops", "t": "Str"},
            ],
        }
    )
    parser = PandocParser(source)
    msg = (
        "Trying to add content but there is no current section, this is probably a "
        "bug, please open an issue on GitHub"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        parser.generate()


def test_unsupported_markup_raises():
    match = re.escape("Markup of type does-not-exist is not supported (yet)")
    with pytest.raises(ValueError, match=match):
        PandocParser(source="", markup_type="does-not-exist")


def test_check_pandoc_installed_no_min_version_works():
    # check that it doesn't raise
    check_pandoc_installed(min_version=None)


def test_check_pandoc_installed_min_version_too_high_raises():
    match = re.escape("Pandoc version too low, expected at least 999.9.9, got")
    with pytest.raises(ValueError, match=match):
        check_pandoc_installed(min_version="999.9.9")


def test_pandoc_not_installed():
    def raise_filenotfound(*args, **kwargs):
        # error raised when trying to run subprocess on non-existing command
        raise FileNotFoundError("[Errno 2] No such file or directory: 'pandoc'")

    with patch("subprocess.run", raise_filenotfound):
        match = re.escape(
            "This feature requires the pandoc library to be installed on your system"
        )
        with pytest.raises(FileNotFoundError, match=match):
            check_pandoc_installed()


def test_pandoc_version_cannot_be_determined():
    mock = Mock()
    with patch("subprocess.run", mock):
        match = re.escape("Could not determine version of pandoc")
        with pytest.raises(RuntimeError, match=match):
            check_pandoc_installed()
