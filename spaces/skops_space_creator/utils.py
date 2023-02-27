"""Utility functions for the app"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

from skops import card
from skops.card._model_card import PlotSection, Section

PAT_MD_IMG = re.compile(
    r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))'
)


def get_rendered_model_card(model_card: card.Card, hf_path: str) -> str:
    # This is a bit hacky:
    # As a space, the model card is created in a temporary hf_path directory,
    # which is where all the files are put. So e.g. if a figure is added, it is
    # found at /tmp/skops-jtyqdgk3/fig.png. However, when the model card is is
    # actually used, we don't want that, since there, the files will be in the
    # cwd. Therefore, we remove the tmp directory everywhere we find it in the
    # file.
    if not hf_path.endswith(os.path.sep):
        hf_path += os.path.sep

    rendered = model_card.render()
    rendered = rendered.replace(hf_path, "")
    return rendered


def process_card_for_rendering(rendered: str) -> tuple[str, str]:
    idx = rendered[1:].index("\n---") + 1
    metadata = rendered[3:idx]
    rendered = rendered[idx + 4 :]  # noqa: E203

    # below is a hack to display the images in streamlit
    # https://discuss.streamlit.io/t/image-in-markdown/13274/10 The problem is

    # that streamlit does not display images in markdown, so we need to replace
    # them with html. However, we only want that in the rendered markdown, not
    # in the card that is produced for the hub
    def markdown_images(markdown):
        # example image markdown:
        # ![Test image](images/test.png "Alternate text")
        images = PAT_MD_IMG.findall(markdown)
        return images

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded

    def img_to_html(img_path, img_alt):
        img_format = img_path.split(".")[-1]
        img_html = (
            f'<img src="data:image/{img_format.lower()};'
            f'base64,{img_to_bytes(img_path)}" '
            f'alt="{img_alt}" '
            'style="max-width: 100%;">'
        )
        return img_html

    def markdown_insert_images(markdown):
        images = markdown_images(markdown)

        for image in images:
            image_markdown = image[0]
            image_alt = image[1]
            image_path = image[2]
            markdown = markdown.replace(
                image_markdown, img_to_html(image_path, image_alt)
            )
        return markdown

    rendered_with_img = markdown_insert_images(rendered)
    return metadata, rendered_with_img


@dataclass(frozen=True)
class SectionInfo:
    return_key: str
    title: str
    content: str
    is_fig: bool
    level: int


def iterate_key_section_content(
    data: dict[str, Section],
    parent_section: str = "",
    parent_keys: list[str] | None = None,
    level: int = 0,
) -> SectionInfo:
    parent_keys = parent_keys or []

    for key, val in data.items():
        if parent_section:
            title = "/".join((parent_section, val.title))
        else:
            title = val.title

        if not getattr(val, "visible", True):
            continue

        return_key = key if not parent_keys else "/".join(parent_keys + [key])
        content = val.content

        is_fig = getattr(val, "is_fig", False)
        if isinstance(val.content, str):
            img_match = PAT_MD_IMG.match(val.content)
            if img_match:  # image section found in parsed model card
                is_fig = True
                img_title = img_match.groupdict()["image_title"]
                img_path = img_match.groupdict()["image_path"]
                content = PlotSection(alt_text=img_title, path=img_path)

        yield SectionInfo(
            return_key=return_key,
            title=title,
            content=content,
            is_fig=is_fig,
            level=level,
        )

        if val.subsections:
            yield from iterate_key_section_content(
                val.subsections,
                parent_section=title,
                parent_keys=parent_keys + [key],
                level=level + 1,
            )
