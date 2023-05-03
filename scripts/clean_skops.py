"""This script removes all old repos under the skops user on HF Hub.

The user is used for the CI and if there are leftover repos, they can be
removed.
"""

import datetime

from huggingface_hub import HfApi
from requests.exceptions import HTTPError

MAX_AGE = 7  # in days

# This is the token for the skops user. TODO remove eventually, see issue #47
token = "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"
client = HfApi(token=token)
user = client.whoami()["name"]
answer = input(
    f"Are you sure you want to delete all repos under {user} older than {MAX_AGE} days?"
    " (y/[n]) "
)
if answer != "y":
    exit(1)

# MODELS

models = [x for x in client.list_models(author=user)]
print(f"Found {len(models)} models, checking their age...")

for model_info in models:
    try:
        info = client.model_info(model_info.modelId)
    except HTTPError:
        # https://github.com/huggingface/moon-landing/issues/6034
        continue

    age = (
        datetime.datetime.now()
        - datetime.datetime.fromisoformat(info.lastModified.rsplit(".", 1)[0])
    ).days
    if age < MAX_AGE:
        print(f"Skipping model: {model_info.modelId}, age: {age}")
        continue
    print(f"deleting {model_info.modelId}, age: {age} days")
    client.delete_repo(model_info.modelId)

# SPACES

spaces = [x for x in client.list_spaces(author=user)]
print(f"Found {len(spaces)} spaces, checking their age...")

for space_info in spaces:
    try:
        info = client.space_info(space_info.id)
    except HTTPError:
        # https://github.com/huggingface/moon-landing/issues/6034
        continue

    age = (
        datetime.datetime.now()
        - datetime.datetime.fromisoformat(info.lastModified.rsplit(".", 1)[0])
    ).days
    if age < MAX_AGE:
        print(f"Skipping space: {space_info.id}, age: {age}")
        continue
    print(f"deleting {space_info.id}, age: {age} days")
    client.delete_repo(space_info.id, repo_type="space")
