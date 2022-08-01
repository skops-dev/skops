"""This script removes all repos under the skops user on HF Hub.

The user is used for the CI and if there are leftover repos, they can be
removed.
"""

import datetime

from huggingface_hub import HfApi

# This is the token for the skops user. TODO remove eventually, see issue #47
token = "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"
client = HfApi()
user = client.whoami(token=token)["name"]
answer = input(
    f"Are you sure you want to delete all repos under {user} older than 7 days? (y/[n])"
)
if answer != "y":
    exit(1)
models = client.list_models(author=user)

print(f"Found {len(models)} models, checking their age...")

for model_info in models:
    info = client.model_info(model_info.modelId, token=token)
    age = (
        datetime.datetime.now()
        - datetime.datetime.fromisoformat(info.lastModified.rsplit(".", 1)[0])
    ).days
    if age < 7:
        print(f"Skipping model: {model_info.modelId}, age: {age}")
        continue
    print(f"deleting {model_info.modelId}, age: {age} days")
    client.delete_repo(model_info.modelId, token=token)
