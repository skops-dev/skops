"""This script removes all repos under skops user.

The user is used for the CI and if there are leftover repos, they can be
removed.
"""

from huggingface_hub import HfApi

# This is the token for the skops user. TODO remove eventually, see issue #47
token = "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"
client = HfApi()
user = client.whoami(token=token)["name"]
answer = input(f"Are you sure you want to delete all repos under {user}? (y/[n])")
if answer != "y":
    exit(1)
models = client.list_models(author=user)
for model_info in models:
    print(f"deleting {model_info.modelId}")
    client.delete_repo(model_info.modelId, token=token)
