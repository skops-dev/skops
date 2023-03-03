# Deploy the app in skops_space_creator as a Hugging Face Space
# requires the HF_HUB_TOKEN to be set as environment variable

from pathlib import Path
from uuid import uuid4

from huggingface_hub import HfApi

import skops
import skops.hub_utils
import skops.hub_utils.tests
from skops.hub_utils.tests.common import HF_HUB_TOKEN

token = HF_HUB_TOKEN
print(token)

client = HfApi(token=token)
user_name = client.whoami(token=token)["name"]
repo_name = f"skops-space-creator-{uuid4()}"
repo_id = f"{user_name}/{repo_name}"
print(f"Creating and pushing to repo: {repo_id}")

space_repo = Path(skops.__path__[0]).parent / "spaces" / "skops_space_creator"

client.create_repo(
    repo_id=repo_id,
    token=token,
    repo_type="space",
    exist_ok=True,
    space_sdk="streamlit",
)
out = client.upload_folder(
    repo_id=repo_id,
    path_in_repo=".",
    folder_path=space_repo,
    commit_message="Create skops-space-creator space",
    token=token,
    repo_type="space",
    create_pr=False,
)

# link to main app, not to "<url>/tree/main/"
url = out.rsplit("/", 3)[0]
print(f"visit the space at {url}")
