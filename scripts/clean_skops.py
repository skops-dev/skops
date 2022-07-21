from huggingface_hub import HfApi

token = "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"
client = HfApi()
user = client.whoami(token=token)["name"]
models = client.list_models(author=user)
for model_info in models:
    print(f"deleting {model_info.modelId}")
    client.delete_repo(model_info.modelId, token=token)
