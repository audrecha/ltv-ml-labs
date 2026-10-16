"""Lab 2: download model weights from Hugging Face into the local cache."""
from huggingface_hub import snapshot_download

for repo in ['google/siglip-base-patch16-224', 'intfloat/multilingual-e5-small',
             'mlx-community/Qwen2.5-VL-7B-Instruct-4bit', 'mlx-community/gemma-3-4b-it-4bit']:
    print(repo, snapshot_download(repo), flush=True)
