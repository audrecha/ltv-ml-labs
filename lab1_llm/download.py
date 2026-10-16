from huggingface_hub import snapshot_download
for r in ["mlx-community/Qwen2.5-7B-Instruct-4bit","mlx-community/Meta-Llama-3.1-8B-Instruct-4bit","mlx-community/Mistral-7B-Instruct-v0.3-4bit"]:
    p=snapshot_download(r); print(r,p,flush=True)
