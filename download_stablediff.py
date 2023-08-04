from huggingface_hub import snapshot_download
snapshot_download(repo_id="stabilityai/stable-diffusion-2-1-base",
                  cache_dir='./checkpoints')