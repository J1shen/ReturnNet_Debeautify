from huggingface_hub import hf_hub_download
"""
hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", 
                filename="v2-1_512-ema-pruned.ckpt",
                cache_dir='./checkpoints')
"""
hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", 
                filename="v1-5-pruned.ckpt",
                cache_dir='./checkpoints')