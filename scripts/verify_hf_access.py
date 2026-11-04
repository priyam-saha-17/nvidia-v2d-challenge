"""Verify gated checkpoint access without downloading weights or printing credentials."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.secrets/huggingface')
from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError

if not get_token():
    raise SystemExit('No project Hugging Face login found. Run the provided hf auth login command.')
api = HfApi()
blocked = []
for repo in ('nvidia/cari4d_commercial', 'facebook/sam-3d-body-dinov3'):
    info = api.model_info(repo, token=True)
    weights = [f.rfilename for f in info.siblings
               if f.rfilename.endswith(('.pt', '.pth', '.ckpt', '.safetensors'))]
    if not weights:
        raise SystemExit(f'No checkpoint files discovered in {repo}')
    # An authenticated HEAD to the actual weight verifies gate access; model cards are public.
    try:
        meta = get_hf_file_metadata(hf_hub_url(repo, weights[0]), token=True)
        print(f'Access verified: {repo}, {weights[0]}, {meta.size} bytes')
    except GatedRepoError:
        blocked.append(repo)
        print(f'Access denied or awaiting author review: {repo}')
if blocked:
    raise SystemExit(1)
