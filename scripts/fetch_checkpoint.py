"""Download one checkpoint snapshot (weights + config + tokenizer) to the HF cache; print path and full commit SHA.
Usage: uv run python scripts/fetch_checkpoint.py <repo> [revision]"""
import sys
from huggingface_hub import HfApi, snapshot_download

repo, rev = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "main")
sha = HfApi().model_info(repo, revision=rev).sha
path = snapshot_download(repo, revision=sha, allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])
print(repo, rev, sha, path)
