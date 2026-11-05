"""t0: fetch the two toy bases (pinned revisions, safetensors + configs + tokenizer only) into the shared HF cache on /mnt/e.
Usage: uv run python toy/scripts/t0_fetch.py   (HF_API_KEY from the environment; never printed)"""
import json, os, pathlib
from huggingface_hub import snapshot_download
CACHE = "/mnt/e/ai-models/huggingface_cache/hub"
REV = {"HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", "EleutherAI/pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"}
out = {}
for repo, rev in REV.items():
    p = snapshot_download(repo, revision=rev, cache_dir=CACHE, token=os.environ["HF_API_KEY"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "merges*", "vocab*"], ignore_patterns=["*onnx*"])
    out[repo] = dict(sha=rev, snapshot=p)
    print(repo, rev, p, flush=True)
pathlib.Path("toy/data/bases.json").write_text(json.dumps(out, indent=1))
