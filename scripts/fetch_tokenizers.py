"""Download tokenizer files only (tokenizer*.json, *.model) for the pilot repos; print revision SHA and file hashes.
Usage: uv run python scripts/fetch_tokenizers.py <out_dir>"""
import hashlib, pathlib, sys
from huggingface_hub import HfApi, snapshot_download

REPOS = ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT", "allenai/OLMo-2-0425-1B-DPO",
         "allenai/OLMo-2-0425-1B-RLVR1", "allenai/OLMo-2-0425-1B-Instruct",
         "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-RL-Zero-Math"]
out = pathlib.Path(sys.argv[1])
for r in REPOS:
    d = out / r.split("/")[1]
    snapshot_download(r, allow_patterns=["tokenizer*.json", "*.model"], local_dir=d)
    sha = HfApi().model_info(r).sha
    for f in sorted(p for p in d.rglob("*") if p.is_file() and ".cache" not in p.parts):
        print(r, sha, f.relative_to(d), f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest())
