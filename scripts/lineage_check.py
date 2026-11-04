"""Lineage gate CLI: uv run python scripts/lineage_check.py <base_repo@rev> <stage_repo@rev> [<stage_repo@rev> ...]
Resolves each repo@rev to a local snapshot (downloads weights if absent), compares base->each stage on weights only,
writes runs/lineage/<pair>.json, prints a table, exits non-zero if any pair FAILs."""
import sys

from huggingface_hub import HfApi, snapshot_download

from rtp import lineage


def resolve(spec: str):
    repo, _, rev = spec.partition("@")
    rev = rev or "main"
    api = HfApi()
    sha = api.model_info(repo, revision=rev).sha
    files = [f.rfilename for f in api.model_info(repo, revision=sha, files_metadata=True).siblings]
    weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
    path = snapshot_download(repo, revision=sha, allow_patterns=[weights, "*.json"])
    return sha, path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    bsha, bpath = resolve(sys.argv[1])
    bad = 0
    for spec in sys.argv[2:]:
        csha, cpath = resolve(spec)
        st = lineage.pair_stats(bpath, cpath)
        lineage.write_record(bsha, csha, st, label=f"{sys.argv[1]} -> {spec}")
        print(f"{'PASS' if st['passed'] else 'FAIL'}  {sys.argv[1]} ({bsha[:12]}) -> {spec} ({csha[:12]})  median_rel={st['median_rel']}  median_cos={st['median_cos']}  {st['reason']}", flush=True)
        bad += not st["passed"]
    sys.exit(1 if bad else 0)
