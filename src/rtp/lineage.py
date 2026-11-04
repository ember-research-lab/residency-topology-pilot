"""lineage - weights-only lineage gate: is a stage a descendant of its claimed base? (fail closed, no override).

For a (parent, child) snapshot pair: median over all tensors of the relative Frobenius distance ||child-parent||/||parent||
and of the cosine of the flattened tensors. PASS iff median distance < MAX_DIST and median cosine > MIN_COS and the key sets
and shapes are identical. Records live in runs/lineage/<parent12>__<child12>.json; downstream code calls require_pass().
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch

MAX_DIST, MIN_COS = 0.5, 0.9
RECORD_DIR = pathlib.Path("runs/lineage")


class LineageError(RuntimeError):
    """Raised when a lineage record is missing or FAIL; there is no override."""


def _keys_and_getter(snap: str):
    """Return (sorted keys, get(key)->float32 tensor) reading straight from the weight files, one tensor at a time."""
    from safetensors import safe_open
    files = sorted(glob.glob(f"{snap}/*.safetensors"))
    if files:
        where = {}
        for f in files:
            with safe_open(f, "pt") as g:
                for k in g.keys():
                    where[k] = f

        def get(k):
            with safe_open(where[k], "pt") as g:
                return g.get_tensor(k).float()
        return sorted(where), get
    bins = glob.glob(f"{snap}/*.bin")
    if not bins:
        raise LineageError(f"no weight files under {snap}")
    sd = torch.load(bins[0], weights_only=True, map_location="cpu")
    return sorted(sd), lambda k: sd[k].float()


def pair_stats(parent_snap: str, child_snap: str) -> dict:
    pk, pget = _keys_and_getter(parent_snap)
    ck, cget = _keys_and_getter(child_snap)
    out = dict(parent=parent_snap, child=child_snap, n_parent=len(pk), n_child=len(ck),
               keys_equal=pk == ck)
    if pk != ck:
        out.update(median_rel=None, median_cos=None, shapes_equal=False, passed=False,
                   reason="key sets differ")
        return out
    rel, cos, shapes = [], [], True
    for k in pk:
        a, b = pget(k), cget(k)
        if a.shape != b.shape:
            shapes = False
            break
        rel.append(float((b - a).norm() / a.norm()))
        cos.append(float(torch.dot(a.flatten().double(), b.flatten().double()) / (a.double().norm() * b.double().norm())))
    if not shapes:
        out.update(median_rel=None, median_cos=None, shapes_equal=False, passed=False, reason="shape mismatch")
        return out
    out.update(shapes_equal=True, median_rel=float(np.median(rel)), median_cos=float(np.median(cos)),
               min_cos=float(np.min(cos)), max_rel=float(np.max(rel)), n_tensors=len(rel))
    out["passed"] = bool(out["median_rel"] < MAX_DIST and out["median_cos"] > MIN_COS)
    out["reason"] = "PASS" if out["passed"] else "median distance >= %.2f or median cosine <= %.2f" % (MAX_DIST, MIN_COS)
    return out


def pair_id(parent_sha: str, child_sha: str) -> str:
    return f"{parent_sha[:12]}__{child_sha[:12]}"


def write_record(parent_sha: str, child_sha: str, stats: dict, label: str = "") -> pathlib.Path:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    p = RECORD_DIR / f"{pair_id(parent_sha, child_sha)}.json"
    p.write_text(json.dumps(dict(stats, parent_sha=parent_sha, child_sha=child_sha, label=label, max_dist=MAX_DIST, min_cos=MIN_COS), indent=1))
    return p


def require_pass(parent_sha: str, child_sha: str) -> dict:
    """Fail closed: raise unless a PASS record exists for exactly this pair."""
    p = RECORD_DIR / f"{pair_id(parent_sha, child_sha)}.json"
    if not p.exists():
        raise LineageError(f"lineage gate record missing for {pair_id(parent_sha, child_sha)}: run scripts/lineage_check.py first")
    rec = json.loads(p.read_text())
    if not rec.get("passed"):
        raise LineageError(f"lineage gate FAIL for {pair_id(parent_sha, child_sha)} (median_rel={rec.get('median_rel')}, median_cos={rec.get('median_cos')})")
    return rec
