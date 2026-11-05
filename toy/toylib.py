"""toylib - shared helpers for the toy seed-replicate run (PREREG-toy-2026-11-04): survivors under a toy tokenizer, generic truncated
extraction (Llama-style and GPT-NeoX), ridge gate, base freezes. Reuses rtp.extract.build_prompts (hooked-token assertions)."""
from __future__ import annotations

import csv, json, pathlib, sys
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from rtp import graph as G, measures as M
from rtp.entities import TEMPLATES_CTRL, TEMPLATES_GEO
from rtp.extract import ExtractError, build_prompts

torch.set_num_threads(4)
ENT = pathlib.Path(__file__).resolve().parents[1] / "data" / "entities"
K = 8
# (set key, csv, column, templates)
SETS = [("geo", "geo_v1.csv", "name", TEMPLATES_GEO), ("nouns", "control_nouns_v1.csv", "text", TEMPLATES_CTRL),
        ("nonce", "control_nonce_v1.csv", "text", TEMPLATES_CTRL), ("fig", "figures_v1.csv", "name", TEMPLATES_GEO)]
SETCODE = {"geo": 0, "nouns": 1, "nonce": 2, "fig": 3}


def read_rows(key: str) -> list[dict]:
    f = next(s for s in SETS if s[0] == key)
    return list(csv.DictReader(open(ENT / f[1], encoding="utf-8")))


def survivors(tok: Tokenizer) -> dict[str, list[int]]:
    """Indices (into the frozen CSV rows) whose 8 templates tokenise as prefix+name with name <= 4 tokens under `tok`."""
    out = {}
    for key, _, col, templates in SETS:
        keep = []
        for i, r in enumerate(read_rows(key)):
            x = r[col]
            if len(tok.encode(" " + x, add_special_tokens=False).ids) > 4 or len(tok.encode(x, add_special_tokens=False).ids) > 4:
                continue
            try:
                build_prompts(tok, [(key, [x], templates)])
            except ExtractError:
                continue
            keep.append(i)
        out[key] = keep
    return out


def make_records(tok: Tokenizer, surv: dict[str, list[int]]) -> list[dict]:
    recs = []
    for key, _, col, templates in SETS:
        rows = read_rows(key)
        strings = [rows[i][col] for i in surv[key]]
        for r in build_prompts(tok, [(key, strings, templates)]):
            r["row"] = surv[key][r["i"]]
            recs.append(r)
    return recs


def backbone(model):
    return model.model if hasattr(model, "model") else model.gpt_neox


def n_layers(model) -> int:
    return model.config.num_hidden_layers


def load(path: str, dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    m.eval()
    return m


@torch.no_grad()
def forward_last(model, recs: list[dict], layer: int, pad_id: int, batch: int = 64) -> np.ndarray:
    """Hooked, truncated forward: residual (output of block `layer`) at the last token of every record. (N,d) float32."""
    bb = backbone(model)
    keep = list(bb.layers)
    bb.layers = torch.nn.ModuleList(keep[:layer])
    cap = {}
    h = bb.layers[layer - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        out = np.zeros((len(recs), model.config.hidden_size), np.float32)
        order = sorted(range(len(recs)), key=lambda i: len(recs[i]["ids"]))
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            T = max(len(recs[i]["ids"]) for i in idx)
            ids = torch.full((len(idx), T), pad_id); mask = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                n = len(recs[i]["ids"]); ids[j, :n] = torch.tensor(recs[i]["ids"]); mask[j, :n] = 1
            bb(input_ids=ids, attention_mask=mask)
            for j, i in enumerate(idx):
                out[i] = cap["h"][j, len(recs[i]["ids"]) - 1].numpy()
        return out
    finally:
        h.remove()
        bb.layers = torch.nn.ModuleList(keep)


@torch.no_grad()
def reference_last(model, recs: list[dict], layer: int) -> np.ndarray:
    out = np.zeros((len(recs), model.config.hidden_size), np.float32)
    for i, r in enumerate(recs):
        o = model(input_ids=torch.tensor([r["ids"]]), output_hidden_states=True)
        out[i] = o.hidden_states[layer][0, -1].numpy()
    return out


def closure_check(model, recs, layer, pad_id, n=10, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    sub = [recs[i] for i in rng.choice(len(recs), n, replace=False)]
    ref = reference_last(model, sub, layer)
    tr = forward_last(model, sub, layer, pad_id, batch=n)
    single = np.concatenate([forward_last(model, [r], layer, pad_id, batch=1) for r in sub])
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return dict(closure_rel=rel(tr, ref), reextract_rel=rel(single, tr), n=n)


def to_sets(last: np.ndarray, recs: list[dict]) -> dict[str, np.ndarray]:
    """(N,d) in record order -> {set: (n_surv, K, d)} in survivor order."""
    out = {}
    for key in SETCODE:
        sel = [i for i, r in enumerate(recs) if r["set"] == key]
        n = len(sel) // K
        a = last[sel].reshape(n, K, -1)
        assert all(recs[sel[j]]["i"] == j // K and recs[sel[j]]["t"] == j % K for j in range(len(sel)))
        out[key] = a
    return out


def ridge_r2(X, Y, folds=5, seed=0, grid=np.logspace(-4, 4, 17)):
    """Pilot step-4 ridge: 5-fold CV R2 pooled over outputs, alpha by inner LOO (verbatim from scripts/step4_freeze.py)."""
    X = X - X.mean(0)
    idx = np.random.default_rng(seed).permutation(len(X)); parts = np.array_split(idx, folds)
    pred = np.zeros_like(Y)
    for te in parts:
        tr = np.setdiff1d(idx, te)
        Xt, mu = X[tr], Y[tr].mean(0)
        Kt = Xt @ Xt.T; w, V = np.linalg.eigh(Kt); sc = w.mean()
        best = None
        for a in grid:
            lam = a * sc
            Hd = (V * (w / (w + lam))) @ V.T
            res = (Y[tr] - mu) - Hd @ (Y[tr] - mu)
            loo = ((res / (1 - np.diag(Hd))[:, None]) ** 2).sum()
            if best is None or loo < best[0]: best = (loo, lam)
        dual = np.linalg.solve(Kt + best[1] * np.eye(len(tr)), Y[tr] - mu)
        pred[te] = X[te] @ (Xt.T @ dual) + mu
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def unit_xyz(rows):
    lat, lon = np.radians([float(g["lat"]) for g in rows]), np.radians([float(g["lon"]) for g in rows])
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def frozen_m(arr: np.ndarray, seed=20261104) -> int:
    """Pilot gap-rule m over 100 (template multiset x 80% entity subsample) bootstrap spectra; arr (n,K,d)."""
    rng = np.random.default_rng(seed); n = len(arr); sp = []
    for _ in range(100):
        t = rng.integers(0, K, K); sub = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        sp.append(np.linalg.eigvalsh(G.laplacian(arr[sub][:, t].mean(1), 7)))
    return G.gap_rule_m(sp)
