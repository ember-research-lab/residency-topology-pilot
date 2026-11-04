"""extract - residual-stream vectors at the last name token (DESIGN 2.2, 6.3 step 3).

HF load (optionally bf16-rounded weights, fp32 compute), forward hooks on blocks (layer l = output of block l,
layer 0 = embeddings), truncation after the deepest probed layer. Layer-outer sizes are small at 1B so the whole
truncated model stays in RAM. Writes runs/<ckpt>/acts.npz + manifest.json.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from .entities import TEMPLATES_CTRL, TEMPLATES_GEO

torch.set_num_threads(4)


class ExtractError(RuntimeError):
    """Raised when an extraction assertion (hooked token, closure, determinism) fails."""


def load_model(path: str, round_bf16: bool = True):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()
    if round_bf16:
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
    return model


def read_sets(ent_dir: pathlib.Path) -> list[tuple[str, str, list[str]]]:
    """[(set_name, column, templates)] -> strings, in frozen file order."""
    spec = [("geo", "geo_v1.csv", "name", TEMPLATES_GEO), ("nouns", "control_nouns_v1.csv", "text", TEMPLATES_CTRL),
            ("nonce", "control_nonce_v1.csv", "text", TEMPLATES_CTRL)]
    return [(s, [r[col] for r in csv.DictReader(open(ent_dir / f, encoding="utf-8"))], t) for s, f, col, t in spec]


def build_prompts(tok: Tokenizer, sets) -> list[dict]:
    """One record per (set, string, template): token ids and the hooked position (last token, asserted to be the
    last subword of the name: the template prefix tokenizes to a strict prefix of the full prompt, the remainder
    is the name, and nothing (EOS/pad) follows)."""
    out = []
    for sname, strings, templates in sets:
        for si, x in enumerate(strings):
            for ti, t in enumerate(templates):
                full = tok.encode(t.replace("{X}", x), add_special_tokens=True).ids
                pre = tok.encode(t.split("{X}")[0].rstrip(), add_special_tokens=True).ids if t.split("{X}")[0].strip() else tok.encode("", add_special_tokens=True).ids
                if full[:len(pre)] != pre or len(full) <= len(pre):
                    raise ExtractError(f"hooked-token assertion failed (prefix) for {t!r} {x!r}")
                name_ids = full[len(pre):]
                if tok.decode(name_ids).strip() != x:
                    raise ExtractError(f"hooked-token assertion failed (decode) for {t!r} {x!r}: {tok.decode(name_ids)!r}")
                out.append(dict(set=sname, i=si, t=ti, ids=full, name_start=len(pre), x=x))
    return out


def _hidden(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def forward_layers(model, recs: list[dict], layers: list[int], pad_id: int, batch: int = 64):
    """Hooked, truncated forward. Returns {layer: (N,d)} last-token and {layer: (N,d)} mean-over-name-tokens."""
    blocks = model.model.layers
    deepest = max(layers)
    keep = list(blocks)
    model.model.layers = torch.nn.ModuleList(keep[:deepest])
    cap: dict[int, torch.Tensor] = {}
    hs = []
    try:
        for l in layers:
            if l == 0:
                hs.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o)))
            else:
                hs.append(model.model.layers[l - 1].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, _hidden(o))))
        last = {l: np.zeros((len(recs), model.config.hidden_size), np.float32) for l in layers}
        mean = {l: np.zeros_like(last[l]) for l in layers}
        order = sorted(range(len(recs)), key=lambda i: len(recs[i]["ids"]))
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            T = max(len(recs[i]["ids"]) for i in idx)
            ids = torch.full((len(idx), T), pad_id); mask = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                n = len(recs[i]["ids"]); ids[j, :n] = torch.tensor(recs[i]["ids"]); mask[j, :n] = 1
            model.model(input_ids=ids, attention_mask=mask)
            for l in layers:
                h = cap[l]
                for j, i in enumerate(idx):
                    n = len(recs[i]["ids"]); s = recs[i]["name_start"]
                    last[l][i] = h[j, n - 1].numpy(); mean[l][i] = h[j, s:n].mean(0).numpy()
        return last, mean
    finally:
        for h in hs:
            h.remove()
        model.model.layers = torch.nn.ModuleList(keep)


@torch.no_grad()
def reference_hidden(model, recs: list[dict], layers: list[int], pad_id: int):
    """Unmodified HF forward with output_hidden_states=True (all layers), last-token vectors, one prompt at a time."""
    out = {l: np.zeros((len(recs), model.config.hidden_size), np.float32) for l in layers}
    for i, r in enumerate(recs):
        o = model.model(input_ids=torch.tensor([r["ids"]]), output_hidden_states=True)
        for l in layers:
            out[l][i] = o.hidden_states[l][0, -1].numpy()
    return out


def run(model_path: str, tok_path: str, ent_dir: str, out_dir: str, layers: list[int], round_bf16: bool, sha: str) -> dict:
    tok = Tokenizer.from_file(tok_path)
    pad_id = tok.token_to_id("<|pad|>") or 0
    sets = read_sets(pathlib.Path(ent_dir))
    recs = build_prompts(tok, sets)
    model = load_model(model_path, round_bf16)
    last, mean = forward_layers(model, recs, layers, pad_id)
    od = pathlib.Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for l in layers:
        for kind, d in (("last", last), ("mean", mean)):
            arrs[f"{kind}_L{l}"] = d[l]
    meta = np.array([[ {"geo": 0, "nouns": 1, "nonce": 2}[r["set"]], r["i"], r["t"], r["name_start"], len(r["ids"])] for r in recs], np.int32)
    np.savez(od / "acts.npz", meta=meta, **arrs)
    man = dict(sha=sha, model_path=model_path, weights="bf16-rounded" if round_bf16 else "float32", compute="float32",
               layers=layers, n_prompts=len(recs), meta_columns=["set(0 geo,1 nouns,2 nonce)", "string_index", "template_index", "name_start", "length"],
               token_ids=[r["ids"] for r in recs])
    (od / "manifest.json").write_text(json.dumps(man))
    return dict(n=len(recs), model=model)
