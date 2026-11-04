"""Steps 3/5: closure check, then full extraction. Usage:
uv run python scripts/step3_extract.py <snapshot_dir> <full_sha> <out_dir> <bf16|f32> [--skip-closure]
Closure (DESIGN 4/6.3): truncated+hooked forward vs unmodified output_hidden_states <= 1e-5 relative on 20 prompts;
re-extraction (batch 20 vs singly) <= 1e-6 relative; hooked token = last subword of the name on every prompt."""
import json, pathlib, sys, time
import numpy as np
from tokenizers import Tokenizer
from rtp import extract as X

snap, sha, out, prec = sys.argv[1:5]
skip = "--skip-closure" in sys.argv
base_snap = sys.argv[sys.argv.index("--base-snap") + 1] if "--base-snap" in sys.argv else None
LAYERS = [0, 4, 8, 12]
t0 = time.time()
tok = Tokenizer.from_file(f"{snap}/tokenizer.json")
pad_id = tok.token_to_id("<|pad|>") or 0
sets = X.read_sets(pathlib.Path("data/entities"))
recs = X.build_prompts(tok, sets)                       # raises on any hooked-token failure (all prompts)
print("prompts", len(recs), "hooked-token assertion OK on every prompt", flush=True)
model = X.load_model(snap, round_bf16=(prec == "bf16"))
print("loaded", round(time.time() - t0), "s", flush=True)
if base_snap:   # BROKEN-if: stage weights identical to base (wrong repo/branch)
    import torch, glob
    from safetensors import safe_open
    sd = model.state_dict(); dw = {}
    for f in sorted(glob.glob(f"{base_snap}/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if any(f"layers.{i}." in k for i in (0, 5, 11)) and ("q_proj" in k or "up_proj" in k):
                    b = g.get_tensor(k).to(torch.bfloat16).float()
                    dw[k] = float((sd[k] - b).norm() / b.norm())
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out, "deltaW.json").write_text(json.dumps(dw, indent=1))
    print("DELTAW max rel", max(dw.values()), "n", len(dw), flush=True)
    if not dw or max(dw.values()) == 0.0:
        sys.exit("BROKEN: stage weights identical to base")
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
if not skip:
    rng = np.random.default_rng(0)
    sub = [recs[i] for i in rng.choice(len(recs), 20, replace=False)]
    ref = X.reference_hidden(model, sub, LAYERS, pad_id)
    tr, _ = X.forward_layers(model, sub, LAYERS, pad_id, batch=20)
    single = [X.forward_layers(model, [r], LAYERS, pad_id, batch=1)[0] for r in sub]
    res = {}
    for l in LAYERS:
        s = np.stack([x[l][0] for x in single])
        res[l] = dict(closure_rel=rel(tr[l], ref[l]), closure_max_abs=float(np.abs(tr[l] - ref[l]).max()),
                      reextract_rel=rel(s, tr[l]), reextract_max_abs=float(np.abs(s - tr[l]).max()))
    print("CLOSURE", json.dumps(res), flush=True)
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out, "closure.json").write_text(json.dumps(res, indent=1))
    if any(v["closure_rel"] > 1e-5 or v["reextract_rel"] > 1e-6 for v in res.values()):
        sys.exit("BROKEN: closure/re-extraction check failed; stopping before step 4")
t1 = time.time()
last, mean = X.forward_layers(model, recs, LAYERS, pad_id)
od = pathlib.Path(out); od.mkdir(parents=True, exist_ok=True)
arrs = {f"{k}_L{l}": d[l] for l in LAYERS for k, d in (("last", last), ("mean", mean))}
meta = np.array([[{"geo": 0, "nouns": 1, "nonce": 2}[r["set"]], r["i"], r["t"], r["name_start"], len(r["ids"])] for r in recs], np.int32)
np.savez(od / "acts.npz", meta=meta, **arrs)
(od / "manifest.json").write_text(json.dumps(dict(sha=sha, snapshot=snap, weights="bf16-rounded" if prec == "bf16" else "float32",
    compute="float32", layers=LAYERS, n_prompts=len(recs), meta_columns=["set(0 geo,1 nouns,2 nonce)", "string_index", "template_index", "name_start", "length"],
    token_ids=[r["ids"] for r in recs])))
print("EXTRACT_DONE", round(time.time() - t1), "s", flush=True)
