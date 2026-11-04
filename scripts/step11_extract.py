"""Step 11 child: Olmo-3 7B extraction at layers {8,16} (1/4 L and 1/2 L), model truncated to 17 blocks, fp32 compute, geography + noun
controls (1000 strings x 8 templates). Closure on 10 prompts. Usage:
uv run python scripts/step11_extract.py <snapshot> <full_sha> <out_dir> --gate PARENT:CHILD   (gate is mandatory, fail closed)"""
import json, pathlib, sys, time
import numpy as np, torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from rtp import extract as X, lineage, cap
cap.require_cap()   # fail closed, no override

snap, sha, out = sys.argv[1:4]
if "--gate" not in sys.argv:
    sys.exit("BROKEN: --gate PARENT:CHILD is mandatory (lineage gate)")
gp, gc = sys.argv[sys.argv.index("--gate") + 1].split(":")
if sha not in (gp, gc):
    sys.exit("BROKEN: this checkpoint is not part of the gated pair")
lineage.require_pass(gp, gc)
LAYERS, NB = [8, 16], 17      # hooks at 8 and 16; one extra block so output_hidden_states[16] is not the final-normed state
t0 = time.time()
tok = Tokenizer.from_file(f"{snap}/tokenizer.json"); pad_id = tok.token_to_id("<|pad|>") or 0
sets = [s for s in X.read_sets(pathlib.Path("data/entities")) if s[0] in ("geo", "nouns")]
recs = X.build_prompts(tok, sets)
print("prompts", len(recs), "hooked-token assertion OK on every prompt", flush=True)
cfg = AutoConfig.from_pretrained(snap); cfg.layer_types = cfg.layer_types[:NB]; cfg.num_hidden_layers = NB
# Streaming load: model skeleton on the meta device, only the needed tensors read (safetensors, per key), kept in bf16 (as released);
# each block is upcast to fp32 only for its forward (pre-hook) and returned to bf16 afterwards, so compute is fp32-from-bf16-weights.
from safetensors import safe_open
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
want = {k: v for k, v in model.state_dict().items() if not k.startswith("lm_head")}
sd = {}
for shard in sorted({idx[k] for k in want}):
    with safe_open(f"{snap}/{shard}", "pt") as g:
        for k in want:
            if idx[k] == shard:
                sd[k] = g.get_tensor(k).to(torch.bfloat16)
missing = model.load_state_dict(sd, strict=False, assign=True)
assert not [k for k in missing.missing_keys if not k.startswith("lm_head")], missing.missing_keys
model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)          # non-persistent buffers were built on meta
model.model.embed_tokens.float(); model.model.norm.float()
for blk in model.model.layers:
    blk.register_forward_pre_hook(lambda m, a, kw: (m.float(), None)[1], with_kwargs=True); blk.register_forward_hook(lambda m, a, o: (m.bfloat16(), None)[1])
model.eval(); del sd
hwm = lambda: max(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 1e6
print("loaded", round(time.time() - t0), "s; blocks", len(model.model.layers), "VmHWM_GB", round(hwm(), 2), flush=True)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
rng = np.random.default_rng(0); sub = [recs[i] for i in rng.choice(len(recs), 10, replace=False)]
ref = X.reference_hidden(model, sub, LAYERS, pad_id)
tr, _ = X.forward_layers(model, sub, LAYERS, pad_id, batch=10)
single = [X.forward_layers(model, [r], LAYERS, pad_id, batch=1)[0] for r in sub]
res = {}
for l in LAYERS:
    s = np.stack([x[l][0] for x in single])
    res[l] = dict(closure_rel=rel(tr[l], ref[l]), closure_max_abs=float(np.abs(tr[l] - ref[l]).max()), reextract_rel=rel(s, tr[l]), reextract_max_abs=float(np.abs(s - tr[l]).max()))
print("CLOSURE", json.dumps(res), "VmHWM_GB", round(hwm(), 2), flush=True)
DRY = "--dry" in sys.argv
od = pathlib.Path(out); od.mkdir(parents=True, exist_ok=True); (od / "closure.json").write_text(json.dumps(res, indent=1))
if any(v["closure_rel"] > 1e-5 or v["reextract_rel"] > 1e-6 for v in res.values()):
    sys.exit("BROKEN: closure/re-extraction check failed")
if DRY:
    print("DRY_DONE VmHWM_GB", round(hwm(), 2), flush=True); sys.exit(0)
t1 = time.time()
last, mean = X.forward_layers(model, recs, LAYERS, pad_id, batch=256)
arrs = {f"{k}_L{l}": d[l] for l in LAYERS for k, d in (("last", last), ("mean", mean))}
meta = np.array([[{"geo": 0, "nouns": 1}[r["set"]], r["i"], r["t"], r["name_start"], len(r["ids"])] for r in recs], np.int32)
np.savez(od / "acts.npz", meta=meta, **arrs)
(od / "manifest.json").write_text(json.dumps(dict(sha=sha, snapshot=snap, weights="bf16 as released", compute="float32", layers=LAYERS, blocks_loaded=NB, n_prompts=len(recs), token_ids=[r["ids"] for r in recs], vmhwm_gb=hwm())))
print("VmHWM_GB final", round(hwm(), 2), flush=True)
print("EXTRACT_DONE", round(time.time() - t1), "s", flush=True)
