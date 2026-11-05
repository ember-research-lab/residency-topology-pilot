"""t1: gate (F-T1) for one toy base: survivors, closure on 10 prompts, base extraction at l*=L/2, geography ridge R2, frozen m/r per set.
Usage (under the cap): uv run python toy/scripts/t1_gate.py <smollm2|pythia> -> toy/data/gate_<key>.json, toy/runs/base_<key>/acts.npz"""
import json, pathlib, resource, sys, time
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0, "toy")
import toylib as T
from rtp import cap, measures as M
cap.require_cap()
key = sys.argv[1]
repo = {"smollm2": "HuggingFaceTB/SmolLM2-360M", "pythia": "EleutherAI/pythia-410m"}[key]
b = json.load(open("toy/data/bases.json"))[repo]
snap = b["snapshot"]; t0 = time.time()
tok = Tokenizer.from_file(f"{snap}/tokenizer.json")
pad_id = tok.token_to_id("<|endoftext|>") or tok.token_to_id("<|padding|>") or 0
surv = T.survivors(tok)
rep = dict(base=repo, sha=b["sha"], survivors={k: len(v) for k, v in surv.items()}, frozen_rows={k: len(T.read_rows(k)) for k in surv})
print("survivors", rep["survivors"], flush=True)
pair_ok = None
nouns = T.read_rows("nouns"); geo = T.read_rows("geo"); gpos = {g["name"]: i for i, g in enumerate(geo)}
gs, ns = set(surv["geo"]), set(surv["nouns"])
rep["paired_geo_noun_survivors"] = sum(1 for i in surv["nouns"] if gpos[nouns[i]["matched_geo"]] in gs)
if len(surv["geo"]) < 300 or len(surv["nouns"]) < 300:
    rep["F_T1"] = "FIRES: survivors below floor 300"; json.dump(rep, open(f"toy/data/gate_{key}.json", "w"), indent=1); sys.exit(rep["F_T1"])
recs = T.make_records(tok, surv)
print("prompts", len(recs), "hooked-token assertion OK", flush=True)
model = T.load(snap); L = T.n_layers(model); ls = L // 2
rep.update(L=L, l_star=ls, weights="float32 as released", compute="float32")
rep["closure"] = T.closure_check(model, recs, ls, pad_id)
print("closure", rep["closure"], flush=True)
assert rep["closure"]["closure_rel"] <= 1e-5 and rep["closure"]["reextract_rel"] <= 1e-6, "F-T4: closure failed"
last = T.forward_last(model, recs, ls, pad_id)
sets = T.to_sets(last, recs)
od = pathlib.Path(f"toy/runs/base_{key}"); od.mkdir(parents=True, exist_ok=True)
np.savez(od / "acts.npz", **{f"H_{k}": v for k, v in sets.items()}, **{f"idx_{k}": np.array(surv[k]) for k in surv})
Hgeo = sets["geo"].mean(1)
rep["geo_ridge_r2"] = T.ridge_r2(Hgeo, T.unit_xyz([geo[i] for i in surv["geo"]]))
rep["F_T1"] = "FIRES (R2 < 0.5): STOP" if rep["geo_ridge_r2"] < 0.5 else "pass (R2 >= 0.5)"
fr = {}
for k in ("geo", "nouns", "nonce", "fig"):
    Hc = sets[k].mean(1); Hc = Hc - Hc.mean(0)
    fr[k] = dict(n=len(Hc), m=T.frozen_m(sets[k]), r=M.r_from_base(Hc))
rep["frozen"] = fr
rep["wall_s"] = round(time.time() - t0); rep["vmhwm_kb"] = int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])
json.dump(rep, open(f"toy/data/gate_{key}.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
