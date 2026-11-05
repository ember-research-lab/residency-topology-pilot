"""t3: extract l* last-token activations for one trained run (4 entity sets, survivors of the base gate), closure on 10 prompts, then delete the
weights. Usage (under the cap): uv run python toy/scripts/t3_extract.py <stage> <seed> -> toy/runs/<stage>_s<seed>/acts.npz"""
import json, pathlib, shutil, sys, time
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0, "toy")
import toylib as T
from rtp import cap
cap.require_cap()
stage, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
snap = json.load(open("toy/data/bases.json"))["HuggingFaceTB/SmolLM2-360M"]["snapshot"]
tok = Tokenizer.from_file(f"{snap}/tokenizer.json"); pad_id = tok.token_to_id("<|endoftext|>")
b = np.load("toy/runs/base_smollm2/acts.npz"); surv = {k: b[f"idx_{k}"].tolist() for k in T.SETCODE}
recs = T.make_records(tok, surv)
wd = pathlib.Path(f"/mnt/e/toy-residency/{stage}_s{seed}"); model = T.load(str(wd)); ls = T.n_layers(model) // 2
assert ls == 16
cl = T.closure_check(model, recs, ls, pad_id); assert cl["closure_rel"] <= 1e-5 and cl["reextract_rel"] <= 1e-6, f"F-T4 closure {cl}"
sets = T.to_sets(T.forward_last(model, recs, ls, pad_id), recs)
for k in sets:   # F-T4: weights identical to base would give identical activations
    assert not np.array_equal(sets[k], b[f"H_{k}"]), "F-T4: identical to base"
od = pathlib.Path(f"toy/runs/{stage}_s{seed}"); od.mkdir(parents=True, exist_ok=True)
np.savez(od / "acts.npz", **{f"H_{k}": v for k, v in sets.items()})
json.dump(dict(closure=cl, wall_s=round(time.time() - t0)), open(od / "extract.json", "w"))
if stage != "sftre": shutil.rmtree(wd)   # sftre = re-trained SFT parents (AMENDMENT 2), kept as the GRPO start points
print("EXTRACTED", stage, seed, cl, round(time.time() - t0), "s; weights " + ("kept" if stage == "sftre" else "deleted"), flush=True)
