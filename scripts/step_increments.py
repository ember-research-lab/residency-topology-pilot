"""AMENDMENT 3 (EXPLORATORY, non-voting): parent-relative stage increments on the frozen base graph.
D_inc = H_child - H_parent applied as stage_vs_base(base, H_b + D_inc). Holm over the 4 stage increments (family 8);
step-branch increments reported raw. Bootstrap B=1000 for the 4 stage increments. -> data/increments_results.json
Usage: uv run python scripts/step_increments.py"""
import json, time, tomllib
import numpy as np
from rtp import analysis as A, lineage

t0 = time.time()
cfg = tomllib.load(open("configs/pilot.toml", "rb")); m, r, B = cfg["frozen"]["m"], cfg["frozen"]["r"], cfg["measure"]["B"]
base_sha = cfg["hosts"]["olmo2_1b"]["base"]["sha"]
for k, v in json.load(open("data/checkpoints.json")).items():          # lineage gate, fail closed
    if isinstance(v, dict) and "sha" in v and v["sha"] != base_sha:
        lineage.require_pass(base_sha, v["sha"])
INC = {"sft": ("olmo2-1b-base", "sft"), "dpo|sft": ("sft", "dpo"), "rlvr1|dpo": ("dpo", "rlvr1"), "instruct|rlvr1": ("rlvr1", "instruct"),
       "step200|rlvr1": ("rlvr1", "inst_step200"), "step800|step200": ("inst_step200", "inst_step800"),
       "step1600|step800": ("inst_step800", "inst_step1600"), "step2600|step1600": ("inst_step1600", "inst_step2600")}
MAIN = list(INC)[:4]
names = ["olmo2-1b-base"] + sorted({x for p in INC.values() for x in p} - {"olmo2-1b-base"})
arr = {}
for n_ in names:
    if n_ == "olmo2-1b-base":
        arr[n_], ref = A.load_geo(n_)
    else:
        arr[n_], _ = A.load_geo(n_, ref)


def stats_for(tpl, ents, fixed, full):
    w = np.bincount(tpl, minlength=8) / len(tpl)
    H = {n_: np.tensordot(arr[n_], w, axes=([1], [0]))[ents] for n_ in names}
    base = A.Ck(H["olmo2-1b-base"]); rng = np.random.default_rng(7); res = {}
    for i, (k, (p, c)) in enumerate(INC.items()):
        if not full and k not in MAIN:
            continue
        rg = np.random.default_rng(1000 + i) if fixed else rng
        res[k] = A.stage_vs_base(base, base.H + (H[c] - H[p]), m, r, 999, rg, full_null=full)
    h = A.holm([res[k][q] for k in MAIN for q in ("p_blue", "p_red")], 8)
    for j, k in enumerate(MAIN):
        (ab, rb), (ar, rr) = h[2 * j], h[2 * j + 1]
        res[k].update(holm_p_blue=ab, holm_p_red=ar, verdict=A.verdict(rb, rr, res[k]["reff_R"]))
    return res


res = stats_for(np.arange(8), np.arange(500), True, True)
for k in INC:
    x = res[k]
    print(k, "ratio %.4f f_low %.5g (null med %.5g) p_blue %.3f p_red %.3f %s (%.0fs)" % (x["ratio"], x["f_low"], x["f_low_null_median"], x["p_blue"], x["p_red"], x.get("verdict", "non-voting"), time.time() - t0), flush=True)
out = {"m": m, "r": r, "B": B, "increments": INC, "point": res}
rng = np.random.default_rng(20261111); rec = {k: [] for k in MAIN}
for b_ in range(B):
    tpl = rng.integers(0, 8, 8); ents = np.sort(rng.choice(500, 400, replace=False))
    rs = stats_for(tpl, ents, False, False)
    for k in MAIN:
        rec[k].append(rs[k]["verdict"])
    if b_ % 50 == 0:
        print("boot", b_, round(time.time() - t0), "s", flush=True)
out["stability"] = {}
for k in MAIN:
    v = np.array(rec[k]); pv = res[k]["verdict"]
    out["stability"][k] = dict(point_verdict=pv, stability=float((v == pv).mean()), counts={u: int((v == u).sum()) for u in set(v)}, stable_claim=bool((v == pv).mean() >= 0.9))
json.dump(out, open("data/increments_results.json", "w"), indent=1)
print("DONE", round(time.time() - t0), "s")
