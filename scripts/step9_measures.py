"""Step 9: all measures + N4' (P=999) + Holm over the 8 preregistered tests (6 present until step 11) + self-calibration.
Usage: uv run python scripts/step9_measures.py   -> data/step9_results.json"""
import json, time, tomllib
import numpy as np
from scipy import stats
from rtp import analysis as A

t0 = time.time()
from rtp import lineage
_ck = json.load(open("data/checkpoints.json")); _base = tomllib.load(open("configs/pilot.toml", "rb"))["hosts"]["olmo2_1b"]["base"]["sha"]
for _k, _v in _ck.items():                     # lineage gate: every compared checkpoint needs a PASS record against the base
    if isinstance(_v, dict) and "sha" in _v and _v["sha"] != _base:   # the base itself is the parent
        lineage.require_pass(_base, _v["sha"])
cfg = tomllib.load(open("configs/pilot.toml", "rb"))["frozen"]; m, r = cfg["m"], cfg["r"]
base_arr, ref = A.load_geo("olmo2-1b-base"); base = A.Ck(base_arr.mean(1))
STAGES = ["sft", "dpo", "rlvr1", "instruct"]
FAMILY = ["sft", "dpo", "instruct"]       # preregistered family; Olmo-3 RL-Zero (step 11) not yet run
res = {"m": m, "r": r, "P": 999}
rng = np.random.default_rng(20261104)
arrs = {s: A.load_geo(s, ref)[0].mean(1) for s in STAGES}
for i, s in enumerate(STAGES):
    res[s] = A.stage_vs_base(base, arrs[s], m, r, 999, np.random.default_rng(1000 + i), full_null=True)   # seed shared with step 10 identity draw
    print(s, "ratio %.4f f_low %.5g p_blue %.4f p_red %.4f reff %.1f  (%.0fs)" % (res[s]["ratio"], res[s]["f_low"], res[s]["p_blue"], res[s]["p_red"], res[s]["reff_R"], time.time() - t0), flush=True)
# Holm over the family of 8 (Olmo-3 pair absent -> p=1, provisional); RLVR1 secondary with its own 2-p Holm
for s_, v in A.family_verdicts(res).items():
    res[s_].update(v)
res["sft"]["verdict_family"] = res["dpo"]["verdict_family"] = res["instruct"]["verdict_family"] = "primary family, provisional (Olmo-3 pair absent)"
res["rlvr1"]["verdict_family"] = "secondary (own Holm over its 2 p-values)"
# r sensitivity 25/50/100 (f_low only, P=999)
res["r_sensitivity"] = {}
for s in STAGES:
    res["r_sensitivity"][s] = {}
    for rr_ in (r // 2, r, 2 * r):
        o = A.stage_vs_base(base, A.Ck(arrs[s]), m, rr_, 999, rng)
        res["r_sensitivity"][s][rr_] = dict(f_low=o["f_low"], p_blue=o["p_blue"], p_red=o["p_red"], lambda_lin=o["lambda_lin"], reff_R=o["reff_R"])
# m sensitivity (non-voting): m = m*-2, m*+2, 10
res["m_sensitivity"] = {}
for s in STAGES:
    res["m_sensitivity"][s] = {}
    for mm in (m - 2, m + 2, 10):
        o = A.stage_vs_base(base, A.Ck(arrs[s]), mm, r, 999, rng)
        res["m_sensitivity"][s][mm] = dict(f_low=o["f_low"], p_blue=o["p_blue"], p_red=o["p_red"], d_O=o["d_O"])
# reference rows: ingredients and RL steps vs base (point, non-voting) + RL-step monotonicity from DPO
refs = {}
for nme in ["ing1", "ing2", "ing3", "inst_step200", "inst_step800", "inst_step1600", "inst_step2600"]:  # vs base
    try:
        Hn = A.load_geo(nme, ref)[0].mean(1)
    except FileNotFoundError:
        continue
    st = A.Ck(Hn); refs[nme] = dict(ratio=float(np.linalg.norm(st.H - base.H) / np.linalg.norm(base.H)), **A.triplet(base, st, m),
                                    norm_D_from_dpo=float(np.linalg.norm(st.H - A.Ck(arrs["dpo"]).H)))
res["reference_rows"] = refs
pairs = {}
for a_, b_ in [("ing1", "ing2"), ("ing1", "ing3"), ("ing2", "ing3"), ("inst_step200", "inst_step800"), ("inst_step800", "inst_step1600"), ("inst_step1600", "inst_step2600")]:
    ca, cb = A.Ck(A.load_geo(a_, ref)[0].mean(1)), A.Ck(A.load_geo(b_, ref)[0].mean(1))
    pairs[f"{a_}|{b_}"] = dict(ratio=float(np.linalg.norm(ca.H - cb.H) / np.linalg.norm(base.H)), **A.triplet(ca, cb, m))
res["epsilon_point_pairs"] = pairs
steps = ["inst_step200", "inst_step800", "inst_step1600", "inst_step2600"]
res["rl_step_norm_D_vs_base"] = {k: refs[k]["ratio"] for k in steps if k in refs}
# self-calibration: a permuted D treated as observed, 50 times (SFT), percentile uniform?
Dsft = A.Ck(arrs["sft"]).H - base.H; pcs = []
for _ in range(50):
    o = A.stage_vs_base(base, base.H + Dsft[rng.permutation(len(Dsft))], m, r, 199, rng)
    pcs.append(o["p_blue"])
ks = stats.kstest(pcs, "uniform"); res["self_calibration"] = dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), n=50, P=199, stage="sft", mean_pct=float(np.mean(pcs)))
print("self-calibration KS p = %.4f" % ks.pvalue)
res["wall_s"] = time.time() - t0
json.dump(res, open("data/step9_results.json", "w"), indent=1)
print("DONE", round(res["wall_s"]), "s")
