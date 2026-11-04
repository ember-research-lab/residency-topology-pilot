"""Step 9: all measures + N4' (P=999) + Holm over the 8 preregistered tests (6 present until step 11) + self-calibration.
Usage: uv run python scripts/step9_measures.py   -> data/step9_results.json"""
import json, time, tomllib
import numpy as np
from scipy import stats
from rtp import analysis as A

t0 = time.time()
cfg = tomllib.load(open("configs/pilot.toml", "rb"))["frozen"]; m, r = cfg["m"], cfg["r"]
base_arr, ref = A.load_geo("olmo2-1b-base"); base = A.Ck(base_arr.mean(1))
STAGES = ["sft", "dpo", "rlvr1", "instruct"]
FAMILY = ["sft", "dpo", "instruct"]       # preregistered family; Olmo-3 RL-Zero (step 11) not yet run
res = {"m": m, "r": r, "P": 999}
rng = np.random.default_rng(20261104)
arrs = {s: A.load_geo(s, ref)[0].mean(1) for s in STAGES}
for s in STAGES:
    res[s] = A.stage_vs_base(base, arrs[s], m, r, 999, rng, full_null=True)
    print(s, "ratio %.4f f_low %.5g p_blue %.4f p_red %.4f reff %.1f  (%.0fs)" % (res[s]["ratio"], res[s]["f_low"], res[s]["p_blue"], res[s]["p_red"], res[s]["reff_R"], time.time() - t0), flush=True)
# Holm: family of 8 = {sft,dpo,instruct,rl_zero} x {blue,red}; the 2 absent Olmo-3 p-values count as 1 (provisional)
ps = [res[s][k] for s in FAMILY for k in ("p_blue", "p_red")]
h = A.holm(ps, 8)
for i, s in enumerate(FAMILY):
    (ab, rb), (ar, rr) = h[2 * i], h[2 * i + 1]
    res[s].update(holm_p_blue=ab, holm_p_red=ar, verdict=A.verdict(rb, rr, res[s]["reff_R"]), verdict_family="primary (provisional, Olmo-3 pair absent)")
h2 = A.holm([res["rlvr1"]["p_blue"], res["rlvr1"]["p_red"]], 2)
res["rlvr1"].update(holm_p_blue=h2[0][0], holm_p_red=h2[1][0], verdict=A.verdict(h2[0][1], h2[1][1], res["rlvr1"]["reff_R"]), verdict_family="secondary (own Holm over its 2 p-values)")
# r sensitivity 25/50/100 (f_low only, P=999)
res["r_sensitivity"] = {}
for s in STAGES:
    res["r_sensitivity"][s] = {}
    for rr_ in (r // 2, r, 2 * r):
        o = A.stage_vs_base(base, A.Ck(arrs[s]), m, rr_, 999, rng)
        res["r_sensitivity"][s][rr_] = dict(f_low=o["f_low"], p_blue=o["p_blue"], p_red=o["p_red"], lambda_lin=o["lambda_lin"], reff_R=o["reff_R"])
# reference rows: ingredients and RL steps vs base (point, non-voting) + RL-step monotonicity from DPO
refs = {}
for nme in ["ing1", "ing2", "ing3", "inst_step200", "inst_step800", "inst_step1600", "inst_step2600"]:
    try:
        Hn = A.load_geo(nme, ref)[0].mean(1)
    except FileNotFoundError:
        continue
    st = A.Ck(Hn); refs[nme] = dict(ratio=float(np.linalg.norm(st.H - base.H) / np.linalg.norm(base.H)), **A.triplet(base, st, m),
                                    norm_D_from_dpo=float(np.linalg.norm(st.H - A.Ck(arrs["dpo"]).H)))
res["reference_rows"] = refs
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
