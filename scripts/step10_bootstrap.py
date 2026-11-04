"""Step 10: bootstrap B=1000 (templates K-of-K with replacement x 80% entity subsample without replacement; m, r, l* frozen).
Per replicate the same draw is applied to base, all stages, ingredients and RL-step branches. Output data/step10_results.json.
Identity draw (all templates, all entities, same seeds as step 9) must reproduce the step-9 point estimate exactly."""
import json, time, tomllib
import numpy as np
from rtp import analysis as A

t0 = time.time()
cfg = tomllib.load(open("configs/pilot.toml", "rb"))["frozen"]; m, r = cfg["m"], cfg["r"]
B = tomllib.load(open("configs/pilot.toml", "rb"))["measure"]["B"]
names = ["olmo2-1b-base", "sft", "dpo", "rlvr1", "instruct", "ing1", "ing2", "ing3", "inst_step200", "inst_step800", "inst_step1600", "inst_step2600"]
arr = {}
for n_ in names:
    arr[n_], ids = A.load_geo(n_, None if n_ == "olmo2-1b-base" else ref) if n_ != "olmo2-1b-base" else A.load_geo(n_)
    if n_ == "olmo2-1b-base":
        ref = ids
STAGES = ["sft", "dpo", "rlvr1", "instruct"]
EPS_PAIRS = {"A": [("ing1", "ing2"), ("ing1", "ing3"), ("ing2", "ing3")],
             "N6": [("inst_step200", "inst_step800"), ("inst_step800", "inst_step1600"), ("inst_step1600", "inst_step2600")]}
KEYS = ("d_O", "d_NP", "R_P")


def draw_stats(tpl, ents, seeds_fixed: bool):
    H = {n_: arr[n_][ents][:, tpl].mean(1) for n_ in names}
    base = A.Ck(H["olmo2-1b-base"]); rng = np.random.default_rng(7)
    res = {}
    for i, s in enumerate(STAGES):
        rg = np.random.default_rng(1000 + i) if seeds_fixed else rng
        res[s] = A.stage_vs_base(base, H[s], m, r, 999, rg)
    for s, v in A.family_verdicts(res).items():
        res[s].update(v)
    eps = {}
    for fam, prs in EPS_PAIRS.items():
        for a_, b_ in prs:
            eps[f"{a_}|{b_}"] = A.triplet(A.Ck(H[a_]), A.Ck(H[b_]), m)
    return res, eps


# ---- identity draw must reproduce step 9 exactly
pt = json.load(open("data/step9_results.json"))
res0, eps0 = draw_stats(np.arange(8), np.arange(500), True)
for s in STAGES:
    for k in ("f_low", "d_O", "d_NP", "R_P", "norm_D", "p_blue", "p_red", "reff_R", "verdict"):
        a, b = res0[s][k], pt[s][k]
        if a != b and not (isinstance(a, float) and abs(a - b) <= 1e-12 * max(1, abs(b))):
            raise SystemExit(f"BROKEN: identity draw != step-9 point estimate at {s}.{k}: {a} vs {b}")
for k, v in eps0.items():
    for q in KEYS:
        if abs(v[q] - pt["epsilon_point_pairs"][k][q]) > 1e-12:
            raise SystemExit(f"BROKEN: identity draw != step-9 epsilon pair {k}.{q}")
print("identity draw reproduces step-9 point estimate: OK", flush=True)

rng = np.random.default_rng(20261110)
rec = {s: {k: [] for k in ("f_low", "p_blue", "p_red", "verdict", "reff_R", "ratio", *KEYS)} for s in STAGES}
rep_eps = {k: {q: [] for q in KEYS} for k in eps0}
for b_ in range(B):
    tpl = rng.integers(0, 8, 8); ents = np.sort(rng.choice(500, 400, replace=False))
    res, eps = draw_stats(tpl, ents, False)
    for s in STAGES:
        for k in rec[s]:
            rec[s][k].append(res[s][k])
    for k in eps:
        for q in KEYS:
            rep_eps[k][q].append(eps[k][q])
    if b_ % 50 == 0:
        print(b_, round(time.time() - t0), "s", flush=True)

out = {"B": B, "m": m, "r": r, "n_entities_per_draw": 400, "P_per_draw": 999, "identity_draw": "OK", "stages": {}}
for s in STAGES:
    v = np.array(rec[s]["verdict"]); pv = pt[s]["verdict"]
    out["stages"][s] = dict(point_verdict=pv, stability=float((v == pv).mean()), verdict_counts={u: int((v == u).sum()) for u in set(v)},
                            stable_claim=bool((v == pv).mean() >= 0.9),
                            ci95={k: [float(np.percentile(rec[s][k], 2.5)), float(np.percentile(rec[s][k], 97.5))] for k in ("f_low", "ratio", *KEYS)})
eps = {}
for q in KEYS:
    ea = max(float(np.percentile(rep_eps[f"{a}|{b}"][q], 95)) for a, b in EPS_PAIRS["A"])
    en = max(float(np.percentile(rep_eps[f"{a}|{b}"][q], 95)) for a, b in EPS_PAIRS["N6"])
    eps[q] = dict(eps_A=ea, eps_N6=en, eps=max(ea, en))
out["epsilon"] = eps
out["epsilon_pairs_p95"] = {k: {q: float(np.percentile(v[q], 95)) for q in KEYS} for k, v in rep_eps.items()}
for s in STAGES:   # legacy P1 (DEGENERATE): within / beyond / INCONCLUSIVE vs epsilon
    out["stages"][s]["legacy_P1_DEGENERATE"] = {}
    for q in KEYS:
        lo, hi = out["stages"][s]["ci95"][q]
        e = eps[q]["eps"]
        out["stages"][s]["legacy_P1_DEGENERATE"][q] = "within eps" if hi < e else ("beyond eps" if lo > e else "INCONCLUSIVE")
out["wall_s"] = time.time() - t0
json.dump(out, open("data/step10_results.json", "w"), indent=1)
print("DONE", round(out["wall_s"]), "s")
