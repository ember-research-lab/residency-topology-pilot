"""Step 11 measures: Olmo-3-7B RL-Zero-Math vs its base at l* = 16 (1/2 L). Freezes m, r from the 7B base alone (step-4 rules), then the
step-9 pipeline (N4' P=999, full null), r/m sensitivity, bootstrap B=1000, and the COMPLETED Holm family of 8 p-values (1B SFT/DPO/Instruct
from data/step9_results.json held at their point values + RL-Zero). -> data/step11_results.json. Usage: uv run python scripts/step11_measures.py"""
import csv, json, time, tomllib
import numpy as np
from rtp import analysis as A, graph as G, measures as M, lineage

t0 = time.time()
LSTAR, K = 16, 8
BASE_SHA, RLZ_SHA = "a81bae42db3975be1671e27b9c9a56da1a9f980f", "dee26a4074b88215cd1326472e1ab14d0259d892"
lineage.require_pass(BASE_SHA, RLZ_SHA)
geo = list(csv.DictReader(open("data/entities/geo_v1.csv", encoding="utf-8"))); n = len(geo)
lat, lon = np.radians([float(g["lat"]) for g in geo]), np.radians([float(g["lon"]) for g in geo])
Y = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
def ridge_r2(X, Y, folds=5, seed=0, grid=np.logspace(-4, 4, 17)):
    """5-fold CV R2 (pooled over the 3 outputs); alpha (relative to mean eigenvalue of XX^T) by inner LOO."""
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
            Hd = (V * (w / (w + lam))) @ V.T                    # hat matrix
            res = (Y[tr] - mu) - Hd @ (Y[tr] - mu)
            loo = ((res / (1 - np.diag(Hd))[:, None]) ** 2).sum()
            if best is None or loo < best[0]: best = (loo, lam)
        lam = best[1]
        dual = np.linalg.solve(Kt + lam * np.eye(len(tr)), Y[tr] - mu)
        pred[te] = X[te] @ (Xt.T @ dual) + mu
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())



arr_b, ref = A.load_geo("olmo3-7b-base", None, layer=LSTAR); arr_s, _ = A.load_geo("olmo3-7b-rlzero", ref, layer=LSTAR)
arr_b8, _ = A.load_geo("olmo3-7b-base", None, layer=8)
fr = {"probe_r2": {"L8": ridge_r2(arr_b8.mean(1), Y), "L16": ridge_r2(arr_b.mean(1), Y)}, "l_star": LSTAR,
      "l_star_note": "1/2 L fixed by the step-11 truncation (3/4 L not extracted); R2 at 1/4 L recorded for reference"}
Hb = arr_b.mean(1); Hc = Hb - Hb.mean(0)
lam, U = G.eigh_sorted(G.laplacian(Hc, 7)); G.check_connected(lam)
rng = np.random.default_rng(20261104); spectra = []
for _ in range(100):
    t = rng.integers(0, K, K); sub = np.sort(rng.choice(n, int(0.8 * n), replace=False))
    spectra.append(np.linalg.eigvalsh(G.laplacian(arr_b[sub][:, t].mean(1), 7)))
m = G.gap_rule_m(spectra); r = M.r_from_base(Hc)
fr.update(m=m, m_point=G.gap_rule_m(lam), r=r, lambda_2=float(lam[1]), spectrum_2_22=[round(float(x), 5) for x in lam[1:22]])
print("freezes", json.dumps(fr), flush=True)
base = A.Ck(Hb); st = arr_s.mean(1)
pt = A.stage_vs_base(base, st, m, r, 999, np.random.default_rng(1000), full_null=True)
print("rlzero ratio %.4f f_low %.5g (null med %.5g) p_blue %.4f p_red %.4f reff %.1f lam_lin %.3f (%.0fs)" % (pt["ratio"], pt["f_low"], pt["f_low_null_median"], pt["p_blue"], pt["p_red"], pt["reff_R"], pt["lambda_lin"], time.time() - t0), flush=True)
s9 = json.load(open("data/step9_results.json"))
fam = [("sft", s9["sft"]), ("dpo", s9["dpo"]), ("instruct", s9["instruct"]), ("rlzero", pt)]
def holm_verdict(p_blue, p_red, reff):
    ps = [x for _, d in fam[:3] for x in (d["p_blue"], d["p_red"])] + [p_blue, p_red]
    h = A.holm(ps, 8)
    return h[6][0], h[7][0], A.verdict(h[6][1], h[7][1], reff)
pt["holm_p_blue"], pt["holm_p_red"], pt["verdict"] = holm_verdict(pt["p_blue"], pt["p_red"], pt["reff_R"])
h_all = A.holm([x for _, d in fam for x in (d["p_blue"], d["p_red"])], 8)
fam_out = {k: dict(holm_p_blue=h_all[2 * i][0], holm_p_red=h_all[2 * i + 1][0]) for i, (k, _) in enumerate(fam)}
sens = {}
for rr in (r // 2, 2 * r):
    o = A.stage_vs_base(base, A.Ck(st), m, rr, 999, np.random.default_rng(5)); sens[f"r={rr}"] = dict(f_low=o["f_low"], p_red=o["p_red"], p_blue=o["p_blue"], lambda_lin=o["lambda_lin"])
for mm in (max(m - 2, 2), m + 2, 10):
    o = A.stage_vs_base(base, A.Ck(st), mm, r, 999, np.random.default_rng(6)); sens[f"m={mm}"] = dict(f_low=o["f_low"], p_red=o["p_red"], p_blue=o["p_blue"], d_O=o["d_O"])
print("point verdict", pt["verdict"], "holm", pt["holm_p_blue"], pt["holm_p_red"], flush=True)
rng = np.random.default_rng(20261112); verd = []; rec = {"f_low": [], "ratio": [], "d_O": [], "d_NP": []}
for b_ in range(1000):
    tpl = rng.integers(0, K, K); ents = np.sort(rng.choice(n, 400, replace=False))
    w = np.bincount(tpl, minlength=K) / K
    Hb_, Hs_ = (np.tensordot(a, w, axes=([1], [0]))[ents] for a in (arr_b, arr_s))
    o = A.stage_vs_base(A.Ck(Hb_), Hs_, m, r, 999, rng)
    verd.append(holm_verdict(o["p_blue"], o["p_red"], o["reff_R"])[2])
    for k in rec: rec[k].append(o[k])
    if b_ % 100 == 0: print("boot", b_, round(time.time() - t0), "s", flush=True)
v = np.array(verd)
out = dict(freezes=fr, point=pt, family_holm=fam_out, sensitivity=sens, bootstrap=dict(B=1000, stability=float((v == pt["verdict"]).mean()), counts={u: int((v == u).sum()) for u in set(v)},
           ci95={k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] for k, x in rec.items()}, note="Holm family of 8: 1B SFT/DPO/Instruct p-values held at their step-9 point values"),
           wall_s=time.time() - t0)
json.dump(out, open("data/step11_results.json", "w"), indent=1)
print("DONE", round(out["wall_s"]), "s stability", out["bootstrap"]["stability"], "verdict", pt["verdict"])
