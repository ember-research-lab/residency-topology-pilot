"""Step 14 (EXPLORATORY, non-voting): (1) F4(a) colour test of the 1B stages on C1/C2 controls at l*=12; (2) smoothness diagnostic on the
Olmo-3 7B pair at l*=16: kernel ridge of D on H_b, colour of the cross-fitted residual, geo-vs-C1 enrichment difference with a paired
entity-subsample bootstrap. Analysis only (no model loads). -> data/step14_smoothness_7b.json. Run under the cgroup cap."""
import csv, json, time
import numpy as np
from rtp import analysis as A, graph as G, cap
cap.require_cap()
t0 = time.time(); K = 8; out = {}
def rows(name, code, layer, f="acts.npz"):
    a = np.load(f"runs/{name}/{f}"); meta = a["meta"]; sel = np.where(meta[:, 0] == code)[0]; n = len(sel) // K
    assert (meta[sel, 1] == np.repeat(np.arange(n), K)).all()
    return a[f"last_L{layer}"][sel].reshape(n, K, -1).astype(np.float64)
def frozen_m(arr):
    rng = np.random.default_rng(20261104); n = len(arr); sp = []
    for _ in range(100):
        t = rng.integers(0, K, K); sub = np.sort(rng.choice(n, int(0.8 * n), replace=False)); sp.append(np.linalg.eigvalsh(G.laplacian(arr[sub][:, t].mean(1), 7)))
    return G.gap_rule_m(sp)
def enr(pt): return pt["f_low"] / pt["f_low_null_median"]
def slim(pt): return {k: pt[k] for k in ("ratio", "f_low", "f_low_null_median", "p_blue", "p_red", "reff_R", "lambda_lin")} | dict(enrichment=enr(pt), percentile=1 - pt["p_red"])
# ---- part 1: 1B controls
SETS = {"geo": 0, "C1_nouns": 1, "C2_nonce": 2}; r = 50; p1 = {}
for sn in ("C1_nouns", "C2_nonce"):
    ab = rows("olmo2-1b-base", SETS[sn], 12); m = frozen_m(ab); base = A.Ck(ab.mean(1)); p1[sn] = {"m": m, "r": r}
    for st in ("sft", "dpo", "rlvr1", "instruct"):
        pt = A.stage_vs_base(base, rows(st, SETS[sn], 12).mean(1), m, r, 999, np.random.default_rng(3000 + SETS[sn]))
        p1[sn][st] = slim(pt); print("1B", sn, st, "ratio %.4f enr %.2f pct %.3f" % (pt["ratio"], enr(pt), 1 - pt["p_red"]), flush=True)
s9 = json.load(open("data/step9_results.json"))
p1["geo_reference_step9"] = {st: dict(ratio=s9[st]["ratio"], enrichment=enr(s9[st]), percentile=1 - s9[st]["p_red"]) for st in ("sft", "dpo", "rlvr1", "instruct")}
out["part1_1b_controls"] = p1; json.dump(out, open("data/step14_smoothness_7b.json", "w"), indent=1)
# ---- part 2: smoothness on 7B
def krr_fit_predict(Xtr, Ytr, Xte, sigma, grid=np.logspace(-6, 2, 17), folds=5, seed=0):
    def kern(A_, B_): return np.exp(-np.maximum((A_ ** 2).sum(1)[:, None] + (B_ ** 2).sum(1)[None] - 2 * A_ @ B_.T, 0) / (2 * sigma ** 2))
    mu = Ytr.mean(0); Kt = kern(Xtr, Xtr); n = len(Xtr); idx = np.random.default_rng(seed).permutation(n); parts = np.array_split(idx, folds); best = None
    for lam in grid:
        sse = 0.0
        for te in parts:
            tr = np.setdiff1d(idx, te); al = np.linalg.solve(Kt[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Ytr[tr] - Ytr[tr].mean(0))
            sse += ((Ytr[te] - Ytr[tr].mean(0) - Kt[np.ix_(te, tr)] @ al) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, lam)
    al = np.linalg.solve(Kt + best[1] * np.eye(n), Ytr - mu)
    return kern(Xte, Xtr) @ al + mu, best[1]
p2 = {}; cache = {}
for sn, code in SETS.items():
    ab, as_ = rows("olmo3-7b-base", code, 16) if code < 2 else rows("olmo3-7b-base", code, 16, "acts13.npz"), None
    as_ = rows("olmo3-7b-rlzero", code, 16) if code < 2 else rows("olmo3-7b-rlzero", code, 16, "acts13.npz")
    Hb = ab.mean(1); Hb = Hb - Hb.mean(0); D = as_.mean(1) - ab.mean(1); D = D - D.mean(0); n = len(Hb)
    d2 = ((Hb[:, None] - Hb[None]) ** 2).sum(-1); sigma = float(np.sqrt(np.median(d2[np.triu_indices(n, 1)])))
    perm = np.random.default_rng(5).permutation(n); te = perm[: n // 5]; tr = perm[n // 5:]
    Dh, lam = krr_fit_predict(Hb[tr], D[tr], Hb[te], sigma); r2 = float(1 - ((D[te] - Dh) ** 2).sum() / ((D[te] - D[tr].mean(0)) ** 2).sum())
    res = np.zeros_like(D); lams = []
    for k, part in enumerate(np.array_split(np.random.default_rng(6).permutation(n), 5)):      # 5-fold cross-fit so every row has a held-out residual
        trk = np.setdiff1d(np.arange(n), part); Dh_k, lk = krr_fit_predict(Hb[trk], D[trk], Hb[part], sigma, seed=k); res[part] = D[part] - Dh_k; lams.append(lk)
    r2_cf = float(1 - (res ** 2).sum() / (D ** 2).sum())
    base = A.Ck(ab.mean(1)); m = frozen_m(ab)
    pt = A.stage_vs_base(base, ab.mean(1) + res, m, r, 999, np.random.default_rng(4000 + code))
    raw = A.stage_vs_base(base, as_.mean(1), m, r, 999, np.random.default_rng(1000 + code))
    p2[sn] = dict(m=m, sigma=sigma, heldout_r2_80_20=r2, lambda_80_20=lam, crossfit_r2_all_rows=r2_cf, lambdas_crossfit=lams, raw=slim(raw), residual_after_smooth_removal=slim(pt))
    cache[sn] = (ab, as_, m)
    print("7B", sn, "heldout R2 %.3f crossfit %.3f | raw enr %.2f pct %.3f | residual enr %.2f pct %.3f" % (r2, r2_cf, enr(raw), 1 - raw["p_red"], enr(pt), 1 - pt["p_red"]), flush=True)
out["part2_smoothness"] = p2; json.dump(out, open("data/step14_smoothness_7b.json", "w"), indent=1)
# paired bootstrap: geography entity subsample (80%, no replacement: duplicates make the self-tuning graph singular) with its matched C1 nouns; P=99 null
geo = [r_["name"] for r_ in csv.DictReader(open("data/entities/geo_v1.csv", encoding="utf-8"))]; mg = [r_["matched_geo"] for r_ in csv.DictReader(open("data/entities/control_nouns_v1.csv", encoding="utf-8"))]
pos = {g_: i for i, g_ in enumerate(geo)}; pair = np.array([pos[x] for x in mg])    # noun j <-> geo pair[j]
inv = {int(g_): j for j, g_ in enumerate(pair)}; assert len(inv) == 500
gb, gs, mgeo = cache["geo"]; nb, ns, mn = cache["C1_nouns"]; B = 1000; rng = np.random.default_rng(14); diffs, eg, en = [], [], []
for b in range(B):
    sub = np.sort(rng.choice(500, 400, replace=False)); sn_ = np.array([inv[int(i)] for i in sub])
    e = []
    for (xb, xs, m_, idx) in ((gb, gs, mgeo, sub), (nb, ns, mn, sn_)):
        base = A.Ck(xb[idx].mean(1)); pt = A.stage_vs_base(base, xs[idx].mean(1), m_, r, 99, rng); e.append(enr(pt))
    eg.append(e[0]); en.append(e[1]); diffs.append(e[0] - e[1])
    if b % 100 == 0: print("boot", b, round(time.time() - t0), flush=True)
d = np.array(diffs)
out["geo_minus_C1_enrichment"] = dict(B=B, subsample="80% of entities without replacement, geography entity paired with its matched noun (matched_geo), null P=99", point_enrich=dict(geo=enr(raw_ := None) if False else p2["geo"]["raw"]["enrichment"], C1=p2["C1_nouns"]["raw"]["enrichment"], C2=p2["C2_nonce"]["raw"]["enrichment"]),
    boot_mean_geo=float(np.mean(eg)), boot_mean_C1=float(np.mean(en)), diff_mean=float(d.mean()), diff_ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))], point_diff=p2["geo"]["raw"]["enrichment"] - p2["C1_nouns"]["raw"]["enrichment"])
out["wall_s"] = round(time.time() - t0); json.dump(out, open("data/step14_smoothness_7b.json", "w"), indent=1); print("DONE", json.dumps(out["geo_minus_C1_enrichment"]), flush=True)
