"""t4: all measures of PREREG-toy sec.2 (+ AMENDMENT 1) from toy/runs/<stage>_s<seed>/acts.npz.
Usage: uv run python toy/scripts/t4_analyze.py <runs_dir> <out.json> <B_real> <B_cal> [stages comma list]
Reuses rtp.analysis.Ck / rtp.measures.f_low_gram; the base graph (U, P_r) is rebuilt per bootstrap draw from the base entity subsample (m, r frozen)."""
import glob, itertools, json, re, sys, time
import numpy as np
sys.path.insert(0, "toy")
from rtp import analysis as A, measures as M, cap
runs, outf, B_REAL, B_CAL = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t0 = time.time()
gate = json.load(open("toy/data/gate_smollm2.json")); FR = gate["frozen"]
base = np.load(f"{runs}/base_smollm2/acts.npz")
import toylib as T
HB = {k: base[f"H_{k}"].mean(1).astype(np.float64) for k in T.SETCODE}
# paired geo<->noun index arrays (full survivors -> rows in survivor order)
geo_rows, nn_rows = T.read_rows("geo"), T.read_rows("nouns")
gpos = {int(r): j for j, r in enumerate(base["idx_geo"])}; gname = {g["name"]: i for i, g in enumerate(geo_rows)}
pair_g, pair_n = [], []
for j, r in enumerate(base["idx_nouns"]):
    g = gname[nn_rows[int(r)]["matched_geo"]]
    if g in gpos: pair_g.append(gpos[g]); pair_n.append(j)
pair_g, pair_n = np.array(pair_g), np.array(pair_n); NP_ = len(pair_g)
stages = {}
for f in sorted(glob.glob(f"{runs}/*_s*/acts.npz")):
    mt = re.search(r"/(sft|dpo|grpo|rlsft)_s(\d+)/acts.npz", f)
    if not mt: continue
    st, sd = mt.groups()
    a = np.load(f); H = {k: a[f"H_{k}"].mean(1).astype(np.float64) for k in T.SETCODE}
    if st == "rlsft":   # AMENDMENT 2: parent = the re-trained SFT of the same seed; store H_RL - H_SFTre + H_base so that (stored - HB) is the parent-relative D
        P = np.load(f"{runs}/sftre_s{sd}/acts.npz"); H = {k: H[k] - P[f"H_{k}"].mean(1).astype(np.float64) + HB[k] for k in T.SETCODE}
    stages.setdefault(st, {})[int(sd)] = H
if len(sys.argv) > 5: stages = {s: v for s, v in stages.items() if s in sys.argv[5].split(",")}
def centred(X): return X - X.mean(0)
def fro(X): return float(np.linalg.norm(centred(X)))
# ---------- f_low machinery with cached base draws
class Draw:   # base graph of one entity subsample
    def __init__(self, Hb, m, r, idx):
        self.idx, self.m = idx, m; ck = A.Ck(Hb[idx]); self.U = ck.U; self.P = ck.V[:, ::-1][:, :r]
    def fl(self, D):
        Dc = centred(D[self.idx]); R = Dc - self.P @ (self.P.T @ Dc)
        e = ((self.U.T @ R) ** 2).sum(1); return float(e[1:self.m + 1].sum() / e[1:].sum())
def make_draws(Hb, m, r, rng, B, pool):
    n = len(pool); return [Draw(Hb, m, r, np.sort(rng.choice(pool, int(0.8 * n), replace=False))) for _ in range(B)]
def make_paired_draws(rng, B):
    g, n = [], []
    for _ in range(B):
        sel = np.sort(rng.choice(NP_, int(0.8 * NP_), replace=False))
        g.append(Draw(HB["geo"], FR["geo"]["m"], FR["geo"]["r"], pair_g[sel])); n.append(Draw(HB["nouns"], FR["nouns"]["m"], FR["nouns"]["r"], pair_n[sel]))
    return g, n
def enrich(draw, D_num, D_bands, resample):
    """E = f_low(num) / median over (resampled) band pairs of f_low."""
    fb = np.array([draw.fl(D) for D in D_bands]); return draw.fl(D_num) / np.median(fb[resample])
def boot_E(draws, D_num, D_bands, rng):
    out = []
    for d in draws:
        fb = np.array([d.fl(D) for D in D_bands]); out.append(d.fl(D_num) / np.median(fb[rng.integers(0, len(fb), len(fb))]))
    return np.array(out)
def boot_delta(gd, nd, Dg, Dn, bg, bn, rng):
    """Paired Delta = E(geo) - E(nouns) per draw; same pair resample for both sets."""
    out = []
    for g, n in zip(gd, nd):
        rs = rng.integers(0, len(bg), len(bg))
        fg = np.array([g.fl(D) for D in bg]); fn = np.array([n.fl(D) for D in bn])
        out.append(g.fl(Dg) / np.median(fg[rs]) - n.fl(Dn) / np.median(fn[rs]))
    return np.array(out)
ci = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
# ---------- KRR smoothness (step-14 recipe)
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
    return kern(Xte, Xtr) @ al + mu
def krr_r2(Hb, D):
    Hc = centred(Hb); D = centred(D); n = len(Hc)
    d2 = ((Hc[:, None] - Hc[None]) ** 2).sum(-1); sigma = float(np.sqrt(np.median(d2[np.triu_indices(n, 1)])))
    perm = np.random.default_rng(5).permutation(n); te, tr = perm[:n // 5], perm[n // 5:]
    Dh = krr_fit_predict(Hc[tr], D[tr], Hc[te], sigma)
    return float(1 - ((D[te] - Dh) ** 2).sum() / ((D[te] - D[tr].mean(0)) ** 2).sum())
# ---------- point measures + real bootstrap
res = dict(config=dict(B_real=B_REAL, B_cal=B_CAL, frozen=FR, base_sha=gate["sha"], n_pairs_geo_noun=int(NP_)), stages={})
rng = np.random.default_rng(20261105)
G_real, N_real = make_paired_draws(rng, B_REAL)
fullD = {}   # per stage: dict(seeds, D_stage[set], pairs[(i,j)], D_pair[set][pair])
for st, sd in stages.items():
    if len(sd) < 2: print("skip", st, "needs >= 2 seeds for a seed band; have", len(sd), flush=True); continue
    seeds = sorted(sd); k = len(seeds); pairs = list(itertools.combinations(seeds, 2))
    Dst = {s: centred(np.mean([sd[i][s] for i in seeds], 0) - HB[s]) for s in T.SETCODE}
    Dpr = {s: [centred(sd[i][s] - sd[j][s]) for i, j in pairs] for s in T.SETCODE}
    fullD[st] = dict(seeds=seeds, pairs=pairs, Dst=Dst, Dpr=Dpr, sd=sd)
    r = dict(seeds=seeds, k=k, pairs=pairs, per_set={})
    for s in T.SETCODE:
        nH = fro(HB[s]); nst = fro(Dst[s]) / nH; nps = [fro(D) / nH for D in Dpr[s]]
        single = [fro(sd[i][s] - HB[s]) / nH for i in seeds]
        r["per_set"][s] = dict(norm_D_stage=nst, norm_D_seed_pairs=nps, norm_D_seed_median=float(np.median(nps)), norm_D_single_seed=single,
                               excess_over_noise=float(nst ** 2 - np.median(nps) ** 2 / (2 * k)), ratio_seed_over_stage=float(np.median(nps) / nst),
                               krr_r2_stage=krr_r2(HB[s], Dst[s]), krr_r2_seed_pairs=[krr_r2(HB[s], D) for D in Dpr[s]])
    # enrichment per set (unpaired sets on their own 80% draws)
    for s in ("nonce", "fig"):
        dr = make_draws(HB[s], FR[s]["m"], FR[s]["r"], rng, B_REAL, np.arange(len(HB[s])))
        full = Draw(HB[s], FR[s]["m"], FR[s]["r"], np.arange(len(HB[s])))
        pt = enrich(full, Dst[s], Dpr[s], np.arange(len(pairs))); bs = boot_E(dr, Dst[s], Dpr[s], rng)
        r["per_set"][s].update(E=pt, E_ci=ci(bs), E_boot_mean=float(bs.mean()))
    for s, pool, pg in (("geo", pair_g, None), ("nouns", pair_n, None)):
        full = Draw(HB[s], FR[s]["m"], FR[s]["r"], pool)
        pt = enrich(full, Dst[s], Dpr[s], np.arange(len(pairs)))
        bs = boot_E(G_real if s == "geo" else N_real, Dst[s], Dpr[s], rng)
        r["per_set"][s].update(E=pt, E_ci=ci(bs), E_boot_mean=float(bs.mean()))
    dl = boot_delta(G_real, N_real, Dst["geo"], Dst["nouns"], Dpr["geo"], Dpr["nouns"], rng)
    r["delta_geo_minus_nouns"] = dict(point=r["per_set"]["geo"]["E"] - r["per_set"]["nouns"]["E"], ci=ci(dl), boot_mean=float(dl.mean()))
    r["delta_fig_minus_nonce"] = dict(point=r["per_set"]["fig"]["E"] - r["per_set"]["nonce"]["E"])
    res["stages"][st] = r; print("real", st, round(time.time() - t0), "s", flush=True)
    json.dump(res, open(outf, "w"), indent=1)
# ---------- calibration: planted rewiring on seed-noise pseudo-stages
G_cal, N_cal = make_paired_draws(np.random.default_rng(777), B_CAL)
Ug = A.Ck(HB["geo"]).U; mg = FR["geo"]["m"]
def planted(N, s, Wd, Ng):
    if s == 0: return N
    Nc, Wc = centred(N), centred(Wd)
    return centred(np.sqrt(1 - s) * Nc / np.linalg.norm(Nc) * np.linalg.norm(Nc) + np.sqrt(s) * Wc / np.linalg.norm(Wc) * np.linalg.norm(Nc))
cal = {}
SH = [0.0, 0.05, 0.10, 0.20]
for st, fd in fullD.items():
    if len(fd["seeds"]) < 3: continue
    pairs = fd["pairs"]; npairs = pairs[:3] if len(pairs) > 3 else pairs
    for ab in npairs:
        ia = pairs.index(ab); rest = [p for p in range(len(pairs)) if p != ia]
        Ng = fd["Dpr"]["geo"][ia] / np.sqrt(2); Nn = fd["Dpr"]["nouns"][ia] / np.sqrt(2)
        bg = [fd["Dpr"]["geo"][p] for p in rest]; bn = [fd["Dpr"]["nouns"][p] for p in rest]
        for rs_ in range(3):
            W = Ug[:, 1:mg + 1] @ np.random.default_rng(900 + rs_).normal(size=(mg, HB["geo"].shape[1]))
            for s in SH:
                Dg = planted(Ng, s, W, Ng)
                dl = boot_delta(G_cal, N_cal, Dg, Nn, bg, bn, np.random.default_rng(5))
                lo, hi = ci(dl); cal.setdefault(str(s), []).append(dict(stage=st, pair=list(ab), rewire_seed=rs_, delta_boot_mean=float(dl.mean()), ci=[lo, hi], detect=bool(lo > 0)))
    print("cal", st, round(time.time() - t0), "s", flush=True)
    res["calibration"] = {s: dict(n=len(v), detection_rate=float(np.mean([x["detect"] for x in v])), mean_delta=float(np.mean([x["delta_boot_mean"] for x in v])), reps=v) for s, v in cal.items()}
    json.dump(res, open(outf, "w"), indent=1)
# ---------- falsifiers
import os
res["grpo_read"] = os.path.exists("toy/data/rl_read.flag")   # AMENDMENT 1/2 read gate (>= 10 points over parent), written by the driver of this run after reading toy/runs/rlsft_s*.json
rd = [st for st in res["stages"] if st not in ("grpo", "rlsft") or res.get("grpo_read")]
if rd:
    med_st = float(np.median([res["stages"][s]["per_set"]["geo"]["norm_D_stage"] for s in rd])); med_sd = float(np.median([res["stages"][s]["per_set"]["geo"]["norm_D_seed_median"] for s in rd]))
    res["F_T2"] = dict(stages_used=rd, median_norm_D_stage=med_st, median_norm_D_seed=med_sd, fires=bool(med_st <= med_sd))
if "calibration" in res:
    c = res["calibration"]; res["F_T3"] = dict(power_at_0_10=c["0.1"]["detection_rate"], size_at_0=c["0.0"]["detection_rate"], fires=bool(c["0.1"]["detection_rate"] < 0.8 or c["0.0"]["detection_rate"] > 0.10))
res["wall_s"] = round(time.time() - t0); json.dump(res, open(outf, "w"), indent=1); print("DONE", json.dumps({k: res.get(k) for k in ("F_T2", "F_T3")}), flush=True)
