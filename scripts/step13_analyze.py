"""Step 13: F4 (token-statistics artefact) and P2 cost readout on the Olmo-3 7B pair at l*=16. Writes data/step13_f4_7b.json and
data/step13_cost_7b.json. Runs under the cgroup cap (CLAUDE.md rule 1). Usage: uv run python scripts/step13_analyze.py"""
import csv, json, time
import numpy as np
from scipy import stats
from wordfreq import zipf_frequency
from rtp import analysis as A, graph as G, measures as M, lineage, cap
cap.require_cap()
t0 = time.time(); LS, K, B = 16, 8, 1000
BASE_SHA, RLZ_SHA = "a81bae42db3975be1671e27b9c9a56da1a9f980f", "dee26a4074b88215cd1326472e1ab14d0259d892"
lineage.require_pass(BASE_SHA, RLZ_SHA)
CODE = {"geo": 0, "nouns": 1, "nonce": 2, "figures": 3}
def rows(name, setn, layer=LS):
    """(n,K,d) last-token vectors of a set from the step-11 (geo, nouns) or step-13 (nonce, figures, layer 0) archives."""
    f = {"geo": "acts.npz", "nouns": "acts.npz", "nonce": "acts13.npz", "figures": "acts13.npz"}[setn] if layer else "acts13_L0.npz"
    a = np.load(f"runs/{name}/{f}"); meta = a["meta"]; sel = np.where(meta[:, 0] == CODE[setn])[0]
    n = len(sel) // K; assert len(sel) == n * K and (meta[sel, 1] == np.repeat(np.arange(n), K)).all() and (meta[sel, 2] == np.tile(np.arange(K), n)).all()
    return a[f"last_L{layer}"][sel].reshape(n, K, -1).astype(np.float64)
def ridge_oof(X, Y, folds=5, seed=0, grid=np.logspace(-4, 4, 17)):
    """Out-of-fold ridge predictions (alpha per fold by inner LOO, relative to mean Gram eigenvalue)."""
    X = X - X.mean(0); Y = Y.reshape(len(Y), -1)
    idx = np.random.default_rng(seed).permutation(len(X)); parts = np.array_split(idx, folds); pred = np.zeros_like(Y)
    for te in parts:
        tr = np.setdiff1d(idx, te); Xt, mu = X[tr], Y[tr].mean(0)
        Kt = Xt @ Xt.T; w, V = np.linalg.eigh(Kt); sc = w.mean(); best = None
        for a in grid:
            lam = a * sc; Hd = (V * (w / (w + lam))) @ V.T; res = (Y[tr] - mu) - Hd @ (Y[tr] - mu)
            loo = ((res / (1 - np.diag(Hd))[:, None]) ** 2).sum()
            if best is None or loo < best[0]: best = (loo, lam)
        dual = np.linalg.solve(Kt + best[1] * np.eye(len(tr)), Y[tr] - mu); pred[te] = X[te] @ (Xt.T @ dual) + mu
    return pred
def r2(Y, P, sel=None):
    Y = Y.reshape(len(Y), -1); sel = np.arange(len(Y)) if sel is None else sel
    return float(1 - ((Y[sel] - P[sel]) ** 2).sum() / ((Y[sel] - Y[sel].mean(0)) ** 2).sum())
def delta_boot(Y, Pb, Ps, seed):
    """Delta R2 (s - b) with entity-bootstrap (resampling out-of-fold predictions) B=1000; one-sided p_drop = P*(delta* >= 0)."""
    rng = np.random.default_rng(seed); n = len(Y); d = np.array([r2(Y, Ps, s) - r2(Y, Pb, s) for s in (rng.integers(0, n, n) for _ in range(B))])
    pt = r2(Y, Ps) - r2(Y, Pb)
    return dict(r2_base=r2(Y, Pb), r2_stage=r2(Y, Ps), delta=pt, ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))], p_drop=float((1 + (d >= 0).sum()) / (B + 1)), boot=d)
geo = list(csv.DictReader(open("data/entities/geo_v1.csv", encoding="utf-8")))
lat, lon = np.radians([float(g["lat"]) for g in geo]), np.radians([float(g["lon"]) for g in geo])
Ygeo = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
figs = list(csv.DictReader(open("data/entities/figures_v1.csv", encoding="utf-8"))); Yfig = np.array([float(f["death_year"]) for f in figs])
fr = json.load(open("data/step11_results.json"))["freezes"]; m_geo, r = fr["m"], fr["r"]

# ---------------- F4 ----------------
f4 = {"definition": "PREREG line 249: fires if (a) C1 shows the same verdict pattern with comparable percentile, (b) base layer-0 low band overlaps the l*=16 low band with O>0.8, (c) |Spearman(u2, Zipf or token count)|>0.6"}
def frozen_m(arr):
    rng = np.random.default_rng(20261104); n = len(arr); sp = []
    for _ in range(100):
        t = rng.integers(0, K, K); sub = np.sort(rng.choice(n, int(0.8 * n), replace=False)); sp.append(np.linalg.eigvalsh(G.laplacian(arr[sub][:, t].mean(1), 7)))
    return G.gap_rule_m(sp)
for setn in ("nouns", "nonce"):
    ab, as_ = rows("olmo3-7b-base", setn), rows("olmo3-7b-rlzero", setn)
    mm = frozen_m(ab); base = A.Ck(ab.mean(1))
    pt = A.stage_vs_base(base, as_.mean(1), mm, r, 999, np.random.default_rng(2000 + CODE[setn]))
    h = A.holm([pt["p_blue"], pt["p_red"]], 2)
    pt.update(m=mm, r=r, percentile_f_low=1 - pt["p_red"], holm2_p_blue=h[0][0], holm2_p_red=h[1][0], verdict=A.verdict(h[0][1], h[1][1], pt["reff_R"]),
              geo_reference=dict(m=m_geo, r=r, p_red=0.001, ratio=0.0104, note="RL-Zero on geography, step 11"))
    pt.pop("t_hat", None); f4[setn] = pt
    print(setn, "m", mm, "ratio %.4f f_low %.5g null med %.5g p_blue %.3f p_red %.4f verdict %s" % (pt["ratio"], pt["f_low"], pt["f_low_null_median"], pt["p_blue"], pt["p_red"], pt["verdict"]), flush=True)
f4["a_fires"] = f4["nouns"]["verdict"] == "red"
Hb = rows("olmo3-7b-base", "geo").mean(1); base = A.Ck(Hb)
def l0_overlap(arr, tag):
    d = arr.mean(1); ndup = int(len(d) - len(np.unique(d.round(6), axis=0)))
    try:
        c0 = A.Ck(d); G.check_connected(c0.lam)
        ov = M.eigenspace_overlap(base.U[:, 1:m_geo + 1], c0.U[:, 1:m_geo + 1]); return dict(O=ov["O"], duplicate_rows=ndup, m=m_geo)
    except Exception as e:
        return dict(error=repr(e), duplicate_rows=ndup)
a0 = np.load("runs/olmo3-7b-base/acts13_L0.npz"); sel0 = np.where(a0["meta"][:, 0] == 0)[0]
f4["b_layer0_overlap"] = dict(
    last_token=l0_overlap(rows("olmo3-7b-base", "geo", layer=0), "last"),
    mean_name_tokens=l0_overlap(a0["mean_L0"][sel0].reshape(500, K, -1).astype(np.float64), "mean"),
    note="layer 0 = embedding output. The preregistered last-token vector is the embedding of the final name subword, identical across templates and shared by entities with the same last subword (duplicate rows make the self-tuning graph singular); the mean over name-token embeddings is the fallback representation (deviation, labelled)")
ok = [v["O"] for v in (f4["b_layer0_overlap"]["last_token"], f4["b_layer0_overlap"]["mean_name_tokens"]) if "O" in v]
f4["b_fires"] = bool(any(o > 0.8 for o in ok)) if ok else None
names = [g["name"] for g in geo]; zipf = np.array([zipf_frequency(x, "en") for x in names]); ntok = np.array([float(g["n_tok_space"]) for g in geo])
u2 = base.U[:, 1]
rz, pz = stats.spearmanr(u2, zipf); rt, pt_ = stats.spearmanr(u2, ntok)
ex = {f"u{j+1}": dict(rho_zipf=float(stats.spearmanr(base.U[:, j], zipf)[0]), rho_ntok=float(stats.spearmanr(base.U[:, j], ntok)[0])) for j in range(1, m_geo + 1)}
f4["c_spearman_u2"] = dict(rho_zipf=float(rz), p_zipf=float(pz), rho_token_count=float(rt), p_token_count=float(pt_), low_band_all_modes=ex, zipf_note="wordfreq zipf_frequency(name,'en'), 0 for OOV; token count = n_tok_space (frozen CSV)")
f4["c_fires"] = bool(max(abs(rz), abs(rt)) > 0.6)
f4["F4_fires"] = bool(f4["a_fires"] or f4["b_fires"] or f4["c_fires"])
f4["wall_s"] = round(time.time() - t0)
json.dump(f4, open("data/step13_f4_7b.json", "w"), indent=1); print("F4", json.dumps({k: f4[k] for k in ("a_fires", "b_fires", "c_fires", "F4_fires")}), "O", f4["b_layer0_overlap"], "rho", rz, rt, flush=True)

# ---------------- P2 cost ----------------
ref = A.load_geo("olmo2-1b-base", None, layer=12)
def r2_1b(name):
    arr = A.load_geo(name, ref[1], layer=12)[0].mean(1); return ridge_oof(arr, Ygeo)
P1 = {n: r2_1b(n) for n in ("olmo2-1b-base", "ing1", "ing2", "ing3", "inst_step200", "inst_step800", "inst_step1600", "inst_step2600")}
def eps_pairs(pairs):
    out = {}
    for i, (a, b) in enumerate(pairs):
        d = delta_boot(Ygeo, P1[a], P1[b], 7 + i); out[f"{a}|{b}"] = dict(delta=d["delta"], abs_boot_p95=float(np.quantile(np.abs(d["boot"]), .95)))
    return out
epsA = eps_pairs([("ing1", "ing2"), ("ing1", "ing3"), ("ing2", "ing3")])
epsN = eps_pairs([("inst_step200", "inst_step800"), ("inst_step800", "inst_step1600"), ("inst_step1600", "inst_step2600")])
eps_A, eps_N = max(v["abs_boot_p95"] for v in epsA.values()), max(v["abs_boot_p95"] for v in epsN.values()); eps = max(eps_A, eps_N)
cost = {"epsilon": dict(eps=eps, eps_A=eps_A, eps_N6=eps_N, pairs_A=epsA, pairs_N6=epsN, unit="delta R2, 1B OLMo-2 geography ridge at l=12, 95th bootstrap percentile of |delta| per pair, max over pairs",
        label="epsilon borrowed from 1B (no 7B ingredient or RL-step activations exist), T3; also used for C2 (no figures activations exist for any 1B run)")}
Pb, Ps = ridge_oof(Hb, Ygeo), ridge_oof(rows("olmo3-7b-rlzero", "geo").mean(1), Ygeo)
c1 = delta_boot(Ygeo, Pb, Ps, 101)
fb, fs = rows("olmo3-7b-base", "figures").mean(1), rows("olmo3-7b-rlzero", "figures").mean(1)
c2 = delta_boot(Yfig, ridge_oof(fb, Yfig), ridge_oof(fs, Yfig), 102)
h = A.holm([c1["p_drop"], c2["p_drop"]], 2)
for nm, c, hh in (("C1_geography", c1, h[0]), ("C2_death_year", c2, h[1])):
    c.pop("boot"); c.update(holm2_p_drop=hh[0], drops_beyond_band=bool(hh[1] and c["ci95"][1] < -eps), improves_beyond_band=bool(c["ci95"][0] > eps)); cost[nm] = c
any_drop = cost["C1_geography"]["drops_beyond_band"] or cost["C2_death_year"]["drops_beyond_band"]
any_imp = cost["C1_geography"]["improves_beyond_band"] or cost["C2_death_year"]["improves_beyond_band"]
cost["F1_condition"] = "RL-Zero red, Holm p_red 0.008, stability 1.0 (step 11)"
cost["P2_verdict"] = "FALSIFIED (no cost beyond band, or an improvement beyond band)" if (not any_drop or any_imp) else "SUPPORTED (conditional on epsilon borrowed from 1B, T3)"
cost["rule"] = "drop beyond band := Holm(2) one-sided p_drop <= 0.05 and CI95 upper < -eps; improvement := CI95 lower > +eps; falsified if no drop or any improvement (PREREG line 257)"
cost["wall_s"] = round(time.time() - t0)
json.dump(cost, open("data/step13_cost_7b.json", "w"), indent=1)
print("eps", eps, {k: (round(cost[k]["delta"], 4), [round(x, 4) for x in cost[k]["ci95"]], cost[k]["holm2_p_drop"]) for k in ("C1_geography", "C2_death_year")}, cost["P2_verdict"], flush=True)
