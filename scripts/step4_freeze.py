"""Step 4: base-only freezes (l* by probe R2, m by gap rule, r by 90% PCs, outlier check, layer-0 overlap, frequency
correlation). Reads runs/olmo2-1b-base/acts.npz; writes runs/olmo2-1b-base/freezes.json and prints it.
Usage: uv run python scripts/step4_freeze.py"""
import csv, json, sys
import numpy as np
from scipy.stats import spearmanr
from wordfreq import zipf_frequency
from rtp import graph as G, measures as M

A = np.load("runs/olmo2-1b-base/acts.npz")
meta = A["meta"]
K, LAYERS, L = 8, [0, 4, 8, 12], 16
geo = list(csv.DictReader(open("data/entities/geo_v1.csv", encoding="utf-8")))
n = len(geo)
sel = meta[:, 0] == 0
assert sel.sum() == n * K and (meta[sel][:, 1] == np.repeat(np.arange(n), K)).all()


def vecs(l, kind="last"):
    return A[f"{kind}_L{l}"][sel].reshape(n, K, -1)


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


out = {"probe_r2": {}}
for l in LAYERS:
    out["probe_r2"][l] = ridge_r2(vecs(l).mean(1), Y)
cand = {l: out["probe_r2"][l] for l in (4, 8, 12)}
best = max(cand.values())
ties = [l for l, v in cand.items() if abs(v - best) < 1e-12]
lstar = 8 if 8 in ties else ties[0]
out["l_star"] = lstar
out["probe_r2_below_0.3_everywhere"] = bool(all(v < 0.3 for v in out["probe_r2"].values()))
H = vecs(lstar).mean(1); Hc = H - H.mean(0)
Lb = G.laplacian(Hc, 7); lam, U = G.eigh_sorted(Lb)
G.check_connected(lam)
out["lambda_2"] = float(lam[1]); out["spectrum_2_22"] = [round(float(x), 5) for x in lam[1:22]]
# m: gap rule averaged over the bootstrap (template multiset x 80% entity subsample)
rng = np.random.default_rng(20261104)
spectra = []
Hk = vecs(lstar)
for _ in range(100):
    t = rng.integers(0, K, K); sub = np.sort(rng.choice(n, int(0.8 * n), replace=False))
    spectra.append(np.linalg.eigvalsh(G.laplacian(Hk[sub][:, t].mean(1), 7)))
m = G.gap_rule_m(spectra); out["m"] = m; out["m_point_estimate"] = G.gap_rule_m(lam)
out["m_boundary"] = m in (4, 20)
out["r"] = M.r_from_base(Hc)
sv = np.linalg.svd(Hc, compute_uv=False); out["r_uncapped_90pct"] = int(np.searchsorted(np.cumsum(sv**2) / (sv**2).sum(), 0.9) + 1)
# outlier check: top-3 native coordinates' share of centred variance
var = Hc.var(0); out["top3_coord_variance_share"] = float(np.sort(var)[-3:].sum() / var.sum()); out["top3_coords"] = [int(i) for i in np.argsort(var)[-3:]]
out["outlier_arm_needed"] = out["top3_coord_variance_share"] > 0.5
# layer-0 overlap: low band of the layer-0 graph (same m) vs l* low band
# layer-0 last-token vectors are raw embeddings of the final subword, so entities sharing a final subword coincide and the
# self-tuning kernel is undefined (zero k-th neighbour distance); the layer-0 control therefore uses the mean over name tokens.
H0l = vecs(0).mean(1); out["layer0_last_token_distinct_vectors"] = int(len({tuple(r) for r in np.round(H0l, 6)}))
H0 = vecs(0, "mean").mean(1); lam0, U0 = G.eigh_sorted(G.laplacian(H0 - H0.mean(0), 7))
ov = M.eigenspace_overlap(U[:, 1:m + 1], U0[:, 1:m + 1]); out["layer0_overlap_O"] = ov["O"]; out["layer0_theta_max"] = ov["theta_max"]
out["layer0_mean_name_distinct_vectors"] = int(len({tuple(r) for r in np.round(H0, 6)}))
# frequency / length correlation of u_2
zipf = np.array([zipf_frequency(g["name"], "en") for g in geo]); ntok = np.array([float(g["n_tok_space"]) for g in geo])
u2 = U[:, 1]
out["spearman_u2_zipf"] = float(spearmanr(u2, zipf)[0]); out["spearman_u2_ntok"] = float(spearmanr(u2, ntok)[0])
out["spearman_u2_logpop"] = float(spearmanr(u2, np.log([float(g["population"]) for g in geo]))[0])
out["F4_flags"] = {
    "layer0_overlap_gt_0.8": bool(out["layer0_overlap_O"] > 0.8), "freq_or_ntok_rho_gt_0.6": bool(max(abs(out["spearman_u2_zipf"]), abs(out["spearman_u2_ntok"])) > 0.6)}
json.dump(out, open("runs/olmo2-1b-base/freezes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
if "--write-config" in sys.argv:
    s = open("configs/pilot.toml").read()
    s = s[:s.index("[frozen]")] + f"""[frozen]  # step 4, base-only (OLMo-2-0425-1B main a1847dff35000b4271fa70afc5db10fd29fedbdf), frozen before any other checkpoint
m = {m}
r = {out['r']}
l_star = {lstar}
probe_r2 = {{ L0 = {out['probe_r2'][0]:.4f}, L4 = {out['probe_r2'][4]:.4f}, L8 = {out['probe_r2'][8]:.4f}, L12 = {out['probe_r2'][12]:.4f} }}
layer0_overlap_O = {out['layer0_overlap_O']:.4f}
spearman_u2_zipf = {out['spearman_u2_zipf']:.4f}
spearman_u2_ntok = {out['spearman_u2_ntok']:.4f}
top3_coord_variance_share = {out['top3_coord_variance_share']:.4f}
"""
    open("configs/pilot.toml", "w").write(s)
