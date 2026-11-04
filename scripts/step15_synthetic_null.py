"""Step 15 (EXPLORATORY, non-voting): synthetic positive control for the smooth null. Analysis only on saved 7B base activations
(geography, C1 nouns, C2 nonce at l*=16). D_syn = D_smooth (random 2-layer tanh MLP of H_b) + D_rewire (U_low M), ||D||/||H_b|| = 0.0104.
Readouts: (i) raw colour vs N4' (P=499); (ii) colour of the cross-fitted KRR residual (step-14 smoother) vs N4' (P=499);
(iii) Freedman-Lane-style null: permute rows of D (P=199), apply the SAME fixed cross-fit smoother operator (lambda per fold fixed from the
real-D fit, not re-selected per permutation: deviation, for cost), colour the residual's f_low against that null.
-> data/step15_synthetic_null.json. Run under the cgroup cap."""
import json, time
import numpy as np
from rtp import analysis as A, measures as M, cap
cap.require_cap()
t0 = time.time(); K = 8; R_ = 50; MSYN = 5; RATIO = 0.0104; SEEDS = 20; P_STD, P_FL = 499, 199
def rows(name, code, layer=16, f="acts.npz"):
    a = np.load(f"runs/{name}/{f}"); meta = a["meta"]; sel = np.where(meta[:, 0] == code)[0]; n = len(sel) // K
    return a[f"last_L{layer}"][sel].reshape(n, K, -1).astype(np.float64)
def loo_lam(Ktr, Y, grid=np.logspace(-6, 2, 17)):
    """lambda by closed-form leave-one-out on the eigendecomposition (fast; replaces the step-14 inner 5-fold CV inside the permutation refits)."""
    w, V = np.linalg.eigh(Ktr); w = np.clip(w, 0, None); C = V.T @ (Y - Y.mean(0)); Gc = C @ C.T; best = None
    for lam in grid:
        g = lam / (w + lam); h = (V ** 2) @ (w / (w + lam)); Vg = V * g
        sse = (((Vg @ Gc) * Vg).sum(1) / (1 - h) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, lam)
    return best[1], w, V, C
FOLDS = None
def crossfit(Kfull, Y, lams=None):
    """Cross-fit smoother of Y (n x k) on the kernel; returns (prediction, lams). lams=None -> select per fold by LOO."""
    n = len(Kfull); pred = np.zeros_like(Y); out = []
    for k, part in enumerate(FOLDS):
        tr = np.setdiff1d(np.arange(n), part); mu = Y[tr].mean(0)
        if lams is None:
            lam, w, V, C = loo_lam(Kfull[np.ix_(tr, tr)], Y[tr])
        else:
            lam = lams[k]; w, V = np.linalg.eigh(Kfull[np.ix_(tr, tr)]); w = np.clip(w, 0, None); C = V.T @ (Y[tr] - mu)
        out.append(lam); pred[part] = Kfull[np.ix_(part, tr)] @ (V @ (C / (w + lam)[:, None])) + mu
    return pred, out
def operator(Kfull, lams):
    n = len(Kfull); Mop = np.zeros((n, n))
    for k, part in enumerate(FOLDS):
        tr = np.setdiff1d(np.arange(n), part); Z = Kfull[np.ix_(part, tr)] @ np.linalg.inv(Kfull[np.ix_(tr, tr)] + lams[k] * np.eye(len(tr)))
        Z = Z - Z.sum(1, keepdims=True) / len(tr); pred = np.zeros((len(part), n)); pred[:, tr] = Z + 1.0 / len(tr); Mop[part] = np.eye(n)[part] - pred
    return Mop
def verdict(pb, pr): return "blue" if pb <= 0.05 else ("red" if pr <= 0.05 else "white")
def run_cell(Hb, base, Kfull, m, D, rng, Pmat):
    U_, S_, Vt_ = np.linalg.svd(D, full_matrices=False); W = Vt_.T; Dk = U_ * S_      # D = Dk W^T; all smoothing/colouring is invariant to the feature rotation
    def fl(X):
        Xc = X - X.mean(0); R = Xc - Pmat @ (Pmat.T @ Xc); return M.f_low(R, base.U, m)
    raw = A.stage_vs_base(base, Hb + D, m, R_, P_STD, rng)
    fit, lams = crossfit(Kfull, Dk); E = Dk - fit; rs = A.stage_vs_base(base, Hb + E @ W.T, m, R_, P_STD, rng); obs = fl(E)
    Mop = operator(Kfull, lams)
    null_old = np.array([fl(Mop @ Dk[rng.permutation(len(Dk))]) for _ in range(P_FL)])                  # iii-old: row-permuted D, fixed smoother (bias measurement)
    null_new = []
    for _ in range(P_FL):                                                                               # iii: D* = fhat(H_b) + E_perm, full refit incl. lambda choice
        Ds = fit + E[rng.permutation(len(E))]; f2, _ = crossfit(Kfull, Ds); null_new.append(fl(Ds - f2))
    null_new = np.array(null_new)
    cell = lambda o: dict(p_blue=o["p_blue"], p_red=o["p_red"], verdict=verdict(o["p_blue"], o["p_red"]), pct=1 - o["p_red"], enrich=o["f_low"] / o["f_low_null_median"])
    def pn(null):
        pb, pr = (1 + (null <= obs).sum()) / (P_FL + 1), (1 + (null >= obs).sum()) / (P_FL + 1)
        return dict(p_blue=float(pb), p_red=float(pr), verdict=verdict(pb, pr), pct=float(1 - pr), enrich=float(obs / np.median(null)), obs=float(obs))
    return dict(raw=cell(raw), resid=cell(rs), iii_old=pn(null_old), iii=pn(null_new))
data = {}
for sn, code, f in (("geo", 0, "acts.npz"), ("C1_nouns", 1, "acts.npz"), ("C2_nonce", 2, "acts13.npz")):
    ab = rows("olmo3-7b-base", code, 16, f); Hb = ab.mean(1); base = A.Ck(Hb); n, d = Hb.shape
    Hc = Hb - Hb.mean(0); g_ = Hc @ Hc.T; dg_ = np.diag(g_); d2 = np.maximum(dg_[:, None] + dg_[None] - 2 * g_, 0); sig = np.sqrt(np.median(d2[np.triu_indices(n, 1)]))
    data[sn] = dict(ab=ab, Hb=Hb, base=base, Kfull=np.exp(-d2 / (2 * sig ** 2)), Hc=Hc, Pmat=base.V[:, ::-1][:, :R_], rs=rows("olmo3-7b-rlzero", code, 16, f))
out = {"design": dict(m_synthetic=MSYN, ratio=RATIO, seeds=SEEDS, P_raw_resid=P_STD, P_perm=P_FL, note="synthetic cells use m=5 for injection and colouring; real-data (iii) uses each set's frozen m (geo 4, C1 5, C2 4); lambda fixed per fold in the permutation null")}
def synth(sn, s, seed):
    dd = data[sn]; rng = np.random.default_rng(15000 + seed); Hc = dd["Hc"]; n, d = Hc.shape
    Hn = Hc / np.sqrt((Hc ** 2).mean()); Ds = np.tanh(Hn @ rng.normal(0, 1 / np.sqrt(d), (d, 256))) @ rng.normal(0, 1 / np.sqrt(256), (256, d)); Ds -= Ds.mean(0)
    Dr = dd["base"].U[:, 1:MSYN + 1] @ rng.normal(size=(MSYN, d)); Dr -= Dr.mean(0)
    D = np.sqrt(1 - s) * Ds / np.linalg.norm(Ds) + (np.sqrt(s) * Dr / np.linalg.norm(Dr) if s > 0 else 0)
    D = D * RATIO * np.linalg.norm(dd["base"].H) / np.linalg.norm(D); return D - D.mean(0), rng
import sys
GROUP = int(sys.argv[1]); GROUPS = [[("geo", 0.0), ("geo", 0.05)], [("geo", 0.10), ("geo", 0.20)], [("C1_nouns", 0.0), ("C2_nonce", 0.0)]]
FOLDS = np.array_split(np.random.default_rng(6).permutation(500), 5)
out["cells"] = {}
for sn, s in GROUPS[GROUP]:
    res = []
    for seed in range(SEEDS):
        D, rng = synth(sn, s, seed); dd = data[sn]; res.append(run_cell(dd["Hb"], dd["base"], dd["Kfull"], MSYN, D, rng, dd["Pmat"]))
        if seed % 5 == 0: print(sn, s, "seed", seed, {k: res[-1][k]["verdict"] for k in ("raw", "resid", "iii_old", "iii")}, round(time.time() - t0), flush=True)
    summ = {}
    for ro in ("raw", "resid", "iii_old", "iii"):
        v = [r[ro]["verdict"] for r in res]; summ[ro] = dict(red=v.count("red") / SEEDS, blue=v.count("blue") / SEEDS, white=v.count("white") / SEEDS, median_enrich=float(np.median([r[ro]["enrich"] for r in res])))
    out["cells"][f"{sn}_s{s}"] = dict(summary=summ, per_seed=res)
    json.dump(out, open(f"data/step15_part{GROUP}.json", "w"), indent=1)
# real RL-Zero displacement under readout (iii) (and (i), (ii) for reference)
frm = {"geo": 4, "C1_nouns": 5, "C2_nonce": 4}; out["real_rlzero"] = {}
for sn, mm in (frm.items() if GROUP == 2 else []):
    dd = data[sn]; D = dd["rs"].mean(1) - dd["Hb"]; D = D - D.mean(0)
    out["real_rlzero"][sn] = dict(m=mm, **run_cell(dd["Hb"], dd["base"], dd["Kfull"], mm, D, np.random.default_rng(777), dd["Pmat"]))
    print("real", sn, {k: (out["real_rlzero"][sn][k]["verdict"], round(out["real_rlzero"][sn][k]["pct"], 3)) for k in ("raw", "resid", "iii_old", "iii")}, flush=True)
out["wall_s"] = round(time.time() - t0); json.dump(out, open(f"data/step15_part{GROUP}.json", "w"), indent=1); print("DONE", flush=True)
