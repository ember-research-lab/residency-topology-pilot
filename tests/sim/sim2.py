"""Sizing simulation for the residency measurement pipeline (stand-in only; T1 on the synthetic, T3 for transfer to LLMs)."""
import numpy as np, sys, json
rng = np.random.default_rng(20261104)
n, D, K, k = 500, 512, 8, 10
m = int(sys.argv[2]) if len(sys.argv) > 2 else 10

def sphere(n):
    lat = np.arcsin(rng.uniform(-1, 1, n)); lon = rng.uniform(-np.pi, np.pi, n)
    return np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], lat
P, LAT = sphere(n)
W = rng.standard_normal((3, 256)) * 2.0; b = rng.uniform(0, 2*np.pi, 256)
A = rng.standard_normal((256, D)) / np.sqrt(256)
def signal(P):
    S = np.cos(P @ W + b) @ A
    S -= S.mean(0); return S / np.sqrt((S**2).sum(1).mean())   # rms row norm 1
S0 = signal(P)

def reps(S, sigma, T=None):
    """per-template reps (K,n,D); noise rms row norm = sigma"""
    return S[None] + sigma/np.sqrt(D) * rng.standard_normal((K, n, D))

def knn_graph(X, k):
    X = X - X.mean(0); X = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ X.T; np.fill_diagonal(C, -np.inf)
    idx = np.argpartition(-C, k, axis=1)[:, :k]
    Wg = np.zeros_like(C); r = np.repeat(np.arange(len(X)), k); Wg[r, idx.ravel()] = 1
    return np.maximum(Wg, Wg.T), idx
def lowmodes(X, k=k, m=m):
    Wg, idx = knn_graph(X, k); d = Wg.sum(1); Dm = 1/np.sqrt(d)
    L = np.eye(len(X)) - Dm[:, None]*Wg*Dm[None]
    lam, U = np.linalg.eigh(L)
    return U[:, 1:m+1], lam, idx
def overlap(Ua, Ub):
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return (s**2).mean(), np.degrees(np.arccos(np.clip(s.min(), -1, 1)))
def nbr(ia, ib):
    return np.mean([len(set(a) & set(b_)) for a, b_ in zip(ia, ib)]) / ia.shape[1]
def procrustes(Xa, Xb):
    Xa = Xa - Xa.mean(0); Xb = Xb - Xb.mean(0)
    Xa /= np.linalg.norm(Xa); Xb /= np.linalg.norm(Xb)
    ev = np.linalg.eigvals((Xa @ Xa.T) @ (Xb @ Xb.T)).real
    return 1 - np.sqrt(np.clip(ev, 0, None)).sum()          # 1 - ||Xa^T Xb||_*  in [0,1]
def cka(Xa, Xb):
    Xa = Xa - Xa.mean(0); Xb = Xb - Xb.mean(0); Ga, Gb = Xa@Xa.T, Xb@Xb.T
    return (Ga*Gb).sum()/np.linalg.norm(Ga)/np.linalg.norm(Gb)
def measures(Xa, Xb):
    Ua, _, ia = lowmodes(Xa); Ub, _, ib = lowmodes(Xb)
    O, th = overlap(Ua, Ub)
    return dict(d_O=1-O, theta=th, d_NP=1-nbr(ia, ib), R_P=procrustes(Xa, Xb), d_CKA=1-cka(Xa, Xb))

out = {}
_, lam0, _ = lowmodes(S0, m=20); out['base_lambda_2_21'] = [round(float(x),4) for x in lam0[1:21]]
# --- nulls ---
G1, G2 = rng.standard_normal((n, D)), rng.standard_normal((n, D))
out['null_indep_gauss'] = measures(G1, G2)
out['null_indep_structured'] = measures(S0, signal(sphere(n)[0]))   # two unrelated sphere clouds
out['rescale_x7'] = measures(S0, 7*S0)
Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
out['orthogonal_rotation'] = measures(S0, S0 @ Q)
out['chance_O'] = m/n; out['chance_NP'] = k/(n-1)

# --- odd readout witness: frozen base probe, reflection along it ---
def ridge(X, y, lam=1.0):
    X = X - X.mean(0); return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@(y-y.mean()))
w = ridge(S0, LAT); w_hat = w/np.linalg.norm(w)
R = np.eye(D) - 2*np.outer(w_hat, w_hat)
Xs_ref = S0 @ R
def oddS(Xb, Xs): return np.corrcoef((Xs - Xs.mean(0)) @ w, LAT)[0, 1]
out['odd_witness'] = dict(S_base=oddS(S0, S0), S_reflected=oddS(S0, Xs_ref), even_on_reflection=measures(S0, Xs_ref))

# --- epsilon_C and power ---
def boot_eps_and_power(sigma, alphas, B=150, frac=0.8, kind='topo'):
    Tb = reps(S0, sigma); Tb2 = reps(S0, sigma)       # base templates; Tb2 unused except self-check
    perm = rng.permutation(n)
    res = {}
    # self-noise: base under two independent template resamples, same entity subsample
    dself = []
    for _ in range(B):
        sub = rng.choice(n, int(frac*n), replace=False)
        t1, t2 = rng.integers(0, K, K), rng.integers(0, K, K)
        mm = measures(Tb[t1][:, sub].mean(0), Tb[t2][:, sub].mean(0)); dself.append(mm)
    eps = {key: float(np.quantile([d[key] for d in dself], 0.95)) for key in ('d_O', 'd_NP', 'R_P')}
    res['eps_C'] = eps
    for a in alphas:
        if kind == 'topo':     # fraction a of signal replaced by a row-permuted copy: destroys neighbourhoods
            Ss = np.sqrt(1-a**2)*S0 + a*S0[perm]
        else:                  # metric-only: anisotropic rescale of the representation axes
            sc = np.exp(a*rng.standard_normal(D)); Ss = S0*sc; Ss /= np.sqrt((Ss**2).sum(1).mean())
        Ts = Ss[None] + (Tb - S0[None])                # same template noise draw = paired (stage shares prompts)
        Ts = Ts + 0.25*sigma/np.sqrt(D)*rng.standard_normal(Ts.shape)  # extra stage-specific noise
        ds = []
        for _ in range(B):
            sub = rng.choice(n, int(frac*n), replace=False); t = rng.integers(0, K, K)
            ds.append(measures(Tb[t][:, sub].mean(0), Ts[t][:, sub].mean(0)))
        row = {}
        for key in ('d_O', 'd_NP', 'R_P', 'theta', 'd_CKA'):
            v = np.array([d[key] for d in ds]); row[key] = [float(np.quantile(v, .05)), float(np.median(v)), float(np.quantile(v, .95))]
        row['detect_O'] = row['d_O'][0] > eps['d_O']; row['equiv_O'] = row['d_O'][2] < eps['d_O']
        row['detect_RP'] = row['R_P'][0] > eps['R_P']; row['detect_NP'] = row['d_NP'][0] > eps['d_NP']
        if a == 0.0: res['paired_floor_95'] = {kk: row[kk][2] for kk in ('d_O','d_NP','R_P','d_CKA')}
        if 'paired_floor_95' in res:
            row['detect_vs_paired_floor'] = {kk: row[kk][0] > res['paired_floor_95'][kk] for kk in ('d_O','d_NP','R_P','d_CKA')}
        res[f'{kind}_a={a}'] = row
    return res
sig = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
out[f'sigma={sig}'] = boot_eps_and_power(sig, [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
out[f'sigma={sig}_metric'] = boot_eps_and_power(sig, [0.0, 0.1, 0.3, 0.6], kind='metric')
print(json.dumps(out, indent=1, default=float))
