"""Null/anchor values of the 'imitation activates, reward prunes' measures + odd readout t_k validation. T1 on stand-in."""
import numpy as np, json, runpy, sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(sys.argv[1])
H, n, S, lat, m, U, Lb = g['H'], g['n'], g['S'], g['lat'], g['m'], g['U'], g['Lb']
rng = np.random.default_rng(5); k = 10
def pr(ev): ev = np.clip(ev, 0, None); return ev.sum()**2/(ev**2).sum()
def knn(X):
    X = X - X.mean(0); X = X/np.linalg.norm(X, axis=1, keepdims=True); C = X@X.T; np.fill_diagonal(C, -np.inf)
    idx = np.argpartition(-C, k, 1)[:, :k]; A = np.zeros((len(X),)*2); A[np.repeat(np.arange(len(X)), k), idx.ravel()] = 1
    return np.maximum(A, A.T)
def comb(A):
    deg = A.sum(1); tri = np.diag(A@A@A)/2; cc = np.where(deg > 1, tri/(deg*(deg-1)/2), 0).mean()
    p = np.bincount(deg.astype(int))/len(deg); p = p[p > 0]; return dict(clust=float(cc), deg_entropy_bits=float(-(p*np.log2(p)).sum()), mean_deg=float(deg.mean()))
def jacc(A, B): e1, e2 = np.triu(A, 1) > 0, np.triu(B, 1) > 0; return float((e1 & e2).sum()/(e1 | e2).sum())
def lap(A):
    d = A.sum(1); return np.eye(len(A)) - A/np.sqrt(np.outer(d, d))
def summary(X, A_ref=None):
    Xc = X - X.mean(0); A = knn(X); lam, V = np.linalg.eigh(lap(A))
    e = ((V.T@Xc)**2).sum(1)
    out = dict(PR_cov=float(pr(np.linalg.eigvalsh(Xc.T@Xc))), PR_lap=float(pr(lam[1:])),
               lowmode_energy=float(e[1:m+1].sum()/e[1:].sum()), **comb(A))
    if A_ref is not None: out['jaccard_to_ref'] = jacc(A, A_ref)
    return out
Ab = knn(H)
res = dict(base_stand_in=summary(H), base_rescaled_x5=summary(5*H, Ab),
           indep_gauss=summary(rng.standard_normal(H.shape), Ab), chance_jaccard=float((k/(n-1))/(2-k/(n-1))))
# PR of n x D white Gaussian (Marchenko-Pastur reference)
# odd readout t_k = u_k^T D a_k, a_k = H^T u_k / ||.||, u_k centred; N4 expectation and witness
Uc = U[:, 1:m+1] - U[:, 1:m+1].mean(0)
def tk(Dsp):
    out = []
    for j in range(m):
        a = H.T@Uc[:, j]; a /= np.linalg.norm(a); out.append(float(Uc[:, j]@(Dsp@a)))
    return np.array(out)
Dw = rng.standard_normal(H.shape)*0.1*np.linalg.norm(H)/np.sqrt(H.size) + 0.3*H.mean(0)   # white + common shift
null = np.array([tk(Dw[rng.permutation(n)]) for _ in range(999)])
res['tk_null_mean_over_sd'] = [round(float(x), 3) for x in null.mean(0)/null.std(0)]
a0 = H.T@Uc[:, 0]; a0 /= np.linalg.norm(a0)
for sgn in (+1, -1):
    Dsp = sgn*0.05*np.outer(Uc[:, 0], a0)*np.linalg.norm(H)
    res[f'witness_mode1_sign{sgn:+d}'] = dict(t_1=float(tk(Dsp)[0]), t_others_maxabs=float(np.abs(tk(Dsp)[1:]).max()),
                                             **{kk: v for kk, v in summary(H+Dsp, Ab).items() if kk in ('PR_cov', 'lowmode_energy', 'jaccard_to_ref')})
# off-span witness: v orthogonal to entity span -> t_k blind
Q, _ = np.linalg.qr(H.T); v = rng.standard_normal(H.shape[1]); v -= Q@(Q.T@v); v /= np.linalg.norm(v)
res['offspan_tk_maxabs'] = float(np.abs(tk(np.outer(lat-lat.mean(), v))).max())
print(json.dumps(res, indent=1))
