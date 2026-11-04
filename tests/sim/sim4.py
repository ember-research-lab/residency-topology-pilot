"""N4' = linear-part-preserving permuted residual. Reuses sim3 objects."""
import numpy as np, json, runpy, sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(sys.argv[1])
H, n, rng, stats, matched, types, f_low, graph_L, d_O = (g[k] for k in ('H','n','rng','stats','matched','types','f_low','graph_L','d_O'))
Uh_, sv, _ = np.linalg.svd(H, full_matrices=False)
r = int(min(np.searchsorted(np.cumsum(sv**2)/np.sum(sv**2), 0.90)+1, n//10))
Pr = Uh_[:, :r]                                    # top-r PC scores basis of H (n x r)
def split(Dsp):
    Dl = Pr@(Pr.T@Dsp); return Dl, Dsp-Dl          # linear-in-H part (via top-r PCs) and residual
out = dict(r=r)
NP = 150
for name, gen in types.items():
    for rho in (0.05, 0.2):
        Dsp = matched(gen(), rho); Dl, R = split(Dsp)
        obs_dO = d_O(graph_L(H+Dsp)); obs_f = f_low(R)
        nd, nf = [], []
        for _ in range(NP):
            p = rng.permutation(n); Rp = R[p]; Rp = Rp - Pr@(Pr.T@Rp)   # keep residual orthogonal to linear part
            nd.append(d_O(graph_L(H+Dl+Rp))); nf.append(f_low(Rp))
        nd, nf = np.array(nd), np.array(nf)
        out[f'{name} rho={rho}'] = dict(lin_share=round(float((Dl**2).sum()/(Dsp**2).sum()),3),
            d_O=dict(obs=float(obs_dO), null_med=float(np.median(nd)), pct_le=round(float((1+(nd<=obs_dO).sum())/(1+NP)),3)),
            f_low_resid=dict(obs=float(obs_f), null_med=float(np.median(nf)), pct_le=round(float((1+(nf<=obs_f).sum())/(1+NP)),3)))
print(json.dumps(out, indent=1))
