"""Power of the f_low(residual) colour test vs N4' at n=500 (Gram-matrix fast form). T1 on the stand-in."""
import numpy as np, json, runpy, sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(sys.argv[1])
H, n, Lb, U, m = g['H'], g['n'], g['Lb'], g['U'], g['m']
rng = np.random.default_rng(11)
Uh_, sv, _ = np.linalg.svd(H, full_matrices=False)
r = int(min(np.searchsorted(np.cumsum(sv**2)/np.sum(sv**2), 0.90)+1, n//10)); Pr = Uh_[:, :r]
Ulow, u1 = U[:, 1:m+1], U[:, :1]
proj = lambda Y: Y - Pr@(Pr.T@Y)
def f_low_perm(G, perm):
    # Y = (I-PrPr^T) P R ; energies via Gram G = R R^T ; P R = R[perm]  =>  u^T Y = ((I-PrPr^T)u)[inv] ... use w = P^T (I-PrPr^T) u
    inv = np.argsort(perm)
    Wl = proj(Ulow)[inv]; W1 = proj(u1)[inv]; Wp = Pr[inv]
    el = np.einsum('ij,ij->', Wl, G@Wl); e1 = np.einsum('ij,ij->', W1, G@W1)
    tot = np.trace(G) - np.einsum('ij,ij->', Wp, G@Wp)
    return el/(tot-e1)
def unit(X): X = X - X.mean(0); return X/np.linalg.norm(X)
# self-check fast form against direct form
R0 = proj(rng.standard_normal((n, 64))); p0 = rng.permutation(n)
Y = proj(R0[p0]); e = ((U.T@Y)**2).sum(1); direct = e[1:m+1].sum()/e[1:].sum()
out = {'selfcheck_fast_minus_direct': float(f_low_perm(R0@R0.T, p0) - direct), 'r': r, 'm': m}
NP, REPS = 499, 20
for reff in (5, 50):
    for kind in ('blue', 'red'):
        for beta in (0.0, 0.05, 0.1, 0.2):
            det = 0
            for _ in range(REPS):
                M = rng.standard_normal((reff, 512))
                Zw = rng.standard_normal((n, reff))@M
                Zc = (Lb@(Lb@rng.standard_normal((n, reff)))) if kind == 'blue' else (Ulow@rng.standard_normal((m, reff)))
                Dsp = np.sqrt(1-beta)*unit(Zw) + np.sqrt(beta)*unit(Zc@M)
                R = proj(Dsp); G = R@R.T; ident = np.arange(n)
                obs = f_low_perm(G, ident)
                null = np.array([f_low_perm(G, rng.permutation(n)) for _ in range(NP)])
                p = (1+((null <= obs) if kind == 'blue' else (null >= obs)).sum())/(1+NP)
                det += p <= 0.05/9
            out[f'reff={reff} {kind} beta={beta}'] = det/REPS
print(json.dumps(out, indent=1))
