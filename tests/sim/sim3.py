"""N4 permuted-displacement null + graph-colour classifier + G8 witness on the synthetic stand-in. T1 on the stand-in only."""
import numpy as np, json
rng = np.random.default_rng(7)
n, Dm, K = 500, 512, 8
lat = np.arcsin(rng.uniform(-1, 1, n)); lon = rng.uniform(-np.pi, np.pi, n)
P = np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
W = rng.standard_normal((3, 256))*2.0; b = rng.uniform(0, 2*np.pi, 256); A = rng.standard_normal((256, Dm))/16
S = np.cos(P@W+b)@A; S -= S.mean(0); S /= np.sqrt((S**2).sum(1).mean())
H = S + 0.5/np.sqrt(Dm)*rng.standard_normal((n, Dm))/np.sqrt(K)   # template-averaged base
H -= H.mean(0)

def graph_L(X, kself=7):
    X = X - X.mean(0); sq = (X**2).sum(1); d2 = np.maximum(sq[:, None]+sq[None]-2*X@X.T, 0)
    sig = np.sqrt(np.sort(d2, 1)[:, kself]); Kk = np.exp(-d2/(sig[:, None]*sig[None])); np.fill_diagonal(Kk, 0)
    q = Kk.sum(1); Kk = Kk/np.outer(q, q)                      # diffusion-maps alpha=1
    dd = Kk.sum(1); Ls = np.eye(len(X)) - Kk/np.sqrt(np.outer(dd, dd))
    return Ls
Lb = graph_L(H); lam, U = np.linalg.eigh(Lb)
rel = [(lam[j+1]-lam[j])/lam[j+1] for j in range(1, 21)]          # gap after mode j (modes 2..j)
m = int(np.argmax(rel[3:20]) + 4)                                  # m in [4,20]: low band = modes 2..m+1? (count m)
m = m  # number of nontrivial low modes
Ul, Uh = U[:, 1:m+1], U[:, m+1:]
def f_low(Dsp):
    E = (U.T @ Dsp)**2; e = E.sum(1); return e[1:m+1].sum()/e[1:].sum()
def d_O(Lp):
    _, V = np.linalg.eigh(Lp); s = np.linalg.svd(Ul.T@V[:, 1:m+1], compute_uv=False); return 1-(s**2).mean()
def coup(Lp):
    E = Lp - Lb; return np.linalg.norm(Ul.T@E@Uh)/np.linalg.norm(E)
def stats(Dsp):
    Lp = graph_L(H+Dsp); return dict(f_low=f_low(Dsp), C=coup(Lp), d_O=d_O(Lp))

def matched(Dsp, rho): Dsp = Dsp - Dsp.mean(0); return Dsp*rho*np.linalg.norm(H)/np.linalg.norm(Dsp)
Z = rng.standard_normal((n, Dm))
types = {
 'red_linear_map(H@A)': lambda: H@rng.standard_normal((Dm, Dm)),
 'white_iid': lambda: rng.standard_normal((n, Dm)),
 'blue_graph_highpass(L^2 Z)': lambda: Lb@(Lb@rng.standard_normal((n, Dm))),
 'topo_rewire(S[perm]-S)': lambda: S[rng.permutation(n)] - S,
}
out = dict(m_gap_rule=m, lam_2_12=[round(float(x), 4) for x in lam[1:12]], white_expect_f_low=m/(n-1))
NP = 150
for name, gen in types.items():
    for rho in (0.05, 0.2):
        Dsp = matched(gen(), rho); obs = stats(Dsp)
        null = [stats(Dsp[rng.permutation(n)]) for _ in range(NP)]
        row = {}
        for k_ in obs:
            v = np.array([x[k_] for x in null])
            row[k_] = dict(obs=round(float(obs[k_]), 5), null_med=round(float(np.median(v)), 5),
                           pct_le=round(float((1+(v <= obs[k_]).sum())/(1+NP)), 4))
        out[f'{name} rho={rho}'] = row
# G8 witness: odd shift +-alpha*c*v with v orthogonal to entity span S; odd readout along anchor v
Q, _ = np.linalg.qr(H.T); v = rng.standard_normal(Dm); v -= Q@(Q.T@v); v /= np.linalg.norm(v)
c = lat - lat.mean()
wit = {}
for sgn in (+1, -1):
    Dsp = sgn*0.1*np.outer(c, v)*np.linalg.norm(H)/np.linalg.norm(np.outer(c, v))
    Lp = graph_L(H+Dsp); s_i = Dsp@v
    wit[sgn] = dict(d_O=float(d_O(Lp)), C=float(coup(Lp)), f_low=float(f_low(Dsp)),
                    odd_corr=float(np.corrcoef(s_i, c)[0, 1]), odd_mean_sign=float(np.sign((s_i*c).mean())))
wit['even_max_abs_diff'] = max(abs(wit[1][k_]-wit[-1][k_]) for k_ in ('d_O', 'C', 'f_low'))
out['G8_witness'] = wit
print(json.dumps(out, indent=1))
