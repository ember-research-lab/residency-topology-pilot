"""Synthetic battery (DESIGN 6.3 step 1). No model. Each test is one 'BROKEN if' condition; none is loosened.

Constructions are reused from tests/sim/*.py via tests/standin.py (same seeds, same draw order).
"""
import numpy as np
import pytest
from scipy import stats

from rtp import graph as G
from rtp import measures as M
from rtp import nulls as N
from standin import n, sim3_base, sim3_graph, sim_sphere_signal


@pytest.fixture(scope="module")
def base():
    b = sim3_base()
    Lb, lam, U, m = sim3_graph(b["H"])
    H = b["H"]
    sv = np.linalg.svd(H, full_matrices=False)
    r = M.r_from_base(H)
    return dict(H=H, S=b["S"], lat=b["lat"], Lb=Lb, lam=lam, U=U, m=m, r=r, P=M.pc_basis(H, r))


def matched(Dsp, rho, H):
    Dsp = Dsp - Dsp.mean(0)
    return Dsp * rho * np.linalg.norm(H) / np.linalg.norm(Dsp)


def test_standin_reproduces_sim3(base):
    assert base["m"] == 4
    assert np.allclose(base["lam"][1:12], [0.0753, 0.1066, 0.1182, 0.1346, 0.1645, 0.1842, 0.1893, 0.2075, 0.2244, 0.2314, 0.2586], atol=5e-5)
    assert base["r"] == 22


def test_white_uniform_percentiles(base):
    """white D -> N4' percentile (p_blue) uniform: KS p > 0.01 over 200 draws."""
    rng = np.random.default_rng(101)
    pcs = []
    for _ in range(200):
        Dsp = matched(rng.standard_normal(base["H"].shape), 0.05, base["H"])
        pcs.append(N.n4prime_f_low(Dsp, base["P"], base["U"], base["m"], 199, rng)["p_blue"])
    ks = stats.kstest(pcs, "uniform")
    print("KS white N4' p_blue: stat=%.4f p=%.4g mean=%.3f" % (ks.statistic, ks.pvalue, np.mean(pcs)))
    assert ks.pvalue > 0.01


def test_injected_blue_and_red(base):
    """blue (L^2 Z) -> p_blue <= 0.002 ; red (linear map H@A, residual) -> p_red <= 0.002, P=999."""
    rng = np.random.default_rng(102)
    H, Lb = base["H"], base["Lb"]
    blue = matched(Lb @ (Lb @ rng.standard_normal(H.shape)), 0.05, H)
    red = matched(H @ rng.standard_normal((H.shape[1], H.shape[1])), 0.05, H)
    rb = N.n4prime_f_low(blue, base["P"], base["U"], base["m"], 999, rng)
    rr = N.n4prime_f_low(red, base["P"], base["U"], base["m"], 999, rng)
    print("blue p_blue=%.4f (f_low %.3g)  red p_red=%.4f (f_low %.3g)" % (rb["p_blue"], rb["obs"], rr["p_red"], rr["obs"]))
    assert rb["p_blue"] <= 0.002
    assert rr["p_red"] <= 0.002


def all_measures(Hb, Hs, k_self=7):
    """Every measure of DESIGN 2.4 on (Hb, Hs), everything rebuilt from the inputs. t_k excluded (checked apart)."""
    Lb = G.laplacian(Hb, k_self); Ls = G.laplacian(Hs, k_self)
    _, Ub = G.eigh_sorted(Lb); _, Us = G.eigh_sorted(Ls)
    m = 4
    Dsp = Hs - Hb
    P = M.pc_basis(Hb, 22)
    _, R, lam_lin = M.split_linear(Dsp, P)
    ib, Ab = G.cosine_knn(Hb); is_, As = G.cosine_knn(Hs)
    out = {}
    out.update(M.eigenspace_overlap(Ub[:, 1:m + 1], Us[:, 1:m + 1]))
    out.update(M.neighbourhood_preservation(ib, is_))
    out["R_P"] = M.procrustes_residual(Hb, Hs)
    out["CKA"] = M.cka_linear(Hb, Hs)
    out["lambda_lin"] = lam_lin
    out["f_low"] = M.f_low(R, Ub, m)
    out["C"] = M.coupling_C(Lb, Ls, Ub, m)
    out["PR_cov_b"], out["PR_cov_s"] = M.pr_cov(Hb), M.pr_cov(Hs)
    out["PR_lap"] = M.pr_lap(G.knn_laplacian(As))
    out["clust"], out["deg_entropy"] = M.clustering_coefficient(As), M.degree_entropy_bits(As)
    out["jaccard"] = M.edge_jaccard(Ab, As)
    out["E_low_own"] = M.e_low(Hs, Us, m)
    out["E_low_base"] = M.e_low(Hs, Ub, m)
    return out


def test_rescale_and_rotation_invariance(base):
    """rescale x5 and a random rotation change every measure by < 1e-12 (joint transform of base and stage)."""
    rng = np.random.default_rng(103)
    Hb = base["H"]
    Hs = Hb + matched(rng.standard_normal(Hb.shape), 0.3, Hb)
    ref = all_measures(Hb, Hs)
    Q, _ = np.linalg.qr(rng.standard_normal((Hb.shape[1],) * 2))
    for name, f in (("x5", lambda X: 5 * X), ("rot", lambda X: X @ Q)):
        alt = all_measures(f(Hb), f(Hs))
        worst = max(alt, key=lambda k: abs(alt[k] - ref[k]))
        print("%s: worst %s %.3g" % (name, worst, abs(alt[worst] - ref[worst])))
        bad = {k: abs(alt[k] - ref[k]) for k in ref if abs(alt[k] - ref[k]) >= 1e-12}
        assert not bad, bad


def test_tk_rescale_covariance_and_rotation(base):
    """t_k is linear in D: rotation leaves it unchanged (<1e-12); a joint x5 rescale multiplies it by 5.
    (DESIGN 2.5 says 'every measure' is scale-invariant; t_k is not - reported in REPORT-steps-0-2.md.)"""
    rng = np.random.default_rng(104)
    H, U, m = base["H"], base["U"], base["m"]
    Dsp = matched(rng.standard_normal(H.shape), 0.1, H)
    t = M.t_k(H, Dsp, U, m)
    Q, _ = np.linalg.qr(rng.standard_normal((H.shape[1],) * 2))
    assert np.abs(M.t_k(H @ Q, Dsp @ Q, U, m) - t).max() < 1e-12
    assert np.abs(M.t_k(5 * H, 5 * Dsp, U, m) - 5 * t).max() < 1e-12


def test_tk_null_witness_offspan(base):
    """t_k mean/sd under N4 < 0.1; witness flips sign; off-span t_k < 1e-10 (sim6 construction)."""
    H, U, m, lat = base["H"], base["U"], base["m"], base["lat"]
    rng = np.random.default_rng(5)
    Uc = U[:, 1:m + 1] - U[:, 1:m + 1].mean(0)
    Dw = rng.standard_normal(H.shape) * 0.1 * np.linalg.norm(H) / np.sqrt(H.size) + 0.3 * H.mean(0)
    null = np.array([M.t_k(H, N.n4_draw(Dw, rng), U, m) for _ in range(999)])
    ratio = null.mean(0) / null.std(0)
    a0 = H.T @ Uc[:, 0]; a0 /= np.linalg.norm(a0)
    tp = M.t_k(H, +0.05 * np.outer(Uc[:, 0], a0) * np.linalg.norm(H), U, m)
    tm = M.t_k(H, -0.05 * np.outer(Uc[:, 0], a0) * np.linalg.norm(H), U, m)
    Q, _ = np.linalg.qr(H.T)
    v = rng.standard_normal(H.shape[1]); v -= Q @ (Q.T @ v); v /= np.linalg.norm(v)
    off = np.abs(M.t_k(H, np.outer(lat - lat.mean(), v), U, m)).max()
    print("t_k null mean/sd", ratio.round(3), "witness", tp[0], tm[0], "others", np.abs(tp[1:]).max(), "offspan", off)
    assert np.abs(ratio).max() < 0.1
    assert tp[0] > 0 > tm[0] and abs(tp[0] + tm[0]) < 1e-9 and abs(tp[0] - 1.135) < 0.005
    assert off < 1e-10


def test_gram_form_f_low_equals_direct(base):
    rng = np.random.default_rng(105)
    H, U, m, P = base["H"], base["U"], base["m"], base["P"]
    Dsp = matched(rng.standard_normal(H.shape), 0.2, H)
    _, R, _ = M.split_linear(Dsp, P)
    assert abs(M.f_low_gram(R @ R.T, U, m) - M.f_low(R, U, m)) < 1e-10
    Gm = R @ R.T
    worst = 0.0
    for _ in range(20):
        perm = rng.permutation(n)
        Rp = R[perm]; Rp = Rp - P @ (P.T @ Rp)
        worst = max(worst, abs(N.f_low_null_gram(Gm, perm, P, U, m) - M.f_low(Rp, U, m)))
    print("gram vs direct worst", worst)
    assert worst < 1e-10


def test_procrustes_matches_direct_nuclear(base):
    rng = np.random.default_rng(106)
    A = base["H"]; B = A + 0.5 * rng.standard_normal(A.shape) * np.linalg.norm(A) / np.sqrt(A.size)
    a, b = A - A.mean(0), B - B.mean(0); a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    direct = 1 - np.linalg.svd(a.T @ b, compute_uv=False).sum()
    assert abs(M.procrustes_residual(A, B) - direct) < 1e-10


def test_null_table_geometry_rows():
    """2.5 rows for d_O, theta_max, d_NP, R_P: sim.py construction (seed 20261104, kNN graph, m=10, k=10)."""
    rng = np.random.default_rng(20261104)
    sphere, signal, S0, _ = sim_sphere_signal(rng)
    def meas(Xa, Xb, m=10):
        ia, Aa = G.cosine_knn(Xa); ib, Ab = G.cosine_knn(Xb)
        _, Ua = np.linalg.eigh(G.knn_laplacian(Aa)); _, Ub = np.linalg.eigh(G.knn_laplacian(Ab))
        o = M.eigenspace_overlap(Ua[:, 1:m + 1], Ub[:, 1:m + 1])
        return dict(d_O=o["d_O"], theta=o["theta_max"], d_NP=M.neighbourhood_preservation(ia, ib)["d_NP"], R_P=M.procrustes_residual(Xa, Xb))
    G1, G2 = rng.standard_normal((n, 512)), rng.standard_normal((n, 512))
    g = meas(G1, G2)
    s = meas(S0, signal(sphere(n)[0]))
    print("null table gauss", g, "structured", s)
    assert round(g["d_O"], 2) == 0.98 and round(g["d_NP"], 2) == 0.98
    assert abs(g["theta"] - 89.8) < 0.05 and abs(g["R_P"] - 0.245) < 0.005
    assert abs(s["R_P"] - 0.818) < 0.005
    assert round(1 - 10 / n, 2) == 0.98 and round(1 - 10 / (n - 1), 2) == 0.98
    for X in (7 * S0, S0 @ np.linalg.qr(rng.standard_normal((512, 512)))[0]):
        z = meas(S0, X)
        assert all(abs(v) < 1e-9 for k_, v in z.items() if k_ != "theta") and z["theta"] < 1e-9, z


def test_null_table_graph_rows(base):
    """2.5 rows PR_cov, PR_lap, clustering, degree entropy, E_low, Jaccard, f_low white (sim6 construction)."""
    H, m, U = base["H"], base["m"], base["U"]
    rng = np.random.default_rng(5)
    def summ(X, A_ref=None):
        Xc = X - X.mean(0)
        _, A = G.cosine_knn(X); Lk = G.knn_laplacian(A); lam, V = np.linalg.eigh(Lk)
        o = dict(PR_cov=M.pr_cov(X), PR_lap=M.pr_lap(Lk), E_low=M.e_low(X, V, m),
                 clust=M.clustering_coefficient(A), deg_entropy=M.degree_entropy_bits(A))
        if A_ref is not None: o["jac"] = M.edge_jaccard(A_ref, A)
        return o
    _, Ab = G.cosine_knn(H)
    base_s = summ(H); resc = summ(5 * H, Ab); gauss = summ(rng.standard_normal(H.shape), Ab)
    print("base", base_s, "\nx5", resc, "\ngauss", gauss)
    r2 = lambda x: round(x, 2)
    assert (r2(base_s["PR_cov"]), r2(gauss["PR_cov"])) == (16.48, 252.02)
    assert (r2(base_s["PR_lap"]), r2(gauss["PR_lap"])) == (459.85, 460.20)
    assert (r2(base_s["clust"]), r2(gauss["clust"])) == (0.56, 0.04)   # table: 0.563 / 0.035
    assert (r2(base_s["deg_entropy"]), r2(gauss["deg_entropy"])) == (2.35, 2.42)
    assert (r2(base_s["E_low"]), r2(gauss["E_low"])) == (0.37, 0.02)
    assert r2(gauss["jac"]) == 0.01 and resc["jac"] == 1.0
    assert r2((10 / (n - 1)) / (2 - 10 / (n - 1))) == 0.01
    assert all(abs(resc[k] - base_s[k]) < 1e-9 for k in base_s)
    # f_low of a white D under plain N4: median = m/(n-1) = 0.0080
    Dw = rng.standard_normal(H.shape)
    med = np.median([M.f_low(N.n4_draw(Dw, rng), U, m) for _ in range(499)])
    assert round(med, 4) == 0.0080 and round(m / (n - 1), 4) == 0.0080, med
