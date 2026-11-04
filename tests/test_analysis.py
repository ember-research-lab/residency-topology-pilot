"""Fast Gram-based analysis path equals the reference measures (1e-8) and still detects injected colour."""
import numpy as np
from rtp import analysis as A, graph as G, measures as M
from standin import sim3_base, sim3_graph


def test_fast_path_matches_reference():
    H = sim3_base()["H"]; rng = np.random.default_rng(1)
    Hs = H + 0.3 * np.linalg.norm(H) / np.sqrt(H.size) * rng.standard_normal(H.shape)
    a, b = A.Ck(H), A.Ck(Hs)
    assert np.abs(a.L - G.laplacian(H)).max() < 1e-12
    ref_o = M.eigenspace_overlap(a.U[:, 1:5], b.U[:, 1:5])["d_O"]
    t = A.triplet(a, b, 4)
    assert abs(t["d_O"] - ref_o) < 1e-12
    assert abs(t["R_P"] - M.procrustes_residual(H, Hs)) < 1e-8
    idx_b, _ = G.cosine_knn(H); idx_s, _ = G.cosine_knn(Hs)
    assert abs(t["d_NP"] - M.neighbourhood_preservation(idx_b, idx_s)["d_NP"]) < 1e-12
    assert abs(A.cka(a, b) - M.cka_linear(H, Hs)) < 1e-10
    assert abs(a.pr_cov() - M.pr_cov(H)) < 1e-8


def test_stage_vs_base_detects_red_and_holm():
    b = sim3_base(); H = b["H"]; Lb, lam, U, m = sim3_graph(H); rng = np.random.default_rng(2)
    base = A.Ck(H)
    Dred = H @ rng.standard_normal((H.shape[1], H.shape[1]))
    Dred = 0.05 * np.linalg.norm(H) * Dred / np.linalg.norm(Dred)
    res = A.stage_vs_base(base, H + Dred, m, 22, 499, rng)
    assert res["p_red"] <= 0.004
    h = A.holm([res["p_blue"], res["p_red"]], 8)
    assert h[1][1] is True and A.verdict(False, True, res["reff_R"]) == "red"
    assert A.verdict(False, False, 5.0) == "UNDERPOWERED" and A.verdict(False, False, 30.0) == "not red"
