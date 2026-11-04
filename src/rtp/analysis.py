"""analysis - per-draw statistics for base vs stage on the frozen entity graph (DESIGN 3.4, 2.4), Gram-based fast path.

A 'draw' is (template multiset t, entity subset e); everything (graph, P_r, U) is recomputed from the draw with m, r, l*
frozen. The identity draw is the point estimate: step 9 and the bootstrap share this code path.
"""
from __future__ import annotations

import numpy as np

from . import graph as G
from . import measures as M
from . import nulls as N


class Ck:
    """Cached per-checkpoint, per-draw quantities from centred representation H (n x d)."""

    def __init__(self, H: np.ndarray, k_self: int = 7, k_nn: int = 10):
        self.H = H - H.mean(0)
        self.G = self.H @ self.H.T
        dg = np.diag(self.G)
        self.w, self.V = np.linalg.eigh(self.G)
        self.w = np.clip(self.w, 0, None)
        self.L = G.laplacian_from_d2(np.maximum(dg[:, None] + dg[None] - 2 * self.G, 0), k_self)
        self.lam, self.U = G.eigh_sorted(self.L)
        C = self.G / np.sqrt(np.outer(dg, dg))
        np.fill_diagonal(C, -np.inf)
        self.idx = np.argpartition(-C, k_nn, axis=1)[:, :k_nn]
        A = np.zeros_like(C)
        A[np.repeat(np.arange(len(C)), k_nn), self.idx.ravel()] = 1.0
        self.A = np.maximum(A, A.T)
        self.fro2 = float(dg.sum())

    def pr_cov(self) -> float:
        return float(self.w.sum() ** 2 / (self.w**2).sum())

    def e_low(self, U: np.ndarray, m: int) -> float:
        e = np.einsum("ij,ij->j", U, self.G @ U)
        return float(e[1:m + 1].sum() / e[1:].sum())


def procrustes(a: Ck, b: Ck) -> float:
    sa, sb = np.sqrt(a.w / a.fro2), np.sqrt(b.w / b.fro2)
    Mx = (sa[:, None] * (a.V.T @ b.V)) * sb[None]
    return float(1 - np.linalg.svd(Mx, compute_uv=False).sum())


def cka(a: Ck, b: Ck) -> float:
    return float((a.G * b.G).sum() / np.linalg.norm(a.G) / np.linalg.norm(b.G))


def reff(R: np.ndarray) -> float:
    """Participation ratio of the squared singular values of R (same convention as PR_cov)."""
    ev = np.clip(np.linalg.eigvalsh(R @ R.T), 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum())


def triplet(a: Ck, b: Ck, m: int) -> dict:
    o = M.eigenspace_overlap(a.U[:, 1:m + 1], b.U[:, 1:m + 1])
    return dict(d_O=o["d_O"], theta_max=o["theta_max"], d_NP=M.neighbourhood_preservation(a.idx, b.idx)["d_NP"], R_P=procrustes(a, b))


def nonvoting(base: Ck, st: Ck, m: int) -> dict:
    out = triplet(base, st, m)
    out.update(CKA=cka(base, st), PR_cov_b=base.pr_cov(), PR_cov_s=st.pr_cov(), dPR_cov=st.pr_cov() - base.pr_cov(),
               clust_b=M.clustering_coefficient(base.A), clust_s=M.clustering_coefficient(st.A),
               dclust=M.clustering_coefficient(st.A) - M.clustering_coefficient(base.A),
               jaccard=M.edge_jaccard(base.A, st.A), E_low_base_graph_b=base.e_low(base.U, m), E_low_base_graph_s=st.e_low(base.U, m),
               dE_low_base_graph=st.e_low(base.U, m) - base.e_low(base.U, m), E_low_own_s=st.e_low(st.U, m),
               C=M.coupling_C(base.L, st.L, base.U, m))
    return out


def holm(p: list[float], family: int = 8) -> list[tuple[float, bool]]:
    """Holm at alpha=0.05 over a family of size `family` (absent members count as p=1, i.e. the most permissive
    reading for the present ones). Returns (adjusted p, reject) in input order."""
    order = np.argsort(p)
    adj = np.zeros(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (family - rank) * p[i])); adj[i] = run
    return [(float(a), bool(a <= 0.05)) for a in adj]


def verdict(reject_blue: bool, reject_red: bool, reff_R: float) -> str:
    if reject_red:
        return "red"
    if reject_blue:
        return "blue"
    return "UNDERPOWERED" if reff_R < 20 else "not red"


def stage_vs_base(base: Ck, Hs: "np.ndarray | Ck", m: int, r: int, n_perm: int, rng: np.random.Generator,
                  full_null: bool = False, P: np.ndarray | None = None) -> dict:
    """Voting statistic f_low(R) vs N4' (Gram form) + non-voting rows for one stage. Hs = stage representation of the
    same draw. P (n x r PC basis of the base) is computed here if not supplied."""
    st = Hs if isinstance(Hs, Ck) else Ck(Hs)
    if P is None:
        P = base.V[:, ::-1][:, :r]          # top-r left singular vectors of centred H_b = top Gram eigenvectors
    Dsp = st.H - base.H
    Dl, R, lam_lin = M.split_linear(Dsp, P)
    Gr = R @ R.T
    obs = M.f_low_gram(Gr, base.U, m)
    perms = [rng.permutation(len(R)) for _ in range(n_perm)]
    null = np.array([N.f_low_null_gram(Gr, p, P, base.U, m) for p in perms])
    pb, pr = N.p_values(obs, null)
    out = dict(norm_D=float(np.linalg.norm(Dsp)), norm_H=float(np.linalg.norm(base.H)), ratio=float(np.linalg.norm(Dsp) / np.linalg.norm(base.H)),
               lambda_lin=lam_lin, f_low=obs, f_low_null_median=float(np.median(null)), p_blue=pb, p_red=pr, reff_R=reff(R),
               t_hat=M.t_k(base.H, Dsp, base.U, m).tolist())
    out.update(nonvoting(base, st, m))
    if full_null:
        nd = {k: [] for k in ("dPR_cov", "dclust", "dE_low_base_graph", "d_O", "d_NP")}
        for p in perms:
            Rp = R[p]; Rp = Rp - P @ (P.T @ Rp)
            cn = Ck(base.H + Dl + Rp)
            nd["dPR_cov"].append(cn.pr_cov() - base.pr_cov())
            nd["dclust"].append(M.clustering_coefficient(cn.A) - M.clustering_coefficient(base.A))
            nd["dE_low_base_graph"].append(cn.e_low(base.U, m) - base.e_low(base.U, m))
            nd["d_O"].append(M.eigenspace_overlap(base.U[:, 1:m + 1], cn.U[:, 1:m + 1])["d_O"])
            nd["d_NP"].append(M.neighbourhood_preservation(base.idx, cn.idx)["d_NP"])
        out["null_q"] = {k: [float(np.quantile(v, q)) for q in (0.05, 0.5, 0.95)] for k, v in nd.items()}
        out["null_pct"] = {k: float((1 + (np.array(v) <= out[k]).sum()) / (1 + len(v))) for k, v in nd.items()}
    return out


def load_geo(name: str, ref_ids=None, layer: int = 12, n: int = 500, K: int = 8):
    """(n,K,d) last-token vectors of the geography set at `layer` for runs/<name>; asserts token ids equal ref_ids."""
    import json
    a = np.load(f"runs/{name}/acts.npz")
    man = json.load(open(f"runs/{name}/manifest.json"))
    if ref_ids is not None and man["token_ids"] != ref_ids:
        raise RuntimeError(f"token ids of {name} differ from the base: pipeline broken")
    meta = a["meta"]
    assert (meta[: n * K, 0] == 0).all() and (meta[: n * K, 1] == np.repeat(np.arange(n), K)).all()
    return a[f"last_L{layer}"][: n * K].reshape(n, K, -1).astype(np.float64), man["token_ids"]


FAMILY = ["sft", "dpo", "instruct"]   # preregistered Holm family members present until step 11 (Olmo-3 RL-Zero absent)


def family_verdicts(res: dict) -> dict:
    """Holm over the 8-test family (absent Olmo-3 pair counts as p=1, provisional) + RLVR1 as secondary (own 2-p Holm).
    res[stage] needs p_blue, p_red, reff_R. Returns {stage: dict(holm_p_blue, holm_p_red, verdict)}."""
    out = {}
    h = holm([res[s][k] for s in FAMILY for k in ("p_blue", "p_red")], 8)
    for i, s in enumerate(FAMILY):
        (ab, rb), (ar, rr) = h[2 * i], h[2 * i + 1]
        out[s] = dict(holm_p_blue=ab, holm_p_red=ar, verdict=verdict(rb, rr, res[s]["reff_R"]))
    h2 = holm([res["rlvr1"]["p_blue"], res["rlvr1"]["p_red"]], 2)
    out["rlvr1"] = dict(holm_p_blue=h2[0][0], holm_p_red=h2[1][0], verdict=verdict(h2[0][1], h2[1][1], res["rlvr1"]["reff_R"]))
    return out
