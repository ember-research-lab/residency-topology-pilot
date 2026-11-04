"""measures - every formula of DESIGN 2.4 items 1-11, exact.

H_b, H_s are (n,d) base/stage representations (centred inside each function); D = H_s - H_b;
U is the full eigenvector matrix of the base L_sym (column 0 trivial), low band = columns 1..m.
"""
from __future__ import annotations

import numpy as np

from .graph import centre


# ---- 1-3 decision-sheet triplet (reported with epsilon, non-voting) -------------------------------
def eigenspace_overlap(Ub: np.ndarray, Us: np.ndarray) -> dict:
    """(1) O = mean sigma_i^2 of U_b^T U_s; d_O = 1 - O; theta_max = largest principal angle (degrees).
    theta_max = arctan2(max sin, min cos) with sin from the complement residual: identical to arccos(min sigma)
    but accurate near 0 (arccos of 1-1e-16 is only good to ~1e-8 rad)."""
    C = Ub.T @ Us
    s = np.linalg.svd(C, compute_uv=False)
    sin_max = np.linalg.norm(Us - Ub @ C, 2)
    return dict(O=float((s**2).mean()), d_O=float(1 - (s**2).mean()),
                theta_max=float(np.degrees(np.arctan2(sin_max, s.min()))))


def neighbourhood_preservation(idx_b: np.ndarray, idx_s: np.ndarray) -> dict:
    """(2) NP = mean_i |N_b(i) & N_s(i)| / k on cosine kNN; d_NP = 1 - NP."""
    k = idx_b.shape[1]
    np_ = np.mean([len(set(a) & set(b)) for a, b in zip(idx_b, idx_s)]) / k
    return dict(NP=float(np_), d_NP=float(1 - np_))


def procrustes_residual(Hb: np.ndarray, Hs: np.ndarray) -> float:
    """(3) R_P = 1 - ||Hb~^T Hs~||_* with unit Frobenius norms. Exact identity (T0) used for accuracy:
    with thin SVDs A = Ua Sa Va^T, B = Ub Sb Vb^T, ||A^T B||_* = ||Sa Ua^T Ub Sb||_* (n x n), equal to
    sum sqrt(eig(G_a G_b)) without squaring tiny singular values."""
    A, B = centre(Hb), centre(Hs)
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    Ua, sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, sb, _ = np.linalg.svd(B, full_matrices=False)
    M = (sa[:, None] * (Ua.T @ Ub)) * sb[None]
    return float(1 - np.linalg.svd(M, compute_uv=False).sum())


def cka_linear(Hb: np.ndarray, Hs: np.ndarray) -> float:
    """Linear CKA (reference only)."""
    A, B = centre(Hb), centre(Hs)
    Ga, Gb = A @ A.T, B @ B.T
    return float((Ga * Gb).sum() / np.linalg.norm(Ga) / np.linalg.norm(Gb))


# ---- 4-6 colour of the displacement (voting statistic is 5) ---------------------------------------
def r_from_base(Hb: np.ndarray, frac: float = 0.90) -> int:
    """r = number of PCs reaching `frac` of base variance, capped at n/10 (frozen from base)."""
    n = Hb.shape[0]
    sv = np.linalg.svd(centre(Hb), compute_uv=False)
    c = np.cumsum(sv**2) / np.sum(sv**2)
    return int(min(np.searchsorted(c, frac) + 1, n // 10))


def pc_basis(Hb: np.ndarray, r: int) -> np.ndarray:
    """P_r: top-r left singular vectors of the centred base (n x r)."""
    Uh, _, _ = np.linalg.svd(centre(Hb), full_matrices=False)
    return Uh[:, :r]


def split_linear(Dsp: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(4) D_lin = P P^T D, R = D - D_lin, lambda_lin = |D_lin|^2/|D|^2."""
    Dl = P @ (P.T @ Dsp)
    R = Dsp - Dl
    return Dl, R, float((Dl**2).sum() / (Dsp**2).sum())


def f_low(R: np.ndarray, U: np.ndarray, m: int) -> float:
    """(5) direct form: sum_{j=2}^{m+1}|u_j^T R|^2 / sum_{j>=2}|u_j^T R|^2."""
    e = ((U.T @ R) ** 2).sum(1)
    return float(e[1:m + 1].sum() / e[1:].sum())


def f_low_gram(G: np.ndarray, U: np.ndarray, m: int) -> float:
    """(5) Gram form with G = R R^T: |u^T R|^2 = u^T G u. Equals f_low to rounding."""
    e = np.einsum("ij,ij->j", U, G @ U)
    return float(e[1:m + 1].sum() / e[1:].sum())


def coupling_C(Lb: np.ndarray, Ls: np.ndarray, U: np.ndarray, m: int) -> float:
    """(6) C = |U_low^T (L_s - L_b) U_high|_F / |L_s - L_b|_F."""
    E = Ls - Lb
    return float(np.linalg.norm(U[:, 1:m + 1].T @ E @ U[:, m + 1:]) / np.linalg.norm(E))


# ---- 7-10 ember-lab-admin's frame ---------------------------------------------------------------------------
def _pr(ev: np.ndarray) -> float:
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum())


def pr_cov(H: np.ndarray) -> float:
    """(7) participation ratio of eig(H_c^T H_c) (computed via the n x n Gram)."""
    Hc = centre(H)
    return _pr(np.linalg.eigvalsh(Hc @ Hc.T))


def pr_lap(L: np.ndarray) -> float:
    """(8) participation ratio of the Laplacian spectrum over j>=2 (non-voting)."""
    return _pr(np.linalg.eigvalsh(L)[1:])


def clustering_coefficient(A: np.ndarray) -> float:
    """(9) mean local clustering coefficient of the binary kNN graph (deg<2 contributes 0)."""
    deg = A.sum(1)
    tri = np.einsum("ij,jk,ki->i", A, A, A) / 2
    return float(np.where(deg > 1, tri / np.maximum(deg * (deg - 1) / 2, 1), 0).mean())


def degree_entropy_bits(A: np.ndarray) -> float:
    """(9) Shannon entropy (bits) of the degree distribution (non-voting)."""
    p = np.bincount(A.sum(1).astype(int)) / len(A)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_jaccard(Ab: np.ndarray, As: np.ndarray) -> float:
    """(9) Jaccard of the undirected edge sets."""
    e1, e2 = np.triu(Ab, 1) > 0, np.triu(As, 1) > 0
    return float((e1 & e2).sum() / (e1 | e2).sum())


def e_low(H: np.ndarray, U: np.ndarray, m: int) -> float:
    """(10) E_low(H) = sum_{j=2}^{m+1}|u_j^T H|^2 / sum_{j>=2}|u_j^T H|^2; U from the own graph (a)
    or the frozen base graph (b), chosen by the caller."""
    e = ((U.T @ centre(H)) ** 2).sum(1)
    return float(e[1:m + 1].sum() / e[1:].sum())


# ---- 11 odd readout -------------------------------------------------------------------------------
def t_k(Hb: np.ndarray, Dsp: np.ndarray, U: np.ndarray, m: int) -> np.ndarray:
    """(11, amended 2026-11-04) normalised signed mode stretch
    t^_k = u~_k^T D a_k / |H_b^T u~_k|, u~_k = u_k - mean(u_k), a_k = H_b^T u~_k / |H_b^T u~_k|, k=1..m.
    Relative stretch of mode k (+1 = amplitude doubled). Flips with D -> -D, invariant to the eigenvector sign
    convention, to a joint rescale of (H_b, D) and to a joint rotation. (The unnormalised form is
    |H_b^T u~_k| times this and is only scale-covariant.)"""
    Uc = U[:, 1:m + 1] - U[:, 1:m + 1].mean(0)
    out = np.empty(m)
    for j in range(m):
        v = Hb.T @ Uc[:, j]
        nv = np.linalg.norm(v)
        out[j] = Uc[:, j] @ (Dsp @ (v / nv)) / nv
    return out
