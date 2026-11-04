"""nulls - N4 and N4' permutation nulls (DESIGN 3.3-3.4).

N4 : permute the rows of D among entities.
N4': split D = D_lin + R with D_lin = P_r P_r^T D, permute the rows of R, re-orthogonalise to P_r,
     keep D_lin fixed; statistic recomputed on D_lin + R_pi.
p_blue = (1+#{T_pi <= T_obs})/(1+P), p_red = (1+#{T_pi >= T_obs})/(1+P).
"""
from __future__ import annotations

import numpy as np

from .measures import split_linear


def n4_draw(Dsp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return Dsp[rng.permutation(len(Dsp))]


def n4prime_draw(Dl: np.ndarray, R: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return the permuted residual R_pi (re-orthogonalised to P); the null displacement is Dl + R_pi."""
    Rp = R[rng.permutation(len(R))]
    return Rp - P @ (P.T @ Rp)


def p_values(t_obs: float, t_null: np.ndarray) -> tuple[float, float]:
    """(p_blue, p_red), one-sided, with the +1 correction (minimum attainable 1/(1+P))."""
    P = len(t_null)
    return (1 + float((t_null <= t_obs).sum())) / (1 + P), (1 + float((t_null >= t_obs).sum())) / (1 + P)


def f_low_null_gram(G: np.ndarray, perm: np.ndarray, P: np.ndarray, U: np.ndarray, m: int) -> float:
    """f_low(proj(R[perm])) from the Gram G = R R^T without forming the permuted matrix.
    With Y = (I-PP^T) Pm R: u^T Y = w^T R, w = ((I-PP^T)u)[inv]; total energy = tr G - |P[inv]^T R|^2.
    Equals the direct form to rounding (checked in the battery)."""
    inv = np.argsort(perm)
    Ulow, u1 = U[:, 1:m + 1], U[:, :1]
    proj = lambda Y: Y - P @ (P.T @ Y)
    Wl, W1, Wp = proj(Ulow)[inv], proj(u1)[inv], P[inv]
    el = np.einsum("ij,ij->", Wl, G @ Wl)
    e1 = np.einsum("ij,ij->", W1, G @ W1)
    tot = np.trace(G) - np.einsum("ij,ij->", Wp, G @ Wp)
    return float(el / (tot - e1))


def n4prime_f_low(Dsp: np.ndarray, Pbasis: np.ndarray, U: np.ndarray, m: int, n_perm: int,
                  rng: np.random.Generator) -> dict:
    """Colour test: observed f_low(R) vs N4' (Gram fast form). Returns obs, null, p_blue, p_red, lambda_lin."""
    from .measures import f_low  # local import: keeps nulls importable standalone
    Dl, R, lam_lin = split_linear(Dsp, Pbasis)
    G = R @ R.T
    obs = f_low(R, U, m)
    null = np.array([f_low_null_gram(G, rng.permutation(len(R)), Pbasis, U, m) for _ in range(n_perm)])
    pb, pr = p_values(obs, null)
    return dict(obs=obs, null=null, p_blue=pb, p_red=pr, lambda_lin=lam_lin)
