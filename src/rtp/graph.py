"""graph - entity graphs and low-band selection (DESIGN 2.3).

Spectral graph: dense self-tuning Gaussian kernel, diffusion-maps alpha=1, symmetric normalised Laplacian.
Combinatorial graph: cosine kNN on centred vectors, symmetrised by union, binary (neighbourhood measures only).
"""
from __future__ import annotations

import numpy as np


class GraphError(ValueError):
    """Raised when a graph is degenerate (disconnected, bad m range)."""


def centre(X: np.ndarray) -> np.ndarray:
    return X - X.mean(0)


def sq_dists(X: np.ndarray) -> np.ndarray:
    sq = (X**2).sum(1)
    return np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0.0)


def laplacian(X: np.ndarray, k_self: int = 7, alpha: float = 1.0) -> np.ndarray:
    """L_sym = I - D^-1/2 K~ D^-1/2 for the self-tuning kernel on centred X.

    W_ij = exp(-|h_i-h_j|^2 / (s_i s_j)), s_i = distance to the k_self-th neighbour;
    K~ = Q^-alpha W Q^-alpha with Q = diag(W 1) (alpha=1: diffusion maps); D = diag(K~ 1).
    Invariant to a global rescale of X (s_i rescales with it) and to orthogonal rotations.
    """
    X = centre(np.asarray(X, dtype=np.float64))
    d2 = sq_dists(X)
    s = np.sqrt(np.sort(d2, axis=1)[:, k_self])
    W = np.exp(-d2 / (s[:, None] * s[None]))
    np.fill_diagonal(W, 0.0)
    q = W.sum(1)
    Kt = W / np.outer(q**alpha, q**alpha)
    d = Kt.sum(1)
    return np.eye(len(X)) - Kt / np.sqrt(np.outer(d, d))


def eigh_sorted(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues ascending and eigenvectors (columns); column 0 is the trivial mode."""
    lam, U = np.linalg.eigh(L)
    return lam, U


def gap_rule_m(lams: np.ndarray | list[np.ndarray], m_min: int = 4, m_max: int = 20) -> int:
    """m in [m_min, m_max] maximising the relative gap (lam_{m+2}-lam_{m+1})/lam_{m+2} (1-indexed modes).

    Low band = modes 2..m+1. With a list of spectra (bootstrap replicates) the relative gap is averaged
    over replicates before the argmax. Raises if the winning m sits on a boundary only when the caller
    asks (see check_m_interior) - DESIGN 6.3 step 4 treats a boundary m as BROKEN.
    """
    spectra = [np.asarray(lams)] if isinstance(lams, np.ndarray) else [np.asarray(x) for x in lams]
    ms = np.arange(m_min, m_max + 1)
    gaps = np.zeros(len(ms))
    for lam in spectra:
        # 0-indexed: lam[m] = mode m+1 (top of the band), lam[m+1] = mode m+2
        gaps += np.array([(lam[m + 1] - lam[m]) / lam[m + 1] for m in ms])
    return int(ms[int(np.argmax(gaps / len(spectra)))])


def check_m_interior(m: int, m_min: int = 4, m_max: int = 20) -> None:
    if m in (m_min, m_max):
        raise GraphError(f"m={m} sits on a boundary of [{m_min},{m_max}]")


def check_connected(lam: np.ndarray, tol: float = 1e-8) -> None:
    if lam[1] < tol:
        raise GraphError(f"graph disconnected: lambda_2={lam[1]:.3e} < {tol}")


def cosine_knn(X: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN on centred vectors. Returns (idx[n,k], symmetrised binary adjacency A[n,n])."""
    X = centre(np.asarray(X, dtype=np.float64))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ X.T
    np.fill_diagonal(C, -np.inf)
    idx = np.argpartition(-C, k, axis=1)[:, :k]
    A = np.zeros_like(C)
    A[np.repeat(np.arange(len(X)), k), idx.ravel()] = 1.0
    return idx, np.maximum(A, A.T)


def knn_laplacian(A: np.ndarray) -> np.ndarray:
    """I - D^-1/2 A D^-1/2 for a binary kNN adjacency (the graph behind PR_lap and the 2.5 E_low values)."""
    d = A.sum(1)
    return np.eye(len(A)) - A / np.sqrt(np.outer(d, d))
