"""Stand-in construction lifted unchanged from tests/sim/sim3.py (lines 1-13) and tests/sim/sim.py (sphere/signal).

The base H is the 500-point sphere embedded by random Fourier features into 512 dims plus template-averaged noise.
"""
import numpy as np

from rtp.graph import eigh_sorted, gap_rule_m, laplacian

n, Dm, K = 500, 512, 8


def sim3_base():
    rng = np.random.default_rng(7)
    lat = np.arcsin(rng.uniform(-1, 1, n)); lon = rng.uniform(-np.pi, np.pi, n)
    P = np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    W = rng.standard_normal((3, 256))*2.0; b = rng.uniform(0, 2*np.pi, 256); A = rng.standard_normal((256, Dm))/16
    S = np.cos(P@W+b)@A; S -= S.mean(0); S /= np.sqrt((S**2).sum(1).mean())
    H = S + 0.5/np.sqrt(Dm)*rng.standard_normal((n, Dm))/np.sqrt(K)
    H -= H.mean(0)
    return dict(H=H, S=S, lat=lat, rng=rng)


def sim3_graph(H):
    Lb = laplacian(H, k_self=7)
    lam, U = eigh_sorted(Lb)
    m = gap_rule_m(lam)
    return Lb, lam, U, m


def sim_sphere_signal(rng):
    """sim.py construction (seed 20261104). Returns (signal fn, S0, LAT, rng) with the exact draw order of sim.py."""
    D = 512
    def sphere(n_):
        lat = np.arcsin(rng.uniform(-1, 1, n_)); lon = rng.uniform(-np.pi, np.pi, n_)
        return np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], lat
    P, LAT = sphere(n)
    W = rng.standard_normal((3, 256)) * 2.0; b = rng.uniform(0, 2*np.pi, 256)
    A = rng.standard_normal((256, D)) / np.sqrt(256)
    def signal(P_):
        S = np.cos(P_ @ W + b) @ A
        S -= S.mean(0); return S / np.sqrt((S**2).sum(1).mean())
    return sphere, signal, signal(P), LAT
