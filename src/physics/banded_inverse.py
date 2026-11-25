#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Selected inversion of a banded symmetric positive-definite matrix: the entries of K^-1 inside
the band, in O(n p^2) time and O(n p) memory, instead of the O(n^3) dense inverse.

Family D needs only the 6x6 diagonal blocks of the marginal covariance of the cgNA+ step
coordinates (`mesoscale.per_step_fluctuations` reads nothing else), and the cgNA+ stiffness is
banded (half-bandwidth 41 at every length, measured at 121, 1,000 and 10,240 bp). The recursion is
Takahashi's (Takahashi, Fagan & Chin 1973; Erisman & Tinney 1975): with K = L L^T (L lower,
banded), L^T Z = L^-1 gives, for i <= j <= i + p,

    Z[i, j] = (delta_ij / L[i, i] - sum_{k=i+1}^{i+p} L[k, i] Z[k, j]) / L[i, i],

swept from the last row to the first; every Z[k, j] it needs lies inside the band and is already
known. The result equals the corresponding entries of the dense inverse to rounding (tested).
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def banded_lower(K, p: int) -> np.ndarray:
    """Lower band storage ab[d, j] = K[j + d, j] (d = 0..p) of a sparse or dense symmetric K."""
    import scipy.sparse as sp
    n = K.shape[0]
    ab = np.zeros((p + 1, n))
    if sp.issparse(K):
        coo = K.tocoo()
        m = (coo.row >= coo.col) & (coo.row - coo.col <= p)
        d = coo.row[m] - coo.col[m]
        np.add.at(ab, (d, coo.col[m]), coo.data[m])
        # duplicate entries were summed; a symmetric csc/csr from cgNA+ has none
    else:
        K = np.asarray(K, float)
        for d in range(p + 1):
            ab[d, :n - d] = np.diagonal(K, -d)
    return ab


def half_bandwidth(K) -> int:
    import scipy.sparse as sp
    if sp.issparse(K):
        coo = K.tocoo()
        return int(np.max(np.abs(coo.row - coo.col))) if coo.nnz else 0
    K = np.asarray(K)
    nz = np.nonzero(K)
    return int(np.max(np.abs(nz[0] - nz[1]))) if len(nz[0]) else 0


def diagonal_blocks_of_inverse(K, starts: Iterable[int], size: int) -> Dict[int, np.ndarray]:
    """{start: (K^-1)[start:start+size, start:start+size]} for each requested block start.
    `size` must not exceed the half-bandwidth + 1 (blocks are read from the band of K^-1)."""
    from scipy.linalg import cholesky_banded
    p = max(half_bandwidth(K), size - 1)
    n = K.shape[0]
    ab = banded_lower(K, p)
    # pad with an identity tail so the recursion needs no edge cases
    abp = np.zeros((p + 1, n + p)); abp[:, :n] = ab; abp[0, n:] = 1.0   # K has no entries coupling into the pad
    Lb = cholesky_banded(abp, lower=True)        # Lb[d, j] = L[j + d, j]
    want = {int(s) for s in starts}
    out: Dict[int, np.ndarray] = {}
    W = np.eye(p + 1)                            # Z on rows/cols [i, i+p] (pad region: identity)
    for i in range(n + p - 1, -1, -1):
        lii = Lb[0, i]
        l = Lb[1:, i]                            # L[i+1 .. i+p, i]
        Zsub = W[:p, :p]                         # Z[i+1 .. i+p, i+1 .. i+p]
        row = -(l @ Zsub) / lii
        zii = (1.0 / lii - l @ row) / lii
        Wn = np.empty((p + 1, p + 1))
        Wn[0, 0] = zii; Wn[0, 1:] = row; Wn[1:, 0] = row; Wn[1:, 1:] = Zsub
        W = Wn
        if i in want:
            out[i] = W[:size, :size].copy()
    return out
