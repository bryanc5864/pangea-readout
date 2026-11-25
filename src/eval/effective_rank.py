# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Effective rank of the channel stack (PLAN.md, "Effective rank").

Participation ratio and eigenvalue spectrum of the channel correlation matrix, with the
number of components needed for 95 percent of the variance, overall and per family, for
the per-base-pair channels (one sample per fragment position) and the whole-sequence
channels (one sample per fragment) separately, since the two have different sampling
units. Computed on the standardised training targets -- every channel at unit variance,
so the spectrum is of the correlation matrix and no channel dominates by its units.

Channels are included when they are finite for at least 90 percent of samples; samples
are included when every included channel is finite. Ensemble-subset channels (finite on
about 10 percent of fragments) are therefore excluded and reported as such, as are the
sin/cos halves of circular channels counted as two coordinates, which they are.

    participation ratio = (sum lambda)^2 / sum lambda^2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from data import InMemoryLandscape  # noqa: E402


def spectrum(M: np.ndarray) -> dict:
    C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    tot = lam.sum()
    cum = np.cumsum(lam) / tot
    return {"n_channels": int(M.shape[1]), "n_samples": int(M.shape[0]),
            "participation_ratio": float(tot ** 2 / (lam ** 2).sum()),
            "n_components_95": int(np.searchsorted(cum, 0.95) + 1),
            "n_components_99": int(np.searchsorted(cum, 0.99) + 1),
            "top_eigenvalue_fraction": [float(x) for x in (lam[:10] / tot)]}


def analyse(Y: np.ndarray, names, families, label, max_rows, rng) -> dict:
    fin = np.isfinite(Y)
    keep = fin.mean(axis=0) >= 0.9
    excluded = [n for n, k in zip(names, keep) if not k]
    Yk = Y[:, keep]
    rows = np.isfinite(Yk).all(axis=1)
    Yk = Yk[rows]
    if len(Yk) > max_rows:
        Yk = Yk[rng.choice(len(Yk), max_rows, replace=False)]
    Yk = Yk.astype(np.float64)
    std = Yk.std(axis=0)
    ok = std > 0
    names_k = [n for n, k in zip(names, keep) if k]
    fam_k = [f for f, k in zip(families, keep) if k]
    Yk = Yk[:, ok]
    names_k = [n for n, o in zip(names_k, ok) if o]
    fam_k = [f for f, o in zip(fam_k, ok) if o]
    out = {"overall": spectrum(Yk), "excluded_low_coverage": excluded, "by_family": {}}
    for fam in sorted(set(fam_k)):
        cols = [i for i, f in enumerate(fam_k) if f == fam]
        if len(cols) >= 2:
            out["by_family"][fam] = spectrum(Yk[:, cols])
    o = out["overall"]
    print(f"{label}: {o['n_channels']} channels, {o['n_samples']} samples -> participation "
          f"ratio {o['participation_ratio']:.1f}, 95% of variance in {o['n_components_95']} "
          f"components, 99% in {o['n_components_99']}", flush=True)
    for fam, d in out["by_family"].items():
        print(f"   family {fam}: {d['n_channels']:3d} channels  PR {d['participation_ratio']:6.1f}  "
              f"n95 {d['n_components_95']:3d}  n99 {d['n_components_99']:3d}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-rows", type=int, default=400000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = InMemoryLandscape(a.shards, cache_dir=a.cache_dir, verbose=False)
    rng = np.random.default_rng(0)
    P = ds.per_bp_np.astype(np.float32)                    # (N, C, L)
    N, C, L = P.shape
    pick = rng.choice(N, size=min(N, max(1, a.max_rows // L + 1)), replace=False)
    Ppos = P[pick].transpose(0, 2, 1).reshape(-1, C)       # one row per fragment position
    res = {"per_bp": analyse(Ppos, ds.pb["names"], ds.pb["family"], "per-bp", a.max_rows, rng),
           "global": analyse(ds.global_np.astype(np.float32), ds.gl["names"], ds.gl["family"],
                             "global", a.max_rows, rng)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
