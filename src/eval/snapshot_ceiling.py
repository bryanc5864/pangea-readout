# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Sampling-noise ceilings of the relaxed-ensemble channels (16 snapshots per fragment).

The worst non-L per-bp channels are the ones estimated from the 16 relaxed snapshots on
the 10 percent ensemble subset: sugar-pucker populations and the spreads of the backbone
torsions. Their sampling noise can be estimated from the targets themselves:
  - a population p estimated from n snapshots has variance p(1 - p)/n, estimated without
    bias by p_hat(1 - p_hat)/(n - 1);
  - a spread s estimated from n snapshots has variance about s^2 / (2(n - 1)) (the
    Gaussian result; torsion spreads are circular, so this is an approximation).
Ceiling R-squared = 1 - mean noise variance / total variance across fragments and
positions, the same definition as `noise_ceiling.py`. Raw target units.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

N_SNAPSHOTS = 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="results/snapshot_ceiling_v2.json")
    ap.add_argument("--max-shards", type=int, default=None)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))[: a.max_shards]
    with h5py.File(paths[0], "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h.attrs["per_bp_names"]]
    pop = [i for i, n in enumerate(names) if ("endo_prob" in n or n.startswith("C_BI_fraction"))]
    spr = [i for i, n in enumerate(names)
           if n.startswith("C_") and n.endswith("_sd") and any(f"_{t}_" in n for t in
              ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"))]
    idx = pop + spr
    vals = {i: [] for i in idx}
    for p in paths:
        with h5py.File(p, "r") as h:
            if "in_ensemble_subset" in h:
                sub = np.asarray(h["in_ensemble_subset"][:], dtype=bool)
                if not sub.any():
                    continue
                Y = h["per_bp"][np.flatnonzero(sub)][:, idx, :]
            else:
                Y = h["per_bp"][:, idx, :]
        for k, i in enumerate(idx):
            v = Y[:, k, :].ravel()
            vals[i].append(v[np.isfinite(v)])
    out = {"n_snapshots": N_SNAPSHOTS, "n_shards": len(paths), "per_bp": {}}
    for i in idx:
        v = np.concatenate(vals[i]).astype(np.float64)
        if v.size < 100:
            continue
        total = float(v.var())
        if i in pop:
            noise = float(np.mean(v * (1 - v)) / (N_SNAPSHOTS - 1))
            kind = "population"
        else:
            noise = float(np.mean(v ** 2) / (2 * (N_SNAPSHOTS - 1)))
            kind = "spread"
        out["per_bp"][names[i]] = {"kind": kind, "n_values": int(v.size), "noise_var": noise,
                                   "total_var": total,
                                   "ceiling_r2": (1 - noise / total) if total > 0 else None}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for n, d in out["per_bp"].items():
        c = d["ceiling_r2"]
        print(f"{n:32s} {d['kind']:10s} n={d['n_values']:8d} ceiling "
              f"{'nan' if c is None else f'{c:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
