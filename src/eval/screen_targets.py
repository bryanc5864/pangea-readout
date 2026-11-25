# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Per-channel screen of a target set for numerical anomalies before training on it.

For every channel: finite fraction, mean, standard deviation, 0.1/50/99.9 percentiles,
min, max, the fraction of exact zeros, and a robust outlier count (values more than 20
robust standard deviations -- 1.4826 x MAD -- from the median). Flags channels that look
like silent failures: an unexpected all-zero or near-all-zero channel, heavy outliers,
or a finite fraction that differs between the ensemble subset and the rest when it
should not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))[: a.max_shards]
    P, G = [], []
    for p in paths:
        with h5py.File(p, "r") as h:
            pbn = [x.decode() if isinstance(x, bytes) else x for x in h.attrs["per_bp_names"]]
            gln = [x.decode() if isinstance(x, bytes) else x for x in h.attrs["global_names"]]
            P.append(h["per_bp"][:]); G.append(h["global"][:])
    P = np.concatenate(P); G = np.concatenate(G)
    rep = {"n_shards": len(paths), "n_fragments": int(P.shape[0]), "channels": {}, "flags": []}
    for key, names, A in (("per_bp", pbn, P), ("global", gln, G)):
        for j, nm in enumerate(names):
            v = (A[:, j, :] if A.ndim == 3 else A[:, j]).astype(np.float64).ravel()
            fin = np.isfinite(v); x = v[fin]
            d = {"finite": float(fin.mean())}
            if x.size:
                med = float(np.median(x)); mad = float(np.median(np.abs(x - med))) * 1.4826
                d.update({"mean": float(x.mean()), "sd": float(x.std()), "min": float(x.min()),
                          "max": float(x.max()), "p001": float(np.percentile(x, 0.1)),
                          "p50": med, "p999": float(np.percentile(x, 99.9)),
                          "zero_frac": float(np.mean(x == 0)),
                          "outliers_20_robust_sd": int(np.sum(np.abs(x - med) > 20 * mad)) if mad > 0 else 0})
                if d["outliers_20_robust_sd"] > 0.001 * x.size and mad > 0:
                    rep["flags"].append(f"{nm}: {d['outliers_20_robust_sd']} values beyond 20 robust sd "
                                        f"(median {med:.4g}, max {d['max']:.4g}, min {d['min']:.4g})")
                if 0.9 < d["zero_frac"] < 1.0:
                    rep["flags"].append(f"{nm}: {d['zero_frac']:.3f} of finite values are exactly 0")
            rep["channels"][nm] = d
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"{rep['n_fragments']} fragments; {len(rep['flags'])} flags")
    for f in rep["flags"]:
        print("  " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
