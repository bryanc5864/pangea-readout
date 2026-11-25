# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Targets for auxiliary windows, restricted to selected families (Family L by default).

Writes compact shards holding only the selected families' channels -- per-bp
(n, channels, length) and global (n, channels) -- with their names, so the trainer can
place them into the full target layout with every other column NaN, which the loss masks.
Shard i takes windows i, i + n_shards, ... (strided, as for the main targets).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import landscape as LS  # noqa: E402
from channels import Resolution  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--families", nargs="+", default=["L"])
    ap.add_argument("--per-bp-families", nargs="+", default=None,
                    help="families whose per-bp channels are stored (default: all of "
                         "--families). Whole-sequence channels are stored for every family; "
                         "per-bp channels already at R2 0.999 gain nothing from more data and "
                         "would cost about 24 GB of GPU memory for 1M windows.")
    a = ap.parse_args()
    pb_fams = a.per_bp_families or a.families
    import h5py

    rows = []
    with open(a.windows) as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            if i % a.n_shards == a.shard:
                rows.append((r["frag_id"], r["sequence"]))
    pb = [n for n in LS.channel_names(Resolution.PER_BP) if n[0] in pb_fams]
    gl = [n for n in LS.channel_names(Resolution.GLOBAL) if n[0] in a.families]
    L = len(rows[0][1]) if rows else 0
    P = np.full((len(rows), len(pb), L), np.nan, dtype=np.float32)
    G = np.full((len(rows), len(gl)), np.nan, dtype=np.float32)
    fails, t0 = 0, time.time()
    for i, (_, s) in enumerate(rows):
        try:
            p, g = LS.landscape_arrays(s, pb, gl, minimised_pdb=None, n_ensemble=0,
                                       families=tuple(a.families))
            P[i], G[i] = p, g
        except Exception as exc:                                  # noqa: BLE001
            fails += 1
            if fails <= 3:
                print(f"window {i} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if i + 1 == 5 and fails == 5:
                raise RuntimeError("the first 5 windows all failed")
    tmp = a.out + ".partial"
    with h5py.File(tmp, "w") as h:
        h.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4)
        h.create_dataset("global", data=G, compression="gzip", compression_opts=4)
        dt = h5py.string_dtype()
        h.create_dataset("frag_id", data=np.array([r[0] for r in rows], dtype=object), dtype=dt)
        h.create_dataset("sequence", data=np.array([r[1] for r in rows], dtype=object), dtype=dt)
        h.attrs["per_bp_names"] = np.array(pb, dtype=object)
        h.attrs["global_names"] = np.array(gl, dtype=object)
        h.attrs["families"] = ",".join(a.families)
        h.attrs["per_bp_families"] = ",".join(pb_fams)
        h.attrs["n_failures"] = fails
        h.attrs["code_root"] = os.path.dirname(os.path.dirname(HERE))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(rows)} windows, {fails} failures, "
          f"{(time.time() - t0) / max(1, len(rows)):.3f} s/window", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
