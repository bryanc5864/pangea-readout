#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M production for the triplex, i-motif and cruciform models (quadruplexes:
`run_family_m3.py`). Rows of a TSV with frag_id, sequence and -- for cruciforms -- the long
window (as famA_prod/windows.tsv); one .npz per fragment (per_bp, global_, names), resumable.
Fragments without a candidate are written too (modelled = 0)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "fold"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--fold", choices=("triplex", "imotif", "cruciform"), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workroot", default=None)
    ap.add_argument("--windows", default=None, help="frag_id -> window TSV (cruciform)")
    ap.add_argument("--window-offset", type=int, default=30)
    a = ap.parse_args()
    if a.fold == "triplex":
        import structure_triplex as Mod
        fn = lambda r, w: Mod.triplex_channels(r["sequence"], a.workroot)            # noqa: E731
    elif a.fold == "imotif":
        import structure_imotif as Mod
        fn = lambda r, w: Mod.imotif_channels(r["sequence"], workroot=a.workroot)    # noqa: E731
    else:
        import structure_cruciform as Mod
        fn = lambda r, w: Mod.cruciform_channels(r["sequence"], w, a.window_offset, a.workroot)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.library), delimiter="\t"))[int(lo): int(hi) + 1]
    wins = {}
    if a.fold == "cruciform":
        want = {r["frag_id"] for r in rows}
        for r in csv.DictReader(open(a.windows), delimiter="\t"):
            if r["frag_id"] in want:
                wins[r["frag_id"]] = r["window"]
    pb, gl = Mod.channel_names_per_bp(), Mod.channel_names_global()
    for r in rows:
        fid = r["frag_id"]
        out = os.path.join(a.out, f"{fid}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        try:
            prof, glob = fn(r, wins.get(fid))
        except Exception as exc:                                     # noqa: BLE001
            print(f"{fid} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            continue
        P = np.stack([np.asarray(prof[k], dtype=np.float32) for k in pb])
        G = np.array([glob[k] for k in gl], dtype=np.float32)
        tmp = out + ".partial.npz"
        np.savez(tmp, per_bp=P, global_=G, per_bp_names=np.array(pb), global_names=np.array(gl),
                 sequence=r["sequence"], seconds=time.time() - t)
        os.replace(tmp, out)
        mk = [k for k in gl if k.endswith("_modelled")][0]
        print(f"{fid} ok modelled={int(glob[mk])} {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
