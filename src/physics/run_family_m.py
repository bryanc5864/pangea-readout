#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M production: explicit quadruplex models for rows of a window/library TSV.

Resumable at fragment granularity (a fragment with an .npz is skipped), so a preempted task
loses at most the fragment it was on. Fragments without a modelled quadruplex are cheap and
are written too (modelled = 0), so the packing step can tell "computed, nothing to model"
from "not computed". `src/data/pack_family_m.py` builds side shards from the .npz files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structure_folds as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True, help="TSV with frag_id and sequence columns")
    ap.add_argument("--rows", required=True, help="0-based inclusive range, e.g. 0-99")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", type=int, default=M.N_MODELS)
    ap.add_argument("--workroot", default=None)
    ap.add_argument("--only-modelled", action="store_true",
                    help="skip rows whose 'modelled' column is 0 (quadruplex windows file)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.windows), delimiter="\t"))[int(lo): int(hi) + 1]
    pb, gl = M.channel_names_per_bp(), M.channel_names_global()
    n_done = n_mod = 0
    for r in rows:
        fid, seq = r["frag_id"], r["sequence"]
        if a.only_modelled and str(r.get("modelled", "1")) == "0":
            continue
        out = os.path.join(a.out, f"{fid}.npz")
        if os.path.exists(out):
            n_done += 1
            continue
        t = time.time()
        try:
            prof, glob = M.fold_channels(seq, n_models=a.n_models, workroot=a.workroot)
            status = "ok"
        except Exception as exc:                                     # noqa: BLE001
            print(f"{fid} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            continue
        P = np.stack([np.asarray(prof[k], dtype=np.float32) for k in pb])
        G = np.array([glob[k] for k in gl], dtype=np.float32)
        tmp = out + ".partial.npz"
        np.savez(tmp, per_bp=P, global_=G, per_bp_names=np.array(pb), global_names=np.array(gl),
                 sequence=seq, seconds=time.time() - t)
        os.replace(tmp, out)
        n_done += 1; n_mod += int(glob["M_g4_modelled"] == 1.0)
        print(f"{fid} {status} modelled={int(glob['M_g4_modelled'])} {time.time() - t:.0f}s", flush=True)
    print(f"{n_done}/{len(rows)} rows done ({n_mod} modelled this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
