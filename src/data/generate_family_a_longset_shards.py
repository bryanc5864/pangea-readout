#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A long-fragment set (PLAN.md: "Keep a smaller set of long fragments for those alone"):
shards of 205 bp genomic segments with Family A channels on their central 105 bp and the
window-level channels over 10, 25, 50 and 100 bp windows.

Each segment was solved as five tiles (`famA_longset/manifest_longset.tsv`: 121 bp windows offset
by 21 bp, each scoring its central 21 bp with the production construction at all nine conditions);
per base pair the tiles' values are converged in flank length (convergence study; and a 121 bp
solve agrees with the 61 bp production solute to RMS 0.02-0.06 kT/e at 50-150 mM, DECISIONS.md),
so their concatenation is the long molecule's per-base-pair profile at every condition --
including 250-500 mM, where a long solute itself does not converge. Window channels per position
(window centred there, defined where it fits in the 105 bp):

* A_lw_qnet_w{L}: the window's net charge, sum over its base pairs of (condensed counterion charge
  of the two phosphates' cells - 2) [e];
* A_lw_lpel_w{L}: the electrostatic persistence length, Odijk's finite-length OSF result
  l = h(kappa L) l_B lambda_eff^2 / (4 kappa^2), h(y) = 1 - 8/(3y) + exp(-y)(y/3 + 5/3 + 8/(3y)),
  L the window's contour length (its rise sum). lambda_eff: Manning's renormalised line charge 1/l_B
  -- the value that reproduces the measured salt dependence of DNA persistence length -- scaled by
  the window's uncompensated phosphate charge relative to the set-wide mean at that condition, so
  the absolute scale is Manning's and the sequence dependence is the PB atmosphere's.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
sys.path.insert(0, HERE)
import apbs_run as AR  # noqa: E402
import electrostatics_family as A  # noqa: E402
import landscape as LS  # noqa: E402
from channels import Resolution  # noqa: E402
from generate_targets import channel_metadata  # noqa: E402

N_TILES, TILE, SEG = 5, 21, 205
STRETCH = N_TILES * TILE
BJERRUM_A = 7.1


def odijk_h(y):
    y = np.asarray(y, float)
    return 1.0 - 8.0 / (3.0 * y) + np.exp(-y) * (y / 3.0 + 5.0 / 3.0 + 8.0 / (3.0 * y))


def tile_profiles(solves_dir, fid):
    """Per tile: Family A per-bp channels on the central 21 bp (dict name -> (21,) array)."""
    out = []
    for k in range(N_TILES):
        p = os.path.join(solves_dir, f"{fid}__L{k}.npz")
        if not os.path.exists(p):
            return None
        z = dict(np.load(p))
        prof, _ = A.assemble(z, n_bp=121)
        lo = (121 - A.SCORED) // 2
        out.append({n: np.asarray(v, float)[lo:lo + A.SCORED] for n, v in prof.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", required=True); ap.add_argument("--solves", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--per-shard", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pb = LS.channel_names(Resolution.PER_BP, ["A"])
    meta = channel_metadata(["A"])
    segs = list(csv.DictReader(open(a.segments), delimiter="\t"))
    stretches = {}
    for r in segs:
        t = tile_profiles(a.solves, r["frag_id"])
        if t is not None:
            stretches[r["frag_id"]] = {n: np.concatenate([tt[n] for tt in t]) for n in t[0]}
    print(f"{len(stretches)} of {len(segs)} segments with all tiles solved", flush=True)
    # set-wide mean uncompensated charge per base pair, per condition
    ref = {}
    for cname, t in A.CONDITIONS:
        v = [2.0 - s[f"A_W_condensed_{t}"] - s[f"A_C_condensed_{t}"] for s in stretches.values()]
        v = np.concatenate(v) if v else np.array([np.nan])
        ref[t] = float(np.nanmean(v))
    for fid, s in stretches.items():
        for cname, t in A.CONDITIONS:
            unc = 2.0 - s[f"A_W_condensed_{t}"] - s[f"A_C_condensed_{t}"]
            xi = s.get("A_manning_xi", np.full(STRETCH, np.nan))        # = l_B / (rise / 2)
            rise = np.where(np.isfinite(xi) & (xi > 0), 2.0 * BJERRUM_A / np.where(xi > 0, xi, 1.0), 3.38)
            cond = AR.PRODUCTION_CONDITIONS_BY_NAME[cname]
            kappa = 1.0 / AR.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
            for w in A.LONG_WINDOWS:
                q = np.full(STRETCH, np.nan); lp = np.full(STRETCH, np.nan)
                h = w // 2
                for i in range(h, STRETCH - (w - h) + 1):
                    sl = slice(i - h, i - h + w)
                    if not np.isfinite(unc[sl]).all():
                        continue
                    q[i] = float(-np.sum(unc[sl]))
                    lam = (1.0 / BJERRUM_A) * float(np.mean(unc[sl])) / ref[t]
                    L = float(np.sum(rise[sl]))
                    lp[i] = float(odijk_h(kappa * L) * BJERRUM_A * lam ** 2 / (4.0 * kappa ** 2)) / 10.0
                s[f"A_lw_qnet_w{w}_{t}"] = q; s[f"A_lw_lpel_w{w}_{t}"] = lp
    ids = sorted(stretches)
    seg_of = {r["frag_id"]: r["segment"] for r in segs}
    for j in range(0, len(ids), a.per_shard):
        part = ids[j:j + a.per_shard]
        P = np.full((len(part), len(pb), STRETCH), np.nan, dtype=np.float32)
        for i, fid in enumerate(part):
            for c, n in enumerate(pb):
                if n in stretches[fid]:
                    P[i, c] = stretches[fid][n]
        seqs = [seg_of[f] for f in part]
        out = os.path.join(a.out, f"longset_{j // a.per_shard:03d}.h5")
        with h5py.File(out + ".tmp", "w") as hh:
            hh.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4, chunks=True)
            dt = h5py.string_dtype()
            hh.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=dt)
            hh.create_dataset("frag_id", data=np.array(part, dtype=object), dtype=dt)
            hh.attrs["stretch_start"] = (SEG - STRETCH) // 2
            hh.attrs["stretch_length"] = STRETCH
            for k, v in meta.items():
                if k.startswith("per_bp_") and isinstance(v, list):
                    hh.attrs[k] = np.array([str(x) for x in v], dtype=object) if not isinstance(v[0], (bool, int, np.integer)) else np.array(v)
            hh.attrs["seq_sha1"] = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
            hh.attrs["uncompensated_reference"] = str(ref)
        os.replace(out + ".tmp", out)
    print(f"wrote {len(ids)} segments; reference uncompensated charge per bp: {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
