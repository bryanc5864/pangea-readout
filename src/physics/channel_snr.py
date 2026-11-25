#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Signal against replicate noise for every Family A channel, on the production construction.

Signal: the spread of a channel across the composition-control fragments -- in total and
within composition-matched groups (base order only, the gate's question) -- pooled over the
scored base pairs. Noise: the spread across replicate solves of one sequence (grid jitter
plus a 0.001 A input perturbation, so the minimiser's sensitivity is included), pooled the
same way. Reported per channel and summarised by channel kind, so a channel whose
base-order variation does not clear its own construction noise is visible before
production rather than after training on it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import electrostatics_family as A  # noqa: E402


def _load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def kind_of(name: str) -> str:
    n = re.sub(r"_(c\d{3}|k150|mg150)$", "", name)
    n = re.sub(r"^A_s\d", "A_s*", n)
    n = re.sub(r"_d\d+$", "_d*", n)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragments", required=True, help="directory of composition-control .npz")
    ap.add_argument("--replicates", required=True, help="directory of replicate .npz of one sequence")
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = {r["frag_id"]: r for r in csv.DictReader(open(a.library), delimiter="\t")}
    F, groups = [], []
    for p in sorted(glob.glob(os.path.join(a.fragments, "*.npz"))):
        fid = os.path.basename(p)[:-4]
        if fid not in lib:
            continue
        pr, gl = A.assemble(_load(p))
        F.append((pr, gl)); groups.append(lib[fid]["control_group"])
    R = [A.assemble(_load(p)) for p in sorted(glob.glob(os.path.join(a.replicates, "*.npz")))]
    if len(F) < 20 or len(R) < 3:
        print(f"too few fragments ({len(F)}) or replicates ({len(R)})"); return 2
    groups = np.asarray(groups)
    rows = []
    names = [n for n in F[0][0] if np.isfinite(F[0][0][n]).any()]
    for n in names:
        X = np.stack([f[0][n] for f in F])[:, 20:41]                     # (frag, 21)
        Q = np.stack([r[0][n] for r in R])[:, 20:41]                     # (rep, 21)
        tot = np.sqrt(np.nanmean(np.nanvar(X, axis=0, ddof=1)))
        within = []
        for g in np.unique(groups):
            G = X[groups == g]
            if len(G) > 2:
                within.append(np.nanvar(G, axis=0, ddof=1))
        win = np.sqrt(np.nanmean(np.stack(within))) if within else np.nan
        noise = np.sqrt(np.nanmean(np.nanvar(Q, axis=0, ddof=1)))
        rows.append({"channel": n, "kind": kind_of(n), "total_sd": tot, "within_group_sd": win,
                     "replicate_sd": noise, "snr_total": tot / noise if noise > 0 else np.nan,
                     "snr_within": win / noise if noise > 0 else np.nan})
    for n in F[0][1]:
        x = np.array([f[1][n] for f in F]); q = np.array([r[1][n] for r in R])
        if not np.isfinite(x).any():
            continue
        tot, noise = np.nanstd(x, ddof=1), np.nanstd(q, ddof=1)
        within = [np.nanvar(x[groups == g], ddof=1) for g in np.unique(groups) if (groups == g).sum() > 2]
        win = np.sqrt(np.nanmean(within)) if within else np.nan
        rows.append({"channel": n, "kind": kind_of(n), "total_sd": tot, "within_group_sd": win,
                     "replicate_sd": noise, "snr_total": tot / noise if noise > 0 else np.nan,
                     "snr_within": win / noise if noise > 0 else np.nan})
    with open(a.out.replace(".json", ".tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    summ = {k: {"n": len(v), "median_snr_total": round(float(np.nanmedian([r["snr_total"] for r in v])), 2),
                "median_snr_within": round(float(np.nanmedian([r["snr_within"] for r in v])), 2),
                "min_snr_within": round(float(np.nanmin([r["snr_within"] for r in v])), 2)}
            for k, v in sorted(by.items())}
    out = {"n_fragments": len(F), "n_groups": int(len(np.unique(groups))), "n_replicates": len(R),
           "n_channels": len(rows), "by_kind": summ,
           "fraction_snr_within_above_5": round(float(np.mean([r["snr_within"] > 5 for r in rows
                                                               if np.isfinite(r["snr_within"])])), 4)}
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:28s} n={v['n']:3d}  SNR total {v['median_snr_total']:7.2f}  within-group {v['median_snr_within']:7.2f}  (min {v['min_snr_within']})")
    print(f"channels with within-group SNR > 5: {out['fraction_snr_within_above_5']:.1%} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
