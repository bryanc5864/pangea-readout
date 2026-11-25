# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""External validation against ORChID hydroxyl-radical cleavage (measured, not solver).

ORChID (Greenbaum, Parker, Tullius; dna.bu.edu/orchid) tabulates measured cleavage at
each of the four positions of every tetramer, on the strand the tetramer is written on.
Hydroxyl radical cleavage is hydrogen abstraction at the deoxyribose C4' and C5', so it
reports backbone solvent accessibility and, through it, minor groove width.

For each channel, the value at every corpus occurrence of each tetramer is averaged at the
same four positions -- Watson-strand channels over Watson occurrences, Crick-strand channels
over Crick occurrences (the reverse complement on the Watson strand, positions reversed),
base-pair channels over both -- and the 256 x 4 averages are correlated with the measured
table. The primary number uses aligned positions only; no offset is searched.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

COMP = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--table", default="/home/bjc75/scratch/pangea/processed/orchid_tetramer_256.tsv")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meas = {}
    with open(a.table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            meas[r["tetramer"]] = [float(r[f"mean_pos{k}"]) for k in range(1, 5)]
    paths = sorted(glob.glob(a.shards))[: a.max_shards]
    sums = {c: defaultdict(lambda: np.zeros(4)) for c in a.channels}
    cnts = {c: defaultdict(lambda: np.zeros(4)) for c in a.channels}
    for p in paths:
        with h5py.File(p, "r") as h:
            names = [x.decode() if isinstance(x, bytes) else x for x in h.attrs["per_bp_names"]]
            seqs = [x.decode() if isinstance(x, bytes) else x for x in h["sequence"][:]]
            cols = {c: names.index(c) for c in a.channels if c in names}
            P = h["per_bp"][:, sorted(cols.values()), :]
            colpos = {c: sorted(cols.values()).index(i) for c, i in cols.items()}
        for f_, s in enumerate(seqs):
            L = len(s)
            for i in range(L - 3):
                t = s[i:i + 4]
                if t not in meas:
                    continue
                for c, j in colpos.items():
                    v = P[f_, j, i:i + 4]
                    if c.endswith("_C"):
                        continue                      # Crick channels use Crick occurrences
                    ok = np.isfinite(v)
                    sums[c][t][ok] += v[ok]; cnts[c][t][ok] += 1
                # the same Watson positions read as the reverse-complement tetramer on Crick
                tc = rc(t)
                for c, j in colpos.items():
                    if c.endswith("_W"):
                        continue
                    v = P[f_, j, i:i + 4][::-1]           # Crick 5'->3' = Watson reversed
                    ok = np.isfinite(v)
                    sums[c][tc][ok] += v[ok]; cnts[c][tc][ok] += 1
    from scipy.stats import pearsonr, spearmanr
    out = {"n_shards": len(paths), "source": "ORChID tetramer table (dna.bu.edu/orchid)",
           "channels": {}}
    tets = sorted(meas)
    M = np.array([meas[t] for t in tets])
    for c in a.channels:
        if c not in sums or not sums[c]:
            continue
        A = np.array([sums[c][t] / np.maximum(cnts[c][t], 1) for t in tets])
        ok = np.isfinite(A) & np.isfinite(M) & (np.array([cnts[c][t] for t in tets]) > 0)
        r_all = pearsonr(A[ok], M[ok])[0]
        per_pos = [float(pearsonr(A[:, k], M[:, k])[0]) for k in range(4)]
        rho = spearmanr(A[ok], M[ok])[0]
        out["channels"][c] = {"pearson_r_all_positions": float(r_all),
                              "spearman_rho_all_positions": float(rho),
                              "pearson_r_by_position": per_pos,
                              "mean_instances_per_tetramer": float(np.mean([cnts[c][t].mean() for t in tets]))}
        print(f"{c:32s} r {r_all:+.3f}  rho {rho:+.3f}  by position "
              + " ".join(f"{x:+.3f}" for x in per_pos), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
