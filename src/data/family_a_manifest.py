#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A production manifest: which fragment is solved at which ionic condition.

PLAN.md: "solve the full set at two anchor conditions chosen to bracket the range, then
solve a subset of 10 to 20 percent at the remaining conditions and learn the condition
residual". The manifest lists every corpus fragment (from the main shards, so the MPRA
stratum is included) with its conditions: the anchors for all, and each remaining
condition for a fixed fraction chosen by a hash of the fragment id (deterministic, the
same fragment keeps its subset membership across reruns, and different conditions get
different subsets so every fragment carries at least the anchors). Solute lengths per
condition are not here; the production job reads them from the convergence study.
"""
from __future__ import annotations

import argparse
import glob
import hashlib

import h5py


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def in_subset(frag_id: str, condition: str, fraction: float) -> bool:
    h = hashlib.sha1(f"{condition}|{frag_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < fraction


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--anchors", nargs="+", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--windows", default=None,
                    help="extend_windows.py output: solve each fragment on its long window")
    ap.add_argument("--subset-key", default=None,
                    help="list only fragments in this hash subset (e.g. 'ensemble')")
    ap.add_argument("--subset-fraction", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    win = {}
    if a.windows:
        import csv
        win = {r["frag_id"]: r["window"] for r in csv.DictReader(open(a.windows), delimiter="\t")}
    n, counts = 0, {c: 0 for c in a.anchors + a.others}
    with open(a.out, "w") as f:
        f.write("frag_id\tstratum\tsequence\tconditions\n")
        for p in sorted(glob.glob(a.shards)):
            with h5py.File(p, "r") as h:
                ids, seqs, st = _decode(h["frag_id"][:]), _decode(h["sequence"][:]), _decode(h["stratum"][:])
            for fid, s, t in zip(ids, seqs, st):
                if a.subset_key and not in_subset(fid, a.subset_key, a.subset_fraction):
                    continue
                conds = list(a.anchors) + [c for c in a.others if in_subset(fid, c, a.fraction)]
                for c in conds:
                    counts[c] += 1
                if win and fid not in win:
                    raise ValueError(f"no window for {fid}")
                f.write(f"{fid}\t{t}\t{win.get(fid, s)}\t{','.join(conds)}\n"); n += 1
    print(f"{n} fragments -> {a.out}; solves per condition: {counts}; total {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
