# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Auxiliary genomic windows for the Family L data-scaling experiment.

Family L (RNA secondary structure) overfits: its validation R-squared peaks early and then
falls while every other family keeps improving, and its peak barely moves with capacity
(per-bp 0.645 at 2M, 0.661 at 50M, 0.654 at 200M). That is a data limit, and RNA folding
targets are cheap, so this draws many more windows from the same 15 panel genomes.

Leakage control is the main split's own criterion, applied by the same code: candidates
are grouped with every fragment of the existing library by MinHash similarity
(`splits.group_by_similarity`, Jaccard >= 0.5 on 11-mers), and any candidate that lands
in a group containing a library fragment is dropped, as is any exact duplicate. The
existing test side therefore has no near-duplicate among the auxiliary windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fasta import Fasta  # noqa: E402
from sample_fragments import sample_genomic  # noqa: E402
import splits as sp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="/home/bjc75/scratch/pangea/genomes/fa/panel_prepared.json")
    ap.add_argument("--library", default="/home/bjc75/scratch/pangea/fragments/library_v2.tsv")
    ap.add_argument("--per-species", type=int, default=70000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lib = []
    with open(a.library) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lib.append(r["sequence"].upper())
    lib_set = set(lib)
    rng = random.Random(a.seed)
    panel = json.load(open(a.panel))
    cand = []
    for g in panel["genomes"]:
        with Fasta(g["fasta"]) as fa:
            recs = sample_genomic(fa, a.per_species, rng, g["species"])
        recs = [r for r in recs if r["sequence"] not in lib_set]
        cand += recs
        print(f"  {g['species']:32s} {len(recs):7d} candidates", flush=True)
    seqs = lib + [r["sequence"] for r in cand]
    print(f"grouping {len(seqs)} sequences ({len(lib)} library + {len(cand)} candidates)", flush=True)
    labels, info = sp.group_by_similarity(seqs, threshold=0.5)
    lib_groups = set(np.asarray(labels[:len(lib)]).tolist())
    keep, seen = [], set()
    for i, r in enumerate(cand):
        g = int(labels[len(lib) + i])
        if g in lib_groups or r["sequence"] in seen:
            continue
        seen.add(r["sequence"])
        keep.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["frag_id", "stratum", "sequence", "source", "contig", "start", "strand"])
        for k, r in enumerate(keep):
            w.writerow([f"aux_{k:07d}", "panel_aux", r["sequence"], r["source"], r["contig"],
                        r["start"], r["strand"]])
    report = {"candidates": len(cand), "kept": len(keep),
              "dropped_similar_to_library": sum(1 for i, r in enumerate(cand)
                                               if int(labels[len(lib) + i]) in lib_groups),
              "grouping": {k: v for k, v in info.items() if isinstance(v, (int, float, str))}}
    json.dump(report, open(a.out + ".report.json", "w"), indent=1)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
