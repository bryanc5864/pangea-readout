#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Rare-event genomic windows: random panel windows kept when Family F detects a rare structure.

Several whole-sequence channels are rare events in the corpus -- one value covers >= 99 percent
of fragments (measured on 23,918 fragments: RNA G-quadruplex gain 99.5 percent zero, quadruplex
tetrad count 99.4 percent at two, triplet-repeat instability 99.4 percent zero, R-loop score 99.0
percent zero; triplex candidates 96.7 percent zero) -- so their R-squared rests on a few hundred
training events. Their labels are cheap, so this importance-samples training windows: uniform
random 61 bp windows from one panel genome (`sample_fragments.sample_genomic`), each scored by
Family F (milliseconds), kept when any event is present, capped per event type. Training-side
only: evaluation stays on the unbiased corpus split, and the merge step applies the main split's
MinHash leakage filter (`merge_rare_windows` below, run once over all genomes).
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
from fasta import Fasta  # noqa: E402
from sample_fragments import sample_genomic  # noqa: E402
import noncanonical as NC  # noqa: E402

EVENTS = {
    "rloop": lambda f: f.get("F_rloop_score", 0.0) > 0,
    "triplet": lambda f: f.get("F_triplet_repeat_instability", 0.0) > 0,
    "triplex": lambda f: f.get("F_triplex_candidate_count", 0.0) > 0,
    "g4": lambda f: f.get("F_g4_best_tetrads", 0.0) >= 3,
    "zdna": lambda f: f.get("F_zdna_tract_len_bp", 0.0) >= 18,          # top 1 percent
}


def sample_one_genome(g: dict, n_candidates: int, cap: int, seed: int):
    rng = random.Random(seed)
    with Fasta(g["fasta"]) as fa:
        recs = sample_genomic(fa, n_candidates, rng, g["species"])
    kept, counts = [], {k: 0 for k in EVENTS}
    for r in recs:
        f = NC.noncanonical_channels(r["sequence"])
        ev = [k for k, test in EVENTS.items() if test(f)]
        if not ev or all(counts[k] >= cap for k in ev):
            continue
        for k in ev:
            counts[k] += 1
        r["events"] = ",".join(ev)
        kept.append(r)
    return kept, counts, len(recs)


def merge_rare_windows(parts, library, out):
    import splits as sp
    lib = [r["sequence"].upper() for r in csv.DictReader(open(library), delimiter="\t")]
    lib_set = set(lib)
    cand, seen = [], set()
    for p in parts:
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r["sequence"] in lib_set or r["sequence"] in seen:
                continue
            seen.add(r["sequence"]); cand.append(r)
    labels, _ = sp.group_by_similarity(lib + [r["sequence"] for r in cand], threshold=0.5)
    lib_groups = set(np.asarray(labels[:len(lib)]).tolist())
    keep = [r for i, r in enumerate(cand) if int(labels[len(lib) + i]) not in lib_groups]
    with open(out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["frag_id", "stratum", "sequence", "source", "contig", "start", "strand", "events"])
        for k, r in enumerate(keep):
            w.writerow([f"rare_{k:07d}", "rare_aux", r["sequence"], r["source"], r["contig"],
                        r["start"], r["strand"], r["events"]])
    ev = {}
    for r in keep:
        for e in r["events"].split(","):
            ev[e] = ev.get(e, 0) + 1
    rep = {"candidates": len(cand), "kept": len(keep), "by_event": ev}
    json.dump(rep, open(out + ".report.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="/home/bjc75/scratch/pangea/genomes/fa/panel_prepared.json")
    ap.add_argument("--library", default="/home/bjc75/scratch/pangea/fragments/library_v2.tsv")
    ap.add_argument("--genome-index", type=int, default=None)
    ap.add_argument("--n-candidates", type=int, default=400000)
    ap.add_argument("--cap", type=int, default=1500, help="windows per event type per genome")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--merge", nargs="*", default=None, help="per-genome TSVs to merge and filter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.merge is not None:
        merge_rare_windows(a.merge, a.library, a.out)
        return 0
    g = json.load(open(a.panel))["genomes"][a.genome_index]
    kept, counts, n = sample_one_genome(g, a.n_candidates, a.cap, a.seed + a.genome_index)
    with open(a.out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "source", "contig", "start", "strand", "events"])
        for r in kept:
            w.writerow([r["sequence"], r["source"], r["contig"], r["start"], r["strand"], r["events"]])
    print(json.dumps({"genome": g["species"], "candidates": n, "kept": len(kept), "by_event": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
