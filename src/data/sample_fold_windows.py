#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Genomic windows centred on quadruplex motifs, for Family M and the rare Family F channels.

Only about 0.4 percent of corpus fragments carry a quadruplex Family M can model (four tracts
of >= 3 G, loops 1-6 nt; DECISIONS.md), too few to learn the explicit-geometry channels from,
and Family F's quadruplex descriptors are rare events in the corpus for the same reason. This
draws 61 bp windows centred on G-tract clusters (>= 4 tracts of >= 3 G with gaps <= 12 nt) or
C-tract clusters (the quadruplex on the other strand) from the same 15 panel genomes, split
into windows Family M models (`fold/g4_select.py`: reason "ok") and the rest. Linear-time
tract grouping, not a nested regular expression (which backtracks badly on G-rich DNA).

Leakage control is the main split's own criterion, as for the auxiliary windows
(`sample_aux_windows.py`): candidates grouped with every library fragment by MinHash
similarity (Jaccard >= 0.5 on 11-mers) are dropped when they share a group with any library
fragment, so no test fragment has a near-duplicate here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics", "fold"))
from fasta import Fasta  # noqa: E402
import splits as sp  # noqa: E402
import g4_select as S  # noqa: E402

L = 61
MAX_GAP = 12


def clusters(seq: str, base: str):
    """(start, end) spans of >= 4 consecutive base{3,} tracts with gaps <= MAX_GAP."""
    tr = [(m.start(), m.end()) for m in re.finditer(base + "{3,}", seq)]
    out, i = [], 0
    while i < len(tr):
        j = i
        while j + 1 < len(tr) and tr[j + 1][0] - tr[j][1] <= MAX_GAP:
            j += 1
        if j - i + 1 >= 4:
            out.append((tr[i][0], tr[j][1]))
        i = j + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="/home/bjc75/scratch/pangea/genomes/fa/panel_prepared.json")
    ap.add_argument("--library", default="/home/bjc75/scratch/pangea/fragments/library_v2.tsv")
    ap.add_argument("--modelled-per-genome", type=int, default=1200)
    ap.add_argument("--other-per-genome", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = [r["sequence"].upper() for r in csv.DictReader(open(a.library), delimiter="\t")]
    lib_set = set(lib)
    rng = random.Random(a.seed)
    panel = json.load(open(a.panel))
    cand, per_genome = [], {}
    for g in panel["genomes"]:
        mod, oth, seen = [], [], set()
        with Fasta(g["fasta"]) as fa:
            for c in fa.names:
                n = fa.length(c)
                if n < L:
                    continue
                s = fa.fetch(c, 0, n).upper()
                for base in ("G", "C"):
                    for lo, hi in clusters(s, base):
                        if hi - lo > L - 10:
                            continue
                        st = (lo + hi) // 2 - L // 2
                        if st < 0 or st + L > n:
                            continue
                        w = s[st:st + L]
                        if any(b not in "ACGT" for b in w) or w in seen or w in lib_set:
                            continue
                        seen.add(w)
                        m = S.select(w)
                        rec = {"sequence": w, "source": g["species"], "contig": c, "start": st,
                               "strand": "+", "modelled": int(m is not None and m.reason == "ok")}
                        (mod if rec["modelled"] else oth).append(rec)
        rng.shuffle(mod); rng.shuffle(oth)
        per_genome[g["species"]] = {"modelled_found": len(mod), "other_found": len(oth)}
        cand += mod[:a.modelled_per_genome] + oth[:a.other_per_genome]
        print(f"  {g['species']:32s} modelled {len(mod):7d}  other {len(oth):7d}", flush=True)
    seqs = lib + [r["sequence"] for r in cand]
    print(f"grouping {len(seqs)} sequences ({len(lib)} library + {len(cand)} candidates)", flush=True)
    labels, info = sp.group_by_similarity(seqs, threshold=0.5)
    lib_groups = set(np.asarray(labels[:len(lib)]).tolist())
    keep = [r for i, r in enumerate(cand) if int(labels[len(lib) + i]) not in lib_groups]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["frag_id", "stratum", "sequence", "source", "contig", "start", "strand", "modelled"])
        for k, r in enumerate(keep):
            w.writerow([f"fold_g4_{k:07d}", "fold_g4", r["sequence"], r["source"], r["contig"],
                        r["start"], r["strand"], r["modelled"]])
    report = {"per_genome": per_genome, "candidates": len(cand), "kept": len(keep),
              "kept_modelled": sum(r["modelled"] for r in keep),
              "dropped_similar_to_library": len(cand) - len(keep)}
    json.dump(report, open(a.out + ".report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_genome"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
