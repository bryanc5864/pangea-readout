#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Long windows for context-limited channels: each corpus fragment with F bp of flank either side.

Low-salt electrostatics needs more context than a 61 bp fragment carries (convergence
study), and the encoder must see whatever sequence the label depends on, so those labels
are computed and trained on windows of 61 + 2F bp centred on the fragment. Panel fragments
take their flanks from the genome they were sampled from (`sample_fragments.py`: 0-based
start, half-open fetch, reverse complement for the minus strand), and the window's centre
is checked against the fragment base for base; everything else (MPRA, synthetic strata)
and any genome edge gets uniform random flanks seeded by the fragment id, which are part
of the window the encoder sees, so the label never depends on unseen sequence.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random

import h5py

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.upper().translate(COMP)[::-1]


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def random_flank(frag_id: str, side: str, n: int) -> str:
    rng = random.Random(int(hashlib.sha1(f"{frag_id}|{side}".encode()).hexdigest()[:12], 16))
    return "".join(rng.choice("ACGT") for _ in range(n))


def fill_non_acgt(frag_id: str, left: str, right: str):
    """Genomic flanks can carry assembly gaps (N) and IUPAC codes, which no structure builder
    accepts; each such position takes a base from the fragment's seeded random flank (the
    policy already used for flanks the genome cannot supply). Returns (left, right, n_filled)."""
    rl, rr = random_flank(frag_id, "L", len(left)), random_flank(frag_id, "R", len(right))
    fl = "".join(c if c in "ACGT" else rl[i] for i, c in enumerate(left.upper()))
    fr = "".join(c if c in "ACGT" else rr[i] for i, c in enumerate(right.upper()))
    n = sum(1 for c in left.upper() + right.upper() if c not in "ACGT")
    return fl, fr, n


def genome_paths(manifest: str, fa_dir: str) -> dict:
    d = json.load(open(manifest))
    files = os.listdir(fa_dir)
    out = {}
    for g in d["genomes"]:
        hit = [f for f in files if f.startswith(g["accession"]) and f.endswith(".fa")]
        if hit:
            out[g["species"]] = os.path.join(fa_dir, hit[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--library", required=True, help="library TSV with panel coordinates")
    ap.add_argument("--manifest", required=True); ap.add_argument("--fa-dir", required=True)
    ap.add_argument("--flank", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyfaidx import Fasta
    lib = {r["frag_id"]: r for r in csv.DictReader(open(a.library), delimiter="\t")}
    gp = genome_paths(a.manifest, a.fa_dir)
    fas = {}
    F = a.flank
    n = stats_genome = stats_random = stats_edge = mismatch = stats_fill = 0
    with open(a.out, "w") as out:
        out.write("frag_id\tstratum\twindow\tflank_source\n")
        for p in sorted(glob.glob(a.shards)):
            with h5py.File(p, "r") as h:
                ids, seqs, st = _decode(h["frag_id"][:]), _decode(h["sequence"][:]), _decode(h["stratum"][:])
            for fid, s, t in zip(ids, seqs, st):
                r = lib.get(fid)
                src = "random"
                left, right = random_flank(fid, "L", F), random_flank(fid, "R", F)
                if r is not None and t == "panel" and r.get("source") in gp and r.get("contig"):
                    fa = fas.setdefault(r["source"], Fasta(gp[r["source"]], sequence_always_upper=True))
                    c, st0, L = r["contig"], int(r["start"]), len(s)
                    clen = len(fa[c])
                    lo, hi = max(0, st0 - F), min(clen, st0 + L + F)
                    g = str(fa[c][lo:hi])
                    gl = random_flank(fid, "L", F - (st0 - lo)) + g if st0 - lo < F else g
                    gl = gl + random_flank(fid, "R", F - (hi - st0 - L)) if hi - st0 - L < F else gl
                    win = gl if r["strand"] == "+" else revcomp(gl)
                    if win[F:F + L] != s:
                        mismatch += 1
                        src = "random"          # never use a window whose centre disagrees
                    else:
                        left, right = win[:F], win[F + L:]
                        src = "genome" if (st0 - lo == F and hi - st0 - L == F) else "genome+random_edge"
                        stats_genome += src == "genome"; stats_edge += src != "genome"
                stats_random += src == "random"
                left, right, filled = fill_non_acgt(fid, left, right)
                if filled:
                    src += "+random_fill"; stats_fill += 1
                out.write(f"{fid}\t{t}\t{left}{s}{right}\t{src}\n"); n += 1
                if a.limit and n >= a.limit:
                    break
            if a.limit and n >= a.limit:
                break
    print(f"{n} windows of {61 + 2 * F} bp -> {a.out}: genome {stats_genome}, genome with random edge "
          f"{stats_edge}, random {stats_random}; non-ACGT flank positions filled in {stats_fill}; "
          f"panel centre mismatches {mismatch}")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
