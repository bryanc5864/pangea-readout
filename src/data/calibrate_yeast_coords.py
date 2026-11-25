#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Calibrate the coordinate mapping of the yeast assays against sacCer3, from data.

The in vitro nucleosome occupancy map is the plan's **positive control**: a physics atlas
is expected to nearly saturate it, and if it does not, the landscape is broken and no
downstream number is trustworthy.  That makes the coordinate mapping load-bearing.  The
map was built on a May 2006 SGD assembly, between sacCer1 and sacCer2, whose chromosome
lengths differ from sacCer3 by small amounts that accumulate along each chromosome, so
pairing its coordinates with sacCer3 sequence is not automatically correct.

The map ships no sequence, so the offset cannot be found by alignment.  It can be found
by exploiting a known relationship instead: intrinsic nucleosome occupancy tracks G+C
content strongly, with GC alone explaining close to half the variance (Tillo and Hughes
2009).  Applying a trial offset to the assay coordinates, extracting sacCer3 sequence,
and measuring the GC-occupancy correlation gives a curve that peaks at the correct
alignment.  A mis-set offset degrades it.

This does two jobs at once.  It calibrates the mapping, and it independently reproduces
the GC-occupancy relationship the positive control rests on, which is worth knowing
before any model is trained.

Output: the correlation as a function of offset per chromosome, the maximising offset,
and the correlation achieved there.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fasta import Fasta  # noqa: E402

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI"]


def load_occupancy(path: str, chromosomes: Sequence[str],
                   max_rows: Optional[int] = None) -> Dict[str, Dict[int, float]]:
    """{chromosome: {position: occupancy}} from the normalised tab file."""
    want = set(chromosomes)
    out: Dict[str, Dict[int, float]] = {c: {} for c in chromosomes}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        f.readline()
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[0] not in want:
                continue
            try:
                out[p[0]][int(p[1])] = float(p[2])
            except ValueError:
                continue
    return out


def windowed_gc(fa: Fasta, contig: str, centres: np.ndarray, window: int,
                offset: int) -> np.ndarray:
    """GC fraction in a window centred on each (1-based) position plus `offset`."""
    half = window // 2
    L = fa.length(contig)
    out = np.full(len(centres), np.nan)
    # Fetch one contiguous block and slice, which is far cheaper than per-window seeks.
    lo = int(centres.min()) + offset - half - 1
    hi = int(centres.max()) + offset + half
    lo_c, hi_c = max(0, lo), min(L, hi)
    if hi_c <= lo_c:
        return out
    block = fa.fetch(contig, lo_c, hi_c).upper()
    for i, c in enumerate(centres):
        s = int(c) + offset - half - 1 - lo_c
        e = s + window
        if s < 0 or e > len(block):
            continue
        sub = block[s:e]
        g = sub.count("G") + sub.count("C")
        a = sub.count("A") + sub.count("T")
        if g + a >= window * 0.9:
            out[i] = g / (g + a)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    av, bv = a[m], b[m]
    if av.std() == 0 or bv.std() == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--occupancy", required=True)
    ap.add_argument("--genome-glob", default="/home/bjc75/scratch/pangea/genomes/fa/GCF_000146045*.fa")
    ap.add_argument("--chromosomes", nargs="*", default=["1", "6", "3", "9"],
                    help="assay chromosome labels (Arabic, Roman order)")
    ap.add_argument("--window", type=int, default=147,
                    help="nucleosome footprint; GC is averaged over this")
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--offsets", type=int, nargs="+",
                    default=list(range(-60, 61, 5)))
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    matches = sorted(glob.glob(a.genome_glob))
    if not matches:
        print(f"no genome FASTA matching {a.genome_glob}", file=sys.stderr)
        return 2
    fa_path = matches[0]

    occ = load_occupancy(a.occupancy, a.chromosomes)
    rng = np.random.default_rng(a.seed)
    report: Dict[str, object] = {
        "occupancy_file": a.occupancy,
        "genome": fa_path,
        "window": a.window,
        "rationale": "intrinsic nucleosome occupancy tracks G+C content, so the offset "
                     "that maximises the GC-occupancy correlation identifies the "
                     "coordinate alignment",
        "chromosomes": {},
    }

    with Fasta(fa_path) as fa:
        contigs = [n for n in fa.names if n.startswith("NC_")]
        for label in a.chromosomes:
            try:
                idx = int(label) - 1
            except ValueError:
                continue
            if idx < 0 or idx >= len(contigs):
                continue
            contig = contigs[idx]
            positions = np.array(sorted(occ.get(label, {}).keys()))
            if len(positions) < 1000:
                report["chromosomes"][label] = {"status": "insufficient data",
                                                "n_positions": int(len(positions))}
                continue
            # Sample interior positions so a window never runs off the end.
            half = a.window // 2 + max(abs(min(a.offsets)), abs(max(a.offsets))) + 5
            usable = positions[(positions > half) &
                               (positions < fa.length(contig) - half)]
            if len(usable) > a.n_sample:
                usable = np.sort(rng.choice(usable, a.n_sample, replace=False))
            values = np.array([occ[label][int(p)] for p in usable])

            curve = {}
            for off in a.offsets:
                gc = windowed_gc(fa, contig, usable, a.window, off)
                curve[off] = pearson(gc, values)
            finite = {k: v for k, v in curve.items() if np.isfinite(v)}
            best = max(finite, key=lambda k: abs(finite[k])) if finite else None
            report["chromosomes"][label] = {
                "status": "ok",
                "refseq_contig": contig,
                "roman": ROMAN[idx] if idx < len(ROMAN) else "?",
                "n_positions_available": int(len(positions)),
                "n_sampled": int(len(usable)),
                "contig_length": fa.length(contig),
                "max_assay_position": int(positions.max()),
                "length_minus_max": int(fa.length(contig) - positions.max()),
                "correlation_by_offset": {str(k): (None if not np.isfinite(v) else round(v, 5))
                                          for k, v in curve.items()},
                "best_offset": best,
                "best_correlation": None if best is None else round(finite[best], 5),
                "correlation_at_zero": None if not np.isfinite(curve.get(0, np.nan))
                else round(curve[0], 5),
            }
            r = report["chromosomes"][label]
            print(f"chr{r['roman']:<5s} n={r['n_sampled']:>6,}  "
                  f"r(offset=0)={r['correlation_at_zero']}  "
                  f"best offset={r['best_offset']:+d} r={r['best_correlation']}",
                  flush=True)

    ok = [v for v in report["chromosomes"].values()
          if isinstance(v, dict) and v.get("status") == "ok"]
    if ok:
        best_offsets = [v["best_offset"] for v in ok]
        report["summary"] = {
            "best_offsets": best_offsets,
            "offsets_agree": len(set(best_offsets)) == 1,
            "median_best_correlation": float(np.median([v["best_correlation"] for v in ok])),
            "median_correlation_at_zero": float(np.median(
                [v["correlation_at_zero"] for v in ok if v["correlation_at_zero"] is not None])),
        }
        s = report["summary"]
        print(f"\nbest offsets per chromosome: {best_offsets} "
              f"(agree: {s['offsets_agree']})")
        print(f"median |r| at best offset: {s['median_best_correlation']:.4f}  "
              f"at offset 0: {s['median_correlation_at_zero']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
