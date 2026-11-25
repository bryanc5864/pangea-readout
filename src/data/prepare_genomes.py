#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Decompress and index the downloaded species panel for random access.

Reads genomes_manifest.json, writes one plain FASTA plus .fai per species under <out>,
and records measured GC content and assembly size per species.  The measured GC is kept
alongside the GC reported by NCBI: the universality analysis regresses per-genome
accuracy on composition, so that number has to come from the sequence we actually
sampled, not from a database field.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fasta import Fasta, decompress  # noqa: E402


def measure_gc(fa: Fasta, max_bases: int = 100_000_000, seed: int = 20261119) -> dict:
    """Base composition over the assembly.

    Large genomes are subsampled, but the subsample is drawn as random windows spread
    over every contig in proportion to contig length, never as a prefix.  Reading the
    first 50 Mb of a human assembly lands entirely inside the start of chromosome 1 and
    returns about 47 percent GC against a true value near 41 percent.  The universality
    analysis regresses per-genome accuracy on composition, so a biased estimate here
    would propagate straight into the headline claim.
    """
    import random as _random

    rng = _random.Random(seed)
    counts = {b: 0 for b in "ACGTN"}
    other = 0
    seen = 0
    total = fa.total_length()
    window = 100_000

    if total <= max_bases:
        targets = [(name, 0, fa.length(name)) for name in fa.names]
    else:
        n_windows = max(1, max_bases // window)
        names = fa.names
        weights = [fa.length(c) for c in names]
        cum, acc = [], 0
        for w in weights:
            acc += w
            cum.append(acc)
        targets = []
        for _ in range(n_windows):
            r = rng.random() * total
            idx = next(i for i, c in enumerate(cum) if r <= c)
            name = names[idx]
            L = fa.length(name)
            if L <= window:
                targets.append((name, 0, L))
            else:
                s = rng.randrange(0, L - window)
                targets.append((name, s, s + window))

    for name, s, e in targets:
        chunk = fa.fetch(name, s, e).upper()
        if not chunk:
            continue
        for b in "ACGTN":
            counts[b] += chunk.count(b)
        other += len(chunk) - sum(chunk.count(b) for b in "ACGTN")
        seen += len(chunk)

    acgt = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    return {
        "bases_counted": seen,
        "gc_measured": (counts["G"] + counts["C"]) / acgt if acgt else None,
        "n_fraction": counts["N"] / seen if seen else None,
        "other_fraction": other / seen if seen else None,
        "gc_sampling": "whole" if total <= max_bases else "random_windows_100kb",
    }


def prepare_one(g: dict, out_dir: str, overwrite: bool) -> dict:
    gz = g["path"]
    stem = os.path.basename(gz).replace("_genomic.fna.gz", "")
    fa_path = os.path.join(out_dir, f"{stem}.fa")
    t0 = time.time()
    decompress(gz, fa_path, overwrite=overwrite)
    with Fasta(fa_path) as fa:
        stats = measure_gc(fa)
        rec = {
            **{k: g[k] for k in
               ("species", "clade", "accession", "gc_percent", "total_sequence_length")},
            "fasta": fa_path,
            "fai": fa_path + ".fai",
            "n_contigs": len(fa.names),
            "indexed_length": fa.total_length(),
            "prepare_seconds": round(time.time() - t0, 1),
            **stats,
        }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    with open(a.manifest) as f:
        manifest = json.load(f)

    recs, errs = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(prepare_one, g, a.out, a.overwrite): g["species"]
                for g in manifest["genomes"]}
        for fu in concurrent.futures.as_completed(futs):
            sp = futs[fu]
            try:
                r = fu.result()
                recs.append(r)
                print(f"[ok] {sp:32s} contigs={r['n_contigs']:6d} "
                      f"len={r['indexed_length']:,} GC_measured={r['gc_measured']:.4f} "
                      f"N={r['n_fraction']:.4f} ({r['prepare_seconds']}s)", flush=True)
            except Exception as e:
                errs.append({"species": sp, "error": f"{type(e).__name__}: {e}"})
                print(f"[FAIL] {sp}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    recs.sort(key=lambda r: r["gc_measured"] if r["gc_measured"] is not None else 0)
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_species": len(recs),
        "genomes": recs,
        "errors": errs,
    }
    path = os.path.join(a.out, "panel_prepared.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")
    if recs:
        gcs = [r["gc_measured"] for r in recs if r["gc_measured"]]
        print(f"measured GC range: {min(gcs)*100:.1f}% to {max(gcs)*100:.1f}%")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
