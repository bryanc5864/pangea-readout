#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""10,240 bp genomic fragments for Family W (window ladder 100-500 bp and the wide branch).

From each of the 15 panel genomes: contigs tiled into non-overlapping 10,240 bp windows (a random
phase per contig), windows with any non-ACGT base dropped, then min(PER_GENOME, all) of them drawn
at random. Small genomes contribute every tile they have (M. jannaschii 1.7 Mb has ~160).

Leakage control. A 10 kb fragment can contain a 61 bp library fragment, and Jaccard similarity on
11-mers (the main split's criterion) cannot see that: the containing window shares 51 of its ~10,000
k-mers. So every library fragment whose canonical 11-mers are at least half present in a long
fragment is listed against it (`contains`), and the trainer puts the long fragment in the same
similarity group as everything it contains, so whole groups still go to one side of every split.
Canonical (strand-independent) k-mers, because library fragments were drawn from either strand.
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

L = 10240
K = 11
MIN_SHARED = 26                                   # distinct shared 11-mers for containment (half of a 61 bp fragment's 51)
CODE = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    CODE[ord(b)] = i


def canonical_kmers(seq: str, k: int = K) -> np.ndarray:
    """Unique canonical k-mer codes (2 bits per base) of an ACGT sequence."""
    a = CODE[np.frombuffer(seq.upper().encode(), dtype=np.uint8)].astype(np.int64)
    if len(a) < k or (a == 255).any():
        good = a != 255
        if len(a) < k:
            return np.empty(0, dtype=np.int64)
    else:
        good = None
    n = len(a) - k + 1
    pw = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(a, k)
    fwd = (win * pw).sum(1)
    rcw = (3 - win)[:, ::-1]
    rev = (rcw * pw).sum(1)
    can = np.minimum(fwd, rev)
    if good is not None:
        ok = np.lib.stride_tricks.sliding_window_view(good, k).all(1)
        can = can[ok]
    return np.unique(can[:n])


def kmer_positions(seq: str, k: int = K):
    """(forward k-mer codes, their start positions) of an ACGT sequence; windows with other bases dropped."""
    a = CODE[np.frombuffer(seq.upper().encode(), dtype=np.uint8)].astype(np.int64)
    if len(a) < k:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    pw = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(a, k)
    ok = ~(win == 255).any(1)
    codes = (win * pw).sum(1)
    pos = np.arange(len(codes))
    return codes[ok], pos[ok]


def rc_seq(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def diagonal_support(frag: str, tile_codes_sorted, tile_pos_sorted) -> int:
    """Largest number of the fragment's k-mers that sit on one alignment diagonal of the tile
    (diagonals within +-1 pooled), over both orientations: a contiguous near-copy of the fragment
    in the tile, not merely its k-mers scattered through it."""
    best = 0
    for f in (frag, rc_seq(frag)):
        fc, fp = kmer_positions(f)
        lo = np.searchsorted(tile_codes_sorted, fc, "left"); hi = np.searchsorted(tile_codes_sorted, fc, "right")
        cnt = hi - lo
        if cnt.sum() == 0:
            continue
        idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        d = tile_pos_sorted[idx] - np.repeat(fp, cnt)
        u, c = np.unique(d, return_counts=True)
        # pool neighbouring diagonals (a single indel or a k-mer straddling a mismatch)
        cc = c.copy()
        if len(u) > 1:
            adj = np.diff(u) == 1
            cc[:-1] += np.where(adj, c[1:], 0); cc[1:] += np.where(adj, c[:-1], 0)
        best = max(best, int(cc.max()))
    return best


def recompute_contains(table: str, library: str, out: str) -> int:
    """Rewrite a long-fragment table's `contains` with positional containment (same tiles)."""
    csv.field_size_limit(1 << 30)
    lib = list(csv.DictReader(open(library), delimiter="\t"))
    ids = [r["frag_id"] for r in lib]
    codes, owner, nk = [], [], np.zeros(len(lib), dtype=np.int64)
    for j, r in enumerate(lib):
        c = canonical_kmers(r["sequence"])
        codes.append(c); owner.append(np.full(len(c), j, dtype=np.int64)); nk[j] = len(c)
    codes = np.concatenate(codes); owner = np.concatenate(owner)
    o = np.argsort(codes, kind="stable"); codes, owner = codes[o], owner[o]
    rows = list(csv.DictReader(open(table), delimiter="\t"))
    n_any = 0
    with open(out, "w") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["frag_id", "source", "clade", "contig", "start", "gc", "n_contained", "contains", "sequence"])
        for k, r in enumerate(rows):
            w = r["sequence"]
            u = canonical_kmers(w)
            lo = np.searchsorted(codes, u, "left"); hi = np.searchsorted(codes, u, "right")
            cnt = hi - lo
            idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            hits = np.bincount(owner[idx], minlength=len(lib)) if len(idx) else np.zeros(len(lib), dtype=np.int64)
            cand = np.flatnonzero((hits * 2 >= nk) & (hits >= MIN_SHARED))
            tc, tp = kmer_positions(w)
            oo = np.argsort(tc, kind="stable"); tc, tp = tc[oo], tp[oo]
            cont = [j for j in cand if diagonal_support(lib[j]["sequence"], tc, tp) >= MIN_SHARED]
            n_any += bool(cont)
            wr.writerow([r["frag_id"], r["source"], r["clade"], r["contig"], r["start"], r["gc"], len(cont),
                         ";".join(ids[j] for j in cont), w])
            if k % 2000 == 0:
                print(k, "tiles; with contained so far", n_any, flush=True)
    rep = {"n_fragments": len(rows), "with_contained_library_fragments": n_any, "k": K, "min_shared_kmers": MIN_SHARED,
           "criterion": "candidates by k-mer set (>= half of the fragment's distinct canonical 11-mers and >= 26), "
                        "kept when >= 26 of its k-mers lie on one alignment diagonal (+-1) of the tile, either strand"}
    json.dump(rep, open(out + ".report.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="/home/bjc75/scratch/pangea/genomes/fa/panel_prepared.json")
    ap.add_argument("--library", default="/home/bjc75/scratch/pangea/fragments/library_v2.tsv")
    ap.add_argument("--per-genome", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recompute-contains", default=None, help="existing table: rewrite only its `contains` column")
    a = ap.parse_args()
    if a.recompute_contains:
        return recompute_contains(a.recompute_contains, a.library, a.out)
    rng = random.Random(a.seed)
    # library k-mer index: sorted canonical codes with their fragment, and each fragment's k-mer count
    lib = list(csv.DictReader(open(a.library), delimiter="\t"))
    ids = [r["frag_id"] for r in lib]
    codes, owner, nk = [], [], np.zeros(len(lib), dtype=np.int64)
    for j, r in enumerate(lib):
        c = canonical_kmers(r["sequence"])
        codes.append(c); owner.append(np.full(len(c), j, dtype=np.int64)); nk[j] = len(c)
    codes = np.concatenate(codes); owner = np.concatenate(owner)
    o = np.argsort(codes, kind="stable"); codes, owner = codes[o], owner[o]
    print(f"library index: {len(lib)} fragments, {len(codes)} canonical {K}-mers", flush=True)
    panel = json.load(open(a.panel))
    rows, per = [], {}
    for g in panel["genomes"]:
        tiles = []                                   # (contig, start) only: sequences re-read for the chosen
        with Fasta(g["fasta"]) as fa:
            for c in fa.names:
                n = fa.length(c)
                if n < L:
                    continue
                s = fa.fetch(c, 0, n).upper()
                ph = rng.randrange(0, (n % L) + 1)
                for st in range(ph, n - L + 1, L):
                    w = s[st:st + L]
                    if w.count("A") + w.count("C") + w.count("G") + w.count("T") == L:
                        tiles.append((c, st))
                del s
            rng.shuffle(tiles)
            take = [(c, st, fa.fetch(c, st, st + L).upper()) for c, st in tiles[:a.per_genome]]
        per[g["species"]] = {"tiles": len(tiles), "taken": len(take)}
        print(f"  {g['species']:32s} tiles {len(tiles):8d}  taken {len(take):5d}", flush=True)
        for c, st, w in take:
            u = canonical_kmers(w)
            lo = np.searchsorted(codes, u, "left"); hi = np.searchsorted(codes, u, "right")
            cnt = hi - lo
            idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            hits = np.bincount(owner[idx], minlength=len(lib)) if len(idx) else np.zeros(len(lib), dtype=np.int64)
            # contained: at least half of the fragment's distinct k-mers AND at least MIN_SHARED of them
            # (a homopolymer or dinucleotide-repeat fragment has one or two distinct 11-mers, and any
            # tile with that repeat would otherwise "contain" it, chaining every tile into one group)
            cont = np.flatnonzero((hits * 2 >= nk) & (hits >= MIN_SHARED))
            rows.append({"source": g["species"], "clade": g["clade"], "contig": c, "start": st, "sequence": w,
                         "gc": round((w.count("G") + w.count("C")) / L, 4),
                         "contains": ";".join(ids[j] for j in cont)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["frag_id", "source", "clade", "contig", "start", "gc", "n_contained", "contains", "sequence"])
        for k, r in enumerate(rows):
            n_c = 0 if not r["contains"] else r["contains"].count(";") + 1
            wr.writerow([f"long_{k:06d}", r["source"], r["clade"], r["contig"], r["start"], r["gc"], n_c, r["contains"], r["sequence"]])
    rep = {"length_bp": L, "per_genome": per, "n_fragments": len(rows),
           "with_contained_library_fragments": sum(1 for r in rows if r["contains"]), "k": K, "containment": 0.5,
           "min_shared_kmers": MIN_SHARED}
    json.dump(rep, open(a.out + ".report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_genome"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
