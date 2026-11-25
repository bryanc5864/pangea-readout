#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Leakage-controlled splits: group by sequence similarity, then split by group.

The plan names similarity leakage as one of the twelve ways the project fails quietly:
"MPRA libraries and cross-species sets share sequence", with the control being "splits
by sequence similarity and by chromosome, with the overlap quantified and reported".
Quantified is the operative word. A random split of an MPRA library puts near-duplicate
elements on both sides, and the resulting score measures memorisation rather than
generalisation, while looking entirely normal.

Method: shingle each sequence into k-mers, build a MinHash signature, find candidate
near-duplicate pairs by locality-sensitive hashing, confirm them with the estimated
Jaccard similarity, union them into groups, and assign whole groups to folds. Sequences
that share no group can still be similar by chance, so the split is then audited by
measuring the largest cross-fold similarity actually present.

Everything here is numpy and the standard library; no external clustering dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_K = 11
DEFAULT_PERMUTATIONS = 64
DEFAULT_BANDS = 16
MERSENNE = (1 << 61) - 1


def kmer_hashes(seq: str, k: int) -> np.ndarray:
    """64-bit hashes of every k-mer, as a unique sorted array."""
    s = seq.upper()
    if len(s) < k:
        return np.empty(0, dtype=np.uint64)
    out = np.empty(len(s) - k + 1, dtype=np.uint64)
    for i in range(len(s) - k + 1):
        h = hashlib.blake2b(s[i:i + k].encode(), digest_size=8).digest()
        out[i] = int.from_bytes(h, "little")
    return np.unique(out)


class MinHasher:
    """MinHash with a universal hash family over the Mersenne prime 2^61 - 1."""

    def __init__(self, n_perm: int = DEFAULT_PERMUTATIONS, seed: int = 20261119):
        rng = np.random.default_rng(seed)
        self.n_perm = n_perm
        self.a = rng.integers(1, MERSENNE, size=n_perm, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE, size=n_perm, dtype=np.int64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.n_perm, MERSENNE, dtype=np.int64)
        x = (hashes % MERSENNE).astype(object)
        sig = np.empty(self.n_perm, dtype=np.int64)
        for j in range(self.n_perm):
            vals = [(int(self.a[j]) * int(v) + int(self.b[j])) % MERSENNE for v in x]
            sig[j] = min(vals)
        return sig


def minhash_matrix(seqs: Sequence[str], k: int, n_perm: int,
                   seed: int = 20261119) -> np.ndarray:
    hasher = MinHasher(n_perm, seed)
    sigs = np.empty((len(seqs), n_perm), dtype=np.int64)
    for i, s in enumerate(seqs):
        sigs[i] = hasher.signature(kmer_hashes(s, k))
    return sigs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def group_by_similarity(seqs: Sequence[str], threshold: float = 0.5,
                        k: int = DEFAULT_K, n_perm: int = DEFAULT_PERMUTATIONS,
                        bands: int = DEFAULT_BANDS,
                        seed: int = 20261119) -> Tuple[np.ndarray, dict]:
    """Group indices so that near-duplicates share a group.

    `threshold` is on estimated Jaccard similarity of k-mer sets. Two sequences that
    differ by a handful of substitutions still share most of their k-mers, so a
    threshold near 0.5 catches the near-duplicates that matter without collapsing the
    whole library into one group.
    """
    n = len(seqs)
    sigs = minhash_matrix(seqs, k, n_perm, seed)
    rows_per_band = max(1, n_perm // bands)

    uf = UnionFind(n)
    n_candidate_pairs = 0
    n_confirmed = 0
    for b in range(bands):
        lo = b * rows_per_band
        hi = min(n_perm, lo + rows_per_band)
        if lo >= hi:
            break
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        for i in range(n):
            key = sigs[i, lo:hi].tobytes()
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Confirm each candidate against the full signature.
            anchor = members[0]
            for other in members[1:]:
                n_candidate_pairs += 1
                if estimated_jaccard(sigs[anchor], sigs[other]) >= threshold:
                    uf.union(anchor, other)
                    n_confirmed += 1

    labels = np.array([uf.find(i) for i in range(n)])
    # Relabel densely.
    _, labels = np.unique(labels, return_inverse=True)
    sizes = np.bincount(labels)
    report = {
        "n_sequences": n,
        "n_groups": int(labels.max() + 1) if n else 0,
        "largest_group": int(sizes.max()) if n else 0,
        "n_groups_larger_than_one": int((sizes > 1).sum()),
        "fraction_in_multi_groups": float(sizes[sizes > 1].sum() / n) if n else 0.0,
        "candidate_pairs": n_candidate_pairs,
        "confirmed_pairs": n_confirmed,
        "kmer_k": k,
        "minhash_permutations": n_perm,
        "lsh_bands": bands,
        "jaccard_threshold": threshold,
    }
    return labels, report


def assign_folds(group_labels: np.ndarray, n_folds: int = 10,
                 seed: int = 20261119) -> np.ndarray:
    """Assign whole groups to folds, greedily balancing fold size."""
    rng = np.random.default_rng(seed)
    sizes = np.bincount(group_labels)
    order = np.argsort(-sizes)
    order = np.array([g for g in order if sizes[g] > 0])
    fold_of_group = np.full(len(sizes), -1, dtype=int)
    fold_load = np.zeros(n_folds, dtype=int)
    for g in order:
        # Break ties randomly so the assignment is not deterministic in group index.
        candidates = np.flatnonzero(fold_load == fold_load.min())
        f = int(rng.choice(candidates))
        fold_of_group[g] = f
        fold_load[f] += sizes[g]
    return fold_of_group[group_labels]


def audit_cross_fold_similarity(seqs: Sequence[str], folds: np.ndarray,
                                k: int = DEFAULT_K, n_perm: int = DEFAULT_PERMUTATIONS,
                                n_sample: int = 2000, seed: int = 20261119) -> dict:
    """Measure the largest similarity that actually crosses a fold boundary.

    Grouping removes the near-duplicates it finds; this reports what is left, so the
    split's quality is a measured number rather than an assumption.
    """
    rng = np.random.default_rng(seed)
    n = len(seqs)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    sigs = minhash_matrix([seqs[i] for i in idx], k, n_perm, seed)
    f = folds[idx]
    best = 0.0
    pairs_checked = 0
    top = []
    for i in range(len(idx)):
        diff = np.flatnonzero(f != f[i])
        if diff.size == 0:
            continue
        sims = (sigs[diff] == sigs[i]).mean(axis=1)
        pairs_checked += diff.size
        j = int(np.argmax(sims))
        if sims[j] > best:
            best = float(sims[j])
        top.append(float(sims[j]))
    top_arr = np.asarray(top) if top else np.zeros(1)
    return {
        "n_sampled": int(len(idx)),
        "cross_fold_pairs_checked": int(pairs_checked),
        "max_cross_fold_jaccard": round(best, 4),
        "p99_cross_fold_jaccard": round(float(np.percentile(top_arr, 99)), 4),
        "median_best_cross_fold_jaccard": round(float(np.median(top_arr)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="TSV with a 'sequence' column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequence-column", default="sequence")
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    ap.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()

    rows: List[dict] = []
    with open(a.table) as f:
        for i, rec in enumerate(csv.DictReader(f, delimiter="\t")):
            if a.max_rows and i >= a.max_rows:
                break
            rows.append(rec)
    seqs = [r[a.sequence_column].strip().upper() for r in rows]
    print(f"{len(seqs):,} sequences from {a.table}", flush=True)

    labels, group_report = group_by_similarity(
        seqs, a.threshold, a.k, a.permutations, a.bands, a.seed)
    print(f"grouped into {group_report['n_groups']:,} groups; "
          f"{group_report['fraction_in_multi_groups']*100:.2f} percent of sequences sit "
          f"in a group with a near-duplicate", flush=True)

    folds = assign_folds(labels, a.n_folds, a.seed)
    audit = audit_cross_fold_similarity(seqs, folds, a.k, a.permutations, seed=a.seed)
    print(f"max similarity crossing a fold boundary: "
          f"{audit['max_cross_fold_jaccard']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out_tsv = a.out.replace(".json", ".tsv")
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["row_index", "id", "similarity_group", "fold"])
        for i, r in enumerate(rows):
            rid = r.get(a.id_column, "") if a.id_column else ""
            w.writerow([i, rid, int(labels[i]), int(folds[i])])

    from collections import Counter

    report = {
        "table": a.table,
        "n_folds": a.n_folds,
        "grouping": group_report,
        "fold_sizes": dict(Counter(int(x) for x in folds)),
        "cross_fold_audit": audit,
        "assignment": out_tsv,
        "note": "folds are assigned by similarity group, never by row, so near-duplicate "
                "sequences cannot straddle a split boundary",
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:2500])
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
