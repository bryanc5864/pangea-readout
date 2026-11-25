#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Tests for leakage-controlled splitting.

A split that leaks produces better numbers, not an error, so these tests are built
around a construction where the right answer is known: planted near-duplicates must end
up in the same group and therefore the same fold, and unrelated sequences must not be
merged.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# src/model/data.py (a regular module) would shadow the src/data namespace package
_MODEL_PATHS = [p for p in sys.path if os.path.normpath(p).endswith(os.path.join("src", "model"))]
for _p in _MODEL_PATHS:
    sys.path.remove(_p)
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data.splits import (  # noqa: E402
    assign_folds, audit_cross_fold_similarity, estimated_jaccard, group_by_similarity,
    kmer_hashes, minhash_matrix,
)
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
sys.path[:0] = _MODEL_PATHS


def planted(n_groups: int = 60, copies: int = 2, length: int = 200,
            n_subs: int = 3, seed: int = 0):
    rng = random.Random(seed)
    seqs, truth = [], []
    for g in range(n_groups):
        base = "".join(rng.choice("ACGT") for _ in range(length))
        seqs.append(base)
        truth.append(g)
        for _ in range(copies):
            s = list(base)
            for p in rng.sample(range(length), n_subs):
                s[p] = rng.choice("ACGT")
            seqs.append("".join(s))
            truth.append(g)
    return seqs, np.array(truth)


def test_kmer_hashes_are_unique_and_sized():
    seq = "ACGTACGTACGTACGT"
    h = kmer_hashes(seq, 11)
    assert h.size == len(set(h.tolist()))
    assert h.size <= len(seq) - 11 + 1
    assert kmer_hashes("ACGT", 11).size == 0


def test_minhash_identical_sequences_match_exactly():
    seqs = ["ACGTTGCAACGTTGCAACGTTGCA"] * 2
    sig = minhash_matrix(seqs, k=11, n_perm=32)
    assert estimated_jaccard(sig[0], sig[1]) == 1.0


def test_minhash_unrelated_sequences_are_dissimilar():
    rng = random.Random(1)
    a = "".join(rng.choice("ACGT") for _ in range(300))
    b = "".join(rng.choice("ACGT") for _ in range(300))
    sig = minhash_matrix([a, b], k=11, n_perm=64)
    assert estimated_jaccard(sig[0], sig[1]) < 0.1


def test_planted_near_duplicates_are_grouped():
    seqs, truth = planted()
    labels, report = group_by_similarity(seqs, threshold=0.5)
    intact = sum(len(set(labels[truth == g])) == 1 for g in range(truth.max() + 1))
    assert intact == truth.max() + 1, f"only {intact} groups recovered intact"
    assert report["n_groups"] == truth.max() + 1


def test_unrelated_sequences_are_not_merged():
    seqs, truth = planted()
    labels, _ = group_by_similarity(seqs, threshold=0.5)
    merged = sum(1 for lab in set(labels.tolist())
                 if len(set(truth[labels == lab])) > 1)
    assert merged == 0


def test_groups_do_not_straddle_folds():
    """The property that actually prevents leakage."""
    seqs, truth = planted()
    labels, _ = group_by_similarity(seqs, threshold=0.5)
    folds = assign_folds(labels, n_folds=5)
    for g in range(truth.max() + 1):
        assert len(set(folds[truth == g])) == 1, f"group {g} straddles folds"


def test_fold_sizes_are_balanced():
    seqs, _ = planted(n_groups=100)
    labels, _ = group_by_similarity(seqs, threshold=0.5)
    folds = assign_folds(labels, n_folds=5)
    counts = Counter(folds.tolist())
    assert len(counts) == 5
    lo, hi = min(counts.values()), max(counts.values())
    assert hi - lo <= max(4, 0.2 * hi), counts


def test_cross_fold_audit_reports_low_similarity_after_grouping():
    seqs, _ = planted()
    labels, _ = group_by_similarity(seqs, threshold=0.5)
    folds = assign_folds(labels, n_folds=5)
    audit = audit_cross_fold_similarity(seqs, folds, n_sample=200)
    assert audit["max_cross_fold_jaccard"] < 0.3
    assert audit["cross_fold_pairs_checked"] > 0


def test_random_split_would_leak():
    """Control: the same data split at random does put near-duplicates on both sides.

    This is what the grouped split prevents, and stating it as a test keeps the
    justification for the extra machinery in the repository rather than in a comment.
    """
    seqs, truth = planted()
    rng = np.random.default_rng(0)
    random_folds = rng.integers(0, 5, size=len(seqs))
    straddling = sum(1 for g in range(truth.max() + 1)
                     if len(set(random_folds[truth == g].tolist())) > 1)
    assert straddling > 0.5 * (truth.max() + 1), (
        "random split unexpectedly kept duplicate groups together")
