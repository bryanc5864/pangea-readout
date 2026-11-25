# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family I: ProBound protein readout, held apart."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
I = pytest.importorskip("protein_readout", reason="physics environment not active")
from channels import REGISTRY, Resolution, reverse_complement  # noqa: E402

SEQ = "GCTAGCTTGACGTCAGGATCCACGTGGCATTTACTGGGAATTCCCCAGATAAGGCTTCCGGAAGT"[:61]
CTX = {I.NUC_KEY: 30.5}


def test_declared_equals_emitted_and_exact_reverse_complement():
    p, g = I.readout_channels(SEQ, CTX)
    assert set(p) == {s.name for s in REGISTRY.by_family("I") if s.resolution is Resolution.PER_BP}
    assert set(g) == {s.name for s in REGISTRY.by_family("I") if s.resolution is Resolution.GLOBAL}
    p2, g2 = I.readout_channels(reverse_complement(SEQ), CTX)
    for k in p:
        np.testing.assert_allclose(p[k], p2[k][::-1], atol=1e-9)
    for k in g:
        assert g[k] == pytest.approx(g2[k], abs=1e-12)


def test_optimal_site_scores_zero_and_raises_occupancy():
    B = I.MODELS["MAX"]
    best = "".join("ACGT"[i] for i in B.argmax(1))
    bg = "T" * 61
    planted = bg[:20] + best + bg[20 + len(best):]
    p0, g0 = I.readout_channels(bg, CTX)
    p1, g1 = I.readout_channels(planted, CTX)
    assert np.allclose(p1["I_MAX_ddG_kT"][20:20 + len(best)], 0.0, atol=1e-9)
    assert g1["I_MAX_occ_q50"] > g0["I_MAX_occ_q50"]
    for k, v in g1.items():
        assert 0.0 <= v <= 1.0, k
    assert g1["I_MAX_occ_nuc"] < g1["I_MAX_occ_q50"]        # competition only removes occupancy


def test_nucleosome_term_needs_its_context():
    _, g = I.readout_channels(SEQ, None)
    assert all(np.isnan(v) for k, v in g.items() if k.endswith("_occ_nuc"))
