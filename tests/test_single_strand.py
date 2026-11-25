# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family J: single-stranded DNA self-structure."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
J = pytest.importorskip("single_strand", reason="physics environment not active")
from channels import REGISTRY, Resolution  # noqa: E402

SEQ = "GGGATCCATTGCGCAATGGATCCCAAAAAGGGATCCATTGCGCAATGGATCCCTTTTTACG"[:61]


def test_emitted_equals_declared_and_strand_identity():
    p, g = J.single_strand_channels(SEQ)
    p2, g2 = J.single_strand_channels(J.reverse_complement(SEQ))
    dp = {s.name for s in REGISTRY.by_family("J") if s.resolution is Resolution.PER_BP}
    dg = {s.name for s in REGISTRY.by_family("J") if s.resolution is Resolution.GLOBAL}
    assert set(p) == dp and set(g) == dg
    for k in p:
        if k.startswith("J_m_"):
            np.testing.assert_allclose(p[k], p2[k.replace("J_m_", "J_p_")], atol=1e-9)
    for k in g:
        if k.startswith("J_m_"):
            assert abs(g[k] - g2[k.replace("J_m_", "J_p_")]) < 1e-9


def test_dna_parameters_are_used_and_restored():
    import RNA
    rna = SEQ.replace("T", "U")
    before = RNA.fold_compound(rna).mfe()[1]
    _, g = J.single_strand_channels(SEQ)
    after = RNA.fold_compound(rna).mfe()[1]
    assert before == after                                   # RNA parameters restored
    import thermodynamics as T
    mfe, _ = T.dna_fold_mfe(SEQ, na_molar=0.15)
    assert abs(g["J_p_mfe_kcal"] - mfe) < 1e-6                # same DNA model as Family E
    assert g["J_p_mfe_kcal"] < -5.0                           # this sequence has strong hairpins
