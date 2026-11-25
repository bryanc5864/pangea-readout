# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family K: RNA duplex and DNA-RNA hybrid channels."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
K = pytest.importorskip("duplex_classes", reason="physics environment not active")
from channels import REGISTRY, Resolution  # noqa: E402

SEQ = "ATGCGTACGTTAGCATCGATCGGCTAGCTAGGATCCGATCGATTGCATGCAAGCTTGCATG"[:61]


@pytest.fixture(scope="module")
def both():
    return K.duplex_class_channels(SEQ), K.duplex_class_channels(K.reverse_complement(SEQ))


def test_emitted_equals_declared(both):
    (p, g), _ = both
    dp = {s.name for s in REGISTRY.by_family("K") if s.resolution is Resolution.PER_BP}
    dg = {s.name for s in REGISTRY.by_family("K") if s.resolution is Resolution.GLOBAL}
    assert set(p) == dp and set(g) == dg


def test_hybrid_strand_identity_is_exact(both):
    (p, g), (p2, g2) = both
    for k, v in p.items():
        if k.startswith("K_hybm_"):
            np.testing.assert_array_equal(v, p2[k.replace("K_hybm_", "K_hybp_")])
    for k, v in g.items():
        if k.startswith("K_hybm_"):
            assert v == g2[k.replace("K_hybm_", "K_hybp_")] or (np.isnan(v) and np.isnan(g2[k.replace("K_hybm_", "K_hybp_")]))


def test_rna_duplex_is_reverse_complement_consistent(both):
    (p, _), (p2, _) = both
    spec = {s.name: s for s in REGISTRY.by_family("K")}
    n = len(SEQ)
    for k, v in p.items():
        s = spec[k]
        if not k.startswith("K_rna_") or s.circular or s.parity.value not in ("symmetric", "antisymmetric"):
            continue
        sign = -1.0 if s.parity.value == "antisymmetric" else 1.0
        idx = np.array([n - 1 - i + s.rc_shift for i in range(n)])
        ok = (idx >= 0) & (idx < n)
        a, b = v[ok], sign * p2[k][idx[ok]]
        m = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[m], b[m], atol=1e-6, rtol=1e-6), k


def test_a_form_signatures(both):
    (p, _), _ = both
    mean = lambda k: float(np.nanmean(p[k][4:-4]))  # noqa: E731
    assert mean("K_rna_slide") < -1.0 and mean("K_rna_roll") > 5.0 and mean("K_rna_twist") < 32.0
    assert mean("K_rna_minor_width_A") > 8.0                 # wide shallow A-form minor groove


def test_hybrid_table_is_self_consistent():
    for d, (h, s, g) in K.HYB_NN.items():
        assert abs(h - 310.15 * s / 1000.0 - g) <= 0.051, d
    for h, s, g in (K.HYB_INIT_GC, K.HYB_INIT_AT):
        assert abs(h - 310.15 * s / 1000.0 - g) <= 0.051


def test_rna_duplex_free_energy_matches_vienna(both):
    import RNA
    (_, g), _ = both
    x = SEQ.replace("T", "U"); c = K.reverse_complement(SEQ).replace("T", "U")
    assert abs(RNA.fold_compound(x + "&" + c).mfe_dimer()[1] - g["K_rna_duplex_dG37_kcal"]) < 1e-6


def test_caps_commute_with_reverse_complement():
    assert K.reverse_complement(K.CAP) == K.CAP
