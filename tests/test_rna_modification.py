# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family N: RNA modification response."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
N = pytest.importorskip("rna_modification", reason="physics environment not active")
from channels import REGISTRY, Resolution, reverse_complement  # noqa: E402

SEQ = "GGGATACTGCAAAGCAGTATCCCAATTGGCCAATTACGATCGTAGCTAGCATCGATCGAT"


def test_emitted_equals_declared_and_strand_identity():
    p, g = N.modification_channels(SEQ)
    p2, g2 = N.modification_channels(reverse_complement(SEQ))
    assert set(p) == {s.name for s in REGISTRY.by_family("N") if s.resolution is Resolution.PER_BP}
    assert set(g) == {s.name for s in REGISTRY.by_family("N") if s.resolution is Resolution.GLOBAL}
    for k in p:
        if k.startswith("N_m_"):
            np.testing.assert_allclose(p[k], p2[k.replace("N_m_", "N_p_")], atol=1e-9)
    for k in g:
        if k.startswith("N_m_"):
            assert g[k] == pytest.approx(g2[k.replace("N_m_", "N_p_")], abs=1e-9, nan_ok=True)


def test_sites_and_published_directions():
    p, g = N.modification_channels(SEQ)
    rna = SEQ.replace("T", "U")
    a = np.array([b == "A" for b in rna]); u = np.array([b == "U" for b in rna])
    assert np.all(np.isfinite(p["N_p_m6A_ddG_ens"][a])) and np.all(np.isnan(p["N_p_m6A_ddG_ens"][~a]))
    assert np.all(np.isfinite(p["N_p_psi_ddG_ens"][u])) and np.all(np.isnan(p["N_p_psi_ddG_ens"][~u]))
    # m6A destabilises paired A (Kierzek 2022); pseudouridine stabilises paired U (Hudson 2013)
    assert g["N_p_m6A_max_ddG"] > 0.3
    assert g["N_p_psi_min_ddG"] < -0.3


def test_two_prime_o_methyl_response():
    import rna_modification as N
    per_nt, g_ome, dd = N.ome_duplex("GGCAUAGCUAUGCC")
    assert per_nt.shape == (14,) and np.isfinite(per_nt).all()
    # hand check of the 2'-O-methyl duplex free energy from Kierzek 2006 Table 3
    rna = "GGCAUAGCUAUGCC"
    want = N.OME_INIT_DG37 + sum(N.OME_NN_DG37[rna[i:i + 2]] for i in range(len(rna) - 1))
    assert abs(g_ome - want) < 1e-9                       # both ends G-C: no terminal AU
    # Kierzek 2006 compare their 0.1 M set with RNA/RNA at 1 M only and attribute the 0.26 kcal/mol
    # per-stack difference to salt; at matched 0.1 M the response is small either way (its sign
    # rests on the salt correction of the RNA/RNA reference), so only its scale is checked here
    assert abs(dd) / (len(rna) - 1) < 0.5
    p, g = N.modification_channels("ACGT" * 15 + "A")
    for tag in ("p", "m"):
        assert np.isfinite(g[f"N_{tag}_ome_duplex_ddG"]) and np.isfinite(p[f"N_{tag}_ome_ddG_step"]).all()
    # strand tie: the m transcript of x is the p transcript of rc(x)
    x = "GATTACAGGCTTACGATCGATCGGCTAAGCTTAGC"
    rc = x.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    _, gx = N.modification_channels(x); _, grc = N.modification_channels(rc)
    assert abs(gx["N_m_ome_duplex_ddG"] - grc["N_p_ome_duplex_ddG"]) < 1e-9
