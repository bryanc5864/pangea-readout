#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Validation of the analytic thermodynamic targets.

A transcription error in a nearest-neighbour table produces numbers that look entirely
reasonable, so the table is checked three ways: against the free energies published
alongside the parameters, against the reverse-complement symmetry the model must obey
exactly, and for internal consistency between dH, dS and dG.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

from analytic_targets import (  # noqa: E402
    NN_DH, NN_DS, T37_K, gc_fraction, melting_temperature, nn_thermodynamics,
    per_step_thermodynamics, sliding_window_thermodynamics,
)
# src/model/data.py (a regular module) would shadow the src/data namespace package
_MODEL_PATHS = [p for p in sys.path if os.path.normpath(p).endswith(os.path.join("src", "model"))]
for _p in _MODEL_PATHS:
    sys.path.remove(_p)
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data.fasta import revcomp  # noqa: E402
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
sys.path[:0] = _MODEL_PATHS

# SantaLucia (1998) unified nearest-neighbour free energies at 37 C, kcal/mol.
# These are published alongside the dH and dS values this module stores, so reproducing
# them from dH and dS is an independent check on the stored table.
PUBLISHED_DG37 = {
    "AA": -1.00, "AT": -0.88, "TA": -0.58,
    "CA": -1.45, "GT": -1.44, "CT": -1.28, "GA": -1.30,
    "CG": -2.17, "GC": -2.24, "GG": -1.84,
}


@pytest.mark.parametrize("step,expected", sorted(PUBLISHED_DG37.items()))
def test_nn_step_free_energies_match_published(step, expected):
    dg = NN_DH[step] - T37_K * NN_DS[step] / 1000.0
    assert abs(dg - expected) < 0.025, f"{step}: computed {dg:.3f} vs published {expected}"


def test_reverse_complement_symmetry_is_exact():
    """A duplex and its reverse complement are the same molecule.

    The nearest-neighbour model must therefore give identical dH, dS and dG. This holds
    only if every complementary step pair carries equal parameters, so it is a direct
    check on the table's internal structure rather than on arithmetic.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(6, 40))
        seq = "".join(rng.choice(list("ACGT"), size=n))
        a = nn_thermodynamics(seq)
        b = nn_thermodynamics(revcomp(seq))
        assert abs(a["dH_kcal_mol"] - b["dH_kcal_mol"]) < 1e-9
        assert abs(a["dS_cal_mol_K"] - b["dS_cal_mol_K"]) < 1e-9
        assert abs(a["dG37_kcal_mol"] - b["dG37_kcal_mol"]) < 1e-9


def test_free_energy_internal_consistency():
    rng = np.random.default_rng(1)
    for _ in range(20):
        seq = "".join(rng.choice(list("ACGT"), size=25))
        t = nn_thermodynamics(seq)
        expect = t["dH_kcal_mol"] - T37_K * t["dS_cal_mol_K"] / 1000.0
        assert abs(t["dG37_kcal_mol"] - expect) < 1e-9


def test_per_step_sums_to_duplex_minus_initiation():
    seq = "GCGCAAATTTGCGCATATAG"
    total = nn_thermodynamics(seq)
    steps = per_step_thermodynamics(seq)
    init_dh = sum(0.1 if b in "GC" else 2.3 for b in (seq[0], seq[-1]))
    init_ds = sum(-2.8 if b in "GC" else 4.1 for b in (seq[0], seq[-1]))
    assert abs(steps["dH_kcal_mol"].sum() + init_dh - total["dH_kcal_mol"]) < 1e-9
    assert abs(steps["dS_cal_mol_K"].sum() + init_ds - total["dS_cal_mol_K"]) < 1e-9


def test_gc_rich_is_more_stable_and_melts_higher():
    at_rich = "ATATATATATATATATATAT"
    gc_rich = "GCGCGCGCGCGCGCGCGCGC"
    assert nn_thermodynamics(gc_rich)["dG37_kcal_mol"] < \
        nn_thermodynamics(at_rich)["dG37_kcal_mol"]
    tm_at = melting_temperature(at_rich, na_molar=1.0)
    tm_gc = melting_temperature(gc_rich, na_molar=1.0)
    assert tm_gc > tm_at + 20.0
    # Sanity on absolute scale for a 20-mer at 1 M sodium.
    assert 20.0 < tm_at < 70.0
    assert 70.0 < tm_gc < 110.0


def test_salt_correction_lowers_tm():
    """Lower sodium destabilises the duplex, so Tm must fall monotonically."""
    seq = "GCGCAAATTTGCGCATATAG"
    tms = [melting_temperature(seq, na_molar=c) for c in (1.0, 0.5, 0.15, 0.05, 0.01)]
    assert all(tms[i] > tms[i + 1] for i in range(len(tms) - 1)), tms
    assert tms[0] - tms[2] > 2.0


def test_concentration_dependence():
    seq = "GCGCAAATTTGCGCATATAG"
    hi = melting_temperature(seq, ct_molar=1e-3)
    lo = melting_temperature(seq, ct_molar=1e-6)
    assert hi > lo


def test_sliding_window_shape_and_centres():
    seq = "".join(np.random.default_rng(2).choice(list("ACGT"), size=61))
    w = sliding_window_thermodynamics(seq, window=21, na_molar=0.15)
    assert len(w["centre"]) == 61 - 21 + 1
    assert w["centre"][0] == 10
    assert len(w["dG37_kcal_mol"]) == len(w["centre"])
    assert np.all(np.isfinite(w["dG37_kcal_mol"]))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        nn_thermodynamics("ACGTN")
    with pytest.raises(ValueError):
        nn_thermodynamics("A")


def test_gc_fraction_ignores_ambiguous():
    assert np.isclose(gc_fraction("GGCCNN"), 1.0)
    assert np.isclose(gc_fraction("ATAT"), 0.0)
    assert np.isclose(gc_fraction("ACGT"), 0.5)
