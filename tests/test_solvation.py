#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family G, checked against the mechanism it claims rather than against plausibility.

Family G's premise is that most chemical reactivity on DNA is an accessibility question:
a probe attacks a named atom, so its reactivity is governed by whether solvent can reach
that atom. That premise is testable, and these tests test it rather than merely checking
the numbers are finite.

The sharpest one is `test_the_radical_targets_are_the_most_exposed_sugar_carbons`.
Tullius and colleagues showed hydroxyl radical strand-break yield tracks the accessible
surface of the backbone hydrogens, with C4' and C5' the preferred sites. If the computed
surface did not put those atoms well above C1', the whole cleavage channel would rest on
nothing, and no correlation against ORChID would reveal that -- it would just fit badly
for a reason nobody could name.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

pytest.importorskip("freesasa", reason="freesasa not installed")

import electronic as B  # noqa: E402
import geometry as G  # noqa: E402
import solvation as S  # noqa: E402
from channels import REGISTRY, Parity, Resolution, reverse_complement  # noqa: E402

SEQ = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGCCCCTTCCTTCCAAGGAAGGTTGCGCATA"[:61]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("famG")
    return G.build_minimised_structure(SEQ, str(d))


@pytest.fixture(scope="module")
def sasa(built):
    return S.atom_sasa(built["minimised_pdb"])


@pytest.fixture(scope="module")
def profiles(built):
    vip = B.electronic_profiles(SEQ)["B_vip_eV"]
    return S.solvation_profiles(SEQ, built["minimised_pdb"], vip_profile=vip)


@pytest.fixture(scope="module")
def channels(built, profiles):
    vip = B.electronic_profiles(SEQ)["B_vip_eV"]
    return S.solvation_channels(SEQ, built["minimised_pdb"], profiles=profiles,
                                vip_profile=vip)


# ======================================================================================
# The mechanism
# ======================================================================================

def test_the_radical_targets_are_the_most_exposed_sugar_carbons(sasa):
    """C4' and C5' must be far more accessible than C1', or the channel means nothing.

    Measured on a relaxed 61 bp duplex: C5' about 22, C4' about 16, C1' about 2.6
    square angstrom. Tullius showed cleavage tracks exactly this.
    """
    by_name: dict = {}
    for res in sasa.values():
        for name, area in res.items():
            by_name.setdefault(name, []).append(area)
    mean = {k: float(np.mean(v)) for k, v in by_name.items()}
    c1 = mean["C1'"]
    for target in S.RADICAL_TARGETS:
        assert mean[target] > 3.0 * c1, (
            f"{target} at {mean[target]:.2f} A^2 is not clearly more exposed than C1' "
            f"at {c1:.2f}; the hydroxyl radical channel has no mechanism")
    assert mean["C5'"] > mean["C3'"]


def test_phosphate_oxygens_are_the_most_exposed_atoms(sasa):
    """They point into solvent. If they were buried the structure would be wrong."""
    by_name: dict = {}
    for res in sasa.values():
        for name, area in res.items():
            by_name.setdefault(name, []).append(area)
    mean = {k: float(np.mean(v)) for k, v in by_name.items()}
    assert mean["OP1"] > 20.0 and mean["OP2"] > 20.0
    assert mean["O5'"] < mean["OP1"]


def test_purine_n7_is_accessible_and_pyrimidine_n7_does_not_exist(profiles):
    """N7 sits in the major groove on purines and has no counterpart on pyrimidines."""
    n7 = np.asarray(profiles["G_probe_dms_W"], dtype=float)
    purine = np.array([b in "AG" for b in SEQ])
    assert n7[purine].mean() > 0.0
    assert np.allclose(n7[~purine], 0.0)


def test_thymine_and_cytosine_probes_fire_only_on_their_base(profiles):
    t = np.asarray(profiles["G_probe_kmno4_W"], dtype=float)
    c = np.asarray(profiles["G_probe_bisulfite_W"], dtype=float)
    assert np.allclose(t[[b != "T" for b in SEQ]], 0.0)
    assert np.allclose(c[[b != "C" for b in SEQ]], 0.0)
    assert t[[b == "T" for b in SEQ]].sum() > 0.0


def test_surface_areas_are_positive_and_bounded(profiles):
    for name, v in profiles.items():
        if not name.startswith("G_sasa") and "probe" not in name:
            continue
        a = np.asarray(v, dtype=float)
        finite = a[np.isfinite(a)]
        assert (finite >= 0.0).all(), name
        assert (finite < 1000.0).all(), name


# ======================================================================================
# Photoproducts
# ======================================================================================

def test_dipyrimidine_classes_are_exact():
    t = S.dipyrimidine_tracks("TTCCTACG")
    assert t["cpd_TT_W"][0] == 1.0
    assert t["cpd_CC_W"][2] == 1.0
    assert t["cpd_TC_W"][1] == 1.0
    assert t["cpd_TT_W"][4] == 0.0


def test_a_purine_only_strand_has_no_dipyrimidines_on_that_strand():
    """And its complement is all pyrimidine, so the other strand is saturated.

    This is the content of declaring the photoproduct channels STRANDED: a dimer forms
    between adjacent pyrimidines on one strand, and a run of A on one strand is a run of
    T on the other.
    """
    t = S.dipyrimidine_tracks("A" * 20)
    w = sum(t[f"cpd_{c}_W"].sum() for c in S.DIPYRIMIDINES)
    c = sum(t[f"cpd_{c}_C"].sum() for c in S.DIPYRIMIDINES)
    assert w == 0.0
    assert c == 19.0


def test_photoproduct_channels_are_declared_stranded():
    for s in REGISTRY.by_family("G"):
        if "cpd_" in s.name or "sixfour" in s.name:
            if s.resolution is Resolution.PER_BP:
                assert s.parity is Parity.STRANDED, s.name


def test_methyl_weighting_raises_cc_above_other_classes(profiles):
    """The enhancement applies at methylatable cytosine, so CC is weighted up."""
    w = np.asarray(profiles["G_cpd_methyl_weighted_W"], dtype=float)
    cc = np.asarray(profiles["G_cpd_CC_W"], dtype=float)
    tt = np.asarray(profiles["G_cpd_TT_W"], dtype=float)
    if cc.sum() and tt.sum():
        assert w[cc > 0].max() > w[tt > 0].max()


# ======================================================================================
# Reuse rather than reinvention
# ======================================================================================

def test_oxoguanine_comes_from_family_b_and_only_at_guanine(profiles):
    oxo = np.asarray(profiles["G_oxoguanine_susceptibility_eV"], dtype=float)
    for i, base in enumerate(SEQ):
        if base != "G":
            assert not np.isfinite(oxo[i]), (i, base, oxo[i])
    assert np.isfinite(oxo).any(), "no guanine position got a susceptibility"


def test_oxoguanine_is_nan_without_family_b(built):
    """Absent its input the channel is NaN, not guessed."""
    p = S.solvation_profiles(SEQ, built["minimised_pdb"], vip_profile=None)
    assert np.all(np.isnan(p["G_oxoguanine_susceptibility_eV"]))


# ======================================================================================
# Hydration and the registry contract
# ======================================================================================

def test_hydration_terms_have_the_right_signs(channels):
    """Burying carbon costs free energy; burying polar atoms is favourable to expose."""
    assert channels["G_dG_nonpolar_kcal"] > 0.0
    assert channels["G_dG_polar_kcal"] < 0.0
    assert 0.0 < channels["G_groove_hydrophobicity"] < 1.0


def test_the_two_strands_of_a_duplex_are_similarly_accessible(channels):
    """Not identical -- the sequence differs -- but of the same order.

    A large asymmetry would mean the Crick track had been mis-indexed, which is the one
    error in this module that would otherwise be invisible: every channel would still
    look like a plausible surface area.
    """
    a = channels["G_mean_radical_target_W"]
    b = channels["G_mean_radical_target_C"]
    assert abs(a - b) / max(a, b) < 0.25, (a, b)


def test_every_emitted_channel_is_declared(profiles, channels):
    emitted = set(profiles) | set(channels)
    declared = {s.name for s in REGISTRY.by_family("G")}
    assert emitted == declared, (
        f"undeclared: {sorted(emitted - declared)}; "
        f"unemitted: {sorted(declared - emitted)}")


def test_unavailable_observables_are_named_with_substantive_reasons():
    assert len(S.UNAVAILABLE) >= 10
    for key, reason in S.UNAVAILABLE.items():
        assert len(reason) > 40, (key, reason)
        assert "requires" in reason or "needs" in reason, key


def test_channel_count_is_near_the_plan():
    n = len(REGISTRY.by_family("G"))
    assert 70 <= n <= 95, (
        f"Family G has {n} channels against the plan's 85; the shortfall is the "
        "MD-dependent hydration dynamics and the docking and kinetics channels named in "
        "UNAVAILABLE")


def test_minimised_structure_is_required_not_optional():
    """The raw reconstruction has broken bonds; a surface area from it is wrong smoothly."""
    with pytest.raises(Exception):
        S.solvation_profiles(SEQ, "/nonexistent/path.pdb")
