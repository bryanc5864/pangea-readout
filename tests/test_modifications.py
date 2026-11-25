#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family H: modification response, checked against measurement and against the model.

Two kinds of check. The first is that the computed response points the way measurement
says it does: CpG methylation stiffens DNA, and hydroxymethylation stiffens it less,
because the hydroxyl hydrogen-bonds to water and partially restores flexibility. The
second is that the comparison is *clean* -- the modified and unmodified sequences are run
through the same parameter set, so a reported delta is the modification and not a change
of parameter set.

The test that matters most is `test_recomputing_the_ground_state_would_use_the_wrong_set`.
`modification_response` originally obtained Family D's mechanics by calling
`analytic_targets.cgna_ground_state`, which defaults to the canonical parameter set. On a
modified sequence that raises `KeyError: 'TM'`, which is how it was found -- but on an
unmodified sequence it would have succeeded while silently comparing two different
parameter sets, and the reported "modification response" would have been partly the
difference between parameter sets.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

import analytic_targets as at  # noqa: E402
import geometry as G  # noqa: E402
import modifications as H  # noqa: E402
from channels import REGISTRY, Parity, Resolution, reverse_complement  # noqa: E402

# CpG-rich, so there is something to modify.
SEQ = "GCGCAATTCGCGAATTGCGCAATTCGCGAATTGCGCAATTCGCGAATTGCGCAATTCGCGA"[:61]
HALF = "GATCCGTAAGCTTGAC"
PALINDROME = HALF + reverse_complement(HALF)


@pytest.fixture(scope="module")
def response():
    return H.modification_response(SEQ)


@pytest.fixture(scope="module")
def channels(response):
    return H.modification_channels(SEQ, response=response)


# ======================================================================================
# What exists and what does not
# ======================================================================================

def test_only_the_modifications_the_parameter_set_covers_are_declared():
    """5fC, 5caC and N6-methyladenine must not appear as channels.

    The cgNA+ modified-base alphabet is ACGT plus M/N and H/K. Declaring a channel for a
    modification with no parameters would put a column in the training set that no run
    could fill.
    """
    names = [s.name for s in REGISTRY.by_family("H")]
    for absent in H.UNAVAILABLE_MODIFICATIONS:
        assert not any(absent in n for n in names), absent
    for present in ("5mC", "5hmC"):
        assert any(present in n for n in names), present


def test_the_parameter_set_alphabet_is_what_this_family_assumes():
    """Read the alphabet out of the shipped file rather than trusting the docstring."""
    import scipy.io
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams.cgnaplus_params import CGNAPLUSPARAMS_PARAMSPATH

    d = scipy.io.loadmat(CGNAPLUSPARAMS_PARAMSPATH + G.PARAM_SET_METHYL + ".mat")
    letters = sorted({c for name in d["sigma_int"].dtype.names for c in name})
    assert letters == ["A", "C", "G", "H", "K", "M", "N", "T"], letters


def test_unavailable_observables_are_named_with_reasons():
    for key, reason in H.UNAVAILABLE_OBSERVABLES.items():
        assert len(reason) > 10, key


def test_channel_count_is_what_the_gap_implies():
    n = len(REGISTRY.by_family("H"))
    assert n == 38, (
        f"Family H has {n} channels; the plan budgets 60 and the shortfall is accounted "
        "for in UNAVAILABLE_MODIFICATIONS and UNAVAILABLE_OBSERVABLES. If a modification "
        "or observable has become available this bound is stale.")


def test_every_emitted_channel_is_declared(response, channels):
    emitted = set(H.modification_profiles(SEQ, response=response)) | set(channels)
    declared = {s.name for s in REGISTRY.by_family("H")}
    assert emitted == declared, (
        f"undeclared: {sorted(emitted - declared)}; "
        f"unemitted: {sorted(declared - emitted)}")


# ======================================================================================
# The comparison has to be clean
# ======================================================================================

def test_the_modified_parameter_set_reproduces_the_canonical_one(response):
    """Unmodified sequence must give identical results in either parameter set.

    This is what makes a delta attributable to the modification. Without it, the
    "modification response" would include the difference between parameter sets.
    """
    a = at.cgna_ground_state(SEQ, include_stiffness=False)
    b = at.cgna_ground_state(SEQ, include_stiffness=False,
                             parameter_set=G.PARAM_SET_METHYL)
    np.testing.assert_allclose(np.asarray(a["gs"]), np.asarray(b["gs"]), atol=1e-12)


def test_recomputing_the_ground_state_would_use_the_wrong_set():
    """Regression: the default parameter set cannot represent a modified sequence.

    It raises rather than returning something wrong, which is the good case. The danger
    was the reverse direction -- computing the *baseline* with the default set and the
    modified sequence with the modified set -- so this pins that the default set really
    does reject the modified alphabet, and the module reuses the state it already has.
    """
    modified = H.MODIFICATIONS[0].apply(SEQ)
    assert modified != SEQ
    with pytest.raises(Exception):
        at.cgna_ground_state(modified, include_stiffness=False)
    # ... and succeeds when told which set to use.
    g = at.cgna_ground_state(modified, include_stiffness=False,
                             parameter_set=G.PARAM_SET_METHYL)
    assert np.isfinite(np.asarray(g["gs"])).all()


def test_terminal_sites_are_left_unmodified():
    """The parameter set's end blocks cover only the 16 unmodified dinucleotides."""
    m = H.MODIFICATIONS[0].apply("CGCGAAATTTCGCG")
    assert m.startswith("CG") and m.endswith("CG"), m
    assert "M" in m


def test_a_sequence_with_no_sites_gives_zero_not_nan():
    """No eligible site is a fact about the sequence, not missing data."""
    r = H.modification_response("AAAATTTTAAAATTTTAAAATTTT")
    c = H.modification_channels("AAAATTTTAAAATTTTAAAATTTT", response=r)
    assert c["H_5mC_n_sites"] == 0.0
    assert c["H_5mC_dpersistence_length_nm"] == 0.0
    assert np.all(np.asarray(r["5mC"]["local"]["twist_deg"]) == 0.0)


# ======================================================================================
# Direction and scale, against measurement
# ======================================================================================

def test_methylation_stiffens_dna(channels):
    """CpG methylation increases the persistence length. Perez et al. 2012."""
    assert channels["H_5mC_dpersistence_length_nm"] > 0.0
    assert channels["H_5hmC_dpersistence_length_nm"] > 0.0


def test_hydroxymethylation_stiffens_less_than_methylation(channels):
    """The hydroxyl hydrogen-bonds to water and partially restores flexibility.

    Recorded as a directional comparison rather than an absolute, because the absolute
    values carry the parameter set's own uncertainty while the ordering is the physical
    statement.
    """
    assert (channels["H_5hmC_dpersistence_length_nm"]
            < channels["H_5mC_dpersistence_length_nm"])


def test_methylation_lowers_twist_and_widens_the_minor_groove(channels):
    for mod in ("5mC", "5hmC"):
        assert channels[f"H_{mod}_mean_dtwist_deg"] < 0.0, mod
        assert channels[f"H_{mod}_mean_dminor_width_A"] > 0.0, mod


def test_the_response_is_small(channels):
    """A methyl group is a small perturbation; a large delta means something is wrong."""
    for mod in ("5mC", "5hmC"):
        assert abs(channels[f"H_{mod}_mean_dtwist_deg"]) < 3.0, mod
        assert abs(channels[f"H_{mod}_mean_dminor_width_A"]) < 1.0, mod
        assert abs(channels[f"H_{mod}_dpersistence_length_nm"]) < 20.0, mod


def test_more_sites_give_a_larger_response():
    """Dose dependence: a sequence with more CpGs must respond more."""
    few = "AAAATTTTCGAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAA"[:61]
    many = "AACGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCAA"[:61]
    cf = H.modification_channels(few)
    cm = H.modification_channels(many)
    assert cm["H_5mC_n_sites"] > cf["H_5mC_n_sites"]
    assert abs(cm["H_5mC_mean_dtwist_deg"]) > abs(cf["H_5mC_mean_dtwist_deg"])


# ======================================================================================
# Parity, measured on an exact palindrome
# ======================================================================================

def _reflect(v, shift=0):
    n = len(v)
    idx = np.arange(n)[::-1] + shift
    ok = (idx >= 0) & (idx < n)
    out = np.full(n, np.nan)
    out[ok] = np.asarray(v, dtype=float)[idx[ok]]
    return out


def _residual(v, target):
    v = np.asarray(v, dtype=float)
    m = np.isfinite(v) & np.isfinite(target)
    if m.sum() < 4:
        return 0.0
    return float(np.abs(v[m] - target[m]).max() / max(np.abs(v[m]).max(), 1e-12))


def test_a_modified_palindrome_is_still_a_palindrome():
    """CpG is self-complementary and M pairs with N, so the symmetry survives.

    This is why the deltas have a parity at all; if modifying broke the palindrome there
    would be nothing to measure.
    """
    for mod in H.MODIFICATIONS:
        m = mod.apply(PALINDROME)
        table = str.maketrans("ACGTMNHK", "TGCANMKH")
        assert m == m.translate(table)[::-1], (mod.name, m)


@pytest.mark.parametrize("mod", ["5mC", "5hmC"])
def test_step_deltas_are_symmetric_on_a_palindrome(mod):
    prof = H.modification_profiles(PALINDROME)
    for key in ("twist_deg", "roll_deg", "rise_A"):
        name = f"H_{mod}_d{key}"
        assert REGISTRY[name].parity is Parity.SYMMETRIC
        v = prof[name]
        assert _residual(v, _reflect(v, -1)) < 1e-9, name


@pytest.mark.parametrize("mod", ["5mC", "5hmC"])
def test_groove_deltas_are_symmetric_under_their_own_offset(mod):
    """SYMMETRIC under an index map whose shift is the groove's own phosphate offset."""
    prof = H.modification_profiles(PALINDROME)
    for key, shift in (("minor_width_A", -G.MINOR_OFFSET),
                       ("major_width_A", -G.MAJOR_OFFSET)):
        name = f"H_{mod}_d{key}"
        assert REGISTRY[name].parity is Parity.SYMMETRIC
        assert REGISTRY[name].rc_shift == shift
        v = prof[name]
        assert _residual(v, _reflect(v, shift)) < 1e-9, name
        # And it genuinely needs the shift, or COMPOSITE would be the wrong call.
        assert _residual(v, _reflect(v, 0)) > 1e-6, name
