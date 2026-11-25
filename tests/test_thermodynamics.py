#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family E, pinned against measured base-pair opening and against itself.

The nearest-neighbour half is already validated in `test_analytic.py` against
SantaLucia's published values. What is checked here is the statistical mechanics built
on top: that the partition function is internally consistent, that it reproduces what
imino exchange measures about base-pair opening, and that superhelical stress does what
superhelical stress does.

Two tests record discrepancies at their computed values rather than at the measured
ones, so the gap is a known open item instead of something a loose tolerance hides:
`test_absolute_opening_constants_sit_below_measurement` and the bubble-size caveat.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

import analytic_targets as at  # noqa: E402
import thermodynamics as E  # noqa: E402
from channels import REGISTRY, Parity, Resolution, reverse_complement  # noqa: E402

SEQ = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGCCCCTCCCTCCCTCCC"
CLAMPED = "GCGCGCGCGCAAATTTAAATTTAAAGCGCGCGCGC"
HALF = "GATCCGTAAGCTTGAC"
PALINDROME = HALF + reverse_complement(HALF)

AT_SLICE = slice(12, 22)
GC_SLICE = slice(3, 7)


# ======================================================================================
# The registry is the contract
# ======================================================================================

def test_every_emitted_channel_is_declared():
    emitted = set(E.thermo_profiles(SEQ)) | set(E.thermo_channels(SEQ))
    undeclared = sorted(n for n in emitted if n not in REGISTRY)
    assert not undeclared, f"emitted but never declared: {undeclared}"


def test_every_declared_channel_is_emitted():
    emitted = set(E.thermo_profiles(SEQ)) | set(E.thermo_channels(SEQ))
    missing = sorted({s.name for s in REGISTRY.by_family("E")} - emitted)
    assert not missing, f"declared but never emitted: {missing}"


def test_family_e_is_near_the_planned_channel_count():
    n = len(REGISTRY.by_family("E"))
    assert 78 <= n <= 95, f"PLAN.md budgets 85 channels for Family E; got {n}"


def test_provisional_channels_say_why():
    for s in REGISTRY.by_family("E"):
        if s.provisional:
            assert len(s.provisional_reason) > 40, s.name


def test_resolutions_match_where_the_channel_is_emitted():
    prof, chan = E.thermo_profiles(SEQ), E.thermo_channels(SEQ)
    for s in REGISTRY.by_family("E"):
        if s.resolution is Resolution.PER_BP:
            assert s.name in prof, s.name
        else:
            assert s.name in chan, s.name


# ======================================================================================
# Internal consistency of the partition function
# ======================================================================================

def test_supercoiled_recursion_reduces_to_the_plain_one_without_a_domain():
    """The O(N^2) stressed recursion and the O(N) plain one must agree exactly.

    They are separate implementations of the same model, so this is the test that says
    the open-base-pair counting index was introduced without changing the physics. The
    limit is K = 0, not sigma = 0: a topologically closed domain at zero linking
    difference still resists opening, because opening generates positive supercoiling.
    """
    k0 = E.BENHAM_K_RT
    try:
        E.BENHAM_K_RT = 0.0
        a = E.helix_coil(SEQ)
        b = E.helix_coil_supercoiled(SEQ, 0.0)
    finally:
        E.BENHAM_K_RT = k0
    np.testing.assert_allclose(a["open_prob"], b["open_prob"], atol=1e-14)
    assert abs(a["mean_runs"] - b["mean_runs"]) < 1e-12


def test_a_closed_domain_at_zero_linking_difference_opens_less_than_a_free_one():
    """Not a tautology: it is why the K = 0 limit above is the right one to compare to."""
    k0 = E.BENHAM_K_RT
    try:
        E.BENHAM_K_RT = 0.0
        free = E.helix_coil_supercoiled(SEQ, 0.0)["mean_open"]
    finally:
        E.BENHAM_K_RT = k0
    closed = E.helix_coil_supercoiled(SEQ, 0.0)["mean_open"]
    assert closed < free


def test_open_run_count_never_exceeds_the_open_base_pair_count():
    """An open run contains at least one open base pair, so runs <= open.

    This failed twice, each time for a different reason, and each failure was a real
    bug: first the run count came from the unconstrained model while the open count came
    from the stressed one, then the count omitted a run beginning at the first base pair.
    """
    for sc in E.SUPERHELICAL_DENSITIES:
        b = E.bubble_statistics(SEQ, sc)
        assert b["mean_runs"] <= b["mean_open_bp"] + 1e-12, sc
        assert b["mean_bubble_bp"] >= 1.0 - 1e-9, f"sigma={sc}: {b['mean_bubble_bp']}"


def test_open_count_distribution_is_normalised():
    r = E.helix_coil_supercoiled(SEQ, -0.06)
    k = np.asarray(r["k_distribution"], dtype=float)
    assert abs(k.sum() - 1.0) < 1e-9
    kk = np.arange(k.size, dtype=float)
    np.testing.assert_allclose(float((k * kk).sum()), r["mean_open"], rtol=1e-9)


def test_opening_probabilities_are_probabilities():
    for sc in E.SUPERHELICAL_DENSITIES:
        p = E.helix_coil_supercoiled(SEQ, sc)["open_prob"]
        assert np.all((p >= 0.0) & (p <= 1.0))


# ======================================================================================
# Measured base-pair opening
# ======================================================================================

def test_at_pairs_open_about_ten_times_more_readily_than_gc():
    """The sharpest measured check available: imino exchange gives a ratio near 10.

    Nothing was fitted to this. It falls out of the SantaLucia stacking parameters and
    Blake's cooperativity, so agreement validates the whole chain.
    """
    p = E.helix_coil(CLAMPED, temperature_C=37.0)["open_prob"]
    ratio = float(np.mean(p[AT_SLICE]) / np.mean(p[GC_SLICE]))
    assert 4.0 < ratio < 25.0, f"AT/GC opening ratio {ratio:.2f}, measured near 10"


def test_absolute_opening_constants_sit_below_measurement():
    """Recorded at the computed value, not the measured one. See DECISIONS.md.

    Imino exchange reports K_op near 1e-5 to 1e-6 for AT and 1e-6 to 1e-7 for GC. The
    model gives about 4.5e-7 and 4.5e-8, roughly 20 times low, because `sigma` is the
    nucleation parameter for cooperative melting while opening one base pair is a less
    disruptive event. Tuning sigma to close the gap would be fitting to the validation,
    so the gap is asserted instead.
    """
    p = E.helix_coil(CLAMPED, temperature_C=37.0)["open_prob"]
    at_op = float(np.mean(p[AT_SLICE]))
    assert 1e-7 < at_op < 2e-6, (
        f"AT opening constant {at_op:.2e} has moved; if it is now in the measured "
        "1e-5 to 1e-6 range the documented discrepancy is stale")


def test_opening_increases_with_temperature():
    prev = -1.0
    for T in E.TEMPERATURES_C:
        m = float(np.nanmean(E.helix_coil(CLAMPED, temperature_C=T)["open_prob"]))
        assert m > prev, f"opening did not increase at {T} C"
        prev = m


def test_ends_fray_more_than_the_interior():
    p = E.helix_coil(CLAMPED, temperature_C=37.0)["open_prob"]
    assert p[0] > np.mean(p[5:-5]) and p[-1] > np.mean(p[5:-5])


def test_lowering_salt_destabilises_the_duplex():
    tms = [at.melting_temperature(SEQ, na_molar=na) for na in E.IONIC_STRENGTHS_M]
    assert all(b > a for a, b in zip(tms, tms[1:])), tms


# ======================================================================================
# Superhelical stress
# ======================================================================================

def test_negative_supercoiling_opens_dna_monotonically():
    means = [E.helix_coil_supercoiled(SEQ, sc)["mean_open"]
             for sc in sorted(E.SUPERHELICAL_DENSITIES)]
    assert all(a > b for a, b in zip(means, means[1:])), means


def test_sidd_destabilises_the_at_rich_region_most():
    g = E.sidd_profile(CLAMPED, -0.06)
    assert np.nanmean(g[AT_SLICE]) < np.nanmean(g[GC_SLICE])


def test_sidd_energies_are_in_the_published_range():
    """Benham's SIDD energies for real sequence sit in the single-digit kcal/mol range."""
    g = E.sidd_profile(SEQ, -0.06)
    interior = g[5:-5]
    assert 2.0 < float(np.nanmin(interior)) < 20.0, float(np.nanmin(interior))


# ======================================================================================
# Parity, measured on an exact palindrome
# ======================================================================================

def test_every_computable_profile_is_symmetric_on_a_palindrome():
    """All of Family E is SYMMETRIC, and the transfer matrix had to be fixed to make it so.

    The original matrix charged the stacking weight on the coil-to-helix transition,
    crediting a step with stacking energy when the base pair beside it is open. Every
    whole-sequence channel stayed exactly symmetric under that error; only the
    per-base-pair profiles revealed it.
    """
    prof = E.thermo_profiles(PALINDROME)
    n = len(PALINDROME)
    failures = []
    for name, v in prof.items():
        v = np.asarray(v, dtype=float)
        ok = False
        for shift in (0, -1):
            idx = np.arange(n)[::-1] + shift
            good = (idx >= 0) & (idx < n)
            t = np.full(n, np.nan)
            t[good] = v[idx[good]]
            m = np.isfinite(v) & np.isfinite(t)
            if m.sum() < 4:
                ok = True            # all-NaN or too short to judge
                break
            if np.abs(v[m] - t[m]).max() / max(np.abs(v[m]).max(), 1e-12) < 1e-9:
                ok = True
                break
        if not ok:
            failures.append(name)
    assert not failures, f"not symmetric on a palindrome: {failures}"


def test_charging_stacking_on_nucleation_breaks_the_profile_symmetry():
    """Negative control for the transfer-matrix fix.

    Re-introduces the original error by hand and asserts the palindrome symmetry fails,
    so the corrected matrix cannot be silently reverted.
    """
    seq = PALINDROME
    w = E._weights(seq, 37.0, 0.15)
    n = len(seq)
    sigma = E.SIGMA_COOP

    def run(bad: bool):
        fwd = np.zeros((n, 2))
        fwd[0] = [1.0, 1.0] if bad else [sigma, 1.0]
        for i in range(1, n):
            si = w[i - 1]
            T = np.array([[si, 1.0], [sigma * si if bad else sigma, 1.0]])
            v = fwd[i - 1] @ T
            fwd[i] = v / v.max()
        bwd = np.zeros((n, 2))
        bwd[-1] = [1.0, 1.0]
        for i in range(n - 2, -1, -1):
            si = w[i]
            T = np.array([[si, 1.0], [sigma * si if bad else sigma, 1.0]])
            v = T @ bwd[i + 1]
            bwd[i] = v / v.max()
        post = fwd * bwd
        return post[:, 1] / post.sum(axis=1)

    good, bad = run(False), run(True)
    assert np.abs(good - good[::-1]).max() / good.max() < 1e-9
    assert np.abs(bad - bad[::-1]).max() / bad.max() > 0.5


def test_global_channels_are_reverse_complement_invariant():
    """Every SYMMETRIC Family E channel, exactly. STRANDED ones are checked below."""
    f = E.thermo_channels(SEQ)
    r = E.thermo_channels(reverse_complement(SEQ))
    for k, a in f.items():
        if REGISTRY[k].parity is not Parity.SYMMETRIC:
            continue
        b = r[k]
        if np.isfinite(a) and np.isfinite(b):
            assert abs(a - b) <= 1e-9 * max(abs(a), 1.0), f"{k}: {a} vs {b}"


def test_strand_specific_structures_really_are_strand_specific():
    """Negative control for the STRANDED declaration on the pH-dependent channels.

    An i-motif is built from one strand's cytosines, so its free energy must differ
    between a sequence and its reverse complement. If it stopped differing, the
    STRANDED declaration would be wrong and a strand-specific structure would be being
    reported as a property of the duplex.
    """
    f = E.thermo_channels(SEQ)
    r = E.thermo_channels(reverse_complement(SEQ))
    stranded = [k for k in f if REGISTRY[k].parity is Parity.STRANDED]
    assert stranded, "no STRANDED channels to check"
    differing = [k for k in stranded
                 if np.isfinite(f[k]) and np.isfinite(r[k]) and abs(f[k] - r[k]) > 1e-6]
    assert differing, f"none of {stranded} differ under reverse complement"


# ======================================================================================
# Hydrogen bonds and alternative structures
# ======================================================================================

def test_hydrogen_bond_counts_are_exact():
    c = E.hbond_counts("ATGC")
    np.testing.assert_array_equal(c, [2.0, 2.0, 3.0, 3.0])


def test_gc_rich_sequence_has_more_hydrogen_bonds():
    assert E.thermo_channels("GCGCGCGCGC")["E_hbond_total"] > \
           E.thermo_channels("ATATATATAT")["E_hbond_total"]


def test_imotif_becomes_more_favourable_at_acidic_ph():
    dgs = [E.ph_dependent_free_energies(SEQ, ph)["imotif_dG_kcal"]
           for ph in sorted(E.PH_VALUES)]
    assert all(a < b for a, b in zip(dgs, dgs[1:])), dgs


def test_a_hairpin_is_found_where_an_inverted_repeat_exists():
    h = E.hairpin_free_energy("GGGGCCCCAAAAGGGGCCCC")
    assert np.isfinite(h["hairpin_dG_kcal"])
    assert h["hairpin_stem_bp"] >= 4


HAIRPIN_DNA = "GGATCCGCGCAAAAGCGCGGATCC"


def test_dna_fold_really_uses_the_dna_parameters():
    """The DNA and RNA parameter sets give different energies for the same sequence.

    If the switch silently failed, the hairpin channel would be an RNA fold of a DNA
    sequence, and nothing else would notice.
    """
    import RNA
    e_dna, _ = E.dna_fold_mfe(HAIRPIN_DNA, na_molar=1.021)
    _, e_rna = RNA.fold_compound(HAIRPIN_DNA).mfe()
    assert e_dna < 0
    assert abs(e_dna - e_rna) > 0.5, (e_dna, e_rna)


def test_dna_fold_restores_the_rna_parameters():
    """ViennaRNA's parameters are global: a DNA fold must leave RNA folding untouched.

    Family L folds both transcripts of every fragment in the same process as Family E,
    so a leaked DNA parameter set would change every RNA channel computed after it.
    """
    import RNA
    rna = "GGGAAACUUCGGUUUCCCAUAUGCGCAUAUGG"
    before = RNA.fold_compound(rna).mfe()[1]
    E.hairpin_free_energy(HAIRPIN_DNA, na_molar=0.15)
    E.cruciform_free_energy(HAIRPIN_DNA, -0.06, 0.15)
    after = RNA.fold_compound(rna).mfe()[1]
    assert before == after


def test_dna_hairpin_is_stabilised_by_salt():
    e = [E.hairpin_free_energy(HAIRPIN_DNA, na)["hairpin_dG_kcal"] for na in (0.01, 0.15, 1.0)]
    assert e[0] > e[1] > e[2], e


def test_dna_hairpin_is_symmetric_under_reverse_complement():
    """The lower of the two strands' folds is a property of the duplex."""
    from channels import reverse_complement
    s = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGCCCCTCCC"
    a = E.hairpin_free_energy(s, 0.15)["hairpin_dG_kcal"]
    b = E.hairpin_free_energy(reverse_complement(s), 0.15)["hairpin_dG_kcal"]
    assert a == pytest.approx(b, abs=1e-9)


def test_dna_loop_table_is_the_published_one():
    """Hairpin loop initiation of the DNA set: 3.4, 3.4, 3.5, 4.2 kcal/mol for 3-6 nt."""
    import RNA
    path = os.path.join(os.path.dirname(RNA.__file__), "..", "..", "..", "..", "share",
                        "ViennaRNA", "dna_mathews2004.par")
    candidates = [path] + [os.path.join(d, "share", "ViennaRNA", "dna_mathews2004.par")
                           for d in (sys.prefix,)]
    par = next((c for c in candidates if os.path.exists(c)), None)
    if par is None:
        pytest.skip("dna_mathews2004.par not found beside this ViennaRNA install")
    lines = open(par).read().split("# hairpin")[1].split("#")[0].split()
    vals = [x for x in lines if x.lstrip("-").isdigit() or x == "INF"]
    assert vals[3:7] == ["340", "340", "350", "420"], vals[:8]


# ======================================================================================
# Windows
# ======================================================================================

def test_window_ladder_skips_windows_longer_than_the_sequence():
    w = E.thermo_window_channels(SEQ)
    assert any(k.endswith("_w10") for k in w)
    assert not any(k.endswith("_w500") for k in w)
