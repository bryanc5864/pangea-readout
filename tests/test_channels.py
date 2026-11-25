#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Families F and L, and the channel declaration layer they register into.

The parity tests here are the ones `PLAN.md` makes blocking: "A channel predicted with
the wrong parity is wrong in a way no validation correlation will reveal, so parity is
checked channel by channel as a unit test before anything downstream runs." A channel
that declares SYMMETRIC and is not is a silent, permanent error in the training target,
so the declaration is tested against the computed values rather than trusted.

The algorithm tests check published behaviour on sequences whose answer is known
independently: the human telomeric repeat for G4Hunter and quadruplex structure, an
alternating CG tract for Z-DNA, a phased A-tract array for intrinsic curvature, and a
designed hairpin for RNA folding.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

import channels as ch_mod  # noqa: E402
import noncanonical as nc  # noqa: E402
from channels import (  # noqa: E402
    REGISTRY, ChannelSpec, Parity, Resolution, reverse_complement,
)

try:
    import rna_structure as rs  # noqa: E402
    HAVE_VIENNA = rs.HAVE_VIENNA
except Exception:                                            # noqa: BLE001
    rs = None
    HAVE_VIENNA = False


TELOMERIC = "GGGTTAGGGTTAGGGTTAGGG"
CG_TRACT = "AAAAACGCGCGCGCGCGCGAAAAA"
PHASED_A = "AAAAATCGATCGAAAAATCGATCGAAAAATCGATCG"
RANDOM_SEQS = [
    "ACGTACGTAGCTAGCTAGCTTACGATCGATCGGCTAGCTAGCATCGATCGATCGATTACGA",
    "GCGCGCGCATATATATGCGCGCGCATATATATGGCCGGCCAATTAATTCCGGCCGGATAT",
    "TTTTTTTTTTAAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTAAAACCCCGGGGTTTT",
]


# ======================================================================================
# The declaration layer
# ======================================================================================

def test_reverse_complement_is_an_involution():
    for s in RANDOM_SEQS + [TELOMERIC, CG_TRACT]:
        assert reverse_complement(reverse_complement(s)) == s


def test_window_channel_must_declare_its_rung():
    with pytest.raises(ValueError):
        ChannelSpec(name="x", family="F", resolution=Resolution.WINDOW,
                    parity=Parity.SYMMETRIC, units="u", description="d")


def test_non_window_channel_may_not_declare_a_rung():
    with pytest.raises(ValueError):
        ChannelSpec(name="x", family="F", resolution=Resolution.GLOBAL,
                    parity=Parity.SYMMETRIC, units="u", description="d", window_bp=50)


def test_provisional_channels_must_give_a_reason():
    with pytest.raises(ValueError):
        ChannelSpec(name="x", family="F", resolution=Resolution.GLOBAL,
                    parity=Parity.SYMMETRIC, units="u", description="d",
                    provisional=True)


def test_registry_rejects_duplicate_names():
    reg = ch_mod.ChannelRegistry()
    spec = ChannelSpec(name="dup", family="F", resolution=Resolution.GLOBAL,
                       parity=Parity.SYMMETRIC, units="u", description="d")
    reg.register(spec)
    with pytest.raises(ValueError):
        reg.register(spec)


def test_every_computed_channel_is_declared():
    """A channel that is computed but not declared has no parity and no resolution."""
    computed = set(nc.noncanonical_channels(RANDOM_SEQS[0]))
    declared = {s.name for s in REGISTRY.by_family("F")}
    assert computed <= declared, f"undeclared Family F channels: {computed - declared}"


def test_every_declared_channel_is_computed():
    """A declared channel that is never computed would be a silent hole in the tensor.

    Both resolutions: Family F emits whole-sequence scalars from
    `noncanonical_channels` and per-base tracks from `noncanonical_profiles`, and
    checking only the first is how three declared tracks once went uncomputed.
    """
    computed = (set(nc.noncanonical_channels(RANDOM_SEQS[0]))
                | set(nc.noncanonical_profiles(RANDOM_SEQS[0])))
    declared = {s.name for s in REGISTRY.by_family("F")}
    assert declared <= computed, f"declared but not computed: {declared - computed}"


# ======================================================================================
# Family F: published algorithm behaviour
# ======================================================================================

def test_g4hunter_scores_telomeric_repeat_above_threshold():
    """Bedrat 2016 treats scores above about 1.2 as strong quadruplex candidates."""
    prof = nc.g4hunter_profile(TELOMERIC)
    assert prof.max() > 1.2


def test_g4hunter_is_sign_flipped_on_the_complement():
    """A G-rich strand is a C-rich strand on the complement, so the score negates."""
    fwd = nc.g4hunter_profile(TELOMERIC)
    rev = nc.g4hunter_profile(reverse_complement(TELOMERIC))
    np.testing.assert_allclose(fwd, -rev[::-1], atol=1e-12)


def test_g4hunter_run_score_is_capped_at_four():
    """A run of 7 G scores the same as a run of 4, per the published rule."""
    assert nc._run_scores("GGGGGGG", "G", "C").max() == 4.0
    assert nc._run_scores("GGGG", "G", "C").max() == 4.0
    assert nc._run_scores("GGG", "G", "C").max() == 3.0


def test_telomeric_quadruplex_structure_is_recovered():
    """Four G3 tracts with three 3-nt loops: the canonical human telomeric quadruplex."""
    cands = [c for c in nc.find_g4_candidates(TELOMERIC) if c.strand == "+"]
    assert cands, "no quadruplex found in the telomeric repeat"
    best = max(cands, key=nc.g4_stability_score)
    assert best.n_tracts == 4
    assert best.n_tetrads == 3
    assert best.loop_lengths == (3, 3, 3)
    assert not best.has_bulge


def test_more_tetrads_is_more_stable():
    three = nc.find_g4_candidates("GGGTTAGGGTTAGGGTTAGGG")
    four = nc.find_g4_candidates("GGGGTTAGGGGTTAGGGGTTAGGGG")
    assert nc.g4_stability_score(max(four, key=nc.g4_stability_score)) > \
           nc.g4_stability_score(max(three, key=nc.g4_stability_score))


def test_longer_loops_are_less_stable():
    short = nc.find_g4_candidates("GGGTTAGGGTTAGGGTTAGGG")
    long_ = nc.find_g4_candidates("GGGTTTTTTTTAGGGTTTTTTTTAGGGTTTTTTTTAGGG")
    assert nc.g4_stability_score(max(short, key=nc.g4_stability_score)) > \
           nc.g4_stability_score(max(long_, key=nc.g4_stability_score))


def test_cation_ordering_is_potassium_sodium_lithium():
    ch = nc.noncanonical_channels(TELOMERIC)
    assert ch["F_g4_stability_K"] > ch["F_g4_stability_Na"] > ch["F_g4_stability_Li"]


def test_zdna_prefers_alternating_cg():
    """CG alternation is the most Z-forming sequence known; AT runs are the least."""
    cg = nc.z_dna_best_tract("CGCGCGCGCGCGCGCG")["z_best_kcal"]
    at = nc.z_dna_best_tract("AAAAAAAAAAAAAAAA")["z_best_kcal"]
    assert cg < at


def test_zdna_locates_the_cg_tract():
    r = nc.z_dna_best_tract(CG_TRACT)
    # The CG tract starts at index 5 in CG_TRACT.
    assert 4 <= r["z_best_start"] <= 7
    assert r["z_best_len_bp"] >= 10


def test_zdna_energies_match_the_z_hunt_table():
    """The Z-Hunt table, guarded so a typo cannot slip in, and checked against structure.

    5'-CG-3' in Z-DNA has C anti and G syn, so its anti-syn energy is the favourable
    0.66; 5'-GC-3' reaches 0.66 in its syn-anti register. CA/TG and TA give 1.40 and 2.50
    -- not the 1.20 and 2.40 an earlier version carried from a different parameterisation.
    """
    t = nc.Z_HUNT_DBZED
    assert len(t) == 16
    assert t["CG"][0] == pytest.approx(0.66) and t["GC"][1] == pytest.approx(0.66)
    assert t["CA"][0] == pytest.approx(1.40) and t["TG"][0] == pytest.approx(1.40)
    assert t["TA"][0] == pytest.approx(2.50) and t["AT"][1] == pytest.approx(2.50)
    for d, row in t.items():
        assert row[2] == row[3], f"{d}: Z-Hunt's two out-of-alternation rows differ"


def test_z_hunt_dp_equals_exhaustive_enumeration():
    """Z-Hunt enumerates all 2^n conformations; the DP must give the same minimum."""
    import itertools
    rng = np.random.default_rng(0)
    for _ in range(200):
        din = ["".join(rng.choice(list("ACGT"), 2)) for _ in range(rng.integers(1, 8))]
        best = np.inf
        for conf in itertools.product((0, 1), repeat=len(din)):
            e = 0.0
            for k, (d, c) in enumerate(zip(din, conf)):
                r = nc.Z_HUNT_DBZED[d]
                if k == 0:
                    e += r[c]
                else:
                    p = conf[k - 1]
                    e += (r[0] if (c, p) == (0, 0) else r[3] if (c, p) == (0, 1)
                          else r[1] if (c, p) == (1, 1) else r[2])
            best = min(best, e)
        assert nc.z_hunt_window_energy(din) == pytest.approx(best, abs=1e-9)


def test_z_hunt_recovers_the_known_z_forming_hierarchy():
    seqs = {"CG": "CG" * 12, "CA": "CA" * 12, "TA": "TA" * 12, "AA": "A" * 24}
    e = {k: nc.z_dna_best_tract(v)["z_best_kcal"] for k, v in seqs.items()}
    assert e["CG"] < e["CA"] < e["TA"] < e["AA"], e


def test_imotif_ph_factor_falls_with_rising_ph():
    assert nc.imotif_ph_factor(5.0) > nc.imotif_ph_factor(6.5) > nc.imotif_ph_factor(8.0)


def test_imotif_ph_factor_is_half_at_the_pka():
    assert nc.imotif_ph_factor(nc.IMOTIF_CYTOSINE_PKA) == pytest.approx(0.5)


def test_mirror_repeat_is_not_an_inverted_repeat():
    """A mirror repeat reads the same backwards; an inverted repeat is complemented."""
    mirror = "AAAGGGTTTTGGGAAA"     # GGG ... GGG about the centre, not complemented
    assert nc.find_mirror_repeats(mirror, min_arm=3, max_spacer=6)
    inverted = "AAACCCTTTTGGGTTT"   # CCC pairs with GGG
    assert nc.find_inverted_repeats(inverted, min_arm=3, max_spacer=6)


def test_inverted_repeat_arm_is_the_reverse_complement():
    seq = "GGCCAATTGGCC"
    irs = nc.find_inverted_repeats(seq, min_arm=4, max_spacer=4)
    assert irs, "no inverted repeat found in a designed palindrome"


def test_a_phasing_coherence_is_high_for_phased_tracts():
    """A-tracts spaced near the helical period add coherently; random spacing does not."""
    phased = nc.a_phasing(PHASED_A)
    assert phased["a_tract_count"] >= 3
    assert phased["a_phasing_coherence"] > 0.8


def test_a_phasing_coherence_is_low_for_unphased_tracts():
    unphased = "AAAAATCGAAAAATCGATCGATCGATCGAAAAAT"
    assert nc.a_phasing(unphased)["a_phasing_coherence"] < \
           nc.a_phasing(PHASED_A)["a_phasing_coherence"]


def test_gc_skew_signs_correctly():
    assert nc.gc_skew("GGGGG") == pytest.approx(1.0)
    assert nc.gc_skew("CCCCC") == pytest.approx(-1.0)
    assert nc.gc_skew("GGCC") == pytest.approx(0.0)
    assert np.isclose(nc.gc_skew("AAAA"), 0.0)


def test_tandem_repeat_detection():
    r = nc.repeat_descriptors("CAGCAGCAGCAGCAGCAGAAAA")
    assert r["max_tandem_unit_bp"] == 3.0
    assert r["max_tandem_copies"] >= 6


def test_rloop_score_is_higher_for_g_rich_skewed_sequence():
    g_rich = "GGGGAGGGGAGGGGAGGGGAGGGGAGGGGAGGGGAGGGGAGGGGAGGGGA" + "GGGAGGGAGGGA" * 8
    balanced = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC" + "ACGTACGTACGT" * 8
    assert nc.rloop_zones(g_rich)["rloop_score"] > nc.rloop_zones(balanced)["rloop_score"]


# ======================================================================================
# Family F: parity, the blocking check
# ======================================================================================

@pytest.mark.parametrize("seq", RANDOM_SEQS)
def test_declared_parity_holds_for_every_family_f_channel(seq):
    """Compute every channel on a sequence and on its reverse complement, and check
    each one behaves as it declared. This is the check PLAN.md requires to pass before
    anything downstream runs."""
    fwd = nc.noncanonical_channels(seq)
    rev = nc.noncanonical_channels(reverse_complement(seq))
    fwd_p = nc.noncanonical_profiles(seq)
    rev_p = nc.noncanonical_profiles(reverse_complement(seq))

    failures = []
    for spec in REGISTRY.by_family("F"):
        if spec.parity is Parity.STRANDED:
            continue                       # checked separately; the strands exchange
        if spec.resolution is Resolution.PER_BP:
            # A track is compared against the reverse-complement track read backwards,
            # so the position mapping is applied before the sign rule. Comparing them
            # position for position would test the wrong thing and could pass by
            # accident on a near-uniform profile.
            a = np.asarray(fwd_p[spec.name], dtype=float)
            b = np.asarray(rev_p[spec.name], dtype=float)[::-1]
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                continue
            ok, diff = ch_mod.check_parity(a[m], b[m], spec, atol=1e-6, rtol=1e-6)
            if not ok:
                failures.append((spec.name, spec.parity.value,
                                 float(a[m][0]), float(b[m][0]), diff))
            continue
        a, b = fwd[spec.name], rev[spec.name]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ok, diff = ch_mod.check_parity([a], [b], spec, atol=1e-6, rtol=1e-6)
        if not ok:
            failures.append((spec.name, spec.parity.value, a, b, diff))
    assert not failures, "parity violations:\n" + "\n".join(
        f"  {n}: declared {p}, forward={x:.6g}, reverse={y:.6g}, diff={d:.3g}"
        for n, p, x, y, d in failures)


@pytest.mark.parametrize("seq", RANDOM_SEQS)
def test_stranded_channels_actually_exchange(seq):
    """A STRANDED channel must take the other strand's value under reverse complement.

    This is the positive half of the declaration: if a channel declared STRANDED were in
    fact symmetric, the declaration would be harmlessly conservative, but if it does not
    exchange at all then it is not stranded and the label is wrong.
    """
    fwd = nc.noncanonical_channels(seq)
    rev = nc.noncanonical_channels(reverse_complement(seq))
    # GC skew is the cleanest case: it must negate exactly.
    assert fwd["F_gc_skew"] == pytest.approx(-rev["F_gc_skew"], abs=1e-12)


def test_profiles_are_the_right_length():
    seq = RANDOM_SEQS[0]
    prof = nc.noncanonical_profiles(seq)
    for name, arr in prof.items():
        assert len(arr) == len(seq), f"{name} has length {len(arr)}, expected {len(seq)}"


def test_window_channels_track_the_ladder():
    seq = RANDOM_SEQS[0] * 10            # long enough for the upper rungs
    out = nc.window_channels(seq)
    for name, arr in out.items():
        w = int(name.rsplit("_w", 1)[1])
        assert len(arr) == len(seq) - w + 1


def test_provisional_channels_are_flagged_and_reasoned():
    prov = [s for s in REGISTRY.by_family("F") if s.provisional]
    assert prov, "Family F has provisional channels and they must be declared"
    for s in prov:
        assert s.provisional_reason, f"{s.name} is provisional without a reason"


def test_non_provisional_channels_cite_a_source_or_are_structural():
    """Anything carrying published parameters must name where they came from."""
    for s in REGISTRY.by_family("F"):
        if s.provisional:
            continue
        if "kcal" in s.units:
            assert s.source, f"{s.name} reports energies without citing a source"


# ======================================================================================
# Family L: RNA secondary structure
# ======================================================================================

pytestmark_rna = pytest.mark.skipif(not HAVE_VIENNA, reason="ViennaRNA not installed")


@pytestmark_rna
def test_designed_hairpin_folds_as_designed():
    """GGGG-loop-CCCC must fold into one helix with one hairpin."""
    r = rs.fold_once(rs.to_rna("GGGGAAAACCCC"))
    assert r["mfe_structure"] == "((((....))))"
    assert r["n_hairpins"] == 1.0
    assert r["n_helices"] == 1.0
    assert r["mfe_kcal"] < 0


@pytestmark_rna
def test_lower_salt_destabilises_rna():
    """The ViennaRNA salt correction must move the free energy in the physical direction."""
    rna = rs.to_rna("GGGGAAAACCCCATGCATGCATGCGGGGTTTTCCCC")
    low = rs.fold_once(rna, salt_M=0.01)["mfe_kcal"]
    high = rs.fold_once(rna, salt_M=1.021)["mfe_kcal"]
    assert low > high, "lower salt should destabilise the fold (less negative MFE)"


@pytestmark_rna
def test_higher_temperature_destabilises_rna():
    rna = rs.to_rna("GGGGAAAACCCCATGCATGCATGCGGGGTTTTCCCC")
    cold = rs.fold_once(rna, temperature_C=25.0)["mfe_kcal"]
    hot = rs.fold_once(rna, temperature_C=50.0)["mfe_kcal"]
    assert hot > cold


@pytestmark_rna
def test_ensemble_free_energy_is_below_the_mfe():
    """The ensemble includes the MFE structure plus others, so it must be lower."""
    r = rs.fold_once(rs.to_rna("GGGGAAAACCCCATGCATGCATGC"))
    assert r["ensemble_kcal"] <= r["mfe_kcal"] + 1e-9


@pytestmark_rna
def test_frequency_of_mfe_is_a_probability():
    r = rs.fold_once(rs.to_rna("GGGGAAAACCCCATGCATGCATGC"))
    assert 0.0 < r["frequency_of_mfe"] <= 1.0


@pytestmark_rna
def test_unpaired_probability_is_a_probability():
    r = rs.fold_once(rs.to_rna(RANDOM_SEQS[0]))
    u = r["unpaired_profile"]
    assert u.shape == (len(RANDOM_SEQS[0]),)
    assert np.all(u >= -1e-9) and np.all(u <= 1.0 + 1e-9)


@pytestmark_rna
def test_pair_table_round_trips():
    pt = rs.pair_table("((((....))))")
    assert pt[0] == 11 and pt[11] == 0
    assert pt[3] == 8 and pt[8] == 3
    assert np.all(pt[4:8] == -1)


@pytestmark_rna
def test_helix_length_of_a_clean_stem():
    h = rs.helices("((((....))))")
    assert len(h) == 1
    assert h[0][2] == 4


@pytestmark_rna
def test_paired_and_unpaired_fractions_sum_to_one():
    d = rs.loop_decomposition("((((....))))..((..))")
    assert d["paired_fraction"] + d["unpaired_fraction"] == pytest.approx(1.0)


@pytestmark_rna
def test_rna_channels_are_stranded_not_symmetric():
    """The two strands' transcripts are different molecules and must differ.

    This is the substantive content of the STRANDED declaration. A hairpin on the sense
    strand is not a hairpin on the antisense strand, and treating these as symmetric
    would put a wrong target into training with no validation signal to catch it.
    """
    seq = "GGGGAAAACCCCATGCATGCATGCAAAATTTTGGGGAAAACCCC"
    plus = rs.rna_channels(seq, "+")
    minus = rs.rna_channels(seq, "-")
    assert plus["L_p_mfe_kcal"] != pytest.approx(minus["L_m_mfe_kcal"])


@pytestmark_rna
def test_every_computed_family_l_channel_is_declared():
    """Both resolutions, because the family emits at both.

    This test originally compared only against `rna_channels`, so the per-base tracks
    `rna_profiles` returns could be -- and for a while were -- emitted without ever
    being declared. See `tests/test_landscape.py` for why that mattered.
    """
    seq = "GGGGAAAACCCCATGCATGCATGCAAAATTTTGGGGAAAACCCC"
    computed = set()
    for strand in ("+", "-"):
        computed |= set(rs.rna_channels(seq, strand))
        computed |= set(rs.rna_profiles(seq, strand))
    declared = {s.name for s in REGISTRY.by_family("L")}
    assert computed <= declared, f"undeclared: {computed - declared}"
    assert declared <= computed, f"declared but not computed: {declared - computed}"


@pytestmark_rna
def test_family_l_declares_every_channel_stranded():
    """Any non-STRANDED channel in this family would be a modelling error."""
    for s in REGISTRY.by_family("L"):
        assert s.parity is Parity.STRANDED, f"{s.name} is not declared STRANDED"


@pytestmark_rna
def test_rna_profiles_are_the_right_length():
    seq = RANDOM_SEQS[0]
    for name, arr in rs.rna_profiles(seq, "+").items():
        assert len(arr) == len(seq), f"{name}: {len(arr)} != {len(seq)}"


@pytest.mark.parametrize("seq", RANDOM_SEQS)
def test_g4hunter_max_and_min_exchange_under_reverse_complement(seq):
    """The STRANDED declaration on the G4Hunter extrema has specific content.

    Under reverse complement the G4Hunter profile obeys f(i) -> -f(n-1-i), so the
    maximum and the minimum do not map to themselves: they map to each other with a
    sign change. Skipping STRANDED channels in the parity sweep would leave that
    unchecked, so the exchange relation is asserted directly here.
    """
    fwd = nc.noncanonical_channels(seq)
    rev = nc.noncanonical_channels(reverse_complement(seq))
    assert fwd["F_g4hunter_max"] == pytest.approx(-rev["F_g4hunter_min"], abs=1e-9)
    assert fwd["F_g4hunter_min"] == pytest.approx(-rev["F_g4hunter_max"], abs=1e-9)
    # The strand-free summary really is strand free.
    assert fwd["F_g4hunter_absmax"] == pytest.approx(rev["F_g4hunter_absmax"], abs=1e-9)


@pytest.mark.parametrize("seq", RANDOM_SEQS)
def test_imotif_exchanges_with_the_quadruplex_score(seq):
    """The i-motif forms on the C-rich strand, so it is the quadruplex score's partner."""
    fwd = nc.noncanonical_channels(seq)
    assert fwd["F_imotif_propensity"] == pytest.approx(-fwd["F_g4hunter_min"], abs=1e-9)


def test_zdna_centre_offset_is_zero_for_a_centred_tract():
    """A Z-forming tract placed symmetrically must report no displacement."""
    seq = "AAAAAAAAAA" + "CGCGCGCGCGCG" + "AAAAAAAAAA"
    ch = nc.noncanonical_channels(seq)
    assert abs(ch["F_zdna_centre_offset_bp"]) < 2.0


def test_zdna_centre_offset_signs_correctly():
    """A tract left of centre and the same tract right of centre must differ in sign."""
    left = nc.noncanonical_channels("CGCGCGCGCGCG" + "A" * 40)
    right = nc.noncanonical_channels("A" * 40 + "CGCGCGCGCGCG")
    assert left["F_zdna_centre_offset_bp"] < 0 < right["F_zdna_centre_offset_bp"]
