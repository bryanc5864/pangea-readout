#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family C, pinned against measured B-DNA geometry and against the library itself.

Family C reports the quantities every published DNA structure paper reports, so unlike
most of the landscape it can be checked directly: groove widths, backbone torsions and
the behaviour of the rigid-base coordinates under reverse complement are all measured
numbers with measured values.

Three of these tests exist because the obvious implementation was wrong and the test is
what caught it. Each of those carries a **negative control** asserting that the
uncorrected version still fails, so the correction cannot silently become a no-op:

  * `test_mid_frame_translations_obey_parity` with
    `test_native_frame_translations_do_not_obey_parity`
  * `test_minimisation_repairs_the_backbone_junctions` with
    `test_raw_reconstruction_breaks_the_backbone_junctions`
  * `test_minimised_torsions_match_canonical_b_dna` with
    `test_raw_torsions_do_not_match_canonical_b_dna`
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

import geometry as G  # noqa: E402
from channels import REGISTRY, Parity, Resolution, reverse_complement  # noqa: E402

SEQ = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGC"
HALF = "GATCCGTAAGCTTGAC"
PALINDROME = HALF + reverse_complement(HALF)

# Measured B-DNA, for the checks that make this family trustworthy rather than plausible.
MINOR_WIDTH_A = 5.7
MAJOR_WIDTH_A = 11.7
PHOSPHATE_RADIUS_A = 9.3
CANONICAL_TORSIONS = {"alpha": -63.0, "beta": 171.0, "gamma": 54.0,
                      "delta": 123.0, "epsilon": -175.0, "zeta": -100.0}


@pytest.fixture(scope="module")
def state():
    return G.geometry_state(SEQ, n_ensemble=100)


@pytest.fixture(scope="module")
def pal_profiles():
    return G.geometry_profiles(PALINDROME, n_ensemble=0, with_conditions=False)


@pytest.fixture(scope="module")
def minimised(tmp_path_factory):
    d = tmp_path_factory.mktemp("famC")
    return G.build_minimised_structure(SEQ, str(d))


# ======================================================================================
# The registry is the contract
# ======================================================================================

def test_every_emitted_channel_is_declared(state):
    prof = G.geometry_profiles(SEQ, state=state)
    chan = G.geometry_channels(SEQ, state=state)
    emitted = set(prof) | set(chan)
    undeclared = sorted(n for n in emitted if n not in REGISTRY)
    assert not undeclared, f"emitted but never declared: {undeclared}"


def test_every_declared_channel_is_emitted(state):
    prof = G.geometry_profiles(SEQ, state=state)
    chan = G.geometry_channels(SEQ, state=state)
    emitted = set(prof) | set(chan)
    declared = {s.name for s in REGISTRY.by_family("C")}
    missing = sorted(declared - emitted)
    assert not missing, f"declared but never emitted: {missing}"


def test_resolutions_match_where_the_channel_is_emitted(state):
    prof = G.geometry_profiles(SEQ, state=state)
    chan = G.geometry_channels(SEQ, state=state)
    for s in REGISTRY.by_family("C"):
        if s.resolution is Resolution.PER_BP:
            assert s.name in prof, f"{s.name} declared PER_BP but not a profile"
        elif s.resolution is Resolution.GLOBAL:
            assert s.name in chan, f"{s.name} declared GLOBAL but not a scalar"


def test_family_c_is_near_the_planned_channel_count():
    n = len(REGISTRY.by_family("C"))
    assert 200 <= n <= 230, f"PLAN.md budgets 215 channels for Family C; got {n}"


def test_provisional_channels_say_why():
    for s in REGISTRY.by_family("C"):
        if s.provisional:
            assert len(s.provisional_reason) > 40, s.name


def test_channels_with_no_parameter_set_are_nan_not_invented(state):
    """Salt and temperature emit NaN. A plausible number would be a fabrication."""
    prof = G.geometry_profiles(SEQ, state=state)
    for name in [f"C_{p}_{k}" for p in ("salt", "temp") for k in G.CONDITION_DELTA_NAMES]:
        assert np.all(np.isnan(prof[name])), f"{name} emitted a value it cannot know"
        assert REGISTRY[name].provisional


ENSEMBLE_ONLY = ([f"C_{t}_{s}_sd" for t in G.TORSION_NAMES for s in ("W", "C")]
                 + [f"C_BI_fraction_{s}" for s in ("W", "C")]
                 + [f"C_sugar_C{k}endo_prob_{s}" for k in (2, 3) for s in ("W", "C")])


def test_ensemble_channels_are_nan_without_the_relaxed_ensemble(state):
    """Spreads and populations exist only on the ensemble subset; elsewhere they are NaN.

    In particular the BI fraction is no longer a single structure's 0/1 indicator
    reported under a population's name.
    """
    prof = G.geometry_profiles(SEQ, state=state)
    for name in ENSEMBLE_ONLY:
        assert np.all(np.isnan(prof[name])), name
        assert not REGISTRY[name].provisional, name


def test_pucker_is_c2_endo_on_the_minimised_structure(minimised):
    """Altona-Sundaralingam on the relaxed structure: B-DNA sugars are C2'-endo."""
    pk = G.sugar_pucker_from_structure(minimised["minimised_pdb"], len(SEQ))
    for tag in ("W", "C"):
        P = pk[f"phase_{tag}"][np.isfinite(pk[f"phase_{tag}"])]
        A = pk[f"amplitude_{tag}"][np.isfinite(pk[f"amplitude_{tag}"])]
        assert len(P) >= len(SEQ) - 2
        assert np.mean((P >= 108) & (P <= 216)) > 0.9, np.round(P, 0)
        assert 28.0 < float(np.mean(A)) < 45.0


def test_minimisation_keeps_alpha_gamma_canonical(minimised):
    """The two-start minimisation leaves no alpha/gamma crankshaft traps.

    A plain minimisation from the reconstruction left about 20 percent of nucleotides
    with alpha g+ and gamma g- together; that state is higher in energy than the
    canonical one by hundreds of kJ/mol per duplex (see `minimize.restrained_minimise`).
    """
    rep = minimised["report"]
    assert rep["energy_final_kJ_mol"] <= min(rep["energy_final_plain_kJ_mol"],
                                             rep["energy_final_canonical_kJ_mol"]) + 1e-6
    tor = G.torsions_from_structure(minimised["minimised_pdb"], len(SEQ))
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0                       # noqa: E731
    for tag in ("W", "C"):
        a = wrap(tor[f"alpha_{tag}"]); a = a[np.isfinite(a)]
        g = wrap(tor[f"gamma_{tag}"]); g = g[np.isfinite(g)]
        assert np.mean((a > -120) & (a < 0)) > 0.95, np.round(a, 0)
        assert np.mean((g > 0) & (g < 120)) > 0.95, np.round(g, 0)


# ======================================================================================
# Measured B-DNA geometry
# ======================================================================================

def test_groove_widths_match_measured_b_dna(state):
    prof = G.geometry_profiles(SEQ, state=state)
    minor = float(np.nanmean(prof["C_minor_width_A"][5:-5]))
    major = float(np.nanmean(prof["C_major_width_A"][5:-5]))
    assert abs(minor - MINOR_WIDTH_A) < 1.0, f"minor groove width {minor:.2f} A"
    assert abs(major - MAJOR_WIDTH_A) < 1.5, f"major groove width {major:.2f} A"
    assert major > minor


def test_groove_offsets_are_the_two_minima_of_the_phosphate_offset_scan(state):
    """The offsets -4 and +4 were found by scanning, not assumed. Re-scan and confirm."""
    g = state["ground"]
    poses = g["poses"]
    import cgna_atomic as ca
    templates = ca.NucleotideTemplates()
    w_letters = list(SEQ)
    c_letters = [G.COMPLEMENT[c] for c in SEQ][::-1]
    _, wphos = G.strand_atoms(w_letters, poses["watson_base"], poses["watson_phos"],
                              templates, G.BASE_ATOMS_NEEDED, G.PHOS_ATOMS_NEEDED)
    _, cphos = G.strand_atoms(c_letters, poses["crick_base"][:, ::-1],
                              poses["crick_phos"][:, ::-1], templates,
                              G.BASE_ATOMS_NEEDED, G.PHOS_ATOMS_NEEDED)
    pw, pc = wphos["P"][0], cphos["P"][0][::-1]
    n = len(SEQ)
    means = {}
    for off in range(-12, 13):
        d = [np.linalg.norm(pw[i] - pc[i + off])
             for i in range(1, n - 1) if 0 <= i + off < n - 1]
        d = [x for x in d if np.isfinite(x)]
        if len(d) > 5:
            means[off] = float(np.mean(d))
    minima = [o for o in means
              if all(means[o] <= means.get(o + s, np.inf) for s in (-1, 1))]
    assert G.MINOR_OFFSET in minima, f"offset {G.MINOR_OFFSET} is not a minimum: {minima}"
    assert G.MAJOR_OFFSET in minima, f"offset {G.MAJOR_OFFSET} is not a minimum: {minima}"


def test_phosphate_radius_is_near_measured_b_dna(state):
    prof = G.geometry_profiles(SEQ, state=state)
    for tag in ("W", "C"):
        r = float(np.nanmean(prof[f"C_phosphate_radius_A_{tag}"][3:-3]))
        assert abs(r - PHOSPHATE_RADIUS_A) < 1.5, f"{tag} phosphate radius {r:.2f} A"


def test_twist_is_near_the_b_form_helical_repeat(state):
    prof = G.geometry_profiles(SEQ, state=state)
    tw = float(np.nanmean(prof["C_twist"]))
    assert 33.0 < tw < 36.0, f"mean twist {tw:.2f} deg/step"
    assert 10.0 < 360.0 / tw < 11.0


# ======================================================================================
# Parity: measured on an exact palindrome before it was declared
# ======================================================================================

def _reflect(v, shift=0):
    n = len(v)
    idx = np.arange(n)[::-1] + shift
    ok = (idx >= 0) & (idx < n)
    out = np.full(n, np.nan)
    out[ok] = np.asarray(v, dtype=float)[idx[ok]]
    return out


def _parity_residual(v, target, sign):
    v = np.asarray(v, dtype=float)
    m = np.isfinite(v) & np.isfinite(target)
    assert m.sum() >= 5
    return float(np.abs(v[m] - sign * target[m]).max() / max(np.abs(v[m]).max(), 1e-12))


def test_mid_frame_translations_obey_parity(pal_profiles):
    """All 12 rigid-base coordinates obey the 3DNA rules exactly on a palindrome."""
    for lab in G.INTRA_NAMES + G.INTER_NAMES:
        spec = REGISTRY[f"C_{lab}"]
        sign = -1.0 if spec.parity is Parity.ANTISYMMETRIC else 1.0
        v = pal_profiles[f"C_{lab}"]
        shift = 0 if lab in G.INTRA_NAMES else -1
        res = _parity_residual(v, _reflect(v, shift), sign)
        assert res < 1e-9, f"{lab} declared {spec.parity.value}, residual {res:.2e}"


def test_native_frame_translations_do_not_obey_parity():
    """Negative control: in cgNA+'s own preceding-frame convention, they fail badly.

    Without this, converting to the mid-step frame could be quietly removed and the
    test above would still pass on the rotations alone, which were always exact.
    """
    st = G.geometry_state(PALINDROME, n_ensemble=0)
    blocks = st["blocks"]
    worst = 0.0
    for kind, labels, shift in (("B", G.INTRA_NAMES, 0), ("S", G.INTER_NAMES, -1)):
        g = np.asarray(blocks[f"{kind}_gs"], dtype=float)
        n = len(g)
        for j, lab in enumerate(labels):
            if j < 3:
                continue                      # rotations are exact in either convention
            v = np.full(len(PALINDROME), np.nan)
            v[:n] = g[:, j]
            sign = -1.0 if lab in G.ANTISYMMETRIC_COORDS else 1.0
            worst = max(worst, _parity_residual(v, _reflect(v, shift), sign))
    assert worst > 0.05, (
        "the raw cgNA+ translations now satisfy parity, so to_mid_frame is a no-op; "
        f"worst residual {worst:.2e}")


def test_to_mid_frame_is_invertible():
    st = G.geometry_state(SEQ, n_ensemble=0)
    g = np.asarray(st["blocks"]["S_gs"], dtype=float)
    m = G.to_mid_frame(g)
    assert not np.allclose(g[:, 3:], m[:, 3:])
    np.testing.assert_allclose(np.linalg.norm(g[:, 3:], axis=1),
                               np.linalg.norm(m[:, 3:], axis=1), atol=1e-12)


def test_groove_channels_obey_parity_under_their_own_index_shift(pal_profiles):
    """Grooves are symmetric under a shift of exactly the groove offset."""
    for label, shift in (("minor", -G.MINOR_OFFSET), ("major", -G.MAJOR_OFFSET)):
        for q in ("width", "depth"):
            v = pal_profiles[f"C_{label}_{q}_A"]
            res = _parity_residual(v, _reflect(v, shift), 1.0)
            assert res < 1e-9, f"{label} {q} under shift {shift}: residual {res:.2e}"
            # Formerly declared COMPOSITE, because "symmetric under an index shift" had
            # no other way to be said. With an explicit rc_shift it is an ordinary
            # symmetric channel that the model can enforce.
            spec = REGISTRY[f"C_{label}_{q}_A"]
            assert spec.parity is Parity.SYMMETRIC and spec.rc_shift == shift


def test_groove_channels_fail_without_the_index_shift(pal_profiles):
    """Negative control: declaring them plainly SYMMETRIC would have been wrong."""
    v = pal_profiles["C_minor_width_A"]
    assert _parity_residual(v, _reflect(v, 0), 1.0) > 0.05


def test_phosphate_descriptors_exchange_strands_exactly(pal_profiles):
    """STRANDED, with the exchange rule stated: radius even, azimuth and offset odd."""
    rules = {"radius_A": 1.0, "azimuth_deg": -1.0, "axial_offset_A": -1.0}
    for key, sign in rules.items():
        w = pal_profiles[f"C_phosphate_{key}_W"]
        c = pal_profiles[f"C_phosphate_{key}_C"]
        res = _parity_residual(w, _reflect(c, 0), sign)
        assert res < 1e-9, f"phosphate {key}: residual {res:.2e}"
        assert REGISTRY[f"C_phosphate_{key}_W"].parity is Parity.STRANDED


def test_no_entry_of_the_step_stiffness_carries_a_simple_parity(pal_profiles):
    """Every one of the 78 entries is COMPOSITE, and this is why.

    The first guess was that only the 36 cross intra-by-inter entries would need
    COMPOSITE and the 42 pure ones would carry a sign. Measurement says every entry
    fails, under either index map and either sign, by residuals of order 1 -- while the
    12 coordinates those entries are built from obey parity to 1e-13 in the same
    computation.

    That contrast is the point of this test. COMPOSITE is a real property of these
    channels, not a blanket applied to whatever would not pass, and if a future change
    made any entry genuinely symmetric this test would catch the stale declaration.
    """
    residuals = {}
    for s in REGISTRY.by_family("C"):
        if not s.name.startswith("C_K_"):
            continue
        assert s.parity is Parity.COMPOSITE, s.name
        v = pal_profiles[s.name]
        residuals[s.name] = min(_parity_residual(v, _reflect(v, sh), sg)
                                for sh in (0, -1, 1) for sg in (1.0, -1.0))
    assert len(residuals) == 78
    best_name = min(residuals, key=residuals.get)
    assert residuals[best_name] > 1e-3, (
        f"{best_name} now carries a simple parity (residual "
        f"{residuals[best_name]:.2e}); its COMPOSITE declaration is stale")


def test_symmetrised_channels_obey_their_declared_parity(state):
    """Symmetrisation removes the cgNA+ asymmetry from the whole-sequence channels."""
    from channels import symmetrise_channels
    fwd = G.geometry_channels(SEQ, n_ensemble=0)
    rev = G.geometry_channels(reverse_complement(SEQ), n_ensemble=0)
    sym_f = symmetrise_channels(fwd, rev)
    sym_r = symmetrise_channels(rev, fwd)
    worst_name, worst = None, 0.0
    for name, a in sym_f.items():
        spec = REGISTRY[name]
        if spec.parity in (Parity.STRANDED, Parity.COMPOSITE):
            continue
        sign = -1.0 if spec.parity is Parity.ANTISYMMETRIC else 1.0
        b = sym_r[name]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = abs(a - sign * b) / max(abs(a), 1e-12)
        if d > worst:
            worst_name, worst = name, d
    assert worst < 1e-9, f"{worst_name} residual {worst:.2e} after symmetrisation"


def test_unsymmetrised_channels_do_not_obey_parity():
    """Negative control: without symmetrisation the cgNA+ asymmetry is visible."""
    fwd = G.geometry_channels(SEQ, n_ensemble=0)
    rev = G.geometry_channels(reverse_complement(SEQ), n_ensemble=0)
    worst = 0.0
    for name, a in fwd.items():
        spec = REGISTRY[name]
        if spec.parity in (Parity.STRANDED, Parity.COMPOSITE):
            continue
        sign = -1.0 if spec.parity is Parity.ANTISYMMETRIC else 1.0
        b = rev[name]
        if np.isfinite(a) and np.isfinite(b):
            worst = max(worst, abs(a - sign * b) / max(abs(a), 1e-12))
    assert worst > 1e-6, "cgNA+ has become exactly RC-symmetric; symmetrisation is moot"


# ======================================================================================
# The backbone: the finding that reshaped this family
# ======================================================================================

def _junction_bonds(pdb_path):
    import frames as fr
    res = fr.read_pdb_residues(pdb_path)
    keys = sorted([k for k in res if k[0] == "A"], key=lambda k: k[1])
    c3o3, o5c5 = [], []
    for i, k in enumerate(keys):
        r = res[k]
        if "C3'" in r and "O3'" in r:
            c3o3.append(np.linalg.norm(r["C3'"] - r["O3'"]))
        if "O5'" in r and "C5'" in r and i > 0:
            o5c5.append(np.linalg.norm(r["O5'"] - r["C5'"]))
    return np.array(c3o3), np.array(o5c5)


def test_raw_reconstruction_breaks_the_backbone_junctions(minimised):
    """Negative control, and the measurement that forced the minimised route.

    cgNA+ does not constrain the covalent link between the sugar, which rides the base
    frame, and the phosphate group, which is its own rigid body. The two bonds that span
    that junction come out visibly wrong.
    """
    c3o3, o5c5 = _junction_bonds(minimised["raw_pdb"])
    assert abs(c3o3.mean() - 1.43) > 0.1, f"C3'-O3' {c3o3.mean():.3f} A"
    assert abs(o5c5.mean() - 1.44) > 0.1, f"O5'-C5' {o5c5.mean():.3f} A"


def test_minimisation_repairs_the_backbone_junctions(minimised):
    c3o3, o5c5 = _junction_bonds(minimised["minimised_pdb"])
    assert abs(c3o3.mean() - 1.43) < 0.03, f"C3'-O3' {c3o3.mean():.3f} A"
    assert abs(o5c5.mean() - 1.44) < 0.03, f"O5'-C5' {o5c5.mean():.3f} A"


def test_minimisation_keeps_the_structure_sequence_dependent(minimised):
    """An unrestrained minimisation would relax every fragment to the same duplex."""
    assert minimised["report"]["restrained_rmsd_angstrom"] < 0.6


def _mean_torsions(pdb_path):
    t = G.torsions_from_structure(pdb_path, len(SEQ))
    return {k: float(G.circ_mean(t[f"{k}_W"][2:-2])) for k in CANONICAL_TORSIONS}


def test_minimised_torsions_match_canonical_b_dna(minimised):
    got = _mean_torsions(minimised["minimised_pdb"])
    for k, want in CANONICAL_TORSIONS.items():
        off = abs((got[k] - want + 180.0) % 360.0 - 180.0)
        assert off < 25.0, f"{k} = {got[k]:.1f} deg, canonical {want}, off {off:.1f}"


def test_raw_torsions_do_not_match_canonical_b_dna(minimised):
    """Negative control: the unminimised torsions are off by tens of degrees."""
    got = _mean_torsions(minimised["raw_pdb"])
    offs = [abs((got[k] - w + 180.0) % 360.0 - 180.0) for k, w in CANONICAL_TORSIONS.items()]
    assert max(offs) > 30.0, f"raw torsions are now canonical: max offset {max(offs):.1f}"


def test_bi_dominates_as_it_does_in_measured_b_dna(minimised):
    t = G.torsions_from_structure(minimised["minimised_pdb"], len(SEQ))
    d = G._wrap180(t["epsilon_W"] - t["zeta_W"])[2:-2]
    assert float(np.nanmean(d < 0.0)) > 0.6, "BI should dominate in relaxed B-DNA"


# ======================================================================================
# The condition axis
# ======================================================================================

def test_methylated_parameter_set_reproduces_the_unmodified_one(state):
    """The methylated set is a strict superset, so unmodified sequence is unchanged.

    Without this, a methylation delta could be an artifact of switching parameter sets
    rather than of the modification.
    """
    a = G.geometry_state(SEQ, n_ensemble=0)
    b = G.geometry_state(SEQ, n_ensemble=0, parameter_set=G.PARAM_SET_METHYL)
    np.testing.assert_allclose(np.asarray(a["blocks"]["gs"]),
                               np.asarray(b["blocks"]["gs"]), atol=1e-12)


def test_methylation_alphabet_avoids_the_terminal_dinucleotides():
    m = G.methylate_cpg("CGCGAAATTTCGCG")
    assert m[:2] == "CG" and m[-2:] == "CG", m
    assert "M" in m and "N" in m


def test_methylation_changes_geometry_in_the_measured_direction(state):
    """CpG methylation reduces twist and widens the minor groove, as measured."""
    prof = G.geometry_profiles(SEQ, state=state)
    dtwist = float(np.nanmean(prof["C_meth_dtwist_deg"][4:-4]))
    dminor = float(np.nanmean(prof["C_meth_dminor_width_A"][4:-4]))
    assert dtwist < 0.0, f"methylation raised twist by {dtwist:.3f} deg"
    assert dminor > 0.0, f"methylation narrowed the minor groove by {dminor:.3f} A"
    assert abs(dtwist) < 3.0 and abs(dminor) < 1.0, "implausibly large for a methyl group"


# ======================================================================================
# Windows
# ======================================================================================

def test_window_ladder_skips_windows_longer_than_the_sequence(state):
    w = G.geometry_window_channels(SEQ, state=state)
    assert any(k.endswith("_w10") for k in w)
    assert not any(k.endswith("_w500") for k in w)


def test_window_means_converge_to_the_global_mean(state):
    w = G.geometry_window_channels(SEQ, state=state)
    glob = float(np.nanmean(G.geometry_profiles(SEQ, state=state)["C_twist"]))
    wide = float(np.nanmean(w["C_twist_w20"]))
    assert abs(wide - glob) < 1.0
