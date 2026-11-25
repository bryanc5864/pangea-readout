#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family D, pinned against measured B-DNA mechanics.

Family D is the one analytic family whose numbers can be checked directly against
textbook measurements, so it is the family that validates the whole cgNA+ chain:
coordinate convention, unit convention, the marginalisation of the stiffness, and the
elastic relations. If the persistence length is right, very little upstream can be
badly wrong. If it is not, nothing downstream is trustworthy.

Two computed moduli do *not* match measurement and are asserted at their computed
values rather than at the measured ones, so that the discrepancy is recorded as a known
open item instead of being hidden by a loose tolerance. See DECISIONS.md.
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
import mesoscale as ms  # noqa: E402
from channels import REGISTRY, Parity  # noqa: E402

SEQ = "ACGTACGTAGCTAGCTAGCTTACGATCGATCGGCTAGCTAGCATCGATCGATCGATTACGA"
POLY_A = "A" * 24


@pytest.fixture(scope="module")
def gs():
    return at.cgna_ground_state(SEQ, include_stiffness=True)


@pytest.fixture(scope="module")
def blocks(gs):
    return ms.step_blocks(gs)


# ======================================================================================
# The coordinate convention, which everything else rests on
# ======================================================================================

def test_rise_coordinate_is_nanometres():
    """B-DNA rises 3.4 angstrom per step. A rise coordinate near 0.34 means nm."""
    g = at.cgna_ground_state(POLY_A, include_stiffness=False)
    names, vals = list(g["param_names"]), np.asarray(g["gs"])
    rise = np.mean([vals[i][5] for i, n in enumerate(names) if n.startswith("S")])
    assert 0.30 < rise < 0.36, f"rise coordinate {rise}, expected about 0.34 nm"


def test_rotational_coordinate_is_a_rotation_vector_not_cayley():
    """Which convention the library actually returns, asked of the library itself.

    `analytic_targets.cgna_ground_state` uses the constructor defaults, under which
    `euler_definition=True` converts the stored Cayley vector into a rotation vector
    Omega = theta n_hat. Asserting that by re-deriving it from the explicitly-Cayley
    call is the only check that cannot be fooled by a remembered convention: the two
    differ by only 3 percent at B-DNA twist, so no magnitude test distinguishes them.
    """
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams

    euler = CGNAPlusParams(POLY_A, aligned_strands=True, include_stiffness=False)
    cayley = CGNAPlusParams(POLY_A, aligned_strands=True, include_stiffness=False,
                            euler_definition=False, group_split=False)
    ge, gc = np.asarray(euler.gs, float), np.asarray(cayley.gs, float)
    names = list(euler.param_names)
    i = [k for k, n in enumerate(names) if n.startswith("S")][len(POLY_A) // 2]

    c = gc[i, :3]
    norm = float(np.linalg.norm(c))
    expected = c / norm * 2.0 * math.atan(norm / 2.0)
    np.testing.assert_allclose(ge[i, :3], expected, atol=1e-9)
    assert not np.allclose(ge[i, :3], c, atol=1e-6), (
        "the default coordinate has become a Cayley vector; mesoscale.helical_path "
        "and every direction-valued channel assume a rotation vector")


def test_poly_a_twist_is_the_reduced_a_tract_value(blocks):
    """Read as a rotation vector, the poly-A twist is near 34 degrees per step."""
    tw = float(np.degrees(blocks["step_gs"][:, 2]).mean())
    assert 32.0 < tw < 36.0, f"poly-A twist {tw:.2f} deg/step"


def test_rotvec_to_rotmat_is_a_rotation():
    for v in (np.zeros(3), np.array([0.1, -0.2, 0.6]), np.array([0.0, 0.0, 0.6028])):
        R = ms._rotvec_to_rotmat(v)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_rotvec_to_rotmat_recovers_the_angle():
    theta = math.radians(34.3)
    v = np.array([0.0, 0.0, theta])
    R = ms._rotvec_to_rotmat(v)
    recovered = math.acos((np.trace(R) - 1.0) / 2.0)
    assert math.degrees(recovered) == pytest.approx(34.3, abs=1e-9)


def test_helical_path_matches_the_library_chain_builder():
    """The reconstruction is pinned against `CGNAPlusConf`, not against a convention.

    This is the test that would have caught the Cayley/rotation-vector error and the
    mid-step/reference-frame error at the same time, and it is the reason both are now
    checked against the authority rather than against a docstring. Agreement has to be
    to machine precision: anything else means the junction convention has drifted.
    """
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams, CGNAPlusConf

    seq = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGC"
    p = CGNAPlusParams(seq, aligned_strands=True, include_stiffness=False)
    lib = np.asarray(CGNAPlusConf.from_params(p).bp_poses)[0]

    g = at.cgna_ground_state(seq, include_stiffness=True)
    path = ms.helical_path(ms.step_blocks(g))

    np.testing.assert_allclose(path["origins"], lib[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(path["frames"], lib[:, :3, :3], atol=1e-9)


# ======================================================================================
# The validating measurement
# ======================================================================================

def test_persistence_length_matches_measured_b_dna(blocks):
    """The headline validation: measured B-DNA bending persistence length is ~50 nm.

    This single number exercises the Cayley convention, the nm translation convention,
    the marginal covariance taken from the inverse stiffness, and l_p = 2a/<theta^2>.
    A 20 percent window is deliberately generous: the point is to catch a convention
    error, which would be wrong by a factor of 5 or 25, not to claim three digits.
    """
    lp = ms.elastic_moduli(blocks)["persistence_length_nm"]
    assert 40.0 < lp < 60.0, f"persistence length {lp:.1f} nm, measured B-DNA is ~50 nm"


def test_stretch_modulus_is_in_the_measured_range(blocks):
    """Measured B-DNA stretch modulus is roughly 1000-1500 pN."""
    s = ms.elastic_moduli(blocks)["stretch_modulus_pN"]
    assert 700.0 < s < 2200.0, f"stretch modulus {s:.0f} pN"


def test_rise_is_physical(blocks):
    assert 0.32 < ms.elastic_moduli(blocks)["rise_nm"] < 0.35


@pytest.mark.xfail(reason="known discrepancy, recorded in DECISIONS.md: the local "
                          "twist modulus from cgNA+ comes out near 40 nm, where "
                          "single-molecule measurements of the effective torsional "
                          "modulus give 75-110 nm. Whether that is a real "
                          "local-vs-effective distinction or an error in this "
                          "reduction is unresolved.",
                   strict=True)
def test_torsional_modulus_matches_single_molecule_measurements(blocks):
    c = ms.elastic_moduli(blocks)["torsional_modulus_nm"]
    assert 75.0 < c < 110.0, f"torsional modulus {c:.1f} nm"


def test_torsional_modulus_is_pinned_at_its_computed_value(blocks):
    """Guard the known value so a silent change upstream is noticed."""
    c = ms.elastic_moduli(blocks)["torsional_modulus_nm"]
    assert 30.0 < c < 55.0, f"torsional modulus {c:.1f} nm moved unexpectedly"


# ======================================================================================
# Closed-form mechanics
# ======================================================================================

def test_marko_siggia_is_monotone_and_bounded():
    prev = -1.0
    for f in (0.01, 0.1, 0.5, 2.0, 10.0, 50.0):
        x = ms.marko_siggia_extension(f, 50.0)
        assert 0.0 <= x < 1.0
        assert x > prev
        prev = x


def test_marko_siggia_satisfies_its_own_equation():
    lp, f = 50.0, 2.0
    x = ms.marko_siggia_extension(f, lp)
    lhs = f * lp / ms.KT_PN_NM
    rhs = x + 1.0 / (4.0 * (1.0 - x) ** 2) - 0.25
    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_osf_persistence_length_falls_with_salt():
    vals = [ms.osf_persistence_length(50.0, I) for I in (0.001, 0.01, 0.1, 1.0)]
    assert vals[0] > vals[1] > vals[2] > vals[3]
    assert all(v >= 50.0 for v in vals), "electrostatics can only stiffen the chain"


def test_osf_approaches_the_intrinsic_value_at_high_salt():
    assert ms.osf_persistence_length(50.0, 5.0) == pytest.approx(50.0, abs=0.5)


def test_loop_closure_cost_falls_with_loop_length():
    vals = [ms.loop_closure_free_energy(n, 50.0, 0.34) for n in (50, 100, 200, 400)]
    assert vals[0] > vals[1] > vals[2] > vals[3]


def test_loop_closure_matches_the_shimada_yamakawa_elastic_term():
    n, lp, rise = 100, 50.0, 0.34
    expected = 2.0 * math.pi**2 * lp / (n * rise)
    assert ms.loop_closure_free_energy(n, lp, rise, 0.0) == pytest.approx(expected)


def test_intrinsic_curvature_reduces_the_closure_cost():
    straight = ms.loop_closure_free_energy(100, 50.0, 0.34, net_bend_deg=0.0)
    bent = ms.loop_closure_free_energy(100, 50.0, 0.34, net_bend_deg=60.0)
    assert bent < straight


def test_cyclizability_oscillates_with_the_helical_period():
    """The twist-register term is what makes measured cyclizability oscillate at ~10.5
    bp, so an in-register length must beat an out-of-register one."""
    in_reg = ms.cyclizability(105, 50.0, 0.34, 100.0)      # 10 exact turns
    out_reg = ms.cyclizability(100, 50.0, 0.34, 100.0)     # 9.52 turns
    assert in_reg > out_reg


def test_twist_writhe_partition_sums_to_one():
    p = ms.twist_writhe_partition(-0.06, 50.0, 100.0)
    assert p["twist_fraction"] + p["writhe_fraction"] == pytest.approx(1.0)
    assert p["delta_twist"] + p["writhe"] == pytest.approx(-0.06)


def test_softer_torsion_puts_more_into_twist():
    soft = ms.twist_writhe_partition(-0.06, 50.0, 30.0)["twist_fraction"]
    stiff = ms.twist_writhe_partition(-0.06, 50.0, 200.0)["twist_fraction"]
    assert stiff > soft


# ======================================================================================
# Curvature and the window ladder
# ======================================================================================

def test_a_phased_tract_is_more_curved_than_a_mixed_sequence():
    """Phased A-tracts are the canonical intrinsically bent sequence."""
    phased = "".join("AAAAAA" + "CGCGC" for _ in range(6))[:60]
    mixed = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    b_ph = ms.step_blocks(at.cgna_ground_state(phased, include_stiffness=True))
    b_mx = ms.step_blocks(at.cgna_ground_state(mixed, include_stiffness=True))
    assert (ms.intrinsic_curvature(b_ph)["curvature_shortening"] >
            ms.intrinsic_curvature(b_mx)["curvature_shortening"])


def test_contour_length_tracks_rise_times_steps(blocks):
    c = ms.intrinsic_curvature(blocks)
    n = int(blocks["n_steps"])
    rise = ms.elastic_moduli(blocks)["rise_nm"]
    assert c["contour_length_nm"] == pytest.approx(n * rise, rel=0.05)


def test_end_to_end_never_exceeds_contour(blocks):
    c = ms.intrinsic_curvature(blocks)
    assert c["end_to_end_nm"] <= c["contour_length_nm"] + 1e-9


def test_window_ladder_lengths_are_consistent(gs):
    out = ms.mesoscale_window_channels(SEQ, gs)
    n_steps = int(ms.step_blocks(gs)["n_steps"])
    for name, arr in out.items():
        w = int(name.rsplit("_w", 1)[1])
        expected = n_steps - min(w, n_steps) + 1
        assert len(arr) == expected, f"{name}: {len(arr)} != {expected}"


def test_longer_windows_give_longer_apparent_persistence_length(gs):
    """Summing independent bending compliance over more steps cannot make the chain
    look floppier per unit length, so the windowed l_p must be stable, not divergent."""
    out = ms.mesoscale_window_channels(SEQ, gs, ladder=(10, 20))
    lp10 = np.nanmean(out["D_bend_lp_w10"])
    lp20 = np.nanmean(out["D_bend_lp_w20"])
    assert 0.5 < lp20 / lp10 < 2.0


# ======================================================================================
# Registry agreement
# ======================================================================================

def test_every_computed_family_d_channel_is_declared(gs):
    computed = set(ms.mesoscale_channels(SEQ, gs))
    declared = {s.name for s in REGISTRY.by_family("D")}
    assert computed <= declared, f"undeclared: {computed - declared}"
    assert declared <= computed, f"declared but not computed: {declared - computed}"


def test_cgna_stiffness_is_not_exactly_rc_symmetric():
    """Record the ground truth's own asymmetry, measured on an exact palindrome.

    On a palindrome RC(seq) == seq, so there is no index-mapping ambiguity: any mirror
    asymmetry is internal to cgNA+. The rotational ground-state coordinates are
    mirror-symmetric to machine precision; the marginal step covariances are not. The
    stiffness is correctly banded and the row extraction matches the parameter
    ordering, so this is a property of the parameter set.

    This test exists so the asymmetry is a recorded, monitored fact rather than a
    surprise, and so that a future upstream fix is noticed rather than silently
    changing every Family D target.
    """
    from channels import reverse_complement
    half = "ACGTTGCAATGCCGATTAGC"
    pal = half + reverse_complement(half)
    assert reverse_complement(pal) == pal

    b = ms.step_blocks(at.cgna_ground_state(pal, include_stiffness=True))
    # Under reverse complement tilt changes sign while roll and twist do not, which is
    # the 3DNA parity convention PLAN.md says to take rather than guess.
    gs_rot = b["step_gs"][:, :3]
    mirrored = gs_rot[::-1] * np.array([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        gs_rot, mirrored, atol=1e-9,
        err_msg="ground-state rotations should mirror exactly under the 3DNA parity")

    cov = np.diag(b["step_cov"]).reshape(-1, 6)
    rel = np.abs(cov - cov[::-1]) / np.abs(cov)
    tilt_max = float(rel[:, 0].max())
    assert tilt_max > 0.05, (
        "cgNA+ step covariances now look mirror-symmetric; if upstream fixed this, "
        "the symmetrisation in channels.symmetrise_channels can be revisited")
    assert tilt_max < 0.8, f"asymmetry grew unexpectedly: {tilt_max:.3f}"


def test_symmetrised_channels_obey_parity_exactly():
    """The blocking parity check, on the targets production actually uses.

    PLAN.md enforces reverse-complement equivariance architecturally, so a target that
    is not exactly symmetric is an error floor the model cannot reach. Symmetrising the
    target removes it, and this asserts the removal is exact rather than approximate.
    """
    import channels as ch_mod
    from channels import reverse_complement

    fwd = ms.mesoscale_channels(SEQ, symmetrise=True)
    rev = ms.mesoscale_channels(reverse_complement(SEQ), symmetrise=True)
    failures = []
    for spec in REGISTRY.by_family("D"):
        if spec.parity is Parity.STRANDED or spec.units in ("bp", "bp index"):
            continue
        a, b = fwd[spec.name], rev[spec.name]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ok, diff = ch_mod.check_parity([a], [b], spec, atol=1e-10, rtol=1e-10)
        if not ok:
            failures.append((spec.name, spec.parity.value, a, b, diff))
    assert not failures, "parity violations after symmetrisation:\n" + "\n".join(
        f"  {n}: declared {p}, fwd={x:.8g}, rev={y:.8g}, diff={d:.3g}"
        for n, p, x, y, d in failures)


def test_unsymmetrised_channels_do_not_obey_parity():
    """The negative control for the test above: without symmetrisation it must fail.

    If this ever passes, the symmetrisation is doing nothing and the test above is
    vacuous.
    """
    from channels import reverse_complement
    fwd = ms.mesoscale_channels(SEQ)
    rev = ms.mesoscale_channels(reverse_complement(SEQ))
    lp_f, lp_r = fwd["D_persistence_length_nm"], rev["D_persistence_length_nm"]
    assert lp_f != pytest.approx(lp_r, abs=1e-9), (
        "raw cgNA+ channels are unexpectedly RC-symmetric")


def test_symmetrisation_preserves_the_validated_persistence_length():
    lp = ms.mesoscale_channels(SEQ, symmetrise=True)["D_persistence_length_nm"]
    assert 40.0 < lp < 60.0


def test_provisional_channels_give_reasons():
    for s in REGISTRY.by_family("D"):
        if s.provisional:
            assert s.provisional_reason
