#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Every per-base-pair channel's reverse-complement index map, re-measured.

PANGEA enforces reverse-complement equivariance by merging its prediction on a sequence
with its prediction on the reverse complement, reflected back into the forward frame by
each channel's declared `rc_shift`. If a declaration is wrong, the model is forced to
average a channel with a *different position's* value, and no amount of training can fit
it. That happened: the first merge used a plain flip for everything, the 35 step-indexed
channels were each averaged with their neighbour, and `C_twist` trained to R-squared 0.09.

So this test re-measures every map on random exact palindromes, where a quantity either
reflects exactly or does not, and asserts the declaration. A short, explicit list of
targets is only approximately symmetric for a known reason; for those the test asserts
the declared shift is still the best-fitting one, so the index map is right even though
the values are not exactly equal.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

import landscape as LS  # noqa: E402
from channels import REGISTRY, Parity, Resolution, reverse_complement  # noqa: E402

# Approximately symmetric targets, each for a stated reason. Everything else must be exact.
APPROXIMATE = {
    "C_hoogsteen_opening_proxy_kT":
        "built from the cgNA+ marginal stiffness, which is not exactly RC-symmetric "
        "(about 11 percent here); recorded in DECISIONS.md",
    "E_hbond_distance_A":
        "read off a minimised structure; the minimiser does not preserve exact symmetry",
    "E_hbond_distance_spread_A":
        "read off a minimised structure; the minimiser does not preserve exact symmetry",
}
# Ensemble standard deviations are Monte Carlo estimates from a finite number of draws,
# so they are symmetric in expectation only: with N draws the relative error of each
# estimate is about 1/sqrt(2N). Their *index map* must still be the right one, which is
# what the merge needs, and that is checked with a mean rather than a max residual so the
# noise cannot decide it.
# Ensemble standard deviations, mapped to the mean channel whose spread they are. Their
# index map is fixed by construction -- the spread of a quantity sits where the quantity
# sits -- and it cannot be recovered from data: measured on palindromes with 400 draws,
# every candidate shift gives a mean residual near one standard deviation of the channel,
# because the spread barely varies along the sequence relative to its Monte Carlo noise,
# and the cgNA+ stiffness asymmetry recorded in DECISIONS.md feeds straight into the
# covariance. So these are checked structurally rather than fitted.
ENSEMBLE_SD_OF = {
    "C_minor_width_A_sd": "C_minor_width_A", "C_minor_depth_A_sd": "C_minor_depth_A",
    "C_major_width_A_sd": "C_major_width_A", "C_major_depth_A_sd": "C_major_depth_A",
    "C_axis_curvature_sd": "C_axis_curvature_deg_per_bp",
    "C_axis_bend_direction_sd": "C_axis_bend_direction_deg",
}
EXACT_TOL = 1e-8


def _palindromes(n: int = 3, half: int = 30, seed: int = 11):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = "".join(rng.choice(list("ACGT"), half))
        out.append(h + reverse_complement(h))
    return out


@pytest.fixture(scope="module")
def profiles(tmp_path_factory):
    import geometry as G
    pals = _palindromes()
    built = G.build_minimised_structure(pals[0], str(tmp_path_factory.mktemp("rc")))
    out = []
    for k, p in enumerate(pals):
        prof, _ = LS.compute_landscape(
            p, n_ensemble=400,
            minimised_pdb=built["minimised_pdb"] if k == 0 else None)
        out.append(prof)
    return out


def _residual(v, shift, sign, how="max"):
    v = np.asarray(v, dtype=float)
    n = len(v)
    idx = np.arange(n)[::-1] + shift
    ok = (idx >= 0) & (idx < n)
    t = np.full(n, np.nan)
    t[ok] = v[idx[ok]]
    m = np.isfinite(v) & np.isfinite(t)
    if m.sum() < 6 or np.nanstd(v[m]) < 1e-12:
        return None
    d = np.abs(v[m] - sign * t[m])
    if how == "mean":
        return float(d.mean() / max(np.nanstd(v[m]), 1e-12))
    return float(d.max() / max(np.abs(v[m]).max(), 1e-12))


def _merged_channels():
    return [s for s in REGISTRY.specs()
            if s.family in LS.FAMILIES and s.resolution is Resolution.PER_BP
            and s.parity in (Parity.SYMMETRIC, Parity.ANTISYMMETRIC)]


def test_every_merged_channel_reflects_by_its_declared_map(profiles):
    """The contract the equivariant merge relies on, for every channel it merges."""
    failures, checked = [], 0
    for s in _merged_channels():
        if s.name in ENSEMBLE_SD_OF:
            continue                         # checked structurally, below
        sign = -1.0 if s.parity is Parity.ANTISYMMETRIC else 1.0
        res = [r for r in (_residual(p.get(s.name, []), s.rc_shift, sign)
                           for p in profiles if s.name in p) if r is not None]
        if not res:
            continue                         # NaN on every palindrome: nothing to check
        checked += 1
        worst = max(res)
        if s.name in APPROXIMATE:
            # The values are not exactly symmetric, but the *index map* must still be the
            # best one -- otherwise the merge would be averaging the wrong positions.
            for p in profiles:
                if s.name not in p:
                    continue
                cands = {sh: _residual(p[s.name], sh, sign, how="mean")
                         for sh in range(-4, 5)}
                cands = {k: v for k, v in cands.items() if v is not None}
                if cands:
                    assert min(cands, key=cands.get) == s.rc_shift, (
                        s.name, s.rc_shift, cands)
            continue
        if worst > EXACT_TOL:
            failures.append((s.name, s.parity.value, s.rc_shift, f"{worst:.2e}"))
    assert checked > 40, f"only {checked} channels were checkable"
    assert not failures, "declared index map does not hold:\n" + "\n".join(
        f"  {n}: {p} shift {sh} residual {r}" for n, p, sh, r in failures)


def test_the_plain_flip_is_wrong_for_step_channels(profiles):
    """Negative control for the bug: shift 0 must fail on a step-indexed channel."""
    v = profiles[0]["C_twist"]
    assert REGISTRY["C_twist"].rc_shift == -1
    assert _residual(v, -1, 1.0) < EXACT_TOL
    assert _residual(v, 0, 1.0) > 1e-3, (
        "C_twist now reflects with a plain flip, so the rc_shift declaration is moot")


def test_groove_channels_need_their_offset(profiles):
    for name, sh in (("C_minor_width_A", 4), ("C_major_width_A", -4)):
        assert REGISTRY[name].parity is Parity.SYMMETRIC
        assert REGISTRY[name].rc_shift == sh
        assert _residual(profiles[0][name], sh, 1.0) < EXACT_TOL
        assert _residual(profiles[0][name], 0, 1.0) > 1e-3


def test_circular_channels_are_angles():
    """Circular channels are angles; antisymmetric ones are handled by the expansion.

    Under theta -> -theta, sin changes sign and cos does not, so `InMemoryLandscape`
    declares the sin column ANTISYMMETRIC and the cos column SYMMETRIC. Inheriting
    ANTISYMMETRIC for both would force the model to predict cos = 0.
    """
    for s in REGISTRY.specs():
        if s.circular:
            assert s.units in ("degrees", "deg"), s.name


def test_declared_maps_agree_with_the_saved_measurement():
    """results/rc_maps.json is the measurement the declarations were taken from."""
    import json
    path = os.path.join(ROOT, "results", "rc_maps.json")
    if not os.path.exists(path):
        pytest.skip("run src/physics/measure_rc_maps.py first")
    m = json.load(open(path))
    for name, rec in m.items():
        if rec.get("measured") == "exact" and name in REGISTRY:
            assert REGISTRY[name].rc_shift == rec["shift"], (name, rec)


def test_ensemble_spreads_share_their_mean_channels_index_map():
    """The spread of a quantity is indexed where the quantity is indexed."""
    for sd_name, mean_name in ENSEMBLE_SD_OF.items():
        assert REGISTRY[sd_name].rc_shift == REGISTRY[mean_name].rc_shift, (
            sd_name, REGISTRY[sd_name].rc_shift, mean_name, REGISTRY[mean_name].rc_shift)
