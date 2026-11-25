#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The registry contract, checked across every family at once.

This file exists because of a gap that each family's own tests could not see. Families F
and L emitted per-base-pair profiles -- the G4Hunter track, the i-motif track, the Z-DNA
step energy, and the RNA unpaired-probability and positional-entropy tracks -- that were
never declared in the channel registry. Families C and E had per-family contract tests
and so were clean; D, F and L had none.

The consequence was not cosmetic. `scripts/generate_targets.py` builds the column set of
the training data **from the registry**, so seven real per-base-pair tracks would have
been silently absent from every training example, and nothing in the pipeline would have
reported a problem. The declared-but-not-emitted direction is worse still: it would put
a permanently-NaN column into the training set.

So the contract is checked here in both directions, over all six families together,
against the single entry point every consumer uses.
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
from channels import REGISTRY, Parity, Resolution  # noqa: E402

# A 61 bp fragment, the length every sequence in library_v2 has.
SEQ = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGCCCCTCCCTCCCTCCCAAATTTGCGCATA"
assert len(SEQ) == 61


@pytest.fixture(scope="module")
def computed():
    return LS.compute_landscape(SEQ, n_ensemble=0)


def test_every_emitted_channel_is_declared(computed):
    prof, glob = computed
    declared = {s.name for s in REGISTRY.specs()}
    undeclared = sorted((set(prof) | set(glob)) - declared)
    assert not undeclared, (
        f"emitted but never declared, so the target generator would drop them: "
        f"{undeclared}")


def test_every_declared_channel_is_emitted(computed):
    prof, glob = computed
    emitted = set(prof) | set(glob)
    missing = sorted({s.name for s in REGISTRY.specs() if s.family in LS.FAMILIES}
                     - emitted)
    assert not missing, (
        f"declared but never emitted, so these would be permanently-NaN columns in the "
        f"training set: {missing}")


def test_declared_resolution_matches_where_the_channel_appears(computed):
    prof, glob = computed
    for s in REGISTRY.specs():
        if s.family not in LS.FAMILIES:
            continue
        if s.resolution is Resolution.PER_BP:
            assert s.name in prof, f"{s.name} declared PER_BP but is a scalar"
            assert s.name not in glob, f"{s.name} is emitted at two resolutions"
        elif s.resolution is Resolution.GLOBAL:
            assert s.name in glob, f"{s.name} declared GLOBAL but is a profile"
            assert s.name not in prof, f"{s.name} is emitted at two resolutions"


def test_no_duplicate_channel_names_across_families():
    names = [s.name for s in REGISTRY.specs()]
    assert len(names) == len(set(names))


def test_every_channel_name_is_prefixed_by_its_family():
    for s in REGISTRY.specs():
        assert s.name.startswith(f"{s.family}_"), (
            f"{s.name} is declared in family {s.family} but is not prefixed by it")


def test_every_production_channel_names_a_source():
    """A channel fit to be a training target has to say where its parameters came from."""
    for s in REGISTRY.specs():
        if s.family in LS.FAMILIES and not s.provisional:
            assert s.source, f"{s.name} is a production channel with no source"


def test_every_provisional_channel_says_why():
    for s in REGISTRY.specs():
        if s.provisional:
            assert len(s.provisional_reason) > 40, s.name


# ======================================================================================
# The packed arrays the generator writes
# ======================================================================================

def test_column_order_is_deterministic():
    """Column order must not depend on import order, or a dataset silently transposes."""
    a = LS.channel_names(Resolution.PER_BP)
    b = LS.channel_names(Resolution.PER_BP)
    assert a == b == sorted(a)


def test_packed_arrays_have_the_declared_shape():
    pb = LS.channel_names(Resolution.PER_BP)
    gl = LS.channel_names(Resolution.GLOBAL)
    P, G = LS.landscape_arrays(SEQ, pb, gl, n_ensemble=0)
    assert P.shape == (len(pb), len(SEQ)), P.shape
    assert G.shape == (len(gl),), G.shape
    assert P.dtype == np.float32 and G.dtype == np.float32


def test_packed_arrays_agree_with_the_dict_form():
    pb = LS.channel_names(Resolution.PER_BP)
    gl = LS.channel_names(Resolution.GLOBAL)
    prof, glob = LS.compute_landscape(SEQ, n_ensemble=0)
    P, G = LS.landscape_arrays(SEQ, pb, gl, n_ensemble=0)
    for i, name in enumerate(pb[:20]):
        np.testing.assert_allclose(P[i], np.asarray(prof[name], dtype=np.float32),
                                   rtol=1e-6, atol=1e-6)
    for i, name in enumerate(gl[:20]):
        a, b = G[i], np.float32(glob[name])
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-5 * max(abs(b), 1.0)


def test_most_channels_are_not_all_nan_on_a_real_fragment():
    """A fragment should light up most of the landscape.

    Without a minimised structure, and with the ensemble off, a known set of channels is
    legitimately NaN: the backbone torsions, the hydrogen-bond distances, the ensemble
    standard deviations, sugar pucker, and the salt and temperature condition deltas. The
    bound below is loose on purpose -- the point is to catch a family that has stopped
    computing anything at all, which a per-family test would not notice if the family
    were simply never called.
    """
    pb = LS.channel_names(Resolution.PER_BP)
    gl = LS.channel_names(Resolution.GLOBAL)
    P, G = LS.landscape_arrays(SEQ, pb, gl, n_ensemble=0)
    live_p = float(np.mean([np.isfinite(row).any() for row in P]))
    live_g = float(np.mean(np.isfinite(G)))
    assert live_p > 0.6, f"only {live_p:.0%} of per-bp channels produced any value"
    assert live_g > 0.8, f"only {live_g:.0%} of global channels produced a value"


def test_every_family_contributes_channels(computed):
    prof, glob = computed
    emitted = set(prof) | set(glob)
    for fam in LS.FAMILIES:
        got = [n for n in emitted if n.startswith(f"{fam}_")]
        assert got, f"family {fam} contributed nothing"


def test_the_registry_is_the_size_the_plan_implies():
    """PLAN.md budgets 215 + 98 + 90 + 85 + 53 + 60 for C, B, D, E, F, L.

    Family B is deliberately short: only the GFN2 tier has been computed, and the SAPT
    decomposition, the DFT tiers, transfer integrals, molecular electrostatic potentials
    and pKa shifts are not declared at all rather than declared empty.
    """
    counts = {f: len(REGISTRY.by_family(f)) for f in LS.FAMILIES}
    assert counts["C"] == 212 and counts["E"] == 89 and counts["D"] == 52
    assert counts["F"] == 63 and counts["L"] == 78
    assert counts["B"] == 34, (
        f"Family B has {counts['B']} channels against the plan's 98; if the DFT or SAPT "
        "tiers have been added this bound is stale")


# ======================================================================================
# Import order
# ======================================================================================

def test_importing_landscape_alone_populates_the_registry():
    """The registry must be complete the moment `landscape` is importable.

    Every family module registers its channels as a side effect of being imported. When
    `landscape` imported them lazily inside `compute_landscape`, a caller that asked for
    the column set *first* -- which is precisely what the target generator does -- saw an
    empty registry. The generator built a dataset with zero channels, wrote it to disk,
    and reported success. Nothing raised.

    This runs the check in a subprocess so it is a genuine fresh interpreter: inside this
    test session every family has already been imported by other tests, which is exactly
    the condition that would hide the bug.
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import landscape as LS\n"
        "from channels import Resolution\n"
        "p = len(LS.channel_names(Resolution.PER_BP))\n"
        "g = len(LS.channel_names(Resolution.GLOBAL))\n"
        "print(p, g)\n"
        "assert p > 200 and g > 250, (p, g)\n"
        % (os.path.join(ROOT, "src", "physics"), os.path.join(ROOT, "src"))
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "importing `landscape` alone did not populate the channel registry:\n"
        f"stdout={r.stdout}\nstderr={r.stderr[-800:]}")


def test_generator_refuses_an_empty_channel_set(monkeypatch):
    """The generator's second line of defence against the same failure."""
    sys.path.insert(0, os.path.join(ROOT, "src", "data"))
    import generate_targets as gt

    monkeypatch.setattr(gt.LS, "channel_names", lambda *a, **k: [])
    monkeypatch.setattr(gt, "read_library", lambda *a, **k: (["x"], ["ACGT" * 15], ["s"]))
    monkeypatch.setattr(sys, "argv",
                        ["generate_targets.py", "--library", "x", "--out", "y"])
    with pytest.raises(RuntimeError, match="registry is empty or partial"):
        gt.main()


# ======================================================================================
# The split has to actually handle what the QC gate waves through
# ======================================================================================

def test_similarity_split_colocates_identical_sequences():
    """The QC gate downgrades duplicate sequences from a problem to a warning on the
    grounds that the similarity split assigns identical sequences to the same side.
    That claim is load-bearing, so it is tested rather than asserted in a message.

    Three sequences in the 191,344-fragment library appear twice, where the Drosophila
    genomic panel and the DeepSTARR MPRA elements reach the same locus by different
    routes. Under a random split those would leak; under the similarity split they must
    not.
    """
    sys.path.insert(0, os.path.join(ROOT, "src", "data"))
    sys.path.insert(0, os.path.join(ROOT, "src", "model"))
    import splits as sp

    rng = np.random.default_rng(0)
    base = ["".join(rng.choice(list("ACGT"), 61)) for _ in range(40)]
    # Plant three exact duplicates, as the real library contains.
    seqs = base + [base[3], base[17], base[29]]

    labels, _ = sp.group_by_similarity(seqs, threshold=0.5, seed=0)
    for original, copy in ((3, 40), (17, 41), (29, 42)):
        assert labels[original] == labels[copy], (
            f"identical sequences at {original} and {copy} landed in different groups, "
            "so the similarity split would leak them between train and validation and "
            "the QC gate's warning-not-problem treatment of duplicates is unsound")

# `test_similarity_split_puts_whole_groups_on_one_side` lives in tests/test_pangea.py:
# it imports `train`, which imports torch, and this file runs in the physics environment
# where torch is deliberately absent. The grouping half of the claim is tested above,
# which needs only `splits` and numpy.
