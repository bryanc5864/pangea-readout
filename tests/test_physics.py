#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Unit tests for the structure-building and sampling code.

These are the checks the plan calls for before anything downstream runs: frame
conventions, strand parity, and the geometry that a correlation-based validation would
never catch.  Run with:  python -m pytest tests/test_physics.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

import cgna_atomic as ca  # noqa: E402
# src/model/data.py (a regular module) would shadow the src/data namespace package
_MODEL_PATHS = [p for p in sys.path if os.path.normpath(p).endswith(os.path.join("src", "model"))]
for _p in _MODEL_PATHS:
    sys.path.remove(_p)
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data.fasta import Fasta, build_fai, gc_fraction, revcomp  # noqa: E402
from data.sample_fragments import (  # noqa: E402
    dinuc_counts, dinuc_shuffle, mono_counts, mono_shuffle,
)
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
sys.path[:0] = _MODEL_PATHS

SEQ = "GCGCAAATTTGCGCATATAGCGCAAATTTGCGC"


# --------------------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------------------


def test_kabsch_recovers_known_transform():
    rng = np.random.default_rng(0)
    ideal = rng.normal(size=(8, 3))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1.0]])
    t = np.array([1.5, -2.0, 0.25])
    actual = ideal @ R.T + t
    tau = ca.kabsch_se3(ideal, actual)
    assert np.allclose(tau[:3, :3], R, atol=1e-8)
    assert np.allclose(tau[:3, 3], t, atol=1e-8)
    assert np.allclose(ca.apply_se3(tau, ideal), actual, atol=1e-8)


def test_kabsch_is_proper_rotation_under_reflection():
    """A reflected target must still give det(R) = +1, not a mirror."""
    rng = np.random.default_rng(1)
    ideal = rng.normal(size=(10, 3))
    actual = ideal * np.array([1.0, 1.0, -1.0])
    tau = ca.kabsch_se3(ideal, actual)
    assert np.isclose(np.linalg.det(tau[:3, :3]), 1.0, atol=1e-8)


def test_se3_inverse_round_trip():
    rng = np.random.default_rng(2)
    ideal = rng.normal(size=(6, 3))
    actual = rng.normal(size=(6, 3))
    tau = ca.kabsch_se3(ideal, actual)
    inv = ca.invert_se3(tau)
    pts = rng.normal(size=(5, 3))
    assert np.allclose(ca.apply_se3(inv, ca.apply_se3(tau, pts)), pts, atol=1e-9)


# --------------------------------------------------------------------------------------
# Templates and structure
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def templates():
    return ca.NucleotideTemplates()


@pytest.fixture(scope="module")
def built(templates):
    gs = ca.CGNAGroundState(SEQ)
    atoms = ca.build_duplex(SEQ, templates, gs)
    return gs, atoms


def test_templates_cover_all_bases(templates):
    assert set(templates.templates) == set("ACGT")
    for letter, atoms in templates.templates.items():
        assert {"P", "OP1", "OP2", "O5'", "O3'", "C1'"} <= set(atoms), letter
        assert len(atoms) >= 29, letter


def test_units_and_crick_convention_detected(built):
    gs, _ = built
    # cgNA+ returns translations in nm under the settings we request.
    assert gs.unit_scale == 10.0
    # CGNAPlusConf requires aligned strands, so Crick frames arrive flipped.
    assert gs.crick_flipped is True


def test_backbone_bond_lengths(built):
    _, atoms = built
    rep = ca.geometry_report(atoms)
    assert 1.5 < rep["P_O5prime_mean"] < 1.7
    assert 1.5 < rep["O3prime_P_mean"] < 1.7
    assert 1.40 < rep["C1prime_N_glycosidic_mean"] < 1.60


def test_watson_crick_pair_geometry(built):
    """The decisive frame-convention test: bases must pair, not stack on each other."""
    _, atoms = built
    rep = ca.base_pair_report(atoms, len(SEQ))
    assert rep["n_pairs_checked"] == len(SEQ)
    assert 10.0 < rep["C1prime_C1prime_mean"] < 11.2
    assert rep["C1prime_C1prime_sd"] < 0.5
    assert 8.4 < rep["glycosidic_N_N_mean"] < 9.5


def test_no_atomic_clashes(built):
    _, atoms = built
    # The only sub-2 A inter-residue contact should be the O3'-P backbone bond.
    assert ca.min_heavy_distance(atoms) > 1.4


def test_structure_is_sequence_dependent(templates):
    """Two sequences of identical composition must give different geometry.

    This is the structural precursor of the composition gate: if cgNA+ geometry did not
    respond to base order, nothing downstream could either.
    """
    a = "ATATATATATATATATATATGCGCGCGCGCGCG"
    b = "ATGCATGCATGCATGCATGCATGCATGCATGCA"
    atoms_a = ca.build_duplex(a, templates, ca.CGNAGroundState(a))
    atoms_b = ca.build_duplex(b, templates, ca.CGNAGroundState(b))
    xa = np.array([at.xyz for at in atoms_a if at.name == "P" and at.chain == "A"])
    xb = np.array([at.xyz for at in atoms_b if at.name == "P" and at.chain == "A"])
    n = min(len(xa), len(xb))
    # Successive phosphate-phosphate steps differ between the two sequences.  An
    # end-to-end span is a poor probe because both are near-canonical B-DNA overall;
    # the sequence dependence lives in the local steps.
    step_a = np.linalg.norm(np.diff(xa[:n], axis=0), axis=1)
    step_b = np.linalg.norm(np.diff(xb[:n], axis=0), axis=1)
    assert float(np.abs(step_a - step_b).mean()) > 0.01
    rmsd = float(np.sqrt(((xa[:n] - xb[:n]) ** 2).sum(axis=1).mean()))
    assert rmsd > 0.1


def test_composition_matched_sequences_give_different_geometry(templates):
    """Same mononucleotide composition, different order, different structure."""
    import random

    rng = random.Random(7)
    seed = "".join(rng.choice("ACGT") for _ in range(33))
    other = mono_shuffle(seed, rng)
    assert mono_counts(seed) == mono_counts(other)
    assert seed != other
    pa = np.array([a.xyz for a in ca.build_duplex(seed, templates,
                                                  ca.CGNAGroundState(seed))
                   if a.name == "P"])
    pb = np.array([a.xyz for a in ca.build_duplex(other, templates,
                                                  ca.CGNAGroundState(other))
                   if a.name == "P"])
    n = min(len(pa), len(pb))
    rmsd = float(np.sqrt(((pa[:n] - pb[:n]) ** 2).sum(axis=1).mean()))
    assert rmsd > 0.05, "composition-matched sequences produced identical geometry"


def test_reverse_complement_parity(templates):
    """Building revcomp(seq) must give the same physical duplex with strands swapped.

    The pair-distance multiset is invariant under that relabelling, so it must match to
    within numerical tolerance.  A parity error in the frame handling breaks this.
    """
    rc = revcomp(SEQ)
    rep_f = ca.base_pair_report(ca.build_duplex(SEQ, templates,
                                                ca.CGNAGroundState(SEQ)), len(SEQ))
    rep_r = ca.base_pair_report(ca.build_duplex(rc, templates,
                                                ca.CGNAGroundState(rc)), len(rc))
    assert np.isclose(rep_f["C1prime_C1prime_mean"], rep_r["C1prime_C1prime_mean"],
                      atol=0.05)
    assert np.isclose(rep_f["glycosidic_N_N_mean"], rep_r["glycosidic_N_N_mean"],
                      atol=0.05)


def test_pdb_round_trip_frames(templates, tmp_path):
    """Frames re-fitted from the written PDB must reproduce the input base pairing."""
    import frames as frames_mod

    pdb = str(tmp_path / "f.pdb")
    atoms = ca.build_duplex(SEQ, templates, ca.CGNAGroundState(SEQ))
    ca.write_pdb(atoms, pdb)
    bp, _ = frames_mod.bp_frames_from_pdb(pdb)
    assert len(bp) == len(SEQ)
    for tau in bp:
        R = tau[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    # Consecutive base pairs are about one rise apart.
    rises = np.linalg.norm(np.diff(bp[:, :3, 3], axis=0), axis=1)
    assert 2.8 < float(np.median(rises)) < 4.2


def test_major_groove_is_plus_x(templates, tmp_path):
    """Phosphates must sit on the -x side of the base-pair frame."""
    import frames as frames_mod

    pdb = str(tmp_path / "g.pdb")
    atoms = ca.build_duplex(SEQ, templates, ca.CGNAGroundState(SEQ))
    ca.write_pdb(atoms, pdb)
    bp, _ = frames_mod.bp_frames_from_pdb(pdb)
    residues = frames_mod.read_pdb_residues(pdb)
    chk = frames_mod.groove_direction_check(bp, residues)
    assert chk, "groove check produced no projections"
    assert chk["major_groove_is_plus_x"] is True, chk


# --------------------------------------------------------------------------------------
# Sampling geometry
# --------------------------------------------------------------------------------------


def test_shell_points_radii_and_angles():
    import apbs_run

    tau = np.eye(4)[None, :, :]
    pts, angles = apbs_run.shell_points(tau, radial_offsets_A=(5.0, 10.0), n_sectors=4)
    assert pts.shape == (1, 4, 2, 3)
    for a in range(4):
        for r, rad in enumerate((5.0, 10.0)):
            assert np.isclose(np.linalg.norm(pts[0, a, r]), rad, atol=1e-9)
    # Sector 0 along +x, sector 1 along +y for 4 sectors.
    assert np.allclose(pts[0, 0, 0], [5.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(pts[0, 1, 0], [0.0, 5.0, 0.0], atol=1e-9)


def test_central_slice():
    import run_fragment

    sl = run_fragment.central_slice(61, 21)
    assert (sl.start, sl.stop) == (20, 41)
    assert run_fragment.central_slice(10, 21) == slice(0, 10)


def test_dxmap_trilinear_interpolation(tmp_path):
    """Interpolate an exactly linear field; trilinear must be exact."""
    import apbs_run

    nx, ny, nz = 5, 5, 5
    origin = np.array([-2.0, -2.0, -2.0])
    delta = np.array([1.0, 1.0, 1.0])
    xs = origin[0] + delta[0] * np.arange(nx)
    ys = origin[1] + delta[1] * np.arange(ny)
    zs = origin[2] + delta[2] * np.arange(nz)
    data = np.zeros((nx, ny, nz))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                data[i, j, k] = 2.0 * x - 3.0 * y + 0.5 * z + 1.0

    path = tmp_path / "t.dx"
    with open(path, "w") as f:
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write(f"origin {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"delta {delta[0]} 0 0\ndelta 0 {delta[1]} 0\ndelta 0 0 {delta[2]}\n")
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n")
        flat = data.ravel()
        for i in range(0, len(flat), 3):
            f.write(" ".join(f"{v:.10g}" for v in flat[i:i + 3]) + "\n")

    m = apbs_run.DXMap.read(str(path))
    assert m.data.shape == (nx, ny, nz)
    probe = np.array([[0.3, -0.7, 1.2], [1.0, 1.0, 1.0]])
    expect = 2.0 * probe[:, 0] - 3.0 * probe[:, 1] + 0.5 * probe[:, 2] + 1.0
    assert np.allclose(m.interpolate(probe), expect, atol=1e-9)
    # Outside the grid returns NaN rather than a silently wrong number.
    assert np.isnan(m.interpolate(np.array([[99.0, 0.0, 0.0]])))[0]


def test_ionic_conditions_well_formed():
    import apbs_run

    assert len(apbs_run.IONIC_CONDITIONS) == 9
    for c in apbs_run.IONIC_CONDITIONS:
        net = sum(i.charge * i.concentration_M for i in c.ions)
        assert abs(net) < 1e-9, f"{c.name} is not electroneutral"
        assert c.ionic_strength_M > 0
    mg = apbs_run.CONDITIONS_BY_NAME["NaCl_150mM_MgCl2_2mM"]
    assert mg.approximate is True


# --------------------------------------------------------------------------------------
# Sequence utilities
# --------------------------------------------------------------------------------------


def test_dinuc_shuffle_preserves_composition():
    import random

    rng = random.Random(3)
    for _ in range(25):
        seq = "".join(rng.choice("ACGT") for _ in range(61))
        s = dinuc_shuffle(seq, rng)
        assert s is not None
        assert mono_counts(s) == mono_counts(seq)
        assert dinuc_counts(s) == dinuc_counts(seq)
        assert s[0] == seq[0] and s[-1] == seq[-1]


def test_revcomp_involution():
    rng = np.random.default_rng(4)
    seq = "".join(rng.choice(list("ACGT"), size=50))
    assert revcomp(revcomp(seq)) == seq


def test_fasta_random_access(tmp_path):
    seq = "".join(np.random.default_rng(5).choice(list("ACGT"), size=503))
    p = tmp_path / "x.fa"
    with open(p, "w") as f:
        f.write(">chr1 test\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i + 60] + "\n")
    with Fasta(str(p)) as fa:
        assert fa.length("chr1") == len(seq)
        assert fa.fetch("chr1", 0, 10) == seq[:10]
        assert fa.fetch("chr1", 61, 130) == seq[61:130]
        assert fa.fetch("chr1", 500, 600) == seq[500:]
    assert np.isclose(gc_fraction("GGCC"), 1.0)
    assert np.isclose(gc_fraction("ATAT"), 0.0)
