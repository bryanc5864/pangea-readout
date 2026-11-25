# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A assembly: exact reverse-complement conventions on a synthetic solver output."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
A = pytest.importorskip("electrostatics_family", reason="physics environment not active")
from channels import REGISTRY, Parity, Resolution  # noqa: E402

CONDS = ["NaCl_050mM", "NaCl_150mM", "NaCl_500mM", "KCl_150mM", "NaCl_150mM_MgCl2_2mM"]


def _fake(rng):
    c, n = len(CONDS), 21
    z = {"conditions": np.array(CONDS),
         "potential_surface": rng.normal(size=(c, n, 6, 6)),
         "ext_field": rng.normal(size=(c, n, 2, 3)), "ext_dEr_dr": rng.normal(size=(c, n, 2)),
         "ext_probe_kT": rng.normal(size=(c, n, 2)),
         "ext_condensed_fraction": rng.normal(size=(c, n, 2)),
         "ext_phosphate_potential": rng.normal(size=(c, n, 2)),
         "ext_groove_log10_enrichment": rng.normal(size=(c, n, 2, 3)),
         "ext_backbone_log10_enrichment": rng.normal(size=(c, n, 2, 3)),
         "ext_manning_xi": rng.normal(size=n), "ext_surface_charge_e_per_A2": rng.normal(size=n),
         "ext_bp_solvation_kJ_mol": rng.normal(size=(c, n)),
         "elec_energy_kJ_mol": rng.normal(size=c), "elec_energy_ref_kJ_mol": rng.normal(size=c)}
    return z


def _rc(z):
    """The exact solver output of the reverse complement: bp reversed, sector s -> 6-s,
    Watson <-> Crick, tangential and axial field negated."""
    sec = [0, 5, 4, 3, 2, 1]
    y = dict(z)
    y["potential_surface"] = z["potential_surface"][:, ::-1][:, :, sec]
    f = z["ext_field"][:, ::-1].copy(); f[..., 1:] *= -1; y["ext_field"] = f
    for k in ("ext_dEr_dr", "ext_probe_kT", "ext_groove_log10_enrichment", "ext_bp_solvation_kJ_mol"):
        y[k] = z[k][:, ::-1]
    for k in ("ext_condensed_fraction", "ext_phosphate_potential", "ext_backbone_log10_enrichment"):
        y[k] = z[k][:, ::-1][:, :, ::-1]
    for k in ("ext_manning_xi", "ext_surface_charge_e_per_A2"):
        y[k] = z[k][::-1]
    return y


def test_every_declared_channel_assembles():
    p, g = A.assemble(_fake(np.random.default_rng(0)))
    assert set(p) == {s.name for s in REGISTRY.by_family("A") if s.resolution is Resolution.PER_BP}
    assert set(g) == {s.name for s in REGISTRY.by_family("A") if s.resolution is Resolution.GLOBAL}
    assert np.isnan(p["A_s0_pot_d0_c150"][:20]).all() and np.isfinite(p["A_s0_pot_d0_c150"][20:41]).all()
    assert np.isnan(p["A_s0_pot_d0_c010"]).all()          # a condition not solved stays NaN


def test_reverse_complement_identities_are_exact():
    z = _fake(np.random.default_rng(1))
    px, gx = A.assemble(z)
    py, gy = A.assemble(_rc(z))
    pairs = {"A_s5_": "A_s1_", "A_s4_": "A_s2_", "A_C_": "A_W_"}
    checked = 0
    for name, v in px.items():
        if not np.isfinite(v).any():
            continue
        partner = next((pairs[q] + name[len(q):] for q in pairs if name.startswith(q)), None)
        par = REGISTRY[name].parity
        if partner:                                       # own frame: no reflection
            np.testing.assert_allclose(v, py[partner], atol=1e-12, err_msg=name); checked += 1
        elif par is Parity.SYMMETRIC:
            np.testing.assert_allclose(v, py[name][::-1], atol=1e-12, err_msg=name); checked += 1
        elif par is Parity.ANTISYMMETRIC:
            np.testing.assert_allclose(v, -py[name][::-1], atol=1e-12, err_msg=name); checked += 1
    assert checked > 150
    for k in gx:
        assert gx[k] == pytest.approx(gy[k], nan_ok=True)


def test_ensemble_mean_and_spread_match_the_snapshots():
    rng = np.random.default_rng(11)
    snaps = [_fake(rng) for _ in range(5)]
    p, g = A.assemble_ensemble(snaps)
    singles = [A.assemble(z) for z in snaps]
    for base in A.ENSEMBLE_PER_BP:
        st = np.stack([s[0][base] for s in singles])
        np.testing.assert_allclose(p[f"{base}_ensmean"], np.nanmean(st, 0), equal_nan=True)
        np.testing.assert_allclose(p[f"{base}_enssd"], np.nanstd(st, 0, ddof=1), equal_nan=True)
    # own-frame storage carries over (sector 5 and Crick are reversed like their parents)
    assert np.isnan(p["A_s5_pot_d1_c150_ensmean"][:20]).all()
    v = np.array([s[1]["A_solvation_c150"] for s in singles])
    assert np.isclose(g["A_solvation_c150_ensmean"], v.mean())
    # too few snapshots: no ensemble values
    p3, g3 = A.assemble_ensemble(snaps[:3])
    assert np.isnan(p3["A_bpsolv_c150_ensmean"]).all() and np.isnan(g3["A_solvation_c150_enssd"])
    # a single ground-state solve declares them but leaves them NaN
    p1, _ = A.assemble(snaps[0])
    assert np.isnan(p1["A_maj_Er_c150_ensmean"]).all()


def test_ensemble_channels_keep_the_strand_tie():
    rng = np.random.default_rng(12)
    snaps = [_fake(rng) for _ in range(4)]
    p, _ = A.assemble_ensemble(snaps)
    q, _ = A.assemble_ensemble([_rc(z) for z in snaps])
    # A_s5(x) = A_s1(rc x) and A_C(x) = A_W(rc x), position by position, for mean and spread
    for a, b in (("A_s1_pot_d1_c150", "A_s5_pot_d1_c150"), ("A_W_condensed_c150", "A_C_condensed_c150")):
        for stat in ("ensmean", "enssd"):
            np.testing.assert_allclose(p[f"{b}_{stat}"], q[f"{a}_{stat}"], equal_nan=True, atol=1e-12)


def test_multi_pass_solves_merge_by_condition(tmp_path):
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "src", "data"))
    import generate_family_a_long_shards as LS_
    rng = np.random.default_rng(13)
    z = _fake(rng)
    z["sequence"] = np.array(["ACGT" * 30 + "A"]); z["bp_origins"] = rng.normal(size=(21, 3))
    a = {k: v for k, v in z.items()}
    keep = [0, 1]                                  # pass 1: NaCl_050mM, NaCl_150mM
    for k, v in z.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(CONDS):
            a[k] = v[keep]
    a["conditions"] = np.array([CONDS[i] for i in keep])
    b = {k: v for k, v in z.items()}
    for k, v in z.items():                        # pass 2: NaCl_500mM only, plus a new array
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(CONDS):
            b[k] = v[[2]]
    b["conditions"] = np.array([CONDS[2]])
    b["ext_bp_solvation_map_kJ_mol"] = rng.normal(size=(1, 21))
    np.savez(tmp_path / "p1.npz", **a); np.savez(tmp_path / "p2.npz", **b)
    m = LS_.load_merged([str(tmp_path / "p1.npz"), str(tmp_path / "p2.npz")])
    assert [str(c) for c in m["conditions"]] == CONDS[:3]
    np.testing.assert_allclose(m["potential_surface"], z["potential_surface"][:3])
    assert np.isnan(m["ext_bp_solvation_map_kJ_mol"][:2]).all()
    np.testing.assert_allclose(m["ext_bp_solvation_map_kJ_mol"][2], b["ext_bp_solvation_map_kJ_mol"][0])
    b2 = dict(b); b2["bp_origins"] = b["bp_origins"] + 1.0     # a different structure is refused
    np.savez(tmp_path / "p3.npz", **b2)
    with pytest.raises(ValueError, match="structures differ"):
        LS_.load_merged([str(tmp_path / "p1.npz"), str(tmp_path / "p3.npz")])


def test_supplementary_pass_fills_missing_solvation(tmp_path):
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "src", "data"))
    import generate_family_a_long_shards as LS_
    rng = np.random.default_rng(14)
    z = _fake(rng)
    z["sequence"] = np.array(["ACGT" * 30 + "A"]); z["bp_origins"] = rng.normal(size=(21, 3))
    a = dict(z); a["ext_bp_solvation_kJ_mol"] = z["ext_bp_solvation_kJ_mol"].copy()
    a["ext_bp_solvation_kJ_mol"][1] = np.nan                  # condition 1's energy had failed
    b = {k: v for k, v in z.items()}
    for k, v in z.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(CONDS):
            b[k] = v[[1]]
    b["conditions"] = np.array([CONDS[1]])
    b["ext_bp_solvation_map_kJ_mol"] = rng.normal(size=(1, 21))
    np.savez(tmp_path / "a.npz", **a); np.savez(tmp_path / "b.npz", **b)
    m = LS_.load_merged([str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    assert [str(c) for c in m["conditions"]] == CONDS
    np.testing.assert_allclose(m["ext_bp_solvation_kJ_mol"][1], z["ext_bp_solvation_kJ_mol"][1])
    np.testing.assert_allclose(m["ext_bp_solvation_map_kJ_mol"][1], b["ext_bp_solvation_map_kJ_mol"][0])
    assert np.isnan(m["ext_bp_solvation_map_kJ_mol"][0]).all()
