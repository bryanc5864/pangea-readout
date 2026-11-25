#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, stage 3: electrostatics of a modelled quadruplex, central ion channel included.

The minimised model (no ions) is charged with pdb2pqr (AMBER) and solved with the same
nonlinear PB construction as Family A (production exclusion radius per condition, exact
0.35 A fine spacing). The stack axis and tetrad planes are re-derived from the minimised
model's own Hoogsteen contacts. Sampled:

* along the axis at 0.25 A steps from 8 A beyond one outer tetrad to 8 A beyond the
  other -- the channel profile; summarised at the tetrad planes, the inter-tetrad ion
  sites (midway between planes), the two outer sites (half a rise outside each outer
  plane, 5' and 3' faces), its minimum and maximum, and 6 A out on either face;
* the four grooves (midway in azimuth between neighbouring guanine columns, 8 A from the
  axis, at each tetrad level) -- mean, minimum and maximum;
* each nucleotide 3 A outward (radially) from its base centroid -- per-residue potential;
* loop nucleotides averaged.

Plus per-atom solvation (solvated minus reference on the same grid; the nonlinear total
energy is not used, DECISIONS.md), the dipole about the centre of geometry, radius of
gyration, and loop end-to-end spans.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import apbs_run as A  # noqa: E402
import g4_templates as T  # noqa: E402

DEBYE_PER_E_A = 4.80320
AXIS_STEP_A = 0.25
AXIS_BEYOND_A = 8.0
GROOVE_RADIUS_A = 8.0
RESIDUE_OUT_A = 3.0


def model_geometry(pdb: str, tetrads=None) -> Dict[str, object]:
    """Stack axis, planes and grooves of a model; `tetrads` (lists of residue keys) reuses the
    tetrads of the minimised model for an MD frame, whose thermally stretched Hoogsteen contacts
    can fall outside the detection cutoff (probe: one frame read as a single tetrad)."""
    atoms, _ = T.read_pdb(pdb)
    res = T.residues(atoms)
    tet = tetrads if tetrads is not None else T.find_tetrads(res)
    if len(tet) < 2:
        raise ValueError(f"{pdb}: {len(tet)} tetrads after minimisation")
    g = T.tetrad_geometry(res, tet)
    axis, cents = np.asarray(g["axis"], float), np.asarray(g["centroids"], float)
    # orient the axis 5' -> 3': the tetrad holding the lowest-numbered guanine is the 5' face
    first = min(range(len(g["order"])), key=lambda i: min(k[1] for k in g["order"][i]))
    if first != 0:
        axis = -axis
        cents = cents[::-1]; g["order"] = g["order"][::-1]
    return {"res": res, "axis": axis, "centroids": cents, "order": g["order"],
            "rise_A": g["rise_A"], "twist_deg": g["twist_deg"]}


def _axial_points(geom) -> np.ndarray:
    c0, ax = geom["centroids"][0], geom["axis"]
    z_last = float((geom["centroids"][-1] - c0) @ ax)
    z = np.arange(-AXIS_BEYOND_A, z_last + AXIS_BEYOND_A + 1e-9, AXIS_STEP_A)
    return z, c0[None, :] + z[:, None] * ax[None, :]


def _perp_basis(ax):
    u = np.cross(ax, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(ax, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(ax, u)


SURFACE_OFFSETS_A = (1.0, 3.0)
PROBE_A = 1.4
MARCH_MAX_A = 30.0
MARCH_STEP_A = 0.25
# Born radii reproducing ion hydration free energies with the Born formula (Rashin & Honig,
# J Phys Chem 1985), the standard PB description of a bound ion.
BOUND_ION = {"KCl_150mM": ("K", 2.172), "NaCl_150mM": ("NA", 1.680)}


def _surface_points(origin: np.ndarray, direction: np.ndarray, xyz: np.ndarray,
                    radii: np.ndarray, offsets=SURFACE_OFFSETS_A):
    """`apbs_run.surface_referenced_points` along one ray: the first radius from which the
    ray stays in solvent (probe 1.4 A), and points at fixed offsets beyond it."""
    march = np.arange(0.0, MARCH_MAX_A, MARCH_STEP_A)
    pts = origin[None, :] + march[:, None] * direction[None, :]
    d = np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=-1)
    free = np.all(d > (radii + PROBE_A)[None, :], axis=1)
    # first index from which every later sample is free
    tail = np.flip(np.cumprod(np.flip(free).astype(int))).astype(bool)
    if not tail.any():
        return None, None
    r0 = float(march[np.argmax(tail)])
    return r0, origin[None, :] + (r0 + np.asarray(offsets))[:, None] * direction[None, :]


def groove_rays(geom):
    """(origin, direction) per tetrad level and groove: from the tetrad centroid, midway in
    azimuth between neighbouring guanine columns."""
    ax = geom["axis"]; u, v = _perp_basis(ax)
    rays = []
    for t, c in zip(geom["order"], geom["centroids"]):
        az = []
        for k in t:
            p = geom["res"][k]["atoms"].get("C1'")
            if p is not None:
                d = p - c - ((p - c) @ ax) * ax
                az.append(np.arctan2(d @ v, d @ u))
        az = np.sort(np.array(az))
        for i in range(len(az)):
            m = (az[i] + az[(i + 1) % len(az)] + (2 * np.pi if i == len(az) - 1 else 0.0)) / 2
            rays.append((c, np.cos(m) * u + np.sin(m) * v))
    return rays


def residue_rays(geom, n_res: int):
    """Per residue: from the nearest stack-axis point through the base centroid."""
    ax, c0 = geom["axis"], geom["centroids"].mean(0)
    rays = [None] * n_res
    for (ch, i), r in geom["res"].items():
        if not 1 <= i <= n_res:
            continue
        b = [r["atoms"][n] for n in ("N1", "C2", "N3", "C4", "C5", "C6") if n in r["atoms"]]
        if len(b) < 3:
            continue
        bc = np.mean(b, axis=0)
        foot = c0 + ((bc - c0) @ ax) * ax
        d = bc - foot
        nd = np.linalg.norm(d)
        if nd > 1e-6:
            rays[i - 1] = (foot, d / nd)
    return rays


def _bound_ion_pqr(pqr_in: str, pqr_out: str, sites: np.ndarray, ion: str, radius: float,
                   keep_solute: bool = True) -> str:
    lines = []
    with open(pqr_in) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and keep_solute:
                lines.append(line.rstrip("\n"))
    for k, x in enumerate(sites, start=1):
        # whitespace-delimited PQR, 11 fields like pdb2pqr's output
        lines.append(f"ATOM {90000 + k} {ion} {ion} I {k} {x[0]:.3f} {x[1]:.3f} {x[2]:.3f} "
                     f"1.0000 {radius:.4f}")
    with open(pqr_out, "w") as f:
        f.write("\n".join(lines) + "\nEND\n")
    return pqr_out


COULOMB_KJ_A = 1389.354576     # e^2 / (4 pi eps0) in kJ/mol * A


def _solve_pair(pqr: str, name: str, work: str, cond, centre, coarse, fine, maps: bool):
    """Solvated and reference solves on one grid; (potential map or None, per-atom solvation)."""
    inp = A.write_apbs_input(os.path.basename(pqr), name, work, cond, centre, coarse, fine,
                             exact_fine_spacing=True, write_maps=maps, per_atom_energies=True)
    so = A.run_apbs(inp, work)
    pot = A.DXMap.read(os.path.join(work, name + "_pot.dx")) if maps else None
    ref = A.write_apbs_input(os.path.basename(pqr), name + "_ref", work, cond, centre, coarse,
                             fine, exact_fine_spacing=True, write_maps=False, reference=True,
                             per_atom_energies=True)
    e_s, e_r = A.parse_atom_energies(so), A.parse_atom_energies(A.run_apbs(ref, work))
    solv = (e_s - e_r) if (e_s is not None and e_r is not None and len(e_s) == len(e_r)) else None
    return pot, solv


def solve(pdb_min: str, work: str, conditions: Sequence[A.IonicCondition], n_res: int,
          loop_residues: Sequence[int], tetrads=None, ion_binding: bool = True) -> Dict[str, object]:
    """APBS on a model at each condition; returns sampled quantities (and the tetrads used)."""
    geom = model_geometry(pdb_min, tetrads)
    pqr = os.path.join(work, "g4.pqr")
    A.run_pdb2pqr(pdb_min, pqr); A.sanitise_pqr(pqr)
    xyz, q, rad, recs = A.read_pqr(pqr)
    centre = xyz.mean(0)
    ext = xyz.max(0) - xyz.min(0)
    z, axial = _axial_points(geom)
    out: Dict[str, object] = {"z_A": z, "rise_A": geom["rise_A"], "twist_deg": geom["twist_deg"],
                              "n_tetrads": len(geom["order"]), "tetrads": geom["order"]}
    rel = xyz - centre
    out["dipole_D"] = float(np.linalg.norm((q[:, None] * rel).sum(0)) * DEBYE_PER_E_A)
    heavy = rad > 1.2
    out["rg_A"] = float(np.sqrt(np.mean(np.sum((xyz[heavy] - xyz[heavy].mean(0)) ** 2, 1))))
    out["net_charge_e"] = float(q.sum())
    zt = np.array([(c - geom["centroids"][0]) @ geom["axis"] for c in geom["centroids"]])
    rise = float(np.mean(np.diff(zt)))
    sites_z = (zt[:-1] + zt[1:]) / 2
    outer5, outer3 = zt[0] - rise / 2, zt[-1] + rise / 2
    site_xyz = geom["centroids"][0][None, :] + sites_z[:, None] * geom["axis"][None, :]
    # surface-referenced points (condition independent)
    g_pts, g_r = [], []
    for o, d in groove_rays(geom):
        r0, pts = _surface_points(o, d, xyz, rad)
        if r0 is not None:
            g_pts.append(pts); g_r.append(r0)
    g_pts = np.array(g_pts)                                   # (n_rays, n_offsets, 3)
    out["groove_depth_A"] = float(np.mean(g_r)) if g_r else float("nan")
    r_pts = np.full((n_res, len(SURFACE_OFFSETS_A), 3), np.nan)
    for i, ray in enumerate(residue_rays(geom, n_res)):
        if ray is not None:
            r0, pts = _surface_points(ray[0], ray[1], xyz, rad)
            if r0 is not None:
                r_pts[i] = pts
    lr = [i - 1 for i in loop_residues if 1 <= i <= n_res]
    for cond in conditions:
        tag = cond.name
        lam = A.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
        coarse = np.full(3, float(ext.max() + 2 * max(20.0, 3 * lam)))
        fine = np.full(3, float(min(coarse[0], ext.max() + 16.0)))
        pot, solv = _solve_pair(pqr, f"g4_{tag}", work, cond, centre, coarse, fine, maps=True)
        prof = pot.interpolate(axial)
        out[f"profile_{tag}"] = prof
        at = lambda zz: float(np.interp(zz, z, prof))                       # noqa: E731
        out[f"phi_sites_{tag}"] = [at(v) for v in sites_z]
        out[f"phi_planes_{tag}"] = [at(v) for v in zt]
        out[f"phi_outer5_{tag}"] = at(outer5)
        out[f"phi_outer3_{tag}"] = at(outer3)
        inside = (z >= outer5) & (z <= outer3)
        out[f"phi_channel_min_{tag}"] = float(np.nanmin(prof[inside]))
        out[f"phi_far5_{tag}"] = at(zt[0] - 6.0)
        out[f"phi_far3_{tag}"] = at(zt[-1] + 6.0)
        for k, off in enumerate(SURFACE_OFFSETS_A):
            gv = pot.interpolate(g_pts[:, k]) if len(g_pts) else np.array([np.nan])
            out[f"phi_groove_d{int(off)}_mean_{tag}"] = float(np.nanmean(gv))
            out[f"phi_groove_d{int(off)}_min_{tag}"] = float(np.nanmin(gv))
            rv = np.full(n_res, np.nan)
            ok = np.isfinite(r_pts[:, k]).all(1)
            rv[ok] = pot.interpolate(r_pts[ok, k])
            out[f"phi_residue_d{int(off)}_{tag}"] = rv
            out[f"phi_loops_d{int(off)}_{tag}"] = float(np.nanmean(rv[lr])) if lr else float("nan")
        out[f"solvation_kJ_mol_{tag}"] = float(np.sum(solv)) if solv is not None else float("nan")
        # electrostatic binding free energy of the two inner-site cations (K+ in KCl, Na+ in
        # NaCl): solvation of complex - quadruplex - ions alone, plus their Coulomb
        # interaction in the solute dielectric (the thermodynamic cycle on one grid)
        ion = BOUND_ION.get(tag) if ion_binding else None
        if ion is not None and solv is not None:
            cpqr = _bound_ion_pqr(pqr, os.path.join(work, f"cplx_{tag}.pqr"), site_xyz, ion[0], ion[1])
            ipqr = _bound_ion_pqr(pqr, os.path.join(work, f"ions_{tag}.pqr"), site_xyz, ion[0], ion[1],
                                  keep_solute=False)
            _, s_c = _solve_pair(cpqr, f"cplx_{tag}", work, cond, centre, coarse, fine, maps=False)
            _, s_i = _solve_pair(ipqr, f"ions_{tag}", work, cond, centre, coarse, fine, maps=False)
            if s_c is not None and s_i is not None:
                d = np.linalg.norm(site_xyz[:, None, :] - xyz[None, :, :], axis=-1)
                e_coul = float(COULOMB_KJ_A / 2.0 * np.sum(1.0 * q[None, :] / d))   # eps_p = 2
                out[f"ion_binding_kJ_mol_{tag}"] = float(np.sum(s_c) - np.sum(solv) - np.sum(s_i) + e_coul)
                out[f"ion_coulomb_kJ_mol_{tag}"] = e_coul
                out[f"ion_desolvation_kJ_mol_{tag}"] = float(np.sum(s_c) - np.sum(solv) - np.sum(s_i))
        for f in os.listdir(work):
            if f.endswith(".dx"):
                os.remove(os.path.join(work, f))
    return out
