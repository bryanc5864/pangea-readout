#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A channels beyond the shell potentials, read from each solve's potential map.

PLAN.md, Family A: "Potential on cylindrical shells at 6 angular sectors and 6 radial
offsets across 9 ionic conditions; field vector components and gradients per groove;
condensed counterion fraction per phosphate per strand; ion-specific occupancy for sodium,
potassium, magnesium and chloride at groove and backbone sites; electrostatic solvation
free energy; electrostatic component of base-pair opening; Manning charge density
parameter and surface charge density; electrostatic persistence length over 4 window
sizes; strand and groove potential asymmetry; complementarity to a model cationic probe;
local dielectric response."

The solver writes the potential map once per condition and deletes it, so every channel
that needs the map has to be extracted in the same pass. This module covers the ones a
map determines. Mean-field Poisson-Boltzmann gives each mobile species the Boltzmann
density c_i(r) = c_i,bulk exp(-z_i phi(r)) wherever an ion fits, with phi in kT/e, which
is what makes ion-specific occupancy and condensed charge readable from phi alone; the
ion-accessible region uses the condition's largest ion radius, as APBS does.

Conventions (all in the base-pair frame, 3DNA/Tsukuba: +x into the major groove, +y
toward strand I, +z along the helix): groove "major" is sector 0, "minor" sector 3; the
field is E = -grad(phi), in kT/(e A), with components radial (along the sector ray),
tangential (z cross radial) and axial (z).

Not here: the electrostatic component of base-pair opening, the windowed persistence
length and the ensemble spread need solves this module does not see (DECISIONS.md).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

AVOGADRO_PER_A3 = 6.02214076e23 / 1e27      # number per A^3 per mol/L
CONDENSATION_RADIUS_A = 7.0
FIELD_OFFSET_A = 1.0
FIELD_STEP_A = 0.5            # first derivatives: +-0.5 A, about 1.5 grid cells
CURVATURE_STEP_A = 1.0        # second derivative: +-1 A; at 0.25 A the second difference
                              # of a trilinear map was interpolation noise (RC rmse/sd 0.52)
GROOVES = {"major": 0.0, "minor": np.pi}
GROOVE_SITE_OFFSETS_A = (0.0, 1.0, 2.0, 3.0, 4.0)
GROOVE_SITE_ANGLES = np.deg2rad((-15.0, 0.0, 15.0))
PROBE_RADIUS_A = 2.0
PROBE_ANGLES = np.deg2rad((-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0))
PROBE_OFFSETS_A = (1.5, 2.0, 2.5, 3.0, 4.0)
PROBE_AXIAL_A = (-1.6, 0.0, 1.6)
BJERRUM_WATER_25C_A = 7.1                   # e^2 / (4 pi eps0 * 78.54 * kT), 298.15 K


def _frames(bp_frames: np.ndarray):
    T = bp_frames[:, :3, :3]
    return T[:, :, 0], T[:, :, 1], T[:, :, 2], bp_frames[:, :3, 3]


def _accessible(points: np.ndarray, atom_xyz: np.ndarray, atom_radii: np.ndarray,
                probe_A: float, tree=None) -> np.ndarray:
    """True where a point is farther than (atom radius + probe) from every atom."""
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(atom_xyz)
    reach = float(atom_radii.max()) + probe_A
    ok = np.ones(len(points), bool)
    for i, nbrs in enumerate(tree.query_ball_point(points, reach)):
        if nbrs:
            d = np.linalg.norm(atom_xyz[nbrs] - points[i], axis=1)
            ok[i] = bool(np.all(d > atom_radii[nbrs] + probe_A))
    return ok


def groove_field(pot, bp_frames: np.ndarray, surface_r: np.ndarray,
                 offset_A: float = FIELD_OFFSET_A, h: float = FIELD_STEP_A,
                 h2: float = CURVATURE_STEP_A) -> Dict[str, np.ndarray]:
    """Field components and radial field gradient 1 A outside the surface in each groove.

    surface_r: (bp, 6 sectors) surface radius along each sector ray (NaN if undefined).
    Returns E (bp, groove, 3) as (radial, tangential, axial) and dEr_dr (bp, groove).
    """
    ex, ey, ez, o = _frames(bp_frames)
    n = len(o)
    E = np.full((n, len(GROOVES), 3), np.nan)
    dE = np.full((n, len(GROOVES)), np.nan)
    sector = {"major": 0, "minor": 3}
    for g, (name, ang) in enumerate(GROOVES.items()):
        r0 = surface_r[:, sector[name]]
        er = np.cos(ang) * ex + np.sin(ang) * ey
        et = np.cross(ez, er)
        p = o + (r0 + offset_A)[:, None] * er
        for k, e in enumerate((er, et, ez)):
            plus, minus = pot.interpolate(p + h * e), pot.interpolate(p - h * e)
            E[:, g, k] = -(plus - minus) / (2 * h)
        plus2, minus2, mid = pot.interpolate(p + h2 * er), pot.interpolate(p - h2 * er), pot.interpolate(p)
        dE[:, g] = -(plus2 - 2 * mid + minus2) / h2 ** 2
    return {"field": E, "dEr_dr": dE}


def groove_ion_enrichment(pot, bp_frames: np.ndarray, surface_r: np.ndarray,
                          atom_xyz: np.ndarray, atom_radii: np.ndarray, ion_radius_A: float,
                          valences: Sequence[float]) -> np.ndarray:
    """log10 mean Boltzmann enrichment exp(-z phi) over each groove site, per valence.

    Site: the groove ray and +-15 deg either side, 0 to 4 A outside the surface, ion
    accessible points only. Returns (bp, groove, valence).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(atom_xyz)
    ex, ey, ez, o = _frames(bp_frames)
    n = len(o)
    out = np.full((n, len(GROOVES), len(valences)), np.nan)
    sector = {"major": 0, "minor": 3}
    for g, (name, ang) in enumerate(GROOVES.items()):
        r0 = surface_r[:, sector[name]]
        pts = []
        for da in GROOVE_SITE_ANGLES:
            d = np.cos(ang + da) * ex + np.sin(ang + da) * ey
            for off in GROOVE_SITE_OFFSETS_A:
                pts.append(o + (r0 + off)[:, None] * d)
        P = np.stack(pts, axis=1)                              # (bp, m, 3)
        flat = P.reshape(-1, 3)
        fin = np.all(np.isfinite(flat), axis=1)
        phi = np.full(len(flat), np.nan)
        phi[fin] = pot.interpolate(flat[fin])
        acc = np.zeros(len(flat), bool)
        acc[fin] = _accessible(flat[fin], atom_xyz, atom_radii, ion_radius_A, tree)
        phi = np.where(acc, phi, np.nan).reshape(P.shape[:2])
        for v, z in enumerate(valences):
            with np.errstate(invalid="ignore"):
                f = np.nanmean(np.exp(-z * phi), axis=1)
            out[:, g, v] = np.log10(f)
    return out


def phosphate_cells(pot, atom_xyz: np.ndarray, atom_radii: np.ndarray, recs: List[dict],
                    ions: Sequence, radius_A: float = CONDENSATION_RADIUS_A) -> Dict[str, object]:
    """Condensed counterion charge and backbone ion enrichment, per phosphate.

    Every map point within radius_A of a phosphorus belongs to its nearest phosphorus
    (a Voronoi partition, so neighbouring phosphates do not count the same ions twice).
    Over the ion-accessible points of each cell: the net excess mobile charge,
    sum_i z_i (n_i - n_i,bulk) dV in units of e, which is the fraction of that
    phosphate's -1 e neutralised; the cell's mean potential; and log10 of the mean
    Boltzmann enrichment for each species. Returns per-phosphate arrays and the
    (chain, resseq) of each phosphorus.
    """
    from scipy.spatial import cKDTree
    P_idx = [i for i, r in enumerate(recs) if r["name"] == "P"]
    if not P_idx:
        return {}
    pxyz = atom_xyz[P_idx]
    nx, ny, nz = pot.data.shape
    gx = pot.origin[0] + pot.delta[0] * np.arange(nx)
    gy = pot.origin[1] + pot.delta[1] * np.arange(ny)
    gz = pot.origin[2] + pot.delta[2] * np.arange(nz)
    lo, hi = pxyz.min(0) - radius_A, pxyz.max(0) + radius_A
    sx = np.flatnonzero((gx >= lo[0]) & (gx <= hi[0]))
    sy = np.flatnonzero((gy >= lo[1]) & (gy <= hi[1]))
    sz = np.flatnonzero((gz >= lo[2]) & (gz <= hi[2]))
    if not (len(sx) and len(sy) and len(sz)):
        return {}
    X, Y, Z = np.meshgrid(gx[sx], gy[sy], gz[sz], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    phi = pot.data[np.ix_(sx, sy, sz)].ravel()
    d, owner = cKDTree(pxyz).query(pts, distance_upper_bound=radius_A)
    near = np.isfinite(d)
    pts, phi, owner = pts[near], phi[near], owner[near]
    r_ion = max(float(i.radius_A) for i in ions)            # APBS default (largest radius)
    acc = _accessible_bulk(pts, atom_xyz, atom_radii, r_ion)
    pts, phi, owner = pts[acc], phi[acc], owner[acc]
    dV = float(np.prod(pot.delta))
    nP = len(P_idx)
    excess = np.zeros(nP)
    for ion in ions:
        nb = float(ion.concentration_M) * AVOGADRO_PER_A3
        contrib = float(ion.charge) * nb * (np.exp(-float(ion.charge) * phi) - 1.0) * dV
        excess += np.bincount(owner, weights=contrib, minlength=nP)
    count = np.bincount(owner, minlength=nP)
    mean_phi = np.bincount(owner, weights=phi, minlength=nP) / np.maximum(count, 1)
    enrich = {}
    for z in sorted({float(i.charge) for i in ions}):
        w = np.bincount(owner, weights=np.exp(-z * phi), minlength=nP) / np.maximum(count, 1)
        enrich[z] = np.log10(np.where(count > 0, w, np.nan))
    return {"chain": [recs[i].get("chain", "") for i in P_idx],
            "resseq": [int(recs[i].get("resseq", -1)) for i in P_idx],
            "condensed_fraction": np.where(count > 0, excess, np.nan),
            "mean_potential": np.where(count > 0, mean_phi, np.nan),
            "log10_enrichment": enrich, "n_points": count}


def _accessible_bulk(points: np.ndarray, atom_xyz: np.ndarray, atom_radii: np.ndarray,
                     probe_A: float) -> np.ndarray:
    """Vectorised accessibility for many points: nearest-neighbour test by radius class."""
    from scipy.spatial import cKDTree
    ok = np.ones(len(points), bool)
    for rad in np.unique(np.round(atom_radii, 4)):
        sel = np.isclose(atom_radii, rad, atol=1e-4)
        d, _ = cKDTree(atom_xyz[sel]).query(points, distance_upper_bound=rad + probe_A)
        ok &= ~(d <= rad + probe_A)
    return ok


def probe_complementarity(pot, bp_frames: np.ndarray, surface_r: np.ndarray,
                          atom_xyz: np.ndarray, atom_radii: np.ndarray) -> np.ndarray:
    """Most favourable energy (kT) of a +1 e probe of radius 2 A in each groove, per bp.

    Search: the groove direction +-30 deg in 10 deg steps, 1.5 to 4 A outside the
    surface, axially -1.6 / 0 / +1.6 A (half a rise either way); accessible to the probe.
    Returns (bp, groove) = min over the site of (+1) * phi.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(atom_xyz)
    ex, ey, ez, o = _frames(bp_frames)
    n = len(o)
    out = np.full((n, len(GROOVES)), np.nan)
    sector = {"major": 0, "minor": 3}
    for g, (name, ang) in enumerate(GROOVES.items()):
        r0 = surface_r[:, sector[name]]
        pts = []
        for da in PROBE_ANGLES:
            d = np.cos(ang + da) * ex + np.sin(ang + da) * ey
            for off in PROBE_OFFSETS_A:
                for az in PROBE_AXIAL_A:
                    pts.append(o + (r0 + off)[:, None] * d + az * ez)
        P = np.stack(pts, axis=1)
        flat = P.reshape(-1, 3)
        fin = np.all(np.isfinite(flat), axis=1)
        phi = np.full(len(flat), np.nan)
        phi[fin] = pot.interpolate(flat[fin])
        acc = np.zeros(len(flat), bool)
        acc[fin] = _accessible(flat[fin], atom_xyz, atom_radii, PROBE_RADIUS_A, tree)
        phi = np.where(acc, phi, np.nan).reshape(P.shape[:2])
        with np.errstate(all="ignore"):
            out[:, g] = np.nanmin(phi, axis=1)
    return out


def manning_and_surface_charge(bp_frames: np.ndarray, surface_r: np.ndarray,
                               bjerrum_A: float = BJERRUM_WATER_25C_A) -> Dict[str, np.ndarray]:
    """Local Manning parameter xi = l_B / b and surface charge density, per bp.

    b is the axial distance per elementary charge: two phosphates per base-pair step, so
    b = rise / 2 with the rise projected on the mean helix axis and averaged over the two
    steps that touch the base pair. Surface charge density: 2 e over the lateral area of
    a cylinder of the mean surface radius and that rise, in e / A^2.
    """
    o = bp_frames[:, :3, 3]
    axis = bp_frames[:, :3, 2].mean(0); axis /= np.linalg.norm(axis)
    step = np.diff(o, axis=0) @ axis
    rise = np.full(len(o), np.nan)
    rise[1:-1] = 0.5 * (step[:-1] + step[1:])
    rise[0], rise[-1] = step[0], step[-1]
    rise = np.abs(rise)
    xi = bjerrum_A / (rise / 2.0)
    R = np.nanmean(surface_r, axis=1)
    sigma = 2.0 / (2 * np.pi * R * rise)
    return {"manning_xi": xi, "surface_charge_e_per_A2": sigma, "rise_A": rise}


VALENCES = (1.0, -1.0, 2.0)          # monovalent cation, anion, divalent cation


class CellCache:
    """Phosphate Voronoi cells (near, accessible map points and their owners), reused by
    every condition whose map has the same grid and largest ion radius -- most of them,
    since the fine grid is set by the fine box from 50 mM up."""

    def __init__(self):
        self._store = {}

    def get(self, pot, atom_xyz, atom_radii, pxyz, r_ion, radius_A=CONDENSATION_RADIUS_A):
        from scipy.spatial import cKDTree
        key = (tuple(np.round(pot.origin, 4)), tuple(np.round(pot.delta, 5)),
               tuple(pot.data.shape), round(float(r_ion), 3))
        if key in self._store:
            return self._store[key]
        nx, ny, nz = pot.data.shape
        g = [pot.origin[k] + pot.delta[k] * np.arange(pot.data.shape[k]) for k in range(3)]
        lo, hi = pxyz.min(0) - radius_A, pxyz.max(0) + radius_A
        s = [np.flatnonzero((g[k] >= lo[k]) & (g[k] <= hi[k])) for k in range(3)]
        if not all(len(x) for x in s):
            self._store[key] = None
            return None
        I, J, K = np.meshgrid(*s, indexing="ij")
        idx = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)
        pts = pot.origin + idx * pot.delta
        d, owner = cKDTree(pxyz).query(pts, distance_upper_bound=radius_A)
        near = np.isfinite(d)
        idx, pts, owner = idx[near], pts[near], owner[near]
        acc = _accessible_bulk(pts, atom_xyz, atom_radii, r_ion)
        entry = (idx[acc], owner[acc])
        self._store[key] = entry
        return entry


def strand_phosphates(recs: List[dict], n_bp: int, scored: slice) -> np.ndarray:
    """Atom indices of the phosphorus 5' of each scored base pair's Watson (column 0) and
    Crick (column 1) nucleotide; -1 where there is none (a 5' terminus).

    Watson residue b+1 (chain A) pairs with Crick residue n-b (chain B), b 0-based.
    """
    where = {(r.get("chain", ""), str(r.get("resseq", ""))): i
             for i, r in enumerate(recs) if r["name"] == "P"}
    rows = []
    for b in range(scored.start, scored.stop):
        rows.append([where.get(("A", str(b + 1)), -1), where.get(("B", str(n_bp - b)), -1)])
    return np.asarray(rows, dtype=int)


def extract_condition(pot, bp_frames_scored: np.ndarray, surface_r: np.ndarray,
                      atom_xyz: np.ndarray, atom_radii: np.ndarray, recs: List[dict],
                      condition, n_bp: int, scored: slice, cache: CellCache) -> Dict[str, np.ndarray]:
    """Every map-derived Family A quantity for one condition, per scored base pair."""
    r_ion = condition.ion_accessibility_radius_A            # the radius APBS used
    out = groove_field(pot, bp_frames_scored, surface_r)
    out["groove_log10_enrichment"] = groove_ion_enrichment(
        pot, bp_frames_scored, surface_r, atom_xyz, atom_radii, r_ion, VALENCES)
    present = {float(i.charge) for i in condition.ions}
    for v, z in enumerate(VALENCES):
        if z not in present:
            out["groove_log10_enrichment"][:, :, v] = np.nan
    out["probe_kT"] = probe_complementarity(pot, bp_frames_scored, surface_r,
                                            atom_xyz, atom_radii)
    # phosphate cells
    P_all = np.asarray([i for i, r in enumerate(recs) if r["name"] == "P"], dtype=int)
    nS = scored.stop - scored.start
    cf = np.full((nS, 2), np.nan); mp = np.full((nS, 2), np.nan)
    be = np.full((nS, 2, len(VALENCES)), np.nan)
    if len(P_all):
        entry = cache.get(pot, atom_xyz, atom_radii, atom_xyz[P_all], r_ion)
        if entry is not None:
            idx, owner = entry
            phi = pot.data[idx[:, 0], idx[:, 1], idx[:, 2]]
            nP = len(P_all)
            dV = float(np.prod(pot.delta))
            excess = np.zeros(nP)
            for ion in condition.ions:
                nb = float(ion.concentration_M) * AVOGADRO_PER_A3
                excess += np.bincount(owner, weights=float(ion.charge) * nb
                                      * (np.exp(-float(ion.charge) * phi) - 1.0) * dV,
                                      minlength=nP)
            cnt = np.bincount(owner, minlength=nP)
            meanphi = np.bincount(owner, weights=phi, minlength=nP) / np.maximum(cnt, 1)
            enr = np.stack([np.log10(np.bincount(owner, weights=np.exp(-z * phi), minlength=nP)
                                     / np.maximum(cnt, 1)) for z in VALENCES], axis=1)
            pos = {int(a): k for k, a in enumerate(P_all)}
            sp = strand_phosphates(recs, n_bp, scored)
            for i in range(nS):
                for s in range(2):
                    a = int(sp[i, s])
                    if a >= 0 and cnt[pos[a]] > 0:
                        k = pos[a]
                        cf[i, s], mp[i, s] = excess[k], meanphi[k]
                        be[i, s] = enr[k]
            for v, z in enumerate(VALENCES):
                if z not in present:
                    be[:, :, v] = np.nan
    out["condensed_fraction"] = cf
    out["phosphate_potential"] = mp
    out["backbone_log10_enrichment"] = be
    return out
