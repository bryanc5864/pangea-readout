#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A, two channels the production pass does not compute (PLAN.md, Family A): the
electrostatic component of base-pair opening, and the local dielectric response.

Per fragment (its production window and construction: cgNA+ ground state, restrained
minimisation converged as in production), at the central base pair c of the scored 21:

**Opening.** The open state rotates each base of pair c about the axis through its own C1'
parallel to the pair's helical z axis, by half of OPENING_DEG each, toward the major groove (the
3DNA "opening" coordinate, the usual exit pathway), and relaxes the two opened sugars with
everything else fixed (AMBER OL15 + GBn2; the opened bases restrained at the production restraint
to their rotated positions; phosphates, C3', C5' and the rest of the molecule held by a stiff
restraint), so the closed and open molecules differ only in the two bases and their sugar rings. Both are solved with the production construction on the same grids at the
10 and 150 mM NaCl at their production solute lengths (the test fragment gave 15.14 kJ/mol at 10, 50 and 150 mM and 15.04 at 500: local, salt-independent):
dG_open,elec = dE_Coulomb(solute dielectric 2, AMBER exclusions and 1-4 scaling) + d(sum of the
per-atom map solvation 1/2 q (phi_solvated - phi_reference) over the scored base pairs).

**Local dielectric response.** At the central pair's surface-referenced points (6 sectors, 1
and 4 A outside the molecular surface, as the potential channels), the reaction-field (image)
potential of a unit probe charge: one Poisson solve with the molecule's dielectric boundary
(solute 2, water 78.54, no ions -- the dielectric part alone) and its charges set to zero,
minus the same solve in uniform water on the identical probe-centred grid. Positive: the
low-dielectric molecule repels a charge there; zero: bulk water. Units kT/e.

One .npz + .json per fragment; resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apbs_run as A        # noqa: E402
import frames as F          # noqa: E402
import run_fragment as rf   # noqa: E402

OPENING_DEG = 60.0
OPEN_CONDITIONS = ("NaCl_010mM", "NaCl_150mM")    # salt-independent to 0.1 kJ/mol over 10-500 mM (test 34293674)
PROBE_OFFSETS_A = (1.0, 4.0)
PROBE_COARSE_A, PROBE_FINE_A, PROBE_DIME = 96.0, 33.6, 97     # 1.0 A / 0.35 A
KJ_PER_KT = 8.314462618e-3 * 298.15
BASE_NAMES = {"N1", "C2", "N3", "C4", "C5", "C6", "N6", "O6", "N2", "N7", "C8", "N9", "O2", "N4",
              "O4", "C7", "H1", "H2", "H21", "H22", "H3", "H41", "H42", "H5", "H6", "H61", "H62",
              "H8", "H71", "H72", "H73"}


def _rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def open_pair(pdb_in: str, pdb_out: str, b: int, n_bp: int, frame: np.ndarray,
              opening_deg: float = OPENING_DEG) -> dict:
    """Rotate both bases of pair b (0-based) toward the major groove and relax them locally."""
    import openmm
    from openmm import app, unit
    pdb = app.PDBFile(pdb_in)
    pos = np.array(pdb.positions.value_in_unit(unit.angstrom))
    z, x, o = frame[:3, 2], frame[:3, 0], frame[:3, 3]
    targets = {("A", b + 1), ("B", n_bp - b)}
    moved, info = set(), {}
    for res in pdb.topology.residues():
        key = (res.chain.id, int(res.id))
        if key not in targets:
            continue
        at = {a.name: a.index for a in res.atoms()}
        c1 = pos[at["C1'"]]
        bidx = [i for n, i in at.items() if n in BASE_NAMES]
        cen0 = pos[bidx].mean(0)
        best = None
        for sgn in (1.0, -1.0):            # the sense that moves the base toward +x (major groove)
            R = _rot(z, np.radians(sgn * opening_deg / 2))
            new = (pos[bidx] - c1) @ R.T + c1
            gain = float((new.mean(0) - cen0) @ x)
            if best is None or gain > best[0]:
                best = (gain, new)
        pos[bidx] = best[1]; moved.update(bidx)
        info[f"{key[0]}{key[1]}_major_groove_shift_A"] = round(best[0], 3)
    if len(info) != 2:
        raise ValueError(f"pair {b}: residues not found ({info})")
    # GBn2 implicit solvent (a vacuum relaxation let the freed phosphates move apart and lowered
    # the Coulomb energy by ~90 kJ/mol -- an artefact unrelated to opening, test 34290512)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addPerParticleParameter("k")
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    opened_res = {a.index for r in pdb.topology.residues() if (r.chain.id, int(r.id)) in targets for a in r.atoms()}
    held = {"P", "OP1", "OP2", "O5'", "C5'", "O3'", "C3'"}     # phosphate and its links stay put
    for a in pdb.topology.atoms():
        xyz = pos[a.index] / 10.0
        if a.index in moved and a.element.symbol != "H":
            rest.addParticle(a.index, [500.0, *xyz])
        elif a.index not in opened_res or a.name in held:
            rest.addParticle(a.index, [1.0e6, *xyz])
    system.addForce(rest)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"),
                         {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")})
    ctx.setPositions(pos / 10.0)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 20000)
    st = ctx.getState(getPositions=True)
    new = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    fixed = [i for i in range(len(pos)) if i not in opened_res]
    info["fixed_atoms_max_shift_A"] = round(float(np.abs(new[fixed] - pos[fixed]).max()), 4)
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(pdb.topology, st.getPositions(), fh, keepIds=True)
    return info


def coulomb_eps2_kJ(pdb_path: str) -> float:
    """Coulomb energy in the solute dielectric (2) with AMBER exclusions and 1-4 scaling."""
    import openmm
    from openmm import app, unit
    pdb = app.PDBFile(pdb_path)
    system = app.ForceField("amber14/DNA.OL15.xml").createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            for i in range(f.getNumParticles()):
                q, s, e = f.getParticleParameters(i); f.setParticleParameters(i, q, s, 0.0)
            for k in range(f.getNumExceptions()):
                i, j, qq, s, e = f.getExceptionParameters(k); f.setExceptionParameters(k, i, j, qq, s, 0.0)
            f.setForceGroup(1)
        else:
            f.setForceGroup(2)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pdb.positions)
    e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return float(e) / 2.0


def _probe_input(pqr: str, name: str, work: str, centre, homogeneous: bool) -> str:
    sd = "78.540" if homogeneous else "2.000"
    body = f"""read
    mol pqr {os.path.basename(pqr)}
end
elec name probe
    mg-auto
    dime {PROBE_DIME} {PROBE_DIME} {PROBE_DIME}
    cglen {PROBE_COARSE_A:.3f} {PROBE_COARSE_A:.3f} {PROBE_COARSE_A:.3f}
    fglen {PROBE_FINE_A:.3f} {PROBE_FINE_A:.3f} {PROBE_FINE_A:.3f}
    cgcent {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}
    fgcent {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}
    mol 1
    lpbe
    bcfl sdh
    pdie {sd}
    sdie 78.540
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.400
    swin 0.3
    temp 298.15
    calcenergy no
    calcforce no
    write pot dx {name}_pot
end
quit
"""
    p = os.path.join(work, name + ".in")
    open(p, "w").write(body)
    return p


def dielectric_images(pqr: str, points: np.ndarray, work: str) -> np.ndarray:
    """Image potential (kT/e) of a unit probe at each point (NaN where the point is undefined)."""
    xyz, q, rad, recs = A.read_pqr(pqr)
    lines = [l for l in open(pqr) if l.startswith(("ATOM", "HETATM"))]
    out = np.full(len(points), np.nan)
    homog = None
    for k, p in enumerate(points):
        if not np.all(np.isfinite(p)):
            continue
        near = np.linalg.norm(xyz - p, axis=1) < 0.5 * PROBE_COARSE_A * 1.5
        w = os.path.join(work, f"probe{k}"); os.makedirs(w, exist_ok=True)
        pq = os.path.join(w, "probe.pqr")
        with open(pq, "w") as f:
            for l, keep, r in zip(lines, near, rad):
                if keep:
                    f.write(f"{l[:30]}{l[30:54]} {0.0:7.4f} {r:6.4f}\n")
            f.write(f"ATOM  99999 PRB  PRB Z 999    {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f} {1.0:7.4f} {0.0:6.4f}\n")
        A.run_apbs(_probe_input(pq, "diel", w, p, False), w)
        v = float(A.DXMap.read(os.path.join(w, "diel_pot.dx")).interpolate(p[None, :])[0])
        if homog is None:
            A.run_apbs(_probe_input(pq, "homog", w, p, True), w)
            homog = float(A.DXMap.read(os.path.join(w, "homog_pot.dx")).interpolate(p[None, :])[0])
        out[k] = v - homog
        shutil.rmtree(w, ignore_errors=True)
    return out


def run_one(fid: str, seq: str, out_dir: str, workroot: str, solute: dict) -> dict:
    work = tempfile.mkdtemp(prefix=f"Ax_{fid}_", dir=workroot)
    t0 = time.time()
    rec = {"frag_id": fid, "sequence": seq, "opening_deg": OPENING_DEG}
    try:
        built = rf.build_structure(seq, work, min_iterations=20000, min_tolerance=1.0)
        closed = str(built["pdb"])
        bp, _ = F.bp_frames_from_pdb(closed)
        n = len(bp); sl = rf.central_slice(n); c = (sl.start + sl.stop) // 2
        rec.update(n_bp=n, scored_bp=[sl.start, sl.stop], centre_bp=c)
        opened = os.path.join(work, "open.pdb")
        rec["open_build"] = open_pair(closed, opened, c, n, bp[c])
        conds = [A.PRODUCTION_CONDITIONS_BY_NAME[x] for x in OPEN_CONDITIONS]
        res = {}
        for tag, pdb in (("closed", closed), ("open", opened)):
            sub = os.path.join(work, tag); os.makedirs(sub)
            s = rf._solve_by_length(pdb, sub, conds, bp, sl, solute, extended=True, reference_solve=True,
                                    exact_fine_spacing=True)
            res[tag] = s
            rec[f"coulomb_eps2_kJ_{tag}"] = coulomb_eps2_kJ(pdb)
        dcoul = rec["coulomb_eps2_kJ_open"] - rec["coulomb_eps2_kJ_closed"]
        sc = np.nansum(res["closed"]["ext_bp_solvation_map_kJ_mol"], axis=1)
        so = np.nansum(res["open"]["ext_bp_solvation_map_kJ_mol"], axis=1)
        dsolv = so - sc
        dG = dcoul + dsolv
        rec["failed_conditions"] = {t: res[t].get("failed_conditions", {}) for t in res}
        # dielectric response at the central pair (closed structure, the full-length PQR)
        pqr_full = os.path.join(work, "full.pqr")
        A.run_pdb2pqr(closed, pqr_full); A.sanitise_pqr(pqr_full)
        xyz, q, rad, _ = A.read_pqr(pqr_full)
        pts, surf_r, _ = A.surface_referenced_points(bp[c:c + 1], xyz, rad, offsets_A=PROBE_OFFSETS_A)
        img = dielectric_images(pqr_full, pts.reshape(-1, 3), os.path.join(work, "probes"))
        np.savez_compressed(
            os.path.join(out_dir, f"{fid}.npz"),
            conditions=np.asarray(OPEN_CONDITIONS), open_dG_elec_kJ_mol=dG, open_dE_coulomb_kJ_mol=np.full(len(conds), dcoul),
            open_dG_solv_kJ_mol=dsolv, closed_bp_solvation_map_kJ_mol=res["closed"]["ext_bp_solvation_map_kJ_mol"],
            dielectric_image_kT=img.reshape(pts.shape[1], pts.shape[2]), probe_offsets_A=np.asarray(PROBE_OFFSETS_A),
            probe_surface_radius_A=surf_r[0], sequence=np.asarray([seq]))
        rec.update(status="ok", open_dG_elec_kJ_mol=[round(float(v), 2) for v in dG],
                   dielectric_image_kT=[round(float(v), 4) for v in img])
    except Exception as exc:                                         # noqa: BLE001
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:600],
                   traceback=traceback.format_exc()[-2500:])
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(os.path.join(out_dir, f"{fid}.json"), "w"), indent=1, default=str)
    shutil.rmtree(work, ignore_errors=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True, help="TSV with frag_id and sequence (the 121 bp window)")
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--solute-bp-json", required=True); ap.add_argument("--workroot", default=None)
    ap.add_argument("--open-conditions", nargs="+", default=None,
                    help="conditions for the opening solves (default: OPEN_CONDITIONS). The rest of the corpus "
                         "uses NaCl_150mM alone: on the first 36 fragments the 10 and 150 mM values differ by at "
                         "most 0.19 kJ/mol (median 0.05) against a 4.6-26.9 kJ/mol spread between fragments")
    a = ap.parse_args()
    if a.open_conditions:
        global OPEN_CONDITIONS
        OPEN_CONDITIONS = tuple(a.open_conditions)
    os.makedirs(a.out, exist_ok=True)
    solute = {k: int(v) for k, v in json.load(open(a.solute_bp_json)).items()}
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.library), delimiter="\t"))[int(lo): int(hi) + 1]
    n_ok = 0
    for r in rows:
        if os.path.exists(os.path.join(a.out, f"{r['frag_id']}.npz")):
            n_ok += 1; continue
        rec = run_one(r["frag_id"], r["sequence"], a.out, a.workroot, solute)
        n_ok += rec["status"] == "ok"
        print(f"{r['frag_id']} {rec['status']} {rec['seconds']}s"
              + (f" dG_open {rec.get('open_dG_elec_kJ_mol')} image {rec.get('dielectric_image_kT')}" if rec["status"] == "ok"
                 else f" :: {rec.get('error', '')[:200]}"), flush=True)
    print(f"{n_ok}/{len(rows)} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
