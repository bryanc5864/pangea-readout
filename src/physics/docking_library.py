#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family G ligand affinities (PLAN.md G: "generic intercalator and minor groove binder affinity"),
which the stage-1 Family G declared unreachable without docking. Receptors, per RC-unique tetramer
context (stage 1, physics environment):

* minor-groove site: the 12-mer GCGC + context + GCGC from its cgNA+ ground state, minimised as the
  production structures; the site is the minor groove of the central step;
* intercalation site: the same 12-mer with the central step opened as an intercalator opens it --
  the 3' half moved rigidly by +3.4 A along the step's helical axis and unwound by 10 degrees about
  it (rise ~6.8 A, the classic intercalation geometry) -- then minimised with the base heavy atoms
  restrained (AMBER OL15 + GBn2) so the sugar-phosphate backbone stretches across the gap.

Docking (stage 2, `dock_run.py`, docking environment): AutoDock Vina 1.2.7 (Eberhardt et al., J
Chem Inf Model 61:3891, 2021) with DAPI into the minor-groove site and proflavine into the
intercalation site, the best pose's Vina score (kcal/mol) as the affinity channel.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PAD = "GCGC"
OPEN_RISE_A, UNWIND_DEG = 3.4, 10.0


def _rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def build_receptors(context: str, work: str) -> Dict[str, object]:
    import cgna_atomic as ca
    import frames as F
    import minimize as MZ
    seq = PAD + context + PAD[::-1]
    n = len(seq); k = len(PAD) + 1                            # central step: pairs k, k+1 (0-based)
    out = {}
    # minor-groove receptor: ground state, minimised as production
    gs = ca.CGNAGroundState(seq)
    raw = os.path.join(work, "mg_raw.pdb"); ca.write_pdb(ca.build_duplex(seq, ca.NucleotideTemplates(), gs), raw)
    mg = os.path.join(work, "mg.pdb")
    MZ.restrained_minimise(raw, mg, max_iterations=20000, tolerance=1.0)
    bp, _ = F.bp_frames_from_pdb(mg)
    # site: 6 A along -x (minor groove) from the central step's mid origin
    o = 0.5 * (bp[k][:3, 3] + bp[k + 1][:3, 3]); x = 0.5 * (bp[k][:3, 0] + bp[k + 1][:3, 0])
    out["minor"] = {"pdb": mg, "centre": (o - 6.0 * x / np.linalg.norm(x)).tolist()}
    # intercalation receptor: open the central step
    gs2 = ca.CGNAGroundState(seq)
    wb, cb, wp, cp = gs2.watson_bases, gs2.crick_bases, gs2.watson_phosphates, gs2.crick_phosphates
    mid = 0.5 * (wb[k] + wb[k + 1])
    z = wb[k][:3, 2] + wb[k + 1][:3, 2]; z /= np.linalg.norm(z)
    c0 = 0.5 * (0.5 * (wb[k][:3, 3] + cb[k][:3, 3]) + 0.5 * (wb[k + 1][:3, 3] + cb[k + 1][:3, 3]))
    R = _rot(z, np.radians(-UNWIND_DEG))
    def move(T):
        T = T.copy(); T[:3, :3] = R @ T[:3, :3]; T[:3, 3] = R @ (T[:3, 3] - c0) + c0 + OPEN_RISE_A * z
        return T
    for arr in (wb, cb):
        for i in range(k + 1, n):
            arr[i] = move(arr[i])
    for arr in (wp, cp):
        for i in range(k + 1, len(arr)):
            if np.all(np.isfinite(arr[i])):
                arr[i] = move(arr[i])
    raw2 = os.path.join(work, "ic_raw.pdb"); ca.write_pdb(ca.build_duplex(seq, ca.NucleotideTemplates(), gs2), raw2)
    ic = os.path.join(work, "ic.pdb")
    _minimise_bases_restrained(raw2, ic)
    bp2, _ = F.bp_frames_from_pdb(ic)
    out["intercalation"] = {"pdb": ic, "centre": (0.5 * (bp2[k][:3, 3] + bp2[k + 1][:3, 3])).tolist()}
    return out


def _minimise_bases_restrained(pdb_in: str, pdb_out: str) -> None:
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    fx = PDBFixer(filename=pdb_in); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(fx.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", 500.0)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    bb = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
    for a in fx.topology.atoms():
        if a.element.symbol != "H" and a.name not in bb:
            rest.addParticle(a.index, pos[a.index])
    system.addForce(rest)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"),
                         {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")})
    ctx.setPositions(fx.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 20000)
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(fx.topology, ctx.getState(getPositions=True).getPositions(), fh, keepIds=True)


def main() -> int:
    import argparse, tempfile, shutil
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "qc"))
    import build_qc_fragments as QF
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True); ap.add_argument("--contexts", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ctx in (a.contexts or QF.unique_contexts(4))[a.shard::a.n_shards]:
        meta = os.path.join(a.out, f"{ctx}.json")
        if os.path.exists(meta):
            continue
        work = tempfile.mkdtemp()
        try:
            r = build_receptors(ctx, work)
            for site, v in r.items():
                dst = os.path.join(a.out, f"{ctx}_{site}.pdb"); shutil.copy(v["pdb"], dst); v["pdb"] = dst
            json.dump(r, open(meta, "w"), indent=1)
            print(ctx, "receptors ok", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(ctx, "failed:", type(exc).__name__, str(exc)[:200], flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
