#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Restrained minimisation of a cgNA+-built duplex, to clean the sugar-phosphate backbone.

`cgna_atomic.build_duplex` places each nucleotide's base, sugar and phosphate from
rigid templates, then overwrites the phosphate group using cgNA+'s own phosphate frames.
That override is the point of the exercise, because the phosphate frames carry
sequence-dependent backbone information a base-slaved template cannot, but it leaves the
sugar slightly inconsistent with the phosphate it now bonds to.

This module repairs that with a short minimisation under AMBER OL15 in implicit solvent:
base heavy atoms and phosphorus/phosphate oxygens are restrained to their cgNA+
positions, and the sugar is free to relax into a chemically valid geometry.  The
restraints are what keep the structure sequence-dependent; an unrestrained minimisation
would relax every fragment toward the same canonical duplex and reintroduce exactly the
ideal-geometry collapse the plan warns about.

Usage:
    python minimize.py --in frag.pdb --out frag_min.pdb [--restraint-k 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Atoms held to their cgNA+ positions. Bases define the frames; P/OP1/OP2/O5'/O3'
# carry the charge and come from the cgNA+ phosphate frames.
BASE_ATOMS = {
    "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9",
    "O2", "O4", "O6", "N2", "N4", "N6", "C7", "C5M",
}
PHOSPHATE_RESTRAINED = {"P", "OP1", "OP2"}

# The alpha/gamma start restraint (see `restrained_minimise`): flat-bottomed, zero
# within ALPHA_GAMMA_FLAT_RAD of the canonical B-DNA values, in its own force group so
# final energies can be compared without it.
ALPHA_G_MINUS_DEG = -65.0
GAMMA_G_PLUS_DEG = 50.0
ALPHA_GAMMA_FLAT_RAD = 0.5
ALPHA_GAMMA_BIAS_K = 500.0           # kJ/mol/rad^2
BIAS_FORCE_GROUP = 31


def _alpha_gamma_bias(topology, openmm):
    """Flat-bottomed torsion restraint on every alpha and gamma, off until switched on."""
    import math

    f = openmm.CustomTorsionForce(
        "kb_alpha_gamma*max(0, abs(d) - wflat)^2; d = atan2(sin(theta-t0), cos(theta-t0))")
    f.addGlobalParameter("kb_alpha_gamma", 0.0)
    f.addGlobalParameter("wflat", ALPHA_GAMMA_FLAT_RAD)
    f.addPerTorsionParameter("t0")
    for chain in topology.chains():
        res = list(chain.residues())
        for i, r in enumerate(res):
            at = {a.name: a.index for a in r.atoms()}
            if i > 0 and all(k in at for k in ("P", "O5'", "C5'")):
                prev = {a.name: a.index for a in res[i - 1].atoms()}
                if "O3'" in prev:
                    f.addTorsion(prev["O3'"], at["P"], at["O5'"], at["C5'"],
                                 [math.radians(ALPHA_G_MINUS_DEG)])
            if all(k in at for k in ("O5'", "C5'", "C4'", "C3'")):
                f.addTorsion(at["O5'"], at["C5'"], at["C4'"], at["C3'"],
                             [math.radians(GAMMA_G_PLUS_DEG)])
    return f


# Parsed force fields, cached across calls within a process.
#
# `app.ForceField(...)` parses the AMBER XML from disk every time it is constructed, and
# this function is called once per fragment. Rebuilding it per fragment was the dominant
# cost of generating stage-1 targets and, run 243 ways in parallel against one NFS
# server, it stopped the array dead: the first 512-way target-generation job produced
# zero shards in 45 minutes because every task was re-reading and re-parsing the same
# force-field files for every fragment.
#
# The cache is keyed on the file tuple, so switching the implicit-solvent model still
# gets its own parse. It is per process, which is the right scope: Slurm array tasks do
# not share memory, and within a task every fragment wants the same force field.
_FORCE_FIELD_CACHE: Dict[Tuple[str, ...], object] = {}


def _force_field(ff_files: Tuple[str, ...], app):
    ff = _FORCE_FIELD_CACHE.get(ff_files)
    if ff is None:
        ff = app.ForceField(*ff_files)
        _FORCE_FIELD_CACHE[ff_files] = ff
    return ff


def restrained_minimise(
    pdb_in: str,
    pdb_out: str,
    restraint_k: float = 500.0,
    max_iterations: int = 200,
    implicit_solvent: bool = False,
    salt_molar: float = 0.15,
    platform_name: Optional[str] = None,
    n_threads: Optional[int] = None,
    starts: Sequence[str] = ("plain", "canonical"),
    tolerance: Optional[float] = None,
    force_fields: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Minimise *pdb_in* with harmonic position restraints; write *pdb_out*.

    restraint_k is in kJ/mol/nm^2.  Returns a report with energies and the RMSD of the
    restrained atoms, which should stay small: a large value means the restraints were
    too weak and the structure has drifted toward a canonical duplex.

    **Two starts, and the lower final energy is kept.** Minimising straight from the
    cgNA+ reconstruction left about 20 percent of nucleotides with alpha in g+ and
    gamma in g- together (correlation 0.999 between the two), evenly along the
    fragment. That concerted pair is a crankshaft: it moves O5' and C5' while leaving
    the phosphate and base atoms -- exactly the restrained ones -- where they are, so
    the minimiser can settle in either basin depending on where the raw build put the
    linkage. It is a trap, not a preference: on two test duplexes, holding alpha in g-
    and gamma in g+ with a flat-bottomed restraint, then switching that restraint off
    and minimising again to convergence, reached energies lower by 342 and 526 kJ/mol
    with no non-canonical alpha/gamma left, and the plain run had already converged, so
    iterations were not the issue.

    "plain" minimises from the build; "canonical" minimises with the flat-bottomed
    alpha/gamma restraint (free within +-29 degrees of g-/g+) and then again without it.
    The final energies are compared *without* that restraint term, so the canonical
    start cannot win by construction; it wins by finding a lower minimum of the same
    energy function.
    """
    import numpy as np
    import openmm
    from openmm import app, unit

    fixed_pdb = pdb_in
    try:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=pdb_in)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        fixed_pdb = pdb_out.replace(".pdb", "_fixed.pdb")
        with open(fixed_pdb, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        topology, positions = fixer.topology, fixer.positions
    except ImportError:
        pdb = app.PDBFile(pdb_in)
        topology, positions = pdb.topology, pdb.positions

    # DNA by default; RNA duplexes take AMBER RNA.OL3 and DNA-RNA hybrids both (Family K)
    ff_files = list(force_fields) if force_fields else ["amber14/DNA.OL15.xml"]
    if implicit_solvent:
        ff_files.append("implicit/gbn2.xml")
    forcefield = _force_field(tuple(ff_files), app)

    # Note: amber14 + implicit/gbn2.xml does not accept implicitSolventSaltConc in
    # current OpenMM, so the GB model runs at zero added salt.  That is acceptable here
    # because this minimisation only repairs local sugar geometry under strong
    # restraints; the salt dependence of the electrostatics is supplied by APBS, which
    # is solved separately at each ionic condition.
    kwargs = dict(
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=None,
    )
    try:
        system = forcefield.createSystem(topology, **kwargs)
    except ValueError:
        kwargs["nonbondedMethod"] = app.NoCutoff
        system = forcefield.createSystem(topology, **kwargs)

    # Harmonic restraints toward the cgNA+ positions.
    restraint = openmm.CustomExternalForce(
        "0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoule_per_mole / unit.nanometer**2)
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)

    pos = np.array(positions.value_in_unit(unit.nanometer))
    restrained_idx: List[int] = []
    for atom in topology.atoms():
        name = atom.name
        if name in BASE_ATOMS or name in PHOSPHATE_RESTRAINED:
            restraint.addParticle(atom.index, pos[atom.index])
            restrained_idx.append(atom.index)
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    # **Thread count is set explicitly, and this is not a micro-optimisation.**
    #
    # OpenMM's CPU platform defaults its `Threads` property to the machine's physical
    # core count -- 48 on these nodes -- and reads that from the hardware rather than
    # from the Slurm allocation or a cgroup. `OMP_NUM_THREADS` does not control it; only
    # `OPENMM_CPU_THREADS` or this property does.
    #
    # A 512-way array with `--cpus-per-task=1` therefore had every task spawning 48
    # threads, thousands of them per node fighting over the cores they were not
    # allocated. The array produced zero shards in 45 minutes and the cause looked like
    # filesystem contention, which it was not.
    #
    # Default: one thread, matching one allocated core. A caller that genuinely owns the
    # node can ask for more.
    threads = n_threads
    if threads is None:
        threads = int(os.environ.get("OPENMM_CPU_THREADS", "1"))
    plat = openmm.Platform.getPlatformByName(platform_name or "CPU")
    props = {}
    if "Threads" in plat.getPropertyNames():
        props["Threads"] = str(max(1, int(threads)))
    bias = _alpha_gamma_bias(topology, openmm)
    bias.setForceGroup(BIAS_FORCE_GROUP)
    system.addForce(bias)
    physical = set(range(32)) - {BIAS_FORCE_GROUP}

    simulation = app.Simulation(topology, system, integrator, plat, props)
    simulation.context.setPositions(positions)
    e0 = simulation.context.getState(getEnergy=True, groups=physical).getPotentialEnergy()

    # `tolerance` (kJ/mol/nm) is OpenMM's L-BFGS stopping criterion on the force; None keeps
    # OpenMM's default (10). With the historical 200-iteration cap the minimiser stops on
    # the cap long before the tolerance, so its end point depends on the path, and tiny
    # rounding differences between runs move atoms (DECISIONS.md, composition gate). A
    # converged run (cap large, tolerance small) removes that path dependence.
    min_kw = {"maxIterations": int(max_iterations)}
    if tolerance is not None:
        min_kw["tolerance"] = float(tolerance) * unit.kilojoule_per_mole / unit.nanometer
    results = {}
    for start in starts:
        simulation.context.setPositions(positions)
        if start == "canonical":
            simulation.context.setParameter("kb_alpha_gamma", ALPHA_GAMMA_BIAS_K)
            simulation.minimizeEnergy(**min_kw)
        elif start != "plain":
            raise ValueError(f"unknown minimisation start {start!r}")
        simulation.context.setParameter("kb_alpha_gamma", 0.0)
        simulation.minimizeEnergy(**min_kw)
        st = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True,
                                         groups=physical)
        f = np.asarray(st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer))
        fn = np.linalg.norm(f, axis=1)
        results[start] = (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                          st.getPositions(asNumpy=True),
                          (float(fn.max()), float(np.sqrt((fn ** 2).mean()))))
    best = min(results, key=lambda k: results[k][0])
    e1 = results[best][0] * unit.kilojoule_per_mole
    final_positions = results[best][1]

    new_pos = np.array(final_positions.value_in_unit(unit.nanometer))
    d = np.linalg.norm(new_pos[restrained_idx] - pos[restrained_idx], axis=1)
    report = {
        "energy_initial_kJ_mol": float(e0.value_in_unit(unit.kilojoule_per_mole)),
        "energy_final_kJ_mol": float(e1.value_in_unit(unit.kilojoule_per_mole)),
        "n_atoms": int(topology.getNumAtoms()),
        "n_restrained": len(restrained_idx),
        "restrained_rmsd_angstrom": float(np.sqrt((d**2).mean()) * 10.0),
        "restrained_max_shift_angstrom": float(d.max() * 10.0),
        "restraint_k_kJ_mol_nm2": restraint_k,
        "start_kept": best,
    }
    for k, (e, _, (fmax, frms)) in results.items():
        report[f"energy_final_{k}_kJ_mol"] = float(e)
        report[f"max_force_{k}_kJ_mol_nm"] = fmax
        report[f"rms_force_{k}_kJ_mol_nm"] = frms
    report["max_iterations"] = int(max_iterations)
    report["tolerance_kJ_mol_nm"] = tolerance

    with open(pdb_out, "w") as f:
        app.PDBFile.writeFile(topology, final_positions, f, keepIds=True)
    if fixed_pdb != pdb_in and os.path.exists(fixed_pdb):
        os.remove(fixed_pdb)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="pdb_in", required=True)
    ap.add_argument("--out", dest="pdb_out", required=True)
    ap.add_argument("--restraint-k", type=float, default=500.0)
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    rep = restrained_minimise(
        a.pdb_in, a.pdb_out, a.restraint_k, a.max_iterations, platform_name=a.platform
    )
    print(json.dumps(rep, indent=2))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
