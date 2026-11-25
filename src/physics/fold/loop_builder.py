#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Template-free loops for folds whose loop type and length no solved structure provides (long
quadruplex loops, i-motif and hairpin loops): PDBFixer builds the missing nucleotides from the
sequence between the two anchors (approximate placement), then AMBER OL15/GBn2 relaxes them with
every known heavy atom restrained -- minimisation, a short anneal (ANNEAL_PS at 400 K, cooled to
300 K over COOL_PS), and a final minimisation -- so the loop leaves the straight-line start and
packs against the fold without moving it. The result is a physically valid conformation, not a
prediction of the solved one (validated against solved long loops, DECISIONS.md).
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

ANNEAL_PS, COOL_PS, K_KNOWN = 20.0, 20.0, 2000.0


def write_gapped_pdb(residues: List[Tuple[str, Dict[str, np.ndarray]]], missing: Dict[int, List[str]],
                     path: str, chain: str = "A") -> str:
    """`residues` in sequence order (resname, heavy atoms); `missing[i]` = resnames of loop
    nucleotides inserted before known residue i (0-based). SEQRES lists the full chain."""
    full, numbering = [], []
    k = 1
    for i, (rn, _) in enumerate(residues):
        for m in missing.get(i, []):
            full.append(m); k += 1
        full.append(rn); numbering.append(k); k += 1
    for m in missing.get(len(residues), []):
        full.append(m); k += 1
    lines = [f"SEQRES {j // 13 + 1:3d} {chain} {len(full):4d}  " + " ".join(f"{r:>3s}" for r in full[j:j + 13])
             for j in range(0, len(full), 13)]
    serial = 1
    for (rn, atoms), num in zip(residues, numbering):
        for n, x in atoms.items():
            nm = n if len(n) == 4 else " " + n.ljust(3)
            lines.append(f"ATOM  {serial:5d} {nm} {rn:>3s} {chain}{num:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {n[0]:>2s}")
            serial += 1
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_gapped_pdb_multi(chains: Dict[str, Tuple[List[Tuple[str, Dict[str, np.ndarray]]], Dict[int, List[str]]]],
                           path: str) -> Tuple[str, Dict[str, List[int]]]:
    """Several chains: {chain: (residues, missing)} as `write_gapped_pdb`; returns the path and,
    per chain, the residue numbers of the known residues."""
    lines, atoms_lines, known = [], [], {}
    serial = 1
    for ch, (residues, missing) in chains.items():
        full, numbering = [], []
        k = 1
        for i, (rn, _) in enumerate(residues):
            for m in missing.get(i, []):
                full.append(m); k += 1
            full.append(rn); numbering.append(k); k += 1
        for m in missing.get(len(residues), []):
            full.append(m); k += 1
        lines += [f"SEQRES {j // 13 + 1:3d} {ch} {len(full):4d}  " + " ".join(f"{r:>3s}" for r in full[j:j + 13])
                  for j in range(0, len(full), 13)]
        for (rn, atoms), num in zip(residues, numbering):
            for n, x in atoms.items():
                nm = n if len(n) == 4 else " " + n.ljust(3)
                atoms_lines.append(f"ATOM  {serial % 100000:5d} {nm} {rn:>3s} {ch}{num:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {n[0]:>2s}")
                serial += 1
        atoms_lines.append("TER")
        known[ch] = numbering
    open(path, "w").write("\n".join(lines + atoms_lines + ["END"]) + "\n")
    return path, known


BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}


def fill_and_relax(gapped_pdb: str, out_pdb: str, known_resseq, seed: int = 0,
                   platform: str = None, free_backbone=()) -> dict:
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    # PDBFixer's own minimisations (added atoms, hydrogens) run on the CPU: left unset they take the fastest
    # platform, CUDA on a GPU node, and every call there leaked a CUDA context (~200 MB, ~15 device handles),
    # which exhausted the GPU within one block of the quadruplex build (DECISIONS 2026-11-24)
    fx = PDBFixer(filename=gapped_pdb, platform=openmm.Platform.getPlatformByName("CPU"))
    fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    pos = np.array(fx.positions.value_in_unit(unit.nanometer))
    # known residues: a set of resseq (one chain) or of (chain, resseq); free_backbone: (chain, resseq)
    # of known residues whose sugar-phosphate stays unrestrained (backbone joins to be closed)
    known = {(None, int(r)) if not isinstance(r, tuple) else (r[0], int(r[1])) for r in known_resseq}
    freebb = {(c, int(r)) for c, r in free_backbone}
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", K_KNOWN)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    built = []
    for a in fx.topology.atoms():
        key = (a.residue.chain.id, int(a.residue.id))
        if key in known or (None, key[1]) in known:
            if a.element.symbol != "H" and not (key in freebb and a.name in BACKBONE):
                rest.addParticle(a.index, pos[a.index])
        else:
            built.append(a.index)
    system.addForce(rest)
    integ = openmm.LangevinMiddleIntegrator(400 * unit.kelvin, 5.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(seed + 1)
    pname = platform or os.environ.get("PANGEA_OPENMM_PLATFORM", "CPU")
    props = {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")} if pname == "CPU" else {"Precision": "mixed"}
    sim = app.Simulation(fx.topology, system, integ, openmm.Platform.getPlatformByName(pname), props)
    sim.context.setPositions(fx.positions)
    sim.minimizeEnergy(tolerance=10.0, maxIterations=5000)
    sim.context.setVelocitiesToTemperature(400 * unit.kelvin, seed + 1)
    sim.step(int(ANNEAL_PS / 0.002))
    n_cool = int(COOL_PS / 0.002)
    for i in range(10):
        integ.setTemperature((400 - 100 * (i + 1) / 10) * unit.kelvin)
        sim.step(max(1, n_cool // 10))
    sim.minimizeEnergy(tolerance=1.0, maxIterations=20000)
    st = sim.context.getState(getPositions=True, getEnergy=True)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(fx.topology, st.getPositions(), fh, keepIds=True)
    return {"n_built_atoms": len(built), "energy_kJ_mol": st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)}
