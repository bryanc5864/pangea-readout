#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, stage 1: thread a parallel G-quadruplex onto solved cores and loops.

Core: the twelve tetrad guanines (four columns x three tetrads) of a solved parallel
structure. Each propeller loop of length l is taken from a solved parallel structure with a
loop of the same length: its two anchoring guanines (the 3' guanine of one column and the
5' guanine of the next) are superposed onto the core's corresponding guanines (Kabsch, on
matching atom names) and the loop nucleotides carried along. Among the library's loops of
that length the one whose anchors fit the core best is used. Base substitution and the
restrained minimisation with the channel K+ ions follow in later stages.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import g4_templates as T

ANCHOR_ATOMS = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
                "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4")
BACKBONE = ("P", "C4'", "C1'")


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """R, t minimising |R P + t - Q|; returns (R, t, rmsd)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1))))
    return R, t, rmsd


class Structure:
    """A solved parallel unimolecular G4: residues, ordered columns, loops."""

    def __init__(self, path: str):
        self.path = path
        atoms, self.ions = T.read_pdb(path)
        self.res = T.residues(atoms)
        tet = T.find_tetrads(self.res)
        self.geom = T.tetrad_geometry(self.res, tet)
        runs = T.g_runs(self.geom["order"])
        self.columns = sorted([sorted(r, key=lambda k: k[1]) for r in runs], key=lambda r: r[0][1])
        self.chain = self.columns[0][0][0]
        self.loops = []
        for i in range(3):
            a, b = self.columns[i][-1], self.columns[i + 1][0]
            keys = [(self.chain, k) for k in range(a[1] + 1, b[1])]
            # usable only if every loop nucleotide is present as a standard residue (gaps in
            # numbering and modified nucleotides stored as HETATM make a loop unusable)
            # ... and its base complete enough to fit a base frame for substitution (an abasic
            # or truncated residue cannot be mutated: validation hit one, 2 base atoms)
            ok = all(k in self.res and all(n in self.res[k]["atoms"] for n in ("P", "C4'", "C1'"))
                     and sum(n in self.res[k]["atoms"] for n in ("N1", "C2", "N3", "C4", "C5", "C6")) >= 5
                     for k in keys)
            self.loops.append({"anchor5": a, "anchor3": b, "residues": keys, "usable": ok})

    def coords(self, key, names) -> Optional[np.ndarray]:
        at = self.res[key]["atoms"]
        if not all(n in at for n in names):
            return None
        return np.array([at[n] for n in names])


def place_loop(core: Structure, i: int, donor: Structure, j: int):
    """Superpose donor loop j's anchors on core loop i's anchors; return (transformed loop
    residues {key: {atom: xyz}}, resnames, anchor rmsd)."""
    names = [n for n in ANCHOR_ATOMS]
    cP, dP = [], []
    for ck, dk in ((core.loops[i]["anchor5"], donor.loops[j]["anchor5"]),
                   (core.loops[i]["anchor3"], donor.loops[j]["anchor3"])):
        common = [n for n in names if n in core.res[ck]["atoms"] and n in donor.res[dk]["atoms"]]
        cP += [core.res[ck]["atoms"][n] for n in common]
        dP += [donor.res[dk]["atoms"][n] for n in common]
    R, t, rmsd = kabsch(np.array(dP), np.array(cP))
    out, rn = [], []
    for k in donor.loops[j]["residues"]:
        out.append({n: R @ x + t for n, x in donor.res[k]["atoms"].items()})
        rn.append(donor.res[k]["resname"])
    return out, rn, rmsd


def best_loop(core: Structure, i: int, library: List[Structure], length: int, exclude=()):
    best = None
    for s in library:
        if s.path in exclude:
            continue
        for j, lp in enumerate(s.loops):
            if len(lp["residues"]) != length or not lp["usable"]:
                continue
            res, rn, rmsd = place_loop(core, i, s, j)
            if best is None or rmsd < best[2]:
                best = (res, rn, rmsd, os.path.basename(s.path), j)
    return best


# ---------------------------------------------------------------------------------------
# Stage 2: full-atom model of a target quadruplex (assembly, base substitution, minimisation)
# ---------------------------------------------------------------------------------------
SUGAR_BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
PHOSPHATE_GROUP = ("P", "OP1", "OP2")
LETTER_TO_RESNAME = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
CLASH_A = 2.5


def _heavy(atoms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Heavy atoms only (NMR models carry hydrogens; PDBFixer re-adds them)."""
    out = {}
    for n, x in atoms.items():
        stem = n.lstrip("0123456789")
        if stem.startswith("H"):
            continue
        out["OP1" if n == "O1P" else "OP2" if n == "O2P" else "C7" if n == "C5M" else n] = x
    return out


_TEMPLATES = None


def _templates():
    """Ideal Tsukuba base atoms (frame fitting) and full nucleotide templates in base frames."""
    global _TEMPLATES
    if _TEMPLATES is None:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        import cgna_atomic as CA
        T_ = CA.NucleotideTemplates()
        _TEMPLATES = (T_.ideal_bases, T_.templates, CA.kabsch_se3, CA.apply_se3)
    return _TEMPLATES


def substitute_base(atoms: Dict[str, np.ndarray], letter_from: str, letter_to: str
                    ) -> Dict[str, np.ndarray]:
    """Replace a nucleotide's base by `letter_to`, keeping its sugar-phosphate backbone.

    The donor's base frame is fitted (Kabsch on the ideal Tsukuba base atoms, as in
    `frames.fit_base_frames`) and the target base's template atoms placed in that frame,
    so the new base keeps the old one's position, orientation and glycosidic geometry.
    """
    ideal, templates, kabsch_se3, apply_se3 = _templates()
    I, A = [], []
    for nm, pos in ideal[letter_from].items():
        for alias in ((nm, "C7") if nm == "C5M" else (nm,)):
            if alias in atoms:
                I.append(pos); A.append(atoms[alias]); break
    if len(I) < 3:
        raise ValueError(f"cannot fit a {letter_from} base frame ({len(I)} atoms)")
    tau = kabsch_se3(np.asarray(I, float), np.asarray(A, float))
    out = {n: x for n, x in atoms.items() if n in SUGAR_BACKBONE}
    for n, x in templates[letter_to].items():
        if n in SUGAR_BACKBONE or n.lstrip("0123456789").startswith("H"):
            continue
        out["C7" if n == "C5M" else n] = apply_se3(tau, np.asarray(x, float)[None])[0]
    return out


def _clashes(xyz: np.ndarray, others: np.ndarray, cut: float = CLASH_A) -> int:
    if len(xyz) == 0 or len(others) == 0:
        return 0
    d = np.linalg.norm(xyz[:, None, :] - others[None, :, :], axis=2)
    return int((d < cut).sum())


def candidate_loops(core: "Structure", i: int, library: List["Structure"], length: int,
                    exclude_sequences=(), seq_of=None, top: int = 8):
    """Donor loops of `length` for core loop i, ranked by anchor superposition RMSD; one per
    donor structure, donors whose sequence is in `exclude_sequences` skipped."""
    out = []
    for s in library:
        if seq_of is not None and seq_of.get(s.path) in exclude_sequences:
            continue
        best = None
        for j, lp in enumerate(s.loops):
            if len(lp["residues"]) != length or not lp["usable"]:
                continue
            res, rn, rmsd = place_loop(core, i, s, j)
            if best is None or rmsd < best[2]:
                best = (res, rn, rmsd, os.path.basename(s.path), j)
        if best is not None:
            out.append(best)
    out.sort(key=lambda b: b[2])
    return out[:top]


def assemble_model(core: "Structure", loops_seq: List[str], donors: List[tuple]
                   ) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """Residues in sequence order: core column 1, loop 1, column 2, ..., column 4.

    `donors[i]` is a `candidate_loops` entry for loop i; its bases are substituted to
    `loops_seq[i]`. Core guanines are kept as solved. Returns [(resname, heavy atoms)].
    """
    out = []
    for c in range(4):
        for k in core.columns[c]:
            out.append(("DG", _heavy(core.res[k]["atoms"])))
        if c < 3:
            res, rn, _, _, _ = donors[c]
            for r, name, letter in zip(res, rn, loops_seq[c]):
                atoms = _heavy(r)
                src = name[-1] if name[-1] in "ACGTU" else "T"
                src = "T" if src == "U" else src
                out.append((LETTER_TO_RESNAME[letter], substitute_base(atoms, src, letter)
                            if src != letter else atoms))
    # 5' terminus carries no phosphate in the AMBER templates
    out[0] = (out[0][0], {n: x for n, x in out[0][1].items() if n not in PHOSPHATE_GROUP})
    return out


def write_model_pdb(residues_: List[Tuple[str, Dict[str, np.ndarray]]], path: str,
                    ions: Optional[List[np.ndarray]] = None) -> str:
    lines, serial = [], 1
    for ri, (rn, atoms) in enumerate(residues_, start=1):
        for n, x in atoms.items():
            nm = n if len(n) == 4 else " " + n.ljust(3)
            lines.append(f"ATOM  {serial:5d} {nm} {rn:>3s} A{ri:4d}    "
                         f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {n[0]:>2s}")
            serial += 1
    for k, x in enumerate(ions or [], start=1):
        lines.append(f"HETATM{serial:5d}  K     K B{k:4d}    "
                     f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           K")
        serial += 1
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def minimise_model(pdb_in: str, pdb_out: str, core_residues: Sequence[int],
                   k_core: float = 2000.0, tolerance: float = 1.0,
                   max_iterations: int = 20000, platform: Optional[str] = None) -> Dict[str, float]:
    """AMBER OL15 + GBn2 minimisation; core heavy atoms restrained (kJ/mol/nm^2), loops free."""
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    # PDBFixer's own minimisations (added atoms, hydrogens) run on the CPU: left unset they take the fastest
    # platform, CUDA on a GPU node, and every call there leaked a CUDA context (~200 MB, ~15 device handles),
    # which exhausted the GPU within one block of the quadruplex build (DECISIONS 2026-11-24)
    fx = PDBFixer(filename=pdb_in, platform=openmm.Platform.getPlatformByName("CPU"))
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(fx.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", k_core)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    core = set(int(r) for r in core_residues)
    n_rest = 0
    for atom in fx.topology.atoms():
        if atom.residue.index + 1 in core and atom.element is not None and atom.element.symbol != "H":
            rest.addParticle(atom.index, pos[atom.index]); n_rest += 1
    system.addForce(rest)
    integ = openmm.VerletIntegrator(0.001)
    pname = platform or "CPU"
    plat = openmm.Platform.getPlatformByName(pname)
    props = {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")} if pname == "CPU" else {"Precision": "mixed"}
    ctx = openmm.Context(system, integ, plat, props)
    ctx.setPositions(fx.positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(ctx, tolerance, max_iterations)
    st = ctx.getState(getEnergy=True, getPositions=True, getForces=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.linalg.norm(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer), axis=1)
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(fx.topology, st.getPositions(), fh, keepIds=True)
    return {"energy_start_kJ_mol": e0, "energy_final_kJ_mol": e1, "max_force": float(f.max()),
            "n_restrained": n_rest, "n_atoms": system.getNumParticles()}


# ---------------------------------------------------------------------------------------
# Stage 4: MD ensemble (PLAN.md: "Solved structures, MD ensembles, APBS ...")
# ---------------------------------------------------------------------------------------
def md_ensemble(pdb_min: str, out_prefix: str, core_residues: Sequence[int], n_frames: int = 5,
                production_ps: float = 50.0, heat_ps: float = 10.0, k_core: float = 2000.0,
                temperature_K: float = 300.0, seed: int = 0, equil_ps: float = 0.0,
                salt_M: float = 0.0, platform: Optional[str] = None) -> Dict[str, object]:
    """Implicit-solvent (GBn2) Langevin MD of a minimised quadruplex model; frames written as
    PDB files `<out_prefix>_f<k>.pdb`, evenly spaced over the production run.

    The tetrad core's heavy atoms are held as in the minimisation (k_core 2000 kJ/mol/nm^2),
    standing in for the channel cations a continuum solvent cannot keep in place: with a weak
    restraint (200) the probe's core drifted and its inner-site potential moved by up to 20 kT/e
    and the ion binding by 150 kJ/mol between frames, an artefact of the missing ions. The
    ensemble samples what is flexible -- the loops and the backbone outside the core. 2 fs steps,
    bonds to hydrogen constrained. `salt_M` screens the GB electrostatics (Debye-Hueckel kappa,
    water at the run temperature); `equil_ps` is run and discarded after heating; `platform`
    (default: PANGEA_OPENMM_PLATFORM, else CPU) selects OpenMM's CUDA/OpenCL/CPU platform.
    """
    import openmm
    from openmm import app, unit
    pdb = app.PDBFile(pdb_min)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    kw = {}
    if salt_M > 0:                    # kappa = 50.33355 sqrt(I / (eps T)) 1/A (OpenMM's GB convention)
        kw["implicitSolventKappa"] = 50.33355 * (salt_M / (78.5 * temperature_K)) ** 0.5 * 10.0 / unit.nanometer
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, **kw)
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", k_core)
    for p_ in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p_)
    core = set(int(r) for r in core_residues)
    for atom in pdb.topology.atoms():
        if atom.residue.index + 1 in core and atom.element is not None and atom.element.symbol != "H":
            rest.addParticle(atom.index, pos[atom.index])
    system.addForce(rest)
    integ = openmm.LangevinMiddleIntegrator(temperature_K * unit.kelvin, 1.0 / unit.picosecond,
                                            0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(int(seed) + 1)
    pname = platform or os.environ.get("PANGEA_OPENMM_PLATFORM", "CPU")
    plat = openmm.Platform.getPlatformByName(pname)
    props = ({"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")} if pname == "CPU"
             else {"Precision": "mixed"})
    sim = app.Simulation(pdb.topology, system, integ, plat, props)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(10 * unit.kelvin, int(seed) + 1)
    n_heat = int(heat_ps / 0.002)
    for i in range(10):                                   # heat 10 K -> T in 10 stages
        integ.setTemperature((10 + (temperature_K - 10) * (i + 1) / 10) * unit.kelvin)
        sim.step(max(1, n_heat // 10))
    if equil_ps > 0:
        sim.step(int(equil_ps / 0.002))
    n_prod = int(production_ps / 0.002)
    every = max(1, n_prod // n_frames)
    frames, energies = [], []
    for k in range(n_frames):
        sim.step(every)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        path = f"{out_prefix}_f{k}.pdb"
        with open(path, "w") as fh:
            app.PDBFile.writeFile(pdb.topology, st.getPositions(), fh, keepIds=True)
        frames.append(path)
        energies.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    x0 = pos[[a.index for a in pdb.topology.atoms() if a.element is not None and a.element.symbol != "H"]]
    def heavy_rmsd(path):
        q = np.array(app.PDBFile(path).positions.value_in_unit(unit.nanometer))
        q = q[[a.index for a in pdb.topology.atoms() if a.element is not None and a.element.symbol != "H"]]
        return float(kabsch(q * 10, x0 * 10)[2])
    return {"frames": frames, "energies_kJ_mol": energies,
            "rmsd_to_minimised_A": [heavy_rmsd(f) for f in frames]}


def md_ensemble_batch(models, n_frames: int = 5, production_ps: float = 500.0, heat_ps: float = 20.0,
                      equil_ps: float = 100.0, k_core: float = 2000.0, temperature_K: float = 300.0,
                      salt_M: float = 0.15, seed: int = 0, platform: Optional[str] = None,
                      spacing_A: float = 80.0, cutoff_nm: float = 2.0) -> List[Dict[str, object]]:
    """`md_ensemble` for many models at once: every model is a replica in one OpenMM system, placed on
    a lattice SPACING_A apart, GBn2 with a non-periodic cutoff of cutoff_nm (replicas cannot interact
    and Born radii stay per replica), so one GPU integrates them together. A single 500-atom model is
    kernel-launch bound (one H200: 250 ps in 20 s, and four concurrent processes only 1.15x faster);
    packed replicas share every launch. `models`: dicts with pdb_min, core_residues, out_prefix.
    Returns one dict per model, as `md_ensemble`."""
    import openmm
    from openmm import app, unit
    if not models:
        return []
    pdbs = [app.PDBFile(m["pdb_min"]) for m in models]
    side = int(np.ceil(len(models) ** (1 / 3)))
    mod = None; offsets, ranges = [], []
    for k, p in enumerate(pdbs):
        X = np.array(p.positions.value_in_unit(unit.angstrom))
        off = np.array([k % side, (k // side) % side, k // (side * side)], float) * spacing_A - X.mean(0)
        offsets.append(off)
        pos = (X + off) * unit.angstrom
        if mod is None:
            mod = app.Modeller(p.topology, pos); start = 0
        else:
            start = mod.topology.getNumAtoms(); mod.add(p.topology, pos)
        ranges.append((start, start + p.topology.getNumAtoms()))
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    kw = {"implicitSolventKappa": 50.33355 * (salt_M / (78.5 * temperature_K)) ** 0.5 * 10.0 / unit.nanometer} if salt_M > 0 else {}
    system = ff.createSystem(mod.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=cutoff_nm * unit.nanometer,
                             constraints=app.HBonds, **kw)
    allpos = np.array(mod.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", k_core)
    for p_ in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p_)
    atoms = list(mod.topology.atoms())
    for (a0, a1), m in zip(ranges, models):
        core = set(int(r) for r in m["core_residues"])
        res_index0 = atoms[a0].residue.index
        for i in range(a0, a1):
            at = atoms[i]
            if at.residue.index - res_index0 + 1 in core and at.element is not None and at.element.symbol != "H":
                rest.addParticle(i, allpos[i])
    system.addForce(rest)
    integ = openmm.LangevinMiddleIntegrator(temperature_K * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(int(seed) + 1)
    pname = platform or os.environ.get("PANGEA_OPENMM_PLATFORM", "CPU")
    props = {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")} if pname == "CPU" else {"Precision": "mixed"}
    sim = app.Simulation(mod.topology, system, integ, openmm.Platform.getPlatformByName(pname), props)
    sim.context.setPositions(mod.positions)
    sim.context.setVelocitiesToTemperature(10 * unit.kelvin, int(seed) + 1)
    n_heat = int(heat_ps / 0.002)
    for i in range(10):
        integ.setTemperature((10 + (temperature_K - 10) * (i + 1) / 10) * unit.kelvin)
        sim.step(max(1, n_heat // 10))
    if equil_ps > 0:
        sim.step(int(equil_ps / 0.002))
    every = max(1, int(production_ps / 0.002) // n_frames)
    frames = [[] for _ in models]
    for f in range(n_frames):
        sim.step(every)
        X = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if not np.all(np.isfinite(X)):
            raise FloatingPointError("non-finite coordinates in batched MD")
        for k, ((a0, a1), m, p) in enumerate(zip(ranges, models, pdbs)):
            path = f"{m['out_prefix']}_f{f}.pdb"
            with open(path, "w") as fh:
                app.PDBFile.writeFile(p.topology, (X[a0:a1] - offsets[k]) * unit.angstrom, fh, keepIds=True)
            frames[k].append(path)
    out = []
    for k, (m, p) in enumerate(zip(models, pdbs)):
        x0 = np.array(p.positions.value_in_unit(unit.angstrom))
        heavy = [a.index for a in p.topology.atoms() if a.element is not None and a.element.symbol != "H"]
        rms = []
        for f in frames[k]:
            q = np.array(app.PDBFile(f).positions.value_in_unit(unit.angstrom))
            rms.append(float(kabsch(q[heavy], x0[heavy])[2]))
        out.append({"frames": frames[k], "rmsd_to_minimised_A": rms})
    return out


def minimise_batch(models, k_core: float = 2000.0, tolerance: float = 1.0, max_iterations: int = 20000,
                   platform: Optional[str] = None, spacing_A: float = 80.0, cutoff_nm: float = 2.0) -> List[Dict[str, float]]:
    """`minimise_model` for many raw models at once (replica system, as `md_ensemble_batch`).
    `models`: dicts with pdb_raw, pdb_out, core_residues. Hydrogens added by PDBFixer per model."""
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    if not models:
        return []
    tops, poss = [], []
    for m in models:
        # CPU for PDBFixer's own minimisations (see minimise_model)
        fx = PDBFixer(filename=m["pdb_raw"], platform=openmm.Platform.getPlatformByName("CPU"))
        fx.findMissingResidues(); fx.missingResidues = {}
        fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
        tops.append(fx.topology); poss.append(np.array(fx.positions.value_in_unit(unit.angstrom)))
    side = int(np.ceil(len(models) ** (1 / 3)))
    mod, offsets, ranges = None, [], []
    for k, (t, X) in enumerate(zip(tops, poss)):
        off = np.array([k % side, (k // side) % side, k // (side * side)], float) * spacing_A - X.mean(0)
        offsets.append(off)
        if mod is None:
            mod = app.Modeller(t, (X + off) * unit.angstrom); start = 0
        else:
            start = mod.topology.getNumAtoms(); mod.add(t, (X + off) * unit.angstrom)
        ranges.append((start, start + t.getNumAtoms()))
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(mod.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=cutoff_nm * unit.nanometer,
                             constraints=None)
    allpos = np.array(mod.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", k_core)
    for p_ in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p_)
    atoms = list(mod.topology.atoms())
    for (a0, a1), m in zip(ranges, models):
        core = set(int(r) for r in m["core_residues"]); r0 = atoms[a0].residue.index
        for i in range(a0, a1):
            at = atoms[i]
            if at.residue.index - r0 + 1 in core and at.element is not None and at.element.symbol != "H":
                rest.addParticle(i, allpos[i])
    system.addForce(rest)
    pname = platform or "CPU"
    props = {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")} if pname == "CPU" else {"Precision": "mixed"}
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(pname), props)
    ctx.setPositions(mod.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, tolerance, max_iterations)
    st_ = ctx.getState(getPositions=True, getForces=True)
    X = st_.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    if not np.all(np.isfinite(X)):
        raise FloatingPointError("non-finite coordinates after batched minimisation")
    # The minimiser's stopping rule is the RMS force of the whole system, which a single unrelaxed
    # replica hides behind 63 relaxed ones (seen: one model left at -426 kJ/mol/nt beside -870).
    # Any replica whose own RMS force is above 3x the tolerance is minimised again on its own.
    Fz = st_.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    redo = [k for k, (a0, a1) in enumerate(ranges) if float(np.sqrt(np.mean(Fz[a0:a1] ** 2))) > 3.0 * tolerance]
    for k in redo:
        a0, a1 = ranges[k]
        sysk = ff.createSystem(tops[k], nonbondedMethod=app.NoCutoff, constraints=None)
        restk = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restk.addGlobalParameter("k", k_core)
        for p_ in ("x0", "y0", "z0"):
            restk.addPerParticleParameter(p_)
        core = set(int(r) for r in models[k]["core_residues"]); r0 = atoms[a0].residue.index
        for i in range(a0, a1):
            at = atoms[i]
            if at.residue.index - r0 + 1 in core and at.element is not None and at.element.symbol != "H":
                restk.addParticle(i - a0, allpos[i] - offsets[k] / 10.0)
        sysk.addForce(restk)
        ik = openmm.VerletIntegrator(0.001)
        ck = openmm.Context(sysk, ik, openmm.Platform.getPlatformByName(pname), props)
        ck.setPositions((X[a0:a1] - offsets[k]) * unit.angstrom)
        openmm.LocalEnergyMinimizer.minimize(ck, tolerance, max_iterations)
        X[a0:a1] = ck.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom) + offsets[k]
    out = []
    # per-replica energy as minimise_model reports it: the replica alone (no cutoff) plus its restraint
    for k, ((a0, a1), m, t) in enumerate(zip(ranges, models, tops)):
        xk = (X[a0:a1] - offsets[k])
        with open(m["pdb_out"], "w") as fh:
            app.PDBFile.writeFile(t, xk * unit.angstrom, fh, keepIds=True)
        sysk = ff.createSystem(t, nonbondedMethod=app.NoCutoff, constraints=None)
        ik = openmm.VerletIntegrator(0.001)
        ck = openmm.Context(sysk, ik, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
        ck.setPositions(xk * unit.angstrom)
        e = ck.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        core = set(int(r) for r in m["core_residues"]); r0 = atoms[a0].residue.index
        sel = [i - a0 for i in range(a0, a1) if atoms[i].residue.index - r0 + 1 in core
               and atoms[i].element is not None and atoms[i].element.symbol != "H"]
        d2 = np.sum(((X[a0:a1][sel] - offsets[k]) / 10.0 - (allpos[a0:a1][sel] - offsets[k] / 10.0)) ** 2)
        out.append({"energy_final_kJ_mol": float(e + 0.5 * k_core * d2), "n_atoms": int(t.getNumAtoms())})
    return out
