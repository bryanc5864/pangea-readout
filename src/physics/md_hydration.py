#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Explicit-solvent MD hydration library (PLAN.md: Family G "minor and major groove hydration
occupancy, residence time, spine continuity and water bridge probability ... interfacial water
entropy" from "MD-derived hydration"; Family K "2-prime hydroxyl hydration and dynamics" from
"hexamer-scale RNA MD").

For every RC-unique tetramer context and class (B-DNA, A-RNA, DNA-RNA hybrid): the 12-mer
GCGC + context + GCGC built from its coarse-grained ground state (cgNA+ / cgRNA+ / cgHYB+, the
geometry every other family uses), AMBER OL15 (DNA) / OL3 (RNA) in OPC water with 0.15 M KCl plus
neutralising K+, rhombic dodecahedron with 1.0 nm padding, PME, hydrogen mass repartitioning (4 fs),
300 K Langevin, 1 bar Monte Carlo barostat: minimisation, 100 ps heating and 500 ps NPT with the
solute's heavy atoms restrained, 1 ns free equilibration, then PRODUCTION_NS sampled every
SAMPLE_PS. Observables per base pair (the central pairs enter the context lookup), from the first
hydration shell of each groove's acceptors (water O within SHELL_A):
  occupancy (mean first-shell waters) of minor- and major-groove acceptors; residence time (mean
  uninterrupted stay of a water in an acceptor's shell); spine continuity (probability that the
  minor-groove acceptors of pair i and of the opposite strand's pair i+1 share a water); water
  bridge probability (a water in the shell of two acceptors of different nucleotides); interfacial
  water translational entropy (first-order inhomogeneous solvation theory: -k rho0 sum g ln g dV over
  the voxels of the groove region, water density g relative to bulk, in the solute's frame);
  RNA strands: 2'-OH first-shell occupancy and residence time, 2'-OH...O4'(i+1) hydrogen-bond
  probability, the HO2' torsion's population in its O3', base and O4' domains, sugar C3'-endo
  fraction and O2' RMSF.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAD = "GCGC"
PRODUCTION_NS, SAMPLE_PS, EQUIL_NS = 10.0, 2.0, 1.0
SHELL_A = 3.4
MINOR = {"A": ("N3",), "G": ("N3", "N2"), "C": ("O2",), "T": ("O2",), "U": ("O2",)}
MAJOR = {"A": ("N7", "N6"), "G": ("N7", "O6"), "C": ("N4",), "T": ("O4",), "U": ("O4",)}
ION_M = 0.15


def build(context: str, cls: str, work: str) -> str:
    import cgna_atomic as ca
    seq = PAD + context + PAD[::-1]
    if cls == "dna":
        atoms = ca.build_duplex(seq)
    else:
        import rna_atomic as R
        import duplex_classes as K
        atoms = R.build_rna_duplex(seq, K.SETS["rna"]) if cls == "rna" else R.build_hybrid(seq, K.SETS["hyb"])
    raw = os.path.join(work, "raw.pdb"); ca.write_pdb(atoms, raw)
    return raw


def simulate(raw_pdb: str, cls: str, work: str, platform: Optional[str] = None, production_ns: float = PRODUCTION_NS,
             sample_ps: float = SAMPLE_PS, seed: int = 0):
    """Returns (topology, solute atom indices, water O indices, frames of solute+water-O positions
    in A, box vectors) -- analysis frames only, no trajectory on disk."""
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    fx = PDBFixer(filename=raw_pdb); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ffs = ["amber14/DNA.OL15.xml"] if cls == "dna" else (["amber14/RNA.OL3.xml"] if cls == "rna"
                                                         else ["amber14/DNA.OL15.xml", "amber14/RNA.OL3.xml"])
    ff = app.ForceField(*ffs, "amber14/opc.xml")
    mod = app.Modeller(fx.topology, fx.positions)
    mod.addSolvent(ff, model="tip4pew", padding=1.0 * unit.nanometer, boxShape="dodecahedron",
                   positiveIon="K+", negativeIon="Cl-", ionicStrength=ION_M * unit.molar, neutralize=True)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, hydrogenMass=1.5 * unit.amu)
    solute = [a.index for a in mod.topology.atoms() if a.residue.name not in ("HOH", "K+", "Cl-", "K", "CL")]
    heavy = [i for i in solute if list(mod.topology.atoms())[i].element.symbol != "H"]
    rest = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    rest.addGlobalParameter("k", 1000.0)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    pos0 = mod.positions
    for i in heavy:
        rest.addParticle(i, pos0[i].value_in_unit(unit.nanometer))
    system.addForce(rest)
    baro = openmm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin)
    system.addForce(baro)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds)
    integ.setRandomNumberSeed(seed + 1)
    pname = platform or os.environ.get("PANGEA_OPENMM_PLATFORM", "CPU")
    props = {"Precision": "mixed"} if pname in ("CUDA", "OpenCL") else {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")}
    sim = app.Simulation(mod.topology, system, integ, openmm.Platform.getPlatformByName(pname), props)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=5000)
    baro.setFrequency(0)                                  # NVT heating
    sim.context.setParameter(openmm.MonteCarloBarostat.Pressure(), 1.0)
    for T in np.linspace(50, 300, 10):
        integ.setTemperature(T * unit.kelvin); sim.step(2500)          # 100 ps
    baro.setFrequency(25)
    sim.step(125000)                                      # 500 ps NPT, restrained
    sim.context.setParameter("k", 0.0)
    sim.step(int(EQUIL_NS * 1000 / 0.004))
    every = int(sample_ps / 0.004)
    n_frames = int(production_ns * 1000 / sample_ps)
    water_o = [a.index for a in mod.topology.atoms() if a.residue.name == "HOH" and a.element is not None and a.element.symbol == "O"]
    idx = np.array(solute + water_o)
    frames, boxes = [], []
    for f in range(n_frames):
        sim.step(every)
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
        X = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        frames.append(np.asarray(X[idx], dtype=np.float32))
        boxes.append(np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom), dtype=np.float32))
    return mod.topology, solute, water_o, np.stack(frames), np.stack(boxes)


RHO0_A3 = 0.0333                     # bulk water number density
KB_KCAL = 0.0019872043


def _wrap(w, centre, box):
    """Water positions imaged to the cell nearest the solute centre (triclinic box rows a, b, c)."""
    inv = np.linalg.inv(box.T)
    d = w - centre
    f = d @ inv.T
    return centre + (d - np.round(f) @ box)


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def analyse(top, solute, water_o, frames, boxes, cls: str, sample_ps: float = SAMPLE_PS) -> Dict[str, object]:
    from scipy.spatial import cKDTree
    atoms = list(top.atoms())
    sol = [atoms[i] for i in solute]
    ns = len(solute)
    chains = sorted({a.residue.chain.id for a in sol})
    res_of = {}
    for k, a in enumerate(sol):
        res_of.setdefault((a.residue.chain.id, int(a.residue.id)), {})[a.name] = k
    nA = max(r for c, r in res_of if c == chains[0])
    n_bp = nA
    base = lambda rn: rn.strip().lstrip("DR")[-1:] if rn else ""                        # noqa: E731
    resname = {(a.residue.chain.id, int(a.residue.id)): a.residue.name for a in sol}
    pair = lambda i: [(chains[0], i + 1), (chains[1], n_bp - i)]                        # noqa: E731
    def acc(i, groove):
        out = []
        for key in pair(i):
            b = base(resname[key]); names = (MINOR if groove == "minor" else MAJOR).get(b, ())
            out += [(key, res_of[key][nm]) for nm in names if nm in res_of[key]]
        return out
    minor = [acc(i, "minor") for i in range(n_bp)]; major = [acc(i, "major") for i in range(n_bp)]
    heavy = np.array([k for k, a in enumerate(sol) if a.element.symbol != "H"])
    core = np.array([k for k in heavy if 3 <= int(sol[k].residue.id) <= n_bp - 2])
    F = len(frames)
    ref = frames[0][:ns][core]
    occ = {g: np.zeros((F, n_bp)) for g in ("minor", "major")}
    shells = {}                                            # acceptor atom -> list of sets per frame
    allacc = sorted({k for L in minor + major for _, k in L})
    o2p = sorted(res_of[key]["O2'"] for key in res_of if "O2'" in res_of[key]) if cls != "dna" else []
    for k in allacc + o2p:
        shells[k] = []
    grid_lo = ref.min(0) - 8.0; vox = 0.5
    gshape = np.ceil((ref.max(0) + 8.0 - grid_lo) / vox).astype(int)
    counts = np.zeros(gshape, dtype=np.int32)
    aligned_solute = []
    for f in range(F):
        X = frames[f][:ns].astype(float); W = frames[f][ns:].astype(float)
        W = _wrap(W, X[heavy].mean(0), boxes[f].astype(float))
        tree = cKDTree(W)
        for k in shells:
            shells[k].append(set(tree.query_ball_point(X[k], SHELL_A)))
        for g, L in (("minor", minor), ("major", major)):
            for i in range(n_bp):
                occ[g][f, i] = sum(len(shells[k][-1]) for _, k in L[i])
        R, t = _kabsch(X[core], ref)
        Wa = W @ R.T + t
        aligned_solute.append(X @ R.T + t)
        ij = np.floor((Wa - grid_lo) / vox).astype(int)
        ok = np.all((ij >= 0) & (ij < gshape), axis=1)
        np.add.at(counts, tuple(ij[ok].T), 1)
    def residence(k):
        seen, runs = {}, []
        for f, s in enumerate(shells[k]):
            for w in s:
                seen.setdefault(w, []).append(f)
        for w, fr in seen.items():
            fr = np.array(fr); br = np.flatnonzero(np.diff(fr) > 1)
            starts = np.concatenate([[0], br + 1]); ends = np.concatenate([br, [len(fr) - 1]])
            runs += list(fr[ends] - fr[starts] + 1)
        return float(np.mean(runs) * sample_ps) if runs else float("nan")
    out: Dict[str, object] = {"n_bp": n_bp, "n_frames": F}
    for g, L in (("minor", minor), ("major", major)):
        out[f"{g}_occupancy"] = occ[g].mean(0)
        out[f"{g}_residence_ps"] = np.array([np.nanmean([residence(k) for _, k in L[i]]) if L[i] else np.nan for i in range(n_bp)])
    spine = np.full(n_bp, np.nan); bridge = np.full(n_bp, np.nan)
    for i in range(n_bp - 1):
        # the spine bridges minor-groove acceptors of adjacent pairs on opposite strands; both
        # one-pair offsets (Watson i with Crick i+1, Crick i with Watson i+1) are counted
        a1 = [k for key, k in minor[i] if key[0] == chains[0]]; b1 = [k for key, k in minor[i + 1] if key[0] == chains[1]]
        a2 = [k for key, k in minor[i] if key[0] == chains[1]]; b2 = [k for key, k in minor[i + 1] if key[0] == chains[0]]
        if (a1 and b1) or (a2 and b2):
            def bridged(f, a, b):
                return bool(a and b and (set().union(*[shells[x][f] for x in a]) & set().union(*[shells[y][f] for y in b])))
            spine[i] = float(np.mean([bridged(f, a1, b1) or bridged(f, a2, b2) for f in range(F)]))
        accs = minor[i] + minor[i + 1] + major[i] + major[i + 1]
        hits = []
        for f in range(F):
            owner = {}
            for key, k in accs:
                for w in shells[k][f]:
                    owner.setdefault(w, set()).add(key)
            hits.append(any(len(v) >= 2 for v in owner.values()))
        bridge[i] = float(np.mean(hits))
    out["spine_continuity"] = spine; out["water_bridge_p"] = bridge
    # first-order inhomogeneous-solvation translational entropy of the groove voxels, per pair
    g = counts / (F * RHO0_A3 * vox ** 3)
    centres = grid_lo + (np.indices(gshape).reshape(3, -1).T + 0.5) * vox
    gg = g.reshape(-1)
    sel = gg > 0
    mean_sol = np.mean(aligned_solute, axis=0)
    for gname, L in (("minor", minor), ("major", major)):
        pts, lab = [], []
        for i in range(n_bp):
            for _, k in L[i]:
                pts.append(mean_sol[k]); lab.append(i)
        tree = cKDTree(np.array(pts)); d, j = tree.query(centres[sel], distance_upper_bound=5.0)
        val = np.full(n_bp, 0.0); lab = np.array(lab)
        term = (gg[sel] * np.log(gg[sel])) * RHO0_A3 * vox ** 3
        for idx_ in range(len(d)):
            if np.isfinite(d[idx_]):
                val[lab[j[idx_]]] += term[idx_]
        out[f"{gname}_minusTS_kcal_mol"] = KB_KCAL * 300.0 * val        # -T S_trans (>= 0: ordering costs)
    if o2p:
        out["o2p_occupancy"] = np.full((len(chains), n_bp), np.nan)
        out["o2p_residence_ps"] = np.full((len(chains), n_bp), np.nan)
        out["o2p_o4p_hbond_p"] = np.full((len(chains), n_bp), np.nan)
        out["c3endo_frac"] = np.full((len(chains), n_bp), np.nan)
        out["o2p_rmsf_A"] = np.full((len(chains), n_bp), np.nan)
        S = np.array(aligned_solute)
        for ci, ch in enumerate(chains):
            for r in range(1, n_bp + 1):
                key = (ch, r)
                if "O2'" not in res_of.get(key, {}):
                    continue
                k = res_of[key]["O2'"]
                out["o2p_occupancy"][ci, r - 1] = float(np.mean([len(s) for s in shells[k]]))
                out["o2p_residence_ps"][ci, r - 1] = residence(k)
                nxt = res_of.get((ch, r + 1), {})
                if "O4'" in nxt:
                    out["o2p_o4p_hbond_p"][ci, r - 1] = float(np.mean(np.linalg.norm(S[:, k] - S[:, nxt["O4'"]], axis=1) < 3.2))
                nm = res_of[key]
                if all(x in nm for x in ("C1'", "C2'", "C3'", "C4'", "O4'")):
                    def tors(a, b, c, d):
                        b0, b1, b2 = a - b, c - b, d - c
                        b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
                        v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
                        w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
                        return np.degrees(np.arctan2(np.sum(np.cross(b1n, v) * w, -1), np.sum(v * w, -1)))
                    P = lambda n: S[:, nm[n]]                                       # noqa: E731
                    nu0 = tors(P("C4'"), P("O4'"), P("C1'"), P("C2'")); nu1 = tors(P("O4'"), P("C1'"), P("C2'"), P("C3'"))
                    nu2 = tors(P("C1'"), P("C2'"), P("C3'"), P("C4'")); nu3 = tors(P("C2'"), P("C3'"), P("C4'"), P("O4'"))
                    nu4 = tors(P("C3'"), P("C4'"), P("O4'"), P("C1'"))
                    ph = np.degrees(np.arctan2((nu4 + nu1 - nu3 - nu0), 2 * nu2 * (np.sin(np.radians(36)) + np.sin(np.radians(72))))) % 360
                    out["c3endo_frac"][ci, r - 1] = float(np.mean((ph < 36) | (ph > 342)))
                x = S[:, k]
                out["o2p_rmsf_A"][ci, r - 1] = float(np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, -1))))
    return out
