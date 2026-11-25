#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family I, "generic amino acid side chain interaction energies in each groove" (PLAN.md I; family
code "Isc", held apart like Family I). For every RC-unique tetramer context: the 12-mer GCGC +
context + GCGC (cgNA+ ground state, minimised as production) and capped dipeptides ACE-X-NME for the
DNA-reading residues X = ARG, LYS, GLN, ASN, SER, THR, HIS (AMBER ff14SB, the DNA in OL15; GBn2 at
0.15 M), the DNA held fixed.

Search (v2, Monte Carlo minimisation, Li & Scheraga 1987): in each groove of the central step,
N_CHAINS chains, each started from a random backbone/side-chain conformer placed with its side-chain
tip near a random groove acceptor of the central pairs, then N_STEPS moves -- a rigid-body move of
the peptide about its tip, or a jump of one rotatable torsion (phi, psi, chi_n) turning either side
of the bond -- each followed by minimisation and a Metropolis test on the complex energy at T_MC. A
flat-bottom restraint keeps the tip within TIP_R of the nearest groove acceptor of the central pairs;
poses that end outside it are not counted (the channel is a groove interaction). Every minimum found
is kept in the output, so convergence is checked from the record itself. The v1 search (independent
minimisations from one rigid conformer) did not converge: three seeds at 60 starts differed by up to
10.8 kcal/mol and 240 starts were not lower than 60 in 4 of 14 channels (DECISIONS.md, 2026-11-23).

Channels per groove and residue (kcal/mol): `best`, the interaction energy E(complex) - E(DNA) -
E(peptide in the same conformation) of the lowest-energy complex found; `boltz`, the interaction
energy averaged over the distinct minima found with Boltzmann weights of their complex energies at
300 K.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESIDUES = ("ARG", "LYS", "GLN", "ASN", "SER", "THR", "HIS")
TIP = {"ARG": "CZ", "LYS": "NZ", "GLN": "CD", "ASN": "CG", "SER": "OG", "THR": "OG1", "HIS": "NE2"}
TIP_A = 3.5                                  # start: tip 2.8 to 2.8 + TIP_A A out from the acceptor
TIP_R, TIP_K = 6.0, 10.0                     # flat-bottom restraint: radius (A), force constant (kcal/mol/A^2)
N_CHAINS, N_STEPS, T_MC = 10, 40, 1.0        # chains, moves per chain, Metropolis temperature (kcal/mol)
KT300 = 0.596                                # kcal/mol
IN_GROOVE = 0.05                             # kcal/mol of restraint energy above which a pose is outside
TORSIONS = {"ARG": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")),
            "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
            "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
            "ASN": (("CA", "CB"), ("CB", "CG")),
            "SER": (("CA", "CB"), ("CB", "OG")),
            "THR": (("CA", "CB"), ("CB", "OG1")),
            "HIS": (("CA", "CB"), ("CB", "CG"))}
BACKBONE = (("N", "CA"), ("CA", "C"))         # phi, psi of the central residue
KCAL = 4.184
PAD = "GCGC"
MINOR_ACC = {"A": ("N3",), "G": ("N3",), "C": ("O2",), "T": ("O2",)}
MAJOR_ACC = {"A": ("N7",), "G": ("N7", "O6"), "C": ("N4",), "T": ("O4",)}


def _dipeptide(res: str):
    """ACE-X-NME from OpenMM's ff14SB residue templates via PDBFixer (an extended backbone)."""
    from pdbfixer import PDBFixer
    from openmm import app
    import io
    # a minimal capped residue in PDB text: PDBFixer builds every missing atom from templates
    pdb = (f"ATOM      1  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C\n"
           f"ATOM      2  N   {res} A   2       1.335   0.000   0.000  1.00  0.00           N\n"
           f"ATOM      3  CA  {res} A   2       2.055   1.247   0.000  1.00  0.00           C\n"
           f"ATOM      4  C   {res} A   2       3.555   1.000   0.000  1.00  0.00           C\n"
           f"ATOM      5  N   NME A   3       4.150   2.150   0.000  1.00  0.00           N\nEND\n")
    fx = PDBFixer(pdbfile=io.StringIO(pdb))
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    return fx.topology, fx.positions


def _graph(ptop):
    """Adjacency (by peptide atom index) of the dipeptide's bonds."""
    n = ptop.getNumAtoms(); adj = [set() for _ in range(n)]
    for b in ptop.bonds():
        adj[b[0].index].add(b[1].index); adj[b[1].index].add(b[0].index)
    return adj


def _side(adj, a: int, b: int) -> Optional[np.ndarray]:
    """Atoms reached from b without crossing the bond a-b (None if a-b is in a ring)."""
    seen, todo = {b}, [b]
    while todo:
        u = todo.pop()
        for v in adj[u]:
            if u == b and v == a:
                continue
            if v == a:
                return None
            if v not in seen:
                seen.add(v); todo.append(v)
    return np.array(sorted(seen))


def _rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _turn(X, a, b, moving, ang):
    """X with the atoms `moving` turned by ang about the a->b bond axis."""
    Y = X.copy(); R = _rot(X[b] - X[a], ang)
    Y[moving] = (X[moving] - X[a]) @ R.T + X[a]
    return Y


def _torsion_moves(ptop, res):
    """Per rotatable bond of the central residue (phi, psi, chi_n): (a, b, b-side atoms, a-side atoms)."""
    adj = _graph(ptop)
    mid = [r for r in ptop.residues()][1]
    idx = {a.name: a.index for a in mid.atoms()}
    out = []
    for x, y in BACKBONE + TORSIONS[res]:
        a, b = idx[x], idx[y]
        sb, sa = _side(adj, a, b), _side(adj, b, a)
        if sb is not None and sa is not None:
            out.append((a, b, sb, sa))
    return out, adj


def _topo_far(adj, heavy):
    """Heavy-atom pairs more than three bonds apart (the pairs checked for internal clashes)."""
    n = len(adj); far = []
    for i in heavy:
        dist = {i: 0}; todo = [i]
        while todo:
            u = todo.pop(0)
            if dist[u] >= 3:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; todo.append(v)
        far += [(i, j) for j in heavy if j > i and j not in dist]
    return np.array(far, dtype=int).reshape(-1, 2)


def _internal_clash(X, far, lim=2.0) -> bool:
    return bool(len(far)) and bool(np.min(np.linalg.norm(X[far[:, 0]] - X[far[:, 1]], axis=1)) < lim)


def _flat_bottom(tip_global: int, acceptors_A: np.ndarray):
    """Flat-bottom restraint on the tip: zero within TIP_R of the nearest acceptor."""
    import openmm
    ds = [f"sqrt((x-{p[0] / 10:.5f})^2+(y-{p[1] / 10:.5f})^2+(z-{p[2] / 10:.5f})^2)" for p in acceptors_A]
    dmin = ds[0]
    for d_ in ds[1:]:
        dmin = f"min({dmin},{d_})"
    f = openmm.CustomExternalForce(f"0.5*kfb*max(0,{dmin}-rfb)^2")
    f.addGlobalParameter("kfb", TIP_K * KCAL * 100.0)          # kcal/mol/A^2 -> kJ/mol/nm^2
    f.addGlobalParameter("rfb", TIP_R / 10.0)
    f.addParticle(tip_global, [])
    f.setForceGroup(2)
    return f


def _context_code(context: str) -> int:
    return int("".join("ACGT".index(c).__str__() for c in context), 4)


def prepare_dna(context: str, work: str, cache: Optional[str] = None) -> str:
    """The minimised 12-mer of a context (shared by every residue's task through `cache`)."""
    import cgna_atomic as ca
    import minimize as MZ
    if cache and os.path.exists(cache):
        return cache
    seq = PAD + context + PAD[::-1]
    raw = os.path.join(work, "dna_raw.pdb"); ca.write_pdb(ca.build_duplex(seq), raw)
    dna = os.path.join(work, "dna.pdb"); MZ.restrained_minimise(raw, dna, max_iterations=20000, tolerance=1.0)
    if cache:
        tmp = f"{cache}.{os.getpid()}.tmp"; shutil.copy(dna, tmp); os.replace(tmp, cache)
        return cache
    return dna


def probe_context(context: str, work: str, seed: int = 0, n_chains: int = None, n_steps: int = None,
                  residues=None, dna_cache: Optional[str] = None) -> Dict[str, object]:
    n_chains = int(n_chains or N_CHAINS); n_steps = int(n_steps or N_STEPS)
    import openmm
    from openmm import app, unit
    from scipy.spatial import cKDTree
    seq = PAD + context + PAD[::-1]; n = len(seq); k = len(PAD) + 1
    dna = prepare_dna(context, work, dna_cache)
    d = app.PDBFile(dna)
    dpos = np.array(d.positions.value_in_unit(unit.angstrom))
    tree = cKDTree(dpos)
    res_of = {(r.chain.id, int(r.id)): r for r in d.topology.residues()}
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}

    def acc(table):
        pts = []
        for p in (k, k + 1):                        # 0-based pairs of the central step
            for key, b in ((("A", p + 1), seq[p]), (("B", n - p), comp[seq[p]])):
                for at in res_of[key].atoms():
                    if at.name in table[b]:
                        pts.append(dpos[at.index])
        return np.array(pts)
    sites = {"minor": acc(MINOR_ACC), "major": acc(MAJOR_ACC)}
    import frames as F
    bp, _ = F.bp_frames_from_pdb(dna)
    axis_o = 0.5 * (bp[k][:3, 3] + bp[k + 1][:3, 3]); axis_z = bp[k][:3, 2] + bp[k + 1][:3, 2]; axis_z /= np.linalg.norm(axis_z)
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    kappa = 50.33355 * (0.15 / (78.5 * 300.0)) ** 0.5 * 10.0 / unit.nanometer
    out: Dict[str, object] = {}
    sysD = ff.createSystem(d.topology, nonbondedMethod=app.NoCutoff, implicitSolventKappa=kappa)
    cD = openmm.Context(sysD, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    cD.setPositions(d.positions)
    eD = cD.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    out["E_dna_kJ"] = eD
    nd = d.topology.getNumAtoms()
    for res in (residues or RESIDUES):
        ri = RESIDUES.index(res)
        rng = np.random.default_rng([seed, _context_code(context), ri])
        ptop, ppos = _dipeptide(res)
        P0 = np.array(ppos.value_in_unit(unit.angstrom))
        tip_i = next(a.index for a in ptop.atoms() if a.name == TIP[res] and a.residue.index == 1)
        ca_i = next(a.index for a in ptop.atoms() if a.name == "CA" and a.residue.index == 1)
        moves, adj = _torsion_moves(ptop, res)
        heavy = [a.index for a in ptop.atoms() if a.element is not None and a.element.symbol != "H"]
        far = _topo_far(adj, heavy)
        sysP = ff.createSystem(ptop, nonbondedMethod=app.NoCutoff, implicitSolventKappa=kappa)
        cP = openmm.Context(sysP, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
        mod = app.Modeller(d.topology, d.positions); mod.add(ptop, ppos)
        for groove, S in sites.items():
            key = f"{groove}_{res}"
            if len(S) == 0:
                out[f"{key}_best"] = out[f"{key}_boltz"] = float("nan"); continue
            sysC = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, implicitSolventKappa=kappa)
            for i in range(nd):
                sysC.setParticleMass(i, 0.0)             # DNA fixed
            rest = openmm.CustomExternalForce("0.5*1e6*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            for p_ in ("x0", "y0", "z0"):
                rest.addPerParticleParameter(p_)
            for i in range(nd):
                rest.addParticle(i, dpos[i] / 10.0)
            rest.setForceGroup(1); sysC.addForce(rest)
            sysC.addForce(_flat_bottom(nd + tip_i, S))
            cC = openmm.Context(sysC, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})

            def minimise(X):
                cC.setPositions(np.vstack([dpos, X]) / 10.0 * unit.nanometer)
                openmm.LocalEnergyMinimizer.minimize(cC, 5.0, 2000)
                st = cC.getState(getPositions=True)
                Xm = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[nd:]
                e0 = cC.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                ef = cC.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                cP.setPositions(Xm * unit.angstrom)
                dtip = float(np.min(np.linalg.norm(S - Xm[tip_i], axis=1)))
                rr = Xm[tip_i] - axis_o; rcyl = float(np.linalg.norm(rr - (rr @ axis_z) * axis_z))
                eP = cP.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                # complex energy relative to the fixed DNA (peptide internal + interaction + restraint)
                return Xm, (e0 + ef - eD) / KCAL, (e0 - eD - eP) / KCAL, ef / KCAL, dtip, rcyl

            def dna_clash(X, lim):
                return bool(np.min(tree.query(X)[0]) < lim)

            def start():
                for _ in range(500):
                    X = P0.copy()
                    for a, b, sb, _sa in moves:              # random conformer: each torsion by 0/120/240 + jitter
                        X = _turn(X, a, b, sb, np.deg2rad(rng.choice([0.0, 120.0, 240.0]) + rng.normal(0, 15)))
                    if _internal_clash(X, far, 2.5):
                        continue
                    acc_pt = S[rng.integers(len(S))]
                    rel = acc_pt - axis_o; u = rel - (rel @ axis_z) * axis_z; u /= np.linalg.norm(u)
                    u = u + 0.35 * rng.normal(size=3); u /= np.linalg.norm(u)
                    v0 = X[ca_i] - X[tip_i]; v0 /= np.linalg.norm(v0)
                    c = np.cross(v0, u); sn, cs = np.linalg.norm(c), float(v0 @ u)
                    Kx = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
                    R1 = np.eye(3) + Kx + Kx @ Kx * ((1 - cs) / (sn ** 2 + 1e-12))
                    R2 = _rot(u, rng.uniform(0, 2 * np.pi))
                    Y = (X - X[tip_i]) @ (R2 @ R1).T + acc_pt + u * rng.uniform(2.8, 2.8 + TIP_A)
                    if not dna_clash(Y, 1.6):
                        return Y
                raise RuntimeError(f"{context} {key}: no clash-free start in 500 tries")

            minima, chain_best, n_acc, n_mov, n_rej, n_out = [], [], 0, 0, 0, 0
            for ch in range(n_chains):
                X, Ec, Ei, Ef, dt, rc = minimise(start())
                cb = Ec if Ef < IN_GROOVE else np.inf
                if Ef < IN_GROOVE:
                    minima.append((ch, 0, Ec, Ei, dt, rc))
                else:
                    n_out += 1
                for stp in range(1, n_steps + 1):
                    if rng.random() < 0.4:                     # rigid-body move about the tip
                        ax = rng.normal(size=3)
                        R = _rot(ax, np.deg2rad(rng.uniform(0, 30)))
                        Y = (X - X[tip_i]) @ R.T + X[tip_i] + rng.normal(0, 0.5, size=3)
                    else:                                      # torsion jump, either side of the bond
                        a, b, sb, sa = moves[rng.integers(len(moves))]
                        ang = np.deg2rad(rng.choice([60.0, 120.0, 180.0, 240.0, 300.0]) + rng.normal(0, 10))
                        Y = _turn(X, a, b, sb, ang) if rng.random() < 0.5 else _turn(X, b, a, sa, ang)
                    n_mov += 1
                    if dna_clash(Y, 1.2) or _internal_clash(Y, far, 2.0):
                        n_rej += 1; continue
                    Ym, Ec2, Ei2, Ef2, dt2, rc2 = minimise(Y)
                    if Ef2 < IN_GROOVE:
                        minima.append((ch, stp, Ec2, Ei2, dt2, rc2)); cb = min(cb, Ec2)
                    else:
                        n_out += 1
                    if Ec2 <= Ec or rng.random() < np.exp(-(Ec2 - Ec) / T_MC):
                        X, Ec, Ei, Ef = Ym, Ec2, Ei2, Ef2; n_acc += 1
                chain_best.append(float(cb))
            if minima:
                M = np.array([(m[2], m[3]) for m in minima])
                ib = int(np.argmin(M[:, 0]))
                _, uniq = np.unique(np.round(M[:, 0], 2), return_index=True)      # distinct minima
                U = M[uniq]; w = np.exp(-(U[:, 0] - U[:, 0].min()) / KT300)
                out[f"{key}_best"] = float(M[ib, 1]); out[f"{key}_boltz"] = float(np.sum(w * U[:, 1]) / np.sum(w))
                out[f"{key}_Ecplx_best"] = float(M[ib, 0])
            else:
                out[f"{key}_best"] = out[f"{key}_boltz"] = float("nan")
            out[f"{key}_search"] = {"chain_best_Ecplx": chain_best, "moves": n_mov, "rejected_clash": n_rej,
                                    "accepted": n_acc, "outside_groove": n_out,
                                    "minima_fields": ["chain", "step", "E_cplx", "E_int", "tip_to_acceptor_A", "tip_radius_A"],
                                    "minima": [[int(a), int(b), round(c, 4), round(e, 4), round(dd, 2), round(rr_, 2)]
                                               for a, b, c, e, dd, rr_ in minima]}
    return out


def main() -> int:
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "qc"))
    import build_qc_fragments as QF
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True); ap.add_argument("--contexts", nargs="*", default=None)
    ap.add_argument("--residues", nargs="*", default=None, choices=RESIDUES)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-chains", type=int, default=N_CHAINS); ap.add_argument("--n-steps", type=int, default=N_STEPS)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "dna"), exist_ok=True)
    items = [(c, r) for c in (a.contexts or QF.unique_contexts(4)) for r in (a.residues or RESIDUES)]
    failed = 0
    for ctx, res in items[a.shard::a.n_shards]:
        f = os.path.join(a.out, f"{ctx}__{res}.json")
        if os.path.exists(f):
            continue
        w = tempfile.mkdtemp()
        try:
            t = time.time()
            r = probe_context(ctx, w, seed=a.seed, n_chains=a.n_chains, n_steps=a.n_steps, residues=[res],
                              dna_cache=os.path.join(a.out, "dna", f"{ctx}.pdb"))
            r.update(context=ctx, residue=res, seed=a.seed, n_chains=a.n_chains, n_steps=a.n_steps, search="mcm-v2",
                     seconds=round(time.time() - t, 1))
            json.dump(r, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
            print(ctx, res, {k: round(v, 2) for k, v in r.items() if k.endswith(("_best", "_boltz"))},
                  f"{r['seconds']} s", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            failed += 1
            print(ctx, res, "failed:", type(exc).__name__, str(exc)[:300], flush=True)
        finally:
            shutil.rmtree(w, ignore_errors=True)
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
