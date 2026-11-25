#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, intramolecular triplexes (H-DNA) as explicit geometry (PLAN.md, Family M).

For a fragment with an H-DNA candidate (Family F's homopurine/homopyrimidine mirror repeat), the
triplex core on the purine strand's arm -- its longest pure purine run of >= MIN_RUN -- is built
(`fold/triplex_build`: the cgNA+ duplex of the run between G-C caps, each third-strand nucleotide
from the medoid solved triplet T*A-T or C+*G-C in its purine's frame), the third-strand cytosines
N3-protonated (DCP: AMBER OL15 DC with RESP base charges and H3, `fold/dcp_forcefield`), and
minimised with the third strand bonded (AMBER OL15 + GBn2; duplex and third-strand base heavy
atoms restrained at the production restraint, the third-strand backbone free, so its O3'-P gaps
close). Solved with nonlinear PB (the production settings: exclusion radius, 0.35 A fine level,
map-based per-atom solvation) at 10, 150 and 500 mM NaCl. Channels: the potential 1 A outside the
molecular surface in the six base-pair-frame sectors of each core base pair (the third strand
fills the major groove, sector 0), their means over the core, the solvation free energy, and the
electrostatic free energy of third-strand binding, dG = dG_solv(triplex) - dG_solv(duplex) -
dG_solv(third strand) + their Coulomb interaction in the solute dielectric, on one grid.
Fragments without a candidate: M_tx_modelled = 0, the rest NaN (masked in training).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fold"))

FAMILY = "M"
MIN_RUN = 6
CONDITIONS = (("Na010", "NaCl_010mM"), ("Na150", "NaCl_150mM"), ("Na500", "NaCl_500mM"))
N_SECT = 6
SECTOR_SETS = (("s0", (0,)), ("s15", (1, 5)), ("s24", (2, 4)), ("s3", (3,)))
COULOMB_KJ_A = 1389.35457
SRC = ("PDB pyrimidine-motif triplets (data/triplex_templates_v1); cgNA+; AMBER OL15/GBn2 with "
       "RESP C+ charges; APBS nonlinear PB")


def channel_names_global() -> List[str]:
    out = ["M_tx_modelled", "M_tx_n_triplets", "M_tx_n_cplus", "M_tx_o3p_max"]
    for tag, _ in CONDITIONS:
        out += [f"M_tx_bind_elec_{tag}", f"M_tx_bind_elec_per_triplet_{tag}", f"M_tx_solvation_{tag}"]
        out += [f"M_tx_{k}_pot_d1_mean_{tag}" for k, _ in SECTOR_SETS]
    return out


def channel_names_per_bp() -> List[str]:
    out = ["M_tx_in_model"]
    for tag, _ in CONDITIONS:
        out += [f"M_tx_{k}_pot_d1_{tag}" for k, _ in SECTOR_SETS]
    return out


def _declare() -> None:
    reg = lambda **kw: REGISTRY.register(ChannelSpec(family=FAMILY, source=SRC, **kw))  # noqa: E731
    GL, PB, S = Resolution.GLOBAL, Resolution.PER_BP, Parity.SYMMETRIC
    reg(name="M_tx_modelled", resolution=GL, parity=S, units="indicator",
        description="1 if the fragment's H-DNA candidate was modelled as a triplex core")
    reg(name="M_tx_n_triplets", resolution=GL, parity=S, units="count", description="triplex core: triplets")
    reg(name="M_tx_n_cplus", resolution=GL, parity=S, units="count", description="triplex core: C+.G-C triplets")
    reg(name="M_tx_o3p_max", resolution=GL, parity=S, units="A",
        description="triplex core: largest third-strand O3'-P distance after minimisation (model check)")
    for tag, cname in CONDITIONS:
        reg(name=f"M_tx_bind_elec_{tag}", resolution=GL, parity=S, units="kJ/mol", condition_dependent=True,
            description=f"triplex core, {cname}: electrostatic free energy of third-strand binding")
        reg(name=f"M_tx_bind_elec_per_triplet_{tag}", resolution=GL, parity=S, units="kJ/mol",
            condition_dependent=True, description=f"triplex core, {cname}: the same per triplet")
        reg(name=f"M_tx_solvation_{tag}", resolution=GL, parity=S, units="kJ/mol", condition_dependent=True,
            description=f"triplex core, {cname}: electrostatic solvation free energy of the triplex")
        for k, secs in SECTOR_SETS:
            what = {"s0": "sector 0 (major groove, holding the third strand)", "s3": "sector 3 (minor groove)",
                    "s15": "sectors 1 and 5 (flanking the major groove), averaged",
                    "s24": "sectors 2 and 4 (backbone sides), averaged"}[k]
            reg(name=f"M_tx_{k}_pot_d1_mean_{tag}", resolution=GL, parity=S, units="kT/e",
                condition_dependent=True, description=f"triplex core, {cname}: potential 1 A out, {what}, core mean")
            reg(name=f"M_tx_{k}_pot_d1_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
                description=f"triplex core, {cname}: potential 1 A outside the surface, {what}")
    reg(name="M_tx_in_model", resolution=PB, parity=S, units="indicator",
        description="1 for base pairs in the modelled triplex core")


_declare()


def _empty(n):
    prof = {k: np.full(n, np.nan) for k in channel_names_per_bp()}
    glob = {k: float("nan") for k in channel_names_global()}
    prof["M_tx_in_model"] = np.zeros(n); glob["M_tx_modelled"] = 0.0
    return prof, glob


def select(seq: str) -> Optional[dict]:
    """The modelled core: (purine run, top-strand positions of its base pairs) or None."""
    import noncanonical as NC
    import triplex_build as TB
    s = seq.upper(); n = len(s)
    cands = NC.triplex_candidates(s)
    if not cands:
        return None
    best = max(cands, key=lambda d: (d["arm"], -d["spacer"]))
    core, a, b = TB.purine_tract(s, int(best["centre"]), int(best["arm"]), int(best["spacer"]))
    runs = [(m.start(), m.end()) for m in re.finditer(r"[AG]+", core)]
    if not runs:
        return None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    if hi - lo < MIN_RUN:
        return None
    c0, arm, sp = int(best["centre"]), int(best["arm"]), int(best["spacer"])
    tract = s[c0 - arm:c0 + sp + arm]
    on_top = sum(x in "AG" for x in tract) / max(1, len(tract)) >= 0.5    # as purine_tract decides
    pos = [a + lo + i for i in range(hi - lo)] if on_top else [n - 1 - (a + lo + i) for i in range(hi - lo)]
    return {"purine": core[lo:hi], "positions": pos}


def _protonate_and_minimise(atoms, work: str) -> Tuple[str, dict]:
    import cgna_atomic as ca
    import dcp_forcefield as DF
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    raw = os.path.join(work, "raw.pdb"); ca.write_pdb(atoms, raw)
    fx = PDBFixer(filename=raw); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    # H3 on every third-strand cytosine (chain C), along the external bisector of C2-N3-C4: written
    # into the residue's own block of a PDB (OpenMM keeps a residue's atoms contiguous), then bonded
    fixed = os.path.join(work, "fixed.pdb")
    with open(fixed, "w") as fh:
        app.PDBFile.writeFile(fx.topology, fx.positions, fh, keepIds=True)
    lines = open(fixed).read().splitlines()
    out_lines, block = [], []

    def flush(block):
        if block and block[0][21] == "C" and block[0][17:20].strip().startswith("DC"):
            X = {l[12:16].strip(): np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) for l in block}
            u = (X["N3"] - X["C2"]) / np.linalg.norm(X["N3"] - X["C2"]) + (X["N3"] - X["C4"]) / np.linalg.norm(X["N3"] - X["C4"])
            h = X["N3"] + 1.01 * u / np.linalg.norm(u)
            l0 = block[-1]
            block = block + [f"ATOM  {99999:5d}  H3  {l0[17:20]} {l0[21]}{l0[22:26]}    {h[0]:8.3f}{h[1]:8.3f}{h[2]:8.3f}  1.00  0.00           H"]
        return block
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            if block and (l[21], l[22:27]) != (block[0][21], block[0][22:27]):
                out_lines += flush(block); block = []
            block.append(l)
        else:
            if block:
                out_lines += flush(block); block = []
            if not l.startswith("CONECT"):
                out_lines.append(l)
    out_lines += flush(block)
    prot = os.path.join(work, "prot.pdb")
    open(prot, "w").write("\n".join(out_lines) + "\n")
    pdb = app.PDBFile(prot)
    third_c = [r for r in pdb.topology.residues() if r.chain.id == "C" and r.name.startswith("DC")]
    for r in third_c:
        at = {a.name: a for a in r.atoms()}
        pdb.topology.addBond(at["N3"], at["H3"])
    mod = app.Modeller(pdb.topology, pdb.positions)
    xml = DF.write_xml(os.path.join(work, "dcp.xml"))
    ff = app.ForceField("amber14/DNA.OL15.xml", xml, "implicit/gbn2.xml")
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", 500.0)
    for p_ in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p_)
    bb = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
    xyz = np.array(mod.positions.value_in_unit(unit.nanometer))
    for a in mod.topology.atoms():
        if a.element.symbol == "H":
            continue
        if a.residue.chain.id in ("A", "B") or a.name not in bb:
            rest.addParticle(a.index, xyz[a.index])
    system.addForce(rest)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"),
                         {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")})
    ctx.setPositions(mod.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 20000)
    st = ctx.getState(getPositions=True, getEnergy=True)
    mn = os.path.join(work, "min.pdb")
    with open(mn, "w") as fh:
        app.PDBFile.writeFile(mod.topology, st.getPositions(), fh, keepIds=True)
    out = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    res_c = [r for r in mod.topology.residues() if r.chain.id == "C"]
    gaps = []
    for r1, r2 in zip(res_c[:-1], res_c[1:]):
        o3 = next(a.index for a in r1.atoms() if a.name == "O3'"); p = next(a.index for a in r2.atoms() if a.name == "P")
        gaps.append(float(np.linalg.norm(out[o3] - out[p])))
    return mn, {"o3p_max": max(gaps) if gaps else float("nan"), "n_cplus": len(third_c), "n_third": len(res_c),
                "energy_kJ_mol": st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)}


def _pqr(min_pdb: str, work: str, chains: str) -> str:
    """PQR of the given chains: pdb2pqr (AMBER) with C+ written as DC, then the C+ base charges
    and H3 (radius of thymine's H3) put back."""
    import json
    import apbs_run as A
    import dcp_forcefield as DF
    q = json.load(open(DF.CHARGES))["base_charges"]
    lines = [l for l in open(min_pdb) if l.startswith(("ATOM", "HETATM")) and l[21] in chains]
    h3 = {}
    keep = []
    for l in lines:
        if l[21] == "C" and l[17:20].strip().startswith("DC") and l[12:16].strip() == "H3":
            h3[(l[21], int(l[22:26]))] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            continue
        keep.append(l)
    src = os.path.join(work, f"in_{chains}.pdb")
    open(src, "w").write("".join(keep) + "END\n")
    out = os.path.join(work, f"tx_{chains}.pqr")
    A.run_pdb2pqr(src, out); A.sanitise_pqr(out)
    xyz, qq, rr, recs = A.read_pqr(out)
    rad_h3 = next((float(r_) for r_, rec in zip(rr, recs) if rec.get("resname", "").startswith("DT")
                   and rec.get("name") == "H3"), 0.6)
    lines_out, serial = [], 1
    for x, q0, r_, rec in zip(xyz, qq, rr, recs):
        key = (rec.get("chain", ""), int(rec.get("resseq", -1)))
        name = rec.get("name", "")
        qv = q[name] if (key in h3 and name in q) else float(q0)
        lines_out.append(f"ATOM {serial:6d} {name:<4s} {rec.get('resname', 'UNK'):>4s} {key[0]} {key[1]:4d} "
                         f"{x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {qv:8.4f} {float(r_):7.4f}\n"); serial += 1
    for (ch, rs), x in h3.items():
        lines_out.append(f"ATOM {serial:6d} H3   DC   {ch} {rs:4d} {x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {q['H3']:8.4f} {rad_h3:7.4f}\n")
        serial += 1
    res = lines_out
    open(out, "w").write("".join(res))
    return out


def _solve(pqr: str, name: str, work: str, cond, centre, coarse, fine, maps=True):
    """Solvated + reference on one grid: (potential map or None, per-atom solvation kJ/mol)."""
    import apbs_run as A
    xyz, q, rad, _ = A.read_pqr(pqr)
    w = os.path.join(work, name); os.makedirs(w, exist_ok=True)
    shutil.copy(pqr, os.path.join(w, "m.pqr"))
    ins = A.write_apbs_input("m.pqr", "solv", w, cond, centre, coarse, fine, exact_fine_spacing=True, calc_energy=False)
    A.run_apbs(ins, w)
    ref = A.write_apbs_input("m.pqr", "ref", w, cond, centre, coarse, fine, exact_fine_spacing=True,
                             reference=True, reference_maps=True, calc_energy=False)
    A.run_apbs(ref, w)
    ps = A.DXMap.read(os.path.join(w, "solv_pot.dx")); pr = A.DXMap.read(os.path.join(w, "ref_pot.dx"))
    kT = 8.314462618e-3 * cond.temperature_K
    solv = 0.5 * q * (ps.interpolate(xyz) - pr.interpolate(xyz)) * kT
    return (ps if maps else None), solv


def triplex_channels(seq: str, workroot: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    import apbs_run as A
    import frames as F
    import triplex_build as TB
    n = len(seq)
    prof, glob = _empty(n)
    sel = select(seq)
    if sel is None:
        return prof, glob
    work = tempfile.mkdtemp(prefix="tx_", dir=workroot or os.environ.get("SLURM_TMPDIR") or None)
    try:
        atoms = TB.build_core(sel["purine"])
        mn, rep = _protonate_and_minimise(atoms, work)
        # base-pair frames of the duplex (chains A, B) of the minimised core
        dup = os.path.join(work, "dup.pdb")
        open(dup, "w").write("".join(l for l in open(mn) if l.startswith(("ATOM", "HETATM")) and l[21] in "AB") + "END\n")
        bp, _ = F.bp_frames_from_pdb(dup)
        ncap = len(TB.CAP)
        core_bp = list(range(ncap, ncap + len(sel["purine"])))
        p_all, p_dup, p_third = _pqr(mn, work, "ABC"), _pqr(mn, work, "AB"), _pqr(mn, work, "C")
        xyz, q, rad, _ = A.read_pqr(p_all)
        xd, qd, _, _ = A.read_pqr(p_dup); xt, qt, _, _ = A.read_pqr(p_third)
        d = np.linalg.norm(xd[:, None, :] - xt[None, :, :], axis=-1)
        e_inter = float(COULOMB_KJ_A / 2.0 * np.sum(qd[:, None] * qt[None, :] / np.maximum(d, 1e-6)))
        pts, _, _ = A.surface_referenced_points(bp[core_bp], xyz, rad, offsets_A=(1.0,))
        centre = xyz.mean(0); ext = xyz.max(0) - xyz.min(0)
        glob.update(M_tx_modelled=1.0, M_tx_n_triplets=float(rep["n_third"]), M_tx_n_cplus=float(rep["n_cplus"]),
                    M_tx_o3p_max=float(rep["o3p_max"]))
        for tag, cname in CONDITIONS:
            cond = A.PRODUCTION_CONDITIONS_BY_NAME[cname]
            lam = A.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
            coarse = ext + 2.0 * max(50.0, 3.0 * lam)
            fine = np.minimum(ext + 24.0, coarse)
            pot, s_all = _solve(p_all, f"all_{tag}", work, cond, centre, coarse, fine)
            _, s_dup = _solve(p_dup, f"dup_{tag}", work, cond, centre, coarse, fine, maps=False)
            _, s_thr = _solve(p_third, f"thr_{tag}", work, cond, centre, coarse, fine, maps=False)
            dG = float(s_all.sum() - s_dup.sum() - s_thr.sum() + e_inter)
            glob[f"M_tx_bind_elec_{tag}"] = dG
            glob[f"M_tx_bind_elec_per_triplet_{tag}"] = dG / max(1.0, rep["n_third"])
            glob[f"M_tx_solvation_{tag}"] = float(s_all.sum())
            flat = pts.reshape(-1, 3); ok = np.isfinite(flat).all(1)
            vals = np.full(len(flat), np.nan); vals[ok] = pot.interpolate(flat[ok])
            vals = vals.reshape(len(core_bp), N_SECT)
            for k, secs in SECTOR_SETS:
                v = np.nanmean(vals[:, list(secs)], axis=1)
                tr = np.full(n, np.nan)
                for i, p in enumerate(sel["positions"]):
                    tr[p] = v[i]
                prof[f"M_tx_{k}_pot_d1_{tag}"] = tr
                glob[f"M_tx_{k}_pot_d1_mean_{tag}"] = float(np.nanmean(v))
        inm = np.zeros(n)
        for p in sel["positions"]:
            inm[p] = 1.0
        prof["M_tx_in_model"] = inm
        return prof, glob
    finally:
        shutil.rmtree(work, ignore_errors=True)
