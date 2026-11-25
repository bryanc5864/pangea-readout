#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Tiers 2 and 3 of the quantum chemistry (PLAN.md): dispersion-corrected DFT and SAPT.

PLAN.md: "Dispersion-corrected density functional theory across all tetramer contexts at
three stacking geometries, giving total interaction energies, frontier orbitals, ionization
potentials, electron affinities ... Symmetry-adapted perturbation theory on a stratified
subset, giving the decomposition into electrostatic, exchange-repulsion, induction and
dispersion terms. The expensive tier supervises a correction to the cheap tier."

Fragments are the tier-1 library's (`build_qc_fragments.py`: capped bases in the cgNA+
ground-state geometry of the padded context), so every tier computes the same objects:
stacking = E(bp1 + bp2) - E(bp1) - E(bp2); hydrogen bond = E(w + c) - E(w) - E(c); both
counterpoise-corrected (Boys-Bernardi) at the DFT tier. SAPT0 decomposes the stacking
interaction into electrostatics, exchange, induction and dispersion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HARTREE_KCAL = 627.509474
HARTREE_EV = 27.211386


def read_xyz(path: str) -> List[str]:
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    return [l for l in lines[2:2 + n] if l.strip()]


def molecule(psi4, frags: List[List[str]], charge: int = 0, mult: int = 1):
    """psi4 molecule from fragments (each a list of xyz lines); neutral singlet fragments."""
    blocks = []
    for k, atoms in enumerate(frags):
        head = f"{charge if k == 0 and len(frags) == 1 else 0} {mult if len(frags) == 1 else 1}"
        blocks.append(head + "\n" + "\n".join(atoms))
    text = "\n--\n".join(blocks) + "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1\n"
    return psi4.geometry(text)


def dft_context(psi4, files: Dict[str, str], method: str, basis: str, ipea: bool) -> Dict[str, float]:
    out: Dict[str, float] = {}
    bp1, bp2 = read_xyz(files["bp1"]), read_xyz(files["bp2"])
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": False})
    t = time.time()
    e = psi4.energy(method, molecule=molecule(psi4, [bp1, bp2]), bsse_type="cp")
    out["stacking_kcal_mol"] = e * HARTREE_KCAL
    out["t_stacking_s"] = round(time.time() - t, 1)
    for tag, w, c in (("bp1", "w1", "c1"), ("bp2", "w2", "c2")):
        e = psi4.energy(method, molecule=molecule(psi4, [read_xyz(files[w]), read_xyz(files[c])]), bsse_type="cp")
        out[f"hbond_{tag}_kcal_mol"] = e * HARTREE_KCAL
    t = time.time()
    e0, wfn = psi4.energy(method, molecule=molecule(psi4, [bp1 + bp2]), return_wfn=True)
    eps = wfn.epsilon_a()
    eps = eps.to_array() if hasattr(eps, "to_array") else eps
    nocc = wfn.nalpha()
    out["dimer_homo_eV"] = float(eps[nocc - 1]) * HARTREE_EV
    out["dimer_lumo_eV"] = float(eps[nocc]) * HARTREE_EV
    out["dimer_gap_eV"] = out["dimer_lumo_eV"] - out["dimer_homo_eV"]
    dip = wfn.variable("SCF DIPOLE")                  # psi4 >= 1.6 returns a numpy array (a.u.)
    dip = dip.to_array() if hasattr(dip, "to_array") else dip
    out["dimer_dipole_debye"] = float(sum(float(x) ** 2 for x in dip) ** 0.5) * 2.541746
    if ipea:
        psi4.set_options({"reference": "uks"})
        ec = psi4.energy(method, molecule=molecule(psi4, [bp1 + bp2], charge=+1, mult=2))
        ea = psi4.energy(method, molecule=molecule(psi4, [bp1 + bp2], charge=-1, mult=2))
        psi4.set_options({"reference": "rks"})
        out["dimer_vip_eV"] = (ec - e0) * HARTREE_EV
        out["dimer_vea_eV"] = (e0 - ea) * HARTREE_EV
    out["t_total_s"] = round(time.time() - t + out["t_stacking_s"], 1)
    return out


def sapt_context(psi4, files: Dict[str, str], basis: str = "jun-cc-pvdz") -> Dict[str, float]:
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": True})
    t = time.time()
    psi4.energy("sapt0", molecule=molecule(psi4, [read_xyz(files["bp1"]), read_xyz(files["bp2"])]))
    g = lambda k: psi4.variable(k) * HARTREE_KCAL
    return {"sapt0_total_kcal_mol": g("SAPT0 TOTAL ENERGY"), "sapt0_elst_kcal_mol": g("SAPT ELST ENERGY"),
            "sapt0_exch_kcal_mol": g("SAPT EXCH ENERGY"), "sapt0_ind_kcal_mol": g("SAPT IND ENERGY"),
            "sapt0_disp_kcal_mol": g("SAPT DISP ENERGY"), "t_sapt_s": round(time.time() - t, 1)}


def sapt_hbond(psi4, files: Dict[str, str], basis: str = "jun-cc-pvdz") -> Dict[str, float]:
    """SAPT0 decomposition of each base pair's hydrogen-bonding interaction (Watson base
    against Crick base), the plan's "hydrogen-bond energy decomposed the same way"."""
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": True})
    out: Dict[str, float] = {}
    t = time.time()
    g = lambda k: psi4.variable(k) * HARTREE_KCAL                 # noqa: E731
    for tag, w, c in (("bp1", "w1", "c1"), ("bp2", "w2", "c2")):
        psi4.energy("sapt0", molecule=molecule(psi4, [read_xyz(files[w]), read_xyz(files[c])]))
        for k, v in (("total", "SAPT0 TOTAL ENERGY"), ("elst", "SAPT ELST ENERGY"),
                     ("exch", "SAPT EXCH ENERGY"), ("ind", "SAPT IND ENERGY"),
                     ("disp", "SAPT DISP ENERGY")):
            out[f"sapt0_hb_{tag}_{k}_kcal_mol"] = g(v)
    out["t_sapt_hb_s"] = round(time.time() - t, 1)
    return out


GROOVE_SITES = {   # (groove, label, atom) per base letter; purine N7/6-position, pyrimidine 4-position
    "A": (("maj", "pur_N7", "N7"), ("maj", "pur_6", "N6"), ("min", "pur_N3", "N3")),
    "G": (("maj", "pur_N7", "N7"), ("maj", "pur_6", "O6"), ("min", "pur_N3", "N3"), ("min", "G_N2", "N2")),
    "C": (("maj", "pyr_4", "N4"), ("min", "pyr_O2", "O2")),
    "T": (("maj", "pyr_4", "O4"), ("min", "pyr_O2", "O2")),
}
MEP_OFFSET_A = 1.5          # where a protein donor/acceptor sits relative to the atom
# Local direction of each site (atom, neighbours): ring-N acceptors along the bisector away
# from their two ring neighbours (the lone pair), carbonyl O along C=O, amino N toward the
# midpoint of its hydrogens (the donor side). A direction from the base-pair centre was tried
# first and put the N7 probe next to H8 (+40 kcal/mol/e at an acceptor).
SITE_FRAME = {"N7": ("ring", ("C5", "C8")), "N3": ("ring", ("C2", "C4")),
              "O6": ("carbonyl", ("C6",)), "O4": ("carbonyl", ("C4",)), "O2": ("carbonyl", ("C2",)),
              "N6": ("amino", ("H61", "H62")), "N4": ("amino", ("H41", "H42")),
              "N2": ("amino", ("H21", "H22"))}


def site_point(atom_xyz, names, coords_of, atom: str):
    """Probe point MEP_OFFSET_A from `atom` along its local donor/acceptor direction."""
    import numpy as np
    kind, nb = SITE_FRAME[atom]
    x = coords_of(atom)
    if kind == "ring":
        u = sum((x - coords_of(n)) / np.linalg.norm(x - coords_of(n)) for n in nb)
    elif kind == "carbonyl":
        u = x - coords_of(nb[0])
    else:
        u = 0.5 * (coords_of(nb[0]) + coords_of(nb[1])) - x
    return x + MEP_OFFSET_A * u / np.linalg.norm(u)
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _arr(x):
    return x.to_array() if hasattr(x, "to_array") else x


def stage_b(psi4, files: Dict[str, str], context: str, method: str, basis: str,
            polarizability: bool = True) -> Dict[str, float]:
    """QC Stage B on the stacked step (w1, c1, w2, c2): in-stack hole/electron site energies,
    effective transfer integrals (Valeev et al. 2006: fragment frontier orbitals of the isolated
    bases projected on the dimer Fock matrix, Lowdin-corrected for overlap), the molecular
    electrostatic potential 1.5 A outside each groove-facing H-bonding atom, MBIS charges (net
    per base and at those atoms), and the static dipole polarizability of the step."""
    import numpy as np
    order = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                        "base_atom_order.json")))["order"]
    w1, w2 = context[1], context[2]
    letters = {"w1": w1, "c1": COMP[w1], "w2": w2, "c2": COMP[w2]}
    tags = ("w1", "c1", "w2", "c2")
    xyz = {k: read_xyz(files[k]) for k in tags}
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": False, "reference": "rks"})
    out: Dict[str, float] = {}
    t0 = time.time()
    dimer_lines = xyz["w1"] + xyz["c1"] + xyz["w2"] + xyz["c2"]
    mol = molecule(psi4, [dimer_lines])
    e0, wd = psi4.energy(method, molecule=mol, return_wfn=True)
    F, S = _arr(wd.Fa()), _arr(wd.S())
    bs = wd.basisset(); nbf = bs.nbf()
    cen = np.array([bs.function_to_center(i) for i in range(nbf)])
    starts = np.cumsum([0] + [len(xyz[k]) for k in tags])
    fo = {}
    for bi, k in enumerate(tags):
        em, wm = psi4.energy(method, molecule=molecule(psi4, [xyz[k]]), return_wfn=True)
        C, nocc = _arr(wm.Ca()), wm.nalpha()
        dfun = np.flatnonzero((cen >= starts[bi]) & (cen < starts[bi + 1]))
        if len(dfun) != wm.basisset().nbf():
            raise RuntimeError(f"basis mapping failed for {k}")
        for which, j in (("hole", nocc - 1), ("elec", nocc)):
            v = np.zeros(nbf); v[dfun] = C[:, j]
            fo[(k, which)] = v
            out[f"site_{which}_{k}_eV"] = float(v @ F @ v) * HARTREE_EV
    def teff(a, b):
        ea, eb, hab, sab = a @ F @ a, b @ F @ b, a @ F @ b, a @ S @ b
        return abs((hab - 0.5 * (ea + eb) * sab) / (1.0 - sab ** 2)) * HARTREE_EV
    for name, (x, y) in (("intraW", ("w1", "w2")), ("intraC", ("c1", "c2")),
                         ("inter_w1c2", ("w1", "c2")), ("inter_c1w2", ("c1", "w2")),
                         ("pair_bp1", ("w1", "c1")), ("pair_bp2", ("w2", "c2"))):
        for which in ("hole", "elec"):
            out[f"t_{which}_{name}_eV"] = teff(fo[(x, which)], fo[(y, which)])
    # MBIS charges and the groove MEP on the dimer density
    psi4.oeprop(wd, "MBIS_CHARGES")
    q = np.asarray(_arr(wd.variable("MBIS CHARGES"))).ravel()
    coords = np.array([[float(t) for t in l.split()[1:4]] for l in dimer_lines])
    pts, keys = [], []
    for bi, k in enumerate(tags):
        names = order[letters[k]]
        if len(names) != len(xyz[k]):
            raise RuntimeError(f"atom names do not match {k}")
        sl = slice(starts[bi], starts[bi + 1])
        out[f"mbis_net_{k}_e"] = float(q[sl].sum())
        coords_of = lambda n, bi=bi, names=names: coords[starts[bi] + names.index(n)]  # noqa: E731
        for groove, label, atom in GROOVE_SITES[letters[k]]:
            ai = starts[bi] + names.index(atom)
            pts.append(site_point(coords[ai], names, coords_of, atom))
            tag_bp = "bp1" if k in ("w1", "c1") else "bp2"
            keys.append(f"{groove}_{label}_{tag_bp}")
            out[f"mbis_{groove}_{label}_{tag_bp}_e"] = float(q[ai])
    cwd = os.getcwd()
    tmp = os.path.join(psi4.core.IOManager.shared_object().get_default_path(), f"esp_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True); os.chdir(tmp)
    try:
        np.savetxt("grid.dat", np.array(pts), fmt="%.6f")          # angstrom (probed)
        psi4.oeprop(wd, "GRID_ESP")
        v = np.atleast_1d(np.loadtxt("grid_esp.dat"))
    finally:
        os.chdir(cwd)
    for k_, val in zip(keys, v):
        out[f"mep_{k_}_kcal_mol_e"] = float(val) * HARTREE_KCAL
    if polarizability:
        t = time.time()
        psi4.properties(method, properties=["DIPOLE_POLARIZABILITIES"], molecule=mol)
        A = np.array([[psi4.variable(f"DIPOLE POLARIZABILITY {a}{b}") for b in "XYZ"] for a in "XYZ"])
        A = 0.5 * (A + A.T)
        ev = np.linalg.eigvalsh(A)
        u = coords[starts[2]:starts[4]].mean(0) - coords[starts[0]:starts[2]].mean(0)
        u /= np.linalg.norm(u)
        out["polar_iso_au"] = float(ev.mean())
        out["polar_aniso_au"] = float(np.sqrt(0.5 * ((ev[0] - ev[1]) ** 2 + (ev[1] - ev[2]) ** 2 + (ev[2] - ev[0]) ** 2)))
        out["polar_axial_au"] = float(u @ A @ u)
        out["t_polar_s"] = round(time.time() - t, 1)
    out["t_stage_b_s"] = round(time.time() - t0, 1)
    return out


# The four ring nitrogens a duplex exposes (the plan's "pKa shifts at four ring nitrogens"): the
# purine N7 in the major groove and N3 in the minor groove, of A and G. The Watson-Crick-face
# sites (A N1, C N3) are excluded: protonating them needs the pair to open (Family E's opening
# channels); computed rigidly in the pair, the added proton lands on the partner's imino
# hydrogen and the "shift" is a clash energy (probe: A N1 -33 pKa units, A N7 -0.2).
PKA_SITES = {"A": (("N7", ("C5", "C8")), ("N3", ("C2", "C4"))),
             "G": (("N7", ("C5", "C8")), ("N3", ("C2", "C4"))),
             "C": (),
             "T": ()}
N_H_A = 1.01
TWO_PT3_RT_KCAL = 2.302585 * 0.0019872043 * 298.15        # 1.364 kcal/mol per pKa unit
PCM_INPUT = """
Units = Angstrom
Medium {
SolverType = IEFPCM
Solvent = Water
}
Cavity {
RadiiSet = UFF
Type = GePol
Scaling = False
Area = {area}
Mode = Implicit
}
"""


def _protonate(lines, names, site, nb):
    import numpy as np
    X = {n: np.array([float(t) for t in l.split()[1:4]]) for n, l in zip(names, lines)}
    u = sum((X[site] - X[a]) / np.linalg.norm(X[site] - X[a]) for a in nb)
    h = X[site] + N_H_A * u / np.linalg.norm(u)
    return lines + [f"H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}"]


def pka_context(psi4, files: Dict[str, str], context: str, method: str, basis: str,
                pcm_area: float = 0.5, solvation: str = "pcm", ddx_lmax: int = 8,
                ddx_n_lebedev: int = 434, ddx_eta: float = 0.1, ddx_fmm: bool = True,
                ddx_maxiter: int = 100) -> Dict[str, float]:
    """Context shift of ring-nitrogen pKa in the stacked step, in implicit water (IEF-PCM).

    For each protonation site of the four bases: dE_ctx = E(step with H+) - E(step) and
    dE_free = E(base with H+) - E(base), the base alone in its own context geometry, both
    in PCM; the shift is dpKa = -(dE_ctx - dE_free) / (2.303 RT). Rigid geometries (the added
    proton along the lone pair at 1.01 A); vibrational and entropic terms assumed to cancel
    between the step and the free base. Cation energies by restricted closed shell (+1, singlet).
    """
    order = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                        "base_atom_order.json")))["order"]
    w1, w2 = context[1], context[2]
    letters = {"w1": w1, "c1": COMP[w1], "w2": w2, "c2": COMP[w2]}
    tags = ("w1", "c1", "w2", "c2")
    xyz = {k: read_xyz(files[k]) for k in tags}
    if solvation == "ddx":
        # domain-decomposition PCM (ddX): smooth in its discretisation (spherical harmonics up to
        # ddx_lmax on Lebedev grids), unlike GePol tesserae, whose element size moved the shifts by
        # ~1 pKa unit between 0.5 and 0.3 A^2 (DECISIONS.md)
        psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": False, "reference": "rks",
                          "ddx": True, "ddx_model": "pcm", "ddx_solvent": "water", "ddx_radii_set": "uff",
                          "ddx_lmax": int(ddx_lmax), "ddx_n_lebedev": int(ddx_n_lebedev), "ddx_eta": float(ddx_eta)})
        # psi4 defaults (fast multipoles on, 100 Jacobi/DIIS iterations) are set only when changed: at lmax 16
        # the multipole solver's local expansion (lmax 6) left the ddPCM system unconverged (DECISIONS.md)
        if not ddx_fmm:
            psi4.set_options({"ddx_fmm": False})
        if int(ddx_maxiter) != 100:
            psi4.set_options({"ddx_maxiter": int(ddx_maxiter)})
    else:
        psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": False, "reference": "rks",
                          "pcm": True, "pcm_scf_type": "total"})
        psi4.pcm_helper(PCM_INPUT.replace("{area}", f"{pcm_area:g}"))
    out: Dict[str, float] = {}
    t0 = time.time()
    step = xyz["w1"] + xyz["c1"] + xyz["w2"] + xyz["c2"]
    e_step = psi4.energy(method, molecule=molecule(psi4, [step]))
    for k in tags:
        names = order[letters[k]]
        e_base = None
        for site, nb in PKA_SITES[letters[k]]:
            if e_base is None:
                e_base = psi4.energy(method, molecule=molecule(psi4, [xyz[k]]))
            base_h = _protonate(xyz[k], names, site, nb)
            e_base_h = psi4.energy(method, molecule=molecule(psi4, [base_h], charge=1, mult=1))
            parts = {t: (base_h if t == k else xyz[t]) for t in tags}
            step_h = parts["w1"] + parts["c1"] + parts["w2"] + parts["c2"]
            e_step_h = psi4.energy(method, molecule=molecule(psi4, [step_h], charge=1, mult=1))
            ddE = ((e_step_h - e_step) - (e_base_h - e_base)) * HARTREE_KCAL
            bp = "bp1" if k in ("w1", "c1") else "bp2"
            out[f"dpka_{letters[k]}{site}_{k}_{bp}"] = -ddE / TWO_PT3_RT_KCAL
    psi4.set_options({"pcm": False, "ddx": False})
    out["t_pka_s"] = round(time.time() - t0, 1)
    out["solvation"] = (f"ddx lmax {ddx_lmax} lebedev {ddx_n_lebedev} eta {ddx_eta:g}" + ("" if ddx_fmm else " fmm off")
                        + ("" if int(ddx_maxiter) == 100 else f" maxiter {int(ddx_maxiter)}")) if solvation == "ddx" else f"pcm area {pcm_area:g}"
    return out


def trimer_context(psi4, files: Dict[str, str], method: str, basis: str) -> Dict[str, float]:
    """Base-pair trimer A-B-C: three-body stacking E3 = E(ABC) - E(AB) - E(BC) - E(AC) + E(A) + E(B)
    + E(C) (each in its own basis; the BSSE of a three-body term is small), the two-body sum, the
    vertical IP (dSCF UKS) and where the hole sits -- the cation-minus-neutral Mulliken charge of
    each pair -- and the HOMO/LUMO energies with the HOMO's Mulliken weight on each pair."""
    import numpy as np
    A, B_, C = read_xyz(files["bpA"]), read_xyz(files["bpB"]), read_xyz(files["bpC"])
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": False, "reference": "rks"})
    t = time.time()
    E = {}
    for k, frag in (("A", A), ("B", B_), ("C", C), ("AB", A + B_), ("BC", B_ + C), ("AC", A + C)):
        E[k] = psi4.energy(method, molecule=molecule(psi4, [frag]))
    e0, wfn = psi4.energy(method, molecule=molecule(psi4, [A + B_ + C]), return_wfn=True)
    out = {"tri_E3_kcal_mol": (e0 - E["AB"] - E["BC"] - E["AC"] + E["A"] + E["B"] + E["C"]) * HARTREE_KCAL,
           "tri_E2_AB_kcal_mol": (E["AB"] - E["A"] - E["B"]) * HARTREE_KCAL,
           "tri_E2_BC_kcal_mol": (E["BC"] - E["B"] - E["C"]) * HARTREE_KCAL,
           "tri_E2_AC_kcal_mol": (E["AC"] - E["A"] - E["C"]) * HARTREE_KCAL}
    nA, nB = len(A), len(B_)
    owner = np.array([0] * nA + [1] * nB + [2] * len(C))
    eps = wfn.epsilon_a(); eps = eps.to_array() if hasattr(eps, "to_array") else eps
    nocc = wfn.nalpha()
    out["tri_homo_eV"] = float(eps[nocc - 1]) * HARTREE_EV; out["tri_lumo_eV"] = float(eps[nocc]) * HARTREE_EV
    Ca = np.asarray(wfn.Ca().to_array() if hasattr(wfn.Ca(), "to_array") else wfn.Ca())
    S = np.asarray(wfn.S().to_array() if hasattr(wfn.S(), "to_array") else wfn.S())
    c = Ca[:, nocc - 1]; pop = c * (S @ c)
    bas = wfn.basisset()
    fcen = np.array([bas.function_to_center(i) for i in range(bas.nbf())])
    w = np.array([pop[owner[fcen] == k].sum() for k in range(3)])
    for k, tag in enumerate("ABC"):
        out[f"tri_homo_w{tag}"] = float(w[k] / w.sum())
    psi4.oeprop(wfn, "MULLIKEN_CHARGES")
    q0 = np.asarray(wfn.atomic_point_charges().to_array() if hasattr(wfn.atomic_point_charges(), "to_array") else wfn.atomic_point_charges())
    psi4.set_options({"reference": "uks"})
    ec, wc = psi4.energy(method, molecule=molecule(psi4, [A + B_ + C], charge=+1, mult=2), return_wfn=True)
    psi4.set_options({"reference": "rks"})
    psi4.oeprop(wc, "MULLIKEN_CHARGES")
    q1 = np.asarray(wc.atomic_point_charges().to_array() if hasattr(wc.atomic_point_charges(), "to_array") else wc.atomic_point_charges())
    dq = q1 - q0
    out["tri_vip_eV"] = (ec - e0) * HARTREE_EV
    for k, tag in enumerate("ABC"):
        out[f"tri_hole_{tag}"] = float(dq[owner == k].sum())
    out["t_trimer_s"] = round(time.time() - t, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--contexts", nargs="*", default=None, help="subset by context name")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--method", default="b3lyp-d3bj"); ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--ipea", action="store_true"); ap.add_argument("--sapt", action="store_true")
    ap.add_argument("--sapt-hbond", action="store_true", help="SAPT0 of each base pair's H-bonding")
    ap.add_argument("--stage-b", action="store_true",
                    help="site energies, transfer integrals, groove MEP, MBIS charges, polarizability")
    ap.add_argument("--no-polarizability", action="store_true")
    ap.add_argument("--pka", action="store_true", help="ring-nitrogen pKa shifts in context (IEF-PCM)")
    ap.add_argument("--pcm-area", type=float, default=0.5, help="GePol cavity element area (A^2)")
    ap.add_argument("--solvation", choices=("pcm", "ddx"), default="pcm")
    ap.add_argument("--ddx-lmax", type=int, default=8); ap.add_argument("--ddx-n-lebedev", type=int, default=434)
    ap.add_argument("--ddx-eta", type=float, default=0.1,
                    help="ddX switching-region width; 0.3 converges in lmax on stacked steps, 0.1 does not (DECISIONS)")
    ap.add_argument("--ddx-no-fmm", action="store_true", help="ddX without fast multipoles (direct sums)")
    ap.add_argument("--ddx-maxiter", type=int, default=100, help="ddX solver iterations (psi4 default 100)")
    ap.add_argument("--no-dft", action="store_true")
    ap.add_argument("--trimer", action="store_true", help="base-pair trimer tier (three-body stacking, IP, hole)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--memory-gb", type=float, default=12.0)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import psi4
    psi4.set_num_threads(a.threads); psi4.set_memory(f"{a.memory_gb:g} GB")
    psi4.core.IOManager.shared_object().set_default_path(a.scratch)
    psi4.core.set_output_file(a.out.replace(".json", ".log"), False)
    idx = json.load(open(a.index))["contexts"]
    if a.contexts:
        idx = [c for c in idx if c["context"] in set(a.contexts)]
    idx = idx[a.shard::a.n_shards]
    res = []
    for c in idx:
        r = {"context": c["context"], "central_step": c["central_step"], "status": "ok"}
        try:
            if not a.no_dft:
                r.update(dft_context(psi4, c["files"], a.method, a.basis, a.ipea))
            if a.sapt:
                r.update(sapt_context(psi4, c["files"]))
            if a.sapt_hbond:
                r.update(sapt_hbond(psi4, c["files"]))
            if a.stage_b:
                r.update(stage_b(psi4, c["files"], c["context"], a.method, a.basis,
                                 polarizability=not a.no_polarizability))
            if a.trimer:
                r.update(trimer_context(psi4, c["files"], a.method, a.basis))
            if a.pka:
                r.update(pka_context(psi4, c["files"], c["context"], a.method, a.basis, a.pcm_area,
                                     a.solvation, a.ddx_lmax, a.ddx_n_lebedev, a.ddx_eta,
                                     not a.ddx_no_fmm, a.ddx_maxiter))
        except Exception as exc:                                     # noqa: BLE001
            r["status"] = "failed"; r["error"] = f"{type(exc).__name__}: {exc}"[:300]
        psi4.core.clean()
        res.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"method": a.method, "basis": a.basis, "index": a.index, "shard": [a.shard, a.n_shards],
               "results": res}, open(a.out, "w"), indent=1)
    return 0 if all(r["status"] == "ok" for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
