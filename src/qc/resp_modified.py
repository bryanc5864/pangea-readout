#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""RESP charges for the RNA modifications of Family N (m6A, inosine, pseudouridine, and 2'-O-methyl
on A, C, G, U), AMBER-compatible, by the protocol of `resp_cplus.py`: the 3',5'-O-dimethyl nucleoside
from a cgRNA+ A-form build (hydrogens by PDBFixer), the modification placed geometrically, the new
atoms and every hydrogen relaxed by GFN2-xTB with the other heavy atoms fixed, the HF/6-31G* potential
on Merz-Kollman surfaces, and a restrained fit in which every atom the modification does not touch
keeps its AMBER RNA.OL3 charge (the sugar for a base modification, the base and the rest of the sugar
for 2'-O-methyl) and the modified group's total equals the unmodified group's (a neutral change).
The methyl caps stand in for the phosphates and are discarded.

Run in the physics environment with --stage geometry (writes the annotated .xyz), then in the
quantum-chemistry environment with --stage fit.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))

SPECIES = {"m6A": ("A", "base"), "ino": ("A", "base"), "psi": ("U", "base"),
           "A0": ("A", "base"), "U0": ("U", "base"),      # unmodified controls: the protocol against OL3's own base charges
           # unmodified controls for 2'-O-methyl: the 2'-OH group (C2', H2', O2', HO2') refitted by the same protocol
           "A0om": ("A", "ome"), "C0om": ("C", "ome"), "G0om": ("G", "ome"), "U0om": ("U", "ome"),
           "Am": ("A", "ome"), "Cm": ("C", "ome"), "Gm": ("G", "ome"), "Um": ("U", "ome")}
BASE_ATOMS = {"N9", "C8", "H8", "N7", "C5", "C6", "N6", "H61", "H62", "N1", "C2", "H2", "N3", "C4", "O6", "N2",
              "H21", "H22", "H1", "O2", "N4", "H41", "H42", "H5", "H6", "O4", "H3", "C6M", "HM0", "HM1", "HM2"}
OME_FREE = {"C2'", "H2'", "O2'", "CM2", "HM20", "HM21", "HM22"}


OL3_XML = os.environ.get("PANGEA_OL3_XML",
                         "/home/bjc75/project/conda_envs/phys/lib/python3.11/site-packages/openmm/app/data/amber14/RNA.OL3.xml")


def ol3_charges(letter):
    import xml.etree.ElementTree as ET
    root = ET.parse(OL3_XML).getroot()          # read as data: the QC environment has no OpenMM
    for r in root.find("Residues"):
        if r.get("name") == letter:
            return {a.get("name"): float(a.get("charge")) for a in r if a.tag == "Atom"}
    raise KeyError(letter)


def _methyl(c_at, d, ref):
    c = c_at
    perp = np.cross(d, ref); perp /= np.linalg.norm(perp)
    hs = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        p = np.cos(ang) * perp + np.sin(ang) * np.cross(d, perp)
        hs.append(c + 1.09 * (np.cos(np.radians(70.5)) * d + np.sin(np.radians(70.5)) * p))
    return hs


def geometry(species, out_xyz):
    from openmm import unit
    from pdbfixer import PDBFixer
    import cgna_atomic as ca
    import duplex_classes as K
    import rna_atomic as R
    letter, kind = SPECIES[species]
    seq = "GCG" + {"A": "A", "U": "T", "G": "G", "C": "C"}[letter] + "CGC"
    work = tempfile.mkdtemp()
    raw = os.path.join(work, "raw.pdb"); ca.write_pdb(R.build_rna_duplex(seq, K.SETS["rna"]), raw)
    fx = PDBFixer(filename=raw); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    pos = np.array(fx.positions.value_in_unit(unit.angstrom))
    res = [r for r in fx.topology.residues() if r.chain.index == 0]
    at = {a.name: pos[a.index] for a in res[3].atoms()}
    p_next = {a.name: pos[a.index] for a in res[4].atoms()}["P"]
    names, xyz = [], []
    for n_, x in at.items():
        if n_ not in ("P", "OP1", "OP2"):
            names.append(n_); xyz.append(x)
    X = dict(zip(names, xyz))
    free = [n_ for n_ in names if n_.startswith("H")]
    if species == "m6A":
        h = max(("H61", "H62"), key=lambda k: np.linalg.norm(X[k] - X["N1"]))
        d = (X[h] - X["N6"]) / np.linalg.norm(X[h] - X["N6"]); c = X["N6"] + 1.45 * d
        del X[h]; X["C6M"] = c
        for k, hh in enumerate(_methyl(c, d, X["C6"] - X["N6"])):
            X[f"HM{k}"] = hh
        free += ["C6M", "HM0", "HM1", "HM2"]
    elif species == "ino":
        d = (X["N6"] - X["C6"]) / np.linalg.norm(X["N6"] - X["C6"])
        for k in ("N6", "H61", "H62"):
            X.pop(k)
        X["O6"] = X["C6"] + 1.23 * d
        u = (X["N1"] - X["C2"]) / np.linalg.norm(X["N1"] - X["C2"]) + (X["N1"] - X["C6"]) / np.linalg.norm(X["N1"] - X["C6"])
        X["H1"] = X["N1"] + 1.01 * u / np.linalg.norm(u)
        free += ["O6", "H1"]
    elif species == "psi":
        # uracil ring attached through C5: positions kept, the two atoms relabelled (element N1 <-> C5)
        ren = {"N1": "C5", "C2": "C4", "O2": "O4", "C4": "C2", "O4": "O2", "C5": "N1", "H5": "H1"}
        X = {ren.get(k, k): v for k, v in X.items()}
        free = [ren.get(k, k) for k in free] + [k for k in ("C5", "C4", "C2", "N1", "O2", "O4") if k in X]
    elif kind == "ome" and not species.endswith("0om"):
        d = (X["HO2'"] - X["O2'"]) / np.linalg.norm(X["HO2'"] - X["O2'"]); c = X["O2'"] + 1.43 * d
        X.pop("HO2'"); X["CM2"] = c
        for k, hh in enumerate(_methyl(c, d, X["C2'"] - X["O2'"])):
            X[f"HM2{k}"] = hh
        free += ["CM2", "HM20", "HM21", "HM22"]
    # 5'/3' methyl caps (the phosphates' stand-ins)
    d5 = at["P"] - X["O5'"]; d5 /= np.linalg.norm(d5); c5 = X["O5'"] + 1.43 * d5
    d3 = p_next - X["O3'"]; d3 /= np.linalg.norm(d3); c3 = X["O3'"] + 1.43 * d3
    X["CM5"] = c5; X["CM3"] = c3
    for k, hh in enumerate(_methyl(c5, d5, X["C5'"] - X["O5'"])):
        X[f"HM5{k}"] = hh
    for k, hh in enumerate(_methyl(c3, d3, X["C3'"] - X["O3'"])):
        X[f"HM3{k}"] = hh
    free += ["CM5", "CM3"] + [f"HM5{k}" for k in range(3)] + [f"HM3{k}" for k in range(3)]
    el = lambda n_: ("H" if n_.startswith("H") else n_[0])                       # noqa: E731
    with open(out_xyz, "w") as f:
        f.write(f"{len(X)}\n{species} free:{','.join(sorted(set(free)))}\n")
        for n_, x in X.items():
            f.write(f"{el(n_)} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {n_}\n")
    return out_xyz


def resp_fit_multi(xyz, pts, esp, fixed, groups, constraints, restrained, a=0.0005, b=0.1, iters=50):
    """`resp_cplus.resp_fit` with several exact linear constraints: `constraints` is a list of
    (atom indices, total charge of those atoms), each imposed by its own Lagrange multiplier
    (here: the whole molecule neutral, and the modified group keeping its unmodified total)."""
    from resp_cplus import BOHR
    n = len(xyz)
    inv = BOHR / np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=-1)
    G = len(groups)
    M = np.zeros((n, G))
    for g, idx in enumerate(groups):
        M[idx, g] = 1.0
    q_fixed = np.zeros(n)
    for i, v in fixed.items():
        q_fixed[i] = v
    A_ = inv @ M; y = esp - inv @ q_fixed
    C = np.zeros((len(constraints), G)); t = np.zeros(len(constraints))
    for c, (atoms, total) in enumerate(constraints):
        sel = np.zeros(n); sel[list(atoms)] = 1.0
        C[c] = sel @ M; t[c] = total - float(sel @ q_fixed)
    qg = np.zeros(G); nc = len(constraints)
    for _ in range(iters):
        w = np.zeros(G)
        for g, idx in enumerate(groups):
            if any(i in restrained for i in idx):
                w[g] = a * len(idx) / np.sqrt(qg[g] ** 2 + b * b)
        K = np.zeros((G + nc, G + nc)); K[:G, :G] = A_.T @ A_ + np.diag(w); K[:G, G:] = C.T; K[G:, :G] = C
        new = np.linalg.solve(K, np.concatenate([A_.T @ y, t]))[:G]
        if np.max(np.abs(new - qg)) < 1e-7:
            qg = new; break
        qg = new
    q = M @ qg + q_fixed
    res = inv @ q - esp
    return q, float(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(esp ** 2)))


def fit(species, xyz_path, out_json, threads=4, memory_gb=8):
    import psi4
    sys.path.insert(0, HERE)
    import resp_cplus as RC
    letter, kind = SPECIES[species]
    lines = open(xyz_path).read().splitlines()
    free = set(lines[1].split("free:")[1].split(","))
    rows = lines[2:]
    els = [l.split()[0] for l in rows]; names = [l.split()[4] for l in rows]
    xyz = np.array([[float(t) for t in l.split()[1:4]] for l in rows])
    # GFN2-xTB optimisation with the A-form conformation held by its torsions (glycosidic chi, two
    # endocyclic sugar torsions fixing the pucker, gamma, delta and the two cap torsions); bond
    # lengths and angles relax. (xTB 6.7.1 does not hold atoms listed in a $fix block: measured, the
    # "fixed" atoms moved 0.4 A in 15 steps, and the builder's distorted bonds sent that run to a
    # 0.02 eV gap; the torsion constraints converge in 40 steps with a 3.2 eV gap.)
    xtb = os.environ.get("XTB_BIN", "/home/bjc75/other/biophyschem/scripts/xtb_wrapper.sh")
    w = tempfile.mkdtemp()
    open(os.path.join(w, "in.xyz"), "w").write(f"{len(els)}\n\n" + "".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for e, x in zip(els, xyz)))
    ix = {n_: i + 1 for i, n_ in enumerate(names)}
    chi = {"base": {"m6A": ("O4'", "C1'", "N9", "C4"), "ino": ("O4'", "C1'", "N9", "C4"), "psi": ("O4'", "C1'", "C5", "C4"),
                    "A0": ("O4'", "C1'", "N9", "C4"), "U0": ("O4'", "C1'", "N1", "C2")},
           "ome": {"Am": ("O4'", "C1'", "N9", "C4"), "Gm": ("O4'", "C1'", "N9", "C4"), "Cm": ("O4'", "C1'", "N1", "C2"),
                   "Um": ("O4'", "C1'", "N1", "C2"), "A0om": ("O4'", "C1'", "N9", "C4"), "G0om": ("O4'", "C1'", "N9", "C4"),
                   "C0om": ("O4'", "C1'", "N1", "C2"), "U0om": ("O4'", "C1'", "N1", "C2")}}[kind][species]
    dih = [chi, ("O4'", "C1'", "C2'", "C3'"), ("C2'", "C3'", "C4'", "O4'"), ("O5'", "C5'", "C4'", "C3'"),
           ("C5'", "C4'", "C3'", "O3'"), ("C4'", "C3'", "O3'", "CM3"), ("C4'", "C5'", "O5'", "CM5")]
    open(os.path.join(w, "xc"), "w").write("$constrain\n force constant=0.5\n" +
                                           "".join(f" dihedral: {ix[a_]},{ix[b_]},{ix[c_]},{ix[d_]},auto\n" for a_, b_, c_, d_ in dih) + "$end\n")
    ok = False
    for extra in ([], ["--etemp", "1000"]):
        r = subprocess.run([xtb, "in.xyz", "--opt", "--gfn", "2", "--input", "xc", *extra], cwd=w, capture_output=True, text=True)
        if r.returncode == 0 and "GEOMETRY OPTIMIZATION CONVERGED" in r.stdout and os.path.exists(os.path.join(w, "xtbopt.xyz")):
            ok = True
            break
    if not ok:
        raise RuntimeError("xTB constrained optimisation did not converge: " + r.stdout[-800:])
    gap = [l for l in r.stdout.splitlines() if "HOMO-LUMO GAP" in l]
    xyz = np.array([[float(t) for t in l.split()[1:4]] for l in open(os.path.join(w, "xtbopt.xyz")).read().splitlines()[2:2 + len(els)]])
    with open(out_json.replace("_resp.json", "_relaxed.xyz"), "w") as f:
        f.write(f"{len(els)}\n{species} GFN2-xTB, A-form torsions constrained; {gap[-1].strip() if gap else ''}\n")
        for e_, n_, x in zip(els, names, xyz):
            f.write(f"{e_} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {n_}\n")
    psi4.set_num_threads(threads); psi4.set_memory(f"{memory_gb} GB")
    psi4.core.set_output_file(out_json.replace(".json", ".log"), False)
    mol = psi4.geometry("0 1\n" + "\n".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" for e, x in zip(els, xyz)) +
                        "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1")
    psi4.set_options({"basis": "6-31g*", "scf_type": "pk", "maxiter": 200})
    e, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
    cwd = os.getcwd(); os.chdir(w)
    pts = RC.mk_points(xyz, els); np.savetxt("grid.dat", pts, fmt="%.8f")
    psi4.oeprop(wfn, "GRID_ESP"); esp = np.loadtxt("grid_esp.dat"); os.chdir(cwd)
    q_ol3 = ol3_charges(letter if species != "psi" else "U")
    idx = {n_: i for i, n_ in enumerate(names)}
    caps = {n_ for n_ in names if n_.startswith(("CM5", "CM3", "HM5", "HM3"))}
    if kind == "base":
        modset = {n_ for n_ in names if n_ in BASE_ATOMS or n_ in ("C6M", "HM0", "HM1", "HM2")}
        target = sum(q_ol3[n_] for n_ in q_ol3 if n_ in BASE_ATOMS)
    else:
        modset = {n_ for n_ in names if n_ in OME_FREE or (species.endswith("0om") and n_ == "HO2'")}
        target = sum(q_ol3[n_] for n_ in ("C2'", "H2'", "O2'", "HO2'"))
    fixed = {idx[n_]: q_ol3[n_] for n_ in names if n_ not in modset and n_ not in caps and n_ in q_ol3}
    unknown = [n_ for n_ in names if n_ not in modset and n_ not in caps and n_ not in q_ol3]
    if unknown:
        raise KeyError(f"atoms without an OL3 charge: {unknown}")
    eq = [[n_ for n_ in names if n_.startswith("HM5")], [n_ for n_ in names if n_.startswith("HM3")]]
    if kind == "ome" and not species.endswith("0om"):
        eq.append([n_ for n_ in names if n_.startswith("HM2")])
    elif species == "m6A":
        eq.append(["HM0", "HM1", "HM2"])
    groups, grouped = [], set()
    for g in eq:
        if g:
            groups.append([idx[n_] for n_ in g]); grouped.update(idx[n_] for n_ in g)
    for n_ in names:
        i = idx[n_]
        if i not in fixed and i not in grouped:
            groups.append([i])
    restrained = {idx[n_] for n_ in names if idx[n_] not in fixed and not n_.startswith("H")}
    mod_idx = [idx[n_] for n_ in modset]
    # exact constraints: the molecule neutral, the modified group at its unmodified total
    q, rrms = resp_fit_multi(xyz, pts, esp, fixed, groups, [(range(len(names)), 0.0), (mod_idx, target)], restrained)
    out = {"species": species, "parent": letter, "kind": kind, "relative_rms": round(rrms, 4),
           "charges": {n_: round(float(q[idx[n_]]), 4) for n_ in sorted(modset)},
           "group_sum": round(float(sum(q[i] for i in mod_idx)), 4), "target_sum": round(target, 4),
           "method": "RESP HF/6-31G* MK, unmodified atoms fixed to AMBER RNA.OL3"}
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("geometry", "fit"), required=True)
    ap.add_argument("--species", nargs="+", default=list(SPECIES)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sp in a.species:
        if a.stage == "geometry":
            geometry(sp, os.path.join(a.out, f"{sp}.xyz")); print(sp, "geometry")
        else:
            try:
                fit(sp, os.path.join(a.out, f"{sp}.xyz"), os.path.join(a.out, f"{sp}_resp.json"))
            except Exception as exc:                                 # noqa: BLE001
                json.dump({"species": sp, "status": f"failed: {type(exc).__name__}: {str(exc)[:500]}"},
                          open(os.path.join(a.out, f"{sp}_failed.json"), "w"))
                print(sp, "failed:", type(exc).__name__, str(exc)[:300], flush=True)
