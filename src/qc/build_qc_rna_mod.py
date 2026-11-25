#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Fragment library for Family N's "DFT on modified stacks" (PLAN.md, Family N): the central
base-pair step of every RC-unique tetramer context in A-form RNA and in the DNA-RNA hybrid (cgRNA+
/ cgHYB+ geometry, `physics/rna_atomic`, hydrogens by PDBFixer), each base capped at its glycosidic
atom as in the Family B library (`build_qc_fragments.py`), unmodified and with one modified base
at each RNA position of the step that can carry it:

* m6A (N6-methyladenosine, at A): the amino hydrogen farther from N1 replaced by a methyl (the anti
  conformer, the one compatible with Watson-Crick pairing);
* inosine (A-to-I, at A): the N6 amino replaced by O6, N1 protonated;
* pseudouridine (at U): the uracil ring attached through C5 -- the same ring positions with N1 and
  C5 exchanged, so Psi pairs with A isosterically and puts its N1-H where U has C5-H;

the modified atoms relaxed with GFN2-xTB with every other atom of the base fixed. Writes the xyz
files and an index in the format `run_psi4.py` reads (one "context" per variant: <context>_<class>
_<mod>_<site>, the unmodified one <context>_<class>_none), so stacking and hydrogen-bond energies
come from the same DFT tier as Family B.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics")); sys.path.insert(0, HERE)
import build_qc_fragments as QF   # noqa: E402
import cgna_atomic as ca          # noqa: E402
import duplex_classes as K        # noqa: E402
import rna_atomic as R            # noqa: E402

XTB = os.environ.get("XTB_BIN", "/home/bjc75/other/biophyschem/scripts/xtb_wrapper.sh")
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "T": "N1"}
SUGAR = QF.SUGAR_PHOSPHATE | {"HO2'", "O2'", "H2'", "H2''", "HO5'", "HO3'"}


def _cap(atoms, letter):
    gly = GLY[letter]
    d = atoms["C1'"] - atoms[gly]
    out = {k: v for k, v in atoms.items() if k not in SUGAR}
    out["H_cap"] = atoms[gly] + d / np.linalg.norm(d) * QF.N_H_BOND_A
    return out


def _bisector(a, b, c):
    u = (a - b) / np.linalg.norm(a - b) + (a - c) / np.linalg.norm(a - c)
    return u / np.linalg.norm(u)


def modify(base, letter, mod):
    """(atoms with element labels, indices to relax) of the modified capped base."""
    at = {k: np.asarray(v, float) for k, v in base.items()}
    el = {k: QF._element_of(k) for k in at}
    free = []
    if mod == "m6A" and letter == "A":
        h = max(("H61", "H62"), key=lambda k: np.linalg.norm(at[k] - at["N1"]))
        d = (at[h] - at["N6"]) / np.linalg.norm(at[h] - at["N6"])
        c = at["N6"] + 1.45 * d
        del at[h]; del el[h]
        at["C6M"] = c; el["C6M"] = "C"
        perp = np.cross(d, at["C6"] - at["N6"]); perp /= np.linalg.norm(perp)
        for k in range(3):
            ang = 2 * np.pi * k / 3
            p = np.cos(ang) * perp + np.sin(ang) * np.cross(d, perp)
            at[f"HM{k}"] = c + 1.09 * (np.cos(np.radians(70.5)) * d + np.sin(np.radians(70.5)) * p)
            el[f"HM{k}"] = "H"
        free = ["C6M", "HM0", "HM1", "HM2"]
    elif mod == "ino" and letter == "A":
        d = (at["N6"] - at["C6"]) / np.linalg.norm(at["N6"] - at["C6"])
        for k in ("N6", "H61", "H62"):
            at.pop(k, None); el.pop(k, None)
        at["O6"] = at["C6"] + 1.23 * d; el["O6"] = "O"
        at["H1"] = at["N1"] + 1.01 * _bisector(at["N1"], at["C2"], at["C6"]); el["H1"] = "H"
        free = ["O6", "H1"]
    elif mod == "psi" and letter in ("U", "T"):
        el["N1"], el["C5"] = "C", "N"          # attachment C at the old N1 site, N-H at the old C5-H
        free = [k for k in at if k not in ("N3", "C6", "H_cap")]
    else:
        return None, None
    return (at, el), free


def relax(at_el, free, work):
    """Hydrogens and new atoms relaxed with the built heavy-atom framework held; a stiffer
    constraint is tried when the framework moved more than 0.05 A."""
    last = None
    for k_c in (2.0, 20.0):
        try:
            return _relax(at_el, free, work, k_c)
        except RuntimeError as exc:
            last = exc
            if "moved" not in str(exc):
                raise
    raise last


def _relax(at_el, free, work, k_c):
    at, el = at_el
    names = list(at)
    with open(os.path.join(work, "in.xyz"), "w") as f:
        f.write(f"{len(names)}\n\n" + "".join(f"{el[n]} {at[n][0]:.6f} {at[n][1]:.6f} {at[n][2]:.6f}\n" for n in names))
    fix = ",".join(str(i + 1) for i, n in enumerate(names) if n not in free)
    # the built heavy-atom framework held rigid by distance constraints among its atoms (xTB 6.7.1
    # does not hold the atoms of a $fix block during optimisation: measured, they moved 0.4 A)
    open(os.path.join(work, "xc"), "w").write(f"$constrain\n force constant={k_c}\n atoms: {fix}\n$end\n")
    for extra in ([], ["--etemp", "1000"], ["--etemp", "3000"]):   # electronic smearing if SCC stalls
        r = subprocess.run([XTB, "in.xyz", "--opt", "--gfn", "2", "--input", "xc", *extra], cwd=work,
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(os.path.join(work, "xtbopt.xyz")):
            break
    else:
        raise RuntimeError("xtb failed: " + r.stdout[-800:])
    new = [list(map(float, l.split()[1:4])) for l in open(os.path.join(work, "xtbopt.xyz")).read().splitlines()[2:2 + len(names)]]
    # xtb may translate/rotate the frame; superpose back on the fixed atoms
    P = np.array([new[i] for i, n in enumerate(names) if n not in free]); Q = np.array([at[n] for n in names if n not in free])
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    Rm = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    moved = np.linalg.norm((P - pc) @ Rm.T + qc - Q, axis=1).max() if len(P) else 0.0
    if moved > 0.05:
        raise RuntimeError(f"constrained heavy atoms moved {moved:.3f} A after superposition")
    # constrained atoms exactly as built; hydrogens and new atoms from the relaxation
    fixed = {n for n in names if n not in free}
    return [(el[n], np.array(at[n]) if n in fixed else Rm @ (np.array(new[i]) - pc) + qc) for i, n in enumerate(names)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True); ap.add_argument("--contexts", nargs="*", default=None)
    a = ap.parse_args()
    from pdbfixer import PDBFixer
    from openmm import app, unit
    os.makedirs(a.out, exist_ok=True)
    ctxs = a.contexts or QF.unique_contexts(4)
    entries, failures = [], []
    for ctx in ctxs:
        for cls in ("rna", "hyb"):
            seq = "GCG" + ctx + "CGC"
            atoms = R.build_rna_duplex(seq, K.SETS["rna"]) if cls == "rna" else R.build_hybrid(seq, K.SETS["hyb"])
            work = tempfile.mkdtemp()
            raw = os.path.join(work, "raw.pdb"); ca.write_pdb(atoms, raw)
            fx = PDBFixer(filename=raw); fx.findMissingResidues(); fx.missingResidues = {}
            fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
            pos = np.array(fx.positions.value_in_unit(unit.angstrom))
            by = {}
            for at_ in fx.topology.atoms():
                by.setdefault((at_.residue.chain.id, int(at_.residue.id)), {})[at_.name] = pos[at_.index]
            n = len(seq); p = 3 + QF.central_step_index(ctx) + 1
            keys = {"w1": ("A", p), "w2": ("A", p + 1), "c1": ("B", n + 1 - p), "c2": ("B", n + 1 - (p + 1))}
            comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
            letter = {"w1": ctx[1], "w2": ctx[2], "c1": comp[ctx[1]], "c2": comp[ctx[2]]}
            rna_strands = ("w1", "w2", "c1", "c2") if cls == "rna" else ("c1", "c2")
            caps = {t: _cap(by[k], "U" if (letter[t] == "T" and t in rna_strands) else letter[t]) for t, k in keys.items()}
            relaxed_caps = {}
            try:
                for t, c in caps.items():              # hydrogens relaxed (xTB), heavy atoms as built
                    hw = tempfile.mkdtemp()
                    el = {k: QF._element_of(k) for k in c}
                    xs = relax((dict(c), el), [k for k in c if el[k] == "H" and k != "H_cap"], hw)
                    relaxed_caps[t] = {k: x for k, (e, x) in zip(c, xs)}
            except Exception as exc:                   # noqa: BLE001
                failures.append({"context": ctx, "class": cls, "stage": "hydrogens", "error": str(exc)[-300:]})
                continue
            caps = relaxed_caps
            variants = [("none", None, None)]
            for t in rna_strands:
                L = letter[t]
                if L == "A":
                    variants += [("m6A", t, L), ("ino", t, L)]
                elif L == "T":
                    variants += [("psi", t, L)]
            for mod, site, L in variants:
                units = {t: [(QF._element_of(k), v) for k, v in caps[t].items()] for t in keys}
                if mod != "none":
                    mw = tempfile.mkdtemp()
                    at_el, free = modify(caps[site], L, mod)
                    try:
                        units[site] = relax(at_el, free, mw)
                    except Exception as exc:           # noqa: BLE001
                        failures.append({"context": ctx, "class": cls, "mod": mod, "site": site, "error": str(exc)[-300:]})
                        continue
                units["bp1"] = units["w1"] + units["c1"]; units["bp2"] = units["w2"] + units["c2"]
                units["dimer"] = units["bp1"] + units["bp2"]
                name = f"{ctx}_{cls}_{mod}" + (f"_{site}" if site else "")
                d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
                files = {}
                for k, xs in units.items():
                    fp = os.path.join(d, f"{k}.xyz")
                    with open(fp, "w") as f:
                        f.write(f"{len(xs)}\n{name} {k}\n" + "".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for e, x in xs))
                    files[k] = fp
                entries.append({"context": name, "central_step": ctx[1:3], "files": files,
                                "n_atoms": {k: len(v) for k, v in units.items()}, "class": cls, "mod": mod, "site": site,
                                "tetramer": ctx})
        print(ctx, "done", flush=True)
    json.dump({"n_contexts": len(entries), "context_length": 4, "geometry_source": "cgRNA+/cgHYB+ ground state, GCG/CGC padding",
               "energy_definitions": ["stacking", "hydrogen_bond_bp1", "hydrogen_bond_bp2"], "contexts": entries, "failures": failures},
              open(os.path.join(a.out, "qc_fragment_index.json"), "w"), indent=1)
    print(len(entries), "fragment sets ->", a.out, ";", len(failures), "failures:", failures[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
