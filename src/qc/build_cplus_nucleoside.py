#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Geometry for the N3-protonated cytosine (C+) charge fit: the middle deoxycytidine of a cgNA+
B-DNA build (hydrogens by PDBFixer, as in production), its phosphates replaced by methyl caps on
O5' and O3' (C at 1.43 A along O-P, tetrahedral hydrogens), and H3 placed on N3 along the
external bisector of C2-N3-C4 at 1.01 A. Writes an annotated .xyz (atom name per line)."""
import argparse, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import cgna_atomic as CA  # noqa: E402


def methyl(o, c_dir, ref):
    """C at 1.43 A from O along c_dir, three H at 1.09 A, tetrahedral, staggered against ref."""
    c = o + 1.43 * c_dir
    a = -c_dir
    perp = np.cross(a, ref - o); perp /= np.linalg.norm(perp)
    hs = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        p = np.cos(ang) * perp + np.sin(ang) * np.cross(a, perp)
        hs.append(c + 1.09 * (np.cos(np.radians(70.5)) * (-a) + np.sin(np.radians(70.5)) * p))
    return c, hs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from openmm import app, unit
    from pdbfixer import PDBFixer
    work = tempfile.mkdtemp()
    seq = "GCCCG"
    gs = CA.CGNAGroundState(seq, CA.DEFAULT_CGNA_ROOT)
    atoms = CA.build_duplex(seq, CA.NucleotideTemplates(CA.DEFAULT_CGNA_ROOT), gs)
    raw = os.path.join(work, "raw.pdb"); CA.write_pdb(atoms, raw)
    fx = PDBFixer(filename=raw); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    pos = np.array(fx.positions.value_in_unit(unit.angstrom))
    res = [r for r in fx.topology.residues() if r.chain.index == 0]
    mid, nxt = res[2], res[3]
    at = {x.name: pos[x.index] for x in mid.atoms()}
    p_next = {x.name: pos[x.index] for x in nxt.atoms()}["P"]
    names, xyz = [], []
    for n, x in at.items():
        if n in ("P", "OP1", "OP2"):
            continue
        names.append(n); xyz.append(x)
    d5 = at["P"] - at["O5'"]; d5 /= np.linalg.norm(d5)
    c5, h5 = methyl(at["O5'"], d5, at["C5'"])
    d3 = p_next - at["O3'"]; d3 /= np.linalg.norm(d3)
    c3, h3 = methyl(at["O3'"], d3, at["C3'"])
    names += ["CM5", "HM51", "HM52", "HM53", "CM3", "HM31", "HM32", "HM33"]
    xyz += [c5, *h5, c3, *h3]
    u = (at["N3"] - at["C2"]) / np.linalg.norm(at["N3"] - at["C2"]) + (at["N3"] - at["C4"]) / np.linalg.norm(at["N3"] - at["C4"])
    names.append("H3"); xyz.append(at["N3"] + 1.01 * u / np.linalg.norm(u))
    xyz = np.array(xyz)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(xyz)) * 9
    print(len(names), "atoms; closest non-bonded-ish contact", round(float(d.min()), 3), "A")
    with open(a.out, "w") as f:
        f.write(f"{len(names)}\nN3-protonated 2'-deoxycytidine, 3',5'-O-dimethyl (charge +1)\n")
        for n, x in zip(names, xyz):
            el = n[0] if not n.startswith("HM") else "H"
            el = "C" if n.startswith("CM") else el
            f.write(f"{el} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {n}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
