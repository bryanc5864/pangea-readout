#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""RESP charges for N3-protonated deoxycytidine (C+), the residue i-motif C.C+ pairs and
pyrimidine-motif C+.G-C triplets need (Family M), in the AMBER convention.

Protocol (Cornell et al. 1995 / Bayly et al. 1993, as used for AMBER modified nucleosides): the
3',5'-O-dimethyl nucleoside cation (`build_cplus_nucleoside.py`; heavy atoms from a cgNA+ B-DNA
build), hydrogens relaxed (GFN2-xTB) with the heavy atoms fixed, the HF/6-31G* electrostatic
potential on Merz-Kollman surfaces (1.4, 1.6, 1.8, 2.0 x MK radii, 1 point/A^2), and a restrained
fit (hyperbolic restraint a = 0.0005, b = 0.1 on the free heavy atoms) with the sugar atoms held at
the AMBER OL15 DC charges, so DC+ differs from DC only in its base (and H3) and the residue's
charge is that of DC plus one. Equivalences: H41/H42 and each methyl cap's hydrogens. The caps
stand in for the phosphates and are discarded. Writes the base charges (and the fit statistics).
"""
import argparse, json, os, re
import numpy as np

MK_RADII = {"H": 1.2, "C": 1.5, "N": 1.5, "O": 1.4}
SUGAR_FIXED = {"C5'": -0.0069, "H5'": 0.0754, "H5''": 0.0754, "C4'": 0.1629, "H4'": 0.1176,
               "O4'": -0.3691, "C1'": -0.0116, "H1'": 0.1963, "C3'": 0.0713, "H3'": 0.0985,
               "C2'": -0.0854, "H2'": 0.0718, "H2''": 0.0718, "O5'": -0.4954, "O3'": -0.5232}
DC_BASE = {"N1": -0.0339, "C6": -0.0183, "H6": 0.2293, "C5": -0.5222, "H5": 0.1863, "C4": 0.8439,
           "N4": -0.9773, "H41": 0.4314, "H42": 0.4314, "N3": -0.7748, "C2": 0.7959, "O2": -0.6548}
BOHR = 0.52917721067


def mk_points(xyz, els, density=1.0):
    pts = []
    rng = np.random.default_rng(0)
    for scale in (1.4, 1.6, 1.8, 2.0):
        R = np.array([MK_RADII[e] * scale for e in els])
        for i, (c, r) in enumerate(zip(xyz, R)):
            n = max(8, int(4 * np.pi * r * r * density))
            k = np.arange(n) + 0.5
            phi = np.arccos(1 - 2 * k / n); th = np.pi * (1 + 5 ** 0.5) * k
            s = c + r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
            d = np.linalg.norm(s[:, None, :] - xyz[None, :, :], axis=-1)
            ok = np.all(d >= R[None, :] - 1e-6, axis=1)
            pts.append(s[ok])
    return np.concatenate(pts)


def resp_fit(xyz, pts, esp, fixed, groups, total, restrained, a=0.0005, b=0.1, iters=50):
    """Minimise |A q - esp|^2 + restraint, q = expand(free group charges) + fixed; sum q = total."""
    n = len(xyz)
    inv = 1.0 / np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=-1)       # 1/A
    inv *= BOHR                                                                    # -> 1/bohr (au)
    G = len(groups)
    M = np.zeros((n, G))
    for g, idx in enumerate(groups):
        M[idx, g] = 1.0
    q_fixed = np.zeros(n)
    for i, v in fixed.items():
        q_fixed[i] = v
    A_ = inv @ M; y = esp - inv @ q_fixed
    tot_free = total - q_fixed.sum()
    qg = np.zeros(G)
    for _ in range(iters):
        q_atoms = M @ qg
        w = np.zeros(G)
        for g, idx in enumerate(groups):
            if any(i in restrained for i in idx):
                w[g] = a * len(idx) / np.sqrt(qg[g] ** 2 + b * b)
        H = A_.T @ A_ + np.diag(w)
        K = np.zeros((G + 1, G + 1)); K[:G, :G] = H; K[:G, G] = M.sum(0); K[G, :G] = M.sum(0)
        rhs = np.concatenate([A_.T @ y, [tot_free]])
        new = np.linalg.solve(K, rhs)[:G]
        if np.max(np.abs(new - qg)) < 1e-7:
            qg = new; break
        qg = new
    q = M @ qg + q_fixed
    res = inv @ q - esp
    rrms = float(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(esp ** 2)))
    return q, rrms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xyz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--memory-gb", type=int, default=8)
    ap.add_argument("--relax", choices=("fix", "torsions"), default="torsions",
                    help="torsions: B-form torsions constrained, rest relaxed; fix: the original $fix block")
    a = ap.parse_args()
    import psi4
    psi4.set_num_threads(a.threads); psi4.set_memory(f"{a.memory_gb} GB")
    psi4.core.set_output_file(a.out.replace(".json", ".log"), False)
    lines = open(a.xyz).read().splitlines()[2:]
    els = [l.split()[0] for l in lines]; names = [l.split()[4] for l in lines]
    xyz = np.array([[float(t) for t in l.split()[1:4]] for l in lines])
    geo = "1 1\n" + "\n".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" for e, x in zip(els, xyz)) + "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1"
    # hydrogens relaxed with the heavy atoms fixed (GFN2-xTB; psi4's optking rejects frozen-atom
    # lists with indices above 9 in this version), then the HF/6-31G* potential
    import subprocess
    xtb = os.environ.get("XTB_BIN", "/apps/software/2024a/software/xtb/6.7.1-gfbf-2024a/bin/xtb")
    with open("in.xyz", "w") as f:
        f.write(f"{len(els)}\nC+\n" + "".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for e, x in zip(els, xyz)))
    heavy = ",".join(str(i + 1) for i, e in enumerate(els) if e != "H")
    if a.relax == "torsions":
        # B-form conformation held by its torsions, everything else relaxed (xTB 6.7.1 does not hold
        # the atoms of a $fix block: the original fit's heavy atoms moved 0.48 A; see DECISIONS)
        ix = {n: i + 1 for i, n in enumerate(names)}
        dih = [("O4'", "C1'", "N1", "C2"), ("O4'", "C1'", "C2'", "C3'"), ("C2'", "C3'", "C4'", "O4'"),
               ("O5'", "C5'", "C4'", "C3'"), ("C5'", "C4'", "C3'", "O3'"), ("C4'", "C3'", "O3'", "CM3"), ("C4'", "C5'", "O5'", "CM5")]
        open("xcontrol", "w").write("$constrain\n force constant=0.5\n" +
                                    "".join(f" dihedral: {ix[p]},{ix[q]},{ix[r]},{ix[t]},auto\n" for p, q, r, t in dih) + "$end\n")
    else:
        open("xcontrol", "w").write(f"$fix\n atoms: {heavy}\n$end\n")
    r = subprocess.run([xtb, "in.xyz", "--opt", "tight", "--chrg", "1", "--gfn", "2", "--input", "xcontrol"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists("xtbopt.xyz"):
        raise RuntimeError("xtb hydrogen relaxation failed: " + r.stdout[-1500:] + r.stderr[-500:])
    new_xyz = np.array([[float(t) for t in l.split()[1:4]] for l in open("xtbopt.xyz").read().splitlines()[2:2 + len(els)]])
    moved_heavy = float(np.abs(new_xyz[[i for i, e in enumerate(els) if e != "H"]] - xyz[[i for i, e in enumerate(els) if e != "H"]]).max())
    xyz = new_xyz
    geo = "1 1\n" + "\n".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" for e, x in zip(els, xyz)) + "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1"
    mol = psi4.geometry(geo)
    psi4.set_options({"basis": "6-31g*", "scf_type": "pk"})
    e, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
    pts = mk_points(xyz, els)
    np.savetxt("grid.dat", pts / BOHR if False else pts, fmt="%.8f")
    psi4.oeprop(wfn, "GRID_ESP")
    esp = np.loadtxt("grid_esp.dat")
    idx = {n: i for i, n in enumerate(names)}
    fixed = {idx[n]: v for n, v in SUGAR_FIXED.items()}
    free = [i for i in range(len(names)) if i not in fixed]
    eq = [["H41", "H42"], ["HM51", "HM52", "HM53"], ["HM31", "HM32", "HM33"]]
    grouped, groups = set(), []
    for g in eq:
        groups.append([idx[n] for n in g]); grouped.update(idx[n] for n in g)
    for i in free:
        if i not in grouped:
            groups.append([i])
    restrained = {i for i in free if els[i] != "H"}
    q, rrms = resp_fit(xyz, pts, esp, fixed, groups, 1.0, restrained)
    base = {n: round(float(q[idx[n]]), 4) for n in list(DC_BASE) + ["H3"]}
    s = sum(base.values())
    drift = round(1.0 + sum(DC_BASE.values()) - s, 4)
    base[max(base, key=lambda k: abs(base[k]))] = round(base[max(base, key=lambda k: abs(base[k]))] + drift, 4)
    out = {"residue": "DC+ (N3-protonated deoxycytidine)", "method": "RESP, HF/6-31G*, MK points, sugar fixed to AMBER OL15 DC",
           "n_points": int(len(pts)), "relative_rms": round(rrms, 4), "hf_energy_hartree": float(e),
           "heavy_atom_max_shift_in_h_relaxation_A": round(moved_heavy, 4), "relax": a.relax,
           "base_charges": base, "base_sum": round(sum(base.values()), 4),
           "dc_base_sum": round(sum(DC_BASE.values()), 4),
           "caps": {n: round(float(q[idx[n]]), 4) for n in ("CM5", "HM51", "CM3", "HM31")}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
