#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""pKa shifts of the purine ring nitrogens in a stacked step: gas-phase DFT plus Poisson-Boltzmann
solvation, the route that converges where continuum solvation inside the SCF does not (ddX and GePol
PCM both fail to converge on stacked steps; DECISIONS.md).

Thermodynamic cycle, per protonation site s of base k in context c (the same rigid geometries and
proton placement as `run_psi4.pka_context`, the proton along the lone pair at 1.01 A):

    dpKa = -[ (E_g(step+H) - E_g(step) + dG_solv(step+H) - dG_solv(step))
             - (E_g(base+H) - E_g(base) + dG_solv(base+H) - dG_solv(base)) ] / (2.303 RT),

E_g the gas-phase DFT energy (no discretisation), dG_solv the electrostatic solvation free energy of
the species' electrostatic-potential charges (Merz-Kollman points fitted to the gas-phase DFT
potential, light hyperbolic restraint) by finite-difference PB: solute dielectric 1 (the charges are
the gas-phase molecule's own), water 78.54, no ions (pure water, as the PCM definition), Bondi radii,
1.4 A probe molecular surface. Each species is solved at three fine spacings (0.30, 0.20, 0.15 A,
focused) and the finest is reported with the spread as its convergence error.
Known approximation: the solute is not repolarised by the solvent (a self-consistent reaction field
would add that); it enters both legs of the cycle and largely cancels in the double difference.

Stage `qc` (psi4 environment) writes each species' energy, charges and geometry; stage `pb` (physics
environment, APBS) adds the solvation energies and the shifts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80}
SPACINGS = (0.30, 0.20, 0.15)
KCAL_PER_KJ = 1 / 4.184
TWO_PT3_RT_KCAL = 2.302585 * 0.0019872043 * 298.15


def _parse(lines):
    els = [l.split()[0] for l in lines]
    xyz = np.array([[float(t) for t in l.split()[1:4]] for l in lines])
    return els, xyz


def qc_stage(index: str, contexts, out: str, method: str, basis: str, threads: int, mem_gb: int) -> None:
    import psi4
    import run_psi4 as R
    import resp_cplus as RC
    psi4.set_num_threads(threads); psi4.set_memory(f"{mem_gb} GB")
    idx = {c["context"]: c for c in json.load(open(index))["contexts"]}
    order = json.load(open(os.path.join(HERE, "data", "base_atom_order.json")))["order"]
    for ctx in contexts:
        f_out = os.path.join(out, f"{ctx}_qc.json")
        if os.path.exists(f_out):
            continue
        psi4.core.set_output_file(os.path.join(out, f"{ctx}_qc.log"), False)
        files = idx[ctx]["files"]
        w1, w2 = ctx[1], ctx[2]
        letters = {"w1": w1, "c1": R.COMP[w1], "w2": w2, "c2": R.COMP[w2]}
        tags = ("w1", "c1", "w2", "c2")
        xyz = {k: R.read_xyz(files[k]) for k in tags}
        species = {"step": (xyz["w1"] + xyz["c1"] + xyz["w2"] + xyz["c2"], 0)}
        sites = []
        for k in tags:
            names = order[letters[k]]
            for site, nb in R.PKA_SITES[letters[k]]:
                base_h = R._protonate(xyz[k], names, site, nb)
                parts = {t: (base_h if t == k else xyz[t]) for t in tags}
                tag = f"{letters[k]}{site}_{k}"
                species[f"base_{k}"] = (xyz[k], 0)
                species[f"base_{k}_H_{site}"] = (base_h, 1)
                species[f"step_H_{tag}"] = (parts["w1"] + parts["c1"] + parts["w2"] + parts["c2"], 1)
                sites.append({"tag": tag, "k": k, "site": site, "bp": "bp1" if k in ("w1", "c1") else "bp2"})
        rec = {"context": ctx, "method": f"{method}/{basis}", "sites": sites, "species": {}}
        psi4.set_options({"basis": basis, "scf_type": "df", "reference": "rks", "freeze_core": False})
        for name, (lines, charge) in species.items():
            els, X = _parse(lines)
            mol = R.molecule(psi4, [lines], charge=charge, mult=1)
            e, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
            w = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(w)
            pts = RC.mk_points(X, els); np.savetxt("grid.dat", pts, fmt="%.8f")
            psi4.oeprop(wfn, "GRID_ESP"); esp = np.loadtxt("grid_esp.dat"); os.chdir(cwd); shutil.rmtree(w, ignore_errors=True)
            groups = [[i] for i in range(len(els))]
            restrained = {i for i, e_ in enumerate(els) if e_ != "H"}
            q, rrms = RC.resp_fit(X, pts, esp, {}, groups, float(charge), restrained)
            rec["species"][name] = {"charge": charge, "E_hartree": float(e), "elements": els, "xyz": X.round(5).tolist(),
                                    "esp_charges": [round(float(v), 5) for v in q], "esp_relative_rms": round(rrms, 4)}
        json.dump(rec, open(f_out, "w"))
        print(ctx, "qc done:", len(species), "species", flush=True)


def _apbs_solv(els, X, q, work: str, h: float) -> float:
    """Electrostatic solvation free energy (kJ/mol) by APBS: water minus vacuum, solute dielectric 1."""
    pqr = os.path.join(work, "m.pqr")
    with open(pqr, "w") as f:
        for i, (e, x, qq) in enumerate(zip(els, X, q)):
            f.write(f"ATOM {i + 1:6d} {e:<4s} MOL A    1 {x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {qq:8.4f} {BONDI[e]:7.4f}\n")
    ext = X.max(0) - X.min(0)
    fine = ext + 2 * (max(BONDI.values()) + 1.4 + 4.0)
    coarse = 2.0 * fine + 20.0
    dime = [int(np.ceil(fl / h / 32.0)) * 32 + 1 for fl in fine]
    glen_f = [(d - 1) * h for d in dime]
    c = X.mean(0)
    txt = f"""read
  mol pqr m.pqr
end
"""
    for name, sdie in (("solv", 78.54), ("ref", 1.0)):
        txt += f"""elec name {name}
  mg-auto
  dime {dime[0]} {dime[1]} {dime[2]}
  cglen {coarse[0]:.2f} {coarse[1]:.2f} {coarse[2]:.2f}
  fglen {glen_f[0]:.2f} {glen_f[1]:.2f} {glen_f[2]:.2f}
  cgcent {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}
  fgcent {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}
  mol 1
  lpbe
  bcfl mdh
  pdie 1.0
  sdie {sdie}
  srfm smol
  chgm spl2
  sdens 10.0
  srad 1.4
  swin 0.3
  temp 298.15
  calcenergy total
  calcforce no
end
"""
    txt += "print elecEnergy solv - ref end\nquit\n"
    open(os.path.join(work, "in.apbs"), "w").write(txt)
    r = subprocess.run([shutil.which("apbs") or "apbs", "in.apbs"], cwd=work, capture_output=True, text=True, timeout=7200)
    for line in r.stdout.splitlines():
        if "Global net ELEC energy" in line:
            return float(line.split("=")[1].split()[0])
    raise RuntimeError("APBS gave no energy: " + r.stdout[-600:] + r.stderr[-300:])


def pb_stage(out: str, contexts) -> None:
    for ctx in contexts:
        f_qc = os.path.join(out, f"{ctx}_qc.json"); f_out = os.path.join(out, f"{ctx}_pka.json")
        if not os.path.exists(f_qc) or os.path.exists(f_out):
            continue
        rec = json.load(open(f_qc))
        solv = {}
        for name, sp in rec["species"].items():
            vals = []
            for h in SPACINGS:
                w = tempfile.mkdtemp()
                try:
                    vals.append(_apbs_solv(sp["elements"], np.array(sp["xyz"]), sp["esp_charges"], w, h) * KCAL_PER_KJ)
                finally:
                    shutil.rmtree(w, ignore_errors=True)
            solv[name] = {"by_spacing_kcal": dict(zip(map(str, SPACINGS), [round(v, 4) for v in vals])),
                          "value_kcal": vals[-1], "convergence_kcal": float(max(vals) - min(vals)),
                          "last_step_kcal": float(abs(vals[-1] - vals[-2]))}
        out_rec = {"context": ctx, "method": rec["method"] + " gas phase + APBS (pdie 1, sdie 78.54, Bondi, no ions)", "solvation": solv}
        H = 627.509474
        for s in rec["sites"]:
            k, tag, site = s["k"], s["tag"], s["site"]
            sp = rec["species"]
            d_step = (sp[f"step_H_{tag}"]["E_hartree"] - sp["step"]["E_hartree"]) * H + solv[f"step_H_{tag}"]["value_kcal"] - solv["step"]["value_kcal"]
            d_base = (sp[f"base_{k}_H_{site}"]["E_hartree"] - sp[f"base_{k}"]["E_hartree"]) * H + solv[f"base_{k}_H_{site}"]["value_kcal"] - solv[f"base_{k}"]["value_kcal"]
            key = f"dpka_{tag}_{s['bp']}"
            out_rec[key] = -(d_step - d_base) / TWO_PT3_RT_KCAL
            err = (solv[f"step_H_{tag}"]["last_step_kcal"] + solv["step"]["last_step_kcal"]
                   + solv[f"base_{k}_H_{site}"]["last_step_kcal"] + solv[f"base_{k}"]["last_step_kcal"])
            out_rec[key + "_grid_err"] = err / TWO_PT3_RT_KCAL
        json.dump(out_rec, open(f_out, "w"), indent=1)
        print(ctx, {k: round(v, 3) for k, v in out_rec.items() if k.startswith("dpka")}, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("qc", "pb"), required=True)
    ap.add_argument("--index", default="/home/bjc75/scratch/pangea/qc/tetramer/qc_fragment_index.json")
    ap.add_argument("--contexts", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="b3lyp-d3bj"); ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--memory-gb", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "qc":
        qc_stage(a.index, a.contexts, a.out, a.method, a.basis, a.threads, a.memory_gb)
    else:
        pb_stage(a.out, a.contexts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
