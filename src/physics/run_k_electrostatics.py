#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family K electrostatics: APBS on the A-form RNA duplex and both DNA-RNA hybrids of a fragment.

PLAN.md, Family K: "electrostatics solved on A-form conformations". For each fragment and each
class Family K already describes -- `rna` (RNA duplex of x), `hybp` (DNA strand x, RNA partner),
`hybm` (DNA strand rc(x)) -- the capped molecule Family K uses for its cgNA+ geometry (G-C caps,
the only ends the cgNA+ RNA and hybrid sets were trained with) is built in all-atom detail
(`rna_atomic`), minimised exactly as Family A's structures are (restrained, vacuum, two starts,
converged to 1 kJ/mol/nm; AMBER RNA.OL3, plus DNA.OL15 for hybrids), and solved at 150 mM NaCl
with Family A's production construction (`run_fragment.solve_conditions`: exclusion radius 2.0
A, exact 0.35 A fine spacing, extended extraction, per-atom and map-based solvation). The central
21 bp are scored; `hybm` stays in the frame of rc(x), as Family K's tie expects. One .npz per
fragment and class; resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apbs_run as A  # noqa: E402
import cgna_atomic as ca  # noqa: E402
import duplex_classes as K  # noqa: E402
import frames as F  # noqa: E402
import minimize as MZ  # noqa: E402
import rna_atomic as R  # noqa: E402
import run_fragment as rf  # noqa: E402
from channels import reverse_complement  # noqa: E402

CLASSES = ("rna", "hybp", "hybm")
FORCE_FIELDS = {"rna": ["amber14/RNA.OL3.xml"],
                "hybp": ["amber14/DNA.OL15.xml", "amber14/RNA.OL3.xml"],
                "hybm": ["amber14/DNA.OL15.xml", "amber14/RNA.OL3.xml"]}
CONDITION = "NaCl_150mM"


def solve_class(seq: str, cls: str, work: str, min_iterations: int = 20000,
                min_tolerance: float = 1.0) -> dict:
    s = reverse_complement(seq.upper()) if cls == "hybm" else seq.upper()
    capped = K.CAP + s + K.CAP
    atoms = (R.build_rna_duplex(capped, K.SETS["rna"]) if cls == "rna"
             else R.build_hybrid(capped, K.SETS["hyb"]))
    raw = ca.write_pdb(atoms, os.path.join(work, "raw.pdb"))
    mn = os.path.join(work, "min.pdb")
    rep = MZ.restrained_minimise(raw, mn, max_iterations=min_iterations, tolerance=min_tolerance,
                                 force_fields=FORCE_FIELDS[cls])
    bp, _ = F.bp_frames_from_pdb(mn)
    sc = rf.central_slice(len(bp))
    out = rf.solve_conditions(mn, work, [A.PRODUCTION_CONDITIONS_BY_NAME[CONDITION]], bp, sc,
                              extended=True, reference_solve=True, exact_fine_spacing=True)
    out["n_bp_solute"] = len(bp)
    out["minimisation"] = {k: v for k, v in rep.items() if isinstance(v, (int, float, str))}
    out["bp_origins"] = bp[sc][:, :3, 3]
    out["molecule"] = capped
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True, help="TSV with frag_id and sequence")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default=None)
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.library), delimiter="\t"))[int(lo): int(hi) + 1]
    n_ok = n_todo = 0
    for r in rows:
        fid, seq = r["frag_id"], r["sequence"]
        for cls in a.classes:
            out_npz = os.path.join(a.out, f"{fid}__{cls}.npz")
            if os.path.exists(out_npz):
                continue
            n_todo += 1
            work = tempfile.mkdtemp(prefix=f"ke_{cls}_", dir=a.workroot)
            t = time.time()
            rec = {"frag_id": fid, "class": cls}
            try:
                o = solve_class(seq, cls, work)
                np.savez_compressed(
                    out_npz, potential_surface=o["potential_surface"], surface_radius_A=o["surface_radius_A"],
                    surface_offsets_A=o["surface_offsets_A"], conditions=np.asarray(o["conditions"]),
                    sequence=np.asarray([seq]), molecule=np.asarray([o["molecule"]]),
                    bp_origins=o["bp_origins"],
                    **{k: v for k, v in o.items() if k.startswith("ext_") or k == "elec_energy_ref_kJ_mol"})
                rec.update(status="ok", seconds=round(time.time() - t, 1),
                           failed_conditions=o.get("failed_conditions", {}),
                           energy_skipped_conditions=o.get("energy_skipped_conditions", []),
                           minimisation=o["minimisation"], n_bp_solute=o["n_bp_solute"])
                n_ok += 1
            except Exception as exc:                                 # noqa: BLE001
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                           traceback=traceback.format_exc()[-2000:])
            json.dump(rec, open(out_npz.replace(".npz", ".json"), "w"), indent=1, default=str)
            print(f"{fid} {cls} {rec['status']} {round(time.time() - t)}s"
                  + ("" if rec["status"] == "ok" else f" :: {rec['error'][:200]}"), flush=True)
            shutil.rmtree(work, ignore_errors=True)
    print(f"{n_ok}/{n_todo} class solves done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
