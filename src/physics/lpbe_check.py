#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Measure the linearisation error: lpbe against npbe, channel by channel, across salt.

`PLAN.md` specifies the nonlinear Poisson-Boltzmann equation. The nonlinear solver
diverges at 500 mM for a 61 bp duplex regardless of box size or grid spacing, while the
linearised solver converges there (results/highsalt_probe.json). Before substituting
lpbe at high salt, the substitution has to be justified by measurement rather than by
the plausible-sounding argument that screening makes linearisation safe.

The measurement: on every condition where **both** equations converge, compute the same
channels twice and report the discrepancy, normalised by the spread of the channel
itself. If the argument is right, the discrepancy falls monotonically with ionic
strength, and its value at the highest convergent condition bounds the error being
accepted at the conditions above it.

If the discrepancy does not fall with salt, the substitution is not justified and the
high-salt conditions are dropped from the plan instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import cgna_atomic as ca  # noqa: E402
import frames as frames_mod  # noqa: E402
import minimize as minimise_mod  # noqa: E402
import run_fragment  # noqa: E402


def solve_one(pqr_name: str, workdir: str, cond: apbs_run.IonicCondition,
              centre, coarse, fine, bp, scored, xyz, radii, equation: str,
              timeout_s: int = 1800) -> Optional[Dict[str, np.ndarray]]:
    name = f"{cond.name}_{equation}"
    path = apbs_run.write_apbs_input(
        pqr_name, name, workdir, cond, centre, coarse, fine,
        fine_spacing_A=run_fragment.FINE_SPACING_A,
    )
    if equation == "lpbe":
        txt = open(path).read().replace("    npbe\n", "    lpbe\n")
        open(path, "w").write(txt)
    t0 = time.time()
    try:
        apbs_run.run_apbs(os.path.basename(path), workdir, timeout_s=timeout_s)
    except Exception:
        return None
    pot_path = os.path.join(workdir, name + "_pot.dx")
    if not os.path.exists(pot_path):
        return None
    pot = apbs_run.DXMap.read(pot_path)
    axis = apbs_run.sample_potential(pot, bp[scored], atom_xyz=xyz, atom_radii=radii)
    surf = apbs_run.sample_potential_surface(pot, bp[scored], xyz, radii)
    for suffix in ("_pot.dx", "_ndens.dx"):
        p = os.path.join(workdir, name + suffix)
        if os.path.exists(p):
            os.remove(p)
    return {
        "surface": surf["potential_surface"],
        "axis": axis["potential"],
        "seconds": time.time() - t0,
    }


def compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return {}
    av, bv = a[m], b[m]
    diff = av - bv
    sd = float(np.std(bv)) or 1.0
    return {
        "n": int(m.sum()),
        "rmse_kT_e": float(np.sqrt((diff**2).mean())),
        "rmse_over_sd": float(np.sqrt((diff**2).mean()) / sd),
        "max_abs_diff_kT_e": float(np.abs(diff).max()),
        "mean_signed_diff_kT_e": float(diff.mean()),
        "pearson_r": float(np.corrcoef(av, bv)[0, 1]),
        "npbe_mean_kT_e": float(np.mean(bv)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--n-sequences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default="/tmp")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seqs = ["".join(rng.choice(list("ACGT"), size=61)) for _ in range(a.n_sequences)]
    conds = [apbs_run.CONDITIONS_BY_NAME[c] for c in a.conditions]

    records: List[dict] = []
    for si, seq in enumerate(seqs):
        work = tempfile.mkdtemp(prefix=f"lpbe_{si}_", dir=a.workroot)
        try:
            atoms = ca.build_duplex(seq, ca.NucleotideTemplates(), ca.CGNAGroundState(seq))
            raw = os.path.join(work, "raw.pdb")
            ca.write_pdb(atoms, raw)
            minim = os.path.join(work, "min.pdb")
            minimise_mod.restrained_minimise(raw, minim)
            pqr = os.path.join(work, "frag.pqr")
            apbs_run.run_pdb2pqr(minim, pqr)
            apbs_run.sanitise_pqr(pqr)
            xyz, q, radii, recs = apbs_run.read_pqr(pqr)
            bp, _ = frames_mod.bp_frames_from_pdb(minim)
            scored = run_fragment.central_slice(len(bp))

            extent = xyz.max(axis=0) - xyz.min(axis=0)
            long_axis = int(np.argmax(extent))
            pts = bp[scored][:, :3, 3]
            centre = pts.mean(axis=0)
            margin = 2.0 * (max(apbs_run.RADIAL_OFFSETS_A) + 6.0)
            fine_base = pts.max(axis=0) - pts.min(axis=0) + margin

            for cond in conds:
                lam = apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
                tpad = max(run_fragment.MIN_TRANSVERSE_PAD_A,
                           run_fragment.N_DEBYE_PAD * lam)
                pad = np.full(3, tpad)
                pad[long_axis] = run_fragment.AXIAL_PAD_A
                coarse = extent + 2.0 * pad
                fine = np.minimum(fine_base, coarse)

                out_n = solve_one(os.path.basename(pqr), work, cond, centre, coarse,
                                  fine, bp, scored, xyz, radii, "npbe")
                out_l = solve_one(os.path.basename(pqr), work, cond, centre, coarse,
                                  fine, bp, scored, xyz, radii, "lpbe")
                rec = {
                    "sequence_index": si,
                    "condition": cond.name,
                    "ionic_strength_M": cond.ionic_strength_M,
                    "debye_length_A": round(lam, 2),
                    "npbe_converged": out_n is not None,
                    "lpbe_converged": out_l is not None,
                }
                if out_n is not None and out_l is not None:
                    rec["surface"] = compare(out_l["surface"], out_n["surface"])
                    rec["axis"] = compare(out_l["axis"], out_n["axis"])
                    rec["npbe_seconds"] = round(out_n["seconds"], 1)
                    rec["lpbe_seconds"] = round(out_l["seconds"], 1)
                records.append(rec)
                s = rec.get("surface", {})
                print(f"[seq {si}] {cond.name:22s} npbe={rec['npbe_converged']!s:5s} "
                      f"lpbe={rec['lpbe_converged']!s:5s} "
                      + (f"rmse/sd={s['rmse_over_sd']:.4f} r={s['pearson_r']:.5f}"
                         if s else ""), flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # Trend across salt, over sequences where both converged.
    by_cond: Dict[str, List[float]] = {}
    for r in records:
        s = r.get("surface")
        if s:
            by_cond.setdefault(r["condition"], []).append(s["rmse_over_sd"])
    trend = {c: {"mean_rmse_over_sd": float(np.mean(v)), "n": len(v)}
             for c, v in by_cond.items()}

    summary = {
        "n_sequences": len(seqs),
        "conditions": a.conditions,
        "per_condition_surface_rmse_over_sd": trend,
        "records": records,
        "interpretation": (
            "If the discrepancy falls monotonically with ionic strength, substituting "
            "lpbe above the highest npbe-convergent condition is justified, and the "
            "value at that condition bounds the error accepted. If it does not fall, "
            "the high-salt conditions are dropped instead."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nmean surface rmse/sd by condition:")
    for c in a.conditions:
        if c in trend:
            print(f"  {c:22s} {trend[c]['mean_rmse_over_sd']:.4f}  (n={trend[c]['n']})")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
