#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Where the electrostatics noise floor comes from: grid placement or the minimiser's path.

Follow-up to the composition gate (DECISIONS.md, 2026-11-21). The gate divides the
base-order variation of composition-matched fragments by a "solver noise" SD measured
from grid-jitter replicates of one sequence. Two facts make that denominator ambiguous:
the replicates' minimised structures differ from each other (surface radius changes in
33-62 percent of cells, base-pair origins by up to 0.03 A), and the minimiser stops on a
200-iteration cap, far from convergence, so its end point depends on the path.

This study separates the two sources on one sequence and one condition, for two
minimisers -- `capped` (200 iterations, OpenMM's default tolerance: the construction the
gate fragments used) and `converged` (tolerance 1 kJ/mol/nm, 20,000-iteration cap):

  grid       one minimised structure, solved at n grid placements (jitter 0.175 A)
  structure  n minimisations of the same raw build, each displaced by N(0, 0.001 A) --
             the PDB file's own precision, standing in for rounding differences between
             runs -- all solved on the identical grid

Channels are the gate's own (`composition_gate.channel_matrix`: per-fragment means over
the scored base pairs), so the SDs compare directly with the gate's `jitter_sd` column.
A projection of the gate under each minimiser is reported and labelled as one; the
binding test is the gate re-run on re-solved fragments.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import composition_gate as gate  # noqa: E402
import frames as frames_mod  # noqa: E402
import run_fragment as rf  # noqa: E402

JITTER_SEQUENCE = "AAAGAACTACATAATATCATGACATACTTTAATTTTAGTAGTTATTTACAGAGCGTAAAAA"
MINIMISERS = {"capped": (200, None), "converged": (20000, 1.0)}


def _coords(pdb: str) -> np.ndarray:
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
           for l in open(pdb) if l.startswith(("ATOM", "HETATM"))]
    return np.asarray(xyz)


def build(seq: str, minimiser: str, perturb_A: float, seed: int, keep: str) -> Dict[str, object]:
    it, tol = MINIMISERS[minimiser]
    work = tempfile.mkdtemp(prefix=f"ns_build_{minimiser}_", dir=keep)
    t = time.time()
    rep = rf.build_structure(seq, work, min_iterations=it, min_tolerance=tol,
                             perturb_A=perturb_A, perturb_seed=seed)
    return {"pdb": str(rep["pdb"]), "minimisation": rep.get("minimisation"),
            "seconds": round(time.time() - t, 1)}


def solve(seq: str, pdb: str, cond_name: str, grid_jitter: float, jitter_seed: int,
          workroot: str) -> Dict[str, object]:
    work = tempfile.mkdtemp(prefix="ns_solve_", dir=workroot)
    try:
        cond = apbs_run.CONDITIONS_BY_NAME[cond_name]
        bp, _ = frames_mod.bp_frames_from_pdb(pdb)
        sl = rf.central_slice(len(bp))
        t = time.time()
        out = rf.solve_conditions(pdb, work, [cond], bp, sl, grid_jitter=grid_jitter,
                                  jitter_seed=jitter_seed)
        return {"status": "ok", "seconds": round(time.time() - t, 1),
                "frag": {"potential": out["potential"], "counterion": out["counterion"],
                         "conditions": [cond_name], "angles": out["angles"],
                         "radii": np.asarray(apbs_run.RADIAL_OFFSETS_A),
                         "sequence": seq, "potential_surface": out["potential_surface"],
                         "surface_radius_A": out["surface_radius_A"],
                         "surface_offsets_A": out["surface_offsets_A"]},
                "coords": _coords(pdb), "bp_origins": bp[sl][:, :3, 3]}
    except Exception as exc:                                       # noqa: BLE001
        return {"status": "error", "reason": repr(exc)[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_and_solve(seq, minimiser, perturb_A, seed, cond_name, workroot):
    keep = tempfile.mkdtemp(prefix="ns_bs_", dir=workroot)
    try:
        b = build(seq, minimiser, perturb_A, seed, keep)
        r = solve(seq, b["pdb"], cond_name, 0.0, 0, workroot)
        r["minimisation"] = b["minimisation"]; r["build_seconds"] = b["seconds"]
        return r
    finally:
        shutil.rmtree(keep, ignore_errors=True)


def sector_of(name: str) -> str:
    parts = [p for p in name.split("|") if p.startswith("sector")]
    return parts[0] if parts else name.split("|")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", default=JITTER_SEQUENCE)
    ap.add_argument("--condition", default="NaCl_150mM")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--perturb-A", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--workroot", default="/tmp")
    ap.add_argument("--gate-channels", default=None,
                    help="composition_gate_channels.tsv, for the labelled projection")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.workroot, exist_ok=True)
    refdir = tempfile.mkdtemp(prefix="ns_ref_", dir=a.workroot)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        refs = {m: pool.submit(build, a.sequence, m, 0.0, 0, refdir) for m in MINIMISERS}
        refs = {m: f.result() for m, f in refs.items()}
        futs = {}
        for m in MINIMISERS:
            for j in range(a.n):
                futs[pool.submit(solve, a.sequence, refs[m]["pdb"], a.condition, 0.175, j,
                                 a.workroot)] = ("grid", m, j)
                futs[pool.submit(build_and_solve, a.sequence, m, a.perturb_A, 1000 + j,
                                 a.condition, a.workroot)] = ("structure", m, j)
        res = {}
        for f in cf.as_completed(futs):
            res[futs[f]] = f.result()
            print("done", futs[f], res[futs[f]].get("status"), res[futs[f]].get("seconds"),
                  flush=True)
    out = {"sequence": a.sequence, "condition": a.condition, "n": a.n,
           "perturb_A": a.perturb_A, "minimisers": {k: {"max_iterations": v[0],
           "tolerance_kJ_mol_nm": v[1]} for k, v in MINIMISERS.items()},
           "reference_builds": {m: {"seconds": r["seconds"], "minimisation": r["minimisation"]}
                                for m, r in refs.items()},
           "arms": {}, "elapsed_s": None}
    sds = {}
    for kind in ("grid", "structure"):
        for m in MINIMISERS:
            ok = [res[(kind, m, j)] for j in range(a.n) if res[(kind, m, j)]["status"] == "ok"]
            fails = [res[(kind, m, j)].get("reason") for j in range(a.n)
                     if res[(kind, m, j)]["status"] != "ok"]
            arm = {"n_ok": len(ok), "failures": fails}
            if len(ok) >= 3:
                frags = {f"r{i}": r["frag"] for i, r in enumerate(ok)}
                Y, names = gate.channel_matrix(frags, list(frags))
                sd = np.nanstd(Y, axis=0, ddof=1)
                sds[(kind, m)] = dict(zip(names, sd))
                by = {}
                for nme, v in zip(names, sd):
                    by.setdefault(sector_of(nme), []).append(v)
                arm["median_sd_by_sector"] = {k: round(float(np.median(v)), 5) for k, v in sorted(by.items())}
                arm["median_sd_all"] = round(float(np.median(sd)), 5)
                C = np.stack([r["coords"] for r in ok]); O = np.stack([r["bp_origins"] for r in ok])
                disp = np.linalg.norm(C - C[0][None], axis=-1)[1:]
                arm["structure_spread"] = {
                    "max_atom_displacement_A": round(float(disp.max()), 4),
                    "median_atom_displacement_A": round(float(np.median(disp)), 5),
                    "max_bp_origin_sd_A": round(float(np.std(O, axis=0).max()), 5)}
                if kind == "structure":
                    mins = [r["minimisation"] for r in ok]
                    arm["minimisation"] = {
                        "start_kept": {s: sum(x.get("start_kept") == s for x in mins) for s in ("plain", "canonical")},
                        "energy_final_kJ_mol": [round(x["energy_final_kJ_mol"], 2) for x in mins],
                        "max_force_kept_kJ_mol_nm": [round(x.get(f"max_force_{x['start_kept']}_kJ_mol_nm", float("nan")), 2) for x in mins],
                        "restrained_rmsd_A": [round(x["restrained_rmsd_angstrom"], 4) for x in mins],
                        "build_seconds": [r["build_seconds"] for r in ok]}
                arm["solve_seconds_median"] = float(np.median([r["seconds"] for r in ok]))
            out["arms"][f"{kind}/{m}"] = arm
    if a.gate_channels and all(k in sds for k in (("grid", "capped"), ("structure", "capped"),
                                                   ("grid", "converged"), ("structure", "converged"))):
        rows = list(csv.DictReader(open(a.gate_channels), delimiter="\t"))
        proj = {}
        for m in MINIMISERS:
            snr, used = [], 0
            for r in rows:
                n = r["channel"]
                if n not in sds[("grid", m)]:
                    continue
                noise_old2 = sds[("grid", "capped")][n] ** 2 + sds[("structure", "capped")][n] ** 2
                noise_new = np.sqrt(sds[("grid", m)][n] ** 2 + sds[("structure", m)][n] ** 2)
                sig = np.sqrt(max(float(r["within_var"]) - noise_old2, 0.0))
                if noise_new > 0:
                    snr.append(sig / noise_new); used += 1
            snr = np.asarray(snr)
            proj[m] = {"n_channels": used, "fraction_snr_above_5": round(float(np.mean(snr > 5)), 4),
                       "median_snr": round(float(np.median(snr)), 3)}
        out["projection_NOT_A_TEST"] = {
            "note": ("signal = within-group variance of the gate fragments minus this study's "
                     "capped-construction noise; noise = this study's grid and structure "
                     "noise for each minimiser. Planning only; the gate is re-run on re-solved "
                     "fragments."), **proj}
    out["elapsed_s"] = round(time.time() - t0, 1)
    shutil.rmtree(refdir, ignore_errors=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1)[:3000])
    for k, v in out["arms"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk in ("n_ok", "median_sd_all", "median_sd_by_sector", "structure_spread", "failures")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
