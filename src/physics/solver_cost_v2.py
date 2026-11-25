#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Measure what actually drives Poisson-Boltzmann cost, so the budget stops being a guess.

Three beliefs are tested here, because each of them has already been wrong once.

1. **Thread scaling.** Every CPU-hour figure quoted so far multiplied a wall time by the
   number of threads in the allocation, which assumes APBS uses them. That has never
   been measured. If APBS is effectively serial, every CPU-hour estimate in
   `DECISIONS.md` is too high by the thread count.

2. **Cost against solute length.** `PLAN.md` revision 2 sets fragment length from the
   Debye length and calls it "the lever that makes the solver budget tractable". But
   `write_apbs_input` derives `dime` from the *fine* box, and `mg-auto` uses one `dime`
   for both levels, so shortening the solute should change the coarse *spacing* and not
   the grid point count. If that reading is right, shorter fragments buy accuracy and
   convergence headroom rather than time, and the budget has to come from somewhere else.
   `results/solver_probe_v1.json` already hints at it (41 bp 36.6 s, 61 bp 34.9 s), but
   those were different sequences at one condition, so it is measured properly here.

3. **Cost against grid points.** If `dime` is the real driver, cost should be close to
   linear in grid points, and the fine box -- set by the 21 bp scored window and the 18 A
   radial reach, both fixed by the plan -- is what the production budget is paying for.

Output is one JSON record per configuration. No configuration is adopted here; this
measures, and `DECISIONS.md` records what is chosen from it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import run_fragment  # noqa: E402
import solver_probe  # noqa: E402


# PLAN.md revision 2: flank needed for ~2% residual contribution, by ionic strength.
# Solute length = 21 bp scored window + 2 x flank.
PLAN_FLANK_BP = {
    "NaCl_010mM": 36,
    "NaCl_050mM": 16,
    "NaCl_100mM": 11,
    "NaCl_150mM": 9,
    "NaCl_250mM": 7,     # plan quotes 7 bp at 300 mM; 250 mM is our highest production
}


def solute_bp_for(cond_name: str, scored_bp: int = 21) -> int:
    flank = PLAN_FLANK_BP.get(cond_name, 9)
    return scored_bp + 2 * flank


def time_one(seq: str, solute_bp: int, cond: apbs_run.IonicCondition, threads: int,
             fine_spacing: float, workroot: str, repeats: int = 1) -> Dict[str, object]:
    """Build, trim, solve. Returns the configuration and its wall time."""
    work = tempfile.mkdtemp(prefix=f"cost_{solute_bp}_{threads}_", dir=workroot)
    try:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        prep = solver_probe.prepare_structure(seq, solute_bp, work)
        bp = prep["bp"]
        scored = run_fragment.central_slice(prep["n_bp"])

        lam = apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
        tpad = max(run_fragment.MIN_TRANSVERSE_PAD_A, run_fragment.N_DEBYE_PAD * lam)

        times: List[float] = []
        rec: Dict[str, object] = {}
        for _ in range(repeats):
            t0 = time.time()
            rec = solver_probe.run_config(
                prep["pdb"], bp, scored, cond,
                fine_spacing=fine_spacing, transverse_pad=tpad,
                axial_pad=run_fragment.AXIAL_PAD_A, equation="npbe", workdir=work,
            )
            times.append(time.time() - t0)
        rec.pop("_channels", None)
        rec.update({
            "threads": threads,
            "solute_bp": int(prep["n_bp"]),
            "requested_solute_bp": solute_bp,
            "transverse_pad_A": round(tpad, 1),
            "debye_length_A": round(lam, 2),
            "wall_seconds_each": [round(t, 1) for t in times],
            "wall_seconds_median": round(float(np.median(times)), 1),
            "cpu_seconds_if_perfect_scaling": round(float(np.median(times)) * threads, 1),
        })
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--lengths", type=int, nargs="+", default=[21, 31, 41, 61, 93])
    ap.add_argument("--condition", default="NaCl_150mM")
    ap.add_argument("--plan-lengths", action="store_true",
                    help="also time each production condition at its plan-derived length")
    ap.add_argument("--fine-spacings", type=float, nargs="+", default=[0.35])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default="/tmp")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    base = "".join(rng.choice(list("ACGT"), size=max(a.lengths + [93])))
    cond = apbs_run.CONDITIONS_BY_NAME[a.condition]

    records: List[dict] = []

    # 1. Thread scaling, at the production configuration.
    print("=== thread scaling (61 bp, 0.35 A, %s) ===" % a.condition, flush=True)
    for t in a.threads:
        r = time_one(base, 61, cond, t, 0.35, a.workroot, repeats=a.repeats)
        r["sweep"] = "threads"
        records.append(r)
        print(f"  threads={t:2d}  wall={r['wall_seconds_median']:7.1f}s  "
              f"grid={r.get('grid_points')}  status={r.get('status')}", flush=True)
        _dump(records, a.out)

    # 2. Cost against solute length, threads held at 1.
    print("\n=== solute length at 1 thread ===", flush=True)
    for n in a.lengths:
        r = time_one(base, n, cond, 1, 0.35, a.workroot, repeats=a.repeats)
        r["sweep"] = "length"
        records.append(r)
        print(f"  solute={n:4d} bp  wall={r['wall_seconds_median']:7.1f}s  "
              f"grid={r.get('grid_points')}  coarse_spacing={r.get('coarse_spacing_A')}  "
              f"status={r.get('status')}", flush=True)
        _dump(records, a.out)

    # 3. Cost against grid points, via fine spacing.
    print("\n=== fine spacing (grid points) at 1 thread ===", flush=True)
    for s in a.fine_spacings:
        if s == 0.35:
            continue
        r = time_one(base, 61, cond, 1, s, a.workroot, repeats=a.repeats)
        r["sweep"] = "spacing"
        records.append(r)
        print(f"  spacing={s:.2f} A  wall={r['wall_seconds_median']:7.1f}s  "
              f"grid={r.get('grid_points')}  status={r.get('status')}", flush=True)
        _dump(records, a.out)

    # 4. Each production condition at its plan-derived solute length.
    if a.plan_lengths:
        print("\n=== production conditions at plan-derived lengths, 1 thread ===",
              flush=True)
        for name in PLAN_FLANK_BP:
            c = apbs_run.CONDITIONS_BY_NAME[name]
            n = solute_bp_for(name)
            seq = "".join(rng.choice(list("ACGT"), size=n))
            r = time_one(seq, n, c, 1, 0.35, a.workroot, repeats=1)
            r["sweep"] = "plan_length"
            records.append(r)
            print(f"  {name:12s} flank={PLAN_FLANK_BP[name]:3d}bp solute={n:4d}bp  "
                  f"wall={r['wall_seconds_median']:7.1f}s  status={r.get('status')}",
                  flush=True)
            _dump(records, a.out)

    _dump(records, a.out, final=True)
    print(f"\n-> {a.out}")
    return 0


def _dump(records: List[dict], out: str, final: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = {
        "records": records,
        "plan_flank_bp": PLAN_FLANK_BP,
        "question": (
            "Does APBS use the threads it is given; does solute length change cost "
            "under mg-auto with dime set from the fine box; and is cost linear in "
            "grid points?"
        ),
        "complete": final,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
