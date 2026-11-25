#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The context convergence study `PLAN.md` revision 2 mandates.

The plan's words: "Take about 2,000 fragments, solve each at 121, 61, 41, 31 and 21 bp,
and plot how the central-window potential converges with flank length at every ionic
condition. The estimates above come from linearized theory, and DNA sits in the nonlinear
counterion condensation regime where that theory is only a guide, so the empirical
convergence curve is what sets the production lengths."

This settles two separate decisions at once, which is why it is on the critical path.

**The solver budget.** Flank length per condition. The plan's linearised estimate is
about 36 bp of flank at 10 mM, 16 at 50, 11 at 100, 9 at 150 and 7 at 300, for roughly a
2 percent residual. Those are predictions from a theory known to be only a guide here,
so the production lengths come from this curve instead.

**The architecture.** Revision 2 states that fragment-level labels become sliding-window
tracks at inference, and that "that extrapolation is only valid where the quantity is
local relative to its window, which is exactly what the convergence study in the ground
truth section measures, so the two decisions are made together rather than separately."
A channel that has not converged by the production flank is a channel whose genome-wide
track is not trustworthy, whatever its validation correlation says.

Method. Every fragment is built once at the longest length, so the central 21 bp keep
exactly the conformation they had in full context. Shortening the solute therefore
changes the electrostatic boundary and nothing else. The central 21 bp are scored at
every length, and every length is compared to the longest converged one, channel by
channel, normalised by the spread of the channel across the fragment set. A single
energy number would hide exactly the per-channel structure PANGEA is trained on.

Reported per condition: the discrepancy curve against flank length, and the shortest
flank whose discrepancy falls below the tolerance. Reported per channel group: which
channels converge last, since those are the ones whose window-ladder tracks are at risk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import frames as frames_mod  # noqa: E402
import run_fragment  # noqa: E402
import solver_probe  # noqa: E402


def fixed_grid_sequence(seq: str, lengths: List[int], cond: apbs_run.IonicCondition,
                        workroot: str, fine_spacing: float = 0.35,
                        timeout_s: int = 3600) -> Dict[int, dict]:
    """Every solute length of one sequence on ONE conformation and ONE grid.

    The first protocol (`channels_for`) rebuilt and re-minimised each length with the
    200-iteration minimiser and sized each length's grid from its own solute, so a length
    comparison also compared minimiser end points (median channel SD 0.0046 kT/e from
    path dependence alone) and fine spacings (0.31 A against 0.34 A along the axis at
    61 bp). Here the duplex is built and minimised once, converged, at the longest length;
    every shorter solute is that structure trimmed (`run_fragment.trim_pdb`); and every
    length is solved with the reference solute's coarse box, fine box and grid dimensions
    at exactly `fine_spacing`. The only thing that changes between lengths is which atoms
    are present -- the flanks this study is about.

    Channels: the axis-referenced shells (buried points NaN) and the surface-referenced
    shells, concatenated on the last axis -> (scored bp, sector, 6 radii + 6 offsets).
    """
    work = tempfile.mkdtemp(prefix="convfix_", dir=workroot)
    out: Dict[int, dict] = {}
    try:
        built = run_fragment.build_structure(seq, work, min_iterations=20000, min_tolerance=1.0)
        full = str(built["pdb"])
        bp, _ = frames_mod.bp_frames_from_pdb(full)
        n = len(bp)
        sc = run_fragment.central_slice(n)
        lam = apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
        tpad = max(run_fragment.MIN_TRANSVERSE_PAD_A, run_fragment.N_DEBYE_PAD * lam)
        grid = None
        for L in sorted(lengths, reverse=True):
            t0 = time.time()
            try:
                pdb = full if L >= n else run_fragment.trim_pdb(full, os.path.join(work, f"L{L}.pdb"), n, L)
                pqr = os.path.join(work, f"L{L}.pqr")
                apbs_run.run_pdb2pqr(pdb, pqr, log=os.path.join(work, f"L{L}_pdb2pqr.log"))
                apbs_run.sanitise_pqr(pqr)
                xyz, q, r, recs = apbs_run.read_pqr(pqr)
                if grid is None:                     # the reference (longest) solute sets the grid
                    extent = xyz.max(axis=0) - xyz.min(axis=0)
                    long_axis = int(np.argmax(extent))
                    pad = np.full(3, tpad); pad[long_axis] = run_fragment.AXIAL_PAD_A
                    coarse = extent + 2.0 * pad
                    pts = bp[sc][:, :3, 3]
                    centre = pts.mean(axis=0)
                    fine = np.full(3, 2.0 * (max(apbs_run.RADIAL_OFFSETS_A) + 6.0))
                    fine[long_axis] = (pts.max(axis=0) - pts.min(axis=0))[long_axis] + 20.0
                    grid = (coarse, np.minimum(fine, coarse), centre)
                coarse, fine, centre = grid
                name = f"{cond.name}_L{L}"
                inp = apbs_run.write_apbs_input(os.path.basename(pqr), name, work, cond, centre,
                                                coarse, fine, fine_spacing_A=fine_spacing,
                                                exact_fine_spacing=True)
                try:
                    apbs_run.run_apbs(inp, work, timeout_s=timeout_s)
                except RuntimeError as exc:
                    msg = str(exc)
                    out[L] = {"status": "failed", "seconds": round(time.time() - t0, 1),
                              "reason": "overflow/abort" if "large potential values" in msg
                              else msg.split(":")[0][:60]}
                    continue
                pot = apbs_run.DXMap.read(os.path.join(work, name + "_pot.dx"))
                ax = apbs_run.sample_potential(pot, bp[sc], atom_xyz=xyz, atom_radii=r)["potential"]
                sf = apbs_run.sample_potential_surface(pot, bp[sc], xyz, r)["potential_surface"]
                out[L] = {"status": "ok", "seconds": round(time.time() - t0, 1),
                          "channels": np.concatenate([ax, sf], axis=-1)}
            except Exception as exc:                                  # noqa: BLE001
                out[L] = {"status": "error", "reason": repr(exc)[:200]}
            finally:
                for suffix in ("_pot.dx", "_ndens.dx"):
                    pth = os.path.join(work, f"{cond.name}_L{L}{suffix}")
                    if os.path.exists(pth):
                        os.remove(pth)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)

SCORED_BP = 21
# Tolerance in units of the channel's own standard deviation across the fragment set.
# Calibrated against decisions already made in this project: coarse padding was accepted
# at 0.0005, a 0.5 A grid was rejected at 0.305, lpbe was rejected at 0.5 to 0.76.
TOLERANCE_RMSE_OVER_SD = 0.05


def channels_for(seq: str, solute_bp: int, cond: apbs_run.IonicCondition,
                 workroot: str, fine_spacing: float = 0.35,
                 timeout_s: int = 3600) -> Optional[Dict[str, object]]:
    """Solve one (sequence, solute length, condition) and return the scored channels."""
    work = tempfile.mkdtemp(prefix=f"conv_{solute_bp}_", dir=workroot)
    try:
        prep = solver_probe.prepare_structure(seq, solute_bp, work)
        bp = prep["bp"]
        n_bp = int(prep["n_bp"])
        # Score the central SCORED_BP of the solute, which is the central
        # SCORED_BP of the original full-length fragment by construction.
        lo = (n_bp - SCORED_BP) // 2
        scored = slice(lo, lo + SCORED_BP)

        lam = apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
        tpad = max(run_fragment.MIN_TRANSVERSE_PAD_A, run_fragment.N_DEBYE_PAD * lam)

        t0 = time.time()
        rec = solver_probe.run_config(
            prep["pdb"], bp, scored, cond, fine_spacing=fine_spacing,
            transverse_pad=tpad, axial_pad=run_fragment.AXIAL_PAD_A,
            equation="npbe", workdir=work, timeout_s=timeout_s,
        )
        if rec.get("status") != "ok" or "_channels" not in rec:
            return {"status": rec.get("status", "failed"),
                    "reason": rec.get("reason"),
                    "seconds": round(time.time() - t0, 1),
                    "solute_bp": n_bp}
        ch = np.asarray(rec["_channels"], dtype=float)   # (scored, sector, radius)
        return {
            "status": "ok",
            "channels": ch,
            "seconds": round(time.time() - t0, 1),
            "solute_bp": n_bp,
            "grid_points": rec.get("grid_points"),
            "finite_fraction": rec.get("potential_finite_fraction"),
        }
    except Exception as exc:                                  # noqa: BLE001
        return {"status": "error", "reason": repr(exc)[:200], "solute_bp": solute_bp}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarise(by_len: Dict[int, List[np.ndarray]], lengths: List[int],
              ref_len: int) -> Dict[str, object]:
    """Compare every length to the reference, normalised by across-fragment spread."""
    if ref_len not in by_len or not by_len[ref_len]:
        return {"error": f"reference length {ref_len} produced no solutions"}
    ref = np.stack(by_len[ref_len])                   # (n_frag, scored, sector, radius)
    # Spread of each channel across the fragment set, at the reference length.
    sd = np.nanstd(ref, axis=0)
    sd_safe = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.nan)

    out: Dict[str, object] = {}
    for n in lengths:
        if n == ref_len or n not in by_len or not by_len[n]:
            continue
        cur = np.stack(by_len[n])
        k = min(len(cur), len(ref))
        diff = cur[:k] - ref[:k]
        m = np.isfinite(diff) & np.isfinite(sd_safe)[None, ...]
        if m.sum() < 10:
            continue
        rmse = np.sqrt(np.nanmean(diff[m] ** 2))
        norm = diff / sd_safe[None, ...]
        rmse_over_sd = float(np.sqrt(np.nanmean(norm[np.isfinite(norm)] ** 2)))
        # Per-radius, since the outer shells are where the flank matters most.
        per_radius = []
        for r in range(ref.shape[-1]):
            dr = norm[..., r]
            dr = dr[np.isfinite(dr)]
            per_radius.append(round(float(np.sqrt(np.mean(dr ** 2))), 4)
                              if dr.size else None)
        out[str(n)] = {
            "n_fragments": int(k),
            "flank_bp": (n - SCORED_BP) // 2,
            "rmse_kT_e": round(float(rmse), 4),
            "rmse_over_sd": round(rmse_over_sd, 4),
            "rmse_over_sd_by_radius": per_radius,
            "within_tolerance": bool(rmse_over_sd <= TOLERANCE_RMSE_OVER_SD),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sequences", type=int, default=24,
                    help="PLAN.md asks for ~2000; the pilot sizes the full run")
    ap.add_argument("--lengths", type=int, nargs="+", default=[121, 61, 41, 31, 21])
    ap.add_argument("--conditions", nargs="+",
                    default=["NaCl_010mM", "NaCl_050mM", "NaCl_100mM",
                             "NaCl_150mM", "NaCl_250mM"])
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default="/tmp")
    ap.add_argument("--fixed-grid", action="store_true",
                    help="one converged conformation and one grid per sequence across lengths "
                         "(fixed_grid_sequence); the protocol all production lengths come from")
    ap.add_argument("--production-conditions", action="store_true",
                    help="use apbs_run.PRODUCTION_CONDITIONS (counterion accessibility radius)")
    ap.add_argument("--save-raw", default=None,
                    help="also write every solved cell's central-window channel vector "
                         "(.npz), so chunks of the plan-scale study (different --seed) can "
                         "be pooled by context_convergence_combine.py")
    ap.add_argument("--workers", type=int, default=1,
                    help="solve this many (sequence, length, condition) cells at once. "
                         "Every cell is an independent APBS subprocess, so this scales "
                         "with allocated cores; APBS itself does not use threads, which "
                         "was measured rather than assumed.")
    a = ap.parse_args()

    lengths = sorted(a.lengths, reverse=True)
    ref_len = lengths[0]
    rng = np.random.default_rng(a.seed)
    seqs = ["".join(rng.choice(list("ACGT"), size=ref_len))
            for _ in range(a.n_sequences)]

    per_condition: Dict[str, object] = {}
    failures: List[dict] = []
    t_start = time.time()
    raw: Dict[str, np.ndarray] = {}

    table = (apbs_run.PRODUCTION_CONDITIONS_BY_NAME if a.production_conditions
             else apbs_run.CONDITIONS_BY_NAME)
    for cname in a.conditions:
        cond = table[cname]
        by_len: Dict[int, List[np.ndarray]] = {n: [] for n in lengths}
        timing: Dict[int, List[float]] = {n: [] for n in lengths}
        print(f"\n=== {cname} (lambda_D = "
              f"{apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K):.1f} A) ===",
              flush=True)

        # Every (sequence, length) cell is an independent solve, so the grid is run as a
        # flat task list rather than a nested loop. With the corrected grid a 121 bp
        # solve takes about 121 s against 22 s at 21 bp, so the serial version of this
        # study no longer fits in a job's walltime: the reference length dominates and a
        # 24-sequence pilot over five conditions would be most of a day.
        cells = [(si, n) for si in range(len(seqs)) for n in lengths]
        results: Dict[Tuple[int, int], Optional[dict]] = {}
        if a.fixed_grid:
            import concurrent.futures as cf
            with cf.ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
                futs = {pool.submit(fixed_grid_sequence, seqs[si], lengths, cond, a.workroot): si
                        for si in range(len(seqs))}
                for fut in cf.as_completed(futs):
                    si = futs[fut]
                    try:
                        per_len = fut.result()
                    except Exception as exc:                        # noqa: BLE001
                        per_len = {n: {"status": "error", "reason": repr(exc)[:200]}
                                   for n in lengths}
                    for n in lengths:
                        results[(si, n)] = per_len.get(n, {"status": "error", "reason": "missing"})
        elif a.workers > 1:
            import concurrent.futures as cf
            with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
                futs = {pool.submit(channels_for, seqs[si], n, cond, a.workroot):
                        (si, n) for si, n in cells}
                for fut in cf.as_completed(futs):
                    key = futs[fut]
                    try:
                        results[key] = fut.result()
                    except Exception as exc:                        # noqa: BLE001
                        results[key] = {"status": "failed",
                                        "reason": f"{type(exc).__name__}: {exc}"[:200]}
        else:
            for si, n in cells:
                results[(si, n)] = channels_for(seqs[si], n, cond, a.workroot)

        # Report in sequence order regardless of completion order, so the log stays
        # readable and comparable between a serial and a parallel run.
        for si in range(len(seqs)):
            row = []
            for n in lengths:
                res = results.get((si, n))
                if res and res.get("status") == "ok":
                    by_len[n].append(res["channels"])
                    timing[n].append(res["seconds"])
                    row.append(f"{n}:{res['seconds']:.0f}s")
                else:
                    failures.append({"condition": cname, "seq": si, "solute_bp": n,
                                     **{k: v for k, v in (res or {}).items()
                                        if k != "channels"}})
                    row.append(f"{n}:FAIL")
            print(f"  seq {si:3d}  " + "  ".join(row), flush=True)
        if a.save_raw:
            for n in lengths:
                for si in range(len(seqs)):
                    res = results.get((si, n))
                    if res and res.get("status") == "ok":
                        raw[f"{cname}__{n}__{si}"] = np.asarray(res["channels"], dtype=np.float64)
            os.makedirs(os.path.dirname(os.path.abspath(a.save_raw)), exist_ok=True)
            np.savez_compressed(a.save_raw, sequences=np.asarray(seqs), seed=a.seed, **raw)

        per_condition[cname] = {
            "ionic_strength_M": cond.ionic_strength_M,
            "debye_length_A": round(
                apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K), 2),
            "plan_flank_bp": {"NaCl_010mM": 36, "NaCl_050mM": 16, "NaCl_100mM": 11,
                              "NaCl_150mM": 9, "NaCl_250mM": 7}.get(cname),
            "n_solved_by_length": {str(n): len(by_len[n]) for n in lengths},
            "median_seconds_by_length": {
                str(n): (round(float(np.median(timing[n])), 1) if timing[n] else None)
                for n in lengths},
            "convergence": summarise(by_len, lengths, ref_len),
        }
        _dump(per_condition, failures, a, ref_len, t_start)

    _dump(per_condition, failures, a, ref_len, t_start, final=True)

    print("\n=== converged flank by condition (tolerance "
          f"{TOLERANCE_RMSE_OVER_SD} sd) ===")
    for cname, d in per_condition.items():
        conv = d.get("convergence", {})
        ok = [int(k) for k, v in conv.items()
              if isinstance(v, dict) and v.get("within_tolerance")]
        shortest = min(ok) if ok else None
        plan = d.get("plan_flank_bp")
        print(f"  {cname:12s} plan flank={plan!s:>4s} bp   "
              f"measured shortest converged solute="
              f"{shortest if shortest else 'none of those tried':>8}"
              + (f" (flank {(shortest - SCORED_BP)//2} bp)" if shortest else ""))
    print(f"\n-> {a.out}")
    return 0


def _dump(per_condition, failures, a, ref_len, t_start, final: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    payload = {
        "n_sequences": a.n_sequences,
        "lengths_bp": a.lengths,
        "reference_length_bp": ref_len,
        "scored_bp": SCORED_BP,
        "tolerance_rmse_over_sd": TOLERANCE_RMSE_OVER_SD,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "per_condition": per_condition,
        "failures": failures,
        "note": (
            "Pilot scale. PLAN.md revision 2 asks for about 2,000 fragments; this run "
            "sizes the full one and shows whether the linearised flank estimates hold "
            "in the nonlinear regime."
        ),
        "complete": final,
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
