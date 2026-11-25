#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Pool the chunks of the plan-scale context convergence study and read off production lengths.

Each chunk (`context_convergence.py --seed S --save-raw chunk_S.npz`) holds, per ionic
condition, solute length and sequence, the central-21-bp potential channels. Cells are
paired **by sequence**, never by list position: `summarise` in `context_convergence.py`
pairs the k-th success at one length with the k-th at the reference, which silently
compares different sequences whenever a cell fails at one length and not the other (it
did not bite in the pilot, where every length either fully succeeded or fully failed).

Reference length per condition: the longest length that solved for at least 90 percent of
sequences -- 121 bp where it converges (low salt), otherwise the next (the nonlinear solver
overflows at 121 bp from 50 mM up; DECISIONS.md, 2026-11-19). Convergence is the RMSE of a
shorter solve against the reference, over the across-fragment SD of the reference,
overall and per radial shell; the production length is the shortest within tolerance.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

TOLERANCE = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="glob of chunk .npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-success", type=float, default=0.9)
    a = ap.parse_args()
    cells = defaultdict(dict)                     # (condition, length) -> {(seed, si): arr}
    n_seq = 0
    for f in sorted(glob.glob(a.chunks)):
        z = np.load(f, allow_pickle=True)
        seed = int(z["seed"]); n_seq += len(z["sequences"])
        for k in z.files:
            m = re.match(r"^(.+)__(\d+)__(\d+)$", k)
            if m:
                cells[(m.group(1), int(m.group(2)))][(seed, int(m.group(3)))] = z[k]
    out = {"n_sequences": n_seq, "tolerance_rmse_over_sd": TOLERANCE, "per_condition": {}}
    for cond in sorted({c for c, _ in cells}):
        lengths = sorted({n for c, n in cells if c == cond}, reverse=True)
        succ = {n: len(cells[(cond, n)]) / max(1, n_seq) for n in lengths}
        ref_len = next((n for n in lengths if succ[n] >= a.min_success), None)
        rec = {"success_fraction_by_length": {str(n): round(v, 4) for n, v in succ.items()},
               "reference_length_bp": ref_len, "by_length": {}}
        if ref_len is None:
            out["per_condition"][cond] = rec
            continue
        ref = cells[(cond, ref_len)]
        R = np.stack(list(ref.values()))
        sd = np.nanstd(R, axis=0); sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.nan)
        for n in lengths:
            if n >= ref_len:
                continue
            keys = sorted(set(cells[(cond, n)]) & set(ref))
            if len(keys) < 10:
                continue
            D = np.stack([cells[(cond, n)][k] - ref[k] for k in keys]) / sd[None]
            fin = np.isfinite(D)
            per_r = [float(np.sqrt(np.nanmean(D[..., r][fin[..., r]] ** 2))) for r in range(D.shape[-1])]
            val = float(np.sqrt(np.mean(D[fin] ** 2)))
            # Split the discrepancy: a common offset (the same for every fragment -- fewer
            # charges make the far field uniformly less negative) is removed by per-channel
            # standardisation at a fixed production length and carries no sequence
            # information; the fragment-specific scatter is the flank sequence changing the
            # label. Convergence is judged on the scatter.
            bias = np.nanmean(D, axis=0); scat = np.nanstd(D, axis=0)
            off_r = [float(np.sqrt(np.nanmean(bias[..., r] ** 2))) for r in range(D.shape[-1])]
            sca_r = [float(np.sqrt(np.nanmean(scat[..., r] ** 2))) for r in range(D.shape[-1])]
            rec["by_length"][str(n)] = {"n_pairs": len(keys), "rmse_over_sd": round(val, 4),
                                        "rmse_over_sd_by_radius": [round(x, 4) for x in per_r],
                                        "offset_over_sd_by_radius": [round(x, 4) for x in off_r],
                                        "scatter_over_sd_by_radius": [round(x, 4) for x in sca_r],
                                        "within_tolerance": val <= TOLERANCE,
                                        "scatter_within_tolerance": max(sca_r) <= TOLERANCE}
        ok = [int(n) for n, v in rec["by_length"].items() if v["within_tolerance"]]
        rec["shortest_converged_bp"] = min(ok) if ok else None
        # Per shell: convergence is shell-dependent (the outer shells feel distant charges),
        # so the production length can be chosen per channel group rather than by the
        # pooled number, which the outermost shells dominate.
        n_sh = len(next(iter(rec["by_length"].values()))["rmse_over_sd_by_radius"]) if rec["by_length"] else 0
        per_shell, per_shell_sc = [], []
        for j in range(n_sh):
            okj = [int(n) for n, v in rec["by_length"].items()
                   if v["rmse_over_sd_by_radius"][j] is not None
                   and v["rmse_over_sd_by_radius"][j] <= TOLERANCE]
            per_shell.append(min(okj) if okj else None)
            oks = [int(n) for n, v in rec["by_length"].items()
                   if v["scatter_over_sd_by_radius"][j] <= TOLERANCE]
            per_shell_sc.append(min(oks) if oks else None)
        rec["shortest_converged_bp_by_shell"] = per_shell
        rec["shortest_scatter_converged_bp_by_shell"] = per_shell_sc
        out["per_condition"][cond] = rec
        print(f"{cond}: reference {ref_len} bp; " + ", ".join(
            f"{n} bp {v['rmse_over_sd']:.3f}{' ok' if v['within_tolerance'] else ''}"
            for n, v in sorted(rec["by_length"].items(), key=lambda kv: -int(kv[0])))
              + f" -> shortest converged {rec['shortest_converged_bp']}; by shell (rmse) "
              f"{rec['shortest_converged_bp_by_shell']}; by shell (scatter) "
              f"{rec['shortest_scatter_converged_bp_by_shell']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
