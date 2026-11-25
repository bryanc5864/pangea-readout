# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Score pre-registered arms against their matched baselines on the held-out test side.

For each (arm, baseline) pair: Family L per-bp and whole-sequence median R2 and their
change, every other family's change (the non-inferiority margins: per-bp -0.001,
whole-sequence -0.002), the minus-strand L channels separately, and the verdict under
the pre-registered rule (gain >= 0.02 on L per-bp, L whole-sequence not lower, all other
families non-inferior). Seed replicates of the baseline, when present, give the
run-to-run spread printed beside each change. Only arms whose final metrics exist are
scored. Writes results/<name>.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PAIRS = [("v3_L10M_tie", "v3_L10M"), ("v3_L10M_tiebpp", "v3_L10M_tie"),
         ("v3_L10M_axial", "v3_L10M"), ("v3_L10M_axtie", "v3_L10M_axial"),
         ("v3_L10M_axtiebpp", "v3_L10M_axtie"), ("v3_L10M_axial_auxL", "v3_L10M_auxL"),
         ("v3_L10M_full_auxL", "v3_L10M_auxL"), ("v3_L10M_auxL", "v3_L10M"),
         ("v3_L50M_auxL", "v3_L50M"), ("v3_L10M_auxAll", "v3_L10M")]
SEEDS = {"v3_L10M": ["v3_L10M_seed1", "v3_L10M_seed2"]}


def load(runs, r):
    p = os.path.join(runs, r, "final_metrics.json")
    return json.load(open(p))["test"] if os.path.exists(p) else None


def channel_r2(runs, r, names_wanted):
    p = os.path.join(runs, r, "final_metrics.npz")
    c = os.path.join(runs, r, "config.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        return {}
    a = np.load(p); cfg = json.load(open(c))
    pos = {n: i for i, n in enumerate(cfg["per_bp_names"])}
    return {n: float(a["test__per_bp__r2"][pos[n]]) for n in names_wanted if n in pos}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs_v3")
    ap.add_argument("--name", default="family_l_arms")
    a = ap.parse_args()
    out = {"pairs": [], "seed_spread": {}}
    for base, reps in SEEDS.items():
        vals = [load(a.runs, r) for r in [base] + reps]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            fams = sorted(vals[0]["per_bp_median_r2_by_family"])
            out["seed_spread"][base] = {
                "n_runs": len(vals),
                "per_bp": {f: float(np.ptp([v["per_bp_median_r2_by_family"][f] for v in vals])) for f in fams},
                "global": {f: float(np.ptp([v["global_median_r2_by_family"][f] for v in vals]))
                           for f in sorted(vals[0]["global_median_r2_by_family"])}}
    lm = ["L_m_unpaired_prob", "L_m_positional_entropy", "L_p_unpaired_prob", "L_p_positional_entropy"]
    rows = ["| arm | baseline | L per-bp | L whole-seq | worst other per-bp change | worst other whole-seq change | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, base in PAIRS:
        A, B = load(a.runs, arm), load(a.runs, base)
        if A is None or B is None:
            continue
        pa, pb = A["per_bp_median_r2_by_family"], B["per_bp_median_r2_by_family"]
        ga, gb = A["global_median_r2_by_family"], B["global_median_r2_by_family"]
        d_lpb, d_lgl = pa["L"] - pb["L"], ga["L"] - gb["L"]
        other_pb = {f: pa[f] - pb[f] for f in pa if f != "L" and f in pb}
        other_gl = {f: ga[f] - gb[f] for f in ga if f != "L" and f in gb}
        wpb = min(other_pb.items(), key=lambda kv: kv[1]); wgl = min(other_gl.items(), key=lambda kv: kv[1])
        noninf = wpb[1] >= -0.001 and wgl[1] >= -0.002
        verdict = ("helps" if d_lpb >= 0.02 and d_lgl >= 0 and noninf else
                   "inconclusive" if d_lpb > 0 and noninf else "does not help")
        ca, cb = channel_r2(a.runs, arm, lm), channel_r2(a.runs, base, lm)
        rec = {"arm": arm, "baseline": base, "L_per_bp": [pb["L"], pa["L"]], "L_global": [gb["L"], ga["L"]],
               "other_per_bp_change": other_pb, "other_global_change": other_gl,
               "L_channels": {n: [cb.get(n), ca.get(n)] for n in lm}, "verdict": verdict,
               "all_per_bp": [B["per_bp_median_r2"], A["per_bp_median_r2"]],
               "all_global": [B["global_median_r2"], A["global_median_r2"]]}
        out["pairs"].append(rec)
        rows.append(f"| {arm} | {base} | {pb['L']:.4f} -> {pa['L']:.4f} ({d_lpb:+.4f}) | "
                    f"{gb['L']:.4f} -> {ga['L']:.4f} ({d_lgl:+.4f}) | {wpb[0]} {wpb[1]:+.4f} | "
                    f"{wgl[0]} {wgl[1]:+.4f} | {verdict} |")
    print("\n".join(rows))
    if out["seed_spread"]:
        print("\nseed spread (max - min over replicates, test medians):", json.dumps(out["seed_spread"]))
    for rec in out["pairs"]:
        ch = rec["L_channels"]
        if any(v[0] is not None and v[1] is not None for v in ch.values()):
            print(f"{rec['arm']}: " + ", ".join(f"{n} {v[0]:.3f}->{v[1]:.3f}" for n, v in ch.items()
                                                 if v[0] is not None and v[1] is not None))
    json.dump(out, open(os.path.join("results", f"{a.name}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
