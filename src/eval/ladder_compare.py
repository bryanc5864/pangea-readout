# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Compare two scaling ladders family by family on the held-out test side.

The v4 ladder repeats the v3 rungs with one architectural change (the Family L strand
tie). Two pre-registered readings (DECISIONS.md): Family L per-bp rises with size under
the tie (it was flat under v3), and Family F's whole-sequence cost of the tie at 10M
shrinks with size, within -0.002 at 200M. This prints, for every rung present in both
ladders, the per-family per-bp and whole-sequence test medians of each and their
difference, writes results/ladder_<a>_vs_<b>.json, and draws the L and F curves.
"""
from __future__ import annotations

import argparse
import json
import os

RUNGS = [("2M", "v3_L2M", "v4_L2M"), ("10M", "v3_L10M", "v3_L10M_tie"),
         ("50M", "v3_L50M", "v4_L50M"), ("200M", "v3_L200M", "v4_L200M")]


def test(runs, r):
    p = os.path.join(runs, r, "final_metrics.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = d["test"]; t["n_parameters"] = d.get("n_parameters")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs_v3")
    ap.add_argument("--out", default="results/ladder_v3_vs_v4.json")
    a = ap.parse_args()
    rows = []
    for size, ra, rb in RUNGS:
        A, B = test(a.runs, ra), test(a.runs, rb)
        if A is None or B is None:
            print(f"{size:5s}: {'missing ' + ra if A is None else ''} {'missing ' + rb if B is None else ''}")
            continue
        row = {"size": size, "v3": ra, "v4": rb, "per_bp": {}, "global": {},
               "all": {"per_bp": [A["per_bp_median_r2"], B["per_bp_median_r2"]],
                       "global": [A["global_median_r2"], B["global_median_r2"]]}}
        for res, key in (("per_bp", "per_bp_median_r2_by_family"), ("global", "global_median_r2_by_family")):
            for fam in sorted(set(A[key]) & set(B[key])):
                row[res][fam] = [A[key][fam], B[key][fam], B[key][fam] - A[key][fam]]
        rows.append(row)
        pb, gl = row["per_bp"], row["global"]
        worst_pb = min(((v[2], f) for f, v in pb.items() if f != "L"), default=(0, "-"))
        worst_gl = min(((v[2], f) for f, v in gl.items() if f != "L"), default=(0, "-"))
        print(f"{size:5s}: L per-bp {pb['L'][0]:.4f} -> {pb['L'][1]:.4f} ({pb['L'][2]:+.4f}) | "
              f"L whole-seq {gl['L'][0]:.4f} -> {gl['L'][1]:.4f} ({gl['L'][2]:+.4f}) | "
              f"F whole-seq {gl['F'][0]:.4f} -> {gl['F'][1]:.4f} ({gl['F'][2]:+.4f}) | "
              f"worst other per-bp {worst_pb[1]} {worst_pb[0]:+.4f}, whole-seq {worst_gl[1]} {worst_gl[0]:+.4f}")
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sizes = {"2M": 2, "10M": 10, "50M": 50, "200M": 200}
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, (res, fam, title) in enumerate((("per_bp", "L", "Family L, per base"),
                                               ("global", "F", "Family F, whole-sequence"))):
            xs = [sizes[r["size"]] for r in rows]
            ax[k].plot(xs, [r[res][fam][0] for r in rows], "o-", label="v3 (no tie)")
            ax[k].plot(xs, [r[res][fam][1] for r in rows], "s-", label="v4 (strand tie)")
            ax[k].set_xscale("log"); ax[k].set_xlabel("parameters (millions)")
            ax[k].set_ylabel("test median R$^2$"); ax[k].set_title(title); ax[k].legend()
        fig.tight_layout()
        os.makedirs("results/figures", exist_ok=True)
        fig.savefig("results/figures/ladder_v3_vs_v4.png", dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
