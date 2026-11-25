# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The scaling study and the universality gap, from the runs' own held-out test metrics.

Reads `final_metrics.json` (written by train.py from the best-EMA checkpoint on the test
side that no training or selection decision saw) for the ladder runs and the universality
runs, and writes:

    results/scaling_v2.json        per rung: parameters, per-family test medians, fraction
                                   of channels above 0.9 / 0.99 / 0.999, medians excluding
                                   rare-event channels, calibration at 68.3 and 95.4 percent
    results/universality_v2.json   per rung and encoder: per-species test medians, and the
                                   universality gap -- in-filter species minus held-out
                                   species, and held-out species against the multi-clade
                                   encoder of the same size
    results/figures/scaling_v2.png accuracy against capacity, per family, on a log axis

PLAN.md: "plot per-channel accuracy against capacity. Then plot the universality gap
against capacity on the same axis."
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RUNS = os.environ.get("RUNS_DIR", "runs_v3")

# The universality comparison is over real species only. The per-species test metrics
# also carry the designed libraries (e.g. "hexamer:polyA"), whose fragments share their
# flanks, so whole-sequence channels are near-constant within a library and within-library
# R-squared is not meaningful there (-7.9 to -32 at 2M); they are reported separately.
SPECIES = {
    "Homo sapiens": "animal", "Mus musculus": "animal", "Danio rerio": "animal",
    "Drosophila melanogaster": "animal", "Caenorhabditis elegans": "animal",
    "Arabidopsis thaliana": "plant", "Zea mays": "plant", "Sorghum bicolor": "plant",
    "Saccharomyces cerevisiae": "fungus", "Plasmodium falciparum": "protist",
    "Dictyostelium discoideum": "protist", "Escherichia coli": "bacterium",
    "Mycobacterium tuberculosis": "bacterium", "Streptomyces coelicolor": "bacterium",
    "Thermus thermophilus": "bacterium", "Methanocaldococcus jannaschii": "archaeon"}


def load(run: str):
    p = os.path.join(RUNS, run, "final_metrics.json")
    c = os.path.join(RUNS, run, "config.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        return None, None
    return json.load(open(p)), json.load(open(c))


def frac_above(arr, t):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a > t)) if a.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", nargs="+", default=["v3_L2M", "v3_L10M", "v3_L50M",
                                                    "v3_L200M", "v3_L1000M"])
    ap.add_argument("--universality", nargs="+", default=["kingdom", "human"])
    ap.add_argument("--sizes", nargs="+", default=["2M", "10M", "50M", "200M"])
    a = ap.parse_args()
    os.makedirs("results/figures", exist_ok=True)

    rows = []
    for run in a.ladder:
        fm, cfg = load(run)
        if fm is None or "test" not in fm:
            print(f"{run}: no test metrics yet"); continue
        z = np.load(os.path.join(RUNS, run, "final_metrics.npz"), allow_pickle=True)
        t = fm["test"]
        pb, gl = z["test__per_bp__r2"], z["test__global__r2"]
        lev = fm["levels"]
        i68, i95 = lev.index(0.683), lev.index(0.954)
        row = {"run": run, "n_parameters": cfg["n_parameters"], "best_epoch": fm.get("best_epoch"),
               "n_test_fragments": t["n_fragments"],
               "per_bp_median_r2": t["per_bp_median_r2"], "global_median_r2": t["global_median_r2"],
               "per_bp_median_r2_excluding_rare": t.get("per_bp_median_r2_excluding_rare_events"),
               "global_median_r2_excluding_rare": t.get("global_median_r2_excluding_rare_events"),
               "per_bp_median_r2_interior": t.get("per_bp_median_r2_interior"),
               "per_bp_by_family_interior": t.get("per_bp_median_r2_interior_by_family"),
               "per_bp_by_family": t["per_bp_median_r2_by_family"],
               "global_by_family": t["global_median_r2_by_family"],
               "calibration_per_bp_68": t["per_bp_calibration_median_coverage"][i68],
               "calibration_per_bp_95": t["per_bp_calibration_median_coverage"][i95],
               "calibration_global_68": t["global_calibration_median_coverage"][i68],
               "calibration_global_95": t["global_calibration_median_coverage"][i95]}
        for thr in (0.9, 0.99, 0.999):
            row[f"per_bp_frac_above_{thr}"] = frac_above(pb, thr)
            row[f"global_frac_above_{thr}"] = frac_above(gl, thr)
        # Windows under both readouts: the dedicated heads, and the pooled per-bp output
        # (the primary readout since RESULTS.md section 17). Runs evaluated before the
        # pooled readout existed carry it in window_diagnostic_test.json.
        diag_p = os.path.join(RUNS, run, "window_diagnostic_test.json")
        diag = json.load(open(diag_p)) if os.path.exists(diag_p) else {}
        for w in (10, 20, 50):
            row[f"window_{w}_median_r2"] = t.get(f"window_{w}_median_r2")
            row[f"window_{w}_median_r2_pooled"] = t.get(
                f"window_{w}_median_r2_pooled_readout",
                diag.get(str(w), {}).get("pooled_per_bp", {}).get("median_r2"))
        rows.append(row)
        print(f"{run:12s} {row['n_parameters']/1e6:8.1f}M  test per-bp {row['per_bp_median_r2']:.4f} "
              f"global {row['global_median_r2']:.4f}  bp>0.99 {row['per_bp_frac_above_0.99']:.3f} "
              f"gl>0.99 {row['global_frac_above_0.99']:.3f}  cal68 {row['calibration_per_bp_68']:.3f}")
    json.dump({"rows": rows, "split": "similarity, 5 percent test side"},
              open("results/scaling_v2.json", "w"), indent=1)

    # Measured genome GC per species, from the species-panel table in RESULTS.md.
    import re
    gc = {}
    for line in open("RESULTS.md"):
        m = re.match(r"\|\s*([\d.]+)%\s*\|\s*\w+\s*\|\s*([^|]+?)\s*\|\s*GCF_", line)
        if m:
            gc[m.group(2)] = float(m.group(1))
    from scipy.stats import spearmanr

    uni = {"rows": [], "metrics_fixed_in": "DECISIONS.md, 'The universality gap: metrics "
                                           "fixed before the 50M and 200M encoders report'"}
    for tag in a.universality:
        for size in a.sizes:
            fm, cfg = load(f"v3_U{tag}_{size}")
            ref, _ = load(f"v3_L{size}")
            if fm is None or "test_by_species" not in fm:
                continue
            sp = {k: v for k, v in fm["test_by_species"].items() if k in SPECIES}
            inn = [v for v in sp.values() if v["in_training_filter"]]
            out = {k: v for k, v in sp.items() if not v["in_training_filter"]}
            rec = {"encoder": tag, "size": size, "n_parameters": cfg["n_parameters"],
                   "in_filter_per_bp": float(np.median([v["per_bp_median_r2"] for v in inn])) if inn else None,
                   "held_out_per_bp": float(np.median([v["per_bp_median_r2"] for v in out.values()])) if out else None,
                   "in_filter_global": float(np.median([v["global_median_r2"] for v in inn])) if inn else None,
                   "held_out_global": float(np.median([v["global_median_r2"] for v in out.values()])) if out else None,
                   "by_species": sp}
            # The five pre-registered gap measures (DECISIONS.md).
            for key, short in (("per_bp_median_r2", "per_bp"), ("global_median_r2", "global")):
                yi = np.array([v[key] for v in inn]); yo = np.array([v[key] for v in out.values()])
                if yi.size and yo.size:
                    rec[f"{short}_gap_median"] = float(np.median(yi) - np.median(yo))
                    rec[f"{short}_gap_worst"] = float(np.median(yi) - yo.min())
                    rec[f"{short}_error_ratio"] = float((1 - np.median(yo)) / max(1e-12, 1 - np.median(yi)))
                    lo = min(gc[k] for k, v in sp.items() if v["in_training_filter"] and k in gc)
                    hi = max(gc[k] for k, v in sp.items() if v["in_training_filter"] and k in gc)
                    ks = [k for k in out if k in gc]
                    dist = [max(0.0, lo - gc[k], gc[k] - hi) for k in ks]
                    if len(set(dist)) > 1:
                        rec[f"{short}_spearman_vs_gc_distance"] = float(
                            spearmanr(dist, [out[k][key] for k in ks]).correlation)
            if ref is not None and "test_by_species" in ref:
                rs = {k: v for k, v in ref["test_by_species"].items() if k in SPECIES}
                common = [k for k in out if k in rs]
                rec["multi_clade_same_species_per_bp"] = float(np.median([rs[k]["per_bp_median_r2"] for k in common])) if common else None
                rec["multi_clade_same_species_global"] = float(np.median([rs[k]["global_median_r2"] for k in common])) if common else None
            uni["rows"].append(rec)
            print(f"universality {tag:7s} {size:5s}: " + "  ".join(
                f"{k} {rec[k]:+.4f}" for k in ("per_bp_gap_median", "global_gap_median",
                                               "global_gap_worst", "global_error_ratio",
                                               "global_spearman_vs_gc_distance") if k in rec))
    json.dump(uni, open("results/universality_v2.json", "w"), indent=1)

    # The bottleneck sweep: rank r at the fixed 10M trunk; the 10M ladder run is the
    # full-rank point. PLAN.md: "the smallest rank at which per-channel accuracy
    # saturates is an independent estimate of the intrinsic dimensionality". Saturation
    # is defined before looking: within 0.001 of the full-rank median test R-squared, for
    # per-bp and global separately.
    sweep = []
    for run in [f"v3_rank{r}" for r in (4, 8, 16, 32, 64, 128, 256)] + ["v3_L10M"]:
        fm, cfg = load(run)
        if fm is None or "test" not in fm:
            continue
        t = fm["test"]
        z = np.load(os.path.join(RUNS, run, "final_metrics.npz"), allow_pickle=True)
        sweep.append({"run": run, "rank": cfg["args"]["rank"],
                      "width": cfg["args"]["width"],
                      "per_bp_median_r2": t["per_bp_median_r2"],
                      "global_median_r2": t["global_median_r2"],
                      "per_bp_frac_above_0.99": frac_above(z["test__per_bp__r2"], 0.99),
                      "global_frac_above_0.99": frac_above(z["test__global__r2"], 0.99)})
    sweep.sort(key=lambda d: d["rank"])
    sat = {}
    if sweep:
        full = max(sweep, key=lambda d: d["rank"])
        for key in ("per_bp_median_r2", "global_median_r2"):
            ok = [d["rank"] for d in sweep if d[key] >= full[key] - 0.001]
            sat[key] = min(ok) if ok else None
        for d in sweep:
            print(f"rank {d['rank']:4d}: test per-bp {d['per_bp_median_r2']:.4f} global "
                  f"{d['global_median_r2']:.4f}  bp>0.99 {d['per_bp_frac_above_0.99']:.3f}")
        print(f"saturating rank (within 0.001 of full rank {full['rank']}): {sat}")
    json.dump({"rows": sweep, "saturating_rank_within_0.001": sat,
               "trunk": "10M rung, width 272"}, open("results/rank_sweep_v2.json", "w"), indent=1)

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = sorted({f for r in rows for f in r["per_bp_by_family"]} |
                      {f for r in rows for f in r["global_by_family"]})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        n = [r["n_parameters"] for r in rows]
        for ax, key, title in ((axes[0], "per_bp_by_family", "per base pair"),
                               (axes[1], "global_by_family", "whole sequence")):
            for f_ in fams:
                y = [1 - r[key].get(f_, np.nan) for r in rows]
                if np.all(~np.isfinite(y)):
                    continue
                ax.plot(n, y, marker="o", label=f"family {f_}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("parameters"); ax.set_ylabel("1 - median test R$^2$")
            ax.set_title(title); ax.grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle("PANGEA scaling on held-out test fragments (similarity split)")
        fig.tight_layout()
        fig.savefig("results/figures/scaling_v2.png", dpi=150)
        print("wrote results/figures/scaling_v2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
