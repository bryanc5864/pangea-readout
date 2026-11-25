# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Compare training runs from what they wrote, and nothing else.

Every number here is read from a run's own `config.json`, `history.jsonl` and
`best_r2.npz` (and `final_metrics.json` when the run produced one). The best epoch is
chosen by the score `train.py` itself uses to keep `best_ema.pt` -- per-base-pair median
R-squared plus global median R-squared on the validation side -- so the row reported for
a run describes the checkpoint that run actually kept.

Usage:
    python src/eval/compare_runs.py runs/v2_S_base runs/v2_M ... --out results/x.json \
        [--channels D_bubble_position_bp F_zdna_centre_offset_bp ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np


def _score(rec: Dict) -> Optional[float]:
    if rec.get("per_bp_median_r2") is None:
        return None
    return (rec.get("per_bp_median_r2") or 0.0) + (rec.get("global_median_r2") or 0.0)


def summarise_run(path: str, channels: List[str]) -> Optional[Dict]:
    hist_p, cfg_p = os.path.join(path, "history.jsonl"), os.path.join(path, "config.json")
    if not (os.path.exists(hist_p) and os.path.exists(cfg_p)):
        return None
    cfg = json.load(open(cfg_p))
    hist = [json.loads(l) for l in open(hist_p) if l.strip()]
    evald = [r for r in hist if _score(r) is not None]
    if not evald:
        return None
    best = max(evald, key=_score)
    a = cfg.get("args", {})
    row = {
        "run": os.path.basename(os.path.normpath(path)),
        "n_parameters_M": round(cfg.get("n_parameters", float("nan")) / 1e6, 2),
        "width": a.get("width"), "blocks": a.get("blocks"), "rank": a.get("rank"),
        "attn_layers": a.get("attn_layers"), "global_readout": a.get("global_readout", "linear"),
        "beta": a.get("beta"), "split": a.get("split"), "lr": a.get("lr"),
        "epochs_planned": a.get("epochs"), "epochs_done": len(hist),
        "best_epoch": best["epoch"], "seconds_per_epoch": float(np.median([r["seconds"] for r in hist])),
        "n_val": cfg.get("split", {}).get("n_val"), "n_test": cfg.get("split", {}).get("n_test", 0),
    }
    for k in ("per_bp_median_r2", "per_bp_mean_r2", "per_bp_frac_above_0.9",
              "per_bp_frac_above_0.99", "global_median_r2", "global_mean_r2",
              "global_frac_above_0.9", "global_frac_above_0.99", "window_10_median_r2",
              "window_20_median_r2", "window_50_median_r2", "per_bp_median_r2_by_family",
              "global_median_r2_by_family", "per_bp_worst", "global_worst"):
        row[k] = best.get(k)
    npz = os.path.join(path, "best_r2.npz")
    if channels and os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        names = {"per_bp": list(z["per_bp_names"]), "global": list(z["global_names"])}
        row["channels"] = {}
        for c in channels:
            for key in ("global", "per_bp"):
                if c in names[key]:
                    row["channels"][c] = float(z[key][names[key].index(c)])
    fm = os.path.join(path, "final_metrics.json")
    if os.path.exists(fm):
        row["final_metrics"] = json.load(open(fm))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--channels", nargs="*", default=[])
    a = ap.parse_args()
    rows = [r for r in (summarise_run(p, a.channels) for p in a.runs) if r]
    hdr = ("run", "params", "epochs", "best", "s/ep", "bp med", "bp>0.9", "bp>0.99",
           "gl med", "gl>0.9", "gl>0.99", "w10", "w50")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        f = lambda v: "-" if v is None else f"{v:.3f}"            # noqa: E731
        print(f"| {r['run']} | {r['n_parameters_M']}M | {r['epochs_done']}/{r['epochs_planned']} "
              f"| {r['best_epoch']} | {r['seconds_per_epoch']:.1f} | {f(r['per_bp_median_r2'])} "
              f"| {f(r['per_bp_frac_above_0.9'])} | {f(r['per_bp_frac_above_0.99'])} "
              f"| {f(r['global_median_r2'])} | {f(r['global_frac_above_0.9'])} "
              f"| {f(r['global_frac_above_0.99'])} | {f(r['window_10_median_r2'])} "
              f"| {f(r['window_50_median_r2'])} |")
    if a.channels:
        print()
        print("| channel | " + " | ".join(r["run"] for r in rows) + " |")
        print("|---|" + "---|" * len(rows))
        for c in a.channels:
            vals = [r.get("channels", {}).get(c) for r in rows]
            print(f"| {c} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
