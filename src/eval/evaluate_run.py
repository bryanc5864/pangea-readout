# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Post-hoc evaluation of a finished run's kept checkpoint, grouped on the test side.

Loads the run's own configuration, `best_ema.pt` and `split_indices.npz`, so the model,
the weights and the test fragments are exactly those the run reported, and scores the
test side grouped by
  - stratum: panel windows, MPRA elements, designed hexamer libraries, composition
    extremes, random sequence, composition controls -- PLAN.md's "synthetic-composition
    stress test" is the comparison of the synthetic strata against genomic ones;
  - GC band of the fragment itself.
Within-group R-squared uses the group's own variance, so a group whose channels are
nearly constant (a designed library sharing its flanks) scores low for that reason;
groups are reported with their sizes and a per-bp and global median per family.
Writes <run>/eval_by_group.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from data import InMemoryLandscape, landscape_from_args  # noqa: E402
from pangea import PANGEA, ChannelPlan, model_kwargs_from_args  # noqa: E402
import train as T  # noqa: E402

GC_BANDS = ((0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-group", type=int, default=50)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    args = cfg["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = landscape_from_args(args)
    plan = ChannelPlan.from_metadata(ds.pb["names"], ds.gl["names"], ds.pb["parity"],
                                     ds.gl["parity"], ds.pb["family"], ds.gl["family"],
                                     ds.pb["rc_shift"])
    if plan.per_bp_names != cfg["per_bp_names"] or plan.global_names != cfg["global_names"]:
        raise RuntimeError("dataset channel layout differs from the run's")
    kw = model_kwargs_from_args(args)
    model = PANGEA(plan, **kw).to(dev)
    ck = torch.load(os.path.join(a.run, "best_ema.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    te = np.load(os.path.join(a.run, "split_indices.npz"))["test"]
    X = torch.from_numpy(ds.one_hot()).to(dev)
    Ypb = torch.from_numpy(ds.per_bp_np).to(dev)
    Ygl = torch.from_numpy(ds.global_np).to(dev)
    meta = T.library_metadata(args.get("library", T.DEFAULT_LIBRARY), ds.sequences)
    strata = meta["stratum"]
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in ds.sequences])
    groups = {}
    for st in sorted(set(strata[te]) - {""}):
        groups[f"stratum:{st}"] = te[strata[te] == st]
    for lo, hi in GC_BANDS:
        groups[f"gc:{lo:.1f}-{min(hi, 1.0):.1f}"] = te[(gc[te] >= lo) & (gc[te] < hi)]
    eb = args.get("eval_batch_size", 4096) or 4096
    out = {"run": os.path.basename(os.path.normpath(a.run)), "best_epoch": ck.get("epoch"),
           "n_parameters": cfg["n_parameters"], "groups": {}}
    for name, idx in groups.items():
        if len(idx) < a.min_group:
            out["groups"][name] = {"n_fragments": int(len(idx)), "skipped": "too few"}
            continue
        r = T.evaluate_full(model, X, Ypb, Ygl, torch.as_tensor(idx, device=dev), eb,
                            args["windows"])
        rec = {"n_fragments": int(len(idx)),
               "gc_mean": float(gc[idx].mean())}
        for key in ("per_bp", "global"):
            r2 = r[key]["r2"]; fam = plan.per_bp_family if key == "per_bp" else plan.global_family
            rec[f"{key}_median_r2"] = float(np.nanmedian(r2))
            rec[f"{key}_by_family"] = {f: float(np.nanmedian(r2[[i for i, x in enumerate(fam) if x == f]]))
                                       for f in sorted(set(fam))
                                       if np.isfinite(r2[[i for i, x in enumerate(fam) if x == f]]).any()}
        if "r2_interior" in r["per_bp"]:
            rec["per_bp_median_r2_interior"] = float(np.nanmedian(r["per_bp"]["r2_interior"]))
        out["groups"][name] = rec
        print(f"{name:24s} n={len(idx):5d}  per-bp {rec['per_bp_median_r2']:.4f}  "
              f"global {rec['global_median_r2']:.4f}", flush=True)
    json.dump(out, open(os.path.join(a.run, "eval_by_group.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.run, 'eval_by_group.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
