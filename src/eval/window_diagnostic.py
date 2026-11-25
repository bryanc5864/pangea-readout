# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Are window outputs limited by the window heads, or by the window targets?

Window targets are the per-bp targets averaged over 10, 20 and 50 bp (pooled_window_
targets). For each window size this scores, on the run's own test side and in R-squared
per channel, (a) the model's window head -- what the run reports -- and (b) the same
pooling applied to the model's own per-bp predictions. If (b) is much higher, the window
heads are the weak link; if both are low, windowed targets are hard (low variance) and
the gap is in the metric, not the model. Writes <run>/window_diagnostic.json.
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
from data import InMemoryLandscape, pooled_window_targets, landscape_from_args  # noqa: E402
from pangea import PANGEA, ChannelPlan, model_kwargs_from_args  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--side", choices=("val", "test"), default="val",
                    help="choosing between readouts is model selection: decide on val")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json"))); args = cfg["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = landscape_from_args(args)
    plan = ChannelPlan.from_metadata(ds.pb["names"], ds.gl["names"], ds.pb["parity"],
                                     ds.gl["parity"], ds.pb["family"], ds.gl["family"],
                                     ds.pb["rc_shift"])
    kw = model_kwargs_from_args(args)
    model = PANGEA(plan, **kw).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(a.run, "best_ema.pt"), map_location=dev,
                                     weights_only=False)["model"])
    te = torch.as_tensor(np.load(os.path.join(a.run, "split_indices.npz"))[a.side], device=dev)
    X = torch.from_numpy(ds.one_hot()).to(dev); Ypb = torch.from_numpy(ds.per_bp_np).to(dev)
    acc = {}
    for s in range(0, len(te), 1024):
        b = te[s:s + 1024]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out = model(X[b].float())
        t = Ypb[b].float(); pb = out["per_bp"][0].float()
        for w in args["windows"]:
            tw = pooled_window_targets(t, w)
            for kind, pred in (("head", out[f"window_{w}"][0].float()),
                               ("pooled_per_bp", pooled_window_targets(pb, w))):
                v = torch.isfinite(tw) & torch.isfinite(pred)
                tz, pz = torch.where(v, tw, 0.0), torch.where(v, pred, 0.0)
                parts = [((pz - tz) ** 2).sum((0, 2)), tz.sum((0, 2)), (tz ** 2).sum((0, 2)),
                         v.sum((0, 2)).float()]
                key = (w, kind)
                acc[key] = parts if key not in acc else [x + y for x, y in zip(acc[key], parts)]
    res = {}
    for (w, kind), (sse, s1, s2, n) in acc.items():
        n = n.clamp(min=1); var = s2 / n - (s1 / n) ** 2
        r2 = torch.where(var > 1e-8, 1 - sse / n / var.clamp(min=1e-12), torch.full_like(var, float("nan")))
        r2 = r2.cpu().numpy()
        res.setdefault(str(w), {})[kind] = {"median_r2": float(np.nanmedian(r2)),
                                           "frac_above_0.99": float(np.nanmean(r2[np.isfinite(r2)] > 0.99))}
    for w, d in res.items():
        print(f"[{a.side}] window {w:>3s}: head median {d['head']['median_r2']:.4f} | pooled per-bp median "
              f"{d['pooled_per_bp']['median_r2']:.4f}  (>0.99: {d['head']['frac_above_0.99']:.3f} vs "
              f"{d['pooled_per_bp']['frac_above_0.99']:.3f})", flush=True)
    json.dump(res, open(os.path.join(a.run, f"window_diagnostic_{a.side}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
