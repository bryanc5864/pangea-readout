# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Split conformal calibration of a finished run's predicted variances (PLAN.md, Aim 1).

The model predicts a mean and a log-variance per target. Conformal recalibration, per
channel: on the validation side, the standardised residual z = |y - mu| / sigma; for each
nominal central level p, the finite-sample conformal quantile q_p = the ceil((n+1)p)-th
smallest z (n validation values). On the test side, the interval mu +- q_p sigma then covers
at least p of targets (marginally, exchangeability assumed). Reported per channel and level:
test coverage of the raw Gaussian interval (z_p sigma) and of the conformal one (q_p sigma),
and the scale factor q_p / z_p, which a release model stores beside its weights.

Per-bp channels pool positions; each channel's values are subsampled to at most
`--max-values` per side (deterministically) to bound memory. Writes <run>/conformal.json and
<run>/conformal_scales.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import NormalDist

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from data import landscape_from_args  # noqa: E402
from pangea import PANGEA, ChannelPlan, model_kwargs_from_args  # noqa: E402

LEVELS = (0.5, 0.683, 0.8, 0.9, 0.954, 0.99)


@torch.no_grad()
def standardised_residuals(model, X, Y, idx, key, batch, max_values, seed):
    """(n_channels,) list of 1-D arrays of z = |y - mu| / sigma on fragments idx."""
    model.eval()
    zs = None
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out = model(X[b].float())
        m, lv = out[key][0].float(), out[key][1].float()
        t = Y[b].float()
        z = (t - m).abs() * torch.exp(-0.5 * lv)
        z = z.transpose(0, 1).reshape(z.shape[1], -1) if z.dim() == 3 else z.T
        tv = (t.transpose(0, 1).reshape(t.shape[1], -1) if t.dim() == 3 else t.T)
        z = z.cpu().numpy(); ok = np.isfinite(tv.cpu().numpy())
        if zs is None:
            zs = [[] for _ in range(z.shape[0])]
        for j in range(z.shape[0]):
            zs[j].append(z[j][ok[j]])
    rng = np.random.default_rng(seed)
    res = []
    for parts in zs:
        v = np.concatenate(parts) if parts else np.zeros(0)
        if v.size > max_values:
            v = rng.choice(v, size=max_values, replace=False)
        res.append(v)
    return res


def conformal_quantile(z: np.ndarray, p: float) -> float:
    n = z.size
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * p))
    return float(np.sort(z)[min(k, n) - 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-values", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
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
    model = PANGEA(plan, **model_kwargs_from_args(args)).to(dev)
    ck = torch.load(os.path.join(a.run, "best_ema.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    sp = np.load(os.path.join(a.run, "split_indices.npz"))
    va, te = sp["val"], sp["test"]
    X = torch.from_numpy(ds.one_hot()).to(dev)
    zq = {p: NormalDist().inv_cdf(0.5 + p / 2.0) for p in LEVELS}
    eb = min(int(args.get("eval_batch_size", 4096) or 4096), 2048)
    out = {"run": os.path.basename(os.path.normpath(a.run)), "best_epoch": ck.get("epoch"),
           "levels": list(LEVELS), "n_val_fragments": int(len(va)), "n_test_fragments": int(len(te))}
    scales = {}
    for key, Y, names in (("per_bp", torch.from_numpy(ds.per_bp_np).to(dev), plan.per_bp_names),
                          ("global", torch.from_numpy(ds.global_np).to(dev), plan.global_names)):
        zv = standardised_residuals(model, X, Y, torch.as_tensor(va, device=dev), key, eb,
                                    a.max_values, a.seed)
        zt = standardised_residuals(model, X, Y, torch.as_tensor(te, device=dev), key, eb,
                                    a.max_values, a.seed + 1)
        S = np.full((len(names), len(LEVELS)), np.nan)
        raw = np.full_like(S, np.nan); conf = np.full_like(S, np.nan)
        for j in range(len(names)):
            if zv[j].size < 50 or zt[j].size < 50:
                continue
            for k, p in enumerate(LEVELS):
                q = conformal_quantile(zv[j], p)
                S[j, k] = q / zq[p]
                raw[j, k] = float(np.mean(zt[j] <= zq[p]))
                conf[j, k] = float(np.mean(zt[j] <= q))
        scales[key] = S
        rec = {}
        for k, p in enumerate(LEVELS):
            rec[f"{p:g}"] = {
                "raw_coverage_median": float(np.nanmedian(raw[:, k])),
                "raw_abs_error_median": float(np.nanmedian(np.abs(raw[:, k] - p))),
                "raw_frac_within_0.02": float(np.nanmean(np.abs(raw[:, k] - p) <= 0.02)),
                "conformal_coverage_median": float(np.nanmedian(conf[:, k])),
                "conformal_abs_error_median": float(np.nanmedian(np.abs(conf[:, k] - p))),
                "conformal_frac_within_0.02": float(np.nanmean(np.abs(conf[:, k] - p) <= 0.02)),
                "scale_median": float(np.nanmedian(S[:, k])),
            }
        out[key] = rec
        print(key, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rec.items()}), flush=True)
    np.savez(os.path.join(a.run, "conformal_scales.npz"), levels=np.array(LEVELS),
             per_bp=scales["per_bp"], global_=scales["global"],
             per_bp_names=np.array(plan.per_bp_names), global_names=np.array(plan.global_names))
    json.dump(out, open(os.path.join(a.run, "conformal.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.run, 'conformal.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
