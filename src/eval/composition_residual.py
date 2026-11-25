# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Composition residualisation of every channel (PLAN.md, Aim 1), and the model scored on it.

Each channel is regressed on composition -- for whole-sequence channels the fragment's GC,
mono- and dinucleotide frequencies; for per-bp channels the same frequencies in an 11 bp
window centred on the base pair -- by ridge regression fitted on training fragments. On the
test side this reports, per channel: R2 of composition alone; the model's R2 on the raw
target; and the model's R2 on the composition residual (target minus the composition fit,
prediction minus the same fit), i.e. how much of the base-order-dependent part, which
composition cannot explain, the model captures. Residualised targets are kept beside the raw
ones (`composition_fits.npz`: coefficients per channel) so any analysis can use either.
Writes <run>/composition_residual.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from data import landscape_from_args  # noqa: E402
from pangea import PANGEA, ChannelPlan, model_kwargs_from_args  # noqa: E402

DINUC = ["".join(p) for p in itertools.product("ACGT", repeat=2)]
WINDOW = 11
RIDGE = 1e-3


def global_features(seqs):
    F = np.zeros((len(seqs), 1 + 4 + 16), dtype=np.float64)
    for i, s in enumerate(seqs):
        n = len(s)
        F[i, 0] = (s.count("G") + s.count("C")) / n
        for k, b in enumerate("ACGT"):
            F[i, 1 + k] = s.count(b) / n
        for k, d in enumerate(DINUC):
            F[i, 5 + k] = sum(1 for j in range(n - 1) if s[j:j + 2] == d) / (n - 1)
    return F


def local_features(seqs):
    """(n_frag, L, 21): composition of the 11 bp window around each position (clipped)."""
    L = len(seqs[0]); h = WINDOW // 2
    idx = {b: k for k, b in enumerate("ACGT")}
    out = np.zeros((len(seqs), L, 21), dtype=np.float32)
    for i, s in enumerate(seqs):
        a = np.array([idx.get(c, 0) for c in s])
        mono = np.eye(4)[a]                                      # (L, 4)
        di = np.zeros((L, 16)); di[:-1] = np.eye(16)[a[:-1] * 4 + a[1:]]
        cm, cd = np.cumsum(np.vstack([np.zeros(4), mono]), 0), np.cumsum(np.vstack([np.zeros(16), di]), 0)
        for j in range(L):
            lo, hi = max(0, j - h), min(L, j + h + 1)
            m = (cm[hi] - cm[lo]) / (hi - lo)
            d = (cd[max(lo, hi - 1)] - cd[lo]) / max(1, hi - 1 - lo)
            out[i, j, 0] = m[1] + m[2]; out[i, j, 1:5] = m; out[i, j, 5:] = d
    return out


def ridge_fit(F, y):
    ok = np.isfinite(y)
    if ok.sum() < 50:
        return None
    X = np.hstack([F[ok], np.ones((ok.sum(), 1))])
    A = X.T @ X + RIDGE * len(X) * np.eye(X.shape[1]); A[-1, -1] -= RIDGE * len(X)
    return np.linalg.solve(A, X.T @ y[ok])


def ridge_pred(F, w):
    return np.hstack([F, np.ones((len(F), 1))]) @ w


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 20:
        return float("nan")
    v = np.var(y[ok])
    return float(1 - np.mean((y[ok] - p[ok]) ** 2) / v) if v > 1e-12 else float("nan")


@torch.no_grad()
def predict(model, X, idx, key, batch):
    model.eval(); outs = []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            outs.append(model(X[b].float())[key][0].float().cpu().numpy())
    return np.concatenate(outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--max-test", type=int, default=6000)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json"))); args = cfg["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = landscape_from_args(args)
    plan = ChannelPlan.from_metadata(ds.pb["names"], ds.gl["names"], ds.pb["parity"],
                                     ds.gl["parity"], ds.pb["family"], ds.gl["family"],
                                     ds.pb["rc_shift"])
    model = PANGEA(plan, **model_kwargs_from_args(args)).to(dev)
    ck = torch.load(os.path.join(a.run, "best_ema.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    sp = np.load(os.path.join(a.run, "split_indices.npz"))
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(sp["train"], size=min(a.max_train, len(sp["train"])), replace=False))
    te = np.sort(rng.choice(sp["test"], size=min(a.max_test, len(sp["test"])), replace=False))
    seqs = ds.sequences
    X = torch.from_numpy(ds.one_hot()).to(dev)
    eb = min(int(args.get("eval_batch_size", 4096) or 4096), 2048)
    out = {"run": os.path.basename(os.path.normpath(a.run)), "n_train_fit": int(len(tr)),
           "n_test": int(len(te)), "window_bp": WINDOW, "channels": {}}
    fits = {}
    # whole-sequence
    Ftr, Fte = global_features([seqs[i] for i in tr]), global_features([seqs[i] for i in te])
    Yg_tr = ds.global_np[tr].astype(np.float64); Yg_te = ds.global_np[te].astype(np.float64)
    Pg = predict(model, X, torch.as_tensor(te, device=dev), "global", eb).astype(np.float64)
    W = np.full((len(plan.global_names), Ftr.shape[1] + 1), np.nan)
    for j, n in enumerate(plan.global_names):
        w = ridge_fit(Ftr, Yg_tr[:, j])
        if w is None:
            continue
        W[j] = w; c = ridge_pred(Fte, w)
        out["channels"][n] = {"family": plan.global_family[j], "resolution": "global",
                              "composition_r2": r2(Yg_te[:, j], c), "model_r2": r2(Yg_te[:, j], Pg[:, j]),
                              "model_r2_on_residual": r2(Yg_te[:, j] - c, Pg[:, j] - c)}
    fits["global"] = W
    # per-bp (interior positions, as the trainer scores them)
    Ltr = local_features([seqs[i] for i in tr]); Lte = local_features([seqs[i] for i in te])
    k0 = 5; k1 = Ltr.shape[1] - 5
    Ppb = predict(model, X, torch.as_tensor(te, device=dev), "per_bp", eb)
    Wp = np.full((len(plan.per_bp_names), 22), np.nan)
    sub = rng.choice(len(tr), size=min(len(tr), 6000), replace=False)
    for j, n in enumerate(plan.per_bp_names):
        Ytr = ds.per_bp_np[tr[sub], j, k0:k1].astype(np.float64).reshape(-1)
        Ftr_ = Ltr[sub, k0:k1].reshape(-1, 21).astype(np.float64)
        w = ridge_fit(Ftr_, Ytr)
        if w is None:
            continue
        Wp[j] = w
        Yte = ds.per_bp_np[te, j, k0:k1].astype(np.float64).reshape(-1)
        c = ridge_pred(Lte[:, k0:k1].reshape(-1, 21).astype(np.float64), w)
        p = Ppb[:, j, k0:k1].astype(np.float64).reshape(-1)
        out["channels"][n] = {"family": plan.per_bp_family[j], "resolution": "per_bp",
                              "composition_r2": r2(Yte, c), "model_r2": r2(Yte, p),
                              "model_r2_on_residual": r2(Yte - c, p - c)}
    fits["per_bp"] = Wp
    summ = {}
    for res in ("per_bp", "global"):
        ch = {k: v for k, v in out["channels"].items() if v["resolution"] == res}
        fams = sorted({v["family"] for v in ch.values()})
        summ[res] = {"median_composition_r2": float(np.nanmedian([v["composition_r2"] for v in ch.values()])),
                     "median_model_r2": float(np.nanmedian([v["model_r2"] for v in ch.values()])),
                     "median_model_r2_on_residual": float(np.nanmedian([v["model_r2_on_residual"] for v in ch.values()])),
                     "by_family": {f: {k: float(np.nanmedian([v[k] for v in ch.values() if v["family"] == f]))
                                       for k in ("composition_r2", "model_r2", "model_r2_on_residual")}
                                   for f in fams}}
    out["summary"] = summ
    np.savez(os.path.join(a.run, "composition_fits.npz"), global_=fits["global"], per_bp=fits["per_bp"],
             global_names=np.array(plan.global_names), per_bp_names=np.array(plan.per_bp_names))
    json.dump(out, open(os.path.join(a.run, "composition_residual.json"), "w"), indent=1)
    print(json.dumps({r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "by_family"}
                      for r, s in summ.items()}))
    for r, s in summ.items():
        print(r, {f: {k: round(v, 3) for k, v in d.items()} for f, d in s["by_family"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
