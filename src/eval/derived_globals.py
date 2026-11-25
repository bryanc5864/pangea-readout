# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Whole-sequence channels that are exact functionals of per-bp channels, read from the per-bp output.

The window outputs improved from 0.94-0.97 to 0.98-0.996 when read as the pooled per-bp
output instead of from dedicated heads, because per-bp is the model's strongest output.
Many whole-sequence channels are the same kind of object -- a mean, maximum or minimum
over positions of a per-bp channel (C_mean_BI_fraction_W, G_max_cleavage_C,
L_p_mean_positional_entropy, ...). This finds them and scores both readouts.

1. Matching, on training fragments' *targets* only: for every whole-sequence channel G
   and every per-bp channel p of the same family, and every functional f in {mean, max,
   min, sd} over the positions where p is defined, the squared correlation of raw G with
   f(raw p). A pair with R2 >= 0.999 is an exact functional; the affine map G = a f + b
   is fitted on the same training targets.
2. Scoring: the derived prediction is a * f(raw per-bp prediction) + b, standardised
   like the target; R-squared on val and test beside the whole-sequence head's. The
   readout is chosen per channel on val (model selection); test is reported.
Writes <run>/derived_globals.json.
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

FUNCTIONALS = ("mean", "max", "min", "sd")


def to_raw(z: np.ndarray, info: dict, cols) -> np.ndarray:
    """Standardised values -> raw units, channel by channel (axis 1)."""
    out = np.empty_like(z, dtype=np.float64)
    for k, j in enumerate(cols):
        x = z[:, k].astype(np.float64) * info["std"][j] + info["mean"][j]
        if info["transform"][j] == "log10":
            x = 10.0 ** x
        out[:, k] = x
    return out


def functional(x: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """x (n, C, L) raw; mask (C, L) positions where each channel is defined."""
    xm = np.where(mask[None], x, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin,
                    "sd": np.nanstd}[how](xm, axis=2)


def pairwise_r2(F: np.ndarray, G: np.ndarray):
    """Squared Pearson correlation for every (per-bp column, global column), pairwise-complete."""
    Mf, Mg = np.isfinite(F).astype(np.float64), np.isfinite(G).astype(np.float64)
    F0, G0 = np.nan_to_num(F), np.nan_to_num(G)
    n = Mf.T @ Mg
    sx, sy = F0.T @ Mg, Mf.T @ G0
    sxx, syy, sxy = (F0 ** 2).T @ Mg, Mf.T @ (G0 ** 2), F0.T @ G0
    with np.errstate(all="ignore"):
        cov = sxy / n - sx * sy / n ** 2
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        r2 = cov ** 2 / (vx * vy)
    r2[(n < 200) | ~np.isfinite(r2) | (vx < 1e-12) | (vy < 1e-12)] = 0.0
    return r2


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-match", type=int, default=20000, help="training fragments for matching")
    ap.add_argument("--exact", type=float, default=0.999)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json"))); args = cfg["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = landscape_from_args(args)
    pb, gl = ds.pb, ds.gl
    plan = ChannelPlan.from_metadata(pb["names"], gl["names"], pb["parity"], gl["parity"],
                                     pb["family"], gl["family"], pb["rc_shift"])
    sp = np.load(os.path.join(a.run, "split_indices.npz"))
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(sp["train"], size=min(a.n_match, len(sp["train"])), replace=False))
    lin = [j for j, t in enumerate(pb["transform"]) if t in ("identity", "log10")]
    glin = [j for j, t in enumerate(gl["transform"]) if t in ("identity", "log10")]

    # 1. matching on training targets
    P = to_raw(ds.per_bp_np[tr][:, lin].astype(np.float32), pb, lin)       # (n, C', L)
    Gr = to_raw(ds.global_np[tr][:, glin].astype(np.float32), gl, glin)    # (n, G')
    defined = np.isfinite(P)
    any_ = defined.any(axis=2)                                            # fragments with the channel
    mask = (defined.sum(0) >= 0.5 * np.maximum(any_.sum(0), 1)[:, None])  # (C', L) structural
    fam_pb = np.array([pb["family"][j] for j in lin]); fam_gl = np.array([gl["family"][j] for j in glin])
    best = {}
    for how in FUNCTIONALS:
        F = functional(P, mask, how)                                      # (n, C')
        r2 = pairwise_r2(F, Gr)
        r2[fam_pb[:, None] != fam_gl[None, :]] = 0.0
        for g in range(len(glin)):
            i = int(np.argmax(r2[:, g]))
            if r2[i, g] >= a.exact and r2[i, g] > best.get(g, (0,))[0]:
                f, y = F[:, i], Gr[:, g]
                ok = np.isfinite(f) & np.isfinite(y)
                slope, icpt = np.polyfit(f[ok], y[ok], 1)
                best[g] = (float(r2[i, g]), i, how, float(slope), float(icpt))
    print(f"{len(best)} of {len(glin)} whole-sequence channels are exact functionals "
          f"(R2 >= {a.exact}) of a per-bp channel", flush=True)

    # 2. scoring
    model = PANGEA(plan, **model_kwargs_from_args(args)).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(a.run, "best_ema.pt"), map_location=dev,
                                     weights_only=False)["model"])
    X = torch.from_numpy(ds.one_hot()).to(dev)
    res = {"n_exact": len(best), "n_candidates": len(glin), "channels": {}}
    sides = {}
    for side in ("val", "test"):
        idx = sp[side]
        head, pbp = [], []
        for s in range(0, len(idx), 1024):
            b = torch.as_tensor(idx[s:s + 1024], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
                out = model(X[b].float())
            head.append(out["global"][0].float().cpu().numpy())
            pbp.append(out["per_bp"][0].float().cpu().numpy())
        head = np.concatenate(head); pbp = np.concatenate(pbp)
        Pp = to_raw(pbp[:, lin], pb, lin)
        Y = ds.global_np[idx].astype(np.float64)
        sides[side] = {}
        for g, (r2m, i, how, slope, icpt) in best.items():
            j = glin[g]
            d_raw = slope * functional(Pp[:, i:i + 1], mask[i:i + 1], how)[:, 0] + icpt
            if gl["transform"][j] == "log10":
                d_raw = np.log10(np.where(d_raw > 0, d_raw, np.nan))
            d = (d_raw - gl["mean"][j]) / gl["std"][j]
            y = Y[:, j]
            ok = np.isfinite(y) & np.isfinite(d)
            if ok.sum() < 50:
                continue
            sst = np.sum((y[ok] - y[ok].mean()) ** 2)
            r2_d = 1 - np.sum((d[ok] - y[ok]) ** 2) / sst
            okh = np.isfinite(y)
            r2_h = 1 - np.sum((head[okh, j] - y[okh]) ** 2) / np.sum((y[okh] - y[okh].mean()) ** 2)
            sides[side][gl["names"][j]] = (float(r2_h), float(r2_d))
    for name in sides["val"]:
        g = [g for g in best if gl["names"][glin[g]] == name][0]
        r2m, i, how, slope, icpt = best[g]
        hv, dv = sides["val"][name]; ht, dt = sides["test"].get(name, (np.nan, np.nan))
        use = "derived" if dv > hv else "head"
        res["channels"][name] = {"functional": how, "per_bp": pb["names"][lin[i]],
                                 "match_r2": r2m, "val": {"head": hv, "derived": dv},
                                 "test": {"head": ht, "derived": dt}, "chosen_on_val": use,
                                 "test_chosen": dt if use == "derived" else ht}
    ch = res["channels"]
    for side in ("val", "test"):
        h = np.array([c[side]["head"] for c in ch.values()]); d = np.array([c[side]["derived"] for c in ch.values()])
        print(f"[{side}] {len(ch)} functional channels: head median {np.median(h):.4f}, derived "
              f"median {np.median(d):.4f}; derived better on {int((d > h).sum())}", flush=True)
    t = np.array([c["test_chosen"] for c in ch.values()]); th = np.array([c["test"]["head"] for c in ch.values()])
    print(f"[test] chosen on val: median {np.median(t):.4f} (head alone {np.median(th):.4f}); "
          f"worst head {np.min(th):.4f} -> chosen {t[np.argmin(th)]:.4f}", flush=True)
    for name, c in sorted(ch.items(), key=lambda kv: kv[1]["test"]["head"])[:15]:
        print(f"   {name:40s} = {c['functional']:4s}({c['per_bp']:28s}) head {c['test']['head']:.4f} "
              f"-> derived {c['test']['derived']:.4f}  [{c['chosen_on_val']}]", flush=True)
    json.dump(res, open(os.path.join(a.run, "derived_globals.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
