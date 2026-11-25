# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Measure the R-squared ceiling that Monte Carlo sampling noise places on each channel.

Every stage-1 target is a deterministic function of the sequence except the ones that
come from sampling the cgNA+ Gaussian ensemble, whose values move with the random seed.
For those, a perfect model of the expected value still scores below R-squared 1, because
the target it is scored against carries sampling noise. This recomputes the landscape for
a sample of fragments under two seeds and reports, per channel,

    noise variance   = mean((y_seed1 - y_seed2)^2) / 2
    ceiling          = 1 - noise variance / total variance across the corpus

on the same scale the training R-squared uses (log10 for the channels the dataset
log-transforms). Channels whose two seeds agree exactly are deterministic: ceiling 1.

Minimisation is skipped (minimised_pdb=None), so channels that need the relaxed
structure are NaN here and excluded; they are deterministic given the structure anyway.

Usage:
    python src/eval/noise_ceiling.py --shards '.../shard_*.h5' --n 300 --out results/x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))

import landscape as LS                      # noqa: E402
from channels import Resolution             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed-a", type=int, default=11)
    ap.add_argument("--seed-b", type=int, default=12345)
    ap.add_argument("--n-ensemble", type=int, default=200)
    ap.add_argument("--log-channels-from", default=None,
                    help="config.json of a training run, to apply its log10 transforms")
    ap.add_argument("--transforms-from-dataset", action="store_true",
                    help="read the log10 list from the dataset cache of --shards instead "
                         "of a run's config.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = sorted(glob.glob(a.shards))
    rng = np.random.default_rng(0)
    seqs, totals_pb, totals_gl = [], [], []
    per = max(1, a.n // min(len(paths), 64) + 1)
    for p in rng.choice(paths, size=min(len(paths), 64), replace=False):
        with h5py.File(p, "r") as h:
            names_pb = [x.decode() if isinstance(x, bytes) else x for x in h.attrs["per_bp_names"]]
            names_gl = [x.decode() if isinstance(x, bytes) else x for x in h.attrs["global_names"]]
            k = h["per_bp"].shape[0]
            pick = rng.choice(k, size=min(per, k), replace=False)
            s = h["sequence"][:]
            seqs += [(x.decode() if isinstance(x, bytes) else x) for x in s[np.sort(pick)]]
            totals_pb.append(h["per_bp"][:])          # corpus variance comes from all rows
            totals_gl.append(h["global"][:])
    seqs = seqs[:a.n]
    P_all = np.concatenate(totals_pb)
    G_all = np.concatenate(totals_gl)

    logs = set()
    if a.transforms_from_dataset:
        # The dataset cache stores its transforms in a JSON field; read just that with
        # numpy, so this runs in the physics environment, which has no torch.
        caches = sorted(glob.glob(os.path.join(os.path.dirname(a.shards),
                                               "landscape_cache_*.npz")),
                        key=os.path.getmtime)
        if not caches:
            raise FileNotFoundError("no dataset cache beside the shards; build it first")
        with np.load(caches[-1], allow_pickle=True) as z:
            tr = json.loads(str(z["meta"]))["transforms"]
        logs = set(tr.get("per_bp_log10", [])) | set(tr.get("global_log10", []))
        print(f"log10 transforms from {os.path.basename(caches[-1])}: {len(logs)} channels")
    elif a.log_channels_from:
        cfg = json.load(open(a.log_channels_from))
        tr = cfg.get("transforms", {})
        logs = set(tr.get("per_bp_log10", [])) | set(tr.get("global_log10", []))

    pb_names = LS.channel_names(Resolution.PER_BP)
    gl_names = LS.channel_names(Resolution.GLOBAL)
    t0 = time.time()
    A_pb, B_pb, A_gl, B_gl = [], [], [], []
    for i, sq in enumerate(seqs):
        pa, ga = LS.landscape_arrays(sq, pb_names, gl_names, minimised_pdb=None,
                                     n_ensemble=a.n_ensemble, seed=a.seed_a + i)
        pb_, gb_ = LS.landscape_arrays(sq, pb_names, gl_names, minimised_pdb=None,
                                       n_ensemble=a.n_ensemble, seed=a.seed_b + i)
        A_pb.append(pa); B_pb.append(pb_); A_gl.append(ga); B_gl.append(gb_)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seqs)}  {(time.time()-t0)/(i+1):.2f} s/fragment", flush=True)
    A_pb, B_pb = np.stack(A_pb), np.stack(B_pb)
    A_gl, B_gl = np.stack(A_gl), np.stack(B_gl)

    def tf(name, v):
        if name in logs:
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.log10(np.where(v > 0, v, np.nan))
        return v

    out = {"n_fragments": len(seqs), "n_ensemble": a.n_ensemble, "per_bp": {}, "global": {}}
    for key, names_now, A, B, names_corpus, C in (
            ("per_bp", pb_names, A_pb, B_pb, names_pb, P_all),
            ("global", gl_names, A_gl, B_gl, names_gl, G_all)):
        for j, nm in enumerate(names_now):
            if nm not in names_corpus:
                continue
            a_, b_ = tf(nm, A[:, j]), tf(nm, B[:, j])
            ok = np.isfinite(a_) & np.isfinite(b_)
            if ok.sum() < 10:
                continue
            noise = 0.5 * float(np.mean((a_[ok] - b_[ok]) ** 2))
            c = tf(nm, C[:, names_corpus.index(nm)])
            c = c[np.isfinite(c)]
            tot = float(np.var(c)) if c.size else float("nan")
            if noise == 0.0:
                continue                                  # deterministic channel
            out[key][nm] = {"noise_var": noise, "total_var": tot,
                            "ceiling_r2": (1.0 - noise / tot) if tot > 0 else None}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for key in ("per_bp", "global"):
        items = sorted(out[key].items(), key=lambda kv: kv[1]["ceiling_r2"] if kv[1]["ceiling_r2"] is not None else 2)
        print(f"{key}: {len(items)} seed-dependent channels; lowest ceilings:")
        for nm, d in items[:15]:
            print(f"   {nm:40s} ceiling R2 {d['ceiling_r2']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
