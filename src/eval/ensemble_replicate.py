# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Measured noise ceiling of the 16-snapshot relaxed-ensemble channels, by replicate.

`snapshot_ceiling.py` estimates the sampling noise of these channels from the targets
alone; for the torsion spreads that estimate assumes a unimodal angle and is only an
upper bound on the ceiling. Here the relaxed ensemble is recomputed for fragments of the
ensemble subset with an independent seed, exactly as the targets were made
(`geometry.minimised_ensemble_stats`, 16 snapshots), and compared with the stored
values. For each channel, noise variance = half the mean squared replicate difference
over (fragment, position); ceiling = 1 - noise / total variance, with the total
variance taken over the whole corpus (`results/snapshot_ceiling_v2.json`). One task
handles every `--n-tasks`-th ensemble fragment of the selected shards and writes the
paired values; `--combine` pools the task files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))

KEYS = {}                                         # stored channel -> ensemble-stats key
for _tag in ("W", "C"):
    for _t in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"):
        KEYS[f"C_{_t}_{_tag}_sd"] = f"{_t}_{_tag}_sd"
    KEYS[f"C_BI_fraction_{_tag}"] = f"BI_fraction_{_tag}"
    KEYS[f"C_sugar_C2endo_prob_{_tag}"] = f"C2endo_prob_{_tag}"
    KEYS[f"C_sugar_C3endo_prob_{_tag}"] = f"C3endo_prob_{_tag}"
SEED_OFFSET = 104729                               # any offset gives independent snapshots


def run_task(a) -> int:
    import geometry as G
    paths = sorted(glob.glob(a.shards))[: a.max_shards]
    jobs = []
    for p in paths:
        with h5py.File(p, "r") as h:
            sub = np.flatnonzero(np.asarray(h["in_ensemble_subset"][:], dtype=bool))
            for i in sub:
                jobs.append((p, int(i)))
    jobs = jobs[a.task::a.n_tasks][: a.per_task]
    out = {k: {"stored": [], "replicate": []} for k in KEYS}
    for p, i in jobs:
        with h5py.File(p, "r") as h:
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in h.attrs["per_bp_names"]]
            seq = h["sequence"][i]; seq = seq.decode() if isinstance(seq, bytes) else str(seq)
            present = [k for k in KEYS if k in names]
            stored = h["per_bp"][i][[names.index(k) for k in present], :]
        wd = tempfile.mkdtemp(prefix="ensrep_")
        try:
            me = G.minimised_ensemble_stats(seq, wd, seed=SEED_OFFSET + i)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        for k, row in zip(present, stored):
            rep = np.asarray(me.get(KEYS[k], np.full(len(row), np.nan)), dtype=float)
            out[k]["stored"].append(np.asarray(row, dtype=float).tolist())
            out[k]["replicate"].append(rep.tolist())
        print(f"{os.path.basename(p)}[{i}] done", flush=True)
        # written after every fragment, so a task cut off by its walltime keeps its pairs
        os.makedirs(a.out_dir, exist_ok=True)
        tmp = os.path.join(a.out_dir, f".task_{a.task:04d}.json.tmp")
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, os.path.join(a.out_dir, f"task_{a.task:04d}.json"))
    return 0


def combine(a) -> int:
    tot = json.load(open(a.corpus))["per_bp"]
    pooled = {k: ([], []) for k in KEYS}
    files = sorted(glob.glob(os.path.join(a.out_dir, "task_*.json")))
    for f in files:
        d = json.load(open(f))
        for k in KEYS:
            pooled[k][0].extend(np.ravel(d[k]["stored"]))
            pooled[k][1].extend(np.ravel(d[k]["replicate"]))
    res = {"n_task_files": len(files), "per_bp": {}}
    for k, (s, r) in pooled.items():
        s, r = np.asarray(s, float), np.asarray(r, float)
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < 50 or k not in tot:
            continue
        noise = 0.5 * float(np.mean((s[ok] - r[ok]) ** 2))
        total = tot[k]["total_var"]
        res["per_bp"][k] = {"n_pairs": int(ok.sum()), "noise_var": noise, "total_var": total,
                            "ceiling_r2": 1 - noise / total if total > 0 else None,
                            "replicate_r": float(np.corrcoef(s[ok], r[ok])[0, 1])}
        print(f"{k:26s} n={ok.sum():6d} ceiling {res['per_bp'][k]['ceiling_r2']:.3f} "
              f"(replicate r {res['per_bp'][k]['replicate_r']:.3f})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="/home/bjc75/scratch/pangea/targets_v2/shard_*.h5")
    ap.add_argument("--max-shards", type=int, default=64)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--out-dir", default="/nfs/roberts/scratch/pi_skr2/bjc75/pangea/ensemble_replicate")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--corpus", default="results/snapshot_ceiling_v2.json")
    ap.add_argument("--out", default="results/ensemble_replicate_ceiling_v2.json")
    a = ap.parse_args()
    return combine(a) if a.combine else run_task(a)


if __name__ == "__main__":
    raise SystemExit(main())
