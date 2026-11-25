# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""External validation of Family D's cyclizability prior against loop-seq (measured).

Basu et al. 2021 (Nature 589:462) measured the intrinsic cyclizability C0 of 50 bp
sequences. Family D's cyclizability channels are an analytic prior -- a Shimada-Yamakawa
ring-closure estimate from the cgNA+ persistence length, rise, torsional modulus and net
curvature -- marked provisional because the plan supervises their residual on loop-seq at
the delta-learning stage. This measures how much of the measured C0 the prior already
carries, on the random library (the cleanest: no genomic or nucleosomal selection), for a
deterministic subsample. Nothing is fitted; correlations are reported per channel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import mesoscale as D  # noqa: E402

CHANNELS = ("D_cyclizability_L63", "D_cyclizability_L73", "D_cyclizability_L84",
            "D_cyclizability_L94", "D_persistence_length_nm", "D_net_bend_deg",
            "D_curvature_shortening", "D_mean_bend_variance_rad2")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="/home/bjc75/scratch/pangea/processed/loopseq_c0.tsv")
    ap.add_argument("--library", default="random")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="results/validation_loopseq.json")
    a = ap.parse_args()
    rows = []
    with open(a.table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["library"] == a.library and set(r["sequence"].upper()) <= set("ACGT"):
                rows.append((r["sequence"].upper(), float(r["target"])))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)
    rows = [rows[i] for i in sorted(pick)]
    vals = {c: [] for c in CHANNELS}
    c0 = []
    t0 = time.time()
    for i, (s, y) in enumerate(rows):
        try:
            out = D.mesoscale_channels(s)
        except Exception:                                          # noqa: BLE001
            continue
        c0.append(y)
        for c in CHANNELS:
            vals[c].append(float(out.get(c, np.nan)))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f} s/seq", flush=True)
    from scipy.stats import pearsonr, spearmanr
    c0 = np.asarray(c0)
    res = {"n": int(len(c0)), "library": a.library,
           "source": "Basu et al. 2021, loop-seq C0 (intrinsic cyclizability)", "channels": {}}
    for c in CHANNELS:
        v = np.asarray(vals[c])
        ok = np.isfinite(v) & np.isfinite(c0)
        if ok.sum() < 50 or np.std(v[ok]) == 0:
            continue
        res["channels"][c] = {"pearson_r": float(pearsonr(v[ok], c0[ok])[0]),
                              "spearman_rho": float(spearmanr(v[ok], c0[ok])[0]),
                              "n": int(ok.sum())}
        d = res["channels"][c]
        print(f"{c:28s} r {d['pearson_r']:+.3f}  rho {d['spearman_rho']:+.3f}  (n={d['n']})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
