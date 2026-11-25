# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The primary gate on every READOUT dataset (PLAN.md): GC plus 5-mer ridge regression.

Features: the sequence's GC fraction and its reverse-complement-collapsed k-mer frequencies
(k = 5: 512 canonical 5-mers), standardised on the training side. Ridge with the penalty
chosen on the validation side over a log grid (closed form, one eigendecomposition). The
split is the READOUT trainer's own (`train_readout.make_split`, same seed), so the ridge,
the sequence model and the physics head are scored on exactly the same test rows.
Regression: Pearson r and R2 on the test side; classification: one-vs-rest ridge on one-hot
labels, accuracy. Writes results/readout/<dataset>_kmer_ridge.json.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
import train_readout as TR  # noqa: E402

COMP = str.maketrans("ACGT", "TGCA")


def canonical_kmers(k: int):
    out, seen = [], set()
    for t in itertools.product("ACGT", repeat=k):
        s = "".join(t); r = s.translate(COMP)[::-1]
        c = min(s, r)
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def featurise(seqs, k: int):
    kmers = canonical_kmers(k); idx = {c: i for i, c in enumerate(kmers)}
    F = np.zeros((len(seqs), 1 + len(kmers)), dtype=np.float64)
    for n, s in enumerate(seqs):
        s = s.upper(); L = len(s)
        F[n, 0] = (s.count("G") + s.count("C")) / max(1, L)
        tot = 0
        for i in range(L - k + 1):
            w = s[i:i + k]
            if any(b not in "ACGT" for b in w):
                continue
            r = w.translate(COMP)[::-1]
            F[n, 1 + idx[min(w, r)]] += 1; tot += 1
        if tot:
            F[n, 1:] /= tot
    return F


def ridge_path(Xtr, Ytr, alphas):
    """Ridge solutions for every alpha via one eigendecomposition of X'X."""
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd
    ym = Ytr.mean(0)
    w_, V = np.linalg.eigh(Z.T @ Z)
    ZtY = Z.T @ (Ytr - ym)
    sols = {}
    for a in alphas:
        B = V @ ((V.T @ ZtY) / (w_ + a * len(Z))[:, None])
        sols[a] = B
    return mu, sd, ym, sols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(TR.DATASETS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--where", nargs="*", default=[], help="column=value filters, as featurize.py")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--crop", default=None, help="START:END of each sequence, as featurize.py --crop")
    ap.add_argument("--outdir", default="results/readout")
    a = ap.parse_args()
    cfg = TR.DATASETS[a.dataset]
    rows = list(csv.DictReader(open(a.table), delimiter="\t"))
    for cond in a.where:
        k_, v_ = cond.split("=", 1); rows = [r for r in rows if r.get(k_) == v_]
    if a.max_rows:
        rows = rows[:a.max_rows]
    seqs = [r["sequence"] for r in rows]
    if a.crop:
        c0, c1 = (int(x) for x in a.crop.split(":"))
        seqs = [x[c0:c1] for x in seqs]                  # the same input every arm read
    cols = {k: [r.get(k) for r in rows] for k in rows[0].keys()}
    tr, va, te = TR.make_split(cfg, seqs, cols, seed=0)
    X = featurise(seqs, a.k)
    alphas = [10.0 ** e for e in np.arange(-4, 3.01, 0.5)]
    out = {"dataset": a.dataset, "features": f"GC + {X.shape[1] - 1} canonical {a.k}-mers",
           "crop": a.crop,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
    if cfg["task"] == "regression":
        y = np.asarray([float(v) for v in cols[cfg["target"]]])
        conds = np.asarray(cols[cfg["condition"]]) if cfg.get("condition") else np.array(["all"] * len(y))
        res = {}
        for c in sorted(set(conds)):
            m = conds == c
            trc, vac, tec = tr[m[tr]], va[m[va]], te[m[te]]
            if min(len(trc), len(vac), len(tec)) < 20:
                continue
            mu, sd, ym, sols = ridge_path(X[trc], y[trc][:, None], alphas)
            def pred(idx, B):
                return ((X[idx] - mu) / sd) @ B[:, 0] + ym[0]
            best = max(alphas, key=lambda al: np.corrcoef(pred(vac, sols[al]), y[vac])[0, 1])
            p = pred(tec, sols[best]); t = y[tec]
            r = float(np.corrcoef(p, t)[0, 1]); r2 = float(1 - np.mean((t - p) ** 2) / np.var(t))
            gc_r = float(np.corrcoef(X[tec, 0], t)[0, 1])
            res[c] = {"alpha": best, "test_pearson_r": r, "test_r2": r2, "gc_only_test_pearson_r": gc_r,
                      "n_test": int(len(tec))}
        out["results"] = res
    else:
        y = np.asarray([int(float(v)) for v in cols[cfg["target"]]])
        K = int(cfg["n_classes"]); Y = np.eye(K)[y]
        mu, sd, ym, sols = ridge_path(X[tr], Y[tr], alphas)
        def acc(idx, B):
            return float(np.mean(np.argmax(((X[idx] - mu) / sd) @ B + ym, 1) == y[idx]))
        best = max(alphas, key=lambda al: acc(va, sols[al]))
        out["results"] = {"all": {"alpha": best, "test_accuracy": acc(te, sols[best]), "n_test": int(len(te))}}
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{a.dataset}_kmer_ridge.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
