# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""External validation of Family E's DNA fold against Array Melt (measured, not solver).

Ke et al. 2025 (Nat Commun 16, doi 10.1038/s41467-025-60455-4; bioRxiv
10.1101/2024.01.08.574731) measured the equilibrium stability of 27,732 two-state DNA
hairpins on a flow cell, in 50 mM Na-HEPES pH 8.0 with 25 mM NaCl; the released
"adjusted" values carry one systematic offset (about 0.7 kcal/mol in dG37) applied by the
authors to match UV melting. Total sodium is not exactly determinable from the buffer, so
the comparison is reported in the ways that do not depend on it -- Pearson and Spearman
correlation -- and as MAE at 1 M and at 75 mM sodium, plus MAE after one fitted constant
offset, which is stated as such. Family E's hairpin channel is the DNA minimum free energy
under ViennaRNA's Mathews 2004 DNA parameters (thermodynamics.dna_fold_mfe), which is what
is scored here; nothing is fitted to the measurements except that one reported offset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import thermodynamics as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="/home/bjc75/scratch/pangea/processed/array_melt_thermo.tsv")
    ap.add_argument("--out", default="results/validation_array_melt.json")
    a = ap.parse_args()
    seqs, dg, series = [], [], []
    with open(a.table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seqs.append(r["sequence"]); dg.append(float(r["dG37_kcal_mol"]))
            series.append(r.get("series", ""))
    dg = np.asarray(dg)
    res = {"n": len(seqs), "source": "Ke et al. 2025, Array Melt, arr_v1_adjusted.csv"}
    from scipy.stats import pearsonr, spearmanr
    for na in (1.0, 0.075):
        pred = np.array([E.dna_fold_mfe(s, na_molar=na, temperature_C=37.0)[0] for s in seqs])
        ok = np.isfinite(pred)
        p, m = pred[ok], dg[ok]
        off = float(np.median(m - p))
        res[f"na_{na:g}M"] = {
            "pearson_r": float(pearsonr(p, m)[0]), "spearman_rho": float(spearmanr(p, m)[0]),
            "mae_kcal_mol": float(np.mean(np.abs(p - m))),
            "fitted_constant_offset_kcal_mol": off,
            "mae_after_offset_kcal_mol": float(np.mean(np.abs(p + off - m))),
            "rmse_after_offset_kcal_mol": float(np.sqrt(np.mean((p + off - m) ** 2)))}
        by = {}
        for s_ in sorted(set(series)):
            sel = np.array([x == s_ for x in series])[ok]
            if sel.sum() >= 200:
                by[s_] = {"n": int(sel.sum()), "pearson_r": float(pearsonr(p[sel], m[sel])[0]),
                          "mae_after_offset": float(np.mean(np.abs(p[sel] + off - m[sel])))}
        res[f"na_{na:g}M"]["by_series"] = by
        d = res[f"na_{na:g}M"]
        print(f"Na {na:g} M: n={ok.sum()}  Pearson r {d['pearson_r']:.3f}  Spearman {d['spearman_rho']:.3f}  "
              f"MAE {d['mae_kcal_mol']:.2f}  offset {off:+.2f}  MAE after offset "
              f"{d['mae_after_offset_kcal_mol']:.2f}  RMSE after offset {d['rmse_after_offset_kcal_mol']:.2f}")
        for s_, v in by.items():
            print(f"     series {s_:22s} n={v['n']:5d}  r {v['pearson_r']:.3f}  MAE(offset) {v['mae_after_offset']:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
