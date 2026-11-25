#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The plan's first gate: does the computed potential separate composition-matched sequences?

From PLAN.md, "Generating the ground truth":

    If every window is built as an ideal fiber model and handed to the solver, the only
    thing varying with sequence is which atoms sit at which fixed positions. The
    resulting potential becomes a near-deterministic function of base composition, and
    the whole electrostatics family collapses into GC content with thousands of
    CPU-hours of extra steps. ... Confirming that the computed potential separates
    sequences matched on base composition is the first check to run, ahead of training
    anything.

The `compctrl` stratum provides groups of fragments with identical mononucleotide
composition (and, for a subset, identical dinucleotide composition) but different base
order.  For each channel this script decomposes the variance into a between-group part,
which composition can explain, and a within-group part, which only base order can
explain, and compares the within-group part against the solver's own numerical noise
measured from grid-jitter replicates of a single sequence.

Three numbers decide the gate, per channel:

  snr        within-group SD divided by the grid-jitter SD.  Below about 5 the
             within-group variation is not distinguishable from solver noise.
  eta        within-group variance as a fraction of total variance.  This is the share
             of the channel that composition cannot account for.
  comp_r2    R-squared of an ordinary least squares fit of the channel on mononucleotide
             counts alone.  Near 1.0 means the channel is a composition feature written
             in physical notation.

The gate passes when a clear majority of electrostatics channels have snr above the
threshold and eta above the floor.  A failure is not a bug to be worked around: it means
the construction is degenerate and the ground truth has to change before any compute is
spent.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BASES = "ACGT"
DEFAULT_SNR_THRESHOLD = 5.0
DEFAULT_ETA_FLOOR = 0.05
DEFAULT_PASS_FRACTION = 0.5


def load_fragment(npz_path: str) -> Optional[dict]:
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None
    keys = set(z.files)
    out = {
        "potential": z["potential"],          # (cond, bp, sector, radius)
        "counterion": z["counterion"],
        "conditions": [str(c) for c in z["conditions"]],
        "angles": z["angles"],
        "radii": z["radii"],
        "sequence": str(z["sequence"][0]),
    }
    # Surface-referenced channels are the primary ones; see DECISIONS.md.
    if "potential_surface" in keys:
        out["potential_surface"] = z["potential_surface"]
        out["surface_radius_A"] = z["surface_radius_A"]
        out["surface_offsets_A"] = z["surface_offsets_A"]
    return out


def channel_matrix(frags: Dict[str, dict], frag_ids: Sequence[str]
                   ) -> Tuple[np.ndarray, List[str]]:
    """Flatten each fragment's potential into a channel vector.

    Channels are (condition, sector, radius), averaged over the scored base pairs, plus
    the per-condition mean counterion density.  Averaging over base pairs is deliberate:
    the gate asks whether the fragment-level electrostatics respond to base order at
    all, and a per-position channel would mix that question with positional alignment.
    """
    rows, names = [], []
    with np.errstate(invalid="ignore"):
        for k, fid in enumerate(frag_ids):
            f = frags[fid]
            vec, this_names = [], []

            pot = f["potential"]                       # (cond, bp, sector, radius)
            per_bp = np.nanmean(pot, axis=1)           # (cond, sector, radius)
            vec.append(per_bp.ravel())
            for ci_name in f["conditions"]:
                for s in range(per_bp.shape[1]):
                    for rad in f["radii"]:
                        this_names.append(f"potAxis|{ci_name}|sector{s}|r{rad:g}")

            if "potential_surface" in f:
                ps = np.nanmean(f["potential_surface"], axis=1)   # (cond, sector, off)
                vec.append(ps.ravel())
                for ci_name in f["conditions"]:
                    for s in range(ps.shape[1]):
                        for off in f["surface_offsets_A"]:
                            this_names.append(f"potSurf|{ci_name}|sector{s}|d{off:g}")
                sr = np.nanmean(f["surface_radius_A"], axis=1)    # (cond, sector)
                vec.append(sr.ravel())
                for ci_name in f["conditions"]:
                    for s in range(sr.shape[1]):
                        this_names.append(f"surfaceRadius|{ci_name}|sector{s}")

            ci = np.nanmean(f["counterion"], axis=1) if f["counterion"].size \
                else np.array([])
            if ci.size:
                vec.append(ci)
                for ci_name in f["conditions"]:
                    this_names.append(f"counterion|{ci_name}")

            rows.append(np.concatenate(vec))
            if k == 0:
                names = this_names
    Y = np.asarray(rows)
    # Drop channels that are undefined for most fragments; they carry no information and
    # would otherwise appear as NaN rows in the gate table.  The requirement is a
    # fraction of the fragments present, not an absolute count: an absolute threshold
    # silently discards every channel when the set is small, which looks like "no
    # channels survived" rather than "the threshold was wrong".
    n = Y.shape[0]
    min_finite = max(3, int(np.ceil(0.5 * n)))
    keep = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).sum() >= min_finite]
    if not keep:
        raise ValueError(
            f"no channel is defined for at least {min_finite} of {n} fragments; "
            "check that the solver output contains the expected arrays")
    return Y[:, keep], [names[j] for j in keep]


def mono_features(seqs: Sequence[str]) -> np.ndarray:
    X = np.zeros((len(seqs), 4))
    for i, s in enumerate(seqs):
        for j, b in enumerate(BASES):
            X[i, j] = s.count(b)
    return X


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    A = np.hstack([np.ones((len(X), 1)), X])
    ok = np.isfinite(y)
    if ok.sum() < A.shape[1] + 1:
        return float("nan")
    beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    resid = y[ok] - A[ok] @ beta
    ss_res = float((resid**2).sum())
    ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def variance_decomposition(values: np.ndarray, groups: Sequence[str]
                           ) -> Tuple[float, float]:
    """Return (between-group variance of group means, mean within-group variance)."""
    by = defaultdict(list)
    for v, g in zip(values, groups):
        if np.isfinite(v):
            by[g].append(v)
    means, withins = [], []
    for g, vs in by.items():
        if len(vs) < 2:
            continue
        means.append(np.mean(vs))
        withins.append(np.var(vs, ddof=1))
    if len(means) < 2:
        return float("nan"), float("nan")
    return float(np.var(means, ddof=1)), float(np.mean(withins))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory of per-fragment .npz")
    ap.add_argument("--library", required=True, help="fragment library TSV")
    ap.add_argument("--jitter-results", default=None,
                    help="directory of grid-jitter replicate .npz for one sequence")
    ap.add_argument("--role", default="all",
                    choices=["all", "mono", "dinuc"],
                    help="restrict to mononucleotide- or dinucleotide-matched members")
    ap.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD)
    ap.add_argument("--eta-floor", type=float, default=DEFAULT_ETA_FLOOR)
    ap.add_argument("--pass-fraction", type=float, default=DEFAULT_PASS_FRACTION)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.library) as f:
        lib = {r["frag_id"]: r for r in csv.DictReader(f, delimiter="\t")}

    frags, ids, groups, seqs = {}, [], [], []
    for path in sorted(glob.glob(os.path.join(a.results, "*.npz"))):
        fid = os.path.basename(path)[:-4]
        meta = lib.get(fid)
        if meta is None or meta["stratum"] != "compctrl":
            continue
        role = meta["control_role"]
        if a.role == "mono" and role not in ("seed", "mono_shuffle"):
            continue
        if a.role == "dinuc" and role not in ("seed", "dinuc_shuffle"):
            continue
        d = load_fragment(path)
        if d is None:
            continue
        frags[fid] = d
        ids.append(fid)
        groups.append(meta["control_group"])
        seqs.append(meta["sequence"])

    if len(ids) < 10:
        print(f"only {len(ids)} composition-control fragments found in {a.results}; "
              "run the solver on the compctrl stratum first", file=sys.stderr)
        return 2

    Y, names = channel_matrix(frags, ids)
    Xm = mono_features(seqs)

    # Numerical noise floor from grid-jitter replicates of a single sequence.
    jitter_sd = None
    if a.jitter_results:
        jf, jids = {}, []
        for path in sorted(glob.glob(os.path.join(a.jitter_results, "*.npz"))):
            d = load_fragment(path)
            if d is not None:
                fid = os.path.basename(path)[:-4]
                jf[fid] = d
                jids.append(fid)
        if len(jids) >= 3:
            Yj, _ = channel_matrix(jf, jids)
            jitter_sd = np.nanstd(Yj, axis=0, ddof=1)

    rows = []
    for c, name in enumerate(names):
        y = Y[:, c]
        between, within = variance_decomposition(y, groups)
        eta = within / (between + within) if np.isfinite(between + within) and \
            (between + within) > 0 else float("nan")
        within_sd = float(np.sqrt(within)) if np.isfinite(within) else float("nan")
        jsd = float(jitter_sd[c]) if jitter_sd is not None and c < len(jitter_sd) else float("nan")
        snr = within_sd / jsd if (np.isfinite(jsd) and jsd > 0) else float("nan")
        rows.append({
            "channel": name,
            "between_var": between,
            "within_var": within,
            "within_sd": within_sd,
            "jitter_sd": jsd,
            "snr": snr,
            "eta": eta,
            "comp_r2": ols_r2(Xm, y),
            "overall_sd": float(np.nanstd(y, ddof=1)),
        })

    finite_eta = [r["eta"] for r in rows if np.isfinite(r["eta"])]
    finite_snr = [r["snr"] for r in rows if np.isfinite(r["snr"])]
    finite_r2 = [r["comp_r2"] for r in rows if np.isfinite(r["comp_r2"])]

    eta_ok = np.mean([e > a.eta_floor for e in finite_eta]) if finite_eta else float("nan")
    snr_ok = np.mean([s > a.snr_threshold for s in finite_snr]) if finite_snr else float("nan")

    if not finite_snr:
        verdict = "INCONCLUSIVE: no grid-jitter noise floor supplied"
        passed = None
    else:
        passed = bool(eta_ok >= a.pass_fraction and snr_ok >= a.pass_fraction)
        verdict = "PASS" if passed else "FAIL"

    summary = {
        "verdict": verdict,
        "passed": passed,
        "role_filter": a.role,
        "n_fragments": len(ids),
        "n_groups": len(set(groups)),
        "n_channels": len(rows),
        "thresholds": {
            "snr_threshold": a.snr_threshold,
            "eta_floor": a.eta_floor,
            "pass_fraction": a.pass_fraction,
        },
        "fraction_channels_eta_above_floor": None if not finite_eta else float(eta_ok),
        "fraction_channels_snr_above_threshold": None if not finite_snr else float(snr_ok),
        "median_eta": float(np.median(finite_eta)) if finite_eta else None,
        "median_snr": float(np.median(finite_snr)) if finite_snr else None,
        "median_composition_r2": float(np.median(finite_r2)) if finite_r2 else None,
        "max_composition_r2": float(np.max(finite_r2)) if finite_r2 else None,
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "channels": rows}, f, indent=2)
    csv_path = a.out.replace(".json", "_channels.tsv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"\nper-channel detail -> {csv_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
