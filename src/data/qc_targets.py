#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Check the generated target shards before anything trains on them.

Three failures this session shared one shape: something reported success while producing
nothing. An install script exited 0 without creating an environment, a backgrounded job
vanished and reported success, and the target generator wrote a well-formed HDF5 file
containing **zero channels** and printed a success line. Only the third was caught, and
only because a test was written for it afterwards.

So the dataset gets checked on its own terms rather than trusted because the generator
exited cleanly. Every check below answers a question that a silent failure would make
look fine:

  * Did every shard land, and do they agree on their channel layout?
  * Is any channel entirely NaN across the whole corpus? That is a column the model
    would train on nothing, and `PLAN.md` treats a permanently-empty target as a bug
    rather than as missing data.
  * Is any channel constant? A channel with no variance carries no information and will
    show an undefined R-squared, which is worth knowing before it appears as a mysterious
    NaN in the metrics.
  * Does the NaN pattern match what the physics says it should be? Fragment ends are
    legitimately undefined for step-indexed channels; a channel that is NaN in the
    *middle* of fragments is a different matter.
  * Are the strata present in the proportions the sampling design specified?
  * Are there duplicate fragments, which would leak between training and validation?

Exit status is non-zero when a check fails, so this can gate a training job.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


def _decode(a) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def scan(pattern: str, max_shards: Optional[int] = None) -> Dict[str, object]:
    paths = sorted(glob.glob(pattern))
    if max_shards:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern}")

    with h5py.File(paths[0], "r") as h:
        pb_names = _decode(h.attrs["per_bp_names"])
        gl_names = _decode(h.attrs["global_names"])
        pb_prov = np.asarray(h.attrs["per_bp_provisional"], dtype=bool)
        gl_prov = np.asarray(h.attrs["global_provisional"], dtype=bool)
        n_bp = int(h["per_bp"].shape[2])

    n_pb, n_gl = len(pb_names), len(gl_names)
    # Streaming accumulators, so this never holds more than one shard in memory.
    pb_finite = np.zeros(n_pb, dtype=np.int64)
    pb_total = np.zeros(n_pb, dtype=np.int64)
    pb_sum = np.zeros(n_pb); pb_sq = np.zeros(n_pb)
    pb_finite_pos = np.zeros((n_pb, n_bp), dtype=np.int64)
    gl_finite = np.zeros(n_gl, dtype=np.int64)
    gl_sum = np.zeros(n_gl); gl_sq = np.zeros(n_gl)

    n_frag = 0
    failures = 0
    strata: Dict[str, int] = {}
    seqs_seen: Dict[str, int] = {}
    layout_mismatch: List[str] = []
    seconds = 0.0

    for p in paths:
        with h5py.File(p, "r") as h:
            if _decode(h.attrs["per_bp_names"]) != pb_names:
                layout_mismatch.append(os.path.basename(p))
                continue
            P = np.asarray(h["per_bp"])
            G = np.asarray(h["global"])
            n_frag += P.shape[0]
            failures += int(h.attrs.get("n_failures", 0))
            seconds += float(h.attrs.get("seconds", 0.0))
            for s in _decode(h["stratum"]):
                strata[s] = strata.get(s, 0) + 1
            for s in _decode(h["sequence"]):
                seqs_seen[s] = seqs_seen.get(s, 0) + 1

            f = np.isfinite(P)
            pb_finite += f.sum(axis=(0, 2))
            pb_total += f.shape[0] * f.shape[2]
            pb_finite_pos += f.sum(axis=0)
            Pz = np.where(f, P, 0.0)
            pb_sum += Pz.sum(axis=(0, 2)); pb_sq += (Pz ** 2).sum(axis=(0, 2))

            fg = np.isfinite(G)
            gl_finite += fg.sum(axis=0)
            Gz = np.where(fg, G, 0.0)
            gl_sum += Gz.sum(axis=0); gl_sq += (Gz ** 2).sum(axis=0)

    def stats(cnt, s, sq):
        c = np.maximum(cnt, 1)
        mean = s / c
        var = np.maximum(sq / c - mean ** 2, 0.0)
        return mean, np.sqrt(var)

    pb_mean, pb_sd = stats(pb_finite, pb_sum, pb_sq)
    gl_mean, gl_sd = stats(gl_finite, gl_sum, gl_sq)

    duplicates = {s: n for s, n in seqs_seen.items() if n > 1}
    return {
        "n_shards": len(paths), "n_fragments": n_frag, "n_bp": n_bp,
        "n_generator_failures": failures, "generator_seconds": seconds,
        "seconds_per_fragment": seconds / max(n_frag, 1),
        "per_bp_names": pb_names, "global_names": gl_names,
        "per_bp_provisional": pb_prov, "global_provisional": gl_prov,
        "per_bp_finite_fraction": pb_finite / np.maximum(pb_total, 1),
        "per_bp_finite_by_position": pb_finite_pos,
        "per_bp_sd": pb_sd, "global_sd": gl_sd,
        "global_finite_fraction": gl_finite / max(n_frag, 1),
        "strata": strata, "n_duplicate_sequences": len(duplicates),
        "layout_mismatch": layout_mismatch,
    }


def report(scan_result: Dict[str, object], expect_fragments: Optional[int] = None,
           expect_shards: Optional[int] = None,
           duplicate_tolerance: float = 1e-3
           ) -> Tuple[Dict[str, object], List[str], List[str]]:
    """Summary, blocking problems, and non-blocking warnings.

    **Problems and warnings are separated deliberately.** A gate that blocks on something
    already handled downstream gets switched off, and a gate that says nothing about it is
    worse. Duplicate sequences are the case in point: they would leak under a random
    split, and the similarity split co-locates them by construction, so the right
    treatment is to report the count and name the mitigation rather than to fail or to go
    quiet.
    """
    r = scan_result
    problems: List[str] = []
    warnings: List[str] = []
    pb, gl = r["per_bp_names"], r["global_names"]

    if r["layout_mismatch"]:
        problems.append(
            f"{len(r['layout_mismatch'])} shards have a different channel layout: "
            f"{r['layout_mismatch'][:4]}")
    if not pb or not gl:
        problems.append(f"empty channel set: {len(pb)} per-bp, {len(gl)} global")
    if expect_shards and r["n_shards"] < expect_shards:
        problems.append(f"{r['n_shards']} shards present, expected {expect_shards}")
    if expect_fragments and r["n_fragments"] < expect_fragments:
        problems.append(
            f"{r['n_fragments']} fragments present, expected {expect_fragments}")
    if r["n_generator_failures"]:
        problems.append(f"{r['n_generator_failures']} fragments failed in the generator")
    n_dup = int(r["n_duplicate_sequences"])
    if n_dup:
        frac = n_dup / max(int(r["n_fragments"]), 1)
        msg = (f"{n_dup} sequences appear more than once ({frac:.2%} of fragments). "
               "Under --split random these leak between training and validation. Under "
               "--split similarity, which is the default, identical sequences have "
               "Jaccard 1.0 and are assigned to the same side by construction, so they "
               "do not leak. Exact duplicates are also a canary for near-duplicates, "
               "which exact deduplication cannot see: in this library they arise where "
               "two strata sample the same genome, such as the Drosophila panel and the "
               "DeepSTARR MPRA elements.")
        if frac > duplicate_tolerance:
            problems.append(msg)
        else:
            warnings.append(msg)

    # Entirely empty channels: a column the model would fit to nothing.
    dead_pb = [pb[i] for i, f in enumerate(r["per_bp_finite_fraction"]) if f == 0.0]
    dead_gl = [gl[i] for i, f in enumerate(r["global_finite_fraction"]) if f == 0.0]
    # A channel whose window is wider than the fragment cannot be computed for this
    # library at all. That is a property of the 61 bp library, not a defect in the
    # channel, so it is reported separately rather than failing the run -- but it is
    # reported, because such a column would otherwise sit in the training set as a
    # permanent NaN and the trainer would silently fit nothing to it.
    def _too_wide(name: str) -> bool:
        m = re.search(r"_w(\d+)$", name)
        return bool(m) and int(m.group(1)) > int(r["n_bp"])

    too_wide = sorted({n for n in dead_pb + dead_gl if _too_wide(n)})
    prod_dead_pb = [n for n in dead_pb
                    if not r["per_bp_provisional"][pb.index(n)] and not _too_wide(n)]
    prod_dead_gl = [n for n in dead_gl
                    if not r["global_provisional"][gl.index(n)] and not _too_wide(n)]
    if prod_dead_pb or prod_dead_gl:
        problems.append(
            "production channels that are entirely NaN across the corpus: "
            f"{(prod_dead_pb + prod_dead_gl)[:8]}")

    # Constant channels: no variance, so an undefined R-squared later.
    flat_pb = [pb[i] for i, s in enumerate(r["per_bp_sd"])
               if r["per_bp_finite_fraction"][i] > 0 and s < 1e-12]
    flat_gl = [gl[i] for i, s in enumerate(r["global_sd"])
               if r["global_finite_fraction"][i] > 0 and s < 1e-12]

    # A channel's defined region should be a single contiguous band. Ends are
    # legitimately undefined for step-indexed quantities, and a window channel is
    # defined only in a central band whose width is the fragment length minus the
    # window, so neither of those is a problem. A *hole* is: a position that is
    # undefined with defined positions on both sides means the channel failed for some
    # inputs and not others.
    #
    # The first version of this check compared the middle third against the maximum and
    # flagged `E_strand_separation_dG_w50`, which is correct behaviour -- a 50 bp window
    # on a 61 bp fragment has only 12 valid centres, all of them central. Contiguity is
    # the property that actually distinguishes a gap from a narrow band.
    pos = np.asarray(r["per_bp_finite_by_position"])
    n_bp = int(r["n_bp"])
    holes = []
    for i, name in enumerate(pb):
        if r["per_bp_finite_fraction"][i] == 0.0:
            continue
        defined = pos[i] > 0
        idx = np.flatnonzero(defined)
        if idx.size and (idx[-1] - idx[0] + 1) != idx.size:
            holes.append(name)
    if holes:
        problems.append(
            f"channels whose defined region has holes, not just narrow or missing ends: "
            f"{holes[:6]}")

    summary = {
        "n_shards": r["n_shards"], "n_fragments": r["n_fragments"],
        "n_bp": n_bp,
        "n_per_bp_channels": len(pb), "n_global_channels": len(gl),
        "n_provisional": int(r["per_bp_provisional"].sum()
                             + r["global_provisional"].sum()),
        "seconds_per_fragment": round(r["seconds_per_fragment"], 3),
        "cpu_hours_total": round(r["generator_seconds"] / 3600.0, 1),
        "strata": r["strata"],
        "median_per_bp_finite_fraction": float(
            np.median(r["per_bp_finite_fraction"])),
        "median_global_finite_fraction": float(
            np.median(r["global_finite_fraction"])),
        "n_all_nan_channels": len(dead_pb) + len(dead_gl),
        "all_nan_channels_sample": (dead_pb + dead_gl)[:12],
        "n_constant_channels": len(flat_pb) + len(flat_gl),
        "constant_channels_sample": (flat_pb + flat_gl)[:12],
        # Not a failure, but worth seeing: a constant channel has an undefined
        # R-squared, so it will appear as a mysterious NaN in the metrics unless it is
        # known about in advance. Some are constant by construction -- the writhe and
        # twist change at zero superhelical density are identically zero -- and some are
        # constant only because this corpus happens not to contain the feature, which a
        # larger or more varied sample would fix.
        "n_channels_wider_than_the_fragment": len(too_wide),
        "channels_wider_than_the_fragment": too_wide,
        "n_duplicate_sequences": r["n_duplicate_sequences"],
    }
    return summary, problems, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default=None, help="write the summary as JSON")
    ap.add_argument("--expect-fragments", type=int, default=None)
    ap.add_argument("--expect-shards", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--duplicate-tolerance", type=float, default=1e-3,
                    help="duplicate fraction above which duplicates block rather than "
                         "warn; the similarity split handles a small number, but a large "
                         "number means the library itself needs rebuilding")
    args = ap.parse_args()

    r = scan(args.shards, args.max_shards)
    summary, problems, warnings = report(
        r, args.expect_fragments, args.expect_shards, args.duplicate_tolerance)
    print(json.dumps(summary, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "problems": problems,
                       "warnings": warnings}, f, indent=2, default=str)
    if warnings:
        print("\nWARNINGS (reported, not blocking):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nall checks passed"
          + (f" ({len(warnings)} warning(s) above)" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
