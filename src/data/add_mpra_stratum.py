#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Add the MPRA stratum to the fragment library.

The plan's sampling design includes "the MPRA library sequences themselves, so every
downstream task has exact coverage".  MPRA elements are 170 to 249 bp while the solver
fragment is 61 bp, so each selected element is tiled with overlapping 61 bp windows and
the element identity and window offset are recorded.  That bookkeeping is what lets
fragment-level physics channels be reassembled into an element-level feature for the
CEILING arm; without it the solver output could not be matched back to an activity
measurement.

Elements are selected stratified by activity decile rather than at random, so the
covered set spans the full range of measured activity instead of concentrating in the
dense middle of the distribution. Tiling every element is not affordable: the lentiMPRA
K562 library alone would need about 1.5 million solves.

Adapters are stripped before tiling where the construct is documented, because the
adapter is identical across elements and would otherwise dominate the flanking context
of the terminal tiles.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fasta import gc_fraction  # noqa: E402
from data.sample_fragments import FIELDS, frag_id  # noqa: E402

FRAGMENT_BP = 61

# Documented constructs. Adapter lengths are stripped from each end before tiling.
ADAPTERS: Dict[str, Tuple[int, int]] = {
    "lentimpra": (15, 15),      # 15 nt 5' adaptor + 200 nt element + 15 nt 3' adaptor
    "deepstarr": (0, 0),
    "jores_plant": (0, 0),
}


def stratified_sample(rows: List[dict], n: int, rng: random.Random,
                      n_bins: int = 10) -> List[dict]:
    """Sample n rows spread evenly across activity deciles."""
    if n >= len(rows):
        return list(rows)
    with_target = [r for r in rows if r.get("_target") is not None]
    if not with_target:
        return rng.sample(rows, n)
    with_target.sort(key=lambda r: r["_target"])
    per_bin = max(1, n // n_bins)
    out: List[dict] = []
    size = len(with_target)
    for b in range(n_bins):
        lo = b * size // n_bins
        hi = (b + 1) * size // n_bins
        chunk = with_target[lo:hi]
        if not chunk:
            continue
        out += rng.sample(chunk, min(per_bin, len(chunk)))
    # Top up to n if rounding left room.
    if len(out) < n:
        remaining = [r for r in with_target if r not in out]
        out += rng.sample(remaining, min(n - len(out), len(remaining)))
    return out[:n]


def tile(seq: str, length: int = FRAGMENT_BP, step: int = 40) -> List[Tuple[int, str]]:
    """Overlapping windows, with the last window flush to the 3' end."""
    if len(seq) < length:
        return []
    starts = list(range(0, len(seq) - length + 1, step))
    if starts[-1] != len(seq) - length:
        starts.append(len(seq) - length)
    return [(s, seq[s:s + length]) for s in starts]


def load_activity(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path) as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            try:
                target = float(rec["target"])
            except (TypeError, ValueError, KeyError):
                target = None
            rows.append({**rec, "_target": target})
    return rows


def build(processed_dir: str, datasets: Dict[str, int], step: int,
          rng: random.Random) -> Tuple[List[dict], dict]:
    out: List[dict] = []
    report: Dict[str, object] = {}
    for name, n_elements in datasets.items():
        path = os.path.join(processed_dir, f"activity_{name}.tsv")
        if not os.path.exists(path):
            report[name] = {"status": "missing", "path": path}
            continue
        rows = load_activity(path)
        # One entry per unique element sequence; lentiMPRA stores two orientations.
        seen: Dict[str, dict] = {}
        for r in rows:
            seen.setdefault(r["sequence"], r)
        unique = list(seen.values())
        chosen = stratified_sample(unique, n_elements, rng)
        lo, hi = ADAPTERS.get(name, (0, 0))
        n_frag = 0
        skipped_short = 0
        for rec in chosen:
            seq = rec["sequence"].upper()
            core = seq[lo:len(seq) - hi] if (lo or hi) else seq
            if any(b not in "ACGT" for b in core):
                continue
            tiles = tile(core, FRAGMENT_BP, step)
            if not tiles:
                skipped_short += 1
                continue
            for offset, frag in tiles:
                out.append({
                    "stratum": "mpra",
                    "sequence": frag,
                    "source": f"{name}:{rec.get('subset', '')}",
                    "contig": rec.get("element_id", "") or "",
                    "start": offset,
                    "strand": "+",
                    "clade": rec.get("species", ""),
                })
                n_frag += 1
        report[name] = {
            "status": "ok",
            "unique_elements_available": len(unique),
            "elements_selected": len(chosen),
            "element_length_after_adapter_strip":
                len(chosen[0]["sequence"]) - lo - hi if chosen else None,
            "adapters_stripped": [lo, hi],
            "fragments_added": n_frag,
            "elements_skipped_too_short": skipped_short,
            "tile_step": step,
        }
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="dir with activity_*.tsv")
    ap.add_argument("--library", required=True, help="existing library TSV to extend")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--lentimpra", type=int, default=4000)
    ap.add_argument("--deepstarr", type=int, default=4000)
    ap.add_argument("--jores", type=int, default=2000)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    new_rows, report = build(a.processed, {
        "lentimpra": a.lentimpra,
        "deepstarr": a.deepstarr,
        "jores_plant": a.jores,
    }, a.step, rng)

    finalised = []
    for r in new_rows:
        rec = {k: r.get(k, "") for k in FIELDS}
        rec["sequence"] = r["sequence"]
        rec["length"] = len(r["sequence"])
        rec["gc"] = round(gc_fraction(r["sequence"]), 6)
        rec["frag_id"] = frag_id(r["sequence"], "mpra")
        finalised.append(rec)

    with open(a.library) as f:
        existing = list(csv.DictReader(f, delimiter="\t"))
    have = {r["frag_id"] for r in existing}
    deduped, dup = [], 0
    for r in finalised:
        if r["frag_id"] in have:
            dup += 1
            continue
        have.add(r["frag_id"])
        deduped.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
        w.writerows(deduped)

    from collections import Counter

    strata = Counter(r["stratum"] for r in existing)
    strata["mpra"] = len(deduped)
    meta = {
        "source_library": a.library,
        "seed": a.seed,
        "tile_step": a.step,
        "mpra_fragments_added": len(deduped),
        "duplicates_dropped": dup,
        "total_fragments": len(existing) + len(deduped),
        "by_stratum": dict(strata),
        "datasets": report,
    }
    with open(a.out.replace(".tsv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
