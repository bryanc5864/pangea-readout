#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Merge `run_xtb.py --shard i --n-shards n` outputs into one summary, refusing gaps."""
from __future__ import annotations

import argparse
import glob
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob of shard JSON files")
    ap.add_argument("--index", required=True, help="the qc_fragment_index.json the shards ran")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.shards))
    parts = [json.load(open(f)) for f in files]
    n = {p["shard"][1] for p in parts}
    if len(n) != 1:
        print(f"shards disagree on n_shards: {n}", file=sys.stderr); return 2
    n = n.pop()
    have = sorted(p["shard"][0] for p in parts)
    if have != list(range(n)):
        print(f"missing shards: {sorted(set(range(n)) - set(have))}", file=sys.stderr); return 2
    want = [c["context"] for c in json.load(open(a.index))["contexts"]]
    results = sorted((r for p in parts for r in p["results"]), key=lambda r: r["context"])
    if sorted(r["context"] for r in results) != sorted(want):
        print("merged contexts do not match the index", file=sys.stderr); return 2
    ok = [r for r in results if r["status"] == "ok"]
    out = {k: parts[0][k] for k in ("xtb", "method", "note") if k in parts[0]}
    out.update({"generated_from": files, "index": a.index, "n_contexts": len(results),
                "n_ok": len(ok), "wall_seconds_sum": round(sum(p["wall_seconds"] for p in parts), 1),
                "results": results})
    vals = [r["stacking_kcal_mol"] for r in ok if "stacking_kcal_mol" in r]
    if vals:
        out["stacking_kcal_mol_range"] = [min(vals), max(vals)]
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{len(ok)}/{len(results)} contexts ok from {len(files)} shards -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
