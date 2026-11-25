# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Choose widths for the scaling ladder (PLAN.md: 2, 10, 50, 200 and 1,000 million).

Parameter counts are taken from the real model built on the real channel plan of a
dataset cache, because at the small end the output heads are a large share of the total
and depend on the channel count. Depth grows modestly with size (the shapes benchmarked in
src/model/bench.py); width is searched in steps of 16 so every attention head count
divides it; head rank is min(width, 512), at or above saturation everywhere (the per-bp
stack needs 69 linear components for 99 percent of its variance).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import InMemoryLandscape  # noqa: E402
from pangea import PANGEA, ChannelPlan  # noqa: E402

RUNGS = [(2e6, 8, 2), (10e6, 12, 4), (50e6, 12, 4), (200e6, 16, 4), (1000e6, 26, 8)]


def count(plan, width, blocks, attn, rank):
    with torch.device("meta"):
        m = PANGEA(plan, width=width, n_blocks=blocks, rank=rank, n_attn=attn,
                   attn_heads=8 if width < 2048 else 16, pair_bias=attn > 0,
                   global_readout="attn_mlp")
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    a = ap.parse_args()
    ds = InMemoryLandscape(a.shards, verbose=False)
    plan = ChannelPlan.from_metadata(ds.pb["names"], ds.gl["names"], ds.pb["parity"],
                                     ds.gl["parity"], ds.pb["family"], ds.gl["family"],
                                     ds.pb["rc_shift"])
    print(f"channel plan: {len(plan.per_bp_names)} per-bp, {len(plan.global_names)} global")
    for target, blocks, attn in RUNGS:
        best = None
        for width in range(32, 4097, 16):
            heads = 8 if width < 2048 else 16
            if width % heads:
                continue
            n = count(plan, width, blocks, attn, min(width, 512))
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (width, n)
            if n > 1.3 * target:
                break
        w, n = best
        print(f"target {target/1e6:6.0f}M: WIDTH={w} BLOCKS={blocks} ATTN={attn} "
              f"RANK={min(w, 512)} -> {n/1e6:.1f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
