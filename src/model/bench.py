# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Time training steps and peak GPU memory for candidate scaling-ladder configurations.

Random one-hot inputs and random targets shaped like the real channel plan (read from a
training run's config.json), the real loss, AdamW, bf16 autocast and the batched
equivariant forward -- everything a training step does except reading real data. Used to
size the 200M and 1B rungs (single GPU or data-parallel) before spending queue time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pangea import PANGEA, ChannelPlan, LandscapeLoss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="a run's config.json, for the channel plan")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="width:blocks:rank:attn entries, e.g. 1280:16:512:4")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--pair-module", choices=("conv", "axial"), default="conv")
    ap.add_argument("--pair-channels", type=int, default=32)
    ap.add_argument("--pair-layers", type=int, default=4)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    pbn, gln = cfg["per_bp_names"], cfg["global_names"]
    # parity and family only shape the merge and the loss; a uniform plan times the same
    plan = ChannelPlan.from_metadata(pbn, gln, ["stranded"] * len(pbn), ["stranded"] * len(gln),
                                     [n[0] for n in pbn], [n[0] for n in gln], [0] * len(pbn))
    dev = "cuda"
    print(torch.cuda.get_device_name(0), flush=True)
    for spec in a.configs:
        w, b, r, at = (int(x) for x in spec.split(":"))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            m = PANGEA(plan, width=w, n_blocks=b, rank=r, window_ladder=(10, 20, 50),
                       n_attn=at, attn_heads=8 if w < 2048 else 16, pair_bias=at > 0,
                       global_readout="attn_mlp", pair_module=a.pair_module,
                       pair_channels=a.pair_channels, pair_layers=a.pair_layers).to(dev)
            n_par = sum(p.numel() for p in m.parameters()) / 1e6
            loss_fn = LandscapeLoss([n[0] for n in pbn], [n[0] for n in gln]).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.zeros(a.batch_size, 4, 61, device=dev)
            x[torch.arange(a.batch_size)[:, None], torch.randint(0, 4, (a.batch_size, 61)),
              torch.arange(61)[None, :]] = 1.0
            tgt = {"per_bp": torch.randn(a.batch_size, len(pbn), 61, device=dev),
                   "global": torch.randn(a.batch_size, len(gln), device=dev)}
            for wdw in (10, 20, 50):
                tgt[f"window_{wdw}"] = torch.randn(a.batch_size, len(pbn), 61, device=dev)
            times = []
            for i in range(a.steps):
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(x)
                loss, _ = loss_fn(out, tgt)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize(); times.append(time.time() - t)
            st = sorted(times[2:])[len(times[2:]) // 2]
            print(f"[{a.pair_module} pair, {a.pair_channels} ch x {a.pair_layers}] "
                  f"width {w:5d} blocks {b:3d} rank {r:4d} attn {at}: {n_par:8.1f}M params, "
                  f"{st*1000:7.1f} ms/step at batch {a.batch_size}, peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:6.1f} GiB, "
                  f"epoch of 181,777 = {st*181777/a.batch_size/60:6.1f} min", flush=True)
            del m, opt, out, loss
        except torch.cuda.OutOfMemoryError:
            print(f"width {w} blocks {b}: out of memory at batch {a.batch_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
