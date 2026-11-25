# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Gradient norms of the landscape loss and of the base-pair loss, to set the base-pair weight.

The training step clips the *total* gradient to norm 1, so a term whose gradient is far
larger than the landscape loss's would shrink every other family's update. This trains
the 10M axial configuration on a few shards for a few hundred steps and prints, every 50
steps, the gradient norm of each term on its own.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import InMemoryLandscape, load_bpp, pooled_window_targets  # noqa: E402
from pangea import PANGEA, ChannelPlan, LandscapeLoss  # noqa: E402
from train import bpp_loss  # noqa: E402


def grad_norm(model, loss) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters()
                                if p.grad is not None)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--bpp-dir", required=True)
    ap.add_argument("--max-shards", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    dev = "cuda"
    ds = InMemoryLandscape(a.shards, max_shards=a.max_shards, verbose=False)
    plan = ChannelPlan.from_metadata(ds.pb["names"], ds.gl["names"], ds.pb["parity"],
                                     ds.gl["parity"], ds.pb["family"], ds.gl["family"],
                                     ds.pb["rc_shift"])
    X = torch.from_numpy(ds.one_hot()).to(dev)
    Ypb, Ygl = torch.from_numpy(ds.per_bp_np).to(dev), torch.from_numpy(ds.global_np).to(dev)
    Q = torch.from_numpy(load_bpp(a.bpp_dir, ds.paths, ds.sequences)).to(dev)
    torch.manual_seed(0)
    m = PANGEA(plan, width=272, n_blocks=12, rank=272, n_attn=4, attn_heads=8,
               global_readout="attn_mlp", pair_bias=True, pair_module="axial",
               pair_channels=64, pair_layers=4, pair_heads=4, strand_tie=True).to(dev)
    loss_fn = LandscapeLoss(ds.pb["family"], ds.gl["family"], beta=0.5).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    for step in range(a.steps + 1):
        b = torch.randint(0, X.shape[0], (512,), device=dev)
        tgt = {"per_bp": Ypb[b].float(), "global": Ygl[b].float()}
        for w in (10, 20, 50):
            tgt[f"window_{w}"] = pooled_window_targets(tgt["per_bp"], w)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(X[b].float(), return_bpp=True)
        lg = out.pop("bpp_logits")
        main_loss, _ = loss_fn(out, tgt)
        lb = bpp_loss(lg, Q[b])
        if step % 50 == 0:
            gm, gb = grad_norm(m, main_loss), grad_norm(m, lb)
            print(f"step {step:4d}: landscape loss {float(main_loss):8.3f} grad {gm:8.3f} | "
                  f"base-pair loss {float(lb):7.3f} grad {gb:8.3f} | ratio {gb / max(gm, 1e-9):6.2f}",
                  flush=True)
        opt.zero_grad(set_to_none=True)
        (main_loss + lb).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
