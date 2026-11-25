# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The physics landscape of arbitrary sequences, from a frozen PANGEA encoder (Aim 2 input).

PANGEA is trained on 61-bp molecules, so the landscape of a longer sequence is defined
window by window: 61-bp windows every `--stride` bp (the last one flush with the end),
each window's per-bp output stitched into the full track with triangular weights that
favour window centres (a position near a window edge sees less of its own context).
Sequences shorter than 61 bp are completed with constant flanks (`--left`/`--right`;
for loop-seq, the construct's own adapters) and only their own positions are kept unless
`--keep-flanks`. Outputs are the model's merged per-bp means in standardised units --
the channels a READOUT head reads -- as float16 (n, channels, length), plus each
sequence's whole-sequence channels averaged over its windows. Writes one .npz per input
chunk so long tables can be split across jobs (`--chunk`, `--n-chunks`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from data import channel_meta_from_args  # noqa: E402
from pangea import PANGEA, ChannelPlan, model_kwargs_from_args, one_hot  # noqa: E402

WINDOW = 61


def load_encoder(run: str, device: str):
    cfg = json.load(open(os.path.join(run, "config.json")))
    args = cfg["args"]
    pb, gl = channel_meta_from_args(args)
    plan = ChannelPlan.from_metadata(pb["names"], gl["names"], pb["parity"], gl["parity"],
                                     pb["family"], gl["family"], pb["rc_shift"])
    model = PANGEA(plan, **model_kwargs_from_args(args)).to(device).eval()
    model.window_readout = "pooled"
    ck = torch.load(os.path.join(run, "best_ema.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model, pb["names"], gl["names"]


def tiles(L: int, stride: int):
    if L <= WINDOW:
        return [0]
    starts = list(range(0, L - WINDOW + 1, stride))
    if starts[-1] != L - WINDOW:
        starts.append(L - WINDOW)
    return starts


@torch.no_grad()
def featurize(model, seqs, stride: int, left: str, right: str, keep_flanks: bool,
              batch: int, device: str):
    n_pb = len(model.plan.per_bp_names); n_gl = len(model.plan.global_names)
    w = 1.0 - np.abs(np.arange(WINDOW) - (WINDOW - 1) / 2) / ((WINDOW + 1) / 2)  # triangle
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("featurize one sequence length per call")
    pad_l = pad_r = 0
    if L < WINDOW:
        need = WINDOW - L
        pad_l = min(len(left), need // 2 + need % 2); pad_r = need - pad_l
        if pad_r > len(right):
            raise ValueError("flanks too short to complete a 61-bp window")
        seqs = [left[len(left) - pad_l:] + s + right[:pad_r] for s in seqs]
    Lw = len(seqs[0])
    starts = tiles(Lw, stride)
    out_pb = np.zeros((len(seqs), n_pb, Lw), dtype=np.float32)
    wsum = np.zeros(Lw, dtype=np.float32)
    for st in starts:
        wsum[st:st + WINDOW] += w
    out_gl = np.zeros((len(seqs), n_gl), dtype=np.float32)
    windows = [(i, st) for i in range(len(seqs)) for st in starts]
    for b0 in range(0, len(windows), batch):
        chunk = windows[b0:b0 + batch]
        x = one_hot([seqs[i][st:st + WINDOW] for i, st in chunk]).to(device)
        out = model(x.float())
        pbm = out["per_bp"][0].float().cpu().numpy()
        glm = out["global"][0].float().cpu().numpy()
        for k, (i, st) in enumerate(chunk):
            out_pb[i, :, st:st + WINDOW] += pbm[k] * w[None, :]
            out_gl[i] += glm[k] / len(starts)
    out_pb /= wsum[None, None, :]
    if L < WINDOW and not keep_flanks:
        out_pb = out_pb[:, :, pad_l:pad_l + L]
    return out_pb.astype(np.float16), out_gl.astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="encoder run directory")
    ap.add_argument("--table", required=True)
    ap.add_argument("--where", nargs="*", default=[], help="column=value filters")
    ap.add_argument("--stride", type=int, default=31)
    ap.add_argument("--left", default=""); ap.add_argument("--right", default="")
    ap.add_argument("--keep-flanks", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--crop", default=None, help="START:END (0-based, end exclusive) of each "
                    "sequence to featurize, e.g. the centre of a splice-site window")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    filt = dict(kv.split("=", 1) for kv in a.where)
    rows = []
    with open(a.table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if all(r.get(k) == v for k, v in filt.items()):
                rows.append(r)
                if a.max_rows and len(rows) >= a.max_rows:
                    break
    rows = rows[a.chunk::a.n_chunks]
    model, pbn, gln = load_encoder(a.run, dev)
    t0 = time.time()
    seqs = [r["sequence"].upper() for r in rows]
    if a.crop:
        c0, c1 = (int(v) for v in a.crop.split(":"))
        seqs = [q[c0:c1] for q in seqs]
    pb, gl = featurize(model, seqs, a.stride, a.left.upper(), a.right.upper(), a.keep_flanks,
                       a.batch, dev)
    el = time.time() - t0
    out = f"{a.out}.chunk{a.chunk:03d}of{a.n_chunks:03d}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    keep = [k for k in rows[0].keys() if k != "sequence"] if rows else []
    np.savez(out, per_bp=pb, glob=gl, sequences=np.asarray(seqs, dtype=object),
             meta=json.dumps({"run": a.run, "table": a.table, "where": filt, "stride": a.stride,
                              "crop": a.crop,
                              "per_bp_names": pbn, "global_names": gln,
                              "columns": {k: [r[k] for r in rows] for k in keep}}))
    print(f"{len(seqs)} sequences of {len(seqs[0]) if seqs else 0} bp -> {out} "
          f"({el:.0f}s, {len(seqs) / max(el, 1e-9):.0f} sequences/s on {dev})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
