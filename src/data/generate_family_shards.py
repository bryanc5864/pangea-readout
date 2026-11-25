#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Side shards for families added after the main corpus: same fragments, new channels.

The v2 corpus (1,024 shards, 191,344 fragments) carries families B-H and L. A family added
later -- K (RNA duplex and DNA-RNA hybrid), J, A -- is generated as one side shard per main
shard, holding only the new family's channels for exactly the same fragments in the same
order, with the main shard's sequences copied so the loader can verify alignment
(`InMemoryLandscape(extra_dirs=...)` refuses a side shard whose sequences differ). The
side shard uses the main shards' metadata schema (names, parity, family, units,
provisional, rc_shift, circular, condition dependence, source), so the trainer treats its
channels exactly like the originals. Nothing in the main shards is touched.

Works on the auxiliary windows the same way (`--shards` pointed at their shard files).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import landscape as LS  # noqa: E402
from channels import REGISTRY, Resolution  # noqa: E402

sys.path.insert(0, HERE)
from generate_targets import channel_metadata  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def one_shard(path: str, out_dir: str, families, per_bp_families=None) -> str:
    out = os.path.join(out_dir, os.path.basename(path))
    with h5py.File(path, "r") as h:
        seqs = _decode(h["sequence"][:])
        ids = _decode(h["frag_id"][:]) if "frag_id" in h else [str(i) for i in range(len(seqs))]
        # Main-shard values a side family reads as context (Family I: the nucleosome free
        # energy its competition terms need), so nothing is recomputed.
        gnames = _decode(h.attrs["global_names"])
        ctx_cols = {k: gnames.index(k) for k in ("D_nucleosome_dG_I0.15_kT",) if k in gnames}
        G_main = np.asarray(h["global"][:], dtype=float) if ctx_cols else None
    sha = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
    if os.path.exists(out):
        try:
            with h5py.File(out, "r") as h:
                if h.attrs.get("seq_sha1") == sha and h.attrs.get("families") == ",".join(families):
                    return f"exists {out}"
        except Exception:                                            # noqa: BLE001
            pass
    # Per-bp channels can be restricted to some families (the auxiliary windows carry
    # Family K's whole-sequence channels but not its 312 per-bp profiles, which would be
    # ~40 GB in memory at one million windows).
    pb = LS.channel_names(Resolution.PER_BP,
                          families if per_bp_families is None
                          else [f for f in families if f in per_bp_families])
    gl = LS.channel_names(Resolution.GLOBAL, families)
    L = len(seqs[0])
    P = np.full((len(seqs), len(pb), L), np.nan, dtype=np.float32)
    G = np.full((len(seqs), len(gl)), np.nan, dtype=np.float32)
    fails, t0 = [], time.time()
    for i, s in enumerate(seqs):
        try:
            ctx = {k: float(G_main[i, j]) for k, j in ctx_cols.items()} if ctx_cols else None
            p, g = LS.landscape_arrays(s, pb, gl, families=families, n_ensemble=0, context=ctx)
            P[i], G[i] = p, g
        except Exception as exc:                                     # noqa: BLE001
            fails.append({"index": i, "error": f"{type(exc).__name__}: {exc}"[:200]})
            if len(fails) <= 2:
                print(f"{os.path.basename(path)}[{i}] failed: {fails[-1]['error']}", file=sys.stderr, flush=True)
    meta = channel_metadata(families)              # every channel of these families ...
    sel = {}                                       # ... restricted to this side shard's
    for key, names in (("per_bp", pb), ("global", gl)):
        allnames = list(meta[f"{key}_names"])
        idx = [allnames.index(n) for n in names]
        for k, v in meta.items():
            if k.startswith(key + "_") and isinstance(v, list) and len(v) == len(allnames):
                sel[k] = [v[j] for j in idx]
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4, chunks=True)
        h.create_dataset("global", data=G, compression="gzip", compression_opts=4)
        dt = h5py.string_dtype()
        h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=dt)
        h.create_dataset("frag_id", data=np.array(ids, dtype=object), dtype=dt)
        for k, v in sel.items():
            if v and isinstance(v[0], (bool, np.bool_)):
                h.attrs[k] = np.array(v, dtype=bool)
            elif v and isinstance(v[0], (int, np.integer)) and not isinstance(v[0], bool):
                h.attrs[k] = np.array(v, dtype=np.int64)
            else:
                h.attrs[k] = np.array([str(x) for x in v], dtype=object)
        h.attrs["seq_sha1"] = sha
        h.attrs["families"] = ",".join(families)
        if per_bp_families is not None:
            h.attrs["per_bp_families_stored"] = ",".join(f for f in families if f in per_bp_families)
        h.attrs["main_shard"] = os.path.abspath(path)
        h.attrs["n_failures"] = len(fails)
        h.attrs["failures_json"] = json.dumps(fails)[:60000]
        h.attrs["code_root"] = os.path.dirname(os.path.dirname(HERE))
    os.replace(tmp, out)
    return (f"wrote {out}: {len(seqs)} fragments, {len(pb)} per-bp + {len(gl)} global "
            f"channels, {len(fails)} failures, {(time.time() - t0) / max(1, len(seqs)):.2f} s/fragment")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob of the main (or auxiliary) shards")
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bp-families", nargs="*", default=None,
                    help="store per-bp channels only for these families (default: all)")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))[a.task::a.n_tasks]
    if not paths:
        raise SystemExit(f"no shards match {a.shards}")
    os.makedirs(a.out, exist_ok=True)
    for p in paths:
        print(one_shard(p, a.out, a.families, a.per_bp_families), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
