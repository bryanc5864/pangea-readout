#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A side shards: the solver's per-fragment outputs, aligned to the main corpus.

Family A is not computed from the sequence (it comes from APBS solves, one `.npz` per
fragment, `run_fragment.py --extended --reference-solve --production-conditions
--exact-fine-spacing`). This writes, for each main shard, a side shard in the same format
as `generate_family_shards.py` -- same fragments in the same order, Family A channel
metadata, sequence checksum -- so `InMemoryLandscape(extra_dirs=...)` merges it like any
other. A fragment with no solve (anchoring solves a subset at most conditions) is NaN; a
solve whose sequence differs from the main shard's is refused, not merged.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
sys.path.insert(0, HERE)
import electrostatics_family as A  # noqa: E402  (registers the Family A channels)
import landscape as LS  # noqa: E402
from channels import Resolution  # noqa: E402
from generate_targets import channel_metadata  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def find_solves(dirs):
    """frag_id -> npz path; a later directory overrides an earlier one."""
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.npz")):
            out[os.path.basename(p)[:-4]] = p
    return out


def one_shard(path: str, out_dir: str, solves: dict) -> str:
    import h5py
    with h5py.File(path, "r") as h:
        seqs = _decode(h["sequence"][:])
        ids = _decode(h["frag_id"][:])
    sha = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
    pb = LS.channel_names(Resolution.PER_BP, ["A"])
    gl = LS.channel_names(Resolution.GLOBAL, ["A"])
    L = len(seqs[0])
    P = np.full((len(seqs), len(pb), L), np.nan, dtype=np.float32)
    G = np.full((len(seqs), len(gl)), np.nan, dtype=np.float32)
    fails, n_have = [], 0
    for i, (fid, s) in enumerate(zip(ids, seqs)):
        p = solves.get(fid)
        if p is None:
            continue
        try:
            z = np.load(p, allow_pickle=False)
            z = {k: z[k] for k in z.files}
            if str(z["sequence"][0]) != s:
                raise ValueError("solve sequence differs from the main shard's")
            prof, glob_ = A.assemble(z, n_bp=L)
            P[i] = np.stack([prof[n] for n in pb]); G[i] = [glob_[n] for n in gl]
            n_have += 1
        except Exception as exc:                                       # noqa: BLE001
            fails.append({"index": i, "frag_id": fid, "error": f"{type(exc).__name__}: {exc}"[:200]})
    meta = channel_metadata(["A"])
    sel = {}                                   # restricted to (and ordered as) pb / gl
    for key, names in (("per_bp", pb), ("global", gl)):
        allnames = list(meta[f"{key}_names"])
        idx = [allnames.index(n) for n in names]
        for k, v in meta.items():
            if k.startswith(key + "_") and isinstance(v, list) and len(v) == len(allnames):
                sel[k] = [v[j] for j in idx]
    out = os.path.join(out_dir, os.path.basename(path))
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        st = h5py.string_dtype()
        h.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4)
        h.create_dataset("global", data=G, compression="gzip", compression_opts=4)
        h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=st)
        h.create_dataset("frag_id", data=np.array(ids, dtype=object), dtype=st)
        for k, v in sel.items():                 # typed exactly as generate_family_shards
            if v and isinstance(v[0], (bool, np.bool_)):
                h.attrs[k] = np.array(v, dtype=bool)
            elif v and isinstance(v[0], (int, np.integer)) and not isinstance(v[0], bool):
                h.attrs[k] = np.array(v, dtype=np.int64)
            else:
                h.attrs[k] = np.array([str(x) for x in v], dtype=object)
        h.attrs["seq_sha1"] = sha; h.attrs["families"] = "A"; h.attrs["main_shard"] = path
        h.attrs["n_solved"] = n_have; h.attrs["n_failures"] = len(fails)
        h.attrs["failures_json"] = json.dumps(fails[:50])
    os.replace(tmp, out)
    return f"wrote {out}: {n_have}/{len(seqs)} fragments with solves, {len(fails)} refused"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--solves", nargs="+", required=True, help="directories of per-fragment .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    solves = find_solves(a.solves)
    t = time.time()
    for p in sorted(glob.glob(a.shards))[a.task::a.n_tasks]:
        print(one_shard(p, a.out, solves), flush=True)
    print(f"{len(solves)} solves indexed; {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
