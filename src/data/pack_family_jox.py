#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Side shards of the oxDNA2 Family J channels (`single_strand_oxdna.py`), aligned to existing
shards, from `run_family_j_oxdna.py` v2 outputs. A fragment without a result is NaN and counted;
an error unless --allow-missing (every corpus fragment is simulated)."""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import single_strand_oxdna as J  # noqa: E402
sys.path.insert(0, HERE)
from generate_targets import channel_metadata  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True); ap.add_argument("--solves", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pb, gl = J.channel_names()
    meta = channel_metadata(["J"])
    sel = {}
    for key, names in (("per_bp", pb), ("global", gl)):
        alln = list(meta[f"{key}_names"]); idx = [alln.index(n) for n in names]
        for k, v in meta.items():
            if k.startswith(key + "_") and isinstance(v, list) and len(v) == len(alln):
                sel[k] = [v[j] for j in idx]
    tot = missing = 0
    for p in sorted(glob.glob(a.shards)):
        with h5py.File(p, "r") as h:
            seqs = _decode(h["sequence"][:]); ids = _decode(h["frag_id"][:])
        n, L = len(seqs), len(seqs[0])
        P = np.full((n, len(pb), L), np.nan, dtype=np.float32); G = np.full((n, len(gl)), np.nan, dtype=np.float32)
        for i, (fid, s) in enumerate(zip(ids, seqs)):
            f = os.path.join(a.solves, f"{fid}.npz")
            if not os.path.exists(f):
                missing += 1
                if not a.allow_missing:
                    raise FileNotFoundError(f"{fid} ({p}[{i}]) has no oxDNA result")
                continue
            z = np.load(f)
            if str(z["sequence"][0]) != s:
                raise ValueError(f"{f}: sequence differs from {p}[{i}]")
            prof, glob_ = J.channels_from_npz(z, s)
            for c, nm in enumerate(pb):
                P[i, c] = prof[nm]
            for c, nm in enumerate(gl):
                G[i, c] = glob_[nm]
        out = os.path.join(a.out, os.path.basename(p))
        with h5py.File(out + ".tmp", "w") as h:
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
                    h.attrs.create(k, [str(x) for x in v], dtype=h5py.string_dtype())
            h.attrs["seq_sha1"] = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
            h.attrs["families"] = "J"; h.attrs["main_shard"] = os.path.abspath(p)
        os.replace(out + ".tmp", out)
        tot += n
    print(f"{tot} fragments -> {a.out}; missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
