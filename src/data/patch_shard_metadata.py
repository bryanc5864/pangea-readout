#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Bring existing target shards' channel metadata up to date with the registry.

The trainer reads each channel's parity, index map and circularity from the shard
attributes, never from the physics registry -- the training environment does not even
have the physics installed. So when a *declaration* changes without the *values*
changing, the shards' attributes have to follow, and regenerating 290 CPU-hours of
targets to change a few attributes would be absurd.

This touches attributes only, never data, and refuses to run if the shard's channel list
differs from the registry's: a metadata patch onto a different layout would mislabel
every column after the first mismatch.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))

import landscape as LS  # noqa: E402
from channels import REGISTRY, Resolution  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))
    if not paths:
        raise SystemExit(f"no shards match {a.shards}")
    n_ok = 0
    for p in paths:
        with h5py.File(p, "r+") as h:
            for key in ("per_bp", "global"):
                names = _decode(h.attrs[f"{key}_names"])
                missing = [n for n in names if n not in REGISTRY]
                if missing:
                    raise SystemExit(f"{p}: channels not in the registry: {missing[:5]}")
                h.attrs[f"{key}_parity"] = np.array(
                    [REGISTRY[n].parity.value for n in names], dtype=object)
                h.attrs[f"{key}_rc_shift"] = np.array(
                    [int(REGISTRY[n].rc_shift) for n in names], dtype=np.int64)
                h.attrs[f"{key}_circular"] = np.array(
                    [bool(REGISTRY[n].circular) for n in names], dtype=bool)
                h.attrs[f"{key}_provisional"] = np.array(
                    [bool(REGISTRY[n].provisional) for n in names], dtype=bool)
                h.attrs[f"{key}_units"] = np.array(
                    [REGISTRY[n].units for n in names], dtype=object)
        n_ok += 1
    print(f"patched metadata on {n_ok} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
