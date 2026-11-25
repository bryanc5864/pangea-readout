#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Side shards of Family M (quadruplexes of every topology class, triplexes, i-motifs, cruciforms),
aligned to existing shards, from the production runners' .npz files.

For each base shard (main corpus or quadruplex windows), one side shard with the same fragments in
the same order (sequences copied, so the loaders verify alignment) and every Family M channel with
the main shards' metadata schema. Per fold type and fragment: the solve's channels if present;
else, if the fragment has no candidate for that fold, modelled = 0 and in_model = 0 (the rest
NaN); else the fragment is missing -- an error unless --allow-missing, since a silently absent
model would read as "no fold".
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics", "fold"))
import landscape as LS  # noqa: E402  (registers every Family M module)
from channels import Resolution  # noqa: E402
import g4_select as S  # noqa: E402
import structure_triplex as TX  # noqa: E402
import imotif_build as IB  # noqa: E402
import cruciform_build as CB  # noqa: E402
sys.path.insert(0, HERE)
from generate_targets import channel_metadata  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob of the base shards")
    ap.add_argument("--g4", required=True, help="run_family_m3 solves folder")
    ap.add_argument("--triplex", required=True); ap.add_argument("--imotif", required=True)
    ap.add_argument("--cruciform", required=True)
    ap.add_argument("--windows", required=True, help="frag_id -> long window TSV (cruciform candidates)")
    ap.add_argument("--window-offset", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pb = LS.channel_names(Resolution.PER_BP, ["M"]); gl = LS.channel_names(Resolution.GLOBAL, ["M"])
    meta = channel_metadata(["M"])
    wins = {r["frag_id"]: r["window"] for r in csv.DictReader(open(a.windows), delimiter="\t")}
    sources = (
        ("g4", a.g4, lambda fid, s: (S.select_v3(s) is not None and S.select_v3(s).reason == "ok"),
         [f"M_g4{X}_modelled" for X in "pha"], [f"M_g4{X}_in_model" for X in "pha"]),
        ("triplex", a.triplex, lambda fid, s: TX.select(s) is not None, ["M_tx_modelled"], ["M_tx_in_model"]),
        ("imotif", a.imotif, lambda fid, s: IB.select(s) is not None, ["M_im_modelled"], ["M_im_in_model"]),
        ("cruciform", a.cruciform, lambda fid, s: fid in wins and CB.select(wins[fid], a.window_offset, len(s)) is not None,
         ["M_cr_modelled"], ["M_cr_in_model"]))
    tot, missing = 0, {k[0]: 0 for k in sources}
    for p in sorted(glob.glob(a.shards)):
        with h5py.File(p, "r") as h:
            seqs = _decode(h["sequence"][:])
            ids = _decode(h["frag_id"][:]) if "frag_id" in h else [str(i) for i in range(len(seqs))]
        n, Lbp = len(seqs), len(seqs[0])
        P = np.full((n, len(pb), Lbp), np.nan, dtype=np.float32)
        G = np.full((n, len(gl)), np.nan, dtype=np.float32)
        for i, (fid, s) in enumerate(zip(ids, seqs)):
            for name, folder, has, mod_g, mod_p in sources:
                f = os.path.join(folder, f"{fid}.npz")
                if os.path.exists(f):
                    z = np.load(f)
                    if str(z["sequence"]) != s:
                        raise ValueError(f"{f}: sequence differs from {p}[{i}]")
                    for j, nm in enumerate(z["per_bp_names"]):
                        P[i, pb.index(str(nm))] = z["per_bp"][j]
                    for j, nm in enumerate(z["global_names"]):
                        G[i, gl.index(str(nm))] = z["global_"][j]
                    continue
                if has(fid, s):
                    missing[name] += 1
                    if not a.allow_missing:
                        raise FileNotFoundError(f"{fid} ({p}[{i}]) has a {name} candidate but no solve")
                    continue
                for nm in mod_g:
                    G[i, gl.index(nm)] = 0.0
                for nm in mod_p:
                    P[i, pb.index(nm)] = 0.0
        out = os.path.join(a.out, os.path.basename(p))
        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as h:
            h.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4, chunks=True)
            h.create_dataset("global", data=G, compression="gzip", compression_opts=4)
            dt = h5py.string_dtype()
            h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=dt)
            h.create_dataset("frag_id", data=np.array(ids, dtype=object), dtype=dt)
            for k, v in meta.items():
                if not isinstance(v, list):
                    continue
                if v and isinstance(v[0], (bool, np.bool_)):
                    h.attrs[k] = np.array(v, dtype=bool)
                elif v and isinstance(v[0], (int, np.integer)) and not isinstance(v[0], bool):
                    h.attrs[k] = np.array(v, dtype=np.int64)
                else:
                    h.attrs.create(k, [str(x) for x in v], dtype=h5py.string_dtype())
            h.attrs["seq_sha1"] = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
            h.attrs["families"] = "M"
            h.attrs["main_shard"] = os.path.abspath(p)
        os.replace(tmp, out)
        tot += n
    print(f"{tot} fragments packed -> {a.out}; missing (allowed: {a.allow_missing}): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
