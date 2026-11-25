#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Pack Family W solves into HDF5 shards (SHARD fragments each): sequence (10,240 bp), metadata
(source genome, clade, contig, start, GC, the library fragments each one contains for the trainer's
leakage grouping), tracks (n, n_tracks, 10240) float32 and scalars (n, n_scalars) float32, with the
channel names and reverse-complement index shifts as attributes. Every fragment of the table must
have a solve: a missing one is reported and stops the packing (no partial shards)."""
import argparse, csv, json, os, sys
import numpy as np
csv.field_size_limit(1 << 30)


def main() -> int:
    import h5py
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True); ap.add_argument("--solves", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--allow-missing", action="store_true", help="report only (dry run); writes nothing")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.table), delimiter="\t"))
    missing = [r["frag_id"] for r in rows if not os.path.exists(os.path.join(a.solves, f"{r['frag_id']}.npz"))]
    print(f"{len(rows)} fragments, {len(missing)} without a solve", flush=True)
    if missing:
        json.dump(missing, open(os.path.join(a.out if os.path.isdir(a.out) else os.path.dirname(a.out), "famW_missing.json"), "w"))
        return 0 if a.allow_missing else 2
    os.makedirs(a.out, exist_ok=True)
    z0 = np.load(os.path.join(a.solves, f"{rows[0]['frag_id']}.npz"))
    tn, sn, sh = [str(x) for x in z0["track_names"]], [str(x) for x in z0["scalar_names"]], z0["rc_shift"]
    L = z0["tracks"].shape[1]
    str_dt = h5py.string_dtype()
    for k0 in range(0, len(rows), a.shard):
        part = rows[k0:k0 + a.shard]; n = len(part)
        path = os.path.join(a.out, f"famW_{k0 // a.shard:04d}.h5"); tmp = path + ".partial"
        with h5py.File(tmp, "w") as h:
            tr = h.create_dataset("tracks", (n, len(tn), L), dtype="f4", chunks=(1, len(tn), L), compression="gzip", compression_opts=4)
            sc = h.create_dataset("scalars", (n, len(sn)), dtype="f4")
            for i, r in enumerate(part):
                z = np.load(os.path.join(a.solves, f"{r['frag_id']}.npz"))
                if [str(x) for x in z["track_names"]] != tn or [str(x) for x in z["scalar_names"]] != sn:
                    raise ValueError(f"{r['frag_id']}: channel layout differs")
                tr[i] = z["tracks"]; sc[i] = z["scalars"].astype(np.float32)
            h.create_dataset("sequence", data=np.array([r["sequence"].encode() for r in part], dtype=f"S{L}"))
            for k in ("frag_id", "source", "clade", "contig", "contains"):
                h.create_dataset(k, data=[r[k] for r in part], dtype=str_dt)
            h.create_dataset("start", data=np.array([int(r["start"]) for r in part], np.int64))
            h.create_dataset("gc", data=np.array([float(r["gc"]) for r in part], np.float32))
            h.attrs.create("track_names", tn, dtype=str_dt); h.attrs.create("scalar_names", sn, dtype=str_dt)
            h.attrs["rc_shift"] = np.asarray(sh, np.int8); h.attrs["family"] = "W"; h.attrs["length_bp"] = L
        os.replace(tmp, path)
        print(path, n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
