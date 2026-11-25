#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family W production: `longrange.long_channels` on rows of the long-fragment table, one .npz per
fragment (tracks float32 (n_tracks, L), scalars float64, names, rc_shift). Resumable."""
import argparse, csv, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
csv.field_size_limit(1 << 30)


def main() -> int:
    import longrange as W
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.rows.split("-"))
    os.makedirs(a.out, exist_ok=True)
    with open(a.table) as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            if i < lo:
                continue
            if i > hi:
                break
            path = os.path.join(a.out, f"{r['frag_id']}.npz")
            if os.path.exists(path):
                continue
            t = time.time()
            try:
                tracks, sc, sh = W.long_channels(r["sequence"])
                tn = sorted(tracks); sn = sorted(sc)
                tmp = path[:-4] + ".partial.npz"
                np.savez_compressed(tmp, tracks=np.stack([tracks[k] for k in tn]).astype(np.float32),
                                    track_names=np.array(tn), rc_shift=np.array([sh[k] for k in tn], np.int8),
                                    scalars=np.array([sc[k] for k in sn], np.float64), scalar_names=np.array(sn),
                                    frag_id=r["frag_id"])
                os.replace(tmp, path)
                print(f"{r['frag_id']} ok {time.time() - t:.0f}s", flush=True)
            except Exception as exc:                                 # noqa: BLE001
                print(f"{r['frag_id']} failed {time.time() - t:.0f}s :: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
