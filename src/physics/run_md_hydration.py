#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Explicit-solvent MD hydration library (`md_hydration.py`): task = contexts[shard::n_shards]
of the RC-unique tetramers for one class; one .npz per (context, class); resumable."""
import argparse, json, os, sys, tempfile, time, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "qc"))
import md_hydration as MH       # noqa: E402
import build_qc_fragments as QF  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", choices=("dna", "rna", "hyb"), required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--contexts", nargs="*", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default=None); ap.add_argument("--production-ns", type=float, default=MH.PRODUCTION_NS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctxs = (a.contexts or QF.unique_contexts(4))[a.shard::a.n_shards]
    for ctx in ctxs:
        out = os.path.join(a.out, f"{ctx}_{a.cls}.npz")
        if os.path.exists(out):
            continue
        t = time.time(); work = tempfile.mkdtemp()
        try:
            raw = MH.build(ctx, a.cls, work)
            top, solute, water_o, frames, boxes = MH.simulate(raw, a.cls, work, a.platform, a.production_ns)
            t_md = time.time() - t
            res = MH.analyse(top, solute, water_o, frames, boxes, a.cls)
            arrays = {k: np.asarray(v) for k, v in res.items()}
            np.savez(out + ".partial.npz", context=ctx, cls=a.cls, seconds_md=t_md, seconds=time.time() - t, **arrays)
            os.replace(out + ".partial.npz", out)
            print(f"{ctx} {a.cls} ok md {t_md:.0f}s total {time.time() - t:.0f}s minor occ {np.round(res['minor_occupancy'][4:8], 2)}", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{ctx} {a.cls} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
