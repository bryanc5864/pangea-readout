# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Do the Watson/Crick partner channels satisfy C(x) = W(rc x) in the targets themselves?

The strand tie for Family L rests on an identity the code guarantees (the minus-strand
transcript of x *is* the plus-strand transcript of rc(x)). A Watson/Crick tie for the
structural families would rest on the structure pipeline being reverse-complement
consistent: the minimised duplex of rc(x) must be the strand-swapped duplex of x. That is
an empirical property of cgNA+ plus a restrained two-start minimisation, so it is
measured here. For a few library sequences, the production target code (minimised
ground state, cgNA+ ensemble; no 16-snapshot relaxed ensemble) runs on x and on rc(x),
and every whole-sequence W/C pair is compared: |C(x) - W(rc x)| relative to the
channel's spread across the sequences, and the same for W(x) against C(rc x).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "physics"))
import geometry as G  # noqa: E402
import landscape as LS  # noqa: E402
from landscape import Resolution  # noqa: E402

COMP = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.upper().translate(COMP)[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="results/rc_consistency_v2.json")
    ap.add_argument("--n-ensemble", type=int, default=3200)
    a = ap.parse_args()
    with h5py.File(a.shard, "r") as h:
        seqs = [x.decode() if isinstance(x, bytes) else str(x) for x in h["sequence"][: a.n]]
    gl_names = LS.channel_names(Resolution.GLOBAL)
    pb_names = LS.channel_names(Resolution.PER_BP)
    vals = {"x": [], "rc": []}
    pvals = {"x": [], "rc": []}
    for k, s in enumerate(seqs):
        for tag, q in (("x", s), ("rc", rc(s))):
            wd = tempfile.mkdtemp(prefix="rccheck_")
            try:
                built = G.build_minimised_structure(q, wd)
                pbv, g = LS.landscape_arrays(q, pb_names, gl_names,
                                             minimised_pdb=built["minimised_pdb"],
                                             n_ensemble=a.n_ensemble, seed=k)
                vals[tag].append(np.asarray(g, dtype=np.float64))
                pvals[tag].append(np.asarray(pbv, dtype=np.float64))
            finally:
                shutil.rmtree(wd, ignore_errors=True)
        print(f"sequence {k + 1}/{len(seqs)} done", flush=True)
    X, R = np.stack(vals["x"]), np.stack(vals["rc"])
    pos = {n: i for i, n in enumerate(gl_names)}
    res = {"n_sequences": len(seqs), "pairs": {}}
    for n in gl_names:
        if not n.endswith("_W"):
            continue
        c = n[:-2] + "_C"
        if c not in pos:
            continue
        w, cc = pos[n], pos[c]
        spread = np.nanstd(np.concatenate([X[:, w], X[:, cc]]))
        d1 = np.abs(X[:, cc] - R[:, w]); d2 = np.abs(X[:, w] - R[:, cc])
        rel = float(np.nanmax(np.concatenate([d1, d2])) / spread) if spread > 0 else float("nan")
        res["pairs"][n[:-2]] = {"max_abs_diff": float(np.nanmax(np.concatenate([d1, d2]))),
                                "spread": float(spread), "max_rel_diff": rel}
        print(f"{n[:-2]:34s} max |C(x) - W(rc x)| = {np.nanmax(np.concatenate([d1, d2])):.4g} "
              f"({rel:.3f} of the spread across sequences)", flush=True)
    # Per-bp W/C pairs, position by position: C(x)[k] against W(rc x)[n-1-k] (base-indexed)
    # and [n-2-k] (step-indexed), split into the two terminal positions of each end and
    # the interior, to see whether a mismatch is local to the ends.
    PX, PR = np.stack(pvals["x"]), np.stack(pvals["rc"])          # (s, C, L)
    ppos = {n: i for i, n in enumerate(pb_names)}
    L = PX.shape[-1]
    res["per_bp"] = {}
    for n in pb_names:
        if not n.endswith("_W") and "_W_" not in n:
            continue
        c = n[:-2] + "_C" if n.endswith("_W") else n.replace("_W_", "_C_", 1)
        if c not in ppos:
            continue
        w, cc = ppos[n], ppos[c]
        spread = np.nanstd(PX[:, [w, cc]])
        best = None
        for off in (1, 2):
            idx = np.array([L - off - k for k in range(L)])
            ok = (idx >= 0) & (idx < L)
            d = np.full((PX.shape[0], L), np.nan)
            d[:, ok] = np.abs(PX[:, cc, ok] - PR[:, w, idx[ok]])
            ends = np.r_[0:2, L - 2:L]
            inter = np.setdiff1d(np.arange(L), ends)
            e = float(np.nanmax(d[:, ends])) if np.isfinite(d[:, ends]).any() else np.nan
            i_ = float(np.nanmedian(d[:, inter])) if np.isfinite(d[:, inter]).any() else np.nan
            if best is None or (np.isfinite(i_) and i_ < best[1]):
                best = (off, i_, e)
        if best is None or spread == 0 or not np.isfinite(spread):
            continue
        res["per_bp"][n[:-2] if n.endswith("_W") else n.replace("_W_", "_")] = {
            "index_map": "n-1-k" if best[0] == 1 else "n-2-k",
            "interior_median_rel": best[1] / spread, "ends_max_rel": best[2] / spread}
    print("\nper-bp W/C pairs: median interior |C(x)[k] - W(rc x)[k']| and worst end value, "
          "relative to the spread", flush=True)
    for n, d in sorted(res["per_bp"].items(), key=lambda kv: -kv[1]["interior_median_rel"]):
        print(f"   {n:36s} interior {d['interior_median_rel']:.3f}  ends {d['ends_max_rel']:.3f}  "
              f"({d['index_map']})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
