# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Base-pair probability matrices for every fragment, from Family L's own folding model.

Family L's weakest channels -- positional entropy, ensemble diversity, unpaired
probability -- are all functionals of one object, the base-pair probability matrix that
ViennaRNA computes for each transcript. Supervising a model with the matrix itself gives
1,830 pair targets per transcript instead of 61 per-base summaries. The model here is
exactly the one `rna_structure.fold_once` uses for the targets (default `RNA.md()`: Turner
2004 at 37 C, the library's default salt and dangles), on both transcripts: "p", the RNA
of the given strand in its own 5'->3' frame, and "m", the RNA of the reverse complement
in *its* own frame -- the frame of the model's reverse-complement pass.

Storage: for each input shard, `<out>/<shard basename>.bpp.npz` with
  bpp      uint8 (n_fragments, 2, n*(n-1)/2), the upper triangle (i < j, row-major) of
           each matrix, quantised as q = round(255 * sqrt(p)) so that small probabilities
           (which carry the entropy) keep resolution: p = (q / 255)^2, smallest nonzero
           1.5e-5, step 0.008 near p = 1;
  seq_sha1 sha1 of the shard's sequences joined by newlines, checked at load time so a
           matrix can never be paired with the wrong fragment.
Shards are independent, so the work is split across processes by shard.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
from rna_structure import _model, to_rna  # noqa: E402

import RNA  # noqa: E402

COMP = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(s: str) -> str:
    return s.upper().translate(COMP)[::-1]


def bpp_matrix(rna: str) -> np.ndarray:
    """(n, n) symmetric base-pair probabilities, exactly as fold_once computes them."""
    fc = RNA.fold_compound(rna, _model())
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    m = np.asarray(fc.bpp(), dtype=np.float64)[1:, 1:]       # ViennaRNA is 1-based
    m = np.triu(m, 1)
    return m + m.T


def quantise(p: np.ndarray) -> np.ndarray:
    return np.rint(255.0 * np.sqrt(np.clip(p, 0.0, 1.0))).astype(np.uint8)


def dequantise(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 255.0) ** 2


def seq_sha1(seqs) -> str:
    return hashlib.sha1("\n".join(seqs).encode()).hexdigest()


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def one_shard(args) -> str:
    path, out_dir = args
    out = os.path.join(out_dir, os.path.basename(path).replace(".h5", ".bpp.npz"))
    with h5py.File(path, "r") as h:
        key = "sequence" if "sequence" in h else "sequences"
        seqs = _decode(h[key][:])
    if os.path.exists(out):
        try:
            if str(np.load(out)["seq_sha1"]) == seq_sha1(seqs):
                return f"exists {out}"
        except Exception:                                            # noqa: BLE001
            pass
    n = len(seqs[0])
    iu = np.triu_indices(n, 1)
    q = np.zeros((len(seqs), 2, len(iu[0])), dtype=np.uint8)
    for k, s in enumerate(seqs):
        if len(s) != n:
            raise ValueError(f"{path}: fragment {k} has length {len(s)}, expected {n}")
        q[k, 0] = quantise(bpp_matrix(to_rna(s.upper()))[iu])
        q[k, 1] = quantise(bpp_matrix(to_rna(reverse_complement(s)))[iu])
    tmp = out + ".tmp.npz"
    np.savez(tmp, bpp=q, seq_sha1=np.asarray(seq_sha1(seqs)), n=np.asarray(n))
    os.replace(tmp, out)
    return f"wrote {out}: {len(seqs)} fragments"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob of target shards (quoted)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--task", type=int, default=0, help="this task's index (array jobs)")
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.shards))[a.task::a.n_tasks]
    if not paths:
        raise SystemExit(f"no shards match {a.shards}")
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(one_shard, [(p, a.out) for p in paths]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
