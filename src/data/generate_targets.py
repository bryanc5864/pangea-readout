#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Generate the stage-1 training targets: the whole analytic landscape, per fragment.

`PLAN.md` trains PANGEA in two stages, and this produces the labels for the first:
"Pretraining on analytically computable targets, available without limit at near-zero
cost for any sequence and any molecule class. These are cgNA+ geometry and stiffness,
nearest-neighbor and secondary-structure thermodynamics, stress-induced destabilization
profiles, non-canonical propensity calculations, and the quantum chemical quantities
looked up from precomputed fragment tables."

All six of those are implemented, so this writes every one of them for every fragment in
the library.

**Measured cost, at the library's 61 bp fragment length.**

    analytic stack (B, C, D, E, F, L)   1.52 s/fragment     81 CPU-h over the library
    restrained minimisation            4.85 s/fragment    258 CPU-h
    total                              6.37 s/fragment    339 CPU-h

which is about 14 minutes of wall clock at this cluster's 1,500-core cap on `day`. Label
generation is not and will not become the bottleneck, so nothing here is traded away for
speed: the ensemble runs at its full 200 draws and the minimisation runs on every
fragment rather than a subset.

**The minimisation is why this is a Slurm array rather than a loop.** Family C's backbone
torsions and Family E's hydrogen-bond distances are wrong without it by up to 63 degrees
and 0.2 angstrom respectively, as `DECISIONS.md` records, so it is run for every fragment
and the resulting structure is shared between the two families. It writes files, so each
task works in node-local scratch and deletes as it goes; a shard of 2,000 fragments would
otherwise leave 2,000 PDB directories behind.

**Column order comes from the channel registry, not from this file.** `landscape`
enumerates the declared channels in sorted order, so the column layout of a shard is
reproducible and independent of import order, and a channel that exists in the physics
but was never declared cannot quietly fail to appear -- `tests/test_landscape.py` asserts
the two sets are equal in both directions.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))

import landscape as LS  # noqa: E402
from channels import REGISTRY, Resolution  # noqa: E402


def read_library(path: str, limit: Optional[int] = None
                 ) -> Tuple[List[str], List[str], List[str]]:
    """Fragment ids, sequences and strata from the library TSV."""
    ids: List[str] = []
    seqs: List[str] = []
    strata: List[str] = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_id = header.index("frag_id")
        i_seq = header.index("sequence")
        i_str = header.index("stratum")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_id, i_seq, i_str):
                continue
            ids.append(parts[i_id])
            seqs.append(parts[i_seq].upper())
            strata.append(parts[i_str])
            if limit and len(ids) >= limit:
                break
    return ids, seqs, strata


def channel_metadata(families=None) -> Dict[str, object]:
    """Everything a consumer needs to interpret a column without importing the physics.

    Written into every shard. A training run that reads only the HDF5 still knows each
    channel's family, resolution, parity, units, whether it is provisional and where its
    parameters came from -- which is what lets the trainer refuse provisional channels
    and lets the loss apply the right parity, rather than those facts living only in
    Python that the training job may not import.
    """
    out: Dict[str, object] = {}
    for res, key in ((Resolution.PER_BP, "per_bp"), (Resolution.GLOBAL, "global")):
        # `families`: a side shard's families, which may include ones held out of the
        # default set (Family I); None keeps the main corpus's default.
        names = LS.channel_names(res, families) if families else LS.channel_names(res)
        out[f"{key}_names"] = names
        out[f"{key}_family"] = [REGISTRY[n].family for n in names]
        out[f"{key}_parity"] = [REGISTRY[n].parity.value for n in names]
        out[f"{key}_units"] = [REGISTRY[n].units for n in names]
        out[f"{key}_provisional"] = [bool(REGISTRY[n].provisional) for n in names]
        out[f"{key}_source"] = [REGISTRY[n].source for n in names]
        out[f"{key}_condition_dependent"] = [
            bool(REGISTRY[n].condition_dependent) for n in names]
        out[f"{key}_rc_shift"] = [int(REGISTRY[n].rc_shift) for n in names]
        out[f"{key}_circular"] = [bool(REGISTRY[n].circular) for n in names]
    return out


# A task whose first this-many fragments all fail aborts rather than producing a shard
# of NaN (see generate_shard).
FAIL_FAST_AFTER = 5


def in_ensemble_subset(frag_id: str, fraction: float) -> bool:
    """Deterministic membership of the ensemble subset, by a hash of the fragment id.

    Deterministic so a requeued task, or a regeneration, computes the expensive
    minimised ensemble for exactly the same fragments.
    """
    import hashlib
    if fraction <= 0:
        return False
    h = int(hashlib.sha1(frag_id.encode()).hexdigest()[:8], 16)
    return (h % 100000) < fraction * 100000


def generate_shard(seqs: Sequence[str], workroot: str, n_ensemble: int = 200,
                   minimise: bool = True, seed: int = 0,
                   progress_every: int = 25,
                   ids: Optional[Sequence[str]] = None,
                   ensemble_fraction: float = 0.0) -> Dict[str, object]:
    """Compute the landscape for a list of sequences and pack it into dense arrays.

    A fragment that fails is recorded and left as NaN rather than aborting the shard: one
    bad sequence in a 191,344-fragment library should cost that fragment, not the run.
    The failure list is written into the shard so the count can never be lost.
    """
    import geometry as G

    pb_names = LS.channel_names(Resolution.PER_BP)
    gl_names = LS.channel_names(Resolution.GLOBAL)
    n = len(seqs)
    L = len(seqs[0]) if n else 0

    P = np.full((n, len(pb_names), L), np.nan, dtype=np.float32)
    Gm = np.full((n, len(gl_names)), np.nan, dtype=np.float32)
    failures: List[Dict[str, str]] = []
    n_ens = 0
    # Minimisation audit, one row per fragment: which start was kept (0 plain, 1
    # canonical alpha/gamma, -1 not minimised) and the two final energies, so the
    # protocol's effect is measured on the whole corpus rather than asserted from two
    # test duplexes.
    min_start = np.full(n, -1, dtype=np.int8)
    min_energy = np.full((n, 2), np.nan, dtype=np.float32)
    in_ens = np.zeros(n, dtype=bool)
    t0 = time.time()

    for i, seq in enumerate(seqs):
        wd = os.path.join(workroot, f"frag{i:06d}")
        mini = None
        try:
            if minimise:
                built = G.build_minimised_structure(seq, wd)
                mini = built["minimised_pdb"]
                rep = built["report"]
                min_start[i] = 1 if rep.get("start_kept") == "canonical" else 0
                min_energy[i, 0] = rep.get("energy_final_plain_kJ_mol", np.nan)
                min_energy[i, 1] = rep.get("energy_final_canonical_kJ_mol", np.nan)
            me = None
            if minimise and ids is not None and in_ensemble_subset(ids[i],
                                                                   ensemble_fraction):
                me = G.minimised_ensemble_stats(seq, os.path.join(wd, "ens"),
                                                seed=seed + i)
                n_ens += 1
                in_ens[i] = True
            p, g = LS.landscape_arrays(seq, pb_names, gl_names,
                                       minimised_pdb=mini, n_ensemble=n_ensemble,
                                       seed=seed + i, min_ensemble=me)
            P[i] = p
            Gm[i] = g
        except Exception as exc:                                  # noqa: BLE001
            failures.append({"index": str(i), "sequence": seq[:20],
                             "error": f"{type(exc).__name__}: {exc}"[:200]})
            # **Fail fast and say why.** Failures used to be recorded silently in the
            # shard, so a 1024-task array whose every fragment failed -- a data file
            # missing from the frozen code snapshot -- ran for an hour, printing only a
            # failure count, before anyone looked. The first errors now go to stderr as
            # they happen, and a task whose first fragments all fail stops at once.
            if len(failures) <= 3:
                print(f"fragment {i} failed: {failures[-1]['error']}", file=sys.stderr,
                      flush=True)
            if i + 1 == FAIL_FAST_AFTER and len(failures) == FAIL_FAST_AFTER:
                raise RuntimeError(
                    f"the first {FAIL_FAST_AFTER} fragments all failed; first error: "
                    f"{failures[0]['error']}")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el/(i+1):.2f} s/fragment  "
                  f"eta {(n-i-1)*el/(i+1)/60:.1f} min  failures {len(failures)}",
                  flush=True)

    return {"per_bp": P, "global": Gm, "failures": failures,
            "seconds": time.time() - t0, "n_ensemble_fragments": n_ens,
            "min_start_kept": min_start, "min_energy_kJ_mol": min_energy,
            "in_ensemble_subset": in_ens,
            "per_bp_names": pb_names, "global_names": gl_names}


def write_shard(path: str, ids: Sequence[str], seqs: Sequence[str],
                strata: Sequence[str], result: Dict[str, object]) -> None:
    import h5py

    meta = channel_metadata()
    with h5py.File(path, "w") as h:
        h.create_dataset("per_bp", data=result["per_bp"], compression="gzip",
                         compression_opts=4, chunks=True)
        h.create_dataset("global", data=result["global"], compression="gzip",
                         compression_opts=4)
        dt = h5py.string_dtype()
        h.create_dataset("frag_id", data=np.array(list(ids), dtype=object), dtype=dt)
        h.create_dataset("sequence", data=np.array(list(seqs), dtype=object), dtype=dt)
        h.create_dataset("stratum", data=np.array(list(strata), dtype=object), dtype=dt)
        for key in ("min_start_kept", "min_energy_kJ_mol", "in_ensemble_subset"):
            if key in result:
                h.create_dataset(key, data=result[key])
        for k, v in meta.items():
            if isinstance(v, list) and v and isinstance(v[0], bool):
                h.attrs[k] = np.array(v, dtype=bool)
            elif isinstance(v, list) and v and isinstance(v[0], int):
                h.attrs[k] = np.array(v, dtype=np.int64)
            elif isinstance(v, list):
                h.attrs[k] = np.array([str(x) for x in v], dtype=object)
        h.attrs["n_fragments"] = len(ids)
        h.attrs["seconds"] = float(result["seconds"])
        h.attrs["n_failures"] = len(result["failures"])
        h.attrs["n_ensemble_fragments"] = int(result.get("n_ensemble_fragments", 0))
        # Provenance: the settings and the code that produced this shard.
        for k, v in (result.get("settings") or {}).items():
            h.attrs[k] = v
        h.attrs["code_root"] = os.path.dirname(os.path.dirname(HERE))
        h.attrs["failures_json"] = json.dumps(result["failures"])[:60000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="library TSV")
    ap.add_argument("--out", required=True, help="output HDF5 shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="read only the first N fragments of the library (for a pilot)")
    ap.add_argument("--n-ensemble", type=int, default=200)
    ap.add_argument("--no-minimise", action="store_true",
                    help="skip the restrained minimisation; the torsion and "
                         "hydrogen-bond-distance channels then come back NaN")
    ap.add_argument("--workroot", default=os.environ.get("SLURM_TMPDIR", "/tmp"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble-fraction", type=float, default=0.10,
                    help="fraction of fragments, chosen deterministically by id, that get "
                         "the minimised cgNA+ ensemble (torsion spreads, BI population, "
                         "pucker states); PLAN.md's ensemble convention is about 10 percent")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a rate line every N fragments; 0 to silence")
    args = ap.parse_args()

    ids, seqs, strata = read_library(args.library, args.limit)
    sel = range(args.shard, len(ids), args.n_shards)
    ids = [ids[i] for i in sel]
    seqs = [seqs[i] for i in sel]
    strata = [strata[i] for i in sel]
    print(f"shard {args.shard}/{args.n_shards}: {len(ids)} fragments, "
          f"length {len(seqs[0]) if seqs else 0}", flush=True)

    pb = LS.channel_names(Resolution.PER_BP)
    gl = LS.channel_names(Resolution.GLOBAL)
    print(f"channels: {len(pb)} per base pair, {len(gl)} whole sequence", flush=True)
    # A dataset with no channels in it is a silent catastrophe, not an edge case: it
    # once happened, because the family modules register their channels on import and
    # the registry was consulted before they had been imported. Refuse rather than
    # write a well-formed file full of nothing.
    if not pb or not gl:
        raise RuntimeError(
            f"the channel registry is empty or partial ({len(pb)} per-bp, {len(gl)} "
            "global). The family modules register on import; check that `landscape` "
            "imports all of them eagerly.")

    workroot = os.path.join(args.workroot, f"pangea_targets_{args.shard}")
    os.makedirs(workroot, exist_ok=True)
    try:
        res = generate_shard(seqs, workroot, n_ensemble=args.n_ensemble,
                             minimise=not args.no_minimise, seed=args.seed,
                             progress_every=args.progress_every, ids=ids,
                             ensemble_fraction=args.ensemble_fraction)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res["settings"] = {"n_ensemble": int(args.n_ensemble),
                       "ensemble_fraction": float(args.ensemble_fraction),
                       "seed": int(args.seed), "minimised": int(not args.no_minimise)}
    write_shard(args.out, ids, seqs, strata, res)
    print(f"wrote {args.out}: {len(ids)} fragments, {len(res['failures'])} failures, "
          f"{res['seconds']/max(len(ids),1):.2f} s/fragment", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
