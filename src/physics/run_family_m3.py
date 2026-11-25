#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M v3 production (`structure_folds`): quadruplex models of every topology class with MD
ensembles, in two stages so the MD can run on a GPU and the Poisson-Boltzmann solves on CPUs.

  --stage build : select, plan, assemble, minimise and run MD for rows of a window TSV; one
                  gzipped JSON per fragment with every structure inlined (<out>/built/<id>.json.gz)
  --stage solve : APBS on every structure of a built fragment; channels (<out>/solves/<id>.npz,
                  the layout `pack_family_m.py` reads)
  --stage both  : the two in one process (CPU MD)

Resumable per fragment at each stage. Fragments without a candidate are written by the solve
stage too (modelled = 0 for every class), so the packer can tell computed from missing.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structure_folds as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stage", choices=("build", "solve", "both"), required=True)
    ap.add_argument("--workroot", default=None); ap.add_argument("--platform", default=None)
    ap.add_argument("--only-modelled", action="store_true")
    ap.add_argument("--production-ps", type=float, default=None)
    ap.add_argument("--equil-ps", type=float, default=None); ap.add_argument("--heat-ps", type=float, default=None)
    ap.add_argument("--batch", type=int, default=1, help="build stage: models per batched replica MD (GPU)")
    ap.add_argument("--block", type=int, default=8, help="build stage: fragments built before one batched MD")
    a = ap.parse_args()
    bdir, sdir = os.path.join(a.out, "built"), os.path.join(a.out, "solves")
    os.makedirs(bdir, exist_ok=True); os.makedirs(sdir, exist_ok=True)
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.windows), delimiter="\t"))[int(lo): int(hi) + 1]
    pb, gl = M.channel_names_per_bp(), M.channel_names_global()
    mds = {k: v for k, v in (("production_ps", a.production_ps), ("equil_ps", a.equil_ps),
                              ("heat_ps", a.heat_ps)) if v is not None} or None
    if a.stage in ("build", "both") and a.platform not in ("CPU", "Reference"):
        # a node whose GPU platform cannot create a context (seen: OpenCL clCreateContext -9999 on the RTX PRO
        # 6000 Blackwell nodes) must fail the task before any fragment, not record every model as failed
        import openmm
        try:
            s_ = openmm.System(); s_.addParticle(1.0)
            openmm.Context(s_, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName(a.platform))
        except Exception as exc:                                     # noqa: BLE001
            print(f"platform {a.platform} unusable on this node: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            return 2
    if a.stage == "build" and a.batch > 1:
        return build_batched(rows, bdir, sdir, a, mds)
    for r in rows:
        fid, seq = r["frag_id"], r["sequence"].upper()
        if a.only_modelled and str(r.get("modelled", "1")) == "0":
            continue
        bpath, spath = os.path.join(bdir, f"{fid}.json.gz"), os.path.join(sdir, f"{fid}.npz")
        work = tempfile.mkdtemp(prefix="famM3_", dir=a.workroot)
        t = time.time()
        try:
            if a.stage in ("build", "both") and not os.path.exists(bpath) and not os.path.exists(spath):
                built = M.build_models(seq, work, md=True, md_settings=mds, platform=a.platform)
                M.save_built(built, bpath)
                nm = sum(m.get("status") == "ok" for ms in built.get("classes", {}).values() for m in ms)
                print(f"{fid} built: {built.get('reason')} n_tet={built.get('n_tet')} models_ok={nm} "
                      f"classes={ {k: len(v) for k, v in built.get('classes', {}).items()} } {time.time() - t:.0f}s", flush=True)
            if a.stage in ("solve", "both") and os.path.exists(bpath) and not os.path.exists(spath):
                built = M.load_built(bpath, work)
                prof, glob = M.solve_models(built, work)
                P = np.stack([np.asarray(prof[k], dtype=np.float32) for k in pb])
                G = np.array([glob[k] for k in gl], dtype=np.float32)
                tmp = spath + ".partial.npz"
                np.savez(tmp, per_bp=P, global_=G, per_bp_names=np.array(pb), global_names=np.array(gl),
                         sequence=seq, seconds=time.time() - t)
                os.replace(tmp, spath)
                mod = {X: int(glob[f"M_g4{X}_modelled"]) for X, _ in M.CLASSES}
                print(f"{fid} solved: modelled {mod} {time.time() - t:.0f}s", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"{fid} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


def build_batched(rows, bdir, sdir, a, mds):
    """Build stage with the MD of many models in one replica system (`g4_build.md_ensemble_batch`):
    fragments are built and minimised a block at a time, then all their models' MD runs together."""
    import g4_build as B
    settings = dict(M.MD_SETTINGS, **(mds or {}))
    todo = [r for r in rows if not os.path.exists(os.path.join(bdir, f"{r['frag_id']}.json.gz"))
            and not os.path.exists(os.path.join(sdir, f"{r['frag_id']}.npz"))
            and not (a.only_modelled and str(r.get("modelled", "1")) == "0")]
    for b0 in range(0, len(todo), a.block):
        block = todo[b0:b0 + a.block]
        t = time.time()
        works, builts = [], []
        for r in block:
            w = tempfile.mkdtemp(prefix="famM3_", dir=a.workroot); works.append(w)
            try:
                builts.append(M.build_models(r["sequence"].upper(), w, md=False, platform=a.platform, minimise=False))
            except Exception as exc:                                 # noqa: BLE001
                print(f"{r['frag_id']} build failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                builts.append(None)
        # batched minimisation of every raw model of the block
        raws = []
        for bi, bt in enumerate(builts):
            if bt is None:
                continue
            n_res = len(bt.get("order", []))
            for X, models in bt.get("classes", {}).items():
                for k, m in enumerate(models):
                    if m.get("status") == "raw":
                        raws.append((bi, X, k, n_res, {"pdb_raw": m["raw"], "pdb_out": os.path.join(os.path.dirname(m["raw"]), "min.pdb"),
                                                       "core_residues": bt["core_idx"]}))
        for j0 in range(0, len(raws), a.batch):
            chunk = raws[j0:j0 + a.batch]
            try:
                reps = B.minimise_batch([c[4] for c in chunk], platform=a.platform)
            except Exception as exc:                                 # noqa: BLE001
                # isolate the failing model(s): each model of the chunk alone, as the single-model path
                print(f"batch minimisation failed ({len(chunk)} models), per model: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                reps = []
                for c in chunk:
                    try:
                        reps.append(B.minimise_model(c[4]["pdb_raw"], c[4]["pdb_out"], c[4]["core_residues"], platform=a.platform))
                    except Exception as e1:                          # noqa: BLE001
                        reps.append(f"failed: minimisation: {type(e1).__name__}: {str(e1)[:200]}")
            for (bi, X, k, n_res, spec), rep in zip(chunk, reps):
                rec = builts[bi]["classes"][X][k]; rec.pop("raw", None)
                if isinstance(rep, str):
                    rec["status"] = rep
                else:
                    rec.update(min=spec["pdb_out"], status="ok", energy_per_nt=rep["energy_final_kJ_mol"] / max(1, n_res))
        jobs = []
        for bi, (bt, w) in enumerate(zip(builts, works)):
            if bt is None:
                continue
            for X, models in bt.get("classes", {}).items():
                for k, m in enumerate(models):
                    if m.get("status") == "ok" and m.get("min"):
                        jobs.append((bi, X, k, {"pdb_min": m["min"], "core_residues": bt["core_idx"],
                                                "out_prefix": os.path.join(w, f"{X}{k}", "md")}))
        for j0 in range(0, len(jobs), a.batch):
            chunk = jobs[j0:j0 + a.batch]
            kw = {x: settings[x] for x in ("n_frames", "heat_ps", "equil_ps", "production_ps", "k_core", "salt_M", "temperature_K")}
            try:
                res = B.md_ensemble_batch([c[3] for c in chunk], platform=a.platform, seed=j0, **kw)
            except Exception as exc:                                 # noqa: BLE001
                # isolate the failing model(s): each model of the chunk alone
                print(f"batch MD failed ({len(chunk)} models), per model: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                res = []
                for i, c in enumerate(chunk):
                    try:
                        res.append(B.md_ensemble_batch([c[3]], platform=a.platform, seed=j0 + i, **kw)[0])
                    except Exception as e1:                          # noqa: BLE001
                        res.append(f"failed: MD: {type(e1).__name__}: {str(e1)[:200]}")
            for (bi, X, k, _), r_ in zip(chunk, res):
                rec = builts[bi]["classes"][X][k]
                if isinstance(r_, str):
                    rec["status"] = r_
                else:
                    rec.update(frames=r_["frames"], md_rmsd=r_["rmsd_to_minimised_A"])
        for r, bt, w in zip(block, builts, works):
            if bt is not None:
                st = [str(m.get("status", "")) for ms in bt.get("classes", {}).values() for m in ms]
                if any("Error initializing context" in x for x in st):
                    # a platform failure is not a property of the fragment: save nothing, so it is rebuilt
                    raise RuntimeError(f"{r['frag_id']}: OpenMM platform failure, record not saved")
                M.save_built(bt, os.path.join(bdir, f"{r['frag_id']}.json.gz"))
            shutil.rmtree(w, ignore_errors=True)
        print(f"block of {len(block)} fragments, {len(jobs)} models: {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
