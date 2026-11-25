#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Tier 1 of the quantum chemistry ladder: GFN2-xTB on the base-pair fragment library.

The plan's electronic-structure family is computed on short stacked fragments and then
generalised to flanking context by the encoder.  It uses three tiers with delta-learning
between them; this is the cheap tier, which covers the broadest context set and whose
predictions the expensive tiers later correct.

Per context we run GFN2-xTB on seven geometries (the stacked dimer, each base pair, and
each of the four capped bases) and derive:

    stacking energy          E(dimer) - E(bp1) - E(bp2)
    hydrogen-bond energy     E(bp) - E(watson base) - E(crick base)
    HOMO, LUMO, gap          frontier orbital energies, per fragment
    vertical IP and EA       from --vipea, which is the quantity that sets where a hole
                             localises and therefore oxidative damage susceptibility
    dipole                   molecular dipole of the dimer
    partial charges          per-atom, written by xtb to its `charges` file

GFN2 includes a dispersion treatment, which matters because dispersion carries much of
the stacking stability.  It does **not** give the electrostatic / exchange-repulsion /
induction / dispersion decomposition the plan wants: that is symmetry-adapted
perturbation theory on the stratified subset, a separate tier.  Nothing here is labelled
as a decomposition.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

HARTREE_EV = 27.211386245988
HARTREE_KCAL = 627.5094740631

UNITS = ("dimer", "bp1", "bp2", "w1", "w2", "c1", "c2")


def find_xtb(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("XTB_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("xtb")
    if found:
        return found
    guess = "/apps/software/2024a/software/xtb/6.7.1-gfbf-2024a/bin/xtb"
    if os.path.exists(guess):
        return guess
    raise RuntimeError("xtb not found; set XTB_BIN or load the xtb module")


def parse_xtb(stdout: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", stdout)
    if m:
        out["energy_Eh"] = float(m.group(1))
    m = re.search(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)\s+eV", stdout)
    if m:
        out["gap_eV"] = float(m.group(1))
    # Orbital table lines look like: "   29        2.0000          -0.3626  -9.8671 (HOMO)"
    m = re.search(r"^\s*\d+\s+[\d.]+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\(HOMO\)",
                  stdout, re.M)
    if m:
        out["homo_eV"] = float(m.group(2))
    m = re.search(r"^\s*\d+\s+(?:[\d.]+\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\(LUMO\)",
                  stdout, re.M)
    if m:
        out["lumo_eV"] = float(m.group(2))
    m = re.search(r"molecular dipole:.*?full:\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)",
                  stdout, re.S)
    if m:
        out["dipole_debye"] = float(m.group(4))
    m = re.search(r"delta SCC IP \(eV\):\s*(-?\d+\.\d+)", stdout)
    if m:
        out["vip_eV"] = float(m.group(1))
    m = re.search(r"delta SCC EA \(eV\):\s*(-?\d+\.\d+)", stdout)
    if m:
        out["vea_eV"] = float(m.group(1))
    return out


def run_xtb_one(xyz: str, xtb: str, workdir: str, vipea: bool = False,
                gfn: int = 2, threads: int = 1, timeout_s: int = 900) -> Dict[str, object]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["XTBPATH"] = env.get("XTBPATH", "")
    cmd = [xtb, os.path.abspath(xyz), "--gfn", str(gfn), "--chrg", "0"]
    cmd.append("--vipea" if vipea else "--sp")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                         env=env, timeout=timeout_s)
    rec: Dict[str, object] = {"seconds": round(time.time() - t0, 2),
                              "returncode": res.returncode}
    # xtb prints its banner and results on stdout but the termination line on stderr,
    # so both streams have to be considered for the success check and for parsing.
    combined = (res.stdout or "") + "\n" + (res.stderr or "")
    if res.returncode != 0 or "normal termination" not in combined:
        rec["status"] = "failed"
        rec["stderr_tail"] = combined[-1500:]
        return rec
    rec["status"] = "ok"
    rec.update(parse_xtb(combined))
    charges_path = os.path.join(workdir, "charges")
    if os.path.exists(charges_path):
        with open(charges_path) as f:
            rec["charges"] = [float(x) for x in f.read().split()]
        os.remove(charges_path)
    return rec


def run_context(ctx_rec: dict, xtb: str, workroot: Optional[str], vipea: bool,
                threads: int) -> dict:
    ctx = ctx_rec["context"]
    out: Dict[str, object] = {"context": ctx, "central_step": ctx_rec["central_step"]}
    work = tempfile.mkdtemp(prefix=f"xtb_{ctx}_", dir=workroot)
    try:
        per: Dict[str, Dict[str, object]] = {}
        for tag in UNITS:
            path = ctx_rec["files"][tag]
            sub = os.path.join(work, tag)
            os.makedirs(sub, exist_ok=True)
            # vertical IP/EA only on the dimer and the pairs; bases inherit via the
            # expensive tiers if needed.
            want = vipea and tag in ("dimer", "bp1", "bp2")
            per[tag] = run_xtb_one(path, xtb, sub, vipea=False, threads=threads)
            if want and per[tag].get("status") == "ok":
                extra = run_xtb_one(path, xtb, sub, vipea=True, threads=threads)
                for k in ("vip_eV", "vea_eV"):
                    if k in extra:
                        per[tag][k] = extra[k]
        out["units"] = per

        def E(tag: str) -> Optional[float]:
            v = per.get(tag, {})
            return v.get("energy_Eh") if v.get("status") == "ok" else None

        e = {t: E(t) for t in UNITS}
        if all(e[t] is not None for t in ("dimer", "bp1", "bp2")):
            out["stacking_kcal_mol"] = (e["dimer"] - e["bp1"] - e["bp2"]) * HARTREE_KCAL
        if all(e[t] is not None for t in ("bp1", "w1", "c1")):
            out["hbond_bp1_kcal_mol"] = (e["bp1"] - e["w1"] - e["c1"]) * HARTREE_KCAL
        if all(e[t] is not None for t in ("bp2", "w2", "c2")):
            out["hbond_bp2_kcal_mol"] = (e["bp2"] - e["w2"] - e["c2"]) * HARTREE_KCAL
        for tag in ("dimer", "bp1", "bp2"):
            v = per.get(tag, {})
            for k in ("homo_eV", "lumo_eV", "gap_eV", "vip_eV", "vea_eV", "dipole_debye"):
                if k in v:
                    out[f"{tag}_{k}"] = v[k]
        out["status"] = "ok" if "stacking_kcal_mol" in out else "incomplete"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="qc_fragment_index.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xtb-bin", default=None)
    ap.add_argument("--workroot", default="/tmp")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--vipea", action="store_true",
                    help="also compute vertical IP and EA (roughly triples the cost)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0,
                    help="run contexts[shard::n_shards] (for array jobs; merge_xtb_shards.py)")
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()

    xtb = find_xtb(a.xtb_bin)
    with open(a.index) as f:
        index = json.load(f)
    contexts = index["contexts"][: a.limit] if a.limit else index["contexts"]
    contexts = contexts[a.shard::a.n_shards]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_context, c, xtb, a.workroot, a.vipea, a.threads): c["context"]
                for c in contexts}
        for fu in concurrent.futures.as_completed(futs):
            ctx = futs[fu]
            try:
                r = fu.result()
            except Exception as exc:
                r = {"context": ctx, "status": "failed",
                     "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            st = r.get("stacking_kcal_mol")
            print(f"[{r['status']:10s}] {ctx}  step {r.get('central_step','??')}  "
                  f"stack={'' if st is None else f'{st:8.3f}'} kcal/mol", flush=True)

    results.sort(key=lambda r: r["context"])
    ok = [r for r in results if r["status"] == "ok"]
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "xtb": xtb,
        "method": "GFN2-xTB single point",
        "n_contexts": len(results),
        "n_ok": len(ok),
        "wall_seconds": round(time.time() - t0, 1),
        "index": a.index,
        "shard": [a.shard, a.n_shards],
        "note": "stacking and hydrogen-bond values are total interaction energies, "
                "not component decompositions; the decomposition requires the SAPT tier",
        "results": results,
    }
    if ok:
        vals = [r["stacking_kcal_mol"] for r in ok if "stacking_kcal_mol" in r]
        if vals:
            summary["stacking_kcal_mol_range"] = [min(vals), max(vals)]
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{len(ok)}/{len(results)} contexts ok in {summary['wall_seconds']}s "
          f"-> {a.out}")
    if "stacking_kcal_mol_range" in summary:
        lo, hi = summary["stacking_kcal_mol_range"]
        print(f"stacking energy range: {lo:.2f} to {hi:.2f} kcal/mol")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
