#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Feasibility probe for the Poisson-Boltzmann stage: cost against accuracy.

The plan budgets 15,000 to 30,000 CPU-hours for about 1.8 million nonlinear solves,
which allows roughly 30 to 60 seconds each.  A first attempt at a 61 bp fragment with a
0.35 A fine grid over the whole scored region took minutes and, with insufficient
coarse padding, aborted outright.  Guessing a cheaper configuration would risk trading
away accuracy invisibly, so this probe measures both together.

It sweeps, on the same sequence and the same conformation:

  solute length     how much duplex is handed to the solver (the conformation always
                    comes from the full cgNA+ context, so only the solver box shrinks)
  fine spacing      grid resolution over the scored region
  transverse pad    coarse-grid padding, which sets the quality of the Debye-Huckel
                    boundary condition and whether npbe converges at all
  equation          npbe against lpbe

and reports, for every configuration, the wall time, whether it converged, and the
sampled shell potentials.  Accuracy is judged by agreement with the most expensive
configuration that converged, channel by channel, rather than by a single energy number:
the channels are what PANGEA is trained on.

Output is a JSON record per configuration, so the production configuration is chosen
from measurement and the choice is auditable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import cgna_atomic as ca  # noqa: E402
import frames as frames_mod  # noqa: E402
import minimize as minimise_mod  # noqa: E402


def central_window(seq: str, keep: int) -> Tuple[str, int]:
    """Central `keep` bases of `seq`, and the offset at which they start."""
    if keep >= len(seq):
        return seq, 0
    start = (len(seq) - keep) // 2
    return seq[start:start + keep], start


def prepare_structure(seq: str, solute_bp: int, workdir: str) -> Dict[str, object]:
    """Build the full-context conformation, then trim the solute handed to the solver.

    Trimming after building is the point: the central base pairs keep the geometry they
    had in the full 61 bp context, so shortening the solute changes only the
    electrostatic boundary, never the conformation being measured.
    """
    gs = ca.CGNAGroundState(seq)
    templates = ca.NucleotideTemplates()
    atoms = ca.build_duplex(seq, templates, gs)

    n = len(seq)
    if solute_bp < n:
        lo = (n - solute_bp) // 2
        hi = lo + solute_bp                      # Watson residues lo+1 .. hi (1-based)
        keep = []
        for a in atoms:
            if a.chain == "A":
                if lo < a.resseq <= hi:
                    keep.append(a)
            else:
                # Crick residue j pairs with Watson n+1-j.
                w = n + 1 - a.resseq
                if lo < w <= hi:
                    keep.append(a)
        # Renumber so the written PDB is a clean duplex.
        a_map, b_map = {}, {}
        for a in keep:
            m = a_map if a.chain == "A" else b_map
            if a.resseq not in m:
                m[a.resseq] = len(m) + 1
        for a in keep:
            m = a_map if a.chain == "A" else b_map
            a.resseq = m[a.resseq]

        # Re-terminate. The residue that becomes the new 5' end still carries the
        # phosphate group it inherited from the full-length chain, which leaves a
        # dangling 5'-phosphate the AMBER DNA templates do not match ("no template
        # found ... is the chain terminated in a way that is unsupported?"). Dropping
        # P, OP1 and OP2 there reproduces the 5'-OH terminus the untrimmed build has.
        first_of_chain = {}
        for a in keep:
            first_of_chain[a.chain] = min(first_of_chain.get(a.chain, a.resseq), a.resseq)
        atoms = [a for a in keep
                 if not (a.resseq == first_of_chain[a.chain]
                         and a.name in ("P", "OP1", "OP2"))]

    raw = os.path.join(workdir, "raw.pdb")
    ca.write_pdb(atoms, raw)
    minim = os.path.join(workdir, "min.pdb")
    min_report = minimise_mod.restrained_minimise(raw, minim)
    bp, _ = frames_mod.bp_frames_from_pdb(minim)
    return {"pdb": minim, "bp": bp, "n_bp": len(bp), "minimisation": min_report}


def run_config(pdb: str, bp: np.ndarray, scored: slice, condition: apbs_run.IonicCondition,
               fine_spacing: float, transverse_pad: float, axial_pad: float,
               equation: str, workdir: str, timeout_s: int = 1800) -> Dict[str, object]:
    pqr = os.path.join(workdir, "frag.pqr")
    if not os.path.exists(pqr):
        apbs_run.run_pdb2pqr(pdb, pqr, log=os.path.join(workdir, "pdb2pqr.log"))
        apbs_run.sanitise_pqr(pqr)
    xyz, q, r, recs = apbs_run.read_pqr(pqr)

    extent = xyz.max(axis=0) - xyz.min(axis=0)
    centre = xyz.mean(axis=0)
    long_axis = int(np.argmax(extent))

    pad = np.full(3, transverse_pad)
    pad[long_axis] = axial_pad
    coarse = extent + 2.0 * pad

    scored_pts = bp[scored][:, :3, 3]
    fine_centre = scored_pts.mean(axis=0)
    rmax = max(apbs_run.RADIAL_OFFSETS_A)
    fine = np.full(3, 2.0 * (rmax + 6.0))
    span = scored_pts.max(axis=0) - scored_pts.min(axis=0)
    fine[long_axis] = span[long_axis] + 20.0
    fine = np.minimum(fine, coarse)

    name = (f"{condition.name}_s{fine_spacing:g}_tp{transverse_pad:g}"
            f"_ap{axial_pad:g}_{equation}")
    dime = [apbs_run._grid_dims(fine[i], fine_spacing) for i in range(3)]

    body_path = apbs_run.write_apbs_input(
        os.path.basename(pqr), name, workdir, condition, fine_centre, coarse, fine,
        fine_spacing_A=fine_spacing,
    )
    if equation == "lpbe":
        with open(body_path) as f:
            text = f.read().replace("    npbe\n", "    lpbe\n")
        with open(body_path, "w") as f:
            f.write(text)

    rec: Dict[str, object] = {
        "condition": condition.name, "equation": equation,
        "fine_spacing_A": fine_spacing, "transverse_pad_A": transverse_pad,
        "axial_pad_A": axial_pad, "n_atoms": int(len(xyz)),
        "total_charge": float(q.sum()),
        "extent_A": [round(float(x), 1) for x in extent],
        "coarse_A": [round(float(x), 1) for x in coarse],
        "fine_A": [round(float(x), 1) for x in fine],
        "dime": dime, "grid_points": int(np.prod(dime)),
        "coarse_spacing_A": [round(float(coarse[i] / dime[i]), 3) for i in range(3)],
    }

    t0 = time.time()
    try:
        exe = apbs_run.which_or_die("apbs")
        res = subprocess.run([exe, os.path.basename(body_path)], cwd=workdir,
                             capture_output=True, text=True, timeout=timeout_s)
        out = res.stdout + "\n" + res.stderr
        rec["seconds"] = round(time.time() - t0, 1)
        rec["returncode"] = res.returncode
        rec["exp_overflow_lines"] = out.count("exp overflows")
        if res.returncode != 0 or "Global net ELEC" not in out:
            rec["status"] = "failed"
            rec["reason"] = ("overflow/abort" if "large potential values" in out
                             else f"rc={res.returncode}")
            return rec
        rec["status"] = "ok"
        rec["elec_energy_kJ_mol"] = apbs_run.parse_elec_energy(out)
        pot_path = os.path.join(workdir, name + "_pot.dx")
        if os.path.exists(pot_path):
            pot = apbs_run.DXMap.read(pot_path)
            sampled = apbs_run.sample_potential(pot, bp[scored])
            v = sampled["potential"]
            rec["potential_mean_by_radius"] = [
                float(x) for x in np.nanmean(v, axis=(0, 1))
            ]
            rec["potential_mean_by_sector"] = [
                float(x) for x in np.nanmean(v, axis=(0, 2))
            ]
            rec["potential_finite_fraction"] = float(np.isfinite(v).mean())
            rec["_channels"] = v.tolist()
            os.remove(pot_path)
        nd = os.path.join(workdir, name + "_ndens.dx")
        if os.path.exists(nd):
            os.remove(nd)
    except subprocess.TimeoutExpired:
        rec["seconds"] = round(time.time() - t0, 1)
        rec["status"] = "timeout"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--fragment-bp", type=int, default=61)
    ap.add_argument("--solute-bp", type=int, nargs="+", default=[61, 41])
    ap.add_argument("--fine-spacing", type=float, nargs="+", default=[0.35, 0.5])
    ap.add_argument("--transverse-pad", type=float, nargs="+", default=[30.0, 50.0])
    ap.add_argument("--axial-pad", type=float, default=40.0)
    ap.add_argument("--equation", nargs="+", default=["npbe"])
    ap.add_argument("--condition", default="NaCl_150mM")
    ap.add_argument("--scored-bp", type=int, default=21)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default="/tmp")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    seq = a.sequence
    if seq is None:
        rng = np.random.default_rng(20261119)
        seq = "".join(rng.choice(list("ACGT"), size=a.fragment_bp))
    cond = apbs_run.CONDITIONS_BY_NAME[a.condition]

    results: List[Dict[str, object]] = []
    for solute_bp in a.solute_bp:
        work = tempfile.mkdtemp(prefix=f"probe_{solute_bp}_", dir=a.workroot)
        try:
            st = prepare_structure(seq, solute_bp, work)
            bp = st["bp"]
            n_bp = st["n_bp"]
            keep = min(a.scored_bp, n_bp)
            start = (n_bp - keep) // 2
            scored = slice(start, start + keep)
            for spacing, tpad, eqn in itertools.product(
                    a.fine_spacing, a.transverse_pad, a.equation):
                rec = run_config(st["pdb"], bp, scored, cond, spacing, tpad,
                                 a.axial_pad, eqn, work, a.timeout)
                rec["solute_bp"] = solute_bp
                rec["n_bp_frames"] = n_bp
                rec["scored_bp"] = [scored.start, scored.stop]
                results.append(rec)
                print(f"[{rec['status']:8s}] solute={solute_bp:3d} h={spacing:.2f} "
                      f"tpad={tpad:4.0f} {eqn:5s} dime={rec['dime']} "
                      f"pts={rec['grid_points']/1e6:5.2f}M "
                      f"t={rec.get('seconds', 0):7.1f}s "
                      f"{rec.get('reason', '')}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # Accuracy against the most expensive converged configuration.
    ok = [r for r in results if r["status"] == "ok" and "_channels" in r]
    reference = max(ok, key=lambda r: r["grid_points"]) if ok else None
    if reference is not None:
        ref = np.asarray(reference["_channels"], dtype=float)
        for r in ok:
            v = np.asarray(r["_channels"], dtype=float)
            if v.shape != ref.shape:
                r["vs_reference"] = "shape mismatch (different scored window)"
                continue
            m = np.isfinite(v) & np.isfinite(ref)
            if m.sum() < 10:
                continue
            diff = v[m] - ref[m]
            denom = np.std(ref[m]) or 1.0
            r["vs_reference"] = {
                "max_abs_diff_kT_e": float(np.abs(diff).max()),
                "rmse_kT_e": float(np.sqrt((diff**2).mean())),
                "rmse_over_sd": float(np.sqrt((diff**2).mean()) / denom),
                "pearson_r": float(np.corrcoef(v[m], ref[m])[0, 1]),
            }
        reference["is_reference"] = True

    for r in results:
        r.pop("_channels", None)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({
            "sequence": seq,
            "condition": a.condition,
            "scored_bp": a.scored_bp,
            "reference_config": None if reference is None else {
                k: reference[k] for k in ("solute_bp", "fine_spacing_A",
                                          "transverse_pad_A", "equation",
                                          "grid_points", "seconds")
            },
            "configs": results,
        }, f, indent=2)
    print(f"\n-> {a.out}")
    for r in results:
        if r["status"] == "ok" and isinstance(r.get("vs_reference"), dict):
            vr = r["vs_reference"]
            print(f"  solute={r['solute_bp']:3d} h={r['fine_spacing_A']:.2f} "
                  f"tpad={r['transverse_pad_A']:4.0f} {r['equation']:5s} "
                  f"t={r['seconds']:7.1f}s  r={vr['pearson_r']:.5f} "
                  f"rmse/sd={vr['rmse_over_sd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
