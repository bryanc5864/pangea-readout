#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, i-motifs as explicit geometry (PLAN.md, Family M).

Models (`fold/imotif_build`): the fragment's C-rich strand threaded on solved tetramolecular
C.C+ cores (3 models: distinct templates and chain assignments), loops built from sequence with
the core restrained (AMBER OL15/GBn2). Electrostatics: each C.C+ pair hemiprotonated -- the two
cytosines carry the mean of DC and DC+ base charges (`physics/data/dcp_charges.json`) and the
shared proton sits at the N3-N3 midpoint with DC+'s H3 charge, so every pair carries +1 without
choosing which cytosine holds it -- solved with nonlinear PB (production settings, map-based
per-atom solvation) at 10, 150 and 500 mM NaCl. Channels: potential at the pair centres (the
cationic core), on the stack axis 6 A beyond each end, in the narrow and wide grooves 1 and 3 A
outside the molecular surface, at the loop nucleotides, per nucleotide, and the solvation free
energy; ensemble means over the models with the across-model spread of the main ones.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fold"))

FAMILY = "M"
N_MODELS = 3
CONDITIONS = (("Na010", "NaCl_010mM"), ("Na150", "NaCl_150mM"), ("Na500", "NaCl_500mM"))
SRC = ("PDB tetramolecular i-motif cores (data/imotif_templates_v1); loops from sequence (PDBFixer + "
       "AMBER OL15/GBn2 relaxation); RESP C+ charges; APBS nonlinear PB")
_Q = (("phi_core", "kT/e", "potential on the stack axis midway between consecutive C.C+ pairs, mean"),
      ("phi_end5", "kT/e", "potential 1 A outside the surface along the stack axis beyond the 5'-face pair"),
      ("phi_end3", "kT/e", "potential 1 A outside the surface along the stack axis beyond the 3'-face pair"),
      ("phi_narrow_d1", "kT/e", "narrow-groove potential 1 A outside the molecular surface"),
      ("phi_narrow_d3", "kT/e", "narrow-groove potential 3 A outside the molecular surface"),
      ("phi_wide_d1", "kT/e", "wide-groove potential 1 A outside the molecular surface"),
      ("phi_wide_d3", "kT/e", "wide-groove potential 3 A outside the molecular surface"),
      ("phi_loops_d1", "kT/e", "loop-nucleotide potential 1 A outside the surface, mean"),
      ("solvation", "kJ/mol", "electrostatic solvation free energy (per-atom map solvation)"))
_SPREAD = ("phi_core", "phi_narrow_d1", "solvation")


def channel_names_global() -> List[str]:
    out = ["M_im_modelled", "M_im_n_pairs"]
    for tag, _ in CONDITIONS:
        out += [f"M_im_{q}_{tag}" for q, _, _ in _Q] + [f"M_im_{q}_sd_{tag}" for q in _SPREAD]
    return out


def channel_names_per_bp() -> List[str]:
    out = ["M_im_in_model"]
    for tag, _ in CONDITIONS:
        out += [f"M_im_res_phi_d1_{tag}", f"M_im_res_phi_d3_{tag}"]
    return out


def _declare() -> None:
    reg = lambda **kw: REGISTRY.register(ChannelSpec(family=FAMILY, source=SRC, **kw))  # noqa: E731
    GL, PB, S = Resolution.GLOBAL, Resolution.PER_BP, Parity.SYMMETRIC
    reg(name="M_im_modelled", resolution=GL, parity=S, units="indicator",
        description="1 if the fragment's C-rich strand was modelled as an i-motif")
    reg(name="M_im_n_pairs", resolution=GL, parity=S, units="count", description="i-motif models: C.C+ pairs")
    for tag, cname in CONDITIONS:
        for q, u, d in _Q:
            reg(name=f"M_im_{q}_{tag}", resolution=GL, parity=S, units=u, condition_dependent=True,
                description=f"i-motif models, {cname}: {d} (mean over models)")
        for q in _SPREAD:
            reg(name=f"M_im_{q}_sd_{tag}", resolution=GL, parity=S, units="as the channel", condition_dependent=True,
                description=f"i-motif models, {cname}: spread of {q} across models")
        reg(name=f"M_im_res_phi_d1_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
            description=f"i-motif models, {cname}: nucleotide potential 1 A outside the surface")
        reg(name=f"M_im_res_phi_d3_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
            description=f"i-motif models, {cname}: nucleotide potential 3 A outside the surface")
    reg(name="M_im_in_model", resolution=PB, parity=S, units="indicator",
        description="1 for base pairs whose C-rich-strand nucleotide is in the i-motif models")


_declare()


def _empty(n):
    prof = {k: np.full(n, np.nan) for k in channel_names_per_bp()}
    glob = {k: float("nan") for k in channel_names_global()}
    prof["M_im_in_model"] = np.zeros(n); glob["M_im_modelled"] = 0.0
    return prof, glob


def _hemiprotonated_pqr(model_pdb: str, pairs: List[Tuple[int, int]], work: str) -> str:
    import apbs_run as A
    q = json.load(open(os.path.join(HERE, "data", "dcp_charges.json")))["base_charges"]
    pqr = os.path.join(work, "im.pqr"); A.run_pdb2pqr(model_pdb, pqr); A.sanitise_pqr(pqr)
    xyz, qq, rr, recs = A.read_pqr(pqr)
    paired = {r for p in pairs for r in p}
    lines, n3, serial = [], {}, 1
    for x, q0, r_, rec in zip(xyz, qq, rr, recs):
        rs, nm, rn = int(rec.get("resseq", -1)), rec.get("name", ""), rec.get("resname", "UNK")
        qv = float(q0)
        if rs in paired and rn.startswith("DC") and nm in q and nm != "H3":
            qv = 0.5 * (float(q0) + q[nm])           # mean of DC and DC+ base charges
        if rs in paired and nm == "N3":
            n3[rs] = np.asarray(x, float)
        lines.append(f"ATOM {serial:6d} {nm:<4s} {rn:>4s} {rec.get('chain', 'A') or 'A'} {rs:4d} "
                     f"{x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {qv:8.4f} {float(r_):7.4f}\n"); serial += 1
    for a, b in pairs:
        if a in n3 and b in n3:
            x = 0.5 * (n3[a] + n3[b])                  # the shared proton
            lines.append(f"ATOM {serial:6d} H3   DC   A {a:4d} {x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {q['H3']:8.4f} {0.6:7.4f}\n")
            serial += 1
    open(pqr, "w").write("".join(lines))
    return pqr


def _surface(origin, direction, xyz, rad, offsets=(1.0, 3.0)):
    import g4_electrostatics as E
    return E._surface_points(origin, direction, xyz, rad, offsets)


def _solve_model(model_pdb: str, info: dict, sel: dict, work: str) -> Dict[str, object]:
    import apbs_run as A
    import g4_templates as T
    import imotif_templates as I
    atoms, _ = T.read_pdb(model_pdb); res = T.residues(atoms)
    pairs_k = I.find_cc_pairs(res)
    pairs = [(a[1], b[1]) for a, b in pairs_k]
    if len(pairs) < 2:
        raise ValueError(f"{len(pairs)} C.C+ pairs left after relaxation")
    pqr = _hemiprotonated_pqr(model_pdb, pairs, work)
    xyz, q, rad, _ = A.read_pqr(pqr)
    cents = np.array([(res[a]["atoms"]["N3"] + res[b]["atoms"]["N3"]) / 2 for a, b in pairs_k])
    ax = np.linalg.svd(cents - cents.mean(0))[2][0]
    # orient 5' -> 3': the pair holding residue 1's tract end... use the first core residue
    first = min(pairs, key=lambda p: min(p))
    if (cents[pairs.index(first)] - cents.mean(0)) @ ax > 0:
        ax = -ax
    z = (cents - cents.mean(0)) @ ax
    c0 = cents.mean(0)
    u = np.cross(ax, [1.0, 0, 0]); u = u if np.linalg.norm(u) > 0.1 else np.cross(ax, [0, 1.0, 0]); u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    strands = {}
    core = info["core_residues"]
    nc = sel["n_c"]
    for k in range(4):
        strands[k] = core[k * nc:(k + 1) * nc]
    rays = []          # (origin, direction, groove class)
    for zi in np.sort(z):
        o = c0 + zi * ax
        az = []
        for k, rs in strands.items():
            pts = [res[(next(iter(res))[0], r)]["atoms"]["C1'"] for r in rs if (next(iter(res))[0], r) in res]
            if not pts:
                continue
            d = np.array([p - o - ((p - o) @ ax) * ax for p in pts])
            dd = d[np.argmin(np.abs([(p - o) @ ax for p in pts]))]
            az.append(np.arctan2(dd @ v, dd @ u))
        az = np.sort(np.array(az))
        if len(az) != 4:
            continue
        gaps = np.array([(az[(i + 1) % 4] - az[i]) % (2 * np.pi) for i in range(4)])
        narrow = set(np.argsort(gaps)[:2])
        for i in range(4):
            m = az[i] + gaps[i] / 2
            rays.append((o, np.cos(m) * u + np.sin(m) * v, "narrow" if i in narrow else "wide"))
    rres = {}
    for (ch, i), r in res.items():
        b = [r["atoms"][n] for n in ("N1", "C2", "N3", "C4", "C5", "C6") if n in r["atoms"]]
        if len(b) >= 3:
            bc = np.mean(b, axis=0); foot = c0 + ((bc - c0) @ ax) * ax; dvec = bc - foot
            if np.linalg.norm(dvec) > 1e-6:
                rres[i] = (foot, dvec / np.linalg.norm(dvec))
    centre = xyz.mean(0); ext = xyz.max(0) - xyz.min(0)
    out = {"n_pairs": len(pairs)}
    loop_res = [r for r in range(1, len(sel["order"]) + 1) if r not in core]
    for tag, cname in CONDITIONS:
        cond = A.PRODUCTION_CONDITIONS_BY_NAME[cname]
        lam = A.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
        coarse = ext + 2.0 * max(50.0, 3.0 * lam); fine = np.minimum(ext + 24.0, coarse)
        w = os.path.join(work, tag); os.makedirs(w, exist_ok=True); shutil.copy(pqr, os.path.join(w, "m.pqr"))
        A.run_apbs(A.write_apbs_input("m.pqr", "solv", w, cond, centre, coarse, fine, exact_fine_spacing=True, calc_energy=False), w)
        A.run_apbs(A.write_apbs_input("m.pqr", "ref", w, cond, centre, coarse, fine, exact_fine_spacing=True,
                                      reference=True, reference_maps=True, calc_energy=False), w)
        ps = A.DXMap.read(os.path.join(w, "solv_pot.dx")); pr = A.DXMap.read(os.path.join(w, "ref_pot.dx"))
        kT = 8.314462618e-3 * cond.temperature_K
        out[f"solvation_{tag}"] = float(np.sum(0.5 * q * (ps.interpolate(xyz) - pr.interpolate(xyz)) * kT))
        # core: axis points midway between consecutive pairs (the pair centres carry the shared
        # protons, where the map holds their own spline-smeared charge)
        zs = np.sort(z)
        mids = c0[None, :] + (0.5 * (zs[:-1] + zs[1:]))[:, None] * ax[None, :]
        out[f"phi_core_{tag}"] = float(np.mean(ps.interpolate(mids)))
        # stack ends: surface-referenced along the axis (loops can cap an end)
        for nm, zend, sgn in (("end5", zs[0], -1.0), ("end3", zs[-1], 1.0)):
            r0, pts = _surface(c0 + zend * ax, sgn * ax, xyz, rad, offsets=(1.0,))
            out[f"phi_{nm}_{tag}"] = float(ps.interpolate(pts)[0]) if pts is not None else float("nan")
        for cls in ("narrow", "wide"):
            v1, v3 = [], []
            for o, d, c in rays:
                if c != cls:
                    continue
                r0, pts = _surface(o, d, xyz, rad)
                if pts is not None:
                    val = ps.interpolate(pts); v1.append(val[0]); v3.append(val[1])
            out[f"phi_{cls}_d1_{tag}"] = float(np.mean(v1)) if v1 else float("nan")
            out[f"phi_{cls}_d3_{tag}"] = float(np.mean(v3)) if v3 else float("nan")
        rp1, rp3 = np.full(len(sel["order"]), np.nan), np.full(len(sel["order"]), np.nan)
        for i, (o, d) in rres.items():
            if 1 <= i <= len(sel["order"]):
                r0, pts = _surface(o, d, xyz, rad)
                if pts is not None:
                    val = ps.interpolate(pts); rp1[i - 1], rp3[i - 1] = val[0], val[1]
        out[f"res_d1_{tag}"], out[f"res_d3_{tag}"] = rp1, rp3
        out[f"phi_loops_d1_{tag}"] = float(np.nanmean(rp1[[r - 1 for r in loop_res]])) if loop_res else float("nan")
        shutil.rmtree(w, ignore_errors=True)
    return out


def imotif_channels(seq: str, n_models: int = N_MODELS, workroot: Optional[str] = None
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    import imotif_build as IB
    n = len(seq)
    prof, glob = _empty(n)
    sel = IB.select(seq)
    if sel is None:
        return prof, glob
    plans = IB.plan_models(sel, n_models)
    if not plans:
        return prof, glob
    work = tempfile.mkdtemp(prefix="im_", dir=workroot or os.environ.get("SLURM_TMPDIR") or None)
    try:
        results, errors = [], []
        for k, p in enumerate(plans):
            mw = os.path.join(work, f"m{k}"); os.makedirs(mw)
            try:
                pdb, info = IB.build_model(sel, p, mw, seed=k)
                results.append(_solve_model(pdb, info, sel, mw))
            except Exception as exc:                                 # noqa: BLE001
                errors.append(f"{p['template']}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
        if not results:
            raise RuntimeError("i-motif candidate, no model built: " + " | ".join(errors))
        glob["M_im_modelled"] = 1.0
        glob["M_im_n_pairs"] = float(np.mean([r["n_pairs"] for r in results]))
        for tag, _ in CONDITIONS:
            for qn, _, _ in _Q:
                v = np.array([r[f"{qn}_{tag}"] for r in results], float)
                glob[f"M_im_{qn}_{tag}"] = float(np.nanmean(v))
                if qn in _SPREAD:
                    glob[f"M_im_{qn}_sd_{tag}"] = float(np.nanstd(v)) if len(v) > 1 else float("nan")
            for d in (1, 3):
                rv = np.nanmean(np.stack([r[f"res_d{d}_{tag}"] for r in results]), axis=0)
                tr = np.full(n, np.nan)
                for i, p in enumerate(sel["positions"]):
                    tr[p] = rv[i]
                prof[f"M_im_res_phi_d{d}_{tag}"] = tr
        inm = np.zeros(n)
        for p in sel["positions"]:
            inm[p] = 1.0
        prof["M_im_in_model"] = inm
        return prof, glob
    finally:
        shutil.rmtree(work, ignore_errors=True)
