#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, cruciforms as explicit geometry (PLAN.md, Family M).

Models (`fold/cruciform_build`): the fragment's best inverted repeat extruded as two hairpins on a
solved stacked-X four-way junction (two templates, 1DCW and 1L4J, so two models), helices from
cgNA+ on the fragment's long window, loops built from sequence, junction joins closed by AMBER
OL15/GBn2 relaxation. Electrostatics: nonlinear PB (production settings, map-based solvation)
at 10, 150 and 500 mM NaCl. Because a junction has no single helix axis, the per-nucleotide
potential is the mean over solvent grid points 0-3 A beyond the probe-accessible surface whose
nearest atom belongs to that nucleotide (both strands' nucleotides at a position averaged).
Channels: the potential at the junction centre (mean of the four junction-proximal base pairs'
C1' midpoints) and in the solvent within 6 A of it -- the cation-binding pocket of four-way
junctions -- the means over hairpin loops, stems and the junction-adjacent flanks, the solvation
free energy, per-nucleotide potentials, and the across-model spread of the main ones.
"""
from __future__ import annotations

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
CONDITIONS = (("Na010", "NaCl_010mM"), ("Na150", "NaCl_150mM"), ("Na500", "NaCl_500mM"))
APBS_TIMEOUT_S = 4 * 3600          # one solve on the whole-model 0.35 A grid: minutes to ~1 h
SRC = ("PDB four-way junctions 1DCW and 1L4J (data/junction_templates_v1); cgNA+ helices; loops from "
       "sequence; AMBER OL15/GBn2 relaxation; APBS nonlinear PB")
_Q = (("phi_junction", "kT/e", "potential at the junction centre"),
      ("phi_junction_shell", "kT/e", "mean solvent potential within 6 A of the junction centre"),
      ("phi_loops", "kT/e", "mean nucleotide potential over the two hairpin loops"),
      ("phi_stems", "kT/e", "mean nucleotide potential over the two hairpin stems"),
      ("phi_flanks", "kT/e", "mean nucleotide potential over the 4 flank base pairs next to the junction"),
      ("solvation", "kJ/mol", "electrostatic solvation free energy (per-atom map solvation)"))
_SPREAD = ("phi_junction", "phi_junction_shell", "solvation")


def channel_names_global() -> List[str]:
    out = ["M_cr_modelled", "M_cr_stem_bp", "M_cr_loop_nt"]
    for tag, _ in CONDITIONS:
        out += [f"M_cr_{q}_{tag}" for q, _, _ in _Q] + [f"M_cr_{q}_sd_{tag}" for q in _SPREAD]
    return out


def channel_names_per_bp() -> List[str]:
    return ["M_cr_in_model"] + [f"M_cr_res_phi_{tag}" for tag, _ in CONDITIONS]


def _declare() -> None:
    reg = lambda **kw: REGISTRY.register(ChannelSpec(family=FAMILY, source=SRC, **kw))  # noqa: E731
    GL, PB, S = Resolution.GLOBAL, Resolution.PER_BP, Parity.SYMMETRIC
    reg(name="M_cr_modelled", resolution=GL, parity=S, units="indicator",
        description="1 if the fragment's best inverted repeat was modelled as an extruded cruciform")
    reg(name="M_cr_stem_bp", resolution=GL, parity=S, units="bp", description="cruciform models: hairpin stem length")
    reg(name="M_cr_loop_nt", resolution=GL, parity=S, units="nt", description="cruciform models: hairpin loop length")
    for tag, cname in CONDITIONS:
        for q, u, d in _Q:
            reg(name=f"M_cr_{q}_{tag}", resolution=GL, parity=S, units=u, condition_dependent=True,
                description=f"cruciform models, {cname}: {d} (mean over models)")
        for q in _SPREAD:
            reg(name=f"M_cr_{q}_sd_{tag}", resolution=GL, parity=S, units="as the channel", condition_dependent=True,
                description=f"cruciform models, {cname}: spread of {q} across models")
        reg(name=f"M_cr_res_phi_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
            description=f"cruciform models, {cname}: solvent potential next to the nucleotides at this position")
    reg(name="M_cr_in_model", resolution=PB, parity=S, units="indicator",
        description="1 for positions inside the modelled cruciform window")


_declare()


def _empty(n):
    prof = {k: np.full(n, np.nan) for k in channel_names_per_bp()}
    glob = {k: float("nan") for k in channel_names_global()}
    prof["M_cr_in_model"] = np.zeros(n); glob["M_cr_modelled"] = 0.0
    return prof, glob


def _residue_potentials(pot, xyz, rad, owner, n_res):
    """Per residue: mean potential over map points 0-3 A beyond the 1.4 A probe surface, nearest
    atom in that residue."""
    from scipy.spatial import cKDTree
    g = [pot.origin[k] + pot.delta[k] * np.arange(pot.data.shape[k]) for k in range(3)]
    lo, hi = xyz.min(0) - 6, xyz.max(0) + 6
    s = [np.flatnonzero((g[k] >= lo[k]) & (g[k] <= hi[k])) for k in range(3)]
    I, J, K = np.meshgrid(*s, indexing="ij")
    idx = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    pts = pot.origin + idx * pot.delta
    d, nn = cKDTree(xyz).query(pts)
    gap = d - rad[nn] - 1.4
    keep = (gap >= 0) & (gap <= 3.0)
    phi = pot.data[idx[keep, 0], idx[keep, 1], idx[keep, 2]]
    o = owner[nn[keep]]
    out = np.full(n_res, np.nan)
    sums = np.bincount(o, weights=phi, minlength=n_res); cnt = np.bincount(o, minlength=n_res)
    ok = cnt > 0
    out[ok] = sums[ok] / cnt[ok]
    return out


def cruciform_channels(seq: str, window: str, frag_offset: int, workroot: Optional[str] = None
                       ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    import apbs_run as A
    import cruciform_build as CB
    import g4_templates as T
    n = len(seq)
    prof, glob = _empty(n)
    if window[frag_offset:frag_offset + n].upper() != seq.upper():
        raise ValueError("window does not contain the fragment at the offset")
    sel = CB.select(window, frag_offset, n)
    if sel is None:
        return prof, glob
    W = len(window)
    work = tempfile.mkdtemp(prefix="cr_", dir=workroot or os.environ.get("SLURM_TMPDIR") or None)
    try:
        results, errors = [], []
        for k, tpl in enumerate(CB.TEMPLATES):
            mw = os.path.join(work, f"m{k}"); os.makedirs(mw)
            try:
                pdb, info = CB.build_model(sel, tpl, mw, seed=k)
            except Exception as exc:                                 # noqa: BLE001
                errors.append(f"{tpl}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            pqr = os.path.join(mw, "cr.pqr"); A.run_pdb2pqr(pdb, pqr); A.sanitise_pqr(pqr)
            xyz, q, rad, recs = A.read_pqr(pqr)
            keys = [(r.get("chain", ""), int(r.get("resseq", -1))) for r in recs]
            uniq = sorted(set(keys), key=lambda kk: (kk[0], kk[1]))
            rid = {kk: i for i, kk in enumerate(uniq)}
            owner = np.array([rid[kk] for kk in keys])
            atoms, _ = T.read_pdb(pdb); res = T.residues(atoms)
            # window position of every residue (top: chain A in order; bottom: chain B)
            nums_a = sorted(r for c, r in res if c == "A"); nums_b = sorted(r for c, r in res if c == "B")
            pos_of = {("A", r): p for r, p in zip(nums_a, info["top_positions"])}
            pos_of.update({("B", r): p for r, p in zip(nums_b, info["bottom_positions"])})
            jres = info["junction_residues"]
            jc = np.mean([res[kk]["atoms"]["C1'"] for kk in jres if kk in res and "C1'" in res[kk]["atoms"]], axis=0)
            (l0, l1), (p0, p1), (r0, r1) = sel["left"], sel["loop"], sel["right"]
            groups = {"loops": set(range(p0, p1)), "stems": set(range(l0, l1)) | set(range(r0, r1)),
                      "flanks": set(range(l0 - 4, l0)) | set(range(r1, r1 + 4))}
            centre = xyz.mean(0); ext = xyz.max(0) - xyz.min(0)
            r = {}
            for tag, cname in CONDITIONS:
                cond = A.PRODUCTION_CONDITIONS_BY_NAME[cname]
                lam = A.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
                coarse = ext + 2.0 * max(50.0, 3.0 * lam); fine = np.minimum(ext + 24.0, coarse)
                w = os.path.join(mw, tag); os.makedirs(w); shutil.copy(pqr, os.path.join(w, "m.pqr"))
                # the reference (uniform dielectric, no ions) goes through APBS's Newton path: the same
                # linear problem, without the banded coarse-level factorisation that on this grid
                # takes over an hour and 27 GB (apbs_run.write_apbs_input)
                A.run_apbs(A.write_apbs_input("m.pqr", "solv", w, cond, centre, coarse, fine, exact_fine_spacing=True,
                                              calc_energy=False, write_ndens=False), w, timeout_s=APBS_TIMEOUT_S)
                A.run_apbs(A.write_apbs_input("m.pqr", "ref", w, cond, centre, coarse, fine, exact_fine_spacing=True,
                                              reference=True, reference_maps=True, calc_energy=False,
                                              reference_equation="npbe"), w, timeout_s=APBS_TIMEOUT_S)
                ps = A.DXMap.read(os.path.join(w, "solv_pot.dx")); pr = A.DXMap.read(os.path.join(w, "ref_pot.dx"))
                kT = 8.314462618e-3 * cond.temperature_K
                r[f"solvation_{tag}"] = float(np.sum(0.5 * q * (ps.interpolate(xyz) - pr.interpolate(xyz)) * kT))
                r[f"phi_junction_{tag}"] = float(ps.interpolate(jc[None])[0])
                # solvent points within 6 A of the centre, outside the probe-inflated atoms
                g = np.stack(np.meshgrid(*[np.arange(-6, 6.01, 0.5)] * 3, indexing="ij"), -1).reshape(-1, 3)
                g = g[np.linalg.norm(g, axis=1) <= 6.0] + jc
                dd = np.min(np.linalg.norm(g[:, None, :] - xyz[None, :, :], axis=-1) - rad[None, :], axis=1)
                acc = g[dd > 1.4]
                r[f"phi_junction_shell_{tag}"] = float(np.mean(ps.interpolate(acc))) if len(acc) else float("nan")
                rp = _residue_potentials(ps, xyz, rad, owner, len(uniq))
                per_pos = {}
                for kk, i in rid.items():
                    p = pos_of.get(kk)
                    if p is not None and np.isfinite(rp[i]):
                        per_pos.setdefault(p, []).append(rp[i])
                pp = {p: float(np.mean(v)) for p, v in per_pos.items()}
                r[f"res_{tag}"] = pp
                for gname, ps_ in groups.items():
                    vals = [pp[p] for p in ps_ if p in pp]
                    r[f"phi_{gname}_{tag}"] = float(np.mean(vals)) if vals else float("nan")
                shutil.rmtree(w, ignore_errors=True)
            results.append(r)
        if not results:
            # a candidate whose every model failed is an error, not "no cruciform"
            raise RuntimeError("cruciform candidate, no model built: " + " | ".join(errors))
        glob["M_cr_modelled"] = 1.0
        glob["M_cr_stem_bp"] = float(sel["arm"]); glob["M_cr_loop_nt"] = float(sel["loop"][1] - sel["loop"][0])
        for tag, _ in CONDITIONS:
            for qn, _, _ in _Q:
                v = np.array([r[f"{qn}_{tag}"] for r in results], float)
                glob[f"M_cr_{qn}_{tag}"] = float(np.nanmean(v))
                if qn in _SPREAD:
                    # spread over the models that have a value (a compact junction can leave no solvent within
                    # 6 A of its centre, so that model's shell potential is NaN); undefined with fewer than two
                    fin = v[np.isfinite(v)]
                    glob[f"M_cr_{qn}_sd_{tag}"] = float(np.std(fin)) if len(fin) > 1 else float("nan")
            tr = np.full(n, np.nan)
            for i in range(n):
                vals = [r[f"res_{tag}"].get(frag_offset + i) for r in results]
                vals = [x for x in vals if x is not None]
                if vals:
                    tr[i] = float(np.mean(vals))
            prof[f"M_cr_res_phi_{tag}"] = tr
        prof["M_cr_in_model"] = np.ones(n)
        return prof, glob
    finally:
        shutil.rmtree(work, ignore_errors=True)
