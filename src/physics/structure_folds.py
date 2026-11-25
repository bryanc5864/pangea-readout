#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M: non-canonical folds as explicit geometry (PLAN.md), quadruplexes (v3).

For a fragment whose best quadruplex candidate (Family F's) has four G tracts (`fold/g4_select.
select_v3`: three-tetrad cores from tracts of >= 3 G, else two-tetrad cores from tracts of >= 2
G), models are threaded on solved structures for each topology class -- parallel (p), hybrid
3+1 (h), antiparallel 2+2 (a) -- from template set v2 (`fold/g4_multi`): a solved core of the
class with each loop from a solved loop of the same type and length, the ensemble spanning the
class's folds (hybrid-1/2, basket/chair) that sequence alone cannot choose between. Each model is
minimised (AMBER OL15 + GBn2, core restrained), run as a GBn2 MD ensemble with the core held
(`g4_build.md_ensemble`, the loops and outer backbone sample; 0.15 M screening) and every
structure solved with nonlinear PB at 150 mM KCl and NaCl (`fold/g4_electrostatics`; frames
reuse the minimised model's tetrads). Channels per class X: the ensemble mean over models and MD
frames, the spread across models (fold and loop uncertainty) and the MD fluctuation (mean
within-model SD over frames) for the main quantities, per-nucleotide potentials and MD RMSF.
A class with no feasible core for the fragment's loops is not modelled (M_g4X_modelled = 0,
NaN elsewhere; masked in training) -- e.g. a 1-nt lateral loop exists in no solved structure.
Requested explicitly, like Family I: it is not in `landscape.FAMILIES`.
"""
from __future__ import annotations

import gzip
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
_REL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "g4_templates_v2")
TEMPLATE_DIR = os.environ.get(
    "PANGEA_G4_TEMPLATES",
    _REL if os.path.exists(os.path.join(_REL, "index.json"))
    else "/home/bjc75/other/biophyschem/data/g4_templates_v2")
CLASSES = (("p", "parallel"), ("h", "hybrid"), ("a", "antiparallel"))
CONDITIONS = (("K150", "KCl_150mM"), ("Na150", "NaCl_150mM"))
MD_SETTINGS = {"n_frames": 5, "heat_ps": 20.0, "equil_ps": 100.0, "production_ps": 500.0,
               "k_core": 2000.0, "salt_M": 0.15, "temperature_K": 300.0}
SRC = ("PDB unimolecular quadruplex structures, all topologies (data/g4_templates_v2); AMBER "
       "OL15/GBn2 minimisation and MD; APBS nonlinear PB; Rashin & Honig 1985 ion radii")

_PER_COND = (
    ("phi_sites", "kT/e", "potential at the inner ion sites (midway between tetrad planes), mean"),
    ("phi_planes", "kT/e", "potential at the tetrad-plane centres, mean"),
    ("phi_outer5", "kT/e", "potential at the 5'-face outer ion site"),
    ("phi_outer3", "kT/e", "potential at the 3'-face outer ion site"),
    ("phi_channel_min", "kT/e", "most negative potential along the central channel"),
    ("phi_far5", "kT/e", "potential on the stack axis 6 A beyond the 5' tetrad"),
    ("phi_far3", "kT/e", "potential on the stack axis 6 A beyond the 3' tetrad"),
    ("phi_groove_d1", "kT/e", "groove potential 1 A outside the molecular surface, mean"),
    ("phi_groove_d1_min", "kT/e", "groove potential 1 A outside the surface, most negative"),
    ("phi_groove_d3", "kT/e", "groove potential 3 A outside the molecular surface, mean"),
    ("phi_loops_d1", "kT/e", "loop-nucleotide potential 1 A outside the surface, mean"),
    ("phi_loops_d3", "kT/e", "loop-nucleotide potential 3 A outside the surface, mean"),
    ("solvation", "kJ/mol", "electrostatic solvation free energy (per-atom, solvated - reference)"),
    ("ion_binding", "kJ/mol", "electrostatic binding free energy of the inner-site cations"),
    ("ion_desolvation", "kJ/mol", "desolvation part of that binding free energy"),
)
_KEY = {"phi_groove_d1": "phi_groove_d1_mean_{c}", "phi_groove_d1_min": "phi_groove_d1_min_{c}",
        "phi_groove_d3": "phi_groove_d3_mean_{c}", "solvation": "solvation_kJ_mol_{c}",
        "ion_binding": "ion_binding_kJ_mol_{c}", "ion_desolvation": "ion_desolvation_kJ_mol_{c}"}
_SPREAD = ("phi_sites", "phi_loops_d1", "ion_binding", "solvation")             # across models
_MDSPREAD = ("phi_sites", "phi_loops_d1", "phi_groove_d1", "solvation")         # over MD frames
_GLOBAL = (
    ("ion_coulomb", "kJ/mol", "Coulomb attraction of the inner-site cations in the solute dielectric (K+)"),
    ("selectivity_KNa", "kJ/mol", "electrostatic K+ minus Na+ binding free energy (negative favours K+)"),
    ("dipole", "debye", "dipole moment about the centre of geometry"),
    ("rg", "A", "heavy-atom radius of gyration"),
    ("groove_depth", "A", "surface radius along the groove directions from the stack axis, mean"),
    ("model_spread", "A", "mean pairwise heavy-atom RMSD between the minimised models"),
    ("energy_per_nt", "kJ/mol", "minimised AMBER/GBn2 energy per nucleotide"),
    ("md_rmsd", "A", "heavy-atom RMSD of the MD frames from their minimised model, mean"),
    ("loop_rmsf", "A", "heavy-atom RMSF of the loop nucleotides over the MD frames, mean"),
    ("fit", "A", "loop anchor superposition score of the models (sum over loops), mean"),
)


def channel_names_global() -> List[str]:
    out = []
    for X, _ in CLASSES:
        out += [f"M_g4{X}_modelled", f"M_g4{X}_n_tetrads"]
        for tag, _ in CONDITIONS:
            out += [f"M_g4{X}_{n}_{tag}" for n, _, _ in _PER_COND]
            out += [f"M_g4{X}_{n}_sd_{tag}" for n in _SPREAD]
            out += [f"M_g4{X}_{n}_mdsd_{tag}" for n in _MDSPREAD]
        out += [f"M_g4{X}_{n}" for n, _, _ in _GLOBAL]
    return out


def channel_names_per_bp() -> List[str]:
    out = []
    for X, _ in CLASSES:
        out += [f"M_g4{X}_in_model", f"M_g4{X}_res_rmsf"]
        for tag, _ in CONDITIONS:
            out += [f"M_g4{X}_res_phi_d1_{tag}", f"M_g4{X}_res_phi_d3_{tag}"]
    return out


def _declare() -> None:
    reg = lambda **kw: REGISTRY.register(ChannelSpec(family=FAMILY, source=SRC, **kw))  # noqa: E731
    GL, PB, S = Resolution.GLOBAL, Resolution.PER_BP, Parity.SYMMETRIC
    for X, topo in CLASSES:
        w = f"{topo} quadruplex models"
        reg(name=f"M_g4{X}_modelled", resolution=GL, parity=S, units="indicator",
            description=f"1 if the fragment's best quadruplex was modelled in 3D as {topo}")
        reg(name=f"M_g4{X}_n_tetrads", resolution=GL, parity=S, units="count",
            description=f"{w}: tetrads in the modelled core (2 or 3)")
        for tag, cname in CONDITIONS:
            for n, u, d in _PER_COND:
                reg(name=f"M_g4{X}_{n}_{tag}", resolution=GL, parity=S, units=u, condition_dependent=True,
                    description=f"{w}, {cname}: {d} (mean over models and MD frames)")
            for n in _SPREAD:
                reg(name=f"M_g4{X}_{n}_sd_{tag}", resolution=GL, parity=S, units="as the channel",
                    condition_dependent=True, description=f"{w}, {cname}: spread of {n} across models")
            for n in _MDSPREAD:
                reg(name=f"M_g4{X}_{n}_mdsd_{tag}", resolution=GL, parity=S, units="as the channel",
                    condition_dependent=True, description=f"{w}, {cname}: MD fluctuation of {n} (mean within-model SD)")
            reg(name=f"M_g4{X}_res_phi_d1_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
                description=f"{w}, {cname}: nucleotide potential 1 A outside the surface")
            reg(name=f"M_g4{X}_res_phi_d3_{tag}", resolution=PB, parity=S, units="kT/e", condition_dependent=True,
                description=f"{w}, {cname}: nucleotide potential 3 A outside the surface")
        for n, u, d in _GLOBAL:
            reg(name=f"M_g4{X}_{n}", resolution=GL, parity=S, units=u, description=f"{w}: {d}")
        reg(name=f"M_g4{X}_in_model", resolution=PB, parity=S, units="indicator",
            description=f"1 for base pairs whose quadruplex-strand nucleotide is in the {topo} models")
        reg(name=f"M_g4{X}_res_rmsf", resolution=PB, parity=S, units="A",
            description=f"{w}: nucleotide heavy-atom RMSF over the MD frames")


_declare()

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        import g4_multi as GM
        _LIB = GM.Library(TEMPLATE_DIR)
    return _LIB


def _empty(n: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    prof = {k: np.full(n, np.nan) for k in channel_names_per_bp()}
    glob = {k: float("nan") for k in channel_names_global()}
    for X, _ in CLASSES:
        prof[f"M_g4{X}_in_model"] = np.zeros(n); glob[f"M_g4{X}_modelled"] = 0.0
    return prof, glob


def _heavy_xyz_by_residue(pdb: str):
    import g4_templates as T
    import g4_build as B
    atoms, _ = T.read_pdb(pdb)
    res = T.residues(atoms)
    return {k[1]: B._heavy(r["atoms"]) for k, r in res.items()}


def _pairwise_rmsd(models: List[str]) -> float:
    import g4_build as B
    xs = []
    for p in models:
        r = _heavy_xyz_by_residue(p)
        xs.append({(i, n): x for i, at in r.items() for n, x in at.items()})
    vals = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            keys = sorted(set(xs[i]) & set(xs[j]))
            P = np.array([xs[i][k] for k in keys]); Q = np.array([xs[j][k] for k in keys])
            vals.append(B.kabsch(P, Q)[2])
    return float(np.mean(vals)) if vals else float("nan")


def build_models(seq: str, work: str, n_models: int = N_MODELS, md: bool = True,
                 md_settings: Optional[dict] = None, platform: Optional[str] = None,
                 minimise: bool = True) -> dict:
    """Stage 1 (structures): select, plan per class, assemble, minimise, MD. Paths under `work`."""
    import g4_select as S
    import g4_build as B
    import g4_multi as GM
    seq = seq.upper()
    sel = S.select_v3(seq)
    out = {"sequence": seq, "reason": None if sel is None else sel.reason, "classes": {}}
    if sel is None or sel.reason != "ok":
        return out
    built = sorted(p for col in sel.core_positions for p in col) + sorted(p for lp in sel.loop_positions for p in lp)
    out.update(strand=sel.strand, n_tet=sel.n_tet, loops=sel.loops, order=sorted(built))
    core_idx, loop_idx = GM.core_and_loop_indices(sel.loops, sel.n_tet)
    out.update(core_idx=core_idx, loop_idx=loop_idx)
    mds = dict(MD_SETTINGS, **(md_settings or {}))
    lib = library()
    for X, topo in CLASSES:
        models = []
        for k, (core, donors, score) in enumerate(GM.plan_models(sel.loops, topo, sel.n_tet, lib, n_models)):
            mw = os.path.join(work, f"{X}{k}"); os.makedirs(mw, exist_ok=True)
            rec = {"core": os.path.basename(core.path)[:-4], "fold": core.fold, "score": float(score),
                   "donors": [None if d is None else (d[3], int(d[4]), round(float(d[2]), 3)) for d in donors]}
            rec["built_loops"] = [i for i, d in enumerate(donors) if d is None]
            try:
                if rec["built_loops"]:
                    # loops without a solved donor are built from sequence on the placed core and donor loops
                    import loop_builder as LB
                    known, missing = _assemble_with_gaps(core, sel.loops, donors)
                    gp = LB.write_gapped_pdb(known, missing, os.path.join(mw, "gap.pdb"))
                    known_num, m_ = [], 0
                    for c in range(4):
                        known_num += list(range(m_ + 1, m_ + 1 + sel.n_tet)); m_ += sel.n_tet
                        if c < 3:
                            if donors[c] is not None:
                                known_num += list(range(m_ + 1, m_ + 1 + len(sel.loops[c])))
                            m_ += len(sel.loops[c])
                    raw = os.path.join(mw, "raw.pdb")
                    LB.fill_and_relax(gp, raw, known_num, seed=k, platform=platform)
                else:
                    resl = B.assemble_model(core, sel.loops, donors)
                    raw = B.write_model_pdb(resl, os.path.join(mw, "raw.pdb"))
                if not minimise:                        # the caller minimises (g4_build.minimise_batch)
                    rec.update(raw=raw, status="raw")
                    models.append(rec)
                    continue
                rep = B.minimise_model(raw, os.path.join(mw, "min.pdb"), core_idx, platform=platform)
                rec.update(min=os.path.join(mw, "min.pdb"), energy_per_nt=rep["energy_final_kJ_mol"] / len(built))
                if md:
                    r = B.md_ensemble(rec["min"], os.path.join(mw, "md"), core_idx, seed=k, platform=platform,
                                      **{x: mds[x] for x in ("n_frames", "heat_ps", "equil_ps", "production_ps",
                                                             "k_core", "salt_M", "temperature_K")})
                    rec.update(frames=r["frames"], md_rmsd=r["rmsd_to_minimised_A"])
                rec["status"] = "ok"
            except Exception as exc:                                 # noqa: BLE001
                rec["status"] = f"failed: {type(exc).__name__}: {str(exc)[:300]}"
            models.append(rec)
        out["classes"][X] = models
    return out


def _assemble_with_gaps(core, loops_seq, donors):
    """`g4_build.assemble_model` with loops whose donor is None left missing: (known residues in
    sequence order, {index of the known residue before which a missing loop goes: resnames})."""
    import g4_build as B
    known, missing = [], {}
    for c in range(4):
        for k in core.columns[c]:
            known.append(("DG", B._heavy(core.res[k]["atoms"])))
        if c < 3:
            if donors[c] is None:
                missing[len(known)] = [B.LETTER_TO_RESNAME[x] for x in loops_seq[c]]
                continue
            res, rn, _, _, _ = donors[c]
            for r, name, letter in zip(res, rn, loops_seq[c]):
                atoms = B._heavy(r)
                src = name[-1] if name[-1] in "ACGTU" else "T"
                src = "T" if src == "U" else src
                known.append((B.LETTER_TO_RESNAME[letter], B.substitute_base(atoms, src, letter) if src != letter else atoms))
    known[0] = (known[0][0], {n: x for n, x in known[0][1].items() if n not in B.PHOSPHATE_GROUP})
    return known, missing


def save_built(built: dict, path: str) -> None:
    """Stage-1 record with every PDB inlined (one gzipped JSON per fragment)."""
    b = json.loads(json.dumps(built))
    for X, models in b.get("classes", {}).items():
        for m in models:
            if m.get("min"):
                m["min_pdb"] = open(m.pop("min")).read()
            if m.get("frames"):
                m["frames_pdb"] = [open(f).read() for f in m.pop("frames")]
    tmp = path + ".partial"
    with gzip.open(tmp, "wt") as f:
        json.dump(b, f)
    os.replace(tmp, path)


def load_built(path: str, work: str) -> dict:
    with gzip.open(path, "rt") as f:
        b = json.load(f)
    for X, models in b.get("classes", {}).items():
        for k, m in enumerate(models):
            mw = os.path.join(work, f"{X}{k}"); os.makedirs(mw, exist_ok=True)
            if "min_pdb" in m:
                p = os.path.join(mw, "min.pdb"); open(p, "w").write(m.pop("min_pdb")); m["min"] = p
            if "frames_pdb" in m:
                fr = []
                for j, txt in enumerate(m.pop("frames_pdb")):
                    p = os.path.join(mw, f"md_f{j}.pdb"); open(p, "w").write(txt); fr.append(p)
                m["frames"] = fr
    return b


def _value(r: dict, n: str, c: str) -> float:
    v = r.get(_KEY.get(n, n + "_{c}").format(c=c), np.nan)
    return float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else float(v)


def solve_models(built: dict, work: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Stage 2 (electrostatics): APBS on every minimised model and MD frame; channels."""
    import g4_electrostatics as E
    import apbs_run as A
    seq = built["sequence"]; n = len(seq)
    prof, glob = _empty(n)
    if built.get("reason") != "ok":
        return prof, glob
    conds = [A.PRODUCTION_CONDITIONS_BY_NAME[c] for _, c in CONDITIONS]
    order, strand = built["order"], built["strand"]
    n_res, loop_idx = len(order), built["loop_idx"]
    fpos = [p if strand == "+" else n - 1 - p for p in order]
    for X, models in built["classes"].items():
        ok = [m for m in models if m.get("status") == "ok" and m.get("min")]
        if not ok:
            continue
        per_model = []              # (static result, [frame results], frame paths)
        for k, m in enumerate(ok):
            sw = os.path.join(work, f"s_{X}{k}"); os.makedirs(sw, exist_ok=True)
            try:
                r0 = E.solve(m["min"], sw, conds, n_res, loop_idx)
            except Exception:                                        # noqa: BLE001
                continue
            frs = []
            for j, f in enumerate(m.get("frames") or []):
                fw = os.path.join(sw, f"f{j}"); os.makedirs(fw, exist_ok=True)
                try:
                    # MD frames: potentials and solvation; the ion-binding PB cycle (two more solves
                    # per condition) on the minimised model only (DECISIONS.md, cost)
                    frs.append((f, E.solve(f, fw, conds, n_res, loop_idx, tetrads=r0["tetrads"], ion_binding=False)))
                except Exception:                                    # noqa: BLE001
                    pass
            per_model.append((m, r0, frs))
        if not per_model:
            continue
        glob[f"M_g4{X}_modelled"] = 1.0
        glob[f"M_g4{X}_n_tetrads"] = float(built["n_tet"])
        ens = lambda pm: [r for _, r in pm[2]] or [pm[1]]                       # noqa: E731
        ION_Q = ("ion_binding", "ion_desolvation")
        for tag, c in CONDITIONS:
            for nme, _, _ in _PER_COND:
                per = [np.array([_value(r, nme, c) for r in ([pm[1]] if nme in ION_Q else ens(pm))]) for pm in per_model]
                allv = np.concatenate(per)
                glob[f"M_g4{X}_{nme}_{tag}"] = float(np.nanmean(allv)) if np.isfinite(allv).any() else float("nan")
                if nme in _SPREAD:
                    means = np.array([np.nanmean(v) for v in per])
                    glob[f"M_g4{X}_{nme}_sd_{tag}"] = float(np.nanstd(means)) if len(means) > 1 else float("nan")
                if nme in _MDSPREAD:
                    sds = [np.nanstd(v, ddof=1) for v in per if len(v) > 1]
                    glob[f"M_g4{X}_{nme}_mdsd_{tag}"] = float(np.nanmean(sds)) if sds else float("nan")
            for d in (1, 3):
                rv = np.nanmean(np.stack([np.asarray(r[f"phi_residue_d{d}_{c}"], float)
                                          for pm in per_model for r in ens(pm)]), axis=0)
                track = np.full(n, np.nan)
                for i, p in enumerate(fpos):
                    track[p] = rv[i]
                prof[f"M_g4{X}_res_phi_d{d}_{tag}"] = track
        allr = [r for pm in per_model for r in ens(pm)]
        mins = [pm[1] for pm in per_model]
        kb = np.array([_value(r, "ion_binding", "KCl_150mM") for r in mins])
        nb = np.array([_value(r, "ion_binding", "NaCl_150mM") for r in mins])
        glob[f"M_g4{X}_ion_coulomb"] = float(np.nanmean([_value(r, "ion_coulomb_kJ_mol", "KCl_150mM") for r in mins]))
        glob[f"M_g4{X}_selectivity_KNa"] = float(np.nanmean(kb - nb))
        glob[f"M_g4{X}_dipole"] = float(np.nanmean([r["dipole_D"] for r in allr]))
        glob[f"M_g4{X}_rg"] = float(np.nanmean([r["rg_A"] for r in allr]))
        glob[f"M_g4{X}_groove_depth"] = float(np.nanmean([r.get("groove_depth_A", np.nan) for r in allr]))
        glob[f"M_g4{X}_model_spread"] = _pairwise_rmsd([pm[0]["min"] for pm in per_model]) if len(per_model) > 1 else float("nan")
        glob[f"M_g4{X}_energy_per_nt"] = float(np.mean([pm[0]["energy_per_nt"] for pm in per_model]))
        glob[f"M_g4{X}_fit"] = float(np.mean([pm[0]["score"] for pm in per_model]))
        mdr = [x for pm in per_model for x in (pm[0].get("md_rmsd") or [])]
        glob[f"M_g4{X}_md_rmsd"] = float(np.mean(mdr)) if mdr else float("nan")
        # per-nucleotide heavy-atom RMSF over each model's frames (core held, so no superposition)
        rmsf_models = []
        for m, _, frs in per_model:
            if len(frs) < 2:
                continue
            fr = [_heavy_xyz_by_residue(f) for f, _ in frs]
            v = np.full(n_res, np.nan)
            for i in range(1, n_res + 1):
                names = sorted(set.intersection(*[set(x.get(i, {})) for x in fr])) if all(i in x for x in fr) else []
                if names:
                    X_ = np.array([[x[i][nm] for nm in names] for x in fr])
                    v[i - 1] = float(np.sqrt(np.mean(np.sum((X_ - X_.mean(0)) ** 2, -1))))
            rmsf_models.append(v)
        if rmsf_models:
            rm = np.nanmean(np.stack(rmsf_models), axis=0)
            track = np.full(n, np.nan)
            for i, p in enumerate(fpos):
                track[p] = rm[i]
            prof[f"M_g4{X}_res_rmsf"] = track
            li = [i - 1 for i in loop_idx]
            glob[f"M_g4{X}_loop_rmsf"] = float(np.nanmean(rm[li])) if li else float("nan")
        inm = np.zeros(n)
        for p in fpos:
            inm[p] = 1.0
        prof[f"M_g4{X}_in_model"] = inm
    return prof, glob


def fold_channels(seq: str, n_models: int = N_MODELS, workroot: Optional[str] = None, md: bool = True,
                  md_settings: Optional[dict] = None, platform: Optional[str] = None
                  ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Family M profiles and whole-sequence values for one fragment (both stages in one process)."""
    work = tempfile.mkdtemp(prefix="famM_", dir=workroot or os.environ.get("SLURM_TMPDIR") or None)
    try:
        built = build_models(seq, work, n_models, md, md_settings, platform)
        return solve_models(built, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
