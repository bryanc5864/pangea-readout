#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Channels from the per-context libraries, mapped onto every fragment by its tetramer contexts
(the convention of `qc_channels`: the step (i, i+1) reads the 4-mer starting at i-1, through its
reverse complement when that is the stored form, with the two pairs exchanged; a base pair's value
averages its two contexts, as the second pair of step i-1 and the first of step i):

* explicit-solvent MD hydration (`md_hydration.py`; the central step of each 12-mer):
  Family G ("Gmd") groove hydration occupancy, residence time, translational -TS, and per step the
  spine continuity and water-bridge probability; Family K ("Kmd") the same for A-RNA and the hybrid
  and the 2'-OH observables (occupancy, residence, O2'...O4'(i+1) hydrogen bond, C3'-endo fraction,
  O2' RMSF) per strand;
* docking (`docking_library.py`, `dock_run.py`): Family G ("Gdk") minor-groove-binder (DAPI) and
  intercalator (proflavine) Vina affinities per step;
* base-pair trimers (`run_psi4.py --trimer`): Family B ("Bt") -- the trimer centred on pair i is
  the t5 trimer of the context at i-1 and the t3 trimer of the context at i-2 (averaged):
  three-body stacking, vertical IP, the hole fraction and HOMO weight on the centre pair.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

COMP = str.maketrans("ACGT", "TGCA")
CENTRAL = (5, 6)                   # the 12-mer's central step (GCGC + context + GCGC), 0-based pairs
SRC_MD = "explicit-solvent MD library (OL15/OL3, OPC, 0.15 M KCl, 10 ns per tetramer context; md_hydration.py)"
SRC_DOCK = "AutoDock Vina 1.2.7 (Eberhardt et al. 2021) on cgNA+ 12-mers: DAPI minor groove, proflavine intercalation"
SRC_TRI = "B3LYP-D3BJ/def2-TZVP on cgNA+ base-pair trimers of every tetramer context (run_psi4.py --trimer)"

MD_BP = (("minor_occupancy", "count", "first-shell waters of the minor-groove acceptors"),
         ("major_occupancy", "count", "first-shell waters of the major-groove acceptors"),
         ("minor_residence_ps", "ps", "mean residence of a water in a minor-groove acceptor's shell"),
         ("major_residence_ps", "ps", "mean residence of a water in a major-groove acceptor's shell"),
         ("minor_minusTS_kcal_mol", "kcal/mol", "-T S_trans of the minor-groove water (first-order IST)"),
         ("major_minusTS_kcal_mol", "kcal/mol", "-T S_trans of the major-groove water (first-order IST)"))
MD_STEP = (("spine_continuity", "probability", "minor-groove spine: a water bridging the two pairs across strands"),
           ("water_bridge_p", "probability", "a water bridging acceptors of two nucleotides of the step"))
MD_O2P = (("o2p_occupancy", "count", "first-shell waters of O2'"), ("o2p_residence_ps", "ps", "residence at O2'"),
          ("o2p_o4p_hbond_p", "probability", "O2'...O4'(i+1) hydrogen bond"),
          ("c3endo_frac", "fraction", "C3'-endo sugar pucker"), ("o2p_rmsf_A", "A", "O2' RMSF"))
TRI = (("E3", "kcal/mol", "three-body stacking of the trimer centred on this pair"),
       ("vip", "eV", "vertical ionisation potential of that trimer"),
       ("hole_centre", "fraction", "fraction of that trimer's hole on this pair"),
       ("homo_centre", "fraction", "HOMO weight on this pair"))
CLASSES_K = (("r", "rna", "A-RNA duplex"), ("h", "hyb", "DNA-RNA hybrid"))


def _declare() -> None:
    PB, GL, S, ST = Resolution.PER_BP, Resolution.GLOBAL, Parity.SYMMETRIC, Parity.STRANDED
    for fam, prefix, what in (("Gmd", "Gmd", "B-DNA"),) + tuple(("Kmd", f"Kmd_{t}", d) for t, _, d in CLASSES_K):
        for n, u, d in MD_BP:
            REGISTRY.register(ChannelSpec(name=f"{prefix}_{n}", family=fam, resolution=PB, parity=S, units=u,
                                          source=SRC_MD, description=f"{what}: {d}"))
        for n, u, d in MD_STEP:
            REGISTRY.register(ChannelSpec(name=f"{prefix}_{n}", family=fam, resolution=PB, parity=S, units=u,
                                          rc_shift=-1, source=SRC_MD, description=f"{what}, step (i, i+1): {d}"))
    for t, _, d in CLASSES_K:
        for n, u, dd in MD_O2P:
            for s in ("W", "C"):
                REGISTRY.register(ChannelSpec(name=f"Kmd_{t}_{n}_{s}", family="Kmd", resolution=PB, parity=ST, units=u,
                                              source=SRC_MD, description=f"{d}, {'Watson' if s == 'W' else 'Crick'} RNA strand: {dd}"))
    for n, u, d in (("mgb_vina", "kcal/mol", "DAPI minor-groove Vina affinity at step (i, i+1)"),
                    ("intercalator_vina", "kcal/mol", "proflavine intercalation Vina affinity at step (i, i+1)")):
        REGISTRY.register(ChannelSpec(name=f"Gdk_{n}", family="Gdk", resolution=PB, parity=S, units=u, rc_shift=-1,
                                      source=SRC_DOCK, description=d))
    for n, u, d in TRI:
        REGISTRY.register(ChannelSpec(name=f"Bt_{n}", family="Bt", resolution=PB, parity=S, units=u, source=SRC_TRI,
                                      description=d))


_declare()


def _rc(s):
    return s.translate(COMP)[::-1]


def _contexts(seq):
    """(i, context, flipped, stored key) for every step (i, i+1) with both flanks."""
    for i in range(1, len(seq) - 2):
        yield i, seq[i - 1:i + 3]


def load_md(folder: str, cls: str) -> Dict[str, dict]:
    out = {}
    for p in glob.glob(os.path.join(folder, f"*_{cls}.npz")):
        z = np.load(p)
        out[str(z["context"])] = {k: np.asarray(z[k]) for k in z.files}
    return out


def md_channels(seq: str, lib: Dict[str, dict], prefix: str, with_o2p: bool = False) -> Dict[str, np.ndarray]:
    seq = seq.upper(); n = len(seq)
    acc = {f"{prefix}_{k}": [[] for _ in range(n)] for k, _, _ in MD_BP}
    prof = {f"{prefix}_{k}": np.full(n, np.nan) for k, _, _ in MD_STEP}
    o2 = {f"{prefix}_{k}_{s}": [[] for _ in range(n)] for k, _, _ in MD_O2P for s in "WC"} if with_o2p else {}
    for i, ctx in _contexts(seq):
        if ctx in lib:
            rec, flipped = lib[ctx], False
        elif _rc(ctx) in lib:
            rec, flipped = lib[_rc(ctx)], True
        else:
            continue
        a, b = CENTRAL
        first, second = (i + 1, i) if flipped else (i, i + 1)
        for k, _, _ in MD_BP:
            acc[f"{prefix}_{k}"][first].append(float(rec[k][a])); acc[f"{prefix}_{k}"][second].append(float(rec[k][b]))
        for k, _, _ in MD_STEP:
            prof[f"{prefix}_{k}"][i] = float(rec[k][a])
        if with_o2p:
            # rows: strand A (Watson of the stored 12-mer), strand B (Crick, 5'->3'); B residue for pair j is 12 - j
            nb = 12
            for k, _, _ in MD_O2P:
                M = rec[k]
                for pos, pair in ((first, a), (second, b)):
                    w, c = M[0, pair], M[1, nb - 1 - pair] if M.shape[0] > 1 else np.nan
                    if flipped:
                        w, c = c, w
                    o2[f"{prefix}_{k}_W"][pos].append(float(w)); o2[f"{prefix}_{k}_C"][pos].append(float(c))
    for k, v in list(acc.items()) + list(o2.items()):
        prof[k] = np.array([np.nanmean(x) if x and np.isfinite(x).any() else np.nan for x in v])
    return prof


def load_dock(folder: str) -> Dict[str, dict]:
    return {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(folder, "*.json"))}


def dock_channels(seq: str, lib: Dict[str, dict]) -> Dict[str, np.ndarray]:
    seq = seq.upper(); n = len(seq)
    prof = {"Gdk_mgb_vina": np.full(n, np.nan), "Gdk_intercalator_vina": np.full(n, np.nan)}
    for i, ctx in _contexts(seq):
        rec = lib.get(ctx) or lib.get(_rc(ctx))
        if rec:
            prof["Gdk_mgb_vina"][i] = rec.get("minor_vina_kcal_mol", np.nan)
            prof["Gdk_intercalator_vina"][i] = rec.get("intercalation_vina_kcal_mol", np.nan)
    return prof


def load_trimers(folder: str) -> Dict[str, dict]:
    out = {}
    for p in glob.glob(os.path.join(folder, "shard_*.json")):
        for r in json.load(open(p))["results"]:
            if r.get("status") == "ok":
                out[r["context"]] = r
    return out


def trimer_channels(seq: str, lib: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """Per pair j: the trimer (j-1, j, j+1) as t5 of the context at j-1 and t3 of the context at j-2."""
    seq = seq.upper(); n = len(seq)
    acc = {f"Bt_{k}": [[] for _ in range(n)] for k, _, _ in TRI}
    for j in range(1, n - 1):
        for start, tag in ((j - 1, "t5"), (j - 2, "t3")):
            if start < 0 or start + 4 > n:
                continue
            ctx = seq[start:start + 4]
            # t5 = pairs 0,1,2 and t3 = pairs 1,2,3 of the context; under reverse complement t5 <-> t3
            rec = lib.get(f"{ctx}_{tag}")
            flipped = False
            if rec is None:
                rec = lib.get(f"{_rc(ctx)}_{'t3' if tag == 't5' else 't5'}"); flipped = True
            if rec is None:
                continue
            acc["Bt_E3"][j].append(rec["tri_E3_kcal_mol"]); acc["Bt_vip"][j].append(rec["tri_vip_eV"])
            acc["Bt_hole_centre"][j].append(rec["tri_hole_B"]); acc["Bt_homo_centre"][j].append(rec["tri_homo_wB"])
    return {k: np.array([np.mean(v) if v else np.nan for v in vals]) for k, vals in acc.items()}
