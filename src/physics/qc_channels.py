#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family B quantum-chemistry channels from the tetramer tier (registered as family "Bq").

The QC tiers (`src/qc/run_psi4.py`) computed, for all 136 reverse-complement-unique
four-base-pair contexts built on cgNA+ geometry: counterpoise B3LYP-D3BJ/def2-TZVP stacking
of the central step and the two base pairs' H-bond energies, SAPT0/jun-cc-pVDZ decompositions
(electrostatics, exchange, induction, dispersion) of both, the dimer frontier orbitals and
dipole, and vertical IP/EA (ground state) -- at the cgNA+ ground state and +-1 sigma along the
central step's softest mode (PLAN.md: "at 3 geometries"). Here every base-pair step of a
fragment takes its own context's values (the step between i and i+1 reads the 4-mer starting
at i-1, through its reverse complement if that is the stored form, with the two base pairs
exchanged), so the channels are an exact function of sequence and cost nothing to compute.
Step quantities sit at index i for step (i, i+1) (rc_shift -1, NaN at the last index and at
the first and last step, which lack a flanking base); a base pair's H-bond values average its
two contexts (as the second pair of step i-1 and the first of step i).

Family code "Bq" so side shards can add these to corpora whose Family B channels already
exist; reports fold it into Family B.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "Bq"
GEOMS = (("gs", "tetramer"), ("sp", "tetramer_soft_plus"), ("sm", "tetramer_soft_minus"))
SAPT = ("total", "elst", "exch", "ind", "disp")
TABLE = os.environ.get("PANGEA_QC_TABLE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "qc", "qc_tetramer_table.json"))
SRC = ("Psi4 1.11: B3LYP-D3BJ/def2-TZVP (counterpoise), SAPT0/jun-cc-pVDZ, on cgNA+ tetramer "
       "geometries (ground state and +-1 sigma along the softest mode)")
COMP = str.maketrans("ACGT", "TGCA")

# Stage B (ground state): strand-specific quantities enter as purine/pyrimidine (per base pair)
# or as max/min over the two strands (per step), so every channel is reverse-complement
# symmetric without a strand tie.
GROOVE_KEYS = ("maj_pur_N7", "maj_pur_6", "maj_pyr_4", "min_pur_N3", "min_pyr_O2", "min_G_N2")
STEP = ([f"Bq_stack_{g}" for g, _ in GEOMS] +
        [f"Bq_stack_sapt_{c}_{g}" for g, _ in GEOMS for c in SAPT] +
        ["Bq_vip", "Bq_vea"] +
        [f"Bq_{q}_{g}" for g, _ in GEOMS for q in ("homo", "lumo", "gap", "dipole")] +
        [f"Bq_t_{w}_{k}_{m}" for w in ("hole", "elec") for k in ("intra", "inter") for m in ("max", "min")] +
        ["Bq_polar_iso", "Bq_polar_aniso", "Bq_polar_axial"])
# per base pair: channel name -> table key template ({bp} = bp1 / bp2)
BP_KEYS = ([(f"Bq_hb_{g}", f"hb_{{bp}}_{g}") for g, _ in GEOMS] +
           [(f"Bq_hb_sapt_{c}_{g}", f"hb_{{bp}}_sapt_{c}_{g}") for g, _ in GEOMS for c in SAPT] +
           [(f"Bq_site_{w}_{b}", f"site_{w}_{b}_{{bp}}") for w in ("hole", "elec") for b in ("pur", "pyr")] +
           [(f"Bq_t_{w}_pair", f"t_{w}_pair_{{bp}}") for w in ("hole", "elec")] +
           [("Bq_mbis_net_pur", "mbis_net_pur_{bp}")] +
           [(f"Bq_mep_{k}", f"mep_{k}_{{bp}}") for k in GROOVE_KEYS] +
           [(f"Bq_mbis_{k}", f"mbis_{k}_{{bp}}") for k in GROOVE_KEYS])
BP = [n for n, _ in BP_KEYS]
GLOBAL = ("Bq_mean_stack_gs", "Bq_mean_stack_disp_gs", "Bq_mean_stack_elst_gs",
          "Bq_mean_hb_gs", "Bq_mean_vip", "Bq_min_vip", "Bq_mean_vea", "Bq_max_vea",
          "Bq_mean_gap_gs", "Bq_min_gap_gs", "Bq_mean_stack_softness",
          "Bq_mean_t_hole_intra_max", "Bq_max_t_hole_intra_max", "Bq_max_site_hole_pur",
          "Bq_mean_polar_iso")
UNITS = {"stack": "kcal/mol", "hb": "kcal/mol", "vip": "eV", "vea": "eV", "homo": "eV",
         "lumo": "eV", "gap": "eV", "dipole": "debye", "softness": "kcal/mol", "t_": "eV",
         "site": "eV", "mep": "kcal/mol/e", "mbis": "e", "polar": "bohr^3"}


def _units(name: str) -> str:
    for k, u in UNITS.items():
        if f"_{k}" in name:
            return u
    return "kcal/mol"


def _declare() -> None:
    for n in STEP:
        REGISTRY.register(ChannelSpec(name=n, family=FAMILY, resolution=Resolution.PER_BP,
                                      parity=Parity.SYMMETRIC, units=_units(n), rc_shift=-1,
                                      description=f"quantum chemistry of this base-pair step: {n[3:]}",
                                      source=SRC))
    for n in BP:
        REGISTRY.register(ChannelSpec(name=n, family=FAMILY, resolution=Resolution.PER_BP,
                                      parity=Parity.SYMMETRIC, units=_units(n),
                                      description=f"quantum chemistry of this base pair: {n[3:]}",
                                      source=SRC))
    for n in GLOBAL:
        REGISTRY.register(ChannelSpec(name=n, family=FAMILY, resolution=Resolution.GLOBAL,
                                      parity=Parity.SYMMETRIC, units=_units(n),
                                      description=f"quantum chemistry, fragment summary: {n[3:]}",
                                      source=SRC))


_declare()
_TAB: Optional[Dict[str, Dict[str, float]]] = None


def load_table(path: str = TABLE) -> Optional[Dict[str, Dict[str, float]]]:
    global _TAB
    if _TAB is None and os.path.exists(path):
        _TAB = json.load(open(path))["contexts"]
    return _TAB


def _lookup(tab, ctx: str) -> Tuple[Optional[Dict[str, float]], bool]:
    """(record, flipped): flipped when the stored context is the reverse complement."""
    if ctx in tab:
        return tab[ctx], False
    rc = ctx.translate(COMP)[::-1]
    if rc in tab:
        return tab[rc], True
    return None, False


def qc_channels(seq: str, table: Optional[Dict[str, Dict[str, float]]] = None
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    seq = seq.upper()
    n = len(seq)
    tab = table if table is not None else load_table()
    prof = {k: np.full(n, np.nan) for k in STEP + BP}
    glob = {k: float("nan") for k in GLOBAL}
    if not tab:
        return prof, glob
    hb = {k: [[] for _ in range(n)] for k in BP}
    for i in range(1, n - 2):                       # step (i, i+1) with flanks i-1 and i+2
        rec, flipped = _lookup(tab, seq[i - 1:i + 3])
        if rec is None:
            continue
        for k in STEP:
            v = rec.get(k[3:])
            if v is not None:
                prof[k][i] = v
        # base pairs: stored bp1 is the context's second base pair; flipped swaps them. In a
        # self-complementary context the two pairs are equivalent by symmetry, so both take
        # their mean -- which also keeps the reverse-complement identity exact.
        first, second = (i + 1, i) if flipped else (i, i + 1)
        ctx = seq[i - 1:i + 3]
        pal = ctx == ctx.translate(COMP)[::-1]
        for k, tmpl in BP_KEYS:
            v1 = rec.get(tmpl.format(bp="bp1")); v2 = rec.get(tmpl.format(bp="bp2"))
            if pal and v1 is not None and v2 is not None:
                v1 = v2 = 0.5 * (v1 + v2)
            for pos, v in ((first, v1), (second, v2)):
                if v is not None:
                    hb[k][pos].append(v)
    for k in BP:
        prof[k] = np.array([np.mean(v) if v else np.nan for v in hb[k]])
    with np.errstate(all="ignore"):
        f = lambda a, fn: float(fn(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")  # noqa: E731
        glob["Bq_mean_stack_gs"] = f(prof["Bq_stack_gs"], np.mean)
        glob["Bq_mean_stack_disp_gs"] = f(prof["Bq_stack_sapt_disp_gs"], np.mean)
        glob["Bq_mean_stack_elst_gs"] = f(prof["Bq_stack_sapt_elst_gs"], np.mean)
        glob["Bq_mean_hb_gs"] = f(prof["Bq_hb_gs"], np.mean)
        glob["Bq_mean_vip"] = f(prof["Bq_vip"], np.mean)
        glob["Bq_min_vip"] = f(prof["Bq_vip"], np.min)
        glob["Bq_mean_vea"] = f(prof["Bq_vea"], np.mean)
        glob["Bq_max_vea"] = f(prof["Bq_vea"], np.max)
        glob["Bq_mean_gap_gs"] = f(prof["Bq_gap_gs"], np.mean)
        glob["Bq_min_gap_gs"] = f(prof["Bq_gap_gs"], np.min)
        soft = 0.5 * (prof["Bq_stack_sp"] + prof["Bq_stack_sm"]) - prof["Bq_stack_gs"]
        glob["Bq_mean_stack_softness"] = f(soft, np.mean)
        glob["Bq_mean_t_hole_intra_max"] = f(prof["Bq_t_hole_intra_max"], np.mean)
        glob["Bq_max_t_hole_intra_max"] = f(prof["Bq_t_hole_intra_max"], np.max)
        glob["Bq_max_site_hole_pur"] = f(prof["Bq_site_hole_pur"], np.max)
        glob["Bq_mean_polar_iso"] = f(prof["Bq_polar_iso"], np.mean)
    return prof, glob
