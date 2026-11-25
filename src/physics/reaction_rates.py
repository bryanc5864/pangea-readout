#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family G hydrolytic damage rates (family code "Gr"; side shards from the main shards' Family E
opening probability), 37 C, pH 7.4:

* depurination: k_ds = 4e-9 s^-1 at 70 C with Ea = 31 kcal/mol (Lindahl & Nyberg, Biochemistry
  11:3610, 1972), i.e. 3.2e-11 s^-1 at 37 C; denatured DNA ~4x faster and otherwise sequence
  independent, so per purine k = k_ds (1 - p_open) + 4 k_ds p_open;
* cytosine deamination: 7e-13 s^-1 double-stranded and 1e-10 s^-1 single-stranded (Frederico,
  Kunkel & Shaw, Biochemistry 29:2532, 1990): per cytosine k = k_ds (1 - p_open) + k_ss p_open;

with p_open the base pair's opening probability at 37 C (Family E, E_open_prob_T37). Channels are
log10 k, per strand (W: the given strand's base at the position; C: its partner), NaN at bases the
reaction does not touch.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

R_KCAL = 1.98720425864083e-3
K_DEP_70 = 4e-9
EA_DEP = 31.0
K_DEP_37 = K_DEP_70 * math.exp(-EA_DEP / R_KCAL * (1.0 / 310.15 - 1.0 / 343.15))
SS_FACTOR_DEP = 4.0
K_DEAM_DS, K_DEAM_SS = 7e-13, 1e-10
SRC_DEP = "Lindahl & Nyberg, Biochemistry 11:3610 (1972); opening probability from Family E"
SRC_DEAM = "Frederico, Kunkel & Shaw, Biochemistry 29:2532 (1990); opening probability from Family E"
COMP = str.maketrans("ACGT", "TGCA")


def _declare():
    for s, which in (("W", "the given strand"), ("C", "the partner strand")):
        REGISTRY.register(ChannelSpec(name=f"Gr_depurination_log10k_{s}", family="Gr", resolution=Resolution.PER_BP,
                                      parity=Parity.STRANDED, units="log10 s^-1", source=SRC_DEP,
                                      description=f"depurination rate of the purine of {which}, 37 C, pH 7.4"))
        REGISTRY.register(ChannelSpec(name=f"Gr_deamination_log10k_{s}", family="Gr", resolution=Resolution.PER_BP,
                                      parity=Parity.STRANDED, units="log10 s^-1", source=SRC_DEAM,
                                      description=f"cytosine deamination rate on {which}, 37 C, pH 7.4"))


_declare()
NAMES = [f"Gr_{q}_log10k_{s}" for q in ("depurination", "deamination") for s in ("W", "C")]


def rate_channels(seq: str, p_open: np.ndarray) -> Dict[str, np.ndarray]:
    seq = seq.upper()
    b_w = np.array(list(seq)); b_c = np.array(list(seq.translate(COMP)))
    p = np.clip(np.asarray(p_open, float), 0.0, 1.0)
    k_dep = K_DEP_37 * ((1 - p) + SS_FACTOR_DEP * p)
    k_deam = K_DEAM_DS * (1 - p) + K_DEAM_SS * p
    out = {}
    for s, b in (("W", b_w), ("C", b_c)):
        out[f"Gr_depurination_log10k_{s}"] = np.where(np.isin(b, ["A", "G"]), np.log10(k_dep), np.nan)
        out[f"Gr_deamination_log10k_{s}"] = np.where(b == "C", np.log10(k_deam), np.nan)
    return out
