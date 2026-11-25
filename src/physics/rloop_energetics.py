#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""R-loop initiation and elongation free energies of a fragment (family code "Fr", side shards).

PLAN.md Family F lists "R-loop initiation and elongation free energy"; Family F's R-loop channels
are the RIZ/REZ pattern scores of the quantitative R-loop-forming-sequence model, so this adds the
free energies themselves from the statistical-mechanical R-loop model of Stolz et al. (PNAS
116:6260, 2019; parameters and B table as in `longrange`). Without superhelical stress (the
model's "unconstrained" mode; the stressed, domain-scale R-loop is Family W's) an R-loop of m >= 2
bp starting at n costs G(n, m) = a + sum of B over its m - 1 steps, a = 10.5 kcal/mol the junction
(initiation) energy and B the RNA:DNA minus DNA:DNA step free energy (elongation). Per strand taken
as the non-template strand (W: the given strand; C: its partner):

* Fr_B: B per step (the elongation free energy profile), W at the step's 5' base pair, C at its 3';
* Fr_elong_dG: mean B over the fragment, kcal/mol per bp;
* Fr_init_dG: the cheapest R-loop in the fragment, min G(n, m) (junction plus best hybrid segment);
* Fr_ens_dG: the R-loop ensemble free energy -RT ln sum exp(-G / RT) over every R-loop, 310 K.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution
import longrange as W

FAMILY = "Fr"
COMP = str.maketrans("ACGT", "TGCA")


def _strand(nontemplate: str) -> Tuple[np.ndarray, float, float, float]:
    P = W.RLOOP; RT = P["R"] * P["T"]
    B = W.rloop_steps(nontemplate); n = len(nontemplate)
    cB = np.concatenate([[0.0], np.cumsum(B)])
    best, terms = math.inf, []
    for s0 in range(0, n - P["minlength"] + 1):
        mm = np.arange(P["minlength"], n - s0 + 1)
        g = P["a"] + (cB[s0 + mm - 1] - cB[s0])
        best = min(best, float(g.min())); terms.append(g)
    g = np.concatenate(terms)
    ens = float(best - RT * math.log(np.exp(-(g - best) / RT).sum()))
    return B, float(B.mean()), best, ens


def rloop_fragment_channels(seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    s = seq.upper(); n = len(s)
    pb: Dict[str, np.ndarray] = {}; gl: Dict[str, float] = {}
    for tag, strand in (("W", s), ("C", s.translate(COMP)[::-1])):
        B, el, ini, ens = _strand(strand)
        t = np.full(n, np.nan)
        if tag == "W":
            t[:n - 1] = B
        else:
            t[1:] = B[::-1]
        pb[f"Fr_B_{tag}"] = t
        gl[f"Fr_elong_dG_{tag}"] = el; gl[f"Fr_init_dG_{tag}"] = ini; gl[f"Fr_ens_dG_{tag}"] = ens
    return pb, gl


def _declare() -> None:
    for t, which in (("W", "the given strand"), ("C", "the partner strand")):
        REGISTRY.register(ChannelSpec(name=f"Fr_B_{t}", family=FAMILY, resolution=Resolution.PER_BP, parity=Parity.STRANDED,
                                      units="kcal/mol", source=W.SRC_RLOOP,
                                      description=f"RNA:DNA minus DNA:DNA step free energy, {which} non-template (elongation profile)"))
        for k, u, d in (("elong_dG", "kcal/mol per bp", "mean elongation free energy over the fragment"),
                        ("init_dG", "kcal/mol", "free energy of the cheapest R-loop: junction plus best hybrid segment, no stress"),
                        ("ens_dG", "kcal/mol", "R-loop ensemble free energy over every R-loop in the fragment, no stress, 310 K")):
            REGISTRY.register(ChannelSpec(name=f"Fr_{k}_{t}", family=FAMILY, resolution=Resolution.GLOBAL, parity=Parity.STRANDED,
                                          units=u, source=W.SRC_RLOOP, description=f"{d}; {which} non-template"))


_declare()
