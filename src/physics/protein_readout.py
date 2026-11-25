#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family I: protein readout, held apart.

PLAN.md: "I. Protein readout, held apart -- Occupancy for a transcription factor panel at
several chemical potentials, generic amino acid side chain interaction energies in each
groove, nucleosome competition terms. Ground truth: ProBound free energy models fit to
SELEX." Held apart means exactly that: these channels live in their own side shards and
enter only the labelled protein-readout variant, never the physics encoder or a headline
number, so that no physics result is carried by base readout.

Models: twelve ProBound fits from MotifCentral v1.0.0 (Rube et al., Nature Biotechnology
40:1520, 2022), one per major DNA-binding structural class (`data/probound_panel_v1.json`
holds the coefficients, fit ids and source HT-SELEX experiments). Each is a single
binding mode with mononucleotide terms: log relative affinity of a window = the sum of
its positions' coefficients, scored on both strands. Each model's top-scoring bases were
checked against the factor's known site before use (CACGTG for MAX, GGGGAATCCCC for
NFKB1, TGTTTAC for FOXA1, AGATAAG for GATA3, CCGGAAGT for GABPA, the CTCF core ...).

Channels per factor:
* per base pair: the binding free energy, in kT relative to the factor's optimal site,
  of the best site covering that base on either strand (taking the best covering site,
  not a site centre, makes the track exactly reverse-complement symmetric);
* per fragment: occupancy theta = Z / (1 + Z), Z = sum over sites of exp(mu + log K_rel),
  at three chemical potentials calibrated on a fixed corpus sample so that the median
  fragment sits at 10, 50 and 90 percent occupancy; and occupancy in competition with a
  nucleosome, Z / (1 + Z + Z_nuc) at the middle potential, with Z_nuc = exp(mu_nuc -
  dG_nuc) from the fragment's own Family D nucleosome free energy (0.15 M), mu_nuc set so
  the median fragment has Z_nuc = 1.

Not yet: the plan's "generic amino acid side chain interaction energies in each groove",
which ProBound does not provide (no sourced model; DECISIONS.md).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "I"
PANEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "probound_panel_v1.json")
OCC_QUANTILES = (10, 50, 90)
NUC_KEY = "D_nucleosome_dG_I0.15_kT"
_COMP = np.array([3, 2, 1, 0])                  # A<->T, C<->G in ACGT order
_IDX = {c: i for i, c in enumerate("ACGT")}


def _panel():
    with open(PANEL_FILE) as f:
        return json.load(f)


_P = _panel()
MODELS = {t["tf"]: np.asarray(t["mononucleotide"], float).reshape(t["size"], 4) for t in _P["tfs"]}
CAL = _P.get("calibration", {})
SRC = "ProBound fit {fid} (MotifCentral v1.0.0; Rube et al., Nat Biotechnol 40:1520, 2022)"


def _declare() -> None:
    for t in _P["tfs"]:
        tf, src = t["tf"], SRC.format(fid=t["fit_id"])
        REGISTRY.register(ChannelSpec(
            name=f"I_{tf}_ddG_kT", family=FAMILY, resolution=Resolution.PER_BP,
            parity=Parity.SYMMETRIC, units="kT", source=src,
            description=f"{tf} ({t['structural_class']}): binding free energy of the best site "
                        f"covering this base, relative to the optimal site"))
        for q in OCC_QUANTILES:
            REGISTRY.register(ChannelSpec(
                name=f"I_{tf}_occ_q{q}", family=FAMILY, resolution=Resolution.GLOBAL,
                parity=Parity.SYMMETRIC, units="probability", source=src, condition_dependent=True,
                description=f"{tf} occupancy at the chemical potential putting the median fragment at {q}%"))
        REGISTRY.register(ChannelSpec(
            name=f"I_{tf}_occ_nuc", family=FAMILY, resolution=Resolution.GLOBAL,
            parity=Parity.SYMMETRIC, units="probability", source=src,
            description=f"{tf} occupancy against nucleosome competition (Family D free energy), middle potential"))


def site_log_affinity(seq: str, B: np.ndarray) -> np.ndarray:
    """log K_rel of every window on both strands, (2, n - w + 1); 0 is the optimal site."""
    x = np.array([_IDX.get(c, -1) for c in seq.upper()])
    w = len(B)
    n = len(x) - w + 1
    best = float(B.max(1).sum())
    out = np.full((2, max(n, 0)), -np.inf)
    if n <= 0 or (x < 0).any():
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, w)          # (n, w)
    pos = np.arange(w)
    out[0] = B[pos[None, :], win].sum(1) - best
    rc_win = _COMP[win[:, ::-1]]                                   # the other strand, 5'->3'
    out[1] = B[pos[None, :], rc_win].sum(1) - best
    return out


def readout_channels(seq: str, context: Optional[Dict[str, float]] = None
                     ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}
    L = len(seq)
    dg_nuc = (context or {}).get(NUC_KEY)
    for tf, B in MODELS.items():
        la = site_log_affinity(seq, B)
        w = len(B)
        best_site = la.max(0)                                      # either strand, per start
        cover = np.full(L, -np.inf)
        for s in range(len(best_site)):
            cover[s:s + w] = np.maximum(cover[s:s + w], best_site[s])
        prof[f"I_{tf}_ddG_kT"] = np.where(np.isfinite(cover), -cover, np.nan)
        logZ0 = float(np.logaddexp.reduce(la.ravel())) if la.size else float("-inf")
        cal = CAL.get(tf, {})
        for q in OCC_QUANTILES:
            mu = cal.get(f"mu_q{q}")
            glob[f"I_{tf}_occ_q{q}"] = (float(1.0 / (1.0 + np.exp(-(mu + logZ0))))
                                        if mu is not None else float("nan"))
        mu50, mun = cal.get("mu_q50"), CAL.get("mu_nuc")
        if mu50 is not None and mun is not None and dg_nuc is not None and np.isfinite(dg_nuc):
            Z = np.exp(mu50 + logZ0); Zn = np.exp(mun - float(dg_nuc))
            glob[f"I_{tf}_occ_nuc"] = float(Z / (1.0 + Z + Zn))
        else:
            glob[f"I_{tf}_occ_nuc"] = float("nan")
    return prof, glob


_declare()
