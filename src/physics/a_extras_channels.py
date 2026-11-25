#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A extras as channels (family code "Ax", side shards): PLAN.md Family A's "electrostatic
component of base-pair opening" and "local dielectric response", from `run_family_a_extras.py`, at
each fragment's central base pair (index 30 of 61; NaN elsewhere, masked):

* Ax_open_elec_c150: dG_open,elec = dE_Coulomb (solute dielectric 2) + d(map solvation), kJ/mol,
  at 150 mM NaCl (salt-independent to <= 0.19 kJ/mol between 10 and 150 mM on the subset solved at
  both); Ax_open_dEcoul and Ax_open_dGsolv_c150 its two parts;
* Ax_s<k>_dielimage_d<1|4>: the reaction-field (image) potential of a unit probe charge 1 and 4 A
  outside the molecular surface in sector k (Family A's six sectors), kT/e.
Sectors keep Family A's parities and its own-frame convention for 4 and 5 (a single central index
maps to itself under reversal), so (Ax_s1_, Ax_s5_) and (Ax_s2_, Ax_s4_) are strand pairs like A's.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "Ax"
CENTRAL = 30
OFFSETS = (1, 4)
SRC = "APBS on the production construction (run_family_a_extras.py): opened-pair GBn2 relaxation; Poisson image potential"


def _sector_parity(s: int) -> Parity:
    return Parity.SYMMETRIC if s in (0, 3) else Parity.STRANDED


def _declare() -> None:
    PB = Resolution.PER_BP
    # 150 mM only: on the subset solved at both, 10 and 150 mM agree to <= 0.19 kJ/mol (median 0.05)
    # against a 4.6-26.9 kJ/mol spread, and the rest of the corpus is solved at 150 mM (DECISIONS)
    for t, c in (("c150", "150 mM"),):
        REGISTRY.register(ChannelSpec(name=f"Ax_open_elec_{t}", family=FAMILY, resolution=PB, parity=Parity.SYMMETRIC,
                                      units="kJ/mol", source=SRC, condition_dependent=True,
                                      description=f"electrostatic free energy of opening this base pair 60 degrees toward the major groove, {c} NaCl"))
        REGISTRY.register(ChannelSpec(name=f"Ax_open_dGsolv_{t}", family=FAMILY, resolution=PB, parity=Parity.SYMMETRIC,
                                      units="kJ/mol", source=SRC, condition_dependent=True,
                                      description=f"solvation part of that opening free energy, {c} NaCl"))
    REGISTRY.register(ChannelSpec(name="Ax_open_dEcoul", family=FAMILY, resolution=PB, parity=Parity.SYMMETRIC,
                                  units="kJ/mol", source=SRC, description="Coulomb part (solute dielectric 2) of that opening free energy"))
    for s in range(6):
        for d in OFFSETS:
            REGISTRY.register(ChannelSpec(name=f"Ax_s{s}_dielimage_d{d}", family=FAMILY, resolution=PB, parity=_sector_parity(s),
                                          units="kT/e", source=SRC,
                                          description=f"image potential of a unit probe {d} A outside the surface, sector {s}"
                                                      + (" (own frame)" if s in (4, 5) else "")))


_declare()


def extras_channels(npz_path: str, n_bp: int = 61) -> Dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}

    def put(name, v):
        a = np.full(n_bp, np.nan); a[CENTRAL] = float(v); out[name] = a
    conds = [str(c) for c in z["conditions"]]
    tag = {"NaCl_150mM": "c150"}
    for k, c in enumerate(conds):
        if c in tag:
            put(f"Ax_open_elec_{tag[c]}", z["open_dG_elec_kJ_mol"][k]); put(f"Ax_open_dGsolv_{tag[c]}", z["open_dG_solv_kJ_mol"][k])
    put("Ax_open_dEcoul", np.asarray(z["open_dE_coulomb_kJ_mol"]).ravel()[0])
    img = np.asarray(z["dielectric_image_kT"])             # (sector, offset)
    for s in range(6):
        for j, d in enumerate(OFFSETS):
            put(f"Ax_s{s}_dielimage_d{d}", img[s, j])
    return out
