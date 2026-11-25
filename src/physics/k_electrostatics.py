#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family K electrostatics channels: PLAN.md Family K's "electrostatics solved on A-form
conformations", from `run_k_electrostatics.py` (APBS NPBE, 150 mM NaCl, Family A's production
construction, central 21 bp of each capped molecule scored).

For each Family K molecule -- `rna` (RNA duplex of x), `hybp` (DNA strand x, RNA partner), `hybm`
(DNA strand rc(x), stored in its own frame) -- the Family A channel set at 150 mM NaCl, assembled by
Family A's own assembler (`electrostatics_family.assemble`: sector potentials at six offsets,
groove fields and probe energies, condensed counterions, strand and groove asymmetries, per-bp
and total solvation, Na+ enrichment, Manning parameter and surface charge) and renamed
K_<molecule>_<channel>. Channels Family A derives from other conditions (salt slopes, K+ and Mg2+
occupancy, long windows, the conformational ensemble) are not part of this set. The RNA duplex
keeps Family A's parities; the hybrids are a stranded pair, K_hybm(x) = K_hybp(rc x) position by
position, as for the rest of Family K.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import electrostatics_family as EF
from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "Kes"    # family code for the side shards (Family K of PLAN.md)
TAGS = ("rna", "hybp", "hybm")
COND_TAG = "c150"
SRC = "APBS nonlinear Poisson-Boltzmann on minimised cgNA+ A-form RNA and hybrid duplexes (run_k_electrostatics.py)"
_SKIP = ("A_lw_", "saltslope", "_ensmean", "_enssd", "A_K_", "A_Mg_")


def _keep(name: str) -> bool:
    if any(k in name for k in _SKIP):
        return False
    if name in ("A_manning_xi", "A_surface_charge") or name.startswith("A_Na_"):
        return True
    return name.endswith(f"_{COND_TAG}")


def _rename(name: str, tag: str) -> str:
    base = name[2:]
    if base.endswith(f"_{COND_TAG}"):
        base = base[: -len(COND_TAG) - 1]
    return f"K_{tag}_es_{base}"


def assemble_k(z, tag: str, n_bp: int = 61) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    prof, glob = EF.assemble(z, n_bp=n_bp)
    return ({_rename(k, tag): v for k, v in prof.items() if _keep(k)},
            {_rename(k, tag): v for k, v in glob.items() if _keep(k)})


def _declare() -> None:
    a = {s.name: s for s in REGISTRY.by_family("A")}
    for tag in TAGS:
        what = {"rna": "RNA duplex", "hybp": "DNA-RNA hybrid, DNA strand x", "hybm": "DNA-RNA hybrid, DNA strand rc(x)"}[tag]
        for n, sp in a.items():
            if not _keep(n):
                continue
            REGISTRY.register(ChannelSpec(
                name=_rename(n, tag), family=FAMILY, resolution=sp.resolution,
                parity=sp.parity if tag == "rna" else Parity.STRANDED, units=sp.units,
                description=f"{sp.description.replace('NaCl_150mM', '150 mM NaCl')} ({what}, A-form)",
                source=SRC, condition_dependent=False, rc_shift=sp.rc_shift if tag == "rna" else 0,
                circular=sp.circular))


_declare()
