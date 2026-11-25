#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family I's "generic amino acid side chain interaction energies in each groove" as channels
(family code "Isc", side shards; held apart with the rest of Family I: the labelled protein-readout
variant only, never the main ladder or a headline number).

From `sidechain_probes.py`: per RC-unique tetramer context, capped dipeptides ACE-X-NME of the seven
DNA-reading residues searched (Monte Carlo minimisation, tip held in the groove) in the minor and the
major groove of the central step on the fixed 12-mer (ff14SB / OL15, GBn2 at 0.15 M); the interaction
energy of the lowest-energy complex and the Boltzmann-weighted interaction energy (kcal/mol). Per base-pair step (i, i+1), looked up by the 4-mer starting at i-1 (or its
reverse complement), as the other context lookups (`context_libraries`): SYMMETRIC, rc_shift -1.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "Isc"
RESIDUES = ("ARG", "LYS", "GLN", "ASN", "SER", "THR", "HIS")
GROOVES = ("minor", "major")
STATS = (("best", "lowest-energy complex:"), ("boltz", "Boltzmann-weighted (300 K)"))
SRC = "capped dipeptides on the fixed 12-mer, Monte Carlo minimisation in the groove, AMBER ff14SB/OL15, GBn2 0.15 M (sidechain_probes.py v2)"
COMP = str.maketrans("ACGT", "TGCA")


def _declare() -> None:
    for g in GROOVES:
        for r in RESIDUES:
            for st, word in STATS:
                REGISTRY.register(ChannelSpec(
                    name=f"Isc_{g}_{r}_{st}", family=FAMILY, resolution=Resolution.PER_BP, parity=Parity.SYMMETRIC,
                    units="kcal/mol", rc_shift=-1, source=SRC,
                    description=f"{word} interaction energy of an {r} side chain (capped dipeptide) in the {g} groove of step (i, i+1)"))


_declare()


def load_table(folder: str) -> Dict[str, Dict[str, float]]:
    """Context -> channel values, merged over the per-(context, residue) files of the v2 search
    (`{context}__{residue}.json`); the search records (`*_search`) are left out."""
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".json"):
            d = json.load(open(os.path.join(folder, f)))
            if "context" in d:
                out.setdefault(d["context"], {}).update({k: v for k, v in d.items() if not k.endswith("_search")})
    return out


def contexts():
    """The 136 reverse-complement-unique tetramers (the lexicographically smaller of each pair)."""
    import itertools
    return sorted({min(c, c.translate(COMP)[::-1]) for c in ("".join(t) for t in itertools.product("ACGT", repeat=4))})


def missing_entries(table):
    """(context, channel) pairs without a finite value: holes that must not reach the shards."""
    canon = {}
    for c, rec in table.items():
        canon.setdefault(min(c, c.translate(COMP)[::-1]), {}).update(rec)
    want = [f"{g}_{r}_{st}" for g in GROOVES for r in RESIDUES for st, _ in STATS]
    return [(c, k) for c in contexts() for k in want
            if not (isinstance(canon.get(c, {}).get(k), (int, float)) and np.isfinite(canon[c][k]))]


def sidechain_channels(seq: str, table) -> Dict[str, np.ndarray]:
    s = seq.upper(); n = len(s)
    prof = {f"Isc_{g}_{r}_{st}": np.full(n, np.nan) for g in GROOVES for r in RESIDUES for st, _ in STATS}
    for i in range(1, n - 2):
        ctx = s[i - 1:i + 3]
        rec = table.get(ctx) or table.get(ctx.translate(COMP)[::-1])
        if rec is None:
            continue
        for g in GROOVES:
            for r in RESIDUES:
                for st, _ in STATS:
                    v = rec.get(f"{g}_{r}_{st}")
                    if v is not None and np.isfinite(v):
                        prof[f"Isc_{g}_{r}_{st}"][i] = float(v)
    return prof
