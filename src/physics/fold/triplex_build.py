#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, triplexes: the core of an intramolecular (H-DNA) pyrimidine-motif triplex.

An H-DNA candidate (Family F's: a homopurine/homopyrimidine mirror repeat, arm a, spacer s) folds
so that one pyrimidine arm returns through the spacer and lies in the major groove of the other
arm's duplex, parallel to the purine strand, Hoogsteen-paired base by base (T*A-T, C+*G-C, the
third-strand cytosine N3-protonated). Built here: the duplex of that arm from its cgNA+ ground
state (`cgna_atomic`), and each third-strand nucleotide placed from the medoid solved triplet of
its type (`data/triplex_templates_v1/triplets.json`) in the Tsukuba frame of its purine partner.
The spacer loop and the displaced purine strand are not part of the core model (recorded). Chains:
A purine strand, B its Watson-Crick partner, C the third strand (5'->3' parallel to A).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cgna_atomic as ca  # noqa: E402

_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data",
                    "triplex_templates_v1", "triplets.json")
TEMPLATES = os.environ.get("PANGEA_TRIPLEX_TEMPLATES", _REL if os.path.exists(_REL)
                           else "/home/bjc75/other/biophyschem/data/triplex_templates_v1/triplets.json")
THIRD = {"A": ("T", "T*A-T"), "G": ("C", "C*G-C")}
CAP = "GC"


def load_templates():
    d = json.load(open(TEMPLATES))["templates"]
    return {k: {n: np.asarray(v, float) for n, v in t.items()} for k, t in d.items()}


def purine_tract(seq: str, centre: int, arm: int, spacer: int) -> Optional[Tuple[str, int, int]]:
    """(purine-strand sequence of the triplex arm, start, end on the fragment) or None.

    The arm 3' of the spacer on the purine-rich strand forms the duplex core; which strand is
    purine-rich decides the orientation."""
    s = seq.upper()
    lo, hi = centre - arm, centre + spacer + arm
    tract = s[lo:hi]
    pur = sum(b in "AG" for b in tract) / max(1, len(tract))
    if pur >= 0.5:
        core = s[centre + spacer:hi]                       # purine strand, 5'->3'
        return core, centre + spacer, hi
    comp = s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    n = len(s)
    # on the complementary strand the purine arm downstream of the spacer
    return comp[n - (centre - 0):n - lo] if False else (comp[n - centre:n - lo], n - centre, n - lo)


def build_core(purine: str, templates=None) -> List["ca.Atom"]:
    """Atoms of the triplex core for a purine-strand sequence (pyrimidines in it get no third
    nucleotide: such steps are recorded by `n_triplets`)."""
    templates = templates or load_templates()
    capped = CAP + purine + CAP[::-1]
    gs = ca.CGNAGroundState(capped)
    atoms = ca.build_duplex(capped, None, gs)
    wb = gs.watson_bases
    third_atoms: List[ca.Atom] = []
    k = 0
    for i, b in enumerate(capped):
        if not (len(CAP) <= i < len(CAP) + len(purine)) or b not in THIRD:
            continue
        letter, key = THIRD[b]
        k += 1
        tau = wb[i]
        for nm, x in templates[key].items():
            if k == 1 and nm in ("P", "OP1", "OP2"):
                continue                                   # 5' end of the third strand
            third_atoms.append(ca.Atom(nm, ca.RESNAME[letter], "C", k, ca.apply_se3(tau, x)[0], ca._element_of(nm)))
    return atoms + third_atoms
