#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M groundwork, i-motifs: hemiprotonated C.C+ pairs, stacks, tracts and loops.

A C.C+ pair is two cytosines with N3-N3 ~2.8 A and both N4-O2 contacts (three H-bonds, one
N3 protonated). Pairs are found from those contacts alone, ordered along the stack axis
(intercalated: consecutive pairs alternate between the two parallel-stranded duplexes),
C-tracts read as runs of sequence-adjacent paired cytosines, and loops as the nucleotides
between tracts of a single chain -- the information threading needs, as for quadruplexes.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

import g4_templates as T

N3N3_MAX_A = 3.2
N4O2_MAX_A = 3.4


def find_cc_pairs(res: Dict[Tuple[str, int], dict]) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    cs = [k for k, r in res.items() if r["resname"].endswith("C") and
          all(x in r["atoms"] for x in ("N3", "N4", "O2"))]
    out = []
    for i, j in itertools.combinations(cs, 2):
        a, b = res[i]["atoms"], res[j]["atoms"]
        if (np.linalg.norm(a["N3"] - b["N3"]) < N3N3_MAX_A and
                np.linalg.norm(a["N4"] - b["O2"]) < N4O2_MAX_A and
                np.linalg.norm(a["O2"] - b["N4"]) < N4O2_MAX_A):
            out.append((i, j))
    return out


def classify(path: str) -> Dict[str, object]:
    atoms, _ = T.read_pdb(path)
    res = T.residues(atoms)
    pairs = find_cc_pairs(res)
    out: Dict[str, object] = {"n_cc_pairs": len(pairs)}
    if len(pairs) < 2:
        return out
    chains = sorted({k[0] for p in pairs for k in p})
    out["chains"] = chains
    if len(chains) != 1:
        return out
    ch = chains[0]
    paired = sorted({k[1] for p in pairs for k in p})
    # tracts: maximal runs of consecutive residue numbers among paired cytosines
    tracts, cur = [], [paired[0]]
    for x in paired[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            tracts.append(cur); cur = [x]
    tracts.append(cur)
    keys = sorted(k for k in res if k[0] == ch)
    seq = "".join(res[k]["resname"][-1] for k in keys)
    loops = [tracts[i + 1][0] - tracts[i][-1] - 1 for i in range(len(tracts) - 1)]
    # stack geometry: pair centroids along the principal axis
    cents = np.array([(res[a]["atoms"]["N3"] + res[b]["atoms"]["N3"]) / 2 for a, b in pairs])
    axis = np.linalg.svd(cents - cents.mean(0))[2][0]
    z = np.sort((cents - cents.mean(0)) @ axis)
    out.update({"unimolecular": True, "tract_lengths": [len(t) for t in tracts],
                "loops": loops, "sequence": seq, "n_tracts": len(tracts),
                "pair_rise_A": round(float(np.median(np.diff(z))), 2) if len(z) > 1 else None})
    return out
