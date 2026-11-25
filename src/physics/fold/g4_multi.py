#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M: quadruplex models of every topology (parallel, hybrid 3+1, antiparallel 2+2) and
tetrad count (2 or 3), threaded on solved structures (template set v2).

A model is a solved core of the requested topology and tetrad count -- its guanines kept as
solved, including the syn/anti pattern the topology requires -- with each loop taken from a
solved structure whose loop at that position has the same type (propeller, lateral or diagonal,
`build_g4_templates_v2`) and length, superposed on the core's anchoring guanines (`g4_build.
place_loop`) and its bases substituted to the target's. Per core, donors are chosen loop by loop
by anchor RMSD plus a clash penalty against what is already placed (as for the parallel models);
a core is feasible when every loop has a donor within MAX_ANCHOR_RMSD. Models are the best
feasible (core, donors) combinations, distinct cores first. The parallel three-tetrad case keeps
its validated construction (the 1KF1 core). A topology with no feasible core for the target's
loops is not modelled for it -- the honest outcome where no solved loop of that type and length
exists (e.g. a one-nucleotide lateral loop).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import g4_build as B

TOPOLOGIES = ("parallel", "hybrid", "antiparallel")
MAX_ANCHOR_RMSD = 2.0
CLASH_WEIGHT = 0.05


class Library:
    """All usable template structures with their topology, tetrad count and loop types."""

    def __init__(self, template_dir: str):
        idx = json.load(open(os.path.join(template_dir, "index.json")))
        self.structs: List[B.Structure] = []
        self.meta: Dict[str, dict] = {}
        for r in idx:
            p = os.path.join(template_dir, r["pdb"] + ".pdb")
            try:
                s = B.Structure(p)
            except Exception:                                   # noqa: BLE001
                continue
            if len(s.loops) != 3:
                continue
            # loop types in the builder's (sequence) order
            for lp, t in zip(s.loops, r["loop_types"]):
                lp["type"] = t
            s.topology, s.n_tetrads, s.sequence = r["topology"], r["n_tetrads"], r.get("sequence", "")
            s.fold = r.get("fold", "")
            s.core_ok = all(s.res[k]["resname"].strip() in ("DG", "G") and
                            all(n in s.res[k]["atoms"] for n in ("P", "C1'", "N9", "O6", "N7", "N1", "N2"))
                            for col in s.columns for k in col[1:] if True) and \
                all(s.res[col[0]]["resname"].strip() in ("DG", "G") for col in s.columns)
            self.structs.append(s); self.meta[s.path] = r

    def cores(self, topology: str, n_tet: int) -> List[B.Structure]:
        out = [s for s in self.structs if s.topology == topology and s.n_tetrads == n_tet and s.core_ok]
        if topology == "parallel" and n_tet == 3:
            out = [s for s in out if os.path.basename(s.path) == "1KF1.pdb"] or out
        return out


def typed_candidates(core: B.Structure, i: int, lib: Library, length: int,
                     exclude_sequences=(), top: int = 12):
    """Donor loops for core loop i: same type and length, one per donor structure, by anchor RMSD."""
    want = core.loops[i].get("type")
    out = []
    for s in lib.structs:
        if s.sequence in exclude_sequences:
            continue
        best = None
        for j, lp in enumerate(s.loops):
            if len(lp["residues"]) != length or not lp["usable"] or lp.get("type") != want:
                continue
            res, rn, rmsd = B.place_loop(core, i, s, j)
            if best is None or rmsd < best[2]:
                best = (res, rn, rmsd, os.path.basename(s.path), j)
        if best is not None:
            out.append(best)
    out.sort(key=lambda b: b[2])
    return out[:top]


BUILT_LOOP_PENALTY_A = 3.0          # score of a loop built from sequence (no solved donor)
MAX_LOOP_SPAN_A = 6.5               # per spanning step: a loop of L nt reaches (L + 1) x this


def _anchor_gap(core: B.Structure, i: int) -> float:
    a = core.res[core.loops[i]["anchor5"]]["atoms"].get("O3'")
    b = core.res[core.loops[i]["anchor3"]]["atoms"]
    b = b.get("P", b.get("O5'"))
    return float(np.linalg.norm(a - b)) if a is not None and b is not None else float("inf")


def _donors_for_core(core: B.Structure, loops_seq: Sequence[str], lib: Library, exclude, rank: int,
                     allow_built: bool = True):
    """Per loop the rank-th donor by anchor RMSD plus clash penalty, or -- where no solved loop of
    that type and length exists -- a loop to be built from sequence (`loop_builder`), provided it
    can span the core's anchor gap; None if some loop can be neither."""
    placed = np.concatenate([np.array(list(B._heavy(core.res[k]["atoms"]).values()))
                             for col in core.columns for k in col])
    chosen, score = [], 0.0
    for i, ls in enumerate(loops_seq):
        cands = [c for c in typed_candidates(core, i, lib, len(ls), exclude) if c[2] <= MAX_ANCHOR_RMSD]
        if not cands:
            if allow_built and _anchor_gap(core, i) <= MAX_LOOP_SPAN_A * (len(ls) + 1):
                chosen.append(None); score += BUILT_LOOP_PENALTY_A
                continue
            return None, None
        scored = []
        for c in cands:
            xyz = np.array([x for r in c[0] for x in B._heavy(r).values()])
            scored.append((c[2] + CLASH_WEIGHT * B._clashes(xyz, placed), c))
        scored.sort(key=lambda t: t[0])
        s, c = scored[min(rank, len(scored) - 1)]
        chosen.append(c); score += s
        placed = np.concatenate([placed, np.array([x for r in c[0] for x in B._heavy(r).values()])])
    return chosen, score


def plan_models(loops_seq: Sequence[str], topology: str, n_tet: int, lib: Library,
                n_models: int = 3, exclude_sequences=()) -> List[Tuple[B.Structure, list, float]]:
    """Up to n_models (core, donors, score). A topology class holds several folds (hybrid-1 and
    hybrid-2, basket and chair, ...: loop types and their sense around the stack) that sequence
    alone does not choose between (leave-one-out validation, DECISIONS.md), so the ensemble takes
    the best-scoring core of each feasible fold first, then the next-best cores, then next-ranked
    donors on the best core; the across-model spread is therefore the fold uncertainty too."""
    per_core = []
    for core in lib.cores(topology, n_tet):
        if core.sequence in exclude_sequences:
            continue
        d, sc = _donors_for_core(core, loops_seq, lib, exclude_sequences, rank=0)
        if d is not None:
            per_core.append((sc, core, d))
    per_core.sort(key=lambda t: t[0])
    out, folds = [], set()
    for sc, core, d in per_core:                       # best core of each fold
        if core.fold not in folds and len(out) < n_models:
            out.append((core, d, sc)); folds.add(core.fold)
    for sc, core, d in per_core:                       # then the next-best cores
        if len(out) >= n_models:
            break
        if all(core is not o[0] for o in out):
            out.append((core, d, sc))
    rank = 1
    while out and len(out) < n_models and rank < 4:
        c0 = out[0][0]
        d, sc = _donors_for_core(c0, loops_seq, lib, exclude_sequences, rank=rank)
        key = lambda dd: [None if x is None else x[3:] for x in dd]                # noqa: E731
        if d is not None and all(key(d) != key(o[1]) for o in out):
            out.append((c0, d, sc))
        rank += 1
    out.sort(key=lambda t: t[2])
    return out


def core_and_loop_indices(loops_seq: Sequence[str], n_tet: int):
    """1-based model residue indices of core guanines and of loop nucleotides."""
    core_idx, loop_idx, m = [], [], 0
    for c in range(4):
        core_idx += list(range(m + 1, m + 1 + n_tet)); m += n_tet
        if c < 3:
            loop_idx += list(range(m + 1, m + 1 + len(loops_seq[c]))); m += len(loops_seq[c])
    return core_idx, loop_idx
