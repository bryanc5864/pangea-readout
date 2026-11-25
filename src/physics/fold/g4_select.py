#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Which quadruplex of a fragment Family M models, and how its G tracts become a core.

The candidate is Family F's best one (`noncanonical.find_g4_candidates`, ranked by
`g4_stability_score`), so the explicit-geometry channels describe the same quadruplex the
propensity channels score. A three-tetrad parallel core needs four tracts of at least
three guanines: with more than four tracts, the four consecutive tracts with the shortest
total loop length are used; in a tract longer than three, the first tract keeps its three
3'-most guanines, the last its three 5'-most, a middle tract its central three, and the
spare guanines join the neighbouring loops. Loops longer than `MAX_LOOP` have too few solved
donors (DECISIONS.md, Family M) and are not modelled. A C-rich candidate is built on the
reverse complement, where it is G-rich, and mapped back.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noncanonical as NC  # noqa: E402

MAX_LOOP = 6
COMP = str.maketrans("ACGT", "TGCA")


@dataclass
class G4Model:
    strand: str                       # "+" built on the fragment, "-" on its reverse complement
    core_positions: List[List[int]]   # 4 x 3 guanine positions, in the built strand's frame
    loops: List[str]                  # three loop sequences (built strand, 5'->3')
    loop_positions: List[List[int]]
    reason: str = "ok"

    def fragment_positions(self, n: int) -> List[int]:
        """Every modelled nucleotide, as a position on the fragment (top strand index)."""
        built = sorted(p for col in self.core_positions for p in col) + \
            sorted(p for lp in self.loop_positions for p in lp)
        return sorted(built) if self.strand == "+" else sorted(n - 1 - p for p in built)


def _tracts(s: str) -> List[Tuple[int, int]]:
    import re
    return [(m.start(), m.end()) for m in re.finditer(r"G{3,}", s)]


def select(seq: str) -> Optional[G4Model]:
    """The modelled quadruplex of `seq`, or None with no qualifying candidate."""
    seq = seq.upper()
    cands = NC.find_g4_candidates(seq)
    if not cands:
        return None
    best = max(cands, key=NC.g4_stability_score)
    n = len(seq)
    built = seq if best.strand == "+" else seq.translate(COMP)[::-1]
    lo, hi = (best.start, best.end) if best.strand == "+" else (n - best.end, n - best.start)
    tr = [t for t in _tracts(built) if t[0] >= lo and t[1] <= hi]
    if len(tr) < 4:
        return G4Model(best.strand, [], [], [], reason="fewer than four tracts of >= 3 G")
    choice = None
    for i in range(len(tr) - 3):
        group = tr[i:i + 4]
        total = sum(group[k + 1][0] - group[k][1] for k in range(3))
        if choice is None or total < choice[0]:
            choice = (total, group)
    group = choice[1]
    cols = []
    for k, (a, b) in enumerate(group):
        if k == 0:
            cols.append(list(range(b - 3, b)))
        elif k == 3:
            cols.append(list(range(a, a + 3)))
        else:
            m = a + (b - a - 3) // 2
            cols.append(list(range(m, m + 3)))
    loop_pos = [list(range(cols[k][-1] + 1, cols[k + 1][0])) for k in range(3)]
    loops = ["".join(built[p] for p in lp) for lp in loop_pos]
    if any(len(lp) == 0 for lp in loops):
        return G4Model(best.strand, cols, loops, loop_pos, reason="zero-length loop")
    if max(len(lp) for lp in loops) > MAX_LOOP:
        return G4Model(best.strand, cols, loops, loop_pos, reason=f"loop longer than {MAX_LOOP}")
    return G4Model(best.strand, cols, loops, loop_pos)


@dataclass
class G4ModelV3(G4Model):
    n_tet: int = 3


def _columns(group, n_tet):
    cols = []
    for k, (a, b) in enumerate(group):
        if k == 0:
            cols.append(list(range(b - n_tet, b)))
        elif k == 3:
            cols.append(list(range(a, a + n_tet)))
        else:
            m = a + (b - a - n_tet) // 2
            cols.append(list(range(m, m + n_tet)))
    return cols


def select_v3(seq: str) -> Optional[G4ModelV3]:
    """Family M v3: Family F's best candidate as a three-tetrad core when four tracts of >= 3 G
    exist, else as a two-tetrad core when four tracts of >= 2 G do (two-tetrad quadruplexes are
    solved structures too: 53 in the template set). No loop-length cap here -- whether a loop can
    be built is decided by the solved loops of its type and length (`g4_multi`)."""
    import re
    seq = seq.upper()
    cands = NC.find_g4_candidates(seq)
    if not cands:
        return None
    best = max(cands, key=NC.g4_stability_score)
    n = len(seq)
    built = seq if best.strand == "+" else seq.translate(COMP)[::-1]
    lo, hi = (best.start, best.end) if best.strand == "+" else (n - best.end, n - best.start)
    for n_tet in (3, 2):
        tr = [(m.start(), m.end()) for m in re.finditer(r"G{%d,}" % n_tet, built)
              if m.start() >= lo and m.end() <= hi]
        if len(tr) < 4:
            continue
        choice = None
        for i in range(len(tr) - 3):
            group = tr[i:i + 4]
            total = sum(group[k + 1][0] - group[k][1] for k in range(3))
            if choice is None or total < choice[0]:
                choice = (total, group)
        cols = _columns(choice[1], n_tet)
        loop_pos = [list(range(cols[k][-1] + 1, cols[k + 1][0])) for k in range(3)]
        loops = ["".join(built[p] for p in lp) for lp in loop_pos]
        if any(len(lp) == 0 for lp in loops):
            return G4ModelV3(best.strand, cols, loops, loop_pos, reason="zero-length loop", n_tet=n_tet)
        return G4ModelV3(best.strand, cols, loops, loop_pos, n_tet=n_tet)
    return G4ModelV3(best.strand, [], [], [], reason="fewer than four tracts of >= 2 G", n_tet=0)
