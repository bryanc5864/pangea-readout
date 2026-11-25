#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, i-motifs: unimolecular models threaded on solved tetramolecular C.C+ cores.

An i-motif's core is two parallel-stranded duplexes of hemiprotonated C.C+ pairs, intercalated
antiparallel; a unimolecular one is the same core with its four C-tracts joined by three loops.
The solved tetramolecular i-motifs (data/imotif_templates_v1: 1BQJ, 1CN0 with 3 C per strand;
1YBL, 1YBN, 1YBR with 4) are exactly such cores. For a fragment's C-rich strand (the complement
of Family F's best quadruplex candidate) with four tracts of >= 3 C: n = min tract length (3, or
4 for longer tracts; spare C join the loops as for quadruplexes), the template's chains are
assigned to tracts so that tracts 1 and 3 form one duplex and run 5'->3' the same way and the
three loop gaps (top, bottom, top) are shortest, and the loops are built from sequence by
`loop_builder` with the core restrained (no solved unimolecular i-motif loops of matching
lengths exist beyond three structures). A loop too short for its gap (anchor distance above
6.5 A per spanning step) makes that assignment infeasible.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import g4_build as B          # noqa: E402
import g4_templates as T      # noqa: E402

_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data", "imotif_templates_v1")
TEMPLATE_DIR = os.environ.get("PANGEA_IMOTIF_TEMPLATES", _REL if os.path.exists(os.path.join(_REL, "index.json"))
                              else "/home/bjc75/other/biophyschem/data/imotif_templates_v1")
COMP = str.maketrans("ACGT", "TGCA")
SPAN_PER_STEP_A = 6.5


def select(seq: str) -> Optional[dict]:
    """C-rich strand, tracts, loops and fragment positions of the modelled i-motif, or None."""
    import noncanonical as NC
    s = seq.upper(); n = len(s)
    cands = NC.find_g4_candidates(s)
    if not cands:
        return None
    best = max(cands, key=NC.g4_stability_score)
    # the C-rich strand is the complement of the candidate's G-rich strand
    if best.strand == "+":
        built, lo, hi, to_top = s.translate(COMP)[::-1], n - best.end, n - best.start, (lambda j: n - 1 - j)
    else:
        built, lo, hi, to_top = s, best.start, best.end, (lambda j: j)
    tr = [(m.start(), m.end()) for m in re.finditer(r"C{3,}", built) if m.start() >= lo and m.end() <= hi]
    if len(tr) < 4:
        return None
    choice = None
    for i in range(len(tr) - 3):
        g = tr[i:i + 4]; tot = sum(g[k + 1][0] - g[k][1] for k in range(3))
        if choice is None or tot < choice[0]:
            choice = (tot, g)
    g = choice[1]
    nc = min(4, min(b - a for a, b in g))
    cols = []
    for k, (a, b) in enumerate(g):
        if k == 0:
            cols.append(list(range(b - nc, b)))
        elif k == 3:
            cols.append(list(range(a, a + nc)))
        else:
            m = a + (b - a - nc) // 2; cols.append(list(range(m, m + nc)))
    loop_pos = [list(range(cols[k][-1] + 1, cols[k + 1][0])) for k in range(3)]
    if any(len(lp) == 0 for lp in loop_pos):
        return None
    order = sorted([p for c in cols for p in c] + [p for lp in loop_pos for p in lp])
    return {"built": built, "n_c": nc, "cols": cols, "loops": ["".join(built[p] for p in lp) for lp in loop_pos],
            "loop_pos": loop_pos, "order": order, "positions": [to_top(j) for j in order]}


def templates(n_c: int) -> List[dict]:
    idx = json.load(open(os.path.join(TEMPLATE_DIR, "index.json")))
    out = []
    for r in idx:
        ok = all(np.all(np.diff(v["resseq"]) == 1) for v in r["chains"].values())
        if r["n_c"] == n_c and ok:
            out.append(r)
    return out


def _assignments(tpl: dict, res, loops: List[str]):
    """Feasible (tract chains, total gap) assignments, shortest gaps first."""
    up = [c for c, v in tpl["chains"].items() if v["dir"] == 1]
    dn = [c for c, v in tpl["chains"].items() if v["dir"] == -1]
    dup = {tuple(d) for d in tpl["duplexes"]}
    out = []
    for t1, t3 in itertools.permutations(up, 2):
        for t2, t4 in itertools.permutations(dn, 2):
            if tuple(sorted((t1, t3))) not in dup or tuple(sorted((t2, t4))) not in dup:
                continue
            chains = [t1, t2, t3, t4]
            gaps, ok = [], True
            for k in range(3):
                a = res[(chains[k], tpl["chains"][chains[k]]["resseq"][-1])]["atoms"]["O3'"]
                b_res = res[(chains[k + 1], tpl["chains"][chains[k + 1]]["resseq"][0])]["atoms"]
                b = b_res.get("P", b_res.get("O5'"))
                d = float(np.linalg.norm(a - b)); gaps.append(d)
                if d > SPAN_PER_STEP_A * (len(loops[k]) + 1):
                    ok = False
            if ok:
                out.append((chains, sum(gaps), gaps))
    out.sort(key=lambda t: t[1])
    return out


def plan_models(sel: dict, n_models: int = 3) -> List[dict]:
    plans = []
    for tpl in templates(sel["n_c"]):
        atoms, _ = T.read_pdb(os.path.join(TEMPLATE_DIR, tpl["pdb"] + ".pdb"))
        res = T.residues(atoms)
        for chains, tot, gaps in _assignments(tpl, res, sel["loops"])[:2]:
            plans.append({"template": tpl["pdb"], "chains": chains, "gap_total_A": tot, "gaps_A": gaps, "res": res, "tpl": tpl})
    plans.sort(key=lambda p: p["gap_total_A"])
    seen, out = set(), []
    for p in plans:                                     # distinct templates first
        if p["template"] not in seen:
            out.append(p); seen.add(p["template"])
    for p in plans:
        if len(out) >= n_models:
            break
        if p not in out:
            out.append(p)
    return out[:n_models]


def build_model(sel: dict, plan: dict, work: str, seed: int = 0) -> Tuple[str, dict]:
    """Core from the template (chains in tract order), loops from sequence; returns (pdb, info)."""
    import loop_builder as LB
    res, tpl = plan["res"], plan["tpl"]
    known, missing = [], {}
    for k, ch in enumerate(plan["chains"]):
        for j, rs in enumerate(tpl["chains"][ch]["resseq"]):
            at = B._heavy(res[(ch, rs)]["atoms"])
            if k == 0 and j == 0:
                at = {n: x for n, x in at.items() if n not in B.PHOSPHATE_GROUP}
            known.append(("DC", at))
        if k < 3:
            missing[len(known)] = [B.LETTER_TO_RESNAME[x] for x in sel["loops"][k]]
    gp = LB.write_gapped_pdb(known, missing, os.path.join(work, "gap.pdb"))
    # model residue numbering: tract, loop, tract, ... (1-based)
    known_num, m = [], 0
    for k in range(4):
        known_num += list(range(m + 1, m + 1 + sel["n_c"])); m += sel["n_c"]
        if k < 3:
            m += len(sel["loops"][k])
    out = os.path.join(work, "model.pdb")
    info = LB.fill_and_relax(gp, out, known_num, seed=seed)
    info.update(core_residues=known_num, template=plan["template"], chains=plan["chains"], gaps_A=plan["gaps_A"])
    return out, info
