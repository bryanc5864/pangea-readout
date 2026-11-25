#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M groundwork: read a G-quadruplex structure and find its tetrads, loops and ions.

A G-tetrad is a cycle of four guanines in which each donates N1-H and N2-H to its
neighbour's O6 and N7 (Hoogsteen face). Tetrads are found from those contacts alone (no
reliance on PDB annotations), stacked tetrads ordered along the helical axis, loops read
as the nucleotides between consecutive G-runs of a strand, and ions (K+/Na+/NH4+) located
relative to the tetrad planes. This is what the threading stage (DECISIONS.md, Family M)
needs from every template.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

HB_MAX_A = 3.3          # heavy-atom distance for a Hoogsteen N1-O6 / N2-N7 contact


def read_pdb(path: str, model: int = 1) -> Tuple[List[dict], List[dict]]:
    """(atoms, ions) of one model; atoms carry chain, resseq, resname, name, xyz."""
    atoms, ions, current = [], [], 1
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = int(line.split()[1])
            if current != model or not line.startswith(("ATOM", "HETATM")):
                continue
            rec = {"name": line[12:16].strip(), "resname": line[17:20].strip(), "chain": line[21],
                   "resseq": int(line[22:26]), "xyz": np.array([float(line[30:38]), float(line[38:46]),
                                                                  float(line[46:54])])}
            if rec["resname"] in ("K", "NA", "NH4", "TL", "SR") or rec["name"] in ("K", "NA"):
                ions.append(rec)
            elif line.startswith("ATOM") or rec["resname"] in ("DG", "DA", "DC", "DT", "G", "A", "C", "U"):
                atoms.append(rec)
    return atoms, ions


def residues(atoms: List[dict]) -> Dict[Tuple[str, int], dict]:
    res: Dict[Tuple[str, int], dict] = {}
    for a in atoms:
        r = res.setdefault((a["chain"], a["resseq"]), {"resname": a["resname"], "atoms": {}})
        r["atoms"][a["name"]] = a["xyz"]
    return res


def find_tetrads(res: Dict[Tuple[str, int], dict]) -> List[List[Tuple[str, int]]]:
    """Cycles of four guanines bound by Hoogsteen contacts, each as an ordered list of keys."""
    gs = [k for k, r in res.items() if r["resname"].endswith("G") and
          all(x in r["atoms"] for x in ("N1", "N2", "O6", "N7"))]
    donor_to = defaultdict(set)              # i donates to j (i's N1/N2 to j's O6/N7)
    for i, j in itertools.permutations(gs, 2):
        ai, aj = res[i]["atoms"], res[j]["atoms"]
        if (np.linalg.norm(ai["N1"] - aj["O6"]) < HB_MAX_A and
                np.linalg.norm(ai["N2"] - aj["N7"]) < HB_MAX_A):
            donor_to[i].add(j)
    tetrads, seen = [], set()
    for a in gs:
        for b in donor_to[a]:
            for c in donor_to[b]:
                for d in donor_to[c]:
                    if a in donor_to[d] and len({a, b, c, d}) == 4:
                        key = frozenset((a, b, c, d))
                        if key not in seen:
                            seen.add(key); tetrads.append([a, b, c, d])
    return tetrads


def tetrad_geometry(res, tetrads) -> Dict[str, object]:
    """Planes, axis, rise and twist between consecutive tetrads, planarity."""
    cents, normals, planarity = [], [], []
    for t in tetrads:
        pts = np.array([res[k]["atoms"][n] for k in t for n in ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9")])
        c = pts.mean(0); u, s, vt = np.linalg.svd(pts - c)
        cents.append(c); normals.append(vt[2]); planarity.append(float(s[2] / np.sqrt(len(pts))))
    cents = np.array(cents)
    axis = np.linalg.svd(cents - cents.mean(0))[2][0] if len(cents) > 1 else normals[0]
    order = np.argsort((cents - cents.mean(0)) @ axis)
    tetrads = [tetrads[i] for i in order]; cents = cents[order]
    rise = [float(abs((cents[i + 1] - cents[i]) @ axis)) for i in range(len(cents) - 1)]
    twist = []
    for i in range(len(tetrads) - 1):
        # angle between the C1'-C1' vector of the first guanine pair, projected on the plane
        v1 = res[tetrads[i][0]]["atoms"].get("C1'", cents[i]) - cents[i]
        best = None
        for k in tetrads[i + 1]:
            v2 = res[k]["atoms"].get("C1'", cents[i + 1]) - cents[i + 1]
            p1, p2 = v1 - (v1 @ axis) * axis, v2 - (v2 @ axis) * axis
            ang = np.degrees(np.arctan2(np.cross(p1, p2) @ axis, p1 @ p2))
            if best is None or abs(ang) < abs(best):
                best = ang
        twist.append(float(best))
    return {"order": tetrads, "centroids": cents, "axis": axis, "rise_A": rise, "twist_deg": twist,
            "planarity_rms_A": planarity}


def ion_sites(ions, geom) -> List[dict]:
    out = []
    for ion in ions:
        z = float((ion["xyz"] - geom["centroids"].mean(0)) @ geom["axis"])
        r = float(np.linalg.norm((ion["xyz"] - geom["centroids"].mean(0)) - z * geom["axis"]))
        out.append({"ion": ion["resname"] or ion["name"], "axial_A": round(z, 2), "radial_A": round(r, 2)})
    return out


def g_runs(tetrads_ordered: List[List[Tuple[str, int]]]) -> List[List[Tuple[str, int]]]:
    """Columns of the stack: each guanine linked to a sequence neighbour (same chain,
    resseq +-1) in the next tetrad. Returns runs as lists of keys in tetrad order."""
    runs = [[g] for g in tetrads_ordered[0]]
    for t in tetrads_ordered[1:]:
        free = list(t)
        for run in runs:
            last = run[-1]
            hit = [g for g in free if g[0] == last[0] and abs(g[1] - last[1]) == 1]
            if hit:
                run.append(hit[0]); free.remove(hit[0])
    return runs


def classify(path: str) -> Dict[str, object]:
    """Topology and loop lengths of a unimolecular G-quadruplex (model 1)."""
    atoms, ions = read_pdb(path)
    res = residues(atoms)
    t = find_tetrads(res)
    out: Dict[str, object] = {"n_tetrads": len(t), "n_ions": len(ions)}
    if len(t) < 2:
        return out
    g = tetrad_geometry(res, t)
    runs = g_runs(g["order"])
    chains = {k[0] for run in runs for k in run}
    out.update({"chains": sorted(chains), "run_lengths": [len(r) for r in runs],
                "rise_A": g["rise_A"], "twist_deg": g["twist_deg"]})
    if len(chains) != 1 or any(len(r) != len(t) for r in runs) or len(runs) != 4:
        return out                                   # multimolecular or irregular stack
    dirs = [int(np.sign(r[-1][1] - r[0][1])) for r in runs]
    order = sorted(runs, key=lambda r: min(k[1] for k in r))
    spans = [(min(k[1] for k in r), max(k[1] for k in r)) for r in order]
    loops = [spans[i + 1][0] - spans[i][1] - 1 for i in range(3)]
    chain = next(iter(chains))
    seq = "".join(res[(chain, i)]["resname"][-1] for i in sorted(k[1] for k in res if k[0] == chain))
    out.update({"unimolecular": True, "parallel": len(set(dirs)) == 1, "strand_dirs": dirs,
                "loops": loops, "spans": spans, "sequence": seq})
    return out
