#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M template set v2: every unimolecular G-quadruplex of 2-4 tetrads, all topologies.

From the downloaded PDB quadruplex set (data only: parsed, never executed), each structure is
classified from its own coordinates (`g4_templates.classify`): tetrads from Hoogsteen contacts,
columns as sequence-contiguous guanines through the stack, strand directions, loops. Kept: one
chain, four columns each spanning every tetrad, 2-4 tetrads, and a model the builder can read
(`g4_build.Structure`). Each loop gets its type from the two columns it joins: propeller
(columns parallel), lateral (antiparallel, adjacent in the tetrad cycle) or diagonal
(antiparallel, opposite). Topology: parallel (4 same-direction columns), hybrid (3+1) or
antiparallel (2+2). Writes the PDB files and index.json to the output folder.
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_templates as T  # noqa: E402
import g4_build as B      # noqa: E402


def loop_types(path):
    s = B.Structure(path)
    t0 = s.geom["order"][0]
    pos = {}
    for ci, col in enumerate(s.columns):
        hit = [k for k in col if k in t0]
        pos[ci] = t0.index(hit[0]) if hit else None
    # direction of each column along the stack: tetrad index of its 3'-most guanine minus that of
    # its 5'-most (columns are stored in sequence order)
    tix = {k: ti for ti, t in enumerate(s.geom["order"]) for k in t}
    dirs = [1 if tix[col[-1]] > tix[col[0]] else -1 for col in s.columns]
    out = []
    for i in range(3):
        if dirs[i] == dirs[i + 1]:
            out.append("propeller")
        elif pos[i] is None or pos[i + 1] is None:
            out.append("unknown")
        else:
            out.append("diagonal" if abs(pos[i] - pos[i + 1]) % 4 == 2 else "lateral")
    # fold descriptor: loop type and the sense in which it progresses around the stack, viewed
    # down the axis from the face holding the first column's 5' guanine (+ anticlockwise,
    # - clockwise; diagonal loops cross the stack and carry no sense), with the first column's
    # direction as reference -- distinguishes e.g. hybrid-1 from hybrid-2 and basket from chair
    import numpy as np
    geom = s.geom
    ax = np.asarray(geom["axis"], float)
    c0 = np.asarray(geom["centroids"], float).mean(0)
    first_face = tix[s.columns[0][0]]
    if first_face != 0:
        ax = -ax                                          # view from the first guanine's face
    u = np.cross(ax, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(ax, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u); v = np.cross(ax, u)
    az = []
    for col in s.columns:
        pts = np.array([s.res[k]["atoms"]["C1'"] for k in col if "C1'" in s.res[k]["atoms"]])
        d = pts.mean(0) - c0
        az.append(np.degrees(np.arctan2(d @ v, d @ u)))
    fold = []
    for i in range(3):
        da = (az[i + 1] - az[i] + 180.0) % 360.0 - 180.0
        t = out[i][0]
        fold.append(t if t == "d" else t + ("+" if da > 0 else "-"))
    return out, [lp["usable"] for lp in s.loops], "".join(fold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index, skipped = [], {}
    for name in sorted(os.listdir(a.pdb_dir)):
        if not name.endswith(".pdb"):
            continue
        p = os.path.join(a.pdb_dir, name)
        try:
            c = T.classify(p)
        except Exception as e:                                   # noqa: BLE001
            skipped[name] = f"classify: {type(e).__name__}"; continue
        if not c.get("unimolecular") or c["n_tetrads"] not in (2, 3, 4):
            skipped[name] = "not unimolecular 2-4 tetrads"; continue
        try:
            lt, usable, fold = loop_types(p)
        except Exception as e:                                   # noqa: BLE001
            skipped[name] = f"builder: {type(e).__name__}: {str(e)[:80]}"; continue
        n_up = sum(1 for d in c["strand_dirs"] if d == c["strand_dirs"][0])
        topo = "parallel" if c["parallel"] else ("hybrid" if n_up in (1, 3) else "antiparallel")
        shutil.copyfile(p, os.path.join(a.out, name))
        index.append({"pdb": name[:-4], "n_tetrads": c["n_tetrads"], "topology": topo,
                      "loops": c["loops"], "loop_types": lt, "fold": fold, "loops_usable": usable,
                      "strand_dirs": c["strand_dirs"], "sequence": c["sequence"], "n_ions": c["n_ions"]})
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    json.dump(skipped, open(os.path.join(a.out, "skipped.json"), "w"), indent=1)
    from collections import Counter
    print(len(index), "templates;", len(skipped), "skipped")
    print(sorted(Counter((r["n_tetrads"], r["topology"]) for r in index).items()))
    print(sorted(Counter((r["n_tetrads"], r["topology"], r["fold"]) for r in index).items()))
    print(sorted(Counter((r["topology"], t, l) for r in index for t, l, u in zip(r["loop_types"], r["loops"], r["loops_usable"]) if u).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
