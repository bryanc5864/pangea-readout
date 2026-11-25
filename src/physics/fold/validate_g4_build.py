#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Leave-one-out validation of the full-atom quadruplex builder against solved structures.

Each solved parallel three-tetrad structure is rebuilt from its sequence alone: the 1KF1
core (1XAV for 1KF1-sequence targets), loops from other solved structures of the same loop
length with every donor of the target's own sequence excluded, bases substituted, then the
restrained minimisation. Reported per target: heavy-atom RMSD of the core and of each loop
to the solved structure (superposed on the core), before and after minimisation, and the
minimiser's final energy and maximum force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_build as B  # noqa: E402
import g4_templates as T  # noqa: E402


def pick_donors(core, loops_seq, library, seq_of, exclude, rank=0):
    """Per loop, the rank-th donor by anchor RMSD plus a clash penalty against what is placed."""
    placed = np.concatenate([np.array(list(B._heavy(core.res[k]["atoms"]).values()))
                             for col in core.columns for k in col])
    chosen = []
    for i, ls in enumerate(loops_seq):
        cands = B.candidate_loops(core, i, library, len(ls), exclude_sequences=exclude,
                                  seq_of=seq_of, top=12)
        if not cands:
            return None
        scored = []
        for c in cands:
            xyz = np.array([x for r in c[0] for x in B._heavy(r).values()])
            scored.append((c[2] + 0.05 * B._clashes(xyz, placed), c))
        scored.sort(key=lambda t: t[0])
        c = scored[min(rank, len(scored) - 1)][1]
        chosen.append(c)
        placed = np.concatenate([placed, np.array([x for r in c[0] for x in B._heavy(r).values()])])
    return chosen


def model_residues(pdb):
    atoms, _ = T.read_pdb(pdb)
    res = T.residues(atoms)
    keys = sorted(res, key=lambda k: k[1])
    return [B._heavy(res[k]["atoms"]) for k in keys]


def rmsd_by_segment(model_res, target, seg_keys_target, model_idx):
    """Superpose the model on the target's core (C1'/C4'/N9 of the 12 guanines), then RMSD
    per segment over matching heavy-atom names (backbone + common base atoms)."""
    P, Q = [], []
    for mi, tk in zip(model_idx["core"], seg_keys_target["core"]):
        for n in ("C1'", "C4'", "N9", "C8", "N7"):
            if n in model_res[mi] and n in target.res[tk]["atoms"]:
                P.append(model_res[mi][n]); Q.append(target.res[tk]["atoms"][n])
    R, t, core_fit = B.kabsch(np.array(P), np.array(Q))
    out = {"core_fit": round(core_fit, 3)}
    for seg in seg_keys_target:
        X, Y = [], []
        for mi, tk in zip(model_idx[seg], seg_keys_target[seg]):
            ta = B._heavy(target.res[tk]["atoms"])
            for n in ("P", "C4'", "C1'", "O4'", "C3'"):
                if n in model_res[mi] and n in ta:
                    X.append(R @ model_res[mi][n] + t); Y.append(ta[n])
        if X:
            out[seg] = round(float(np.sqrt(np.mean(np.sum((np.array(X) - np.array(Y)) ** 2, 1)))), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--classification", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-targets", type=int, default=1000)
    ap.add_argument("--workroot", default=None)
    a = ap.parse_args()
    rows = json.load(open(a.classification))
    seq_by_id = {r["pdb"]: r.get("sequence", "") for r in rows}
    ids = [r["pdb"] for r in rows if r.get("unimolecular") and r.get("parallel")
           and r["n_tetrads"] == 3 and max(r["loops"]) <= 6]
    lib = []
    for pid in ids:
        try:
            lib.append(B.Structure(os.path.join(a.pdb_dir, pid + ".pdb")))
        except Exception:                                            # noqa: BLE001
            pass
    seq_of = {s.path: seq_by_id[os.path.basename(s.path)[:-4]] for s in lib}
    core_main = next(s for s in lib if os.path.basename(s.path) == "1KF1.pdb")
    core_alt = next(s for s in lib if os.path.basename(s.path) == "1XAV.pdb")
    results, seen = [], set()
    for target in lib:
        tseq = seq_of[target.path]
        if tseq in seen or len(results) >= a.max_targets:
            continue
        if not all(lp["usable"] for lp in target.loops):
            continue
        seen.add(tseq)
        core = core_alt if tseq == seq_of[core_main.path] else core_main
        loops_seq = []
        for lp in target.loops:
            s = "".join(target.res[k]["resname"][-1] for k in lp["residues"])
            loops_seq.append(s.replace("U", "T"))
        if any(set(s) - set("ACGT") for s in loops_seq):
            continue
        donors = pick_donors(core, loops_seq, lib, seq_of, exclude={tseq})
        rec = {"pdb": os.path.basename(target.path)[:-4], "loops": loops_seq,
               "core": os.path.basename(core.path)[:-4]}
        if donors is None:
            rec["status"] = "no donor"; results.append(rec); continue
        rec["donors"] = [(d[3], d[4], round(d[2], 2)) for d in donors]
        work = tempfile.mkdtemp(dir=a.workroot)
        try:
            resl = B.assemble_model(core, loops_seq, donors)
            raw = B.write_model_pdb(resl, os.path.join(work, "raw.pdb"))
            # index bookkeeping: model residue order = col1, loop1, col2, loop2, col3, loop3, col4
            idx = {"core": [], "loop1": [], "loop2": [], "loop3": []}
            tk = {"core": [], "loop1": [], "loop2": [], "loop3": []}
            m = 0
            for c in range(4):
                for k in target.columns[c]:
                    idx["core"].append(m); tk["core"].append(k); m += 1
                if c < 3:
                    for k in target.loops[c]["residues"]:
                        idx[f"loop{c+1}"].append(m); tk[f"loop{c+1}"].append(k); m += 1
            rec["before"] = rmsd_by_segment([r for _, r in resl], target, tk, idx)
            core_res = [i + 1 for i in idx["core"]]
            t0 = time.time()
            rep = B.minimise_model(raw, os.path.join(work, "min.pdb"), core_res)
            rec["minimiser"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rep.items()}
            rec["minimiser"]["seconds"] = round(time.time() - t0, 1)
            rec["after"] = rmsd_by_segment(model_residues(os.path.join(work, "min.pdb")), target, tk, idx)
            rec["status"] = "ok"
        except Exception as exc:                                     # noqa: BLE001
            rec["status"] = f"error: {type(exc).__name__}: {str(exc)[:300]}"
        results.append(rec)
        print(json.dumps(rec), flush=True)
    ok = [r for r in results if r["status"] == "ok"]
    summ = {"n_targets": len(results), "n_ok": len(ok)}
    for when in ("before", "after"):
        for seg in ("core", "loop1", "loop2", "loop3"):
            v = [r[when][seg] for r in ok if seg in r[when]]
            if v:
                summ[f"{when}_{seg}_median_A"] = round(float(np.median(v)), 3)
    json.dump({"summary": summ, "targets": results}, open(a.out, "w"), indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
