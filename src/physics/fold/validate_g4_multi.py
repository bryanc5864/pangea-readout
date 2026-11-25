#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Leave-one-out validation of the multi-topology quadruplex builder (`g4_multi`).

Every solved unimolecular structure of the template set v2 with usable loops is rebuilt from
its own sequence with neither the core nor any donor loop taken from it or from a structure of
identical sequence; the model (assembled, then minimised as in production) is superposed on the
target's core and the RMSD of the core and of each loop reported, by topology and tetrad count.
"""
import argparse, json, os, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_build as B          # noqa: E402
import g4_multi as GM         # noqa: E402
import validate_g4_build as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--templates", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default=None)
    ap.add_argument("--topologies", nargs="+", default=list(GM.TOPOLOGIES))
    a = ap.parse_args()
    lib = GM.Library(a.templates)
    results, seen = [], set()
    for target in lib.structs:
        if target.topology not in a.topologies or target.n_tetrads not in (2, 3):
            continue
        if target.topology == "parallel" and target.n_tetrads == 3:
            continue                                   # validated separately (g4_build_validation)
        key = (target.sequence, target.topology, target.n_tetrads)
        if key in seen or not all(lp["usable"] for lp in target.loops):
            continue
        seen.add(key)
        loops_seq = ["".join(target.res[k]["resname"].strip()[-1] for k in lp["residues"]).replace("U", "T")
                     for lp in target.loops]
        rec = {"pdb": os.path.basename(target.path)[:-4], "topology": target.topology,
               "n_tetrads": target.n_tetrads, "loops": loops_seq,
               "loop_types": [lp.get("type") for lp in target.loops]}
        if any(set(s) - set("ACGT") or not s for s in loops_seq):
            rec["status"] = "non-standard loop"; results.append(rec); continue
        plan = GM.plan_models(loops_seq, target.topology, target.n_tetrads, lib, n_models=3,
                              exclude_sequences={target.sequence})
        rec["fold"] = getattr(target, "fold", "")
        if not plan:
            rec["status"] = "no feasible core"; results.append(rec); print(json.dumps(rec), flush=True); continue
        rec["models"] = []
        for core, donors, score in plan:
            m_rec = {"core": os.path.basename(core.path)[:-4], "core_fold": core.fold, "score": round(score, 2),
                     "donors": [(d[3], d[4], round(d[2], 2)) for d in donors]}
            work = tempfile.mkdtemp(dir=a.workroot)
            try:
                resl = B.assemble_model(core, loops_seq, donors)
                raw = B.write_model_pdb(resl, os.path.join(work, "raw.pdb"))
                idx = {"core": [], "loop1": [], "loop2": [], "loop3": []}
                tk = {"core": [], "loop1": [], "loop2": [], "loop3": []}
                m = 0
                for c in range(4):
                    for k in target.columns[c]:
                        idx["core"].append(m); tk["core"].append(k); m += 1
                    if c < 3:
                        for k in target.loops[c]["residues"]:
                            idx[f"loop{c+1}"].append(m); tk[f"loop{c+1}"].append(k); m += 1
                core_res = [i + 1 for i in idx["core"]]
                B.minimise_model(raw, os.path.join(work, "min.pdb"), core_res)
                m_rec["after"] = V.rmsd_by_segment(V.model_residues(os.path.join(work, "min.pdb")), target, tk, idx)
                m_rec["status"] = "ok"
            except Exception as exc:                                 # noqa: BLE001
                m_rec["status"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
            rec["models"].append(m_rec)
        ok_m = [m for m in rec["models"] if m["status"] == "ok"]
        rec["status"] = "ok" if ok_m else "error"
        if ok_m:
            best = min(ok_m, key=lambda m: m["after"]["core"])
            rec["best"] = best["after"]; rec["best_core_fold"] = best["core_fold"]
            rec["first"] = ok_m[0]["after"]
            rec["fold_in_ensemble"] = any(m["core_fold"] == rec["fold"] for m in ok_m)
        results.append(rec)
        print(json.dumps(rec), flush=True)
    summ = {}
    for topo in a.topologies:
        for nt in (2, 3):
            sel = [r for r in results if r["topology"] == topo and r["n_tetrads"] == nt]
            ok = [r for r in sel if r["status"] == "ok"]
            if not sel:
                continue
            s = {"n_targets": len(sel), "n_ok": len(ok),
                 "no_feasible_core": sum(r["status"] == "no feasible core" for r in sel),
                 "fold_in_ensemble": sum(bool(r.get("fold_in_ensemble")) for r in ok)}
            for when in ("first", "best"):
                v = [r[when]["core"] for r in ok]
                if v:
                    s[f"{when}_model_core_median_A"] = round(float(np.median(v)), 3)
                    s[f"{when}_model_core_below_2p5A"] = int(sum(x < 2.5 for x in v))
                by = {}
                for r in ok:
                    for i in range(3):
                        val = r[when].get(f"loop{i+1}")
                        if val is not None:
                            by.setdefault(f"{r['loop_types'][i]}_{len(r['loops'][i])}", []).append(val)
                s[f"{when}_model_loop_median_A"] = {k: [len(v), round(float(np.median(v)), 2)] for k, v in sorted(by.items())}
            summ[f"{topo}_{nt}"] = s
    json.dump({"summary": summ, "targets": results}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
