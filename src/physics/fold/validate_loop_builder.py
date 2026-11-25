#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Validation of `loop_builder`: solved quadruplex loops removed and rebuilt from sequence alone,
the rest of the structure as solved. Reports per loop the backbone RMSD to the solved loop (after
superposition on the core), the largest O3'-P bond, and the closest heavy-atom contact between the
rebuilt loop and the rest (a clash or a knot through the core would show as < 2 A)."""
import argparse, json, os, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_build as B         # noqa: E402
import g4_multi as GM        # noqa: E402
import g4_templates as T     # noqa: E402
import loop_builder as LB    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--templates", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-length", type=int, default=3); ap.add_argument("--max-targets", type=int, default=14)
    ap.add_argument("--workroot", default=None)
    a = ap.parse_args()
    lib = GM.Library(a.templates)
    picks, seen = [], set()
    for s in sorted(lib.structs, key=lambda s: -max(len(l["residues"]) for l in s.loops)):
        for j, lp in enumerate(s.loops):
            L = len(lp["residues"])
            if not lp["usable"] or L < a.min_length or (L, lp.get("type")) in seen:
                continue
            seen.add((L, lp.get("type"))); picks.append((s, j))
    picks = picks[: a.max_targets]
    out = []
    for s, j in picks:
        keys = sorted(s.res, key=lambda k: k[1])
        loop = list(s.loops[j]["residues"])
        known = [(s.res[k]["resname"].strip(), B._heavy(s.res[k]["atoms"])) for k in keys if k not in loop]
        first = keys.index(loop[0])
        missing = {first: [s.res[k]["resname"].strip() for k in loop]}
        work = tempfile.mkdtemp(dir=a.workroot)
        rec = {"pdb": os.path.basename(s.path)[:-4], "loop": j, "length": len(loop), "type": s.loops[j].get("type")}
        try:
            gp = LB.write_gapped_pdb(known, missing, os.path.join(work, "gap.pdb"))
            known_num = [i + 1 for i in range(len(keys)) if keys[i] not in loop]
            t = time.time()
            LB.fill_and_relax(gp, os.path.join(work, "fill.pdb"), known_num)
            rec["seconds"] = round(time.time() - t, 1)
            atoms, _ = T.read_pdb(os.path.join(work, "fill.pdb"))
            res = T.residues(atoms)
            by_num = {k[1]: B._heavy(r["atoms"]) for k, r in res.items()}
            core_keys = [k for col in s.columns for k in col]
            P, Q = [], []
            for k in core_keys:
                i = keys.index(k) + 1
                for n in ("C1'", "C4'", "N9"):
                    if n in by_num.get(i, {}) and n in s.res[k]["atoms"]:
                        P.append(by_num[i][n]); Q.append(s.res[k]["atoms"][n])
            R, tt, _ = B.kabsch(np.array(P), np.array(Q))
            X, Y = [], []
            for k in loop:
                i = keys.index(k) + 1
                for n in ("P", "C4'", "C1'", "O4'", "C3'"):
                    if n in by_num.get(i, {}) and n in s.res[k]["atoms"]:
                        X.append(R @ by_num[i][n] + tt); Y.append(s.res[k]["atoms"][n])
            rec["loop_rmsd_A"] = round(float(np.sqrt(np.mean(np.sum((np.array(X) - np.array(Y)) ** 2, 1)))), 2)
            nums = [keys.index(k) + 1 for k in loop]
            o3p = []
            for i in range(min(nums) - 1, max(nums) + 1):
                if "O3'" in by_num.get(i, {}) and "P" in by_num.get(i + 1, {}):
                    o3p.append(float(np.linalg.norm(by_num[i]["O3'"] - by_num[i + 1]["P"])))
            rec["o3p_max_A"] = round(max(o3p), 3) if o3p else None
            lx = np.array([x for i in nums for x in by_num[i].values()])
            rest = np.array([x for i, at in by_num.items() if i not in nums and abs(min(abs(i - m) for m in nums)) > 1 for x in at.values()])
            rec["min_contact_A"] = round(float(np.min(np.linalg.norm(lx[:, None] - rest[None], axis=-1))), 2)
            rec["status"] = "ok"
        except Exception as exc:                                     # noqa: BLE001
            rec["status"] = f"error: {type(exc).__name__}: {str(exc)[:300]}"
        out.append(rec); print(json.dumps(rec), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
