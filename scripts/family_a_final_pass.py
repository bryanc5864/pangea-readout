#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A final pass: list what the production arrays left undone and write resume manifests.

For each (manifest, output folder) pair: rows whose .npz is missing (never reached, killed by a
time or memory limit, or failed) go to `<name>_resume.tsv`, all their conditions; rows whose
.npz exists but whose record lists failed conditions go to `<name>_fix.tsv` with those
conditions only (solved into a separate folder; the long-shard merge fills a condition's NaNs
from later passes).
"""
import csv, json, os, sys

def main(pairs):
    for name, manifest, outdir in pairs:
        rows = list(csv.DictReader(open(manifest), delimiter="\t"))
        cols = list(rows[0].keys())
        if name == "ensemble":                     # one solve per snapshot: <fid>__s<k>.npz
            for k in range(1, 9):
                miss = [r for r in rows if not os.path.exists(os.path.join(outdir, f"{r['frag_id']}__s{k}.npz"))]
                out = os.path.join(os.path.dirname(manifest), f"ensemble_resume_s{k}.tsv")
                with open(out, "w") as f:
                    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
                    w.writeheader(); w.writerows(miss)
                print(f"ensemble snapshot {k}: {len(miss)} rows -> {out}", flush=True)
            continue
        resume, fix = [], []
        for r in rows:
            fid = r["frag_id"]
            npz, js = os.path.join(outdir, fid + ".npz"), os.path.join(outdir, fid + ".json")
            if not os.path.exists(npz):
                resume.append(r); continue
            if os.path.exists(js):
                try:
                    fc = json.load(open(js)).get("failed_conditions") or {}
                except Exception:
                    fc = {}
                if fc:
                    keep = [c for c in r["conditions"].split(",") if c in fc]
                    if keep:
                        fix.append(dict(r, conditions=",".join(keep)))
        base = os.path.join(os.path.dirname(manifest), name)
        for tag, lst in (("resume", resume), ("fix", fix)):
            with open(f"{base}_{tag}.tsv", "w") as f:
                w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
                w.writeheader(); w.writerows(lst)
            print(f"{name}: {len(lst)} rows -> {base}_{tag}.tsv", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]
    main([tuple(a[i:i + 3]) for i in range(0, len(a), 3)])
