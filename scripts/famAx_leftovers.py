#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A extras leftovers: the fragments whose solve is missing and that no live (pending or
running) task of any source array will compute -- the rows of tasks that timed out, failed or were
cancelled. Sources are listed in a TSV (array_id, library, rows, task_offset): each array task i
ran rows [(i + task_offset) * rows, ... + rows) of its library, as slurm_family_a_extras*.sh does.
Writes the leftovers as a new library (frag_id, sequence) for the next pass; the runner skips any
fragment whose npz exists, so a fragment listed twice is never solved twice."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess


def live_tasks(array_id: str) -> set:
    r = subprocess.run(["squeue", "-j", array_id, "-h", "-r", "-o", "%i"], capture_output=True, text=True)
    if r.returncode != 0:
        if "Invalid job id" in r.stderr:
            return set()
        raise SystemExit(f"squeue failed for {array_id}: {r.stderr.strip()}")
    out = set()
    for tok in r.stdout.split():
        base, _, t = tok.partition("_")
        if base != array_id or not t.isdigit():
            raise SystemExit(f"unexpected squeue element {tok!r} for array {array_id}")
        out.add(int(t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", required=True, help="TSV: array_id, library, rows, task_offset")
    ap.add_argument("--solves", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    have = {f[:-4] for f in os.listdir(a.solves) if f.endswith(".npz")}
    seqs, order, covered = {}, [], set()
    for s in csv.DictReader(open(a.sources), delimiter="\t"):
        live, rows, off = live_tasks(s["array_id"]), int(s["rows"]), int(s["task_offset"])
        with open(s["library"]) as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                fid = r["frag_id"]
                if fid not in seqs:
                    seqs[fid] = r["sequence"]; order.append(fid)
                if i // rows - off in live:
                    covered.add(fid)
        print(f"{s['array_id']}: {len(live)} live tasks", flush=True)
    left = [f for f in order if f not in have and f not in covered]
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; leftovers libraries are never overwritten")
    with open(a.out + ".tmp", "w") as f:
        f.write("frag_id\tsequence\n")
        for fid in left:
            f.write(f"{fid}\t{seqs[fid]}\n")
    os.replace(a.out + ".tmp", a.out)
    print(f"fragments {len(order)}, solved {len(have & set(order))}, covered by live tasks "
          f"{len(covered - have)}, leftovers {len(left)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
