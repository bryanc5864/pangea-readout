#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Remaining label work and the training-start estimate. Counts finished units (npz files) per family,
multiplies the units left by measured per-unit costs, converts to slot-hours (slots = max(CPUs,
GB / 13.3), the day nodes' GB per CPU), and divides by the allocation our pangea-* jobs received over
trailing windows (sacct; the week partition excluded, as it admits only jobs of > 24 h). Writes
results/eta_<YYYY-MM-DD_HHMM>.json; +1 day for packing before training."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

G = "/nfs/roberts/scratch/pi_skr2/bjc75/pangea"
GBC = 13.3


def n_npz(d: str) -> int:
    return sum(1 for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else 0


def gb(s: str) -> float:
    s = s.strip().rstrip("n").rstrip("c")
    if not s:
        return 0.0
    u = s[-1].upper()
    v = float(s[:-1]) if u in "KMGT" else float(s)
    return v * {"K": 1 / 1048576, "M": 1 / 1024, "G": 1, "T": 1024}.get(u, 1 / 1073741824)


def allocation(now: dt.datetime) -> dict:
    out = subprocess.run(["sacct", "-u", os.environ.get("USER", "bjc75"), "-S", (now - dt.timedelta(hours=49)).strftime("%Y-%m-%dT%H:%M:%S"),
                          "-E", now.strftime("%Y-%m-%dT%H:%M:%S"), "-X", "-n", "-P", "-o", "JobName,AllocCPUS,ReqMem,Start,End,Partition"],
                         capture_output=True, text=True, check=True).stdout
    res = {}
    for W in (6, 12, 24, 48):
        t0 = now - dt.timedelta(hours=W); c = s = 0.0
        for line in out.splitlines():
            name, cpus, mem, st, en, part = line.split("|")
            if not name.startswith("pangea") or st in ("", "Unknown", "None") or part == "week" or "gpu" in part:
                continue
            a = dt.datetime.fromisoformat(st); b = now if en in ("", "Unknown", "None") else dt.datetime.fromisoformat(en)
            h = (min(b, now) - max(a, t0)).total_seconds() / 3600
            if h <= 0 or int(cpus) == 0:
                continue
            c += h * int(cpus); s += h * max(int(cpus), gb(mem) / GBC)
        res[f"last {W} h"] = {"cpus": round(c / W), "slots": round(s / W)}
    return res


def main() -> int:
    now = dt.datetime.now().replace(second=0, microsecond=0)
    done = {k: n_npz(f"{G}/{d}") for k, d in (("ax", "famA_extras/solves"), ("k", "famK_prod/solves"), ("j", "famJox/solves_v2"),
                                               ("im", "famM_im/solves"), ("tx", "famM_tx/solves"), ("cr", "famM_cr/solves"))}
    # (units left, CPU-seconds per unit, CPUs, GB requested). Sources: A extras 2,025 s per fragment (4,979 fragments,
    # DECISIONS 2026-11-23 18:55), 8 GB since 2026-11-24 11:25; K 95 s per unit (eta_2026-11-23_2230); J 434 s per
    # fragment (34318878's 18 finished tasks: 978,713 s for 2,253 fragments); M solve/i-motif/triplex and DFT as in
    # eta_2026-11-23_2230; N modified-stack DFT and side chains are estimates (no production task has finished).
    items = {
        "A extras": (191344 - done["ax"], 2025, 1, 8),
        "K RNA/hybrid electrostatics": (3 * 191344 - done["k"], 95, 1, 8),
        "J oxDNA2 v2": (191344 - done["j"], 434, 1, 2),
        "M quadruplex solve stage": (73191, 2397, 1, 12),
        "M i-motif": (25868 - done["im"], 2461.5, 1, 16),
        "M triplex": (7021 - done["tx"], 983, 1, 16),
        "G trimer DFT": (int(sys.argv[1]) if len(sys.argv) > 1 else 58, 4.05 * 3600 * 8, 8, 64),
        "N modified-stack DFT (est. ~7 h per shard)": (128, 7 * 3600 * 8, 8, 48),
        "I side chains (est. ~1 h per item, pending the convergence test)": (952, 3600, 1, 3),
    }
    cr_s, cr_gb = (6919 + 8156 + 5463) / 3, 32   # v2 cost test 34489354, stems 5, 8, 12 bp; stem 28 pending

    def tot(ncr: int, scale: float = 1.0):
        c = s = 0.0
        for u, sec, cpu, g in list(items.values()) + [(ncr, cr_s * scale, 1, cr_gb)]:
            h = max(u, 0) * sec / 3600; c += h; s += h / cpu * max(cpu, g / GBC)
        return c, s

    alloc = allocation(now)
    out = {"computed": now.isoformat(), "finished_units": done, "units_left": {k: v[0] for k, v in items.items()},
           "allocation_without_week": alloc, "scopes": {}}
    for scope, ncr, scale in (("cruciforms: stems >= 8 bp (12,559)", 12559 - done["cr"], 1.0),
                              ("cruciforms: stems >= 8 bp, if long stems cost 2x", 12559 - done["cr"], 2.0),
                              ("cruciforms: all (43,607)", 43607 - done["cr"], 1.0),
                              ("(reference) no cruciforms", 0, 1.0)):
        c, s = tot(ncr, scale); eta = {}
        for wn, a in alloc.items():
            d = s / a["slots"] / 24.0
            eta[f"{wn} ({a['cpus']} CPUs, {a['slots']} slots)"] = {"days_labels": round(d, 1),
                                                                   "training_start": (now + dt.timedelta(days=d + 1)).strftime("%b %d")}
        out["scopes"][scope] = {"cpu_h": round(c), "slot_h": round(s), "eta": eta}
    out["note"] = ("CPU partitions only (GPU stages run alongside). Not included: pKa production (test pending). "
                   "+1 day packing before training.")
    path = f"results/eta_{now.strftime('%Y-%m-%d_%H%M')}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(path); print("finished", done)
    for k, v in out["scopes"].items():
        print(f"{k}: {v['cpu_h']:,} CPU-h, {v['slot_h']:,} slot-h")
        for wn, e in v["eta"].items():
            print("    ", wn, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
