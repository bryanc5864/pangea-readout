# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Aim 2 summary: the three arms, the bottleneck sweep and the decoder, per dataset.

Reads results/readout/*.json and prints, per dataset and encoder: test metric of seq,
phys, seq+phys; the fraction of the sequence model's score the physics head reaches; the
physics bottleneck curve (score against k); and the decoder's per-base accuracy and bits
recovered. Where a sourced assay ceiling exists (`CEILINGS`, from DECISIONS.md), the
fractions of the ceiling each arm reaches are reported as well -- all three numbers, as
the plan asks, never the ratio alone.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

# explainable-variance ceilings (R2 scale) with their source; see DECISIONS.md
CEILINGS = {"loopseq": (0.869, "Basu et al. 2021 Ext. Data Fig. 3c, technical replicates r = 0.869, "
                                "nucleosomal library; used as the explainable-variance fraction")}


def main() -> int:
    rows = defaultdict(dict)
    for f in sorted(glob.glob("results/readout/*.json")):
        d = json.load(open(f))
        enc = os.path.basename(str(d.get("encoder", "")).rstrip("/"))
        tag = d.get("tag") or ""
        rows[(d["dataset"] + (f"[{tag}]" if tag else ""), enc)][d["arm"]] = d
    out = {}
    for (ds, enc), arms in sorted(rows.items()):
        key = "r2" if "seq" in arms and "r2" in arms["seq"]["test"] else "accuracy"
        get = lambda arm: arms[arm]["test"].get(key) if arm in arms else None   # noqa: E731
        s, p, sp = get("seq"), get("phys"), get("seq+phys")
        rec = {"metric": key, "seq": s, "phys": p, "seq+phys": sp,
               "phys_over_seq": (p / s) if (s and p is not None) else None,
               "sweep": {int(a.split(":")[1]): arms[a]["test"].get(key)
                         for a in arms if a.startswith("phys:")},
               "n_train": arms[next(iter(arms))]["n_train"],
               "n_test": arms[next(iter(arms))]["n_test"]}
        if ds.split("[")[0] in CEILINGS and key == "r2":
            c, src = CEILINGS[ds.split("[")[0]]
            rec["ceiling"] = {"value": c, "source": src,
                              "seq_fraction": s / c if s is not None else None,
                              "phys_fraction": p / c if p is not None else None}
        dec = {a: arms[a]["test"] for a in arms if a.startswith("decode")}
        if dec:
            rec["decoder"] = dec
        by = {a: arms[a].get("test_by_library") or arms[a].get("test_by_subset")
              for a in ("seq", "phys", "seq+phys") if a in arms}
        if any(by.values()):
            rec["by_group"] = by
        out[f"{ds}::{enc}"] = rec
        fmt = lambda v: "-" if v is None else f"{v:.4f}"                 # noqa: E731
        print(f"\n{ds} (encoder {enc}, test n={rec['n_test']}, metric {key})")
        print(f"   seq {fmt(s)} | phys {fmt(p)} | seq+phys {fmt(sp)} | phys/seq {fmt(rec['phys_over_seq'])}")
        if rec["sweep"]:
            print("   bottleneck k -> " + ", ".join(f"{k}: {fmt(v)}" for k, v in sorted(rec["sweep"].items())))
        if "ceiling" in rec:
            c = rec["ceiling"]
            print(f"   ceiling {c['value']:.3f}: seq reaches {fmt(c['seq_fraction'])}, phys {fmt(c['phys_fraction'])}")
        for a, t in sorted(dec.items()):
            print(f"   {a}: per-base accuracy {t['per_base_accuracy']:.4f}, bits recovered "
                  f"{t['bits_recovered']:.3f} of {t['entropy_bits']:.3f}")
        # Rate-distortion: each physics representation's readout score beside the bits of
        # sequence its decoder recovers (the information it could have used).
        rd = []
        for arm, d in arms.items():
            if arm.startswith(("phys", "global")):
                dec = ("decode" + arm[4:]) if arm.startswith("phys") else "decodeglobal"
                if arm.startswith("physsel:"):
                    dec = "decodesel:" + arm.split(":", 1)[1]
                bits = arms[dec]["test"]["bits_recovered"] if dec in arms else None
                rd.append({"representation": arm, "score": d["test"].get(key), "bits_per_base": bits})
        if rd:
            rec["rate_distortion"] = sorted(rd, key=lambda r: (r["bits_per_base"] is None,
                                                               -(r["bits_per_base"] or 0)))
            print("   rate-distortion (representation: bits recovered per base -> score):")
            for r in rec["rate_distortion"]:
                b = "  n/a" if r["bits_per_base"] is None else f"{r['bits_per_base']:.3f}"
                print(f"      {r['representation']:16s} {b} -> {fmt(r['score'])}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/readout_summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
