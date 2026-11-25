#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Collect the QC tier shards into one table keyed by tetramer context (`qc_channels.py`).

Keys follow `qc_channels`: stack_<g>, stack_sapt_<c>_<g>, hb_bp1_<g>, hb_bp2_<g>,
hb_bp1_sapt_<c>_<g>, hb_bp2_sapt_<c>_<g>, homo/lumo/gap/dipole_<g>, vip, vea, for the
geometries gs (ground), sp, sm (+-1 sigma along the softest mode). A context missing from a
geometry, or failed there, simply lacks those keys (NaN downstream).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

GEOMS = (("gs", "dft_tetramer"), ("sp", "dft_tetramer_soft_plus"), ("sm", "dft_tetramer_soft_minus"))
SAPT = ("total", "elst", "exch", "ind", "disp")


COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
GROOVE_KEYS = ("maj_pur_N7", "maj_pur_6", "maj_pyr_4", "min_pur_N3", "min_pyr_O2", "min_G_N2")


def stage_b(t, r, put):
    """Stage B record -> table keys: purine/pyrimidine per base pair, max/min over strands per step."""
    if "t_stage_b_s" not in r:
        return
    ctx = r["context"]
    for bp, (wt, ct, wl) in (("bp1", ("w1", "c1", ctx[1])), ("bp2", ("w2", "c2", ctx[2]))):
        pur, pyr = (wt, ct) if wl in "AG" else (ct, wt)
        for w in ("hole", "elec"):
            put(f"site_{w}_pur_{bp}", r.get(f"site_{w}_{pur}_eV"))
            put(f"site_{w}_pyr_{bp}", r.get(f"site_{w}_{pyr}_eV"))
            put(f"t_{w}_pair_{bp}", r.get(f"t_{w}_pair_{bp}_eV"))
        put(f"mbis_net_pur_{bp}", r.get(f"mbis_net_{pur}_e"))
        for k in GROOVE_KEYS:
            put(f"mep_{k}_{bp}", r.get(f"mep_{k}_{bp}_kcal_mol_e"))
            put(f"mbis_{k}_{bp}", r.get(f"mbis_{k}_{bp}_e"))
    for w in ("hole", "elec"):
        a, b = r.get(f"t_{w}_intraW_eV"), r.get(f"t_{w}_intraC_eV")
        if a is not None and b is not None:
            put(f"t_{w}_intra_max", max(a, b)); put(f"t_{w}_intra_min", min(a, b))
        a, b = r.get(f"t_{w}_inter_w1c2_eV"), r.get(f"t_{w}_inter_c1w2_eV")
        if a is not None and b is not None:
            put(f"t_{w}_inter_max", max(a, b)); put(f"t_{w}_inter_min", min(a, b))
    for k in ("iso", "aniso", "axial"):
        put(f"polar_{k}", r.get(f"polar_{k}_au"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/nfs/roberts/scratch/pi_skr2/bjc75/pangea/qc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tab, report = {}, {}
    for g, d in GEOMS:
        n_ok = n_fail = 0
        for f in sorted(glob.glob(os.path.join(a.root, d, "shard_*.json"))):
            for r in json.load(open(f)).get("results", []):
                if r.get("status") != "ok":
                    n_fail += 1
                    continue
                n_ok += 1
                t = tab.setdefault(r["context"], {})
                put = lambda k, v: t.__setitem__(k, float(v)) if v is not None else None  # noqa: E731
                put(f"stack_{g}", r.get("stacking_kcal_mol"))
                put(f"hb_bp1_{g}", r.get("hbond_bp1_kcal_mol"))
                put(f"hb_bp2_{g}", r.get("hbond_bp2_kcal_mol"))
                for c in SAPT:
                    put(f"stack_sapt_{c}_{g}", r.get(f"sapt0_{c}_kcal_mol"))
                    put(f"hb_bp1_sapt_{c}_{g}", r.get(f"sapt0_hb_bp1_{c}_kcal_mol"))
                    put(f"hb_bp2_sapt_{c}_{g}", r.get(f"sapt0_hb_bp2_{c}_kcal_mol"))
                for q in ("homo", "lumo", "gap"):
                    put(f"{q}_{g}", r.get(f"dimer_{q}_eV"))
                put(f"dipole_{g}", r.get("dimer_dipole_debye"))
                if g == "gs":
                    put("vip", r.get("dimer_vip_eV")); put("vea", r.get("dimer_vea_eV"))
                    stage_b(t, r, put)
        report[g] = {"ok": n_ok, "failed": n_fail}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"contexts": tab, "report": report}, open(a.out, "w"), indent=1)
    print(json.dumps({"contexts": len(tab), "by_geometry": report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
