#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Side shards of the channels that are exact functions of sequence through per-context libraries
or of channels already in the main shards: Gmd / Kmd (explicit-solvent MD hydration), Gdk (docking),
Bt (base-pair trimers) via `context_libraries`, and Gr (hydrolytic damage rates) via
`reaction_rates` from the main shards' E_open_prob_T37. One side shard per base shard, same
fragments in the same order, the main shards' metadata schema."""
from __future__ import annotations

import argparse
import glob
import json
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
import context_libraries as CL  # noqa: E402
import reaction_rates as RR  # noqa: E402
import rloop_energetics as FR  # noqa: E402
import k_electrostatics as KE  # noqa: E402
import a_extras_channels as AX  # noqa: E402
import sidechain_channels as SC  # noqa: E402
import modification_electrostatics as NE  # noqa: E402
import landscape as LS  # noqa: E402
from channels import Resolution  # noqa: E402
sys.path.insert(0, HERE)
from generate_targets import channel_metadata  # noqa: E402


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None, help="md_hydration folder")
    ap.add_argument("--dock", default=None, help="dock/scores folder")
    ap.add_argument("--trimers", default=None, help="qc/trimer_tzvp folder")
    ap.add_argument("--rates", action="store_true", help="Gr from the shards' E_open_prob_T37")
    ap.add_argument("--rloop", action="store_true", help="Fr (R-loop free energies) from the sequence")
    ap.add_argument("--kes", default=None, help="famK_prod/solves folder: Kes (Family K A-form electrostatics)")
    ap.add_argument("--aextras", default=None, help="famA_extras/solves folder: Ax (opening and dielectric response)")
    ap.add_argument("--isc", default=None, help="isc/scores folder: Isc (Family I side-chain groove energies; held apart)")
    ap.add_argument("--nes", default=None, help="Family N modification-electrostatics table folder: Nes")
    ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    fams = (["Gmd", "Kmd"] if a.md else []) + (["Gdk"] if a.dock else []) + (["Bt"] if a.trimers else []) + (["Gr"] if a.rates else []) \
        + (["Fr"] if a.rloop else []) + (["Kes"] if a.kes else []) + (["Ax"] if a.aextras else []) + (["Isc"] if a.isc else []) + (["Nes"] if a.nes else [])
    pb = LS.channel_names(Resolution.PER_BP, fams)
    gn = LS.channel_names(Resolution.GLOBAL, fams)
    meta = channel_metadata(fams)
    md = {c: CL.load_md(a.md, c) for c in ("dna", "rna", "hyb")} if a.md else {}
    dock = CL.load_dock(a.dock) if a.dock else {}
    tri = CL.load_trimers(a.trimers) if a.trimers else {}
    isc = SC.load_table(a.isc) if a.isc else {}
    nes = NE.load_table(a.nes) if a.nes else {}
    if a.isc:
        lack = SC.missing_entries(isc)
        if lack:
            raise SystemExit(f"Isc table incomplete: {len(lack)} (context, channel) values missing, e.g. {lack[:3]}")
    if a.nes:
        want = {(m, pt, c) for m, b, _ in NE.MODS for pt in NE.PARTNERS for c in NE.contexts(b)}
        lack = sorted(want - set(nes))
        if lack:
            raise SystemExit(f"Nes table incomplete: {len(lack)} of {len(want)} entries missing, e.g. {lack[:3]}")
    print("libraries:", {k: len(v) for k, v in md.items()}, "dock", len(dock), "trimers", len(tri), "side chains", len(isc), flush=True)
    os.makedirs(a.out, exist_ok=True)
    missing = []
    for p in sorted(glob.glob(a.shards))[a.task::a.n_tasks]:
        with h5py.File(p, "r") as h:
            seqs = _decode(h["sequence"][:]); ids = _decode(h["frag_id"][:])
            names = _decode(h.attrs["per_bp_names"]) if a.rates else []
            iop = names.index("E_open_prob_T37") if a.rates else None
            popen = h["per_bp"][:, iop, :] if a.rates else None
        n, L = len(seqs), len(seqs[0])
        P = np.full((n, len(pb), L), np.nan, dtype=np.float32)
        Gm = np.full((n, len(gn)), np.nan, dtype=np.float32)
        for i, s in enumerate(seqs):
            prof, glob_ = {}, {}
            if md:
                prof.update(CL.md_channels(s, md["dna"], "Gmd"))
                prof.update(CL.md_channels(s, md["rna"], "Kmd_r", with_o2p=True))
                prof.update(CL.md_channels(s, md["hyb"], "Kmd_h", with_o2p=True))
            if dock:
                prof.update(CL.dock_channels(s, dock))
            if tri:
                prof.update(CL.trimer_channels(s, tri))
            if a.rates:
                prof.update(RR.rate_channels(s, popen[i]))
            if a.rloop:
                pr_, gl_ = FR.rloop_fragment_channels(s)
                prof.update(pr_); glob_.update(gl_)
            if isc:
                prof.update(SC.sidechain_channels(s, isc))
            if nes:
                prof.update(NE.es_channels(s, nes))
            if a.aextras:
                f = os.path.join(a.aextras, f"{ids[i]}.npz")
                if os.path.exists(f):
                    prof.update(AX.extras_channels(f, n_bp=L))
                else:
                    missing.append(ids[i])
            if a.kes:
                for tag in KE.TAGS:
                    f = os.path.join(a.kes, f"{ids[i]}__{tag}.npz")
                    if not os.path.exists(f):
                        missing.append(f"{ids[i]}__{tag}"); continue
                    pr_, gl_ = KE.assemble_k(dict(np.load(f, allow_pickle=True)), tag, n_bp=L)
                    prof.update(pr_); glob_.update(gl_)
            for c, nm in enumerate(pb):
                if nm in prof:
                    P[i, c] = prof[nm]
            for c, nm in enumerate(gn):
                if nm in glob_:
                    Gm[i, c] = glob_[nm]
        out = os.path.join(a.out, os.path.basename(p))
        with h5py.File(out + ".tmp", "w") as h:
            h.create_dataset("per_bp", data=P, compression="gzip", compression_opts=4, chunks=True)
            h.create_dataset("global", data=Gm)
            dt = h5py.string_dtype()
            h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=dt)
            h.create_dataset("frag_id", data=np.array(ids, dtype=object), dtype=dt)
            for k, v in meta.items():
                if not isinstance(v, list):
                    continue
                if v and isinstance(v[0], (bool, np.bool_)):
                    h.attrs[k] = np.array(v, dtype=bool)
                elif v and isinstance(v[0], (int, np.integer)) and not isinstance(v[0], bool):
                    h.attrs[k] = np.array(v, dtype=np.int64)
                else:
                    h.attrs.create(k, [str(x) for x in v], dtype=h5py.string_dtype())
            h.attrs["seq_sha1"] = hashlib.sha1("\n".join(seqs).encode()).hexdigest()
            h.attrs["families"] = ",".join(fams); h.attrs["main_shard"] = os.path.abspath(p)
        os.replace(out + ".tmp", out)
        print(os.path.basename(p), "packed", flush=True)
    if missing:
        # a missing solve is a hole in the labels: listed, and the task fails so it is not mistaken for done
        json.dump(missing, open(os.path.join(a.out, f"missing_task{a.task}.json"), "w"))
        print(f"{len(missing)} solves missing (listed in missing_task{a.task}.json)", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
