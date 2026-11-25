#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A long-window shards for `data.LongWindowSet`, aligned to the main corpus.

Each fragment's Family A solves were run on its long window (`extend_windows.py`; the
manifest's sequence column), so the solver's scored 21 bp are the window's centre, which
is the fragment's centre. Per main shard this writes one long shard with the windows, the
fragments (for the alignment check), the per-bp channels on the 21 central positions, the
whole-window channels, and the window geometry. Fragments without a solve are NaN (they
are skipped when long windows are sampled); a solve whose window differs from the windows
file is refused.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "physics"))
sys.path.insert(0, HERE)
import electrostatics_family as A  # noqa: E402
import landscape as LS  # noqa: E402
from channels import Resolution  # noqa: E402
from generate_targets import channel_metadata  # noqa: E402
from generate_family_a_shards import find_solves  # noqa: E402

SCORED = A.SCORED


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


# solver-output arrays without a leading condition axis (geometry, sampling layout)
NO_COND_KEYS = {"surface_offsets_A", "angles", "radii", "bp_origins", "sequence",
                "ext_manning_xi", "ext_surface_charge_e_per_A2"}
MAX_STRUCTURE_RMS_A = 0.05
# per-phosphate arrays: their length is the solute's phosphate count, which differs between passes (the
# 10 mM pass solves the 121 bp window, the higher-salt passes a trimmed 61 bp solute); rows of another
# length are NaN, the rule run_fragment applies within one file. No Family A channel reads them.
PER_PHOSPHATE_KEYS = {"counterion"}


def find_all_solves(dirs):
    """frag_id -> [npz paths] across directories (a fragment solved in passes: the main
    production and a supplementary condition pass)."""
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.npz")):
            out.setdefault(os.path.basename(p)[:-4], []).append(p)
    return out


def load_merged(paths):
    """One solver record from several files of the same fragment: conditions concatenated
    (a condition already present is kept from the first file), arrays a file lacks padded
    with NaN, and the structures required to agree (base-pair origins within
    MAX_STRUCTURE_RMS_A), since every condition must describe the same conformation."""
    zs = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        zs.append({k: z[k] for k in z.files})
    base = zs[0]
    for other in zs[1:]:
        if str(other["sequence"][0]) != str(base["sequence"][0]):
            raise ValueError("solve files of one fragment carry different sequences")
        d = np.asarray(other["bp_origins"], float) - np.asarray(base["bp_origins"], float)
        rms = float(np.sqrt(np.mean(np.sum(d ** 2, -1))))
        if rms > MAX_STRUCTURE_RMS_A:
            raise ValueError(f"structures differ between solve files ({rms:.3f} A rms)")
        bc, oc = [str(c) for c in base["conditions"]], [str(c) for c in other["conditions"]]
        # a condition in both files: fill what the first lacks (e.g. solvation re-solved in a
        # supplementary pass after APBS's energy evaluation had failed in the first)
        for j, c in enumerate(oc):
            if c not in set(bc):
                continue
            i = bc.index(c)
            for k in (set(base) & set(other)) - NO_COND_KEYS - {"conditions"}:
                vb, vo = base[k], other[k]
                if (vb.ndim >= 1 and vo.ndim >= 1 and vb.shape[0] == len(bc) and vo.shape[0] == len(oc)
                        and vb.shape[1:] == vo.shape[1:] and np.issubdtype(vb.dtype, np.floating)):
                    vb = vb.copy(); vb[i] = np.where(np.isnan(vb[i]), vo[j], vb[i]); base[k] = vb
            for k in set(other) - set(base) - NO_COND_KEYS - {"conditions"}:
                vo = other[k]
                if vo.ndim >= 1 and vo.shape[0] == len(oc):
                    vb = np.full((len(bc),) + vo.shape[1:], np.nan); vb[i] = vo[j]; base[k] = vb
        new = [j for j, c in enumerate(oc) if c not in set(bc)]
        if not new:
            continue
        nb, nn = len(bc), len(new)
        keys = (set(base) | set(other)) - NO_COND_KEYS - {"conditions"}
        for k in keys:
            vb, vo = base.get(k), other.get(k)
            if vo is not None and vo.ndim >= 1 and vo.shape[0] == len(oc):
                add = vo[new]
            elif vb is not None and vb.ndim >= 1 and vb.shape[0] == nb:
                add = np.full((nn,) + vb.shape[1:], np.nan)
            else:
                continue
            if vb is None or vb.ndim < 1 or vb.shape[0] != nb:
                vb = np.full((nb,) + add.shape[1:], np.nan)
            if add.shape[1:] != vb.shape[1:]:
                if k not in PER_PHOSPHATE_KEYS:
                    raise ValueError(f"{k}: per-condition shape {vb.shape[1:]} in one solve file, {add.shape[1:]} in another")
                add = np.full((nn,) + vb.shape[1:], np.nan)
            base[k] = np.concatenate([vb.astype(float), add.astype(float)], axis=0)
        base["conditions"] = np.array(bc + [oc[j] for j in new])
    return base


def one_shard(path, out_dir, solves, windows, ensemble=None):
    import h5py
    with h5py.File(path, "r") as h:
        seqs, ids = _decode(h["sequence"][:]), _decode(h["frag_id"][:])
    pb = LS.channel_names(Resolution.PER_BP, ["A"]); gl = LS.channel_names(Resolution.GLOBAL, ["A"])
    wins = [windows[f] for f in ids]
    W = len(wins[0]); off = (W - SCORED) // 2
    P = np.full((len(ids), len(pb), SCORED), np.nan, np.float32)
    G = np.full((len(ids), len(gl)), np.nan, np.float32)
    n_have, n_ens, fails = 0, 0, []
    for i, (fid, w) in enumerate(zip(ids, wins)):
        p = solves.get(fid)
        if not p:
            continue
        try:
            z = load_merged(p)
            if str(z["sequence"][0]) != w:
                raise ValueError("solve sequence differs from the window")
            prof, g = A.assemble(z, n_bp=SCORED)      # exactly the 21 scored positions
            snaps = (ensemble or {}).get(fid, [])
            if snaps:
                zs = []
                for q in snaps:
                    zq = np.load(q, allow_pickle=False); zq = {k: zq[k] for k in zq.files}
                    if str(zq["sequence"][0]) != w:
                        raise ValueError(f"snapshot {os.path.basename(q)} differs from the window")
                    zs.append(zq)
                ep, eg = A.assemble_ensemble(zs, n_bp=SCORED)
                prof.update(ep); g.update(eg); n_ens += int(len(zs) >= A.MIN_SNAPSHOTS)
            P[i] = np.stack([prof[n] for n in pb]); G[i] = [g[n] for n in gl]; n_have += 1
        except Exception as exc:                                              # noqa: BLE001
            fails.append({"frag_id": fid, "error": f"{type(exc).__name__}: {exc}"[:200]})
    meta = channel_metadata(["A"]); sel = {}
    for key, names in (("per_bp", pb), ("global", gl)):
        alln = list(meta[f"{key}_names"]); idx = [alln.index(n) for n in names]
        for k, v in meta.items():
            if k.startswith(key + "_") and isinstance(v, list) and len(v) == len(alln):
                sel[k] = [v[j] for j in idx]
    out = os.path.join(out_dir, os.path.basename(path)); tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        st = h5py.string_dtype()
        h.create_dataset("window", data=np.array(wins, dtype=object), dtype=st)
        h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=st)
        h.create_dataset("frag_id", data=np.array(ids, dtype=object), dtype=st)
        h.create_dataset("per_bp_central", data=P, compression="gzip", compression_opts=4)
        h.create_dataset("global", data=G, compression="gzip", compression_opts=4)
        for k, v in sel.items():
            if v and isinstance(v[0], (bool, np.bool_)):
                h.attrs[k] = np.array(v, dtype=bool)
            elif v and isinstance(v[0], (int, np.integer)) and not isinstance(v[0], bool):
                h.attrs[k] = np.array(v, dtype=np.int64)
            else:
                h.attrs[k] = np.array([str(x) for x in v], dtype=object)
        h.attrs["center_offset"] = off; h.attrs["window_length"] = W
        h.attrs["n_solved"] = n_have; h.attrs["n_failures"] = len(fails); h.attrs["n_ensemble"] = n_ens
        h.attrs["failures_json"] = json.dumps(fails[:50]); h.attrs["main_shard"] = path
    os.replace(tmp, out)
    return (f"wrote {out}: {n_have}/{len(ids)} windows with solves ({n_ens} with an ensemble), "
            f"{len(fails)} refused")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True); ap.add_argument("--windows", required=True)
    ap.add_argument("--solves", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", default=None, help="directory of snapshot solves <frag_id>__s<k>.npz")
    ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    windows = {r["frag_id"]: r["window"] for r in csv.DictReader(open(a.windows), delimiter="\t")}
    solves = find_all_solves(a.solves)
    ensemble = {}
    if a.ensemble:
        import re
        for f in os.listdir(a.ensemble):
            m = re.match(r"^(.+)__s(\d+)\.npz$", f)
            if m:
                ensemble.setdefault(m.group(1), []).append(os.path.join(a.ensemble, f))
    for p in sorted(glob.glob(a.shards))[a.task::a.n_tasks]:
        print(one_shard(p, a.out, solves, windows, ensemble), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
