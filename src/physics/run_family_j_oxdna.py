#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family J extensions from oxDNA2: single-strand conformation, stacking and elasticity.

PLAN.md, Family J: "persistence length and stretching elasticity ... self-structure propensity";
ground truth "single-strand coarse-grained parameters, force spectroscopy". Each fragment's two
strands (p = the given strand, m = its reverse complement) are simulated as free single strands
with oxDNA2 and its sequence-dependent parameters (oxDNA 3.7 built from the official source;
Langevin-like "john" thermostat, dt 0.005, 25 C, 0.15 M monovalent salt):

* equilibrium: 1e5 steps relaxation, 8e5 steps sampled every 1e4 (80 frames): end-to-end distance
  and radius of gyration (nm), persistence length from the decay of backbone-bond tangent
  correlations (nm), the mean oxDNA2 stacking energy of each step and the probability that a
  step is unstacked (stacking energy above -0.1 oxDNA units), per nucleotide as the mean of the
  steps touching it;
* elasticity: constant forces of 5, 10 and 20 pN pulling the two ends apart, 2e5 steps each after
  5e4 of relaxation: mean end-to-end extension along the force, per nucleotide (nm/nt).

Run in the oxDNA environment (`conda_envs/oxdna`). The classic topology lists each strand 3'->5',
so the sequence is written reversed and every per-nucleotide result mapped back to the 5'->3'
frame of its strand. Also per nucleotide over the same 80 frames (added 2026-11-22, PLAN.md J
"charge density and electrostatics without a complementary strand, hydration ... self-structure"):
the oxDNA2 Debye-Hueckel energy (charges on the backbone sites, 0.15 M; half of each pair's term),
the backbone charge spacing (nm per step, so the local linear charge density is its inverse),
the probability of a hydrogen-bonded partner (self-structure), the number of base-site contacts
within 0.85 nm excluding neighbours (burial), and the probability that the base is exposed
(unstacked on both sides and unpaired). One .npz per fragment; resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time

import numpy as np

SEQDEP = os.environ.get("PANGEA_OXDNA_SEQDEP",
                        "/home/bjc75/project/conda_envs/oxdna/src/oxDNA/oxDNA2_sequence_dependent_parameters.txt")
LENGTH_NM = 0.8518            # oxDNA length unit
FORCE_PN = 48.63              # oxDNA force unit at the standard energy scale
STACK_TERM = 2                # DNA2 pair term: stacking
HB_TERM = 4                   # DNA2 pair term: hydrogen bonding
DH_TERM = 7                   # DNA2 pair term: Debye-Hueckel electrostatics (charges on backbone sites)
HB_BELOW = -0.1               # oxDNA energy units: a hydrogen-bonded pair
CONTACT_SU = 1.0              # base-site contact distance (oxDNA length units, 0.85 nm)
UNSTACKED_ABOVE = -0.1        # oxDNA energy units
FORCES_PN = (5.0, 10.0, 20.0)
COMP = str.maketrans("ACGT", "TGCA")


def _write_system(work, seq):
    """Topology and a straight initial strand; returns the 3'->5' sequence used."""
    s35 = seq[::-1]
    n = len(s35)
    with open(os.path.join(work, "ss.top"), "w") as f:
        f.write(f"{n} 1\n")
        for i, b in enumerate(s35):
            f.write(f"1 {b} {i - 1 if i > 0 else -1} {i + 1 if i < n - 1 else -1}\n")
    with open(os.path.join(work, "ss.dat"), "w") as f:
        f.write("t = 0\nb = 300 300 300\nE = 0 0 0\n")
        for i in range(n):
            f.write(f"{100.0 + 0.7 * i} 150 150 0 1 0 1 0 0 0 0 0 0 0 0\n")
    return s35


def _inp(work, steps, seed, conf="ss.dat", extra=None):
    d = {"interaction_type": "DNA2", "salt_concentration": "0.15", "sim_type": "MD", "backend": "CPU",
         "steps": str(steps), "dt": "0.005", "T": "25C", "thermostat": "john", "newtonian_steps": "103",
         "diff_coeff": "2.5", "verlet_skin": "0.5", "topology": os.path.join(work, "ss.top"),
         "conf_file": os.path.join(work, conf), "trajectory_file": os.path.join(work, "traj.dat"),
         "lastconf_file": os.path.join(work, "last.dat"), "print_conf_interval": str(10 ** 9),
         "print_energy_every": str(10 ** 9), "energy_file": os.path.join(work, "energy.dat"),
         "time_scale": "linear", "refresh_vel": "1", "restart_step_counter": "1",
         "use_average_seq": "0", "seq_dep_file": SEQDEP, "seed": str(seed), "no_stdout_energy": "1"}
    if extra:
        d.update(extra)
    return d


def _run(oxpy, inp, chunks, every, sampler=None):
    with oxpy.Context(print_coda=False):
        i_ = oxpy.InputFile()
        for k, v in inp.items():
            i_[k] = v
        m = oxpy.OxpyManager(i_)
        out = []
        for _ in range(chunks):
            m.run(every)
            if sampler is not None:
                out.append(sampler(m))
        try:                                   # write the final state to lastconf_file
            m.print_configuration(True)
        except TypeError:
            m.print_configuration()
    return out


def simulate_strand(seq: str, seed: int, work: str, eq_steps=100000, prod_steps=800000, every=10000,
                    force_relax=50000, force_steps=200000):
    import oxpy
    s35 = _write_system(work, seq)
    n = len(s35)

    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]

    def sample(m):
        ci = m.config_info()
        ps = ci.particles()
        X = np.array([[p.pos[0], p.pos[1], p.pos[2]] for p in ps])
        BB = X + np.array([np.asarray(p.int_centers[0]) for p in ps])          # backbone (charge) sites
        BS = X + np.array([np.asarray(p.int_centers[2]) for p in ps])          # base sites
        inter = ci.interaction
        e = np.array([inter.pair_interaction_term(STACK_TERM, ps[i], ps[i + 1]) for i in range(n - 1)])
        dh = np.zeros(n); hb = np.zeros(n)
        for i, j in pairs:
            v = inter.pair_interaction_term(DH_TERM, ps[i], ps[j])
            dh[i] += 0.5 * v; dh[j] += 0.5 * v
            h = inter.pair_interaction_term(HB_TERM, ps[i], ps[j])
            if h < HB_BELOW:
                hb[i] = 1.0; hb[j] = 1.0
        d = np.linalg.norm(BS[:, None, :] - BS[None, :, :], axis=-1)
        near = (d < CONTACT_SU) & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 2)
        return X, e, dh, hb, near.sum(1).astype(float), np.linalg.norm(BB[1:] - BB[:-1], axis=1)

    _run(oxpy, _inp(work, eq_steps, seed), 1, eq_steps)
    samples = _run(oxpy, _inp(work, prod_steps, seed + 1, conf="last.dat"), prod_steps // every, every, sample)
    X = np.stack([s[0] for s in samples]); E = np.stack([s[1] for s in samples])     # (f, n, 3), (f, n-1)
    DH = np.stack([s[2] for s in samples]); HB = np.stack([s[3] for s in samples])
    CT = np.stack([s[4] for s in samples]); BBd = np.stack([s[5] for s in samples]) * LENGTH_NM
    ree = np.linalg.norm(X[:, -1] - X[:, 0], axis=1) * LENGTH_NM
    rg = np.sqrt(np.mean(np.sum((X - X.mean(1, keepdims=True)) ** 2, -1), 1)) * LENGTH_NM
    b = X[:, 1:] - X[:, :-1]
    bl = np.linalg.norm(b, axis=-1)
    t = b / bl[..., None]
    mean_b = float(bl.mean()) * LENGTH_NM
    smax = min(15, n - 2)
    corr = np.array([np.mean(np.sum(t[:, :-s] * t[:, s:], -1)) for s in range(1, smax + 1)])
    ok = corr > 0.05
    lp = float(-mean_b / np.polyfit(np.arange(1, smax + 1)[ok], np.log(corr[ok]), 1)[0]) if ok.sum() >= 3 else float("nan")
    e_step = E.mean(0)                                   # per step, 3'->5' frame
    p_unst = (E > UNSTACKED_ABOVE).mean(0)

    def per_nt(step_vals):                               # mean of the steps touching each nucleotide
        v = np.full(n, np.nan)
        for i in range(n):
            touch = [step_vals[j] for j in (i - 1, i) if 0 <= j < n - 1]
            v[i] = float(np.mean(touch))
        return v[::-1]                                    # back to the 5'->3' frame
    ext = {}
    for f_pn in FORCES_PN:
        f_u = f_pn / FORCE_PN
        ff = os.path.join(work, "forces.txt")
        with open(ff, "w") as fh:
            fh.write("{\ntype = string\nparticle = 0\nF0 = %.6f\nrate = 0\ndir = -1, 0, 0\n}\n" % f_u)
            fh.write("{\ntype = string\nparticle = %d\nF0 = %.6f\nrate = 0\ndir = 1, 0, 0\n}\n" % (n - 1, f_u))
        extra = {"external_forces": "1", "external_forces_file": ff}

        def dx(m):
            ps = m.config_info().particles()
            return (ps[n - 1].pos[0] - ps[0].pos[0]) * LENGTH_NM
        _run(oxpy, _inp(work, force_relax, seed + 7, conf="last.dat", extra=extra), 1, force_relax)
        xs = _run(oxpy, _inp(work, force_steps, seed + 11, conf="last.dat", extra=extra), force_steps // every, every, dx)
        ext[f_pn] = float(np.mean(xs)) / (n - 1)
    # per nucleotide, 3'->5' frame until mapped back: exposed = unstacked on both sides and
    # without a hydrogen-bonded partner in that frame
    unst = E > UNSTACKED_ABOVE                                            # (f, n-1)
    left = np.concatenate([np.ones((len(E), 1), bool), unst], 1)          # step i-1 (absent = unstacked)
    right = np.concatenate([unst, np.ones((len(E), 1), bool)], 1)         # step i
    exposed = left & right & (HB < 0.5)
    spacing_step = BBd.mean(0)                                            # nm per backbone step
    return {"ree_nm": float(ree.mean()), "rg_nm": float(rg.mean()), "lp_nm": lp,
            "mean_stack_E": float(e_step.mean()), "unstacked_frac": float(p_unst.mean()),
            "stack_E_nt": per_nt(e_step), "unstacked_p_nt": per_nt(p_unst),
            "ext_nm_per_nt": ext, "n_frames": int(len(samples)),
            # electrostatics without a complementary strand (oxDNA2 Debye-Hueckel, 0.15 M), the
            # strand's own charge spacing along the backbone, self-structure and solvent exposure
            "dh_E_nt": DH.mean(0)[::-1], "hbonded_p_nt": HB.mean(0)[::-1],
            "contacts_nt": CT.mean(0)[::-1], "exposed_p_nt": exposed.mean(0)[::-1],
            "charge_spacing_nm_nt": per_nt(spacing_step),
            "mean_dh_E": float(DH.sum(1).mean() / n), "hbonded_frac": float(HB.mean()),
            "mean_charge_spacing_nm": float(BBd.mean()), "exposed_frac": float(exposed.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workroot", default=None)
    ap.add_argument("--prod-steps", type=int, default=800000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, _, hi = a.rows.partition("-")
    rows = list(csv.DictReader(open(a.library), delimiter="\t"))[int(lo): int(hi) + 1]
    for r in rows:
        fid, seq = r["frag_id"], r["sequence"].upper()
        out = os.path.join(a.out, f"{fid}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time(); res = {}
        seed = int.from_bytes(fid.encode()[-6:], "big") % (2 ** 31)
        try:
            for tag, s in (("p", seq), ("m", seq.translate(COMP)[::-1])):
                work = tempfile.mkdtemp(prefix="jox_", dir=a.workroot)
                try:
                    res[tag] = simulate_strand(s, seed + (0 if tag == "p" else 1000), work, prod_steps=a.prod_steps)
                finally:
                    shutil.rmtree(work, ignore_errors=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"{fid} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        arrays = {}
        for tag, v in res.items():
            for k in ("ree_nm", "rg_nm", "lp_nm", "mean_stack_E", "unstacked_frac", "mean_dh_E",
                      "hbonded_frac", "mean_charge_spacing_nm", "exposed_frac"):
                arrays[f"{tag}_{k}"] = np.float64(v[k])
            for k in ("stack_E_nt", "unstacked_p_nt", "dh_E_nt", "hbonded_p_nt", "contacts_nt",
                      "exposed_p_nt", "charge_spacing_nm_nt"):
                arrays[f"{tag}_{k}"] = v[k]
            for f_pn, x in v["ext_nm_per_nt"].items():
                arrays[f"{tag}_ext_f{int(f_pn)}_nm_per_nt"] = np.float64(x)
        tmp = out + ".partial.npz"
        np.savez(tmp, sequence=np.asarray([seq]), **arrays); os.replace(tmp, out)
        print(f"{fid} ok {time.time() - t0:.0f}s lp {res['p']['lp_nm']:.2f}/{res['m']['lp_nm']:.2f} nm", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
