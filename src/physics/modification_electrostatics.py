#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family N, the electrostatic response to an RNA modification (family code "Nes", side shards).

PLAN.md Family N: "Induced change in the main channels for N6-methyladenosine, pseudouridine and
2-prime-O-methyl, applied across the RNA and hybrid classes". Electrostatics is a main channel
(Family K solves it on A-form RNA and hybrids). For each modification -- m6A, pseudouridine (psi),
inosine (ino; A-to-I editing, beyond the plan's list, as in Family N's thermodynamics), and
2'-O-methyl on each base (Am, Cm, Gm, Um) -- with an RNA partner strand (RNA duplex) or a DNA partner
strand (hybrid), and each 3 nt context on the modified strand (5' neighbour, site, 3' neighbour):

* the 13 bp molecule GC GCG <context> CGC GC (the modified strand; Family K's G-C caps) is built in
  all-atom A-form (`rna_atomic`, cgNA+ RNA or hybrid set) and minimised as Family K's molecules are;
* the unmodified PQR comes from pdb2pqr (AMBER; equal to RNA.OL3's charges); the modified PQR replaces
  the site nucleotide's modified group with its RESP charges (`src/qc/resp_modified.py`: HF/6-31G*,
  unmodified atoms fixed to OL3, the group's total kept) and places the new atoms by superposing the
  RESP geometry (GFN2-xTB, A-form torsions held) on the site's base (or sugar, for 2'-O-methyl); new
  atoms take pdb2pqr's AMBER radii of the same atom type in the same molecule;
* the reference is the unmodified nucleotide with its base (or 2'-OH group) charges from the same RESP
  protocol (`CONTROL`), so the difference isolates the modification;
* both are solved with Family K's construction (150 mM NaCl, exclusion radius 2.0 A, exact 0.35 A
  fine spacing) on identical grids, and the response is modified minus reference at the site's base
  pair: the potential 1 and 4 A outside the surface in the major groove, the minor groove, and the
  two backbone sectors of each strand, the site pair's map solvation energy, and the solvation of
  the site with its two neighbours on each side.
Channels are per position of each transcript (the strand x and rc(x), each in its own 5'->3' frame),
looked up by that position's 3 nt context: stranded, Nes_m_*(x) = Nes_p_*(rc x) in its own frame,
like Family N (a strand pair in `pangea.STRAND_PAIRS`).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAMILY = "Nes"
MODS: Tuple[Tuple[str, str, str], ...] = (("m6A", "A", "base"), ("psi", "U", "base"), ("ino", "A", "base"),
                                          ("Am", "A", "ome"), ("Cm", "C", "ome"), ("Gm", "G", "ome"), ("Um", "U", "ome"))
PARTNERS = ("rna", "dna")
FLANK5, FLANK3 = "GCGCG", "CGCGC"          # cap GC + GCG, CGC + cap GC
SITE_INDEX = len(FLANK5) + 1                # 0-based index of the site in the 13 bp molecule
N_BP = len(FLANK5) + 3 + len(FLANK3)
QUANTITIES = (("dpot_maj_d1", "kT/e"), ("dpot_maj_d4", "kT/e"), ("dpot_min_d1", "kT/e"), ("dpot_min_d4", "kT/e"),
              ("dpot_own_d1", "kT/e"), ("dpot_par_d1", "kT/e"), ("dbpsolv", "kJ/mol"), ("dsolv5", "kJ/mol"))
CONDITION = "NaCl_150mM"
SRC = ("APBS NPBE on all-atom A-form RNA and hybrid 13-mers, RESP charges for the modified nucleotides "
       "(src/qc/resp_modified.py) and AMBER RNA.OL3 / DNA.OL15 elsewhere")
RESP_DIR = os.environ.get("PANGEA_RESP_MOD_DIR", "/nfs/roberts/scratch/pi_skr2/bjc75/pangea/tmp/respmod")
PSI_FROM_U = {"C5": "N1", "C4": "C2", "O4": "O2", "C2": "C4", "O2": "O4", "N1": "C5", "H1": "H5"}   # psi label -> U position
# The reference for each modification is the unmodified nucleotide fitted by the same RESP protocol
# (its base, or its 2'-OH group), so the response isolates the modification: the protocol alone moves
# adenosine's and uridine's base charges by 0.075 and 0.081 e RMS from OL3's (RESP controls).
CONTROL = {"m6A": "A0", "ino": "A0", "psi": "U0", "Am": "A0om", "Cm": "C0om", "Gm": "G0om", "Um": "U0om"}


def contexts(base: str) -> List[str]:
    return [a + base + c for a in "ACGU" for c in "ACGU"]


def _strands(ctx_rna: str, partner: str) -> Tuple[str, str, str]:
    """(builder input in the DNA alphabet, chain of the site, its resseq)."""
    rna = FLANK5 + ctx_rna + FLANK3
    dna_alpha = rna.replace("U", "T")
    if partner == "rna":
        return dna_alpha, "A", SITE_INDEX + 1
    comp = str.maketrans("ACGT", "TGCA")
    return dna_alpha.translate(comp)[::-1], "B", SITE_INDEX + 1     # DNA strand = rc(RNA); RNA = chain B


def build_unmodified(ctx_rna: str, partner: str, work: str):
    import cgna_atomic as ca
    import duplex_classes as K
    import minimize as MZ
    import rna_atomic as R
    import apbs_run as A
    seq, chain, resseq = _strands(ctx_rna, partner)
    atoms = R.build_rna_duplex(seq, K.SETS["rna"]) if partner == "rna" else R.build_hybrid(seq, K.SETS["hyb"])
    raw = ca.write_pdb(atoms, os.path.join(work, "raw.pdb"))
    mn = os.path.join(work, "min.pdb")
    ff = ["amber14/RNA.OL3.xml"] if partner == "rna" else ["amber14/DNA.OL15.xml", "amber14/RNA.OL3.xml"]
    MZ.restrained_minimise(raw, mn, max_iterations=20000, tolerance=1.0, force_fields=ff)
    pqr = os.path.join(work, "unmod.pqr")
    A.run_pdb2pqr(mn, pqr, log=os.path.join(work, "pdb2pqr.log"))
    A.sanitise_pqr(pqr)
    return mn, pqr, chain, resseq


def _read_resp(mod: str):
    d = json.load(open(os.path.join(RESP_DIR, f"{mod}_resp.json")))
    rows = open(os.path.join(RESP_DIR, f"{mod}_relaxed.xyz")).read().splitlines()[2:]
    names = [r.split()[4] for r in rows]
    xyz = np.array([[float(t) for t in r.split()[1:4]] for r in rows])
    return d, dict(zip(names, xyz))


def _kabsch(P, Q):
    """Rotation R and translation t with R @ P_i + t ~ Q_i."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def modified_pqr(unmod_pqr: str, mod: str, chain: str, resseq: int, out: str) -> Dict[str, float]:
    import apbs_run as A
    xyz, q, r, recs = A.read_pqr(unmod_pqr)
    resp, X = _read_resp(mod)
    kind = next(k for m, _, k in MODS if m == mod)
    site = [i for i, rec in enumerate(recs) if rec.get("chain") == chain and int(rec.get("resseq", -1)) == resseq]
    at = {recs[i]["name"]: i for i in site}
    rad = {}
    for i, rec in enumerate(recs):                         # AMBER radii by (residue type, atom) from this molecule
        rad.setdefault((rec.get("resname", "")[-1:], rec["name"]), float(r[i]))
    R_C, R_N, R_H1, R_HN = rad[("G", "C8")], rad[("G", "N7")], rad[("G", "H5'")], rad[("G", "H1")]
    R_Ocarb, R_Oeth = rad[("G", "O6")], rad[("G", "O4'")]
    charges = resp["charges"]
    new_atoms: List[Tuple[str, np.ndarray, float, float]] = []
    drop = set()
    q_new = {}
    if kind == "base":
        if mod == "psi":
            common = [(u, p) for p, u in PSI_FROM_U.items() if not p.startswith("H")] + [("N3", "N3"), ("C6", "C6")]
        else:
            common = [(n, n) for n in ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9") if n in at and n in X]
        P = np.array([X[p] for _, p in common]); Q = np.array([xyz[at[u]] for u, _ in common])
        Rm, t = _kabsch(P, Q)
        place = lambda n: Rm @ X[n] + t                    # noqa: E731
        if mod == "m6A":
            h_far = max(("H61", "H62"), key=lambda h: np.linalg.norm(xyz[at[h]] - xyz[at["N1"]]))
            h_keep = "H62" if h_far == "H61" else "H61"
            resp_keep = next(h for h in ("H61", "H62") if h in charges)
            drop.add(h_far)
            q_new.update({n: charges[n] for n in charges if n in at and n not in ("H61", "H62")})
            q_new[h_keep] = charges[resp_keep]
            new_atoms += [("C6M", place("C6M"), charges["C6M"], R_C)] + \
                         [(f"HM{k}", place(f"HM{k}"), charges[f"HM{k}"], R_H1) for k in range(3)]
        elif mod == "ino":
            drop |= {"N6", "H61", "H62"}
            q_new.update({n: charges[n] for n in charges if n in at})
            new_atoms += [("O6", place("O6"), charges["O6"], R_Ocarb), ("H1", place("H1"), charges["H1"], R_HN)]
        else:                                              # psi: same positions, relabelled atoms
            ren = {u: p for p, u in PSI_FROM_U.items()}     # U label at this position -> psi label
            for u, i in at.items():
                p = ren.get(u, u)
                if p in charges:
                    q_new[u] = charges[p]
            new_radius = {"C5": R_N, "N1": R_C, "H5": R_HN}  # U position -> radius of the psi atom now there (N1, C5, H1)
            for u, rv in new_radius.items():
                r[at[u]] = rv
    else:                                                   # 2'-O-methyl: superpose on the sugar
        common = [n for n in ("C1'", "C2'", "C3'", "C4'", "O4'", "O2'") if n in at and n in X]
        Rm, t = _kabsch(np.array([X[n] for n in common]), np.array([xyz[at[n]] for n in common]))
        place = lambda n: Rm @ X[n] + t                    # noqa: E731
        drop.add("H2''")                                   # pdb2pqr's name for RNA HO2'
        for n in ("C2'", "H2'", "O2'"):
            q_new[n] = charges[n]
        r[at["O2'"]] = R_Oeth
        meth = [("CM2", place("CM2"), charges["CM2"], R_C)] + \
               [(f"HM2{k}", place(f"HM2{k}"), charges[f"HM2{k}"], R_H1) for k in range(3)]
        # the isolated nucleoside's methyl orientation can collide with the duplex (1.6-2.0 A seen);
        # rotate the methyl about the C2'-O2' bond to the rotamer farthest from every other atom
        o2, c2 = xyz[at["O2'"]], xyz[at["C2'"]]
        ax = (o2 - c2) / np.linalg.norm(o2 - c2)
        others = np.array([xyz[i] for i in range(len(xyz)) if not (i in site and recs[i]["name"] in ("O2'", "C2'", "H2''"))])
        best, best_d = None, -1.0
        for deg in range(0, 360, 5):
            t_ = np.radians(deg)
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rr = np.eye(3) + np.sin(t_) * Kx + (1 - np.cos(t_)) * Kx @ Kx
            pos = [o2 + Rr @ (x - o2) for _, x, _, _ in meth]
            dmin = min(float(np.min(np.linalg.norm(others - x_, axis=1))) for x_ in pos)
            if dmin > best_d:
                best_d, best = dmin, pos
        new_atoms += [(n_, x_, q_, r_) for (n_, _, q_, r_), x_ in zip(meth, best)]
    lines, serial = [], 1
    for i, rec in enumerate(recs):
        nm = rec["name"]
        if i in site and nm in drop:
            continue
        qv = q_new[nm] if (i in site and nm in q_new) else float(q[i])
        lines.append(f"ATOM {serial:6d} {nm:<4s} {rec.get('resname', 'UNK'):>4s} {rec['chain']} {int(rec['resseq']):4d} "
                     f"{xyz[i][0]:9.3f} {xyz[i][1]:9.3f} {xyz[i][2]:9.3f} {qv:8.4f} {float(r[i]):7.4f}\n"); serial += 1
    res = recs[site[0]].get("resname", "UNK")
    for nm, x, qv, rv in new_atoms:
        lines.append(f"ATOM {serial:6d} {nm:<4s} {res:>4s} {chain} {resseq:4d} {x[0]:9.3f} {x[1]:9.3f} {x[2]:9.3f} {qv:8.4f} {rv:7.4f}\n")
        serial += 1
    open(out, "w").write("".join(lines))
    xm, qm, rm, _ = A.read_pqr(out)
    others = np.array([i for i in range(len(xyz)) if i not in site])
    d_new = min((float(np.min(np.linalg.norm(xyz[others] - x, axis=1))) for _, x, _, _ in new_atoms), default=float("nan"))
    return {"dq_total": float(qm.sum() - q.sum()), "n_new": len(new_atoms), "n_dropped": len(drop & set(at)),
            "min_new_to_other_A": d_new}


def control_pqr(unmod_pqr: str, control: str, chain: str, resseq: int, out: str) -> float:
    """The unmodified molecule with the site's base (or 2'-OH group) charges from the control fit."""
    import apbs_run as A
    xyz, q, r, recs = A.read_pqr(unmod_pqr)
    d = json.load(open(os.path.join(RESP_DIR, f"{control}_resp.json")))["charges"]
    ren = {"HO2'": "H2''"}                                   # pdb2pqr's name for RNA HO2'
    want = {ren.get(k, k): v for k, v in d.items()}
    lines, n_set = [], 0
    for i, rec in enumerate(recs):
        qv = float(q[i])
        if rec.get("chain") == chain and int(rec.get("resseq", -1)) == resseq and rec["name"] in want:
            qv = want[rec["name"]]; n_set += 1
        lines.append(f"ATOM {i + 1:6d} {rec['name']:<4s} {rec.get('resname', 'UNK'):>4s} {rec['chain']} {int(rec['resseq']):4d} "
                     f"{xyz[i][0]:9.3f} {xyz[i][1]:9.3f} {xyz[i][2]:9.3f} {qv:8.4f} {float(r[i]):7.4f}\n")
    if n_set != len(want):
        raise ValueError(f"control {control}: set {n_set} of {len(want)} charges")
    open(out, "w").write("".join(lines))
    return float(sum(want.values()))


def solve_pair(mod: str, partner: str, ctx: str, work: str, unmod_cache: Dict[Tuple[str, str], dict]) -> dict:
    import apbs_run as A
    import frames as F
    import run_fragment as rf
    base = next(b for m, b, _ in MODS if m == mod)
    key = (partner, ctx)
    cond = [A.PRODUCTION_CONDITIONS_BY_NAME[CONDITION]]
    if key not in unmod_cache:
        uw = os.path.join(work, f"unmod_{partner}_{ctx}"); os.makedirs(uw, exist_ok=True)
        mn, pqr, chain, resseq = build_unmodified(ctx, partner, uw)
        bp, _ = F.bp_frames_from_pdb(mn)
        xyz, _, _, _ = A.read_pqr(pqr)
        unmod_cache[key] = {"min": mn, "pqr": pqr, "chain": chain, "resseq": resseq, "bp": bp,
                            "extent": xyz.max(0) - xyz.min(0), "ref": {}}
    u = unmod_cache[key]
    ctl = CONTROL[mod]
    if ctl not in u["ref"]:
        cw = os.path.join(work, f"ref_{ctl}_{partner}_{ctx}"); os.makedirs(cw, exist_ok=True)
        cp = os.path.join(cw, "ref.pqr")
        control_pqr(u["pqr"], ctl, u["chain"], u["resseq"], cp)
        u["ref"][ctl] = rf.solve_conditions(u["min"], cw, cond, u["bp"], slice(0, len(u["bp"])), extended=True,
                                            reference_solve=True, exact_fine_spacing=True, pqr_in=cp, grid_extent=u["extent"])
    mw = os.path.join(work, f"{mod}_{partner}_{ctx}"); os.makedirs(mw, exist_ok=True)
    mp = os.path.join(mw, "mod.pqr")
    diag = modified_pqr(u["pqr"], mod, u["chain"], u["resseq"], mp)
    om = rf.solve_conditions(u["min"], mw, cond, u["bp"], slice(0, len(u["bp"])), extended=True, reference_solve=True,
                             exact_fine_spacing=True, pqr_in=mp, grid_extent=u["extent"])
    ou = u["ref"][ctl]
    k = N_BP - 1 - SITE_INDEX if u["chain"] == "B" else SITE_INDEX      # the site's base-pair index
    ps_u, ps_m = np.asarray(ou["potential_surface"][0]), np.asarray(om["potential_surface"][0])
    own, par = ((1, 2), (4, 5)) if u["chain"] == "A" else ((4, 5), (1, 2))
    d = ps_m[k] - ps_u[k]                                            # (sector, offset)
    su = np.asarray(ou["ext_bp_solvation_map_kJ_mol"][0]); sm = np.asarray(om["ext_bp_solvation_map_kJ_mol"][0])
    res = {"mod": mod, "partner": partner, "context": ctx, "site_bp": int(k), "reference": ctl,
           "dpot_maj_d1": float(d[0, 1]), "dpot_maj_d4": float(d[0, 3]), "dpot_min_d1": float(d[3, 1]), "dpot_min_d4": float(d[3, 3]),
           "dpot_own_d1": float(np.nanmean(d[list(own), 1])), "dpot_par_d1": float(np.nanmean(d[list(par), 1])),
           "dbpsolv": float(sm[k] - su[k]), "dsolv5": float(np.sum(sm[k - 2:k + 3]) - np.sum(su[k - 2:k + 3])),
           "unmod_pot_maj_d1": float(ps_u[k, 0, 1]), "unmod_pot_min_d1": float(ps_u[k, 3, 1])}
    res.update({f"pqr_{a}": v for a, v in diag.items()})
    return res


# --------------------------------------------------------------------------------------
# Channels: per transcript position, looked up by the 3 nt context
# --------------------------------------------------------------------------------------
def load_table(folder: str) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out = {}
    for f in os.listdir(folder):
        if f.endswith(".json") and not f.startswith("failed"):
            d = json.load(open(os.path.join(folder, f)))
            if "mod" in d:
                out[(d["mod"], d["partner"], d["context"])] = d
    return out


def es_channels(seq: str, table) -> Dict[str, np.ndarray]:
    s = seq.upper().replace("T", "U"); n = len(s)
    comp = str.maketrans("ACGU", "UGCA")
    prof = {}
    for tag, t in (("p", s), ("m", s.translate(comp)[::-1])):
        for mod, base, _ in MODS:
            for partner in PARTNERS:
                arr = {q: np.full(n, np.nan) for q, _ in QUANTITIES}
                for i in range(1, n - 1):
                    if t[i] != base:
                        continue
                    rec = table.get((mod, partner, t[i - 1:i + 2]))
                    if rec is None:
                        continue
                    for q, _ in QUANTITIES:
                        arr[q][i] = rec[q]
                for q, _ in QUANTITIES:
                    prof[f"Nes_{tag}_{mod}_{partner}_{q}"] = arr[q]
    return prof


def _declare() -> None:
    from channels import REGISTRY, ChannelSpec, Parity, Resolution
    for mod, base, _ in MODS:
        for partner in PARTNERS:
            for q, u in QUANTITIES:
                for tag in ("p", "m"):
                    REGISTRY.register(ChannelSpec(
                        name=f"Nes_{tag}_{mod}_{partner}_{q}", family=FAMILY, resolution=Resolution.PER_BP,
                        parity=Parity.STRANDED, units=u, source=SRC,
                        description=f"change in {q[1:].replace('_', ' ')} when this {base} of transcript "
                                    f"{'x' if tag == 'p' else 'rc(x)'} carries {mod}, {'RNA' if partner == 'rna' else 'DNA'} partner strand"))


_declare()


def main() -> int:
    import argparse, shutil, tempfile, traceback
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True); ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--mods", nargs="*", default=[m for m, _, _ in MODS])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # work items grouped by (partner, context, parent base) so a task reuses each unmodified solve
    groups = {}
    for mod, base, _ in MODS:
        if mod not in a.mods:
            continue
        for partner in PARTNERS:
            for ctx in contexts(base):
                groups.setdefault((partner, ctx), []).append(mod)
    keys = sorted(groups)[a.task::a.n_tasks]
    for partner, ctx in keys:
        cache = {}
        work = tempfile.mkdtemp(prefix="nes_")
        try:
            for mod in groups[(partner, ctx)]:
                f = os.path.join(a.out, f"{mod}_{partner}_{ctx}.json")
                if os.path.exists(f):
                    continue
                try:
                    r = solve_pair(mod, partner, ctx, work, cache)
                    json.dump(r, open(f, "w"), indent=1)
                    print(mod, partner, ctx, {k: round(v, 3) for k, v in r.items() if k.startswith(("dpot", "dbp", "dsolv"))}, flush=True)
                except Exception as exc:                     # noqa: BLE001
                    json.dump({"mod": mod, "partner": partner, "context": ctx, "status": f"failed: {type(exc).__name__}: {str(exc)[:400]}",
                               "trace": traceback.format_exc()[-1500:]}, open(os.path.join(a.out, f"failed_{mod}_{partner}_{ctx}.json"), "w"))
                    print(mod, partner, ctx, "failed:", type(exc).__name__, str(exc)[:300], flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
