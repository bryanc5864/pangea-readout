#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Build the quantum chemistry fragment library: capped base-pair dimers per context.

Family B of the landscape (electronic structure) is a property of short stacked
fragments rather than of long sequences, so the plan computes it once on an exhaustive
fragment library and lets the encoder generalise to flanking context.  This module
generates that library's geometries.

The important choice is that geometries come from **cgNA+**, not from an idealised
B-DNA step.  A stacking energy computed on a fixed geometry for every context would
depend on sequence only through which bases are present, which is the same degeneracy
the plan warns about for the electrostatics.  Here the central step's rise, twist, roll
and slide respond to the flanking bases before any quantum chemistry runs.

For a tetramer context X1 X2 X3 X4 the central base-pair step is X2-X3.  A duplex is
built with the tetramer centred inside fixed padding so cgNA+ end effects do not reach
the step, the two central base pairs are extracted, and the sugar is replaced by a
hydrogen on the glycosidic nitrogen.  Capping at N9/N1 with H is the standard way to
isolate base stacking and hydrogen bonding from backbone contributions.

Each context yields four geometries, which is what an interaction energy needs:

    dimer   both base pairs           (4 bases)
    bp1     the 5' base pair          (2 bases)
    bp2     the 3' base pair          (2 bases)
    bases   each of the 4 bases alone (for hydrogen-bond decomposition)

so that  E_stack = E(dimer) - E(bp1) - E(bp2)  and
         E_hbond(bp) = E(bp) - E(base_watson) - E(base_crick).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "physics"))
import cgna_atomic as ca  # noqa: E402

BASES = "ACGT"
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
# Atoms belonging to the base proper; everything else is sugar or phosphate.
SUGAR_PHOSPHATE = {
    "P", "OP1", "OP2", "O5'", "C5'", "H5'", "H5''", "C4'", "H4'", "O4'",
    "C1'", "H1'", "C3'", "H3'", "C2'", "H2'", "H2''", "O3'", "O2'", "HO2'",
}
GLYCOSIDIC = {"DA": "N9", "DG": "N9", "DT": "N1", "DC": "N1"}
N_H_BOND_A = 1.012          # N-H bond length used for the cap


def rc(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def canonical_context(ctx: str) -> str:
    """Reverse-complement canonical form, so a context and its complement share a label.

    A base-pair step and its reverse complement are the same physical object, so
    computing both wastes half the budget and invites inconsistent labels.
    """
    return min(ctx, rc(ctx))


def unique_contexts(k: int = 4) -> List[str]:
    seen = []
    taken = set()
    for tup in itertools.product(BASES, repeat=k):
        ctx = "".join(tup)
        c = canonical_context(ctx)
        if c not in taken:
            taken.add(c)
            seen.append(c)
    return seen


def cap_base(atoms: Dict[str, np.ndarray], resname: str) -> Dict[str, np.ndarray]:
    """Strip sugar and phosphate, add a hydrogen on the glycosidic nitrogen."""
    gly = GLYCOSIDIC[resname]
    if gly not in atoms or "C1'" not in atoms:
        raise KeyError(f"missing {gly} or C1' for {resname}")
    n_pos = atoms[gly]
    direction = atoms["C1'"] - n_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("degenerate glycosidic direction")
    h_pos = n_pos + direction / norm * N_H_BOND_A
    out = {k: v for k, v in atoms.items() if k not in SUGAR_PHOSPHATE}
    out["H_cap"] = h_pos
    return out


def central_step_index(context: str) -> int:
    """0-based index i of the central step (context[i], context[i+1]) of an even context.

    4-mer -> 1 (two bases either side counting the step's own), 6-mer -> 2. The first
    hexamer library used the tetramer's index for every length, which put a hexamer's
    "central" step one base off centre, with one context base on its left and three on
    its right (DECISIONS.md, 2026-11-21).
    """
    if len(context) % 2 or len(context) < 2:
        raise ValueError(f"context length must be even, got {len(context)}")
    return len(context) // 2 - 1


GEOMETRIES = ("gs", "soft_plus", "soft_minus")


def soft_mode_excess(seq: str, step_index: int, sign: float) -> np.ndarray:
    """cgNA+ tangent displacement putting one step at +-1 sigma along its softest mode.

    The step's six inter-base-pair coordinates have marginal covariance Sigma_SS (the
    S-block of the inverse of the full stiffness: marginal, not conditional). With v its
    top eigenvector and lambda = sigma^2 its eigenvalue, the displacement of the whole
    duplex that puts the step at +-sigma along v, every other coordinate at its
    conditional mean, is delta = +- Sigma[:, S] v / sqrt(lambda). The library composes it
    as gs o exp(delta) (CGNAGroundState(excess=...)). These are PLAN.md's "three stacking
    geometries": the ground state and the thermal range of the step's dominant
    fluctuation, from cgNA+'s own stiffness rather than an arbitrary twist offset.
    """
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams
    pr = CGNAPlusParams(seq.upper(), aligned_strands=True, include_stiffness=True)
    names = list(pr.param_names)
    K = pr.stiffmat
    K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
    S = [i for i, n in enumerate(names) if n.startswith("S")]
    b = S[step_index]
    rows = np.arange(6 * b, 6 * b + 6)
    E = np.zeros((K.shape[0], 6)); E[rows, np.arange(6)] = 1.0
    Sig_cols = np.linalg.solve(K, E)                    # Sigma[:, S] without forming K^-1
    Sss = Sig_cols[rows]
    w, V = np.linalg.eigh(0.5 * (Sss + Sss.T))
    v, lam = V[:, -1], w[-1]
    delta = float(sign) * (Sig_cols @ v) / np.sqrt(lam)
    return delta.reshape(-1, 6)


def build_context_geometries(context: str, pad: str = "GC", flank: int = 3,
                             geometry: str = "gs") -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Capped geometries for the central step of *context*.

    The duplex is `pad*flank + context + pad*flank`, so the central step sits at least
    three base pairs from either end and cgNA+ end effects do not reach it.
    """
    left = (pad * ((flank // len(pad)) + 1))[:flank]
    right = (pad * ((flank // len(pad)) + 1))[:flank]
    seq = left + context + right
    n = len(seq)

    excess = None
    if geometry != "gs":
        # the central step is step p-1 (0-based) between Watson p and p+1 (1-based)
        excess = soft_mode_excess(seq, flank + central_step_index(context),
                                  +1.0 if geometry == "soft_plus" else -1.0)
    gs = ca.CGNAGroundState(seq, excess=excess)
    templates = ca.NucleotideTemplates()
    atoms = ca.build_duplex(seq, templates, gs, keep_hydrogens=True)

    by_res: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    resname: Dict[Tuple[str, int], str] = {}
    for a in atoms:
        by_res.setdefault((a.chain, a.resseq), {})[a.name] = a.xyz
        resname[(a.chain, a.resseq)] = a.resname

    # Central step: Watson positions p and p+1 (1-based), p = flank + 1 + the step's
    # 0-based index in the context. Chain A residue i pairs with chain B residue n+1-i.
    p = flank + 1 + central_step_index(context)
    units: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    labelled: Dict[str, Dict[str, np.ndarray]] = {}
    for tag, key in (
        ("w1", ("A", p)), ("w2", ("A", p + 1)),
        ("c1", ("B", n + 1 - p)), ("c2", ("B", n + 1 - (p + 1))),
    ):
        rn = resname[key]
        labelled[tag] = cap_base(by_res[key], rn)
        units[tag] = _to_xyz(labelled[tag])

    # bp1 = w1 + c1 (the 5' pair), bp2 = w2 + c2
    units["bp1"] = units["w1"] + units["c1"]
    units["bp2"] = units["w2"] + units["c2"]
    units["dimer"] = units["bp1"] + units["bp2"]
    return units


def build_context_trimers(context: str, pad: str = "GC", flank: int = 3) -> Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]]:
    """PLAN.md B: "stacked dimers and trimers across all tetramer contexts". The two base-pair
    trimers of a tetramer context that contain its central step, from the same padded cgNA+
    ground state as the dimer library: t5 = pairs 0, 1, 2 of the context and t3 = pairs 1, 2, 3.
    Returns {t5|t3: {bpA, bpB, bpC, trimer}} (bpA the 5'-most pair)."""
    left = (pad * ((flank // len(pad)) + 1))[:flank]
    right = (pad * ((flank // len(pad)) + 1))[:flank]
    seq = left + context + right
    n = len(seq)
    gs = ca.CGNAGroundState(seq)
    atoms = ca.build_duplex(seq, ca.NucleotideTemplates(), gs, keep_hydrogens=True)
    by_res: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    resname: Dict[Tuple[str, int], str] = {}
    for a in atoms:
        by_res.setdefault((a.chain, a.resseq), {})[a.name] = a.xyz
        resname[(a.chain, a.resseq)] = a.resname

    def pair(p):                       # Watson position p (1-based) and its partner
        w, c = ("A", p), ("B", n + 1 - p)
        return _to_xyz(cap_base(by_res[w], resname[w])) + _to_xyz(cap_base(by_res[c], resname[c]))
    p0 = flank + 1                     # the context's first pair
    out = {}
    for tag, first in (("t5", p0), ("t3", p0 + 1)):
        u = {"bpA": pair(first), "bpB": pair(first + 1), "bpC": pair(first + 2)}
        u["trimer"] = u["bpA"] + u["bpB"] + u["bpC"]
        out[tag] = u
    return out


def _element_of(name: str) -> str:
    if name == "H_cap":
        return "H"
    stripped = name.lstrip("0123456789")
    return "H" if stripped.startswith("H") else (stripped[0] if stripped else "C")


def _to_xyz(atoms: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    return [(_element_of(k), v) for k, v in atoms.items()]


def write_xyz(units: Sequence[Tuple[str, np.ndarray]], path: str, comment: str = "") -> str:
    with open(path, "w") as f:
        f.write(f"{len(units)}\n{comment}\n")
        for el, xyz in units:
            f.write(f"{el:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--pad", default="GC")
    ap.add_argument("--flank", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contexts", nargs="*", default=None)
    ap.add_argument("--geometry", choices=GEOMETRIES, default="gs",
                    help="ground state, or +-1 sigma along the central step's softest mode")
    ap.add_argument("--trimers", action="store_true", help="base-pair trimers (t5, t3) per context")
    a = ap.parse_args()
    if a.trimers:
        contexts = a.contexts or unique_contexts(a.context_length)
        os.makedirs(a.out, exist_ok=True)
        index = []
        for ctx in contexts:
            for tag, u in build_context_trimers(ctx, a.pad, a.flank).items():
                d = os.path.join(a.out, f"{ctx}_{tag}"); os.makedirs(d, exist_ok=True)
                rec = {"context": f"{ctx}_{tag}", "tetramer": ctx, "trimer": tag, "central_step": ctx[1:3],
                       "files": {}, "n_atoms": {}}
                for k, xs in u.items():
                    pth = os.path.join(d, f"{k}.xyz"); write_xyz(xs, pth, comment=f"{ctx} {tag} {k}")
                    rec["files"][k] = pth; rec["n_atoms"][k] = len(xs)
                index.append(rec)
            print(f"[ok] {ctx} trimers", flush=True)
        json.dump({"n_contexts": len(index), "context_length": a.context_length, "pad": a.pad, "flank": a.flank,
                   "geometry_source": "cgNA+ ground state of the padded duplex (as the dimer library)",
                   "contexts": index, "failures": []},
                  open(os.path.join(a.out, "qc_fragment_index.json"), "w"), indent=1)
        print(len(index), "trimers ->", a.out)
        return 0

    contexts = a.contexts or unique_contexts(a.context_length)
    if a.limit:
        contexts = contexts[: a.limit]
    os.makedirs(a.out, exist_ok=True)

    index = []
    failures = []
    for ctx in contexts:
        d = os.path.join(a.out, ctx)
        os.makedirs(d, exist_ok=True)
        try:
            units = build_context_geometries(ctx, a.pad, a.flank, geometry=a.geometry)
        except Exception as e:
            failures.append({"context": ctx, "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL] {ctx}: {e}", file=sys.stderr, flush=True)
            continue
        i0 = central_step_index(ctx)
        rec = {"context": ctx, "central_step": ctx[i0:i0 + 2], "files": {},
               "n_atoms": {}, "revcomp": rc(ctx)}
        for tag in ("dimer", "bp1", "bp2", "w1", "w2", "c1", "c2"):
            path = os.path.join(d, f"{tag}.xyz")
            write_xyz(units[tag], path, comment=f"{ctx} {tag}")
            rec["files"][tag] = path
            rec["n_atoms"][tag] = len(units[tag])
        # Consistency: the dimer must be exactly the union of the two pairs.
        assert rec["n_atoms"]["dimer"] == rec["n_atoms"]["bp1"] + rec["n_atoms"]["bp2"]
        index.append(rec)
        print(f"[ok] {ctx}  step {ctx[1:3]}  dimer {rec['n_atoms']['dimer']} atoms",
              flush=True)

    meta = {
        "n_contexts": len(index),
        "context_length": a.context_length,
        "pad": a.pad,
        "flank": a.flank,
        "capping": "sugar and phosphate removed; hydrogen placed on the glycosidic "
                   f"nitrogen along the former N-C1' direction at {N_H_BOND_A} angstrom",
        "geometry_source": "cgNA+ ground state for the padded duplex, so the central "
                           "step responds to its flanking context",
        "energy_definitions": {
            "stacking": "E(dimer) - E(bp1) - E(bp2)",
            "hydrogen_bond_bp1": "E(bp1) - E(w1) - E(c1)",
            "hydrogen_bond_bp2": "E(bp2) - E(w2) - E(c2)",
        },
        "contexts": index,
        "failures": failures,
    }
    with open(os.path.join(a.out, "qc_fragment_index.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{len(index)} contexts written to {a.out} ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
