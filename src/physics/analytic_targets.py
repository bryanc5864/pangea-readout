#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Analytic landscape targets: available in unlimited quantity at near-zero cost.

The plan trains PANGEA in two stages.  The first uses targets that can be computed for
any sequence without a solver, so tens of millions of windows are affordable and the
trunk learns the local sequence-to-physics map before any expensive label is touched.
This module provides those targets:

  Family C   cgNA+ ground-state rigid-body coordinates and the stiffness matrix
  Family E   nearest-neighbour duplex thermodynamics with a salt correction

Nearest-neighbour parameters are the SantaLucia unified set (SantaLucia, PNAS 95:1460,
1998), and the sodium correction is Owczarzy et al., Biochemistry 43:3537 (2004).  Both
are validated in `tests/test_analytic.py` against values published with those papers,
because a silent sign or unit error in a thermodynamic table would propagate into every
Family E channel while still looking plausible.

Units: enthalpy kcal/mol, entropy cal/(mol K), free energy kcal/mol at 37 C, Tm in C.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R_CAL = 1.9872          # gas constant, cal/(mol K)
T37_K = 310.15
KELVIN = 273.15

# SantaLucia unified nearest-neighbour parameters.
# Key is the 5'->3' dinucleotide of the Watson strand; the complementary step is implied.
NN_DH: Dict[str, float] = {
    "AA": -7.9, "TT": -7.9,
    "AT": -7.2,
    "TA": -7.2,
    "CA": -8.5, "TG": -8.5,
    "GT": -8.4, "AC": -8.4,
    "CT": -7.8, "AG": -7.8,
    "GA": -8.2, "TC": -8.2,
    "CG": -10.6,
    "GC": -9.8,
    "GG": -8.0, "CC": -8.0,
}
NN_DS: Dict[str, float] = {
    "AA": -22.2, "TT": -22.2,
    "AT": -20.4,
    "TA": -21.3,
    "CA": -22.7, "TG": -22.7,
    "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0,
    "GA": -22.2, "TC": -22.2,
    "CG": -27.2,
    "GC": -24.4,
    "GG": -19.9, "CC": -19.9,
}
# Helix initiation, by the identity of the terminal base pair.
INIT_DH_AT, INIT_DS_AT = 2.3, 4.1
INIT_DH_GC, INIT_DS_GC = 0.1, -2.8


def nn_thermodynamics(seq: str) -> Dict[str, float]:
    """Duplex dH, dS and dG37 for a perfectly complementary duplex.

    Returns enthalpy in kcal/mol, entropy in cal/(mol K) and free energy in kcal/mol.
    """
    s = seq.upper()
    if any(b not in "ACGT" for b in s) or len(s) < 2:
        raise ValueError("sequence must be at least 2 nt of ACGT")
    dh = ds = 0.0
    for i in range(len(s) - 1):
        step = s[i:i + 2]
        dh += NN_DH[step]
        ds += NN_DS[step]
    for terminal in (s[0], s[-1]):
        if terminal in "GC":
            dh += INIT_DH_GC
            ds += INIT_DS_GC
        else:
            dh += INIT_DH_AT
            ds += INIT_DS_AT
    dg37 = dh - T37_K * ds / 1000.0
    return {"dH_kcal_mol": dh, "dS_cal_mol_K": ds, "dG37_kcal_mol": dg37}


def melting_temperature(seq: str, ct_molar: float = 1e-4,
                        na_molar: float = 1.0,
                        self_complementary: bool = False) -> float:
    """Tm in Celsius at total strand concentration `ct_molar` and sodium `na_molar`.

    For non-self-complementary duplexes with equal strand concentrations the standard
    form uses Ct/4.  The sodium correction is Owczarzy 2004 equation 22, applied to the
    1 M reference Tm.
    """
    t = nn_thermodynamics(seq)
    dh_cal = t["dH_kcal_mol"] * 1000.0
    ds = t["dS_cal_mol_K"]
    ct_term = ct_molar if self_complementary else ct_molar / 4.0
    tm_1m = dh_cal / (ds + R_CAL * math.log(ct_term))
    if na_molar == 1.0:
        return tm_1m - KELVIN
    fgc = gc_fraction(seq)
    ln_na = math.log(na_molar)
    inv = 1.0 / tm_1m + (4.29 * fgc - 3.95) * 1e-5 * ln_na + 9.40e-6 * ln_na**2
    return 1.0 / inv - KELVIN


def gc_fraction(seq: str) -> float:
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    return gc / (gc + at) if (gc + at) else 0.0


def per_step_thermodynamics(seq: str) -> Dict[str, np.ndarray]:
    """Per-base-step dH, dS and dG37, which is what a positional channel needs."""
    s = seq.upper()
    n = len(s) - 1
    dh = np.zeros(n)
    ds = np.zeros(n)
    for i in range(n):
        step = s[i:i + 2]
        dh[i] = NN_DH[step]
        ds[i] = NN_DS[step]
    return {
        "dH_kcal_mol": dh,
        "dS_cal_mol_K": ds,
        "dG37_kcal_mol": dh - T37_K * ds / 1000.0,
    }


def sliding_window_thermodynamics(seq: str, window: int = 21,
                                  na_molar: float = 0.15) -> Dict[str, np.ndarray]:
    """Window-level duplex stability along a sequence, reported at window centres."""
    s = seq.upper()
    if len(s) < window:
        raise ValueError("sequence shorter than window")
    centres, dg, tm = [], [], []
    for i in range(len(s) - window + 1):
        sub = s[i:i + window]
        t = nn_thermodynamics(sub)
        centres.append(i + window // 2)
        dg.append(t["dG37_kcal_mol"])
        tm.append(melting_temperature(sub, na_molar=na_molar))
    return {
        "centre": np.asarray(centres),
        "dG37_kcal_mol": np.asarray(dg),
        "Tm_C": np.asarray(tm),
        "window": window,
        "na_molar": na_molar,
    }


# --------------------------------------------------------------------------------------
# cgNA+ ground state and stiffness
# --------------------------------------------------------------------------------------


def cgna_ground_state(seq: str, cgna_root: Optional[str] = None,
                      include_stiffness: bool = True,
                      parameter_set: Optional[str] = None) -> Dict[str, object]:
    """cgNA+ ground-state coordinate vector, parameter names and stiffness matrix.

    The coordinate vector interleaves intra-base-pair, inter-base-pair and phosphate
    degrees of freedom; `param_names` is returned alongside so downstream code never has
    to guess the layout.  The stiffness matrix is returned as a dense array only when
    asked for, since it is banded and large.

    Note the caveat the cgNA+ authors state: with the default flags the stiffness is the
    Hessian with respect to Lie-group tangent fluctuations, not additive differences of
    the stored coordinates, so evaluating a quadratic form with naive coordinate
    differences carries an O(|Omega|) error. Any use of these blocks has to respect that.
    """
    import cgna_atomic as ca

    ca.add_cgnaplus_to_path(cgna_root) if cgna_root else ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams  # noqa: E402

    # `parameter_set` is not optional in practice for modified sequences: the default
    # set has no entries for the modified alphabet and raises a KeyError on a key like
    # 'TM'. It is exposed here so a caller working with modified bases cannot reach the
    # default set by omission.
    kw = dict(aligned_strands=True, include_stiffness=include_stiffness)
    if parameter_set is not None:
        kw["parameter_set_name"] = parameter_set
    p = CGNAPlusParams(seq.upper(), **kw)
    out: Dict[str, object] = {
        "sequence": seq.upper(),
        "gs": np.asarray(p.gs, dtype=float),
        "param_names": list(p.param_names),
    }
    if include_stiffness:
        stiff = p.stiffmat
        dense = stiff.toarray() if hasattr(stiff, "toarray") else np.asarray(stiff)
        out["stiffness"] = dense
        out["stiffness_shape"] = dense.shape
        # Diagonal stiffness per coordinate is the cheap, well-defined summary.
        out["stiffness_diagonal"] = np.diag(dense).copy()
    return out


def groove_geometry(pdb_path: str) -> Dict[str, np.ndarray]:
    """Minor and major groove widths from phosphate-phosphate distances.

    The conventional definition uses P(i) on one strand against P(j) on the other with a
    fixed index offset: offset 3 tracks the minor groove and offset 8 the major groove in
    B-DNA.  Reported as the raw distance; subtracting twice the phosphate van der Waals
    radius (5.8 angstrom) gives the usual "width" convention, and both are returned so
    the convention in use is explicit.
    """
    import frames as fr

    residues = fr.read_pdb_residues(pdb_path)
    a_keys = sorted([k for k in residues if k[0] == "A"], key=lambda k: k[1])
    b_keys = sorted([k for k in residues if k[0] == "B"], key=lambda k: k[1])
    n = min(len(a_keys), len(b_keys))

    def p_of(keys, i):
        r = residues.get(keys[i])
        return r.get("P") if r else None

    out: Dict[str, np.ndarray] = {}
    for label, offset in (("minor", 3), ("major", 8)):
        vals, idx = [], []
        for i in range(n):
            j = n - 1 - i + offset
            if j < 0 or j >= len(b_keys):
                continue
            pa, pb = p_of(a_keys, i), p_of(b_keys, j)
            if pa is None or pb is None:
                continue
            vals.append(float(np.linalg.norm(pa - pb)))
            idx.append(i)
        out[f"{label}_groove_PP_distance_A"] = np.asarray(vals)
        out[f"{label}_groove_width_A"] = np.asarray(vals) - 5.8
        out[f"{label}_groove_index"] = np.asarray(idx)
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="GCGCAAATTTGCGCATATAGCGCAAATTTGCGC")
    ap.add_argument("--na", type=float, default=0.15)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--stiffness", action="store_true")
    a = ap.parse_args()

    t = nn_thermodynamics(a.sequence)
    print(f"sequence            : {a.sequence} ({len(a.sequence)} nt, "
          f"GC {gc_fraction(a.sequence)*100:.1f}%)")
    print(f"dH                  : {t['dH_kcal_mol']:.2f} kcal/mol")
    print(f"dS                  : {t['dS_cal_mol_K']:.2f} cal/(mol K)")
    print(f"dG37                : {t['dG37_kcal_mol']:.2f} kcal/mol")
    print(f"Tm at 1 M Na        : {melting_temperature(a.sequence, na_molar=1.0):.2f} C")
    print(f"Tm at {a.na*1000:.0f} mM Na     : "
          f"{melting_temperature(a.sequence, na_molar=a.na):.2f} C")
    w = sliding_window_thermodynamics(a.sequence, a.window, a.na)
    print(f"window dG37 range   : {w['dG37_kcal_mol'].min():.2f} to "
          f"{w['dG37_kcal_mol'].max():.2f} kcal/mol over {len(w['centre'])} windows")
    if a.stiffness:
        gs = cgna_ground_state(a.sequence)
        print(f"cgNA+ gs vector     : {gs['gs'].shape}")
        print(f"cgNA+ stiffness     : {gs['stiffness_shape']}")
        print(f"first param names   : {gs['param_names'][:12]}")
