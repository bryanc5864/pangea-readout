#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family B: electronic structure and quantum chemistry, tier 1.

`PLAN.md` budgets 98 channels for Family B and describes three tiers with delta-learning
between them: semi-empirical tight binding across the broadest context coverage,
dispersion-corrected DFT across all tetramer contexts at three stacking geometries, and
symmetry-adapted perturbation theory on a stratified subset.

**Only the first tier has been computed, and this module emits only what that tier
supports.** `results/qc_tetramer_gfn2.json` holds GFN2-xTB single points for all 136
reverse-complement-unique tetramer contexts, every one converged, built on cgNA+
ground-state geometry so the central step responds to its flanking context. That gives
total interaction energies, frontier orbitals, vertical ionization potentials and
electron affinities, dipoles and partial charges.

It does **not** give the decomposition into electrostatic, exchange-repulsion, induction
and dispersion terms, which is the quantity that makes Family B a chemistry result rather
than an energy table, and which needs the SAPT tier. It does not give transfer integrals,
molecular electrostatic potentials at the hydrogen-bonding atoms, pKa shifts or the
polarizability tensor either. Those channels are **not declared here**. Declaring a
channel and filling it with NaN is right when the quantity is defined but unavailable for
this sequence; declaring one whose ground truth has never been computed at all would put
a column in the training set that no run could ever fill.

**The whole family is a lookup, and that is the design rather than a shortcut.** As
`PLAN.md` puts it, these quantities "are properties of short stacked fragments, not of
long sequences, so the ground truth is computed once on an exhaustive fragment library
and then generalized to flanking context by the encoder". The table is exhaustive at
tetramer resolution: every one of the 256 tetramers resolves, 136 directly and 120
through reverse complement, with no gaps.

**Reverse complement is handled by swapping, not by averaging.** The table is keyed on
`min(t, rc(t))`. When a tetramer resolves through its reverse complement, the two base
pairs of the central step exchange, so the 5-prime and 3-prime hydrogen-bond energies and
the per-base-pair orbital quantities are swapped on the way out. Getting that wrong would
be invisible in any symmetric channel and would silently transpose the two asymmetric
ones.

References:
- Bannwarth C, Ehlert S, Grimme S. GFN2-xTB. J Chem Theory Comput 15:1652, 2019.
- Genereux JC, Barton JK. Mechanisms for DNA charge transport. Chem Rev 110:1642, 2010.
  (holes localize on guanine; pyrimidines act as barriers)
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from channels import (  # noqa: E402
    REGISTRY,
    WINDOW_LADDER_BP,
    ChannelSpec,
    Parity,
    Resolution,
)

FAMILY = "B"

GFN2 = "GFN2-xTB (Bannwarth, Ehlert & Grimme, JCTC 15:1652, 2019)"
BARTON = "Genereux & Barton, Chem Rev 110:1642, 2010"

DEFAULT_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "qc_tetramer_gfn2.json")

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

# Quantities that belong to one base pair of the step and therefore exchange when the
# tetramer is resolved through its reverse complement.
_SWAP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("hbond_bp1_kcal_mol", "hbond_bp2_kcal_mol"),
    ("bp1_homo_eV", "bp2_homo_eV"),
    ("bp1_lumo_eV", "bp2_lumo_eV"),
    ("bp1_gap_eV", "bp2_gap_eV"),
    ("bp1_vip_eV", "bp2_vip_eV"),
    ("bp1_vea_eV", "bp2_vea_eV"),
    ("bp1_dipole_debye", "bp2_dipole_debye"),
)

_TABLE_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[c] for c in reversed(seq.upper()))


def load_table(path: str = DEFAULT_TABLE) -> Dict[str, Dict[str, float]]:
    """The tetramer table, keyed by context, cached so a sweep pays the read once."""
    if path in _TABLE_CACHE:
        return _TABLE_CACHE[path]
    with open(path) as f:
        raw = json.load(f)
    table = {r["context"]: r for r in raw["results"] if r.get("status") == "ok"}
    if not table:
        raise RuntimeError(f"no converged contexts in {path}")
    _TABLE_CACHE[path] = table
    return table


def lookup(tetramer: str, table: Optional[Dict[str, Dict[str, float]]] = None
           ) -> Optional[Dict[str, float]]:
    """One tetramer's quantum chemistry, resolving through reverse complement if needed.

    When the reverse complement is what the table holds, the two base pairs of the
    central step have exchanged, so every per-base-pair quantity is swapped before being
    returned. The symmetric quantities -- stacking energy, the dimer's own orbitals and
    dipole -- are unaffected, which is exactly why the swap has to be explicit: nothing
    downstream would notice it missing.
    """
    tab = table if table is not None else load_table()
    t = tetramer.upper()
    if any(b not in COMPLEMENT for b in t):
        return None
    if t in tab:
        return dict(tab[t])
    rc = reverse_complement(t)
    if rc not in tab:
        return None
    rec = dict(tab[rc])
    for a, b in _SWAP_PAIRS:
        if a in rec and b in rec:
            rec[a], rec[b] = rec[b], rec[a]
    rec["resolved_via_rc"] = True
    return rec


def tetramer_contexts(seq: str) -> List[Tuple[int, str]]:
    """(central-step index, tetramer) for each position with a full flanking context.

    The tetramer at sequence positions i..i+3 has its central step between i+1 and i+2,
    so it is reported at index i+1. The first base pair and the last two have no complete
    context and stay NaN rather than borrowing a neighbour's value.
    """
    s = seq.upper()
    return [(i + 1, s[i:i + 4]) for i in range(len(s) - 3)]


# ======================================================================================
# Charge transport
# ======================================================================================

# Guanine is the lowest-ionization-potential base, so holes localize there; the barrier a
# pyrimidine presents is the difference in site energy. Both come out of the table rather
# than from a tabulated constant.
KT_EV_298 = 0.025693


def hole_site_energies(vip: np.ndarray) -> np.ndarray:
    """Hole site energy in eV: the energy to put a hole on this step, which is its VIP."""
    return np.asarray(vip, dtype=float)


def hole_localization(vip: np.ndarray, kT_eV: float = KT_EV_298) -> np.ndarray:
    """Equilibrium probability that a hole sits on each step, by Boltzmann weighting.

    A hole goes where the ionization potential is lowest, so the weight is
    exp(-(VIP_i - VIP_min)/kT). This is the equilibrium distribution over sites, not a
    transport calculation: it says where a hole rests, not how fast it gets there, and
    the plan's hopping rates need the transfer integrals that only the DFT tier supplies.
    """
    v = np.asarray(vip, dtype=float)
    m = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if not m.any():
        return out
    w = np.exp(-(v[m] - np.nanmin(v[m])) / kT_eV)
    tot = w.sum()
    out[m] = w / tot if tot > 0 else np.nan
    return out


def oxidative_susceptibility(vip: np.ndarray) -> np.ndarray:
    """How much more oxidizable each step is than the fragment's mean, in eV.

    Negative means easier to oxidize than average. Reported as a difference so it is a
    property of sequence context rather than of the absolute GFN2 energy scale, which
    carries a systematic offset that the DFT tier would correct.
    """
    v = np.asarray(vip, dtype=float)
    if not np.isfinite(v).any():
        return np.full(v.shape, np.nan)
    return v - float(np.nanmean(v))


# ======================================================================================
# Assembly
# ======================================================================================

_SCALAR_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("B_stacking_kcal_mol", "stacking_kcal_mol"),
    ("B_hbond_5p_kcal_mol", "hbond_bp1_kcal_mol"),
    ("B_hbond_3p_kcal_mol", "hbond_bp2_kcal_mol"),
    ("B_homo_eV", "dimer_homo_eV"),
    ("B_lumo_eV", "dimer_lumo_eV"),
    ("B_gap_eV", "dimer_gap_eV"),
    ("B_vip_eV", "dimer_vip_eV"),
    ("B_vea_eV", "dimer_vea_eV"),
    ("B_dipole_debye", "dimer_dipole_debye"),
)


def electronic_profiles(seq: str, table: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Every per-base-pair Family B channel, each an array of length n_bp."""
    seq = seq.upper()
    n = len(seq)
    tab = table if table is not None else load_table()
    out = {name: np.full(n, np.nan) for name, _ in _SCALAR_FIELDS}
    for extra in ("B_hbond_mean_kcal_mol", "B_bp_homo_eV", "B_bp_lumo_eV",
                  "B_bp_gap_eV", "B_bp_vip_eV", "B_bp_vea_eV", "B_bp_dipole_debye"):
        out[extra] = np.full(n, np.nan)

    for idx, tet in tetramer_contexts(seq):
        rec = lookup(tet, tab)
        if rec is None:
            continue
        for name, key in _SCALAR_FIELDS:
            v = rec.get(key)
            if v is not None:
                out[name][idx] = float(v)
        b1, b2 = rec.get("hbond_bp1_kcal_mol"), rec.get("hbond_bp2_kcal_mol")
        if b1 is not None and b2 is not None:
            out["B_hbond_mean_kcal_mol"][idx] = 0.5 * (float(b1) + float(b2))
        for short, key in (("homo_eV", "homo_eV"), ("lumo_eV", "lumo_eV"),
                           ("gap_eV", "gap_eV"), ("vip_eV", "vip_eV"),
                           ("vea_eV", "vea_eV"), ("dipole_debye", "dipole_debye")):
            a, b = rec.get(f"bp1_{key}"), rec.get(f"bp2_{key}")
            if a is not None and b is not None:
                out[f"B_bp_{short}"][idx] = 0.5 * (float(a) + float(b))

    out["B_hole_site_energy_eV"] = hole_site_energies(out["B_vip_eV"])
    out["B_hole_localization_prob"] = hole_localization(out["B_vip_eV"])
    out["B_oxidative_susceptibility_eV"] = oxidative_susceptibility(out["B_vip_eV"])
    out["B_electron_site_energy_eV"] = -np.asarray(out["B_vea_eV"], dtype=float)
    return out


def electronic_channels(seq: str, table: Optional[Dict] = None,
                        symmetrise: bool = False) -> Dict[str, float]:
    """Every whole-sequence Family B scalar channel."""
    if symmetrise:
        from channels import symmetrise_channels
        fwd = electronic_channels(seq, table, False)
        rev = electronic_channels(reverse_complement(seq), table, False)
        return symmetrise_channels(fwd, rev)

    p = electronic_profiles(seq, table)
    out: Dict[str, float] = {}
    out["B_total_stacking_kcal_mol"] = float(np.nansum(p["B_stacking_kcal_mol"]))
    out["B_mean_stacking_kcal_mol"] = float(np.nanmean(p["B_stacking_kcal_mol"]))
    out["B_min_stacking_kcal_mol"] = float(np.nanmin(p["B_stacking_kcal_mol"]))
    out["B_max_stacking_kcal_mol"] = float(np.nanmax(p["B_stacking_kcal_mol"]))
    out["B_total_hbond_kcal_mol"] = float(np.nansum(p["B_hbond_mean_kcal_mol"]))
    out["B_mean_gap_eV"] = float(np.nanmean(p["B_gap_eV"]))
    out["B_min_gap_eV"] = float(np.nanmin(p["B_gap_eV"]))
    out["B_mean_vip_eV"] = float(np.nanmean(p["B_vip_eV"]))
    out["B_min_vip_eV"] = float(np.nanmin(p["B_vip_eV"]))
    out["B_mean_vea_eV"] = float(np.nanmean(p["B_vea_eV"]))
    out["B_mean_dipole_debye"] = float(np.nanmean(p["B_dipole_debye"]))
    out["B_cumulative_dipole_debye"] = float(np.nansum(p["B_dipole_debye"]))
    # How deep the best hole trap is relative to the rest of the fragment. A large
    # positive value means one site dominates; near zero means the hole is delocalized.
    out["B_hole_trap_depth_eV"] = float(
        np.nanmean(p["B_vip_eV"]) - np.nanmin(p["B_vip_eV"]))
    out["B_max_hole_localization"] = float(np.nanmax(p["B_hole_localization_prob"]))
    return out


def electronic_window_channels(seq: str, ladder: Sequence[int] = WINDOW_LADDER_BP,
                               table: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Family B on the revision-2 window ladder."""
    p = electronic_profiles(seq, table)
    keys = ("B_stacking_kcal_mol", "B_vip_eV", "B_gap_eV", "B_hbond_mean_kcal_mol")
    n = len(seq)
    out: Dict[str, np.ndarray] = {}
    for w in ladder:
        if w > n:
            continue
        for k in keys:
            v = p[k]
            means = np.full(n, np.nan)
            for i in range(n - w + 1):
                seg = v[i:i + w]
                if np.isfinite(seg).any():
                    means[i + w // 2] = float(np.nanmean(seg))
            out[f"{k}_w{w}"] = means
    return out


# ======================================================================================
# Registry
#
# Parity is measured on an exact palindrome, as everywhere else. The step-indexed
# quantities are SYMMETRIC; the 5-prime and 3-prime hydrogen-bond energies exchange with
# each other, which is what `lookup` handles and what makes them STRANDED.
# ======================================================================================

_TIER1 = ("computed with GFN2-xTB, the cheap tier. PLAN.md supervises a correction to "
          "this tier from dispersion-corrected DFT and SAPT; neither has been run, so "
          "the absolute energy scale carries a systematic semi-empirical offset even "
          "where the sequence ranking is informative.")


def _b(name: str, resolution: Resolution, units: str, desc: str,
       parity: Parity = Parity.SYMMETRIC, source: str = GFN2,
       provisional: bool = False, reason: str = "") -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=resolution, parity=parity,
                       units=units, description=desc, provisional=provisional,
                       provisional_reason=reason, source=source)


_SPECS_B: List[ChannelSpec] = [
    _b("B_stacking_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "total base-stacking interaction energy of this step in its tetramer context"),
    _b("B_hbond_5p_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "Watson-Crick hydrogen-bond energy of the 5-prime base pair of this step",
       Parity.STRANDED),
    _b("B_hbond_3p_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "Watson-Crick hydrogen-bond energy of the 3-prime base pair of this step",
       Parity.STRANDED),
    _b("B_hbond_mean_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "mean hydrogen-bond energy of the two base pairs of this step"),
    _b("B_homo_eV", Resolution.PER_BP, "eV", "HOMO of the stacked dimer"),
    _b("B_lumo_eV", Resolution.PER_BP, "eV", "LUMO of the stacked dimer"),
    _b("B_gap_eV", Resolution.PER_BP, "eV", "frontier orbital gap of the stacked dimer"),
    _b("B_vip_eV", Resolution.PER_BP, "eV",
       "vertical ionization potential of the stacked dimer"),
    _b("B_vea_eV", Resolution.PER_BP, "eV",
       "vertical electron affinity of the stacked dimer"),
    _b("B_dipole_debye", Resolution.PER_BP, "debye", "dipole moment of the dimer"),
    _b("B_bp_homo_eV", Resolution.PER_BP, "eV", "mean base-pair HOMO"),
    _b("B_bp_lumo_eV", Resolution.PER_BP, "eV", "mean base-pair LUMO"),
    _b("B_bp_gap_eV", Resolution.PER_BP, "eV", "mean base-pair frontier gap"),
    _b("B_bp_vip_eV", Resolution.PER_BP, "eV", "mean base-pair ionization potential"),
    _b("B_bp_vea_eV", Resolution.PER_BP, "eV", "mean base-pair electron affinity"),
    _b("B_bp_dipole_debye", Resolution.PER_BP, "debye", "mean base-pair dipole"),
    _b("B_hole_site_energy_eV", Resolution.PER_BP, "eV",
       "energy to place a hole on this step", Parity.SYMMETRIC, BARTON),
    _b("B_electron_site_energy_eV", Resolution.PER_BP, "eV",
       "energy to place an excess electron on this step", Parity.SYMMETRIC, BARTON),
    _b("B_hole_localization_prob", Resolution.PER_BP, "probability",
       "equilibrium probability a hole rests on this step; where it rests, not how fast "
       "it gets there, since hopping rates need the transfer integrals the DFT tier "
       "would supply", Parity.SYMMETRIC, BARTON),
    _b("B_oxidative_susceptibility_eV", Resolution.PER_BP, "eV",
       "ionization potential relative to the fragment mean; negative is more oxidizable",
       Parity.SYMMETRIC, BARTON),
    _b("B_total_stacking_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "summed stacking energy"),
    _b("B_mean_stacking_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "mean stacking energy per step"),
    _b("B_min_stacking_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "most stacked step"),
    _b("B_max_stacking_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "least stacked step, where the duplex is easiest to unstack"),
    _b("B_total_hbond_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "summed hydrogen-bond energy"),
    _b("B_mean_gap_eV", Resolution.GLOBAL, "eV", "mean frontier orbital gap"),
    _b("B_min_gap_eV", Resolution.GLOBAL, "eV", "smallest frontier gap"),
    _b("B_mean_vip_eV", Resolution.GLOBAL, "eV", "mean ionization potential"),
    _b("B_min_vip_eV", Resolution.GLOBAL, "eV",
       "most oxidizable step, the likeliest site of guanine oxidation damage",
       Parity.SYMMETRIC, BARTON),
    _b("B_mean_vea_eV", Resolution.GLOBAL, "eV", "mean electron affinity"),
    _b("B_mean_dipole_debye", Resolution.GLOBAL, "debye", "mean step dipole"),
    _b("B_cumulative_dipole_debye", Resolution.GLOBAL, "debye",
       "summed step dipole magnitude; a scalar proxy for the helical dipole, which "
       "would need the vector sum in a common frame"),
    _b("B_hole_trap_depth_eV", Resolution.GLOBAL, "eV",
       "mean minus minimum ionization potential; large means one site traps the hole, "
       "near zero means it stays delocalized", Parity.SYMMETRIC, BARTON),
    _b("B_max_hole_localization", Resolution.GLOBAL, "probability",
       "largest single-site hole probability", Parity.SYMMETRIC, BARTON),
]

for _s in _SPECS_B:
    object.__setattr__(_s, "description", _s.description + ". " + _TIER1)

REGISTRY.extend(_SPECS_B)


# --- measured index maps ------------------------------------------------------------
# Every Family B per-base-pair channel except the two strand-exchanging hydrogen-bond
# energies is a property of the tetramer's central *step*, reported at its 5-prime base
# pair, so it reflects as i -> n-2-i. Measured exact on random palindromes
# (results/rc_maps.json). Before this was declared, PANGEA's merge averaged each of these
# with its neighbour's value.
from channels import declare_index_maps as _dim  # noqa: E402
_dim(_SPECS_B, step=[s.name for s in _SPECS_B
                     if s.resolution is Resolution.PER_BP
                     and s.parity in (Parity.SYMMETRIC, Parity.ANTISYMMETRIC)])
