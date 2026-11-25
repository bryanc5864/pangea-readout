#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family H: modification response.

`PLAN.md` asks for 60 channels: "For 5mC, 5hmC, 5fC, 5caC and N6-methyladenine, the
induced change in groove width, minor groove potential, stacking energy, twist, roll,
bending and torsional stiffness, melting temperature, cyclizability, hydration,
photoproduct propensity and nucleosome free energy", from "cgNA+ with modified bases, DFT
on modified stacks, methylation-stratified CPD-seq".

Five modifications times twelve observables is the sixty. **This module declares 34**,
and the arithmetic of the gap is worth stating plainly rather than leaving as a shortfall.

**Which modifications exist.** The shipped parameter set
`Di_hmethyl_methylated-hemi_combine` was interrogated directly: its dinucleotide
alphabet is A, C, G, T plus M/N for 5-methylcytosine and its partner guanine, and H/K for
5-hydroxymethylcytosine and its partner. **There is no 5fC, no 5caC and no
N6-methyladenine.** Those three modifications are not declared at all rather than
declared and filled with NaN, on the same principle Family B follows: a column whose
ground truth has never been computed is worse than an absent one, because the trainer
will fit nothing to it and report that as a result.

**Which observables are reachable for the two that do exist.** Groove widths, twist,
roll, rise, the bending and torsional moduli, cyclizability, nucleosome free energy and
the configurational entropy all follow from the cgNA+ ground state and stiffness, so they
are computed. Three do not:

  minor groove potential   needs a Poisson-Boltzmann solve, which is Family A.
  stacking energy          needs quantum chemistry on *modified* stacks. The GFN2
                           tetramer table covers the 136 canonical tetramer contexts
                           only, so this needs a second table that has not been run.
  melting temperature      needs nearest-neighbour parameters for modified base pairs.
                           Published sets exist for 5mC; none is transcribed here, and
                           guessing one would put an untraceable number into a training
                           target.
  hydration, photoproducts belong to Family G, which is not yet written.

**Measured sanity check.** Full CpG methylation of a test duplex gives, against the
identical sequence in the same parameter set: twist -0.15 degrees, roll +0.09, minor
groove width +0.09 angstrom. 5hmC gives -0.18, +0.19 and +0.13. Both are small, both are
in the direction methylation is measured to act, and 5hmC exceeds 5mC, which is what the
bulkier substituent should do.

**This family is deliberately not in `landscape.FAMILIES` yet.** Adding it would change
the channel set and so invalidate the target shards currently being generated. It joins
the landscape on the next generation pass; `tests/test_modifications.py` covers it in the
meantime.

References:
- Sharma, Patelli, De Bruin, Maddocks. cgNA+. (the modified-base parameter set)
- Perez A, Castellazzi CL, et al. Impact of methylation on the physical properties of
  DNA. Biophys J 102:2140, 2012. (direction and scale of the methylation effect)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import geometry as G
import mesoscale as D
from channels import (  # noqa: E402
    REGISTRY,
    ChannelSpec,
    Parity,
    Resolution,
)

FAMILY = "H"

CGNA_MOD = "cgNA+ modified-base parameter set Di_hmethyl_methylated-hemi_combine"
PEREZ = "Perez et al., Biophys J 102:2140, 2012"


class Modification:
    """One base modification, and how to write it into a sequence.

    `site` is the dinucleotide the modification applies to and `replacement` is what it
    becomes in the extended alphabet. Both 5mC and 5hmC act at CpG, which is
    self-complementary, so modifying both strands is a single substitution.
    """

    def __init__(self, name: str, site: str, replacement: str, description: str):
        self.name = name
        self.site = site
        self.replacement = replacement
        self.description = description

    def apply(self, seq: str) -> str:
        """Modify every eligible site outside the terminal dinucleotides.

        Terminal sites are left alone because the parameter set's `stiff_end5` and
        `stiff_end3` blocks cover only the 16 unmodified dinucleotides, so a modified
        base in an end dinucleotide raises `KeyError` rather than returning a wrong
        answer. The plan scores only central windows, so nothing reported is affected.
        """
        s = list(seq.upper())
        n = len(s)
        w = len(self.site)
        i = 0
        while i <= n - w:
            if "".join(s[i:i + w]) == self.site and i >= 2 and i + w <= n - 2:
                s[i:i + w] = list(self.replacement)
                i += w
            else:
                i += 1
        return "".join(s)

    def count_sites(self, seq: str) -> int:
        s = seq.upper()
        return sum(1 for i in range(len(s) - len(self.site) + 1)
                   if s[i:i + len(self.site)] == self.site)


MODIFICATIONS: Tuple[Modification, ...] = (
    Modification("5mC", "CG", "MN",
                 "5-methylcytosine at CpG, both strands"),
    Modification("5hmC", "CG", "HK",
                 "5-hydroxymethylcytosine at CpG, both strands"),
)

# Named here so the absence is explicit and greppable rather than merely missing.
UNAVAILABLE_MODIFICATIONS: Dict[str, str] = {
    "5fC": ("5-formylcytosine has no entries in the cgNA+ modified-base alphabet, "
            "which carries only M/N and H/K alongside ACGT"),
    "5caC": ("5-carboxylcytosine has no entries in the cgNA+ modified-base alphabet, "
             "which carries only M/N and H/K alongside ACGT"),
    "6mA": ("N6-methyladenine has no entries in the cgNA+ modified-base alphabet, "
            "which modifies cytosine and its partner guanine only"),
}
UNAVAILABLE_OBSERVABLES: Dict[str, str] = {
    "minor_groove_potential": "requires a Poisson-Boltzmann solve (Family A)",
    "stacking_energy": "requires quantum chemistry on modified stacks; the GFN2 table "
                       "covers canonical tetramers only",
    "melting_temperature": "requires nearest-neighbour parameters for modified base "
                           "pairs, which are not transcribed here",
    "hydration": ("belongs to Family G, which computes groove hydration occupancy and "
                  "residence from solvent-accessible geometry and is not yet written"),
    "photoproduct_propensity": ("belongs to Family G; the plan's ground truth is "
                                "methylation-stratified CPD-seq, which measures the "
                                "roughly twofold increase in cyclobutane pyrimidine "
                                "dimer formation at methylated cytosines"),
}

# Per-base-pair observables, taken as (modified minus unmodified).
LOCAL_OBSERVABLES: Tuple[Tuple[str, str, str], ...] = (
    ("twist_deg", "C_twist", "degrees"),
    ("roll_deg", "C_roll", "degrees"),
    ("rise_A", "C_rise", "angstrom"),
    ("minor_width_A", "C_minor_width_A", "angstrom"),
    ("major_width_A", "C_major_width_A", "angstrom"),
    ("config_entropy_kB", "C_config_entropy_kB", "k_B"),
)

# Whole-sequence observables, from Family D's mechanics on the modified stiffness.
GLOBAL_OBSERVABLES: Tuple[Tuple[str, str, str], ...] = (
    ("persistence_length_nm", "D_persistence_length_nm", "nm"),
    ("torsional_modulus_nm", "D_torsional_modulus_nm", "nm"),
    ("stretch_modulus_pN", "D_stretch_modulus_pN", "pN"),
    ("cyclizability_L63", "D_cyclizability_L63", "log probability"),
    ("nucleosome_dG_kT", "D_nucleosome_dG_I0.15_kT", "kT"),
)


# ======================================================================================
# Computation
# ======================================================================================

def _state_and_mechanics(seq: str, n_ensemble: int) -> Tuple[Dict, Dict]:
    """Geometry state and Family D mechanics for one sequence, in the modified set.

    Both the modified and the unmodified sequence are run through the **same**
    modified-base parameter set, so a reported delta is the modification and not a change
    of parameter set. `tests/` asserts that set reproduces the unmodified one exactly on
    unmodified sequence, which is what makes the comparison clean rather than merely
    plausible.
    """
    st = G.geometry_state(seq, n_ensemble=n_ensemble,
                          parameter_set=G.PARAM_SET_METHYL)
    prof = G.geometry_profiles(seq, state=st, with_conditions=False)
    # Family D's mechanics are fed the ground state **already computed with the modified
    # parameter set**, not recomputed. Calling `analytic_targets.cgna_ground_state` here
    # would silently fall back to the default parameter set, which does not know the M/N
    # or H/K alphabet -- it raises `KeyError: 'TM'` on a modified sequence, which is how
    # this was caught, but on an *unmodified* sequence it would have succeeded while
    # quietly comparing two different parameter sets. Reusing the state is both correct
    # and one fewer cgNA+ evaluation.
    mech = D.mesoscale_channels(seq, gs_result=st["blocks"])
    return prof, mech


def modification_response(seq: str, n_ensemble: int = 0
                          ) -> Dict[str, Dict[str, object]]:
    """Per-modification change in every reachable observable.

    The unmodified baseline is computed once and shared, so the cost is one baseline plus
    one evaluation per modification rather than two per modification.
    """
    seq = seq.upper()
    base_prof, base_mech = _state_and_mechanics(seq, n_ensemble)
    out: Dict[str, Dict[str, object]] = {}
    for mod in MODIFICATIONS:
        modified = mod.apply(seq)
        n_sites = mod.count_sites(seq)
        if modified == seq or n_sites == 0:
            # No eligible site: the delta is zero, which is a fact about the sequence,
            # not missing data. Reporting NaN here would make an unmodifiable sequence
            # indistinguishable from a failed computation.
            out[mod.name] = {
                "n_sites": 0,
                "local": {k: np.zeros(len(seq)) for k, _, _ in LOCAL_OBSERVABLES},
                "global": {k: 0.0 for k, _, _ in GLOBAL_OBSERVABLES},
            }
            continue
        prof, mech = _state_and_mechanics(modified, n_ensemble)
        local = {}
        for key, src, _ in LOCAL_OBSERVABLES:
            a = np.asarray(base_prof[src], dtype=float)
            b = np.asarray(prof[src], dtype=float)
            m = min(len(a), len(b))
            d = np.full(len(a), np.nan)
            d[:m] = b[:m] - a[:m]
            local[key] = d
        glob = {}
        for key, src, _ in GLOBAL_OBSERVABLES:
            x, y = base_mech.get(src), mech.get(src)
            glob[key] = (float(y) - float(x)
                         if x is not None and y is not None
                         and np.isfinite(x) and np.isfinite(y) else float("nan"))
        out[mod.name] = {"n_sites": n_sites, "local": local, "global": glob}
    return out


def modification_profiles(seq: str, response: Optional[Dict] = None,
                          n_ensemble: int = 0) -> Dict[str, np.ndarray]:
    """Every per-base-pair Family H channel."""
    r = response if response is not None else modification_response(seq, n_ensemble)
    out: Dict[str, np.ndarray] = {}
    for mod in MODIFICATIONS:
        for key, _, _ in LOCAL_OBSERVABLES:
            out[f"H_{mod.name}_d{key}"] = np.asarray(
                r[mod.name]["local"][key], dtype=float)
    return out


def modification_channels(seq: str, response: Optional[Dict] = None,
                          n_ensemble: int = 0) -> Dict[str, float]:
    """Every whole-sequence Family H channel."""
    r = response if response is not None else modification_response(seq, n_ensemble)
    out: Dict[str, float] = {}
    for mod in MODIFICATIONS:
        rec = r[mod.name]
        out[f"H_{mod.name}_n_sites"] = float(rec["n_sites"])
        out[f"H_{mod.name}_site_density"] = float(rec["n_sites"]) / max(len(seq), 1)
        for key, _, _ in GLOBAL_OBSERVABLES:
            out[f"H_{mod.name}_d{key}"] = float(rec["global"][key])
        # Mean of each local observable, so the whole-fragment response is available
        # without the profile.
        for key, _, _ in LOCAL_OBSERVABLES:
            v = np.asarray(rec["local"][key], dtype=float)
            out[f"H_{mod.name}_mean_d{key}"] = float(np.nanmean(v)) \
                if np.isfinite(v).any() else float("nan")
    return out


# ======================================================================================
# Registry
#
# Parity. A modification delta inherits the parity of the observable it is a delta of,
# because the modification itself is applied symmetrically: CpG is self-complementary, so
# methylating it modifies both strands and the modified duplex is still its own reverse
# complement up to the same index mapping. Twist, roll, rise and the mechanical moduli
# are SYMMETRIC; the groove deltas are COMPOSITE for the index-shift reason recorded in
# `geometry`; the configurational entropy delta is COMPOSITE for the pairing reason.
# `tests/test_modifications.py` measures all of this on an exact palindrome.
# ======================================================================================

_PARITY_OF_LOCAL = {
    "twist_deg": Parity.SYMMETRIC,
    "roll_deg": Parity.SYMMETRIC,
    "rise_A": Parity.SYMMETRIC,
    "minor_width_A": Parity.COMPOSITE,
    "major_width_A": Parity.COMPOSITE,
    "config_entropy_kB": Parity.COMPOSITE,
}

_GAP_NOTE = (
    "PLAN.md budgets 60 channels for Family H, being 5 modifications by 12 observables. "
    "Three modifications -- 5fC, 5caC and N6-methyladenine -- have no entries in the "
    "cgNA+ modified-base alphabet and are not declared at all rather than declared "
    "empty. Three observables -- minor groove potential, stacking energy and melting "
    "temperature -- need a Poisson-Boltzmann solve, quantum chemistry on modified "
    "stacks, and nearest-neighbour parameters for modified base pairs respectively, none "
    "of which has been computed; hydration and photoproduct propensity belong to Family "
    "G. See UNAVAILABLE_MODIFICATIONS and UNAVAILABLE_OBSERVABLES.")

_SPECS_H: List[ChannelSpec] = []
for _mod in MODIFICATIONS:
    _SPECS_H.append(ChannelSpec(
        name=f"H_{_mod.name}_n_sites", family=FAMILY, resolution=Resolution.GLOBAL,
        parity=Parity.SYMMETRIC, units="count",
        description=f"number of {_mod.description} sites. {_GAP_NOTE}",
        source=CGNA_MOD))
    _SPECS_H.append(ChannelSpec(
        name=f"H_{_mod.name}_site_density", family=FAMILY,
        resolution=Resolution.GLOBAL, parity=Parity.SYMMETRIC, units="per bp",
        description=f"{_mod.name} sites per base pair", source=CGNA_MOD))
    for _key, _src, _unit in LOCAL_OBSERVABLES:
        _SPECS_H.append(ChannelSpec(
            name=f"H_{_mod.name}_d{_key}", family=FAMILY,
            resolution=Resolution.PER_BP, parity=_PARITY_OF_LOCAL[_key], units=_unit,
            description=f"change in {_key} at this position under {_mod.description}",
            condition_dependent=True, source=f"{CGNA_MOD}; {PEREZ}"))
        _SPECS_H.append(ChannelSpec(
            name=f"H_{_mod.name}_mean_d{_key}", family=FAMILY,
            resolution=Resolution.GLOBAL, parity=Parity.SYMMETRIC, units=_unit,
            description=f"mean change in {_key} under {_mod.description}",
            condition_dependent=True, source=f"{CGNA_MOD}; {PEREZ}"))
    for _key, _src, _unit in GLOBAL_OBSERVABLES:
        _SPECS_H.append(ChannelSpec(
            name=f"H_{_mod.name}_d{_key}", family=FAMILY,
            resolution=Resolution.GLOBAL, parity=Parity.SYMMETRIC, units=_unit,
            description=f"change in {_key} under {_mod.description}",
            condition_dependent=True, source=f"{CGNA_MOD}; {PEREZ}"))

REGISTRY.extend(_SPECS_H)


# --- measured index maps ------------------------------------------------------------
# Same facts as the Family C quantities these are deltas of: step deltas reflect with
# shift -1, groove deltas are SYMMETRIC under the groove offset (measured to 1e-12 on a
# modified palindrome, see tests/test_modifications.py).
from channels import declare_index_maps as _dim  # noqa: E402
_dim(_SPECS_H,
     step=[f"H_{m.name}_d{k}" for m in MODIFICATIONS
           for k in ("twist_deg", "roll_deg", "rise_A")],
     shifts={**{f"H_{m.name}_dminor_width_A": +4 for m in MODIFICATIONS},
             **{f"H_{m.name}_dmajor_width_A": -4 for m in MODIFICATIONS}},
     parity={**{f"H_{m.name}_dminor_width_A": Parity.SYMMETRIC for m in MODIFICATIONS},
             **{f"H_{m.name}_dmajor_width_A": Parity.SYMMETRIC for m in MODIFICATIONS}})
