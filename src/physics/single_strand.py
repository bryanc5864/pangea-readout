#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family J: single-stranded DNA.

PLAN.md: "J. Single-stranded DNA -- stacking free energy and stacking propensity,
persistence length and stretching elasticity, charge density and electrostatics without a
complementary strand, hydration, chemical reactivity, self-structure propensity. Ground
truth: single-strand coarse-grained parameters, force spectroscopy, chemical probing."

**Implemented here (stage 1): self-structure.** Each strand of the fragment, as a single
DNA molecule, folded with ViennaRNA's DNA parameter set (Mathews et al. 2004 nearest-
neighbour DNA energies with their hairpin, internal-loop, mismatch and salt tables -- the
same published set Family E's hairpin channels use): per-base unpaired probability and
positional entropy, ensemble and minimum free energies, ensemble diversity and the MFE
structure's helix and loop statistics, at 0.15 M monovalent salt and 37 C, with a salt and
a temperature axis. **Not yet:** single-strand stacking (needs a sourced stacking table),
persistence length and stretching (needs force-spectroscopy-parameterised coarse
graining), electrostatics without a complementary strand (Family A's pipeline), hydration
and reactivity.

**Strands.** "J_p_" is the given strand x, "J_m_" the other strand rc(x), each folded and
stored in its own 5'->3' frame; so J_m(x) = J_p(rc x) exactly, a stranded pair handled by
PANGEA's strand tie like Family L's transcripts.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import rna_structure as R
import thermodynamics as T
from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "J"
SALT_PRIMARY = 0.15
SALTS = (0.01, 1.0)
TEMPERATURES_C = (25.0, 55.0)
SRC = ("ViennaRNA 2.8 DNA parameter set: Mathews et al. 2004 nearest-neighbour DNA "
       "parameters, with the library's salt correction")
_OXSRC = ("oxDNA2 sequence-dependent stacking (Snodin et al., J Chem Phys 142:234901, 2015; "
          "Sulc et al., J Chem Phys 137:135101, 2012)")
_COMP = str.maketrans("ACGT", "TGCA")
STACK_TEMPERATURES_C = (25.0, 37.0, 55.0)
OXDNA_UNIT_KCAL = 4.142e-20 * 6.02214076e23 / 4184.0     # oxDNA energy unit, kcal/mol
_OX = None


def _oxdna():
    global _OX
    if _OX is None:
        import json as _json
        import os as _os
        _OX = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                            "data", "oxdna2_stacking.json")))
    return _OX


def step_stacking_kcal(seq: str, temperature_C: float) -> np.ndarray:
    """oxDNA2 single-strand stacking strength of every 5'->3' step, kcal/mol (n - 1,).

    oxDNA applies STCK_X_Y to a nucleotide p and its 3' neighbour q as F1_EPS[q][p], so a
    5'-YX-3' step takes STCK_X_Y, scaled as (1 - f + 9 f T / 3000 K) (DNAInteraction.cpp).
    That ordering reproduces SantaLucia's: 5'-AT-3' stacks harder than 5'-TA-3', 5'-GC-3'
    harder than 5'-CG-3'.
    """
    d = _oxdna()["values"]
    f = d["STCK_FACT_EPS"]
    scale = 1.0 - f + 9.0 * f * (temperature_C + 273.15) / 3000.0
    s = seq.upper()
    return np.array([d[f"STCK_{s[i + 1]}_{s[i]}"] * scale * OXDNA_UNIT_KCAL
                     for i in range(len(s) - 1)], dtype=float)

_SCALARS = (("mfe_kcal", "kcal/mol", "minimum free energy of the single strand's fold"),
            ("ensemble_kcal", "kcal/mol", "ensemble free energy of the single strand"),
            ("mfe_ensemble_gap", "kcal/mol", "MFE minus ensemble free energy"),
            ("frequency_of_mfe", "probability", "Boltzmann weight of the MFE structure"),
            ("ensemble_diversity", "bp", "mean base-pair distance within the ensemble"),
            ("centroid_distance", "bp", "distance from the centroid to the ensemble"),
            ("mean_positional_entropy", "bits", "mean per-base pairing entropy"),
            ("max_positional_entropy", "bits", "maximum per-base pairing entropy"),
            ("mean_unpaired_prob", "probability", "mean probability of being unpaired"),
            ("n_helices", "count", "helices in the MFE structure"),
            ("mean_helix_len", "bp", "mean helix length in the MFE structure"),
            ("max_helix_len", "bp", "longest helix in the MFE structure"),
            ("n_hairpins", "count", "hairpin loops in the MFE structure"),
            ("mean_hairpin_size", "nt", "mean hairpin loop size"),
            ("min_hairpin_size", "nt", "smallest hairpin loop"),
            ("paired_fraction", "fraction", "paired fraction of the MFE structure"),
            ("exterior_fraction", "fraction", "exterior-loop fraction of the MFE structure"))


def reverse_complement(s: str) -> str:
    return s.upper().translate(_COMP)[::-1]


def _declare() -> None:
    for tag in ("p", "m"):
        which = "the given strand" if tag == "p" else "the reverse-complement strand"
        for name, units, desc in (("unpaired_prob", "probability", "probability of being unpaired"),
                                  ("positional_entropy", "bits", "per-base pairing entropy")):
            REGISTRY.register(ChannelSpec(
                name=f"J_{tag}_{name}", family=FAMILY, resolution=Resolution.PER_BP,
                parity=Parity.STRANDED, units=units,
                description=f"single-stranded DNA self-structure, {which}: {desc}", source=SRC))
        for name, units, desc in _SCALARS:
            REGISTRY.register(ChannelSpec(
                name=f"J_{tag}_{name}", family=FAMILY, resolution=Resolution.GLOBAL,
                parity=Parity.STRANDED, units=units,
                description=f"single-stranded DNA self-structure, {which}: {desc}", source=SRC))
        for s in SALTS:
            for q in ("mfe", "ensemble"):
                REGISTRY.register(ChannelSpec(
                    name=f"J_{tag}_{q}_salt{s:g}", family=FAMILY, resolution=Resolution.GLOBAL,
                    parity=Parity.STRANDED, units="kcal/mol", condition_dependent=True,
                    description=f"single-strand {q} free energy at {s:g} M salt, {which}",
                    source=SRC))
        for t in STACK_TEMPERATURES_C:
            REGISTRY.register(ChannelSpec(
                name=f"J_{tag}_stack_eps_T{t:g}", family=FAMILY, resolution=Resolution.PER_BP,
                parity=Parity.STRANDED, units="kcal/mol", source=_OXSRC,
                description=f"oxDNA2 single-strand stacking strength at {t:g} C, mean of the steps "
                            f"touching this base, {which}"))
            REGISTRY.register(ChannelSpec(
                name=f"J_{tag}_mean_stack_eps_T{t:g}", family=FAMILY, resolution=Resolution.GLOBAL,
                parity=Parity.STRANDED, units="kcal/mol", source=_OXSRC,
                description=f"oxDNA2 single-strand stacking strength at {t:g} C, mean over steps, {which}"))
        for t in TEMPERATURES_C:
            for q, u in (("mfe", "kcal/mol"), ("paired_fraction", "fraction")):
                REGISTRY.register(ChannelSpec(
                    name=f"J_{tag}_{q}_T{t:g}", family=FAMILY, resolution=Resolution.GLOBAL,
                    parity=Parity.STRANDED, units=u, condition_dependent=True,
                    description=f"single-strand {q} at {t:g} C, {which}", source=SRC))


def single_strand_channels(seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Every Family J channel for one fragment: (per-bp profiles, whole-sequence)."""
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}
    with T._DNAParameters():
        for tag, s in (("p", seq.upper()), ("m", reverse_complement(seq))):
            base = R.fold_once(s, salt_M=SALT_PRIMARY)
            prof[f"J_{tag}_unpaired_prob"] = np.asarray(base["unpaired_profile"], dtype=float)
            prof[f"J_{tag}_positional_entropy"] = np.asarray(base["positional_entropy"], dtype=float)
            for name, _, _ in _SCALARS:
                glob[f"J_{tag}_{name}"] = float(base[name])
            for sv in SALTS:
                r = R.fold_once(s, salt_M=sv)
                glob[f"J_{tag}_mfe_salt{sv:g}"] = float(r["mfe_kcal"])
                glob[f"J_{tag}_ensemble_salt{sv:g}"] = float(r["ensemble_kcal"])
            for t in STACK_TEMPERATURES_C:
                st = step_stacking_kcal(s, t)
                per = np.empty(len(s))
                per[0], per[-1] = st[0], st[-1]
                per[1:-1] = 0.5 * (st[:-1] + st[1:])
                prof[f"J_{tag}_stack_eps_T{t:g}"] = per
                glob[f"J_{tag}_mean_stack_eps_T{t:g}"] = float(st.mean())
            for t in TEMPERATURES_C:
                r = R.fold_once(s, salt_M=SALT_PRIMARY, temperature_C=t)
                glob[f"J_{tag}_mfe_T{t:g}"] = float(r["mfe_kcal"])
                glob[f"J_{tag}_paired_fraction_T{t:g}"] = float(r["paired_fraction"])
    return prof, glob


_declare()
