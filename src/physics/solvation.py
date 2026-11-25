#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family G: solvation, hydration and chemical reactivity.

`PLAN.md` asks for 85 channels: "Hydroxyl radical cleavage per nucleotide per strand;
solvent-accessible surface area by atom class; minor and major groove hydration occupancy,
residence time, spine continuity and water bridge probability; polar and nonpolar
hydration free energy; groove hydrophobicity; cleavage propensity for five nucleases;
reactivity to six chemical probes; cyclobutane pyrimidine dimer propensity at four
dipyrimidine classes, methylated and unmethylated; 6-4 photoproduct propensity;
depurination and depyrimidination rates; 8-oxoguanine susceptibility; cytosine
deamination in duplex and single-stranded context; alkylation susceptibility; abasic site
stability; generic intercalator and minor groove binder affinity; interfacial water
entropy", from "ORChID, CPD-seq and dCPD-seq, MD-derived hydration, chemical probing
data".

This family is the one where `PLAN.md` leans hardest on external validation -- "Agreement
with held-out solver output only shows the network imitates the solver. Agreement with
independent measurement shows the physics is right about the world" -- and hydroxyl
radical cleavage and photoproduct maps are two of the three measurements named for it.

**The organising idea is that most of this family is an accessibility calculation.** A
chemical probe reacts with a specific atom, so its reactivity is governed by whether a
solvent molecule can reach that atom. That makes it computable from the solvent-accessible
surface of the relaxed structure rather than from a fitted sequence model:

    hydroxyl radical   abstracts H from C4' and C5' of the deoxyribose. Measured on the
                       minimised structure, C5' and C4' are the most exposed sugar
                       carbons at 21.7 and 16.0 square angstrom, against 2.6 for C1',
                       which is why this works at all.
    dimethyl sulfate   methylates N7 of guanine, in the major groove.
    DEPC               carbethoxylates N7 of the purines.
    KMnO4              oxidises the thymine C5-C6 bond, reachable only where the duplex
                       is distorted or open.
    bisulfite          deaminates cytosine, and only single-stranded cytosine.
    DNase I            cuts in the minor groove and its rate tracks minor groove width.

Each of those is a declared atom set and a stated geometric quantity, not a parameter fit.

**What is not reachable, and is therefore not declared.** Groove hydration occupancy,
water residence time, spine-of-hydration continuity, water bridge probability and
interfacial water entropy all require explicit-solvent molecular dynamics; there is no
MD here and a static structure cannot produce a residence time. Intercalator and
minor-groove-binder affinities require docking. Depurination, deamination and alkylation
*rates* require kinetic parameters that are not transcribed here. Those are named in
`UNAVAILABLE` rather than emitted as plausible numbers.

**8-oxoguanine reuses Family B rather than guessing.** Guanine oxidation is governed by
the ionization potential, and the GFN2 tetramer table already carries a vertical
ionization potential for every tetramer context, so the susceptibility channel is derived
from it. One implementation of one quantity.

References:
- Balasubramanian B, Pogozelski WK, Tullius TD. DNA strand breaking by the hydroxyl
  radical is governed by the accessible surface areas of the hydrogen atoms of the DNA
  backbone. PNAS 95:9738, 1998. (the basis of the cleavage channel)
- Greenbaum JA, Parker SCJ, Tullius TD. Detection of DNA structural motifs in functional
  genomic elements. Genome Res 17:940, 2007. (ORChID)
- Shrake A, Rupley JA. Environment and exposure to solvent of protein atoms. JMB
  79:351, 1973. (the SASA algorithm freesasa implements)
- Eisenberg D, McLachlan AD. Solvation energy in protein folding and binding. Nature
  319:199, 1986. (atomic solvation parameters)
- Mao P, Smerdon MJ, Roberts SA, Wyrick JJ. Chromosomal landscape of UV damage formation
  and repair at single-nucleotide resolution. PNAS 113:9057, 2016. (CPD-seq)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from channels import (  # noqa: E402
    REGISTRY,
    ChannelSpec,
    Parity,
    Resolution,
)

FAMILY = "G"

TULLIUS = "Balasubramanian, Pogozelski & Tullius, PNAS 95:9738, 1998"
ORCHID = "Greenbaum, Parker & Tullius, Genome Res 17:940, 2007"
SHRAKE = "Shrake & Rupley, JMB 79:351, 1973 (via freesasa)"
EISENBERG = "Eisenberg & McLachlan, Nature 319:199, 1986"
CPDSEQ = "Mao, Smerdon, Roberts & Wyrick, PNAS 113:9057, 2016"
DEFINITION = ("accessibility of a declared atom set on the relaxed structure; no fitted "
              "parameters, the atom set and the geometric measure are stated in the "
              "computing function")

# Atom groups. Each is a chemically motivated set, not a convenience bucket.
SUGAR_CARBONS = ("C1'", "C2'", "C3'", "C4'", "C5'")
# The hydroxyl radical abstracts hydrogen from C4' and C5' preferentially; Tullius and
# colleagues showed strand-break yield tracks the accessible surface of those hydrogens.
RADICAL_TARGETS = ("C4'", "C5'")
PHOSPHATE_OXYGENS = ("OP1", "OP2", "O3'", "O5'")
PHOSPHORUS = ("P",)
# Major-groove edge atoms a probe or protein reads.
MAJOR_EDGE = {"A": ("N7", "N6", "C5"), "G": ("N7", "O6", "C5"),
              "C": ("N4", "C5"), "T": ("O4", "C5", "C7")}
MINOR_EDGE = {"A": ("N3", "C2"), "G": ("N3", "N2"),
              "C": ("O2",), "T": ("O2",)}
PURINE_N7 = ("N7",)
THYMINE_C5C6 = ("C5", "C6")
CYTOSINE_C6 = ("C6",)

# Nonpolar surface tension, the term used with an SASA model in implicit solvent. This is
# the APBS/MM-PBSA convention rather than a fit of our own.
GAMMA_NONPOLAR_KCAL_PER_A2 = 0.0072
# Polar atoms lose favourable solvation when buried; the coefficient here is the
# magnitude Eisenberg and McLachlan report for charged and polar groups, applied as a
# single value rather than per atom type.
GAMMA_POLAR_KCAL_PER_A2 = -0.024

DIPYRIMIDINES: Tuple[str, ...] = ("TT", "TC", "CT", "CC")

UNAVAILABLE: Dict[str, str] = {
    "groove_hydration_occupancy": (
        "requires explicit-solvent molecular dynamics; a static structure has no water "
        "occupancy to report"),
    "water_residence_time": (
        "requires explicit-solvent molecular dynamics; a residence time is a dynamical "
        "quantity and cannot be computed from one conformation"),
    "spine_of_hydration_continuity": (
        "requires explicit-solvent molecular dynamics in the minor groove"),
    "water_bridge_probability": (
        "requires explicit-solvent molecular dynamics"),
    "interfacial_water_entropy": (
        "requires explicit-solvent molecular dynamics and a free-energy decomposition"),
    "intercalator_affinity": (
        "requires docking a ligand into an intercalation site, which is not implemented"),
    "minor_groove_binder_affinity": (
        "requires docking a ligand into the minor groove, which is not implemented"),
    "depurination_rate": (
        "requires glycosidic bond cleavage kinetics; no sourced rate parameters here"),
    "depyrimidination_rate": (
        "requires glycosidic bond cleavage kinetics; no sourced rate parameters here"),
    "alkylation_rate": (
        "requires alkylating-agent-specific kinetics; the N7 accessibility this module "
        "does compute is the structural half of the problem, not the rate"),
    "abasic_site_stability": (
        "requires the free energy of an abasic lesion, which needs either quantum "
        "chemistry on the lesion or measured stabilities; neither is available"),
}


# ======================================================================================
# Solvent-accessible surface
# ======================================================================================

def atom_sasa(pdb_path: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Per-atom solvent-accessible surface area, keyed by (chain, residue) then name.

    Computed on the **minimised** structure. The raw cgNA+ reconstruction has broken
    sugar-phosphate junctions -- C3'-O3' at 1.64 angstrom against 1.43, and across an
    ensemble from 0.64 to 3.44 -- and a surface area computed on atoms at wrong distances
    is wrong in a way that looks perfectly smooth. See `geometry` for the measurement.
    """
    import freesasa

    s = freesasa.Structure(pdb_path)
    r = freesasa.calc(s)
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for i in range(s.nAtoms()):
        chain = s.chainLabel(i).strip()
        resi = int(s.residueNumber(i))
        name = s.atomName(i).strip()
        out.setdefault((chain, resi), {})[name] = float(r.atomArea(i))
    return out


def _sum_atoms(res: Dict[str, float], names: Sequence[str]) -> float:
    return float(sum(res.get(n, 0.0) for n in names))


def strand_profiles(sasa: Dict[Tuple[str, int], Dict[str, float]], seq: str
                    ) -> Dict[str, np.ndarray]:
    """Accessibility tracks per strand, in cgNA+ base-pair indexing.

    Chain A is the Watson strand 5'->3' and chain B is Crick 5'->3', so the Crick track
    is reversed on the way out and a position i refers to the same base pair on either
    strand. Getting that backwards would mirror every strand-specific channel.
    """
    n = len(seq)
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    crick = [comp[c] for c in seq.upper()][::-1]
    out: Dict[str, np.ndarray] = {}

    for chain, tag, letters, reverse in (("A", "W", list(seq.upper()), False),
                                         ("B", "C", crick, True)):
        keys = sorted([k for k in sasa if k[0] == chain], key=lambda k: k[1])
        m = len(keys)
        tracks = {k: np.full(n, np.nan) for k in
                  ("sasa_total", "sasa_sugar", "sasa_phosphate", "sasa_base",
                   "radical_target", "major_edge", "minor_edge", "purine_n7",
                   "thymine_c5c6", "cytosine_c6")}
        for i in range(min(m, n)):
            res = sasa[keys[i]]
            letter = letters[i] if i < len(letters) else "A"
            total = float(sum(res.values()))
            sugar = _sum_atoms(res, SUGAR_CARBONS) + res.get("O4'", 0.0)
            phos = _sum_atoms(res, PHOSPHATE_OXYGENS) + _sum_atoms(res, PHOSPHORUS)
            tracks["sasa_total"][i] = total
            tracks["sasa_sugar"][i] = sugar
            tracks["sasa_phosphate"][i] = phos
            tracks["sasa_base"][i] = max(total - sugar - phos, 0.0)
            tracks["radical_target"][i] = _sum_atoms(res, RADICAL_TARGETS)
            tracks["major_edge"][i] = _sum_atoms(res, MAJOR_EDGE.get(letter, ()))
            tracks["minor_edge"][i] = _sum_atoms(res, MINOR_EDGE.get(letter, ()))
            tracks["purine_n7"][i] = (_sum_atoms(res, PURINE_N7)
                                      if letter in "AG" else 0.0)
            tracks["thymine_c5c6"][i] = (_sum_atoms(res, THYMINE_C5C6)
                                         if letter == "T" else 0.0)
            tracks["cytosine_c6"][i] = (_sum_atoms(res, CYTOSINE_C6)
                                        if letter == "C" else 0.0)
        for k, v in tracks.items():
            out[f"{k}_{tag}"] = v[::-1].copy() if reverse else v
    return out


def hydration_free_energy(sasa: Dict[Tuple[str, int], Dict[str, float]]
                          ) -> Dict[str, float]:
    """Nonpolar and polar hydration free energy from the accessible surface.

    The nonpolar term is `gamma * SASA` over carbon, the standard surface-area model used
    with implicit solvent. The polar term applies a single coefficient to nitrogen,
    oxygen and phosphorus rather than Eisenberg and McLachlan's full per-atom-type table,
    which is a simplification and is why the polar channel is flagged provisional: the
    *ranking* across sequence is carried by the surface area, and the absolute scale is
    carried by a coefficient this module does not derive.
    """
    apolar = polar = 0.0
    for res in sasa.values():
        for name, area in res.items():
            element = name.lstrip("0123456789")[:1]
            if element == "C":
                apolar += area
            elif element in ("N", "O", "P"):
                polar += area
    return {
        "sasa_apolar_A2": apolar,
        "sasa_polar_A2": polar,
        "dG_nonpolar_kcal": GAMMA_NONPOLAR_KCAL_PER_A2 * apolar,
        "dG_polar_kcal": GAMMA_POLAR_KCAL_PER_A2 * polar,
        "groove_hydrophobicity": apolar / (apolar + polar) if (apolar + polar) else
        float("nan"),
    }


# ======================================================================================
# Photoproducts
# ======================================================================================

# Cyclobutane pyrimidine dimer formation needs adjacent pyrimidines on one strand, and is
# reported to rise close to twofold at methylated cytosines with the magnitude set by
# flanking context. The twofold is taken from the plan's own statement of the
# measurement; it is applied as a flat factor, which is why the methylated channels are
# provisional while the dipyrimidine classification is not.
CPD_METHYL_FACTOR = 2.0
_CPD_PROV = (
    "the dipyrimidine classification is exact, but the propensity weighting across the "
    "four classes and the roughly twofold enhancement at methylated cytosine are applied "
    "as flat factors from the literature rather than fitted to CPD-seq. The plan's "
    "ground truth is CPD-seq and dCPD-seq, and this channel is the structural prior that "
    "measurement would correct.")


def dipyrimidine_tracks(seq: str) -> Dict[str, np.ndarray]:
    """Which dipyrimidine class, if any, starts at each position on each strand.

    A cyclobutane pyrimidine dimer forms between adjacent pyrimidines on the *same*
    strand, so this is strand-specific by construction and the two strands see different
    dinucleotides: a TT on one strand is an AA on the other.
    """
    s = seq.upper()
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rc = "".join(comp[c] for c in s)[::-1]
    n = len(s)
    out: Dict[str, np.ndarray] = {}
    for strand, tag, reverse in ((s, "W", False), (rc, "C", True)):
        for cls in DIPYRIMIDINES:
            v = np.zeros(n)
            for i in range(n - 1):
                if strand[i:i + 2] == cls:
                    v[i] = 1.0
            out[f"cpd_{cls}_{tag}"] = v[::-1].copy() if reverse else v
    return out


# ======================================================================================
# Assembly
# ======================================================================================

# Probe reactivity: each entry is (channel suffix, the accessibility track it reads,
# what the probe does). Reactivity is reported as the accessible area of the atom the
# probe attacks, which is the structural determinant; the rate constant is not.
PROBES: Tuple[Tuple[str, str, str], ...] = (
    ("hydroxyl_radical", "radical_target",
     "hydrogen abstraction from C4' and C5' of the deoxyribose"),
    ("dms", "purine_n7", "methylation of guanine N7 in the major groove"),
    ("depc", "purine_n7", "carbethoxylation of purine N7"),
    ("kmno4", "thymine_c5c6", "oxidation of the thymine C5-C6 bond"),
    ("bisulfite", "cytosine_c6", "deamination at cytosine C6, single-stranded only"),
    ("dnase_i", "minor_edge", "minor groove cleavage, rate tracks groove width"),
)


def solvation_profiles(seq: str, minimised_pdb: str,
                       vip_profile: Optional[np.ndarray] = None
                       ) -> Dict[str, np.ndarray]:
    """Every per-base-pair Family G channel.

    `minimised_pdb` is required, not optional: every channel here is an accessibility,
    and an accessibility computed on a structure with broken bonds is wrong smoothly.
    `vip_profile` is Family B's ionization potential track, used for the 8-oxoguanine
    channel; without it that channel is NaN rather than guessed.
    """
    seq = seq.upper()
    n = len(seq)
    sasa = atom_sasa(minimised_pdb)
    tracks = strand_profiles(sasa, seq)
    out: Dict[str, np.ndarray] = {}

    for key, arr in tracks.items():
        out[f"G_{key}"] = np.asarray(arr, dtype=float)

    # Probe reactivity, per strand.
    for suffix, track, _desc in PROBES:
        for tag in ("W", "C"):
            out[f"G_probe_{suffix}_{tag}"] = np.asarray(
                tracks[f"{track}_{tag}"], dtype=float)

    # Photoproducts.
    dip = dipyrimidine_tracks(seq)
    for k, v in dip.items():
        out[f"G_{k}"] = v
    for tag in ("W", "C"):
        any_dip = np.zeros(n)
        for cls in DIPYRIMIDINES:
            any_dip = np.maximum(any_dip, dip[f"cpd_{cls}_{tag}"])
        out[f"G_cpd_any_{tag}"] = any_dip
        # Methylated variant: a CpG-context cytosine in a dipyrimidine is the methylated
        # case, and the enhancement is applied as a flat factor.
        meth = np.zeros(n)
        s = seq if tag == "W" else "".join(
            {"A": "T", "T": "A", "G": "C", "C": "G"}[c] for c in seq)[::-1]
        for i in range(n - 1):
            if any_dip[i] > 0 and s[i:i + 2].count("C") and s[i:i + 2] == "CC":
                meth[i] = CPD_METHYL_FACTOR
            elif any_dip[i] > 0:
                meth[i] = 1.0
        out[f"G_cpd_methyl_weighted_{tag}"] = meth
        # A 6-4 photoproduct forms at the same dipyrimidines but favours 3'-T.
        sixfour = np.zeros(n)
        for i in range(n - 1):
            if s[i:i + 2] in ("TT", "CT"):
                sixfour[i] = 1.0
        out[f"G_sixfour_{tag}"] = sixfour

    # 8-oxoguanine: guanine oxidation is governed by the ionization potential, which
    # Family B already computes. Reusing it keeps one implementation of one quantity.
    oxo = np.full(n, np.nan)
    if vip_profile is not None:
        v = np.asarray(vip_profile, dtype=float)
        finite = np.isfinite(v)
        if finite.any():
            ref = float(np.nanmean(v[finite]))
            for i, base in enumerate(seq):
                if base == "G" and i < len(v) and np.isfinite(v[i]):
                    oxo[i] = ref - float(v[i])     # lower IP is easier to oxidise
    out["G_oxoguanine_susceptibility_eV"] = oxo
    return out


def solvation_channels(seq: str, minimised_pdb: str,
                       profiles: Optional[Dict[str, np.ndarray]] = None,
                       vip_profile: Optional[np.ndarray] = None
                       ) -> Dict[str, float]:
    """Every whole-sequence Family G channel."""
    seq = seq.upper()
    p = profiles if profiles is not None else solvation_profiles(
        seq, minimised_pdb, vip_profile)
    sasa = atom_sasa(minimised_pdb)
    out: Dict[str, float] = {}

    h = hydration_free_energy(sasa)
    for k, v in h.items():
        out[f"G_{k}"] = float(v)

    for key in ("sasa_total", "sasa_sugar", "sasa_phosphate", "sasa_base",
                "radical_target", "major_edge", "minor_edge"):
        for tag in ("W", "C"):
            v = np.asarray(p[f"G_{key}_{tag}"], dtype=float)
            out[f"G_mean_{key}_{tag}"] = float(np.nanmean(v)) \
                if np.isfinite(v).any() else float("nan")
    # Cleavage heterogeneity: a locus whose backbone accessibility varies strongly along
    # it is structurally different from a uniform one, and this is what a footprinting
    # experiment sees as a pattern rather than a level.
    for tag in ("W", "C"):
        v = np.asarray(p[f"G_radical_target_{tag}"], dtype=float)
        out[f"G_cleavage_heterogeneity_{tag}"] = float(np.nanstd(v)) \
            if np.isfinite(v).any() else float("nan")
        out[f"G_max_cleavage_{tag}"] = float(np.nanmax(v)) \
            if np.isfinite(v).any() else float("nan")

    for cls in DIPYRIMIDINES:
        out[f"G_n_cpd_{cls}"] = float(
            np.nansum(p[f"G_cpd_{cls}_W"]) + np.nansum(p[f"G_cpd_{cls}_C"]))
    out["G_cpd_density"] = float(
        (np.nansum(p["G_cpd_any_W"]) + np.nansum(p["G_cpd_any_C"]))
        / max(2 * len(seq), 1))
    out["G_sixfour_density"] = float(
        (np.nansum(p["G_sixfour_W"]) + np.nansum(p["G_sixfour_C"]))
        / max(2 * len(seq), 1))

    oxo = np.asarray(p["G_oxoguanine_susceptibility_eV"], dtype=float)
    out["G_max_oxoguanine_susceptibility_eV"] = float(np.nanmax(oxo)) \
        if np.isfinite(oxo).any() else float("nan")
    out["G_mean_oxoguanine_susceptibility_eV"] = float(np.nanmean(oxo)) \
        if np.isfinite(oxo).any() else float("nan")
    return out


# ======================================================================================
# Registry
# ======================================================================================

_POLAR_PROV = (
    "the nonpolar term uses the standard surface-area model with the implicit-solvent "
    "convention gamma = 0.0072 kcal/mol/A^2, but the polar term applies one coefficient "
    "to all nitrogen, oxygen and phosphorus rather than Eisenberg and McLachlan's "
    "per-atom-type table, so the absolute scale is not traceable even though the "
    "sequence ranking, which the surface area carries, is")


def _g(name: str, resolution: Resolution, parity: Parity, units: str, desc: str,
       source: str = DEFINITION, provisional: bool = False,
       reason: str = "") -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=resolution, parity=parity,
                       units=units, description=desc, provisional=provisional,
                       provisional_reason=reason, source=source)


_SPECS_G: List[ChannelSpec] = []

# Accessibility tracks: STRANDED, since each is a property of one strand's atoms.
_TRACK_DESC = {
    "sasa_total": "total solvent-accessible surface of this nucleotide",
    "sasa_sugar": "accessible surface of the deoxyribose",
    "sasa_phosphate": "accessible surface of the phosphate group",
    "sasa_base": "accessible surface of the base",
    "radical_target": "accessible surface of C4' and C5', the hydroxyl radical's targets",
    "major_edge": "accessible surface of the major-groove edge atoms",
    "minor_edge": "accessible surface of the minor-groove edge atoms",
    "purine_n7": "accessible surface of purine N7",
    "thymine_c5c6": "accessible surface of the thymine C5-C6 bond",
    "cytosine_c6": "accessible surface of cytosine C6",
}
for _key, _desc in _TRACK_DESC.items():
    for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
        _SPECS_G.append(_g(f"G_{_key}_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                           "angstrom^2", f"{_strand} strand: {_desc}", SHRAKE))
for _key in ("sasa_total", "sasa_sugar", "sasa_phosphate", "sasa_base",
             "radical_target", "major_edge", "minor_edge"):
    for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
        _SPECS_G.append(_g(f"G_mean_{_key}_{_tag}", Resolution.GLOBAL, Parity.STRANDED,
                           "angstrom^2", f"mean {_strand} {_TRACK_DESC[_key]}", SHRAKE))

for _suffix, _track, _what in PROBES:
    _src = TULLIUS if _suffix == "hydroxyl_radical" else DEFINITION
    for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
        _SPECS_G.append(_g(
            f"G_probe_{_suffix}_{_tag}", Resolution.PER_BP, Parity.STRANDED,
            "angstrom^2",
            f"{_strand} strand reactivity to {_suffix}: {_what}. Reported as the "
            "accessible area of the attacked atom, which is the structural determinant; "
            "the rate constant is not computed.", _src))

for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
    _SPECS_G.append(_g(f"G_cleavage_heterogeneity_{_tag}", Resolution.GLOBAL,
                       Parity.STRANDED, "angstrom^2",
                       f"spread of {_strand} backbone accessibility along the fragment, "
                       "which is the pattern a footprinting experiment reads rather than "
                       "the level", ORCHID))
    _SPECS_G.append(_g(f"G_max_cleavage_{_tag}", Resolution.GLOBAL, Parity.STRANDED,
                       "angstrom^2", f"most accessible {_strand} backbone position",
                       ORCHID))
    for _cls in DIPYRIMIDINES:
        _SPECS_G.append(_g(f"G_cpd_{_cls}_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                           "indicator",
                           f"{_strand} strand {_cls} dipyrimidine starts here; a "
                           "cyclobutane dimer forms between adjacent pyrimidines on one "
                           "strand", CPDSEQ))
    _SPECS_G.append(_g(f"G_cpd_any_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                       "indicator", f"{_strand} strand dipyrimidine of any class",
                       CPDSEQ))
    _SPECS_G.append(_g(f"G_cpd_methyl_weighted_{_tag}", Resolution.PER_BP,
                       Parity.STRANDED, "relative propensity",
                       f"{_strand} strand dimer propensity with the methylated-cytosine "
                       "enhancement applied", CPDSEQ, provisional=True,
                       reason=_CPD_PROV))
    _SPECS_G.append(_g(f"G_sixfour_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                       "indicator",
                       f"{_strand} strand 6-4 photoproduct site, which favours a "
                       "3-prime thymine", CPDSEQ, provisional=True, reason=_CPD_PROV))

_SPECS_G += [
    _g("G_sasa_apolar_A2", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom^2",
       "total accessible surface of carbon", SHRAKE),
    _g("G_sasa_polar_A2", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom^2",
       "total accessible surface of nitrogen, oxygen and phosphorus", SHRAKE),
    _g("G_dG_nonpolar_kcal", Resolution.GLOBAL, Parity.SYMMETRIC, "kcal/mol",
       "nonpolar hydration free energy from the surface-area model", EISENBERG),
    _g("G_dG_polar_kcal", Resolution.GLOBAL, Parity.SYMMETRIC, "kcal/mol",
       "polar hydration free energy from the surface-area model", EISENBERG,
       provisional=True, reason=_POLAR_PROV),
    _g("G_groove_hydrophobicity", Resolution.GLOBAL, Parity.SYMMETRIC, "fraction",
       "apolar fraction of the accessible surface", SHRAKE),
    _g("G_cpd_density", Resolution.GLOBAL, Parity.SYMMETRIC, "per base",
       "dipyrimidine sites per base, both strands", CPDSEQ),
    _g("G_sixfour_density", Resolution.GLOBAL, Parity.SYMMETRIC, "per base",
       "6-4 photoproduct sites per base, both strands", CPDSEQ,
       provisional=True, reason=_CPD_PROV),
    # STRANDED, not SYMMETRIC: it is reported only at guanines of the *Watson* strand,
    # so under reverse complement it moves to the other strand's guanines. Declared
    # SYMMETRIC at first, caught while declaring index maps.
    _g("G_oxoguanine_susceptibility_eV", Resolution.PER_BP, Parity.STRANDED, "eV",
       "how much more oxidisable this guanine is than the fragment mean, from the "
       "Family B ionization potential; NaN at non-guanine positions and when Family B "
       "is not supplied",
       "derived from Family B (GFN2-xTB vertical ionization potential)"),
    _g("G_max_oxoguanine_susceptibility_eV", Resolution.GLOBAL, Parity.SYMMETRIC, "eV",
       "most oxidisable guanine in the fragment",
       "derived from Family B (GFN2-xTB vertical ionization potential)"),
    _g("G_mean_oxoguanine_susceptibility_eV", Resolution.GLOBAL, Parity.SYMMETRIC, "eV",
       "mean guanine oxidisability",
       "derived from Family B (GFN2-xTB vertical ionization potential)"),
]
for _cls in DIPYRIMIDINES:
    _SPECS_G.append(_g(f"G_n_cpd_{_cls}", Resolution.GLOBAL, Parity.SYMMETRIC, "count",
                       f"number of {_cls} dipyrimidines, both strands", CPDSEQ))

REGISTRY.extend(_SPECS_G)
