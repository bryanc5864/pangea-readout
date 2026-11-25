#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family K: RNA duplex and DNA-RNA hybrid.

PLAN.md: "K. RNA duplex and DNA-RNA hybrid -- A-form geometry and the full stiffness
matrix, groove dimensions which differ sharply from B-form, electrostatics solved on
A-form conformations, duplex thermodynamics, 2-prime hydroxyl hydration and dynamics.
Ground truth: cgNA+ for duplex RNA and hybrids, hexamer-scale RNA MD, APBS on A-form
geometry."

**Implemented here (stage 1):** every ground-state geometry and local-stiffness channel
that Family C computes, from the cgNA+ RNA (`Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends`)
and DNA-RNA hybrid (`cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends`) parameter sets, with groove
geometry measured at the A-form offsets; and duplex nearest-neighbour thermodynamics --
RNA duplexes from Turner 2004 through ViennaRNA, hybrids from Banerjee et al. 2020
(Nucleic Acids Res 48:12042, Table 2 as corrected in Nucleic Acids Res 49:10796, 2021;
100 mM NaCl). **Not yet:** electrostatics on A-form (Family A's pipeline), 2'-hydroxyl
hydration (needs MD). Family D's persistence and torsion estimators are deliberately not
reused: they assume the helical axis passes through the base-pair centres, which holds
for B-DNA and not for A-form (they give dsRNA 45 nm against dsDNA 50 nm, the reverse of
the cgNA+ authors' result that dsRNA is stiffer); a configuration-sampling estimator is
needed first.

**Three molecules per locus.**
- `K_rna_*`: the RNA duplex of x and its complement. It is reverse-complement symmetric
  like B-DNA and inherits Family C's parities and index maps.
- `K_hybp_*`: the hybrid whose DNA strand is x. cgNA+ reads a hybrid sequence 5'->3'
  along the DNA strand, Watson DNA and Crick RNA (Sharma, Patelli, de Bruin, Maddocks,
  bioRxiv 2026, 10.64898/2026.08.09.740023, Methods).
- `K_hybm_*`: the hybrid whose DNA strand is rc(x) -- RNA paired to the other strand --
  computed on rc(x) and stored in its own 5'->3' frame. So K_hybm(x) = K_hybp(rc x)
  exactly, position by position: a stranded pair handled by PANGEA's strand tie, like
  Family L's transcripts.

**End caps.** Both parameter sets were trained only with G-C end pairs (cgNA+ README).
Every molecule is computed with "GC" added at both ends and the cap positions removed.
"GC" is its own reverse complement, so capping commutes exactly with reverse
complementing and the RNA duplex keeps its symmetry. A step into a cap is not a property
of the fragment and is NaN.

**Groove offsets.** Family C measures groove widths at the phosphate index offsets that
minimise the cross-strand P-P distance, -4 and +4 in B-DNA. The same scan on cgNA+ RNA and
hybrid ground states puts the minima at -3 (minor) and +6 (major) (DECISIONS.md), so K
uses those, with the matching reverse-complement index map.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import geometry as G
from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "K"
SETS = {"rna": "Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends",
        "hyb": "cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends"}
MOLECULES: Tuple[Tuple[str, str, bool], ...] = (("rna", "rna", False),
                                               ("hybp", "hyb", False),
                                               ("hybm", "hyb", True))
CAP = "GC"
A_FORM_GROOVE_OFFSETS = (-3, 6)
CGNA_SRC = ("cgNA+ (Sharma, Patelli, de Bruin, Maddocks); parameter sets "
            "Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends and "
            "cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends")
TURNER_SRC = "Turner 2004 nearest-neighbour RNA parameters (ViennaRNA 2.8)"
HYB_SRC = ("Banerjee et al., Nucleic Acids Res 48:12042 (2020), Table 2, corrected in "
           "Nucleic Acids Res 49:10796 (2021); 100 mM NaCl")

# Banerjee et al. 2020 Table 2, corrected 2021: RNA 5'->3' dinucleotide (its DNA partner
# antiparallel) -> (dH kcal/mol, dS cal/mol/K, dG37 kcal/mol).
HYB_NN = {"AA": (-7.8, -22.9, -0.7), "AC": (-10.1, -27.7, -1.5), "AG": (-9.4, -26.1, -1.3),
          "AU": (-5.8, -17.4, -0.4), "CA": (-9.8, -27.7, -1.2), "CC": (-9.5, -25.1, -1.7),
          "CG": (-9.0, -24.5, -1.4), "CU": (-6.1, -18.4, -0.4), "GA": (-8.6, -22.9, -1.5),
          "GC": (-10.6, -27.7, -2.0), "GG": (-13.3, -35.5, -2.3), "GU": (-9.3, -25.5, -1.4),
          "UA": (-6.6, -19.7, -0.5), "UC": (-6.5, -16.4, -1.4), "UG": (-8.9, -23.5, -1.6),
          "UU": (-7.4, -24.5, 0.2)}
HYB_INIT_GC = (0.0, -6.4, 2.0)      # at least one terminal rG-dC / rC-dG pair
HYB_INIT_AT = (0.0, -8.4, 2.6)      # only rA-dT / rU-dA terminal pairs

# Excluded: ensemble spreads, condition deltas and relaxed-structure torsions (no A-form
# relaxed structure or ensemble yet), and the Hoogsteen-opening proxy -- a B-DNA concept
# (a purine flipping syn) that is also not reverse-complement consistent under the RNA set
# (0.39 of its spread, against exactly 0 for every other channel; DECISIONS.md).
_EXCLUDE = re.compile(r"(_sd$|_sd_|_meth_|_salt_|_temp_|BI_fraction|sugar_|_alpha|_beta|"
                      r"_gamma|_delta|_epsilon|_zeta|eps_minus_zeta|_chi|hbond|hoogsteen)")
_COMP = str.maketrans("ACGT", "TGCA")


def reverse_complement(s: str) -> str:
    return s.upper().translate(_COMP)[::-1]


def _rc_shift_for(name: str, rc_shift: int) -> int:
    """The A-form index maps of the groove channels (offset -3 -> +3, +6 -> -6)."""
    if "minor_width" in name or "minor_depth" in name:
        return -A_FORM_GROOVE_OFFSETS[0]
    if "major_width" in name or "major_depth" in name:
        return -A_FORM_GROOVE_OFFSETS[1]
    return rc_shift


def _source_c_specs(resolution: Resolution) -> List[ChannelSpec]:
    return [s for s in REGISTRY.by_family("C")
            if s.resolution is resolution and not _EXCLUDE.search(s.name)]


def _k_name(tag: str, c_name: str) -> str:
    return f"K_{tag}_" + c_name[2:]


def _declare() -> None:
    for tag, _, _ in MOLECULES:
        stranded = tag != "rna"
        for res in (Resolution.PER_BP, Resolution.GLOBAL):
            for c in _source_c_specs(res):
                REGISTRY.register(ChannelSpec(
                    name=_k_name(tag, c.name), family=FAMILY, resolution=res,
                    parity=Parity.STRANDED if stranded else c.parity, units=c.units,
                    description=f"{c.description} ({'RNA duplex' if tag == 'rna' else 'DNA-RNA hybrid, DNA strand ' + ('x' if tag == 'hybp' else 'rc(x)')})",
                    source=CGNA_SRC,
                    rc_shift=0 if stranded else (_rc_shift_for(c.name, c.rc_shift)
                                                 if res is Resolution.PER_BP else 0),
                    circular=c.circular))
        # nearest-neighbour thermodynamics: one step value per base-pair step, and the duplex
        src = TURNER_SRC if tag == "rna" else HYB_SRC
        REGISTRY.register(ChannelSpec(
            name=f"K_{tag}_nn_dG37_step_kcal", family=FAMILY, resolution=Resolution.PER_BP,
            parity=Parity.STRANDED if stranded else Parity.SYMMETRIC, units="kcal/mol",
            description="nearest-neighbour stacking free energy of the step at 37 C",
            source=src, rc_shift=0 if stranded else -1))
        REGISTRY.register(ChannelSpec(
            name=f"K_{tag}_duplex_dG37_kcal", family=FAMILY, resolution=Resolution.GLOBAL,
            parity=Parity.STRANDED if stranded else Parity.SYMMETRIC, units="kcal/mol",
            description="duplex formation free energy at 37 C, nearest-neighbour model",
            source=src))
    REGISTRY.register(ChannelSpec(
        name="K_rna_duplex_dG37_salt0.1_kcal", family=FAMILY, resolution=Resolution.GLOBAL,
        parity=Parity.SYMMETRIC, units="kcal/mol",
        description="RNA duplex free energy at 37 C and 0.1 M monovalent salt "
                    "(ViennaRNA salt correction)", source=TURNER_SRC, condition_dependent=True))
    for tag in ("hybp", "hybm"):
        for q, u in (("dH", "kcal/mol"), ("dS", "cal/mol/K")):
            REGISTRY.register(ChannelSpec(
                name=f"K_{tag}_duplex_{q}", family=FAMILY, resolution=Resolution.GLOBAL,
                parity=Parity.STRANDED, units=u,
                description=f"hybrid duplex {q} (nearest-neighbour sum plus initiation)",
                source=HYB_SRC))


def _molecule(seq: str, tag: str) -> str:
    s = seq.upper()
    return reverse_complement(s) if tag == "hybm" else s


def _cgna_part(seq: str, tag: str, set_key: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    s = _molecule(seq, tag)
    n = len(s)
    capped = CAP + s + CAP
    st = G.geometry_state(capped, n_ensemble=0, parameter_set=SETS[set_key],
                          groove_offsets=A_FORM_GROOVE_OFFSETS)
    prof = G.geometry_profiles(capped, state=st, n_ensemble=0, with_conditions=False)
    glob = G.geometry_channels(capped, state=st, n_ensemble=0, profiles=prof)
    c0 = len(CAP)
    out_p: Dict[str, np.ndarray] = {}
    for c in _source_c_specs(Resolution.PER_BP):
        v = np.asarray(prof.get(c.name, np.full(n + 2 * c0, np.nan)), dtype=float)[c0:c0 + n].copy()
        if c.rc_shift == -1:              # a step quantity: the step into the 3' cap is not ours
            v[n - 1] = np.nan
        out_p[_k_name(tag, c.name)] = v
    out_g = {_k_name(tag, c.name): float(glob.get(c.name, np.nan))
             for c in _source_c_specs(Resolution.GLOBAL)}
    return out_p, out_g


def _to_rna(s: str) -> str:
    return s.upper().replace("T", "U")


def rna_duplex_thermo(seq: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-step stacking free energy and duplex free energy of the RNA duplex of x."""
    import RNA
    x = _to_rna(seq); c = _to_rna(reverse_complement(seq)); n = len(x)
    out: Dict[str, float] = {}
    for key, salt in (("K_rna_duplex_dG37_kcal", None), ("K_rna_duplex_dG37_salt0.1_kcal", 0.1)):
        md = RNA.md()
        if salt is not None:
            md.salt = salt
        fc = RNA.fold_compound(x + "&" + c, md)
        out[key] = float(fc.eval_structure("(" * n + ")" * n))
    fc = RNA.fold_compound(x + "&" + c, RNA.md())
    step = np.full(n, np.nan)
    for i in range(1, n):                 # stack (i, 2n+1-i) on (i+1, 2n-i), 1-based
        e = fc.eval_int_loop(i, 2 * n + 1 - i, i + 1, 2 * n - i)
        step[i - 1] = e / 100.0           # dcal/mol -> kcal/mol
    return step, out


def hybrid_thermo(seq: str, tag: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Banerjee et al. nearest-neighbour hybrid thermodynamics.

    The DNA strand is the molecule's own sequence s (x for hybp, rc(x) for hybm); the RNA
    strand is complementary, read 5'->3' as rc(s). Steps are reported in the DNA strand's
    5'->3' frame: step i spans DNA positions i, i+1, whose RNA partners form the RNA
    dinucleotide rc(s)[n-2-i : n-i].
    """
    s = _molecule(seq, tag)
    n = len(s)
    rna = _to_rna(reverse_complement(s))
    step = np.full(n, np.nan)
    dH = dS = 0.0
    for i in range(n - 1):
        k = n - 2 - i                     # RNA dinucleotide 5'->3' covering this DNA step
        h, sv, g = HYB_NN[rna[k:k + 2]]
        step[i] = g; dH += h; dS += sv
    gc_end = any(b in "GC" for b in (s[0], s[-1]))
    h0, s0, _ = HYB_INIT_GC if gc_end else HYB_INIT_AT
    dH += h0; dS += s0
    dG = dH - 310.15 * dS / 1000.0
    return step, {f"K_{tag}_duplex_dG37_kcal": dG, f"K_{tag}_duplex_dH": dH,
                  f"K_{tag}_duplex_dS": dS}


def duplex_class_channels(seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Every Family K channel for one fragment: (per-bp profiles, whole-sequence)."""
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}
    for tag, set_key, _ in MOLECULES:
        p, g = _cgna_part(seq, tag, set_key)
        prof.update(p); glob.update(g)
    step, g = rna_duplex_thermo(seq)
    prof["K_rna_nn_dG37_step_kcal"] = step; glob.update(g)
    for tag in ("hybp", "hybm"):
        step, g = hybrid_thermo(seq, tag)
        prof[f"K_{tag}_nn_dG37_step_kcal"] = step; glob.update(g)
    return prof, glob


_declare()
