#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family N: RNA modification response (stage 1).

PLAN.md: "N. RNA modification response -- Induced change in the main channels for
N6-methyladenosine, pseudouridine and 2-prime-O-methyl, applied across the RNA and hybrid
classes. Ground truth: cgNA+ with modified bases, DFT on modified stacks."

**Stage 1, implemented here:** the response of the RNA class's secondary structure
(Family L's model: ViennaRNA, Turner 2004, 37 C) to modifying one position at a time, for
both transcripts of the fragment, with ViennaRNA's modified-base nearest-neighbour
parameters, whose published sources are:

* m6A -- Kierzek E, Zhang X, Watson RM, Kennedy SD, Szabat M, Kierzek R, Mathews DH.
  Secondary structure prediction for RNA sequences including N6-methyladenosine. Nature
  Communications 13:1271 (2022). doi:10.1038/s41467-022-28817-4
* pseudouridine -- Hudson GA, Bloomingdale RJ, Znosko BM. Thermodynamic contribution and
  nearest-neighbor parameters of pseudouridine-adenosine base pairs in
  oligoribonucleotides. RNA 19:1474-1482 (2013). doi:10.1261/rna.039610.113
* inosine (A-to-I editing; an addition beyond the plan's list) -- Wright DJ, Rice JL,
  Yanker DM, Znosko BM. Biochemistry 46:4625-4634 (2007), doi:10.1021/bi0616910; and
  Wright DJ, Force CR, Znosko BM. Nucleic Acids Research 46:12099-12108 (2018),
  doi:10.1093/nar/gky907.

Per position i of a transcript (in its own 5'->3' frame), where the base can carry the
modification (A for m6A and inosine, U for pseudouridine; NaN elsewhere): the change in
ensemble free energy and in i's unpaired probability when i alone is modified. Per
transcript: the mean, most stabilising and most destabilising single-site change, the
destabilised fraction of sites, and the change when every site is modified (in vitro
transcription with a modified triphosphate makes exactly that molecule).

Inosine is modelled as ViennaRNA requires: the site holds the parameter set's fallback
base (G) and the inosine parameters override its pairing energies, against the
unmodified A. **2'-O-methyl** (added 2026-11-21): the duplex of each fully 2'-O-methylated
transcript with its RNA complement, from Kierzek et al. 2006's nearest-neighbour set (0.1 M NaCl),
against the RNA/RNA duplex at the same salt (`ome_duplex`). **Not yet:** the hybrid class (no
modified-base DNA/RNA hybrid parameters located), geometry (cgNA+ ships no modified RNA set).

Strands are STRANDED with N_m(x) = N_p(rc x), tied exactly like Family L.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution, reverse_complement

FAMILY = "N"
# (tag, method, base in the unmodified transcript, base the site must hold for ViennaRNA)
MODS: Tuple[Tuple[str, str, str, str], ...] = (
    ("m6A", "sc_mod_m6A", "A", "A"),
    ("psi", "sc_mod_pseudouridine", "U", "U"),
    ("ino", "sc_mod_inosine", "A", "G"),
)
DESTABILISED_KCAL = 0.1
SRC = {"m6A": "Kierzek et al., Nat Commun 13:1271, 2022 (ViennaRNA 2.8 modified-base set)",
       "psi": "Hudson, Bloomingdale & Znosko, RNA 19:1474, 2013 (ViennaRNA 2.8)",
       "ino": "Wright et al., Biochemistry 46:4625, 2007; Wright, Force & Znosko, NAR 46:12099, 2018 (ViennaRNA 2.8)"}
_GLOBALS = (("mean_ddG", "kcal/mol", "mean single-site change in ensemble free energy"),
            ("min_ddG", "kcal/mol", "most stabilising single-site change"),
            ("max_ddG", "kcal/mol", "most destabilising single-site change"),
            ("frac_destabilised", "fraction", "fraction of sites destabilised by > 0.1 kcal/mol"),
            ("all_sites_ddG", "kcal/mol", "change in ensemble free energy with every site modified"))


# 2'-O-methyl RNA / RNA heteroduplexes, INN-HB nearest-neighbour model, 0.1 M NaCl, pH 7:
# Kierzek E, Mathews DH, Ciesielska A, Turner DH, Kierzek R. Nucleic Acids Res 34:3609-3614
# (2006), doi:10.1093/nar/gkl232, Table 3. Stacks m(5'-XY)/r(3'-X'Y'): the 2'-O-methylated
# strand read 5'->3', paired with unmodified RNA. dG37 in kcal/mol.
OME_NN_DG37 = {"AA": -0.55, "AU": -0.84, "UU": -0.94, "UA": -1.20, "AC": -1.60, "AG": -1.81,
               "CA": -1.89, "UC": -1.90, "UG": -1.94, "GA": -2.06, "CU": -2.17, "GU": -2.17,
               "CG": -2.35, "CC": -2.78, "GG": -2.82, "GC": -3.02}
OME_INIT_DG37 = 3.32
OME_TERMINAL_AU_DG37 = 0.30
OME_SALT_M = 0.1
OME_SRC = ("Kierzek, Mathews, Ciesielska, Turner & Kierzek, NAR 34:3609, 2006, Table 3 (2'-O-methyl "
           "RNA/RNA, 0.1 M NaCl); reference RNA/RNA: Turner 2004 with ViennaRNA's salt correction to 0.1 M")


def ome_duplex(rna: str) -> Tuple[np.ndarray, float, float]:
    """2'-O-methyl response of the duplex formed by `rna` (fully 2'-O-methylated) with its RNA
    complement, against the RNA/RNA duplex, both at 0.1 M NaCl and 37 C.

    Returns (per-nucleotide ddG: mean of the steps touching each nucleotide, kcal/mol;
    duplex dG of the 2'-O-methyl/RNA duplex; duplex ddG against RNA/RNA)."""
    import RNA
    n = len(rna)
    comp = rna.translate(str.maketrans("ACGU", "UGCA"))[::-1]
    md = RNA.md(); md.salt = OME_SALT_M
    fc = RNA.fold_compound(rna + "&" + comp, md)
    g_rna = float(fc.eval_structure("(" * n + ")" * n))
    step = np.full(n - 1, np.nan)
    for i in range(1, n):                 # stack (i, 2n+1-i) on (i+1, 2n-i), 1-based
        e_rna = fc.eval_int_loop(i, 2 * n + 1 - i, i + 1, 2 * n - i) / 100.0
        step[i - 1] = OME_NN_DG37[rna[i - 1:i + 1]] - e_rna
    g_ome = OME_INIT_DG37 + sum(OME_NN_DG37[rna[i:i + 2]] for i in range(n - 1))
    g_ome += OME_TERMINAL_AU_DG37 * sum(1 for b in (rna[0], rna[-1]) if b in "AU")
    per_nt = np.full(n, np.nan)
    for i in range(n):
        touching = [step[j] for j in (i - 1, i) if 0 <= j < n - 1]
        per_nt[i] = float(np.mean(touching)) if touching else np.nan
    return per_nt, g_ome, g_ome - g_rna


def _declare() -> None:
    for tag in ("p", "m"):
        which = "transcript of the given strand" if tag == "p" else "transcript of the other strand"
        REGISTRY.register(ChannelSpec(
            name=f"N_{tag}_ome_ddG_step", family=FAMILY, resolution=Resolution.PER_BP,
            parity=Parity.STRANDED, units="kcal/mol", source=OME_SRC, condition_dependent=True,
            description=f"2'-O-methylation of the {which} paired with RNA: change in duplex "
                        f"stacking free energy, mean of the steps touching this nucleotide (0.1 M)"))
        REGISTRY.register(ChannelSpec(
            name=f"N_{tag}_ome_duplex_ddG", family=FAMILY, resolution=Resolution.GLOBAL,
            parity=Parity.STRANDED, units="kcal/mol", source=OME_SRC, condition_dependent=True,
            description=f"fully 2'-O-methylated {which} with its RNA complement: duplex free energy "
                        f"minus that of the RNA/RNA duplex (37 C, 0.1 M)"))
        REGISTRY.register(ChannelSpec(
            name=f"N_{tag}_ome_duplex_dG", family=FAMILY, resolution=Resolution.GLOBAL,
            parity=Parity.STRANDED, units="kcal/mol", source=OME_SRC, condition_dependent=True,
            description=f"fully 2'-O-methylated {which} with its RNA complement: duplex free energy "
                        f"(37 C, 0.1 M)"))
    for tag in ("p", "m"):
        which = "transcript of the given strand" if tag == "p" else "transcript of the other strand"
        for mod, _, base, _ in MODS:
            for name, units, desc in (("ddG_ens", "kcal/mol", "ensemble free-energy change"),
                                      ("dunpaired", "probability", "change in the site's unpaired probability")):
                REGISTRY.register(ChannelSpec(
                    name=f"N_{tag}_{mod}_{name}", family=FAMILY, resolution=Resolution.PER_BP,
                    parity=Parity.STRANDED, units=units, source=SRC[mod],
                    description=f"{mod} at this {base} of the {which}: {desc} (NaN where not {base})"))
            for name, units, desc in _GLOBALS:
                REGISTRY.register(ChannelSpec(
                    name=f"N_{tag}_{mod}_{name}", family=FAMILY, resolution=Resolution.GLOBAL,
                    parity=Parity.STRANDED, units=units, source=SRC[mod],
                    description=f"{mod}, {which}: {desc}"))


def _pf(rna: str, mod=None, sites=None):
    import RNA
    fc = RNA.fold_compound(rna, RNA.md())
    if mod is not None:
        getattr(fc, mod)(list(sites))
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    _, g = fc.pf()
    bpp = np.asarray(fc.bpp())[1:, 1:]
    paired = bpp.sum(0) + bpp.sum(1)
    return float(g), 1.0 - paired


def modification_channels(seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Every Family N channel for one fragment: (per-position profiles, whole-transcript)."""
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}
    for tag, dna in (("p", seq.upper()), ("m", reverse_complement(seq.upper()))):
        rna = dna.replace("T", "U")
        n = len(rna)
        g0, u0 = _pf(rna)
        for mod, method, base, hold in MODS:
            ddg = np.full(n, np.nan); du = np.full(n, np.nan)
            sites = [i for i, b in enumerate(rna) if b == base]
            for i in sites:
                s = rna if hold == base else rna[:i] + hold + rna[i + 1:]
                g, u = _pf(s, method, [i + 1])
                ddg[i], du[i] = g - g0, u[i] - u0[i]
            prof[f"N_{tag}_{mod}_ddG_ens"], prof[f"N_{tag}_{mod}_dunpaired"] = ddg, du
            v = ddg[sites] if sites else np.array([])
            glob[f"N_{tag}_{mod}_mean_ddG"] = float(v.mean()) if v.size else float("nan")
            glob[f"N_{tag}_{mod}_min_ddG"] = float(v.min()) if v.size else float("nan")
            glob[f"N_{tag}_{mod}_max_ddG"] = float(v.max()) if v.size else float("nan")
            glob[f"N_{tag}_{mod}_frac_destabilised"] = (float(np.mean(v > DESTABILISED_KCAL))
                                                        if v.size else float("nan"))
            if sites:
                s_all = rna if hold == base else "".join(hold if b == base else b for b in rna)
                g_all, _ = _pf(s_all, method, [i + 1 for i in sites])
                glob[f"N_{tag}_{mod}_all_sites_ddG"] = g_all - g0
            else:
                glob[f"N_{tag}_{mod}_all_sites_ddG"] = 0.0
        per_nt, g_ome, dd = ome_duplex(rna)
        prof[f"N_{tag}_ome_ddG_step"] = per_nt
        glob[f"N_{tag}_ome_duplex_dG"], glob[f"N_{tag}_ome_duplex_ddG"] = g_ome, dd
    return prof, glob


_declare()
