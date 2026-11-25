#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family J from oxDNA2 single-strand simulations (`run_family_j_oxdna.py`, both strands of
every fragment, 25 C, 0.15 M): conformation and elasticity (end-to-end distance, radius of
gyration, persistence length, extension at 5/10/20 pN), stacking (mean stacking energy and the
probability a step is unstacked), electrostatics without a complementary strand (oxDNA2
Debye-Hueckel energy of the backbone charges, and the local linear charge density = 1/backbone
spacing), self-structure (probability of a hydrogen-bonded partner), hydration (burial by base
contacts and the probability a base is fully exposed), and chemical reactivity as the exposure of
each probe's reactive atom, mechanistically: permanganate attacks thymine's C5=C6 face, which
stacking protects (unstacked probability at T); DMS (A N1, C N3), glyoxal/kethoxal (G N1/N2) and
CMCT (T N3, G N1) attack Watson-Crick-face atoms, which pairing protects (1 - paired probability at
those bases); other bases NaN. Stranded (J_m(x) = J_p(rc x)); requested explicitly, from the
production runs' .npz files (side shards).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "J"
OX_KCAL = 4.142e-20 * 6.02214076e23 / 4184.0          # oxDNA energy unit, kcal/mol
SRC = ("oxDNA2 MD of the free single strand (Snodin et al. 2015 sequence-dependent parameters), "
       "25 C, 0.15 M; oxDNA 3.7")
PER_NT = (("ox_stack_E", "kcal/mol", "stacking energy of the steps touching this base, mean"),
          ("ox_unstacked_p", "probability", "probability the steps touching this base are unstacked"),
          ("ox_dh_E", "kcal/mol", "Debye-Hueckel electrostatic energy of this nucleotide's backbone charge"),
          ("ox_charge_density", "e/nm", "local linear charge density (1 / backbone spacing)"),
          ("ox_paired_p", "probability", "probability of a hydrogen-bonded partner (self-structure)"),
          ("ox_contacts", "count", "base-site contacts within 0.85 nm, neighbours excluded (burial)"),
          ("ox_exposed_p", "probability", "probability the base is unstacked on both sides and unpaired"),
          ("ox_kmno4", "probability", "permanganate reactivity proxy: unstacked probability at T"),
          ("ox_dms", "probability", "DMS reactivity proxy: unpaired probability at A (N1) and C (N3)"),
          ("ox_glyoxal", "probability", "glyoxal/kethoxal reactivity proxy: unpaired probability at G"),
          ("ox_cmct", "probability", "CMCT reactivity proxy: unpaired probability at T (N3) and G (N1)"))
GLOBAL = (("ox_ree", "nm", "end-to-end distance, mean"), ("ox_rg", "nm", "radius of gyration, mean"),
          ("ox_lp", "nm", "persistence length from backbone tangent correlations"),
          ("ox_ext_f5", "nm/nt", "extension along a 5 pN force"), ("ox_ext_f10", "nm/nt", "extension at 10 pN"),
          ("ox_ext_f20", "nm/nt", "extension at 20 pN"),
          ("ox_mean_stack_E", "kcal/mol", "stacking energy per step, mean"),
          ("ox_unstacked_frac", "fraction", "unstacked fraction of steps"),
          ("ox_mean_dh_E", "kcal/mol", "Debye-Hueckel energy per nucleotide, mean"),
          ("ox_mean_charge_density", "e/nm", "linear charge density, strand mean"),
          ("ox_paired_frac", "fraction", "hydrogen-bonded fraction of nucleotides"),
          ("ox_exposed_frac", "fraction", "fully exposed fraction of bases"))


def _declare() -> None:
    for tag in ("p", "m"):
        which = "the given strand" if tag == "p" else "the reverse-complement strand"
        for n, u, d in PER_NT:
            REGISTRY.register(ChannelSpec(name=f"J_{tag}_{n}", family=FAMILY, resolution=Resolution.PER_BP,
                                          parity=Parity.STRANDED, units=u, source=SRC,
                                          description=f"oxDNA2 single strand, {which}: {d}"))
        for n, u, d in GLOBAL:
            REGISTRY.register(ChannelSpec(name=f"J_{tag}_{n}", family=FAMILY, resolution=Resolution.GLOBAL,
                                          parity=Parity.STRANDED, units=u, source=SRC,
                                          description=f"oxDNA2 single strand, {which}: {d}"))


_declare()


def channel_names() -> Tuple[List[str], List[str]]:
    pb = [f"J_{t}_{n}" for t in ("p", "m") for n, _, _ in PER_NT]
    gl = [f"J_{t}_{n}" for t in ("p", "m") for n, _, _ in GLOBAL]
    return pb, gl


def channels_from_npz(z, seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Per-strand arrays of `run_family_j_oxdna.py` v2 -> channels (each strand 5'->3')."""
    comp = str.maketrans("ACGT", "TGCA")
    prof, glob = {}, {}
    for tag, s in (("p", seq.upper()), ("m", seq.upper().translate(comp)[::-1])):
        b = np.array(list(s))
        paired = np.asarray(z[f"{tag}_hbonded_p_nt"], float)
        unst = np.asarray(z[f"{tag}_unstacked_p_nt"], float)
        prof[f"J_{tag}_ox_stack_E"] = np.asarray(z[f"{tag}_stack_E_nt"], float) * OX_KCAL
        prof[f"J_{tag}_ox_unstacked_p"] = unst
        prof[f"J_{tag}_ox_dh_E"] = np.asarray(z[f"{tag}_dh_E_nt"], float) * OX_KCAL
        prof[f"J_{tag}_ox_charge_density"] = 1.0 / np.asarray(z[f"{tag}_charge_spacing_nm_nt"], float)
        prof[f"J_{tag}_ox_paired_p"] = paired
        prof[f"J_{tag}_ox_contacts"] = np.asarray(z[f"{tag}_contacts_nt"], float)
        prof[f"J_{tag}_ox_exposed_p"] = np.asarray(z[f"{tag}_exposed_p_nt"], float)
        prof[f"J_{tag}_ox_kmno4"] = np.where(b == "T", unst, np.nan)
        prof[f"J_{tag}_ox_dms"] = np.where((b == "A") | (b == "C"), 1.0 - paired, np.nan)
        prof[f"J_{tag}_ox_glyoxal"] = np.where(b == "G", 1.0 - paired, np.nan)
        prof[f"J_{tag}_ox_cmct"] = np.where((b == "T") | (b == "G"), 1.0 - paired, np.nan)
        g = lambda k: float(z[f"{tag}_{k}"])                                     # noqa: E731
        glob.update({f"J_{tag}_ox_ree": g("ree_nm"), f"J_{tag}_ox_rg": g("rg_nm"), f"J_{tag}_ox_lp": g("lp_nm"),
                     f"J_{tag}_ox_ext_f5": g("ext_f5_nm_per_nt"), f"J_{tag}_ox_ext_f10": g("ext_f10_nm_per_nt"),
                     f"J_{tag}_ox_ext_f20": g("ext_f20_nm_per_nt"),
                     f"J_{tag}_ox_mean_stack_E": g("mean_stack_E") * OX_KCAL,
                     f"J_{tag}_ox_unstacked_frac": g("unstacked_frac"),
                     f"J_{tag}_ox_mean_dh_E": g("mean_dh_E") * OX_KCAL,
                     f"J_{tag}_ox_mean_charge_density": 1.0 / g("mean_charge_spacing_nm"),
                     f"J_{tag}_ox_paired_frac": g("hbonded_frac"), f"J_{tag}_ox_exposed_frac": g("exposed_frac")})
    return prof, glob
