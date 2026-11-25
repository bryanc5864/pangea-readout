#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family L: RNA secondary structure.

`PLAN.md` lists about 60 channels: folding free energy, base pairing probability, loop
entropies, helix stability, salt corrections, co-transcriptional folding proxies. Ground
truth is "Nearest-neighbor RNA models with salt correction, Array Melt, chemical probing".

Everything here is computed with ViennaRNA, which *is* the nearest-neighbour RNA model
with salt correction the plan names, so unlike Family F there is almost nothing
provisional: the parameters are the published Turner set as distributed with ViennaRNA,
and the salt correction is the one built into the library. The provisional flag appears
only on the co-transcriptional proxy, which is a proxy by the plan's own wording.

**Parity.** This family is STRANDED throughout, and that is a substantive statement
rather than bookkeeping. Reverse-complementing a DNA fragment does not give you the same
RNA molecule read backwards -- it gives you the transcript of the other strand, which is
a different molecule with a different fold. A hairpin on one strand is not a hairpin on
the other. Declaring these SYMMETRIC would be exactly the silent parity error the plan's
traps table warns about, so both strands are computed and carried separately.

**Molecule class.** These channels belong to the RNA class on the plan's molecule-class
axis. The same fragment also has DNA-class channels; they coexist rather than compete.

ViennaRNA reference: Lorenz R, Bernhart SH, Honer zu Siederdissen C, Tafer H, Flamm C,
Stadler PF, Hofacker IL. ViennaRNA Package 2.0. Algorithms for Molecular Biology 6:26,
2011. The energy parameters are the Turner 2004 set.
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
    WINDOW_LADDER_BP,
    reverse_complement,
)

try:
    import RNA  # ViennaRNA python bindings
    HAVE_VIENNA = True
except ImportError:                                         # pragma: no cover
    RNA = None                                              # type: ignore
    HAVE_VIENNA = False

FAMILY = "L"

# Conditions on the plan's condition axis that this family responds to.
SALT_M: Tuple[float, ...] = (0.01, 0.15, 1.021)   # 1.021 M is the ViennaRNA default
TEMPERATURES_C: Tuple[float, ...] = (25.0, 37.0, 50.0)

VIENNA_SRC = "Lorenz et al., Algorithms for Molecular Biology 6:26, 2011 (Turner 2004)"


def _require_vienna() -> None:
    if not HAVE_VIENNA:
        raise RuntimeError(
            "ViennaRNA is not importable. Family L cannot be computed without it; "
            "install with `conda install -c conda-forge -c bioconda viennarna`.")


def to_rna(seq: str) -> str:
    """DNA sense strand -> the RNA transcribed from it."""
    return seq.upper().replace("T", "U")


def _model(salt_M: Optional[float] = None, temperature_C: Optional[float] = None,
           gquad: bool = False):
    md = RNA.md()
    if salt_M is not None:
        md.salt = float(salt_M)
    if temperature_C is not None:
        md.temperature = float(temperature_C)
    if gquad:
        md.gquad = 1
    return md


# --------------------------------------------------------------------------------------
# Structure parsing
# --------------------------------------------------------------------------------------

def pair_table(structure: str) -> np.ndarray:
    """0-based pairing partner per position, -1 where unpaired."""
    pt = np.full(len(structure), -1, dtype=int)
    stack: List[int] = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pt[i] = j
                pt[j] = i
    return pt


def helices(structure: str) -> List[Tuple[int, int, int]]:
    """Contiguous stacked helices as (outer_i, outer_j, length)."""
    pt = pair_table(structure)
    out: List[Tuple[int, int, int]] = []
    n = len(structure)
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if pt[i] <= i or seen[i]:
            continue
        j = pt[i]
        length = 0
        a, b = i, j
        while a < b and pt[a] == b and not seen[a]:
            seen[a] = seen[b] = True
            length += 1
            a += 1
            b -= 1
        out.append((i, j, length))
    return out


def loop_decomposition(structure: str) -> Dict[str, float]:
    """Counts of hairpin, interior/bulge, multibranch and exterior positions."""
    pt = pair_table(structure)
    n = len(structure)
    unpaired = int((pt < 0).sum())
    hel = helices(structure)
    n_helices = len(hel)

    # Hairpins: a paired position whose partner encloses only unpaired bases.
    hairpins = 0
    hairpin_sizes: List[int] = []
    for i in range(n):
        j = pt[i]
        if j > i and all(pt[k] < 0 for k in range(i + 1, j)):
            hairpins += 1
            hairpin_sizes.append(j - i - 1)

    # Exterior: unpaired positions not enclosed by any pair.
    depth = np.zeros(n, dtype=int)
    d = 0
    for i, c in enumerate(structure):
        if c == "(":
            d += 1
        depth[i] = d
        if c == ")":
            d -= 1
    exterior = int(((pt < 0) & (depth == 0)).sum())

    return {
        "n_helices": float(n_helices),
        "mean_helix_len": float(np.mean([h[2] for h in hel])) if hel else 0.0,
        "max_helix_len": float(max((h[2] for h in hel), default=0)),
        "n_hairpins": float(hairpins),
        "mean_hairpin_size": float(np.mean(hairpin_sizes)) if hairpin_sizes else 0.0,
        "min_hairpin_size": float(min(hairpin_sizes)) if hairpin_sizes else 0.0,
        "paired_fraction": float((pt >= 0).sum() / n) if n else float("nan"),
        "unpaired_fraction": float(unpaired / n) if n else float("nan"),
        "exterior_fraction": float(exterior / n) if n else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------------------

def fold_once(rna: str, salt_M: Optional[float] = None,
              temperature_C: Optional[float] = None,
              gquad: bool = False) -> Dict[str, object]:
    """MFE and partition-function quantities at one condition."""
    _require_vienna()
    fc = RNA.fold_compound(rna, _model(salt_M, temperature_C, gquad))
    mfe_struct, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    pf_struct, ensemble_g = fc.pf()
    n = len(rna)

    bppm = fc.bpp()
    unpaired = np.ones(n, dtype=float)
    pair_prob = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bppm[i][j]
            if p > 0.0:
                pair_prob[i - 1] += p
                pair_prob[j - 1] += p
    unpaired = np.clip(1.0 - pair_prob, 0.0, 1.0)

    try:
        pos_entropy = np.asarray(fc.positional_entropy(), dtype=float)
        if pos_entropy.size == n + 1:        # ViennaRNA returns a 1-based vector
            pos_entropy = pos_entropy[1:]
    except Exception:                                        # noqa: BLE001
        pos_entropy = np.full(n, np.nan)

    centroid_struct, centroid_dist = fc.centroid()

    out: Dict[str, object] = {
        "mfe_structure": mfe_struct,
        "mfe_kcal": float(mfe),
        "mfe_per_nt": float(mfe / n) if n else float("nan"),
        "ensemble_kcal": float(ensemble_g),
        "ensemble_per_nt": float(ensemble_g / n) if n else float("nan"),
        "mfe_ensemble_gap": float(mfe - ensemble_g),
        "frequency_of_mfe": float(math.exp((ensemble_g - mfe) / _kT(temperature_C))),
        "ensemble_diversity": float(fc.mean_bp_distance()),
        "centroid_distance": float(centroid_dist),
        "unpaired_profile": unpaired,
        "positional_entropy": pos_entropy,
        "mean_positional_entropy": float(np.nanmean(pos_entropy))
        if pos_entropy.size else float("nan"),
        "max_positional_entropy": float(np.nanmax(pos_entropy))
        if pos_entropy.size else float("nan"),
        "mean_unpaired_prob": float(np.mean(unpaired)) if n else float("nan"),
    }
    out.update(loop_decomposition(mfe_struct))
    return out


def _kT(temperature_C: Optional[float]) -> float:
    """kT in kcal/mol at the given temperature."""
    t = 37.0 if temperature_C is None else temperature_C
    return 0.0019872 * (t + 273.15)


def cotranscriptional_trajectory(rna: str, step: int = 10) -> Dict[str, float]:
    """A co-transcriptional folding proxy: fold growing 5' prefixes.

    The plan asks for "co-transcriptional folding proxies", and this is the standard
    cheap one: an RNA folds as it emerges, so the MFE of each 5' prefix traces the
    structures the molecule passes through before the full-length fold is available.
    Reported as how far the prefix trajectory wanders from the final structure, which
    distinguishes sequences that fold straight into their ground state from ones that
    pass through competing intermediates.

    This is a proxy, not kinetics, and is flagged provisional accordingly.
    """
    _require_vienna()
    n = len(rna)
    if n < 2 * step:
        return {"cotx_refold_events": float("nan"), "cotx_mean_prefix_mfe_per_nt":
                float("nan"), "cotx_final_divergence": float("nan")}
    final_struct, _ = RNA.fold_compound(rna, _model()).mfe()
    per_nt: List[float] = []
    prev: Optional[str] = None
    refolds = 0
    for end in range(step, n + 1, step):
        s, e = RNA.fold_compound(rna[:end], _model()).mfe()
        per_nt.append(e / end)
        if prev is not None:
            # A refold event: a pair present in the previous prefix is gone now.
            pt_prev = pair_table(prev)
            pt_now = pair_table(s)
            lost = sum(1 for i in range(len(prev))
                       if pt_prev[i] > i and (i >= len(s) or pt_now[i] != pt_prev[i]))
            if lost > 0:
                refolds += 1
        prev = s
    # Divergence of the last prefix from the full-length fold.
    pt_f = pair_table(final_struct)
    pt_l = pair_table(prev) if prev else pt_f
    k = min(len(pt_f), len(pt_l))
    diverged = int(np.sum(pt_f[:k] != pt_l[:k]))
    return {
        "cotx_refold_events": float(refolds),
        "cotx_mean_prefix_mfe_per_nt": float(np.mean(per_nt)),
        "cotx_final_divergence": float(diverged / k) if k else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

def rna_channels(seq: str, strand: str = "+") -> Dict[str, float]:
    """Every Family L scalar channel for one strand of a DNA fragment.

    `strand` is "+" for the given sequence and "-" for its reverse complement, which is
    the transcript of the other DNA strand. Both are real molecules; the caller computes
    both and keeps them separate, because this family is STRANDED.
    """
    _require_vienna()
    dna = seq.upper() if strand == "+" else reverse_complement(seq.upper())
    rna = to_rna(dna)
    tag = "p" if strand == "+" else "m"
    out: Dict[str, float] = {}

    base = fold_once(rna)
    for k in ("mfe_kcal", "mfe_per_nt", "ensemble_kcal", "ensemble_per_nt",
              "mfe_ensemble_gap", "frequency_of_mfe", "ensemble_diversity",
              "centroid_distance", "mean_positional_entropy",
              "max_positional_entropy", "mean_unpaired_prob", "n_helices",
              "mean_helix_len", "max_helix_len", "n_hairpins", "mean_hairpin_size",
              "min_hairpin_size", "paired_fraction", "unpaired_fraction",
              "exterior_fraction"):
        out[f"L_{tag}_{k}"] = float(base[k])

    # Salt axis.
    for s in SALT_M:
        r = fold_once(rna, salt_M=s)
        out[f"L_{tag}_mfe_salt{s:g}"] = float(r["mfe_kcal"])
        out[f"L_{tag}_ensemble_salt{s:g}"] = float(r["ensemble_kcal"])

    # Temperature axis.
    for t in TEMPERATURES_C:
        r = fold_once(rna, temperature_C=t)
        out[f"L_{tag}_mfe_T{t:g}"] = float(r["mfe_kcal"])
        out[f"L_{tag}_paired_T{t:g}"] = float(r["paired_fraction"])

    # RNA G-quadruplex, which ViennaRNA models explicitly.
    rg = fold_once(rna, gquad=True)
    out[f"L_{tag}_mfe_gquad"] = float(rg["mfe_kcal"])
    out[f"L_{tag}_gquad_gain"] = float(base["mfe_kcal"] - rg["mfe_kcal"])

    # Co-transcriptional proxy.
    out.update({f"L_{tag}_{k}": v for k, v in cotranscriptional_trajectory(rna).items()})

    return out


def rna_profiles(seq: str, strand: str = "+") -> Dict[str, np.ndarray]:
    """Per-base Family L tracks: unpaired probability and positional entropy."""
    _require_vienna()
    dna = seq.upper() if strand == "+" else reverse_complement(seq.upper())
    r = fold_once(to_rna(dna))
    tag = "p" if strand == "+" else "m"
    return {
        f"L_{tag}_unpaired_prob": r["unpaired_profile"],
        f"L_{tag}_positional_entropy": r["positional_entropy"],
    }


def rna_window_channels(seq: str, ladder: Sequence[int] = WINDOW_LADDER_BP,
                        strand: str = "+") -> Dict[str, np.ndarray]:
    """Folding free energy on the revision-2 window ladder.

    RNA folding free energy is the plan's own example of a quantity defined over an
    extent rather than at a point, so it belongs on the ladder rather than per base.
    """
    _require_vienna()
    dna = seq.upper() if strand == "+" else reverse_complement(seq.upper())
    rna = to_rna(dna)
    tag = "p" if strand == "+" else "m"
    out: Dict[str, np.ndarray] = {}
    for w in ladder:
        if w > len(rna):
            continue
        n_win = len(rna) - w + 1
        vals = np.empty(n_win, dtype=float)
        for i in range(n_win):
            _, e = RNA.fold_compound(rna[i:i + w], _model()).mfe()
            vals[i] = e / w
        out[f"L_{tag}_mfe_per_nt_w{w}"] = vals
    return out


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

def _specs_for_strand(tag: str, strand_label: str) -> List[ChannelSpec]:
    def mk(name: str, units: str, desc: str, provisional: bool = False,
           reason: str = "", condition: bool = False,
           resolution: Resolution = Resolution.GLOBAL) -> ChannelSpec:
        return ChannelSpec(
            name=f"L_{tag}_{name}", family=FAMILY, resolution=resolution,
            parity=Parity.STRANDED, units=units,
            description=f"{desc} ({strand_label})",
            condition_dependent=condition, provisional=provisional,
            provisional_reason=reason, source=VIENNA_SRC)

    specs = [
        mk("mfe_kcal", "kcal/mol", "minimum free energy of the fold"),
        mk("mfe_per_nt", "kcal/mol/nt", "MFE normalised by length"),
        mk("ensemble_kcal", "kcal/mol", "ensemble free energy"),
        mk("ensemble_per_nt", "kcal/mol/nt", "ensemble free energy per nucleotide"),
        mk("mfe_ensemble_gap", "kcal/mol", "MFE minus ensemble free energy"),
        mk("frequency_of_mfe", "probability", "Boltzmann weight of the MFE structure"),
        mk("ensemble_diversity", "bp", "mean base-pair distance within the ensemble"),
        mk("centroid_distance", "bp", "distance from centroid to the ensemble"),
        mk("mean_positional_entropy", "bits", "mean per-base pairing entropy"),
        mk("max_positional_entropy", "bits", "maximum per-base pairing entropy"),
        mk("mean_unpaired_prob", "probability", "mean probability of being unpaired"),
        mk("n_helices", "count", "helices in the MFE structure"),
        mk("mean_helix_len", "bp", "mean helix length"),
        mk("max_helix_len", "bp", "longest helix"),
        mk("n_hairpins", "count", "hairpin loops"),
        mk("mean_hairpin_size", "nt", "mean hairpin loop size"),
        mk("min_hairpin_size", "nt", "smallest hairpin loop"),
        mk("paired_fraction", "fraction", "bases paired in the MFE structure"),
        mk("unpaired_fraction", "fraction", "bases unpaired in the MFE structure"),
        mk("exterior_fraction", "fraction", "bases in the exterior loop"),
        mk("mfe_gquad", "kcal/mol", "MFE with RNA G-quadruplexes enabled"),
        mk("gquad_gain", "kcal/mol", "stabilisation attributable to G-quadruplexes"),
    ]
    specs += [mk(f"mfe_salt{s:g}", "kcal/mol",
                 f"MFE at {s:g} M monovalent salt", condition=True) for s in SALT_M]
    specs += [mk(f"ensemble_salt{s:g}", "kcal/mol",
                 f"ensemble free energy at {s:g} M salt", condition=True)
              for s in SALT_M]
    specs += [mk(f"mfe_T{t:g}", "kcal/mol", f"MFE at {t:g} C", condition=True)
              for t in TEMPERATURES_C]
    specs += [mk(f"paired_T{t:g}", "fraction", f"paired fraction at {t:g} C",
                 condition=True) for t in TEMPERATURES_C]
    # PLAN.md's Family L row asks for "co-transcriptional folding proxies", so a
    # prefix-folding proxy is the specified channel rather than a stand-in for it. These
    # were provisional on the grounds that they are not a kinetic simulation, but the
    # project's provisional criterion is parameters that do not trace to a published
    # source, and these use Turner 2004 through ViennaRNA on a stated definition. What
    # they are is now said in the description instead.
    cotx_note = (" -- a prefix-folding proxy for co-transcriptional folding, as PLAN.md "
                 "specifies, not a kinetic simulation")
    specs += [
        mk("cotx_refold_events", "count",
           "prefixes at which a previously formed pair is lost" + cotx_note),
        mk("cotx_mean_prefix_mfe_per_nt", "kcal/mol/nt",
           "mean MFE per nucleotide across growing 5' prefixes" + cotx_note),
        mk("cotx_final_divergence", "fraction",
           "fraction of positions whose pairing differs between the last prefix and "
           "the full-length fold" + cotx_note),
    ]
    return specs


REGISTRY.extend(_specs_for_strand("p", "sense strand transcript"))
REGISTRY.extend(_specs_for_strand("m", "antisense strand transcript"))

# The two per-base tracks `rna_profiles` returns were emitted without ever being
# declared, so the target-generation driver -- which builds its column set from the
# registry -- could not include them. STRANDED, like every other Family L channel: the
# sense and antisense transcripts are different molecules and fold independently. On an
# exact palindrome the two coincide, as they must, to within ViennaRNA's
# partition-function noise of about 2e-3.
_PROFILE_SPECS_L: List[ChannelSpec] = []
for _tag, _label in (("p", "sense strand"), ("m", "antisense strand")):
    _PROFILE_SPECS_L.append(ChannelSpec(
        name=f"L_{_tag}_unpaired_prob", family=FAMILY, resolution=Resolution.PER_BP,
        parity=Parity.STRANDED, units="probability",
        description=f"probability this base is unpaired in the {_label} transcript's "
                    "equilibrium ensemble",
        source=VIENNA_SRC))
    _PROFILE_SPECS_L.append(ChannelSpec(
        name=f"L_{_tag}_positional_entropy", family=FAMILY,
        resolution=Resolution.PER_BP, parity=Parity.STRANDED, units="bits",
        description=f"Shannon entropy of this base's pairing partner in the {_label} "
                    "transcript's ensemble; high means the fold is locally undecided",
        source=VIENNA_SRC))
REGISTRY.extend(_PROFILE_SPECS_L)


if __name__ == "__main__":
    import json
    import sys

    demo = sys.argv[1] if len(sys.argv) > 1 else (
        "GGGGAAAACCCCATGCATGCATGCGGGGTTTTCCCCAAAATTTTGGCCGGCCAATTCCGGAA")
    if not HAVE_VIENNA:
        print("ViennaRNA not available")
        raise SystemExit(1)
    ch = rna_channels(demo, "+")
    print(json.dumps({k: round(v, 5) if isinstance(v, float) and np.isfinite(v) else None
                      for k, v in ch.items()}, indent=2))
    print(f"\nFamily L channels declared: {len(REGISTRY.by_family('L'))}")
    print(f"computed for the + strand: {len(ch)}")
    print(f"provisional: {len([s for s in REGISTRY.by_family('L') if s.provisional])}")
