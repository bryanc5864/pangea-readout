#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family F: non-canonical structure propensities.

`PLAN.md` lists 53 channels here: quadruplex free energy under potassium, sodium and
lithium with topology class probabilities, loop and bulge descriptors and tetrad count;
i-motif at three pH values; Z-DNA transition free energy at three ionic strengths with
junction energy and boundary position; intramolecular triplex at three pH values;
cruciform extrusion at three superhelical densities; slipped-strand and repeat hairpin
propensity; A-phased bend descriptors; R-loop initiation and elongation free energy;
single-strandedness; quadruplex-duplex competition; triplet repeat instability; branch
migration propensity; transition cooperativity length.

These are string algorithms over the sequence, so they are cheap -- microseconds per
fragment against tens of seconds for a Poisson-Boltzmann solve -- which is why this
family is in the analytic pretraining set rather than the expensive one. The experimental
supervision (G4-seq, S1-END-seq, KAS-seq, permanganate and S1 footprinting) enters at the
delta-learning stage, not here.

**What is rigorously parameterised and what is not.** The project rule is that no number
is invented. Where a published algorithm exists it is implemented as published and cited.
Where the plan asks for an absolute free energy whose parameter set I could not source,
the channel is computed as a propensity and marked `provisional=True` in the registry, so
it cannot silently become a training target. `DECISIONS.md` records each one and what
would close it. The provisional channels are the quadruplex and i-motif absolute free
energies and the cation and pH axes attached to them; the structural descriptors,
G4Hunter, the mirror- and inverted-repeat detectors, the R-loop model and the A-phasing
analysis are not provisional.

Sources, per algorithm:

- **G4Hunter**: Bedrat A, Lacroix L, Mergny JL. Re-evaluation of G-quadruplex propensity
  with G4Hunter. Nucleic Acids Research 44:1746, 2016. A run of n consecutive G scores
  +min(n,4) per base, a run of n consecutive C scores -min(n,4) per base, A and T score
  0, and the G4Hunter score of a window is the mean over that window.
- **Z-DNA**: the dinucleotide B-to-Z transition free energies that I could source are the
  experimentally determined low-salt values, 0.66 kcal/mol for d(CG), 1.2 for d(CA)/d(TG)
  and 2.4 for d(TA), with a nucleation cost of about 5.0 kcal/mol per junction used by
  the SIBZ treatment (Zhabinskaya & Benham, PLOS Computational Biology 7:e1001051, 2011,
  which attributes the dinucleotide table to Ho, Ellison, Quigley & Rich, EMBO J
  5:2737, 1986). The full ten-dinucleotide anti-syn / syn-anti table in that paper is
  published as an image and I was not able to read it, so the non-alternating
  dinucleotides carry a documented high-energy default rather than their true values, and
  the absolute Z free energy is marked provisional. The *relative* ranking of alternating
  purine-pyrimidine tracts, which is what the channel is for, does not depend on that gap.
- **R-loop forming sequences**: the RIZ/REZ decomposition of Wongsurawat T et al.,
  Quantitative model of R-loop forming structures reveals a novel level of RNA-DNA hybrid
  interference. Nucleic Acids Research 40:e16, 2012 -- a G-cluster initiation zone
  followed by a G-rich elongation zone, scored with GC skew.
- **A-phased bending**: A-tracts repeating in phase with the helical period. The period is
  taken from the cgNA+ ground state at run time where available, and otherwise from the
  canonical 10.5 bp per turn.

Reverse-complement parity is declared for every channel and tested in `tests/`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
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

FAMILY = "F"

# --------------------------------------------------------------------------------------
# Published parameters
# --------------------------------------------------------------------------------------

G4HUNTER_RUN_CAP = 4           # Bedrat 2016: runs longer than 4 score as 4
G4HUNTER_WINDOW_BP = 25        # the window the method is calibrated on

# B-to-Z transition free energies, kcal/mol per dinucleotide, for all 16 dinucleotides in
# each conformation, from Ho, Ellison, Quigley & Rich, EMBO J 5:2737, 1986, as hard-coded
# in the Z-Hunt program (zhun3.c, Ho lab, github.com/Ho-Lab-Colostate/zhunt, array
# `dbzed[4][16]`), read from the source text. Rows:
#
#   0  anti-syn (AS), when the conformation alternates across the boundary with the
#      previous dinucleotide
#   1  syn-anti (SA), likewise
#   2  syn-anti where the boundary breaks alternation (syn-syn across it)
#   3  anti-syn where the boundary breaks alternation (anti-anti across it)
#
# Rows 2 and 3 are numerically identical in the program. The AS/SA reading was checked
# against known Z-DNA structure rather than taken from the array comments: 5'-CG-3' in
# Z-DNA has C anti and G syn, and row 0 gives CG 0.66; 5'-GC-3' has G syn and C anti, and
# row 1 gives GC 0.66; CA/TG, the other classic Z-former, gives 1.40 in its Z register.
#
# **This replaces a four-value table and a 5.0 kcal/mol placeholder, and corrects two of
# the four.** The earlier values CA/TG 1.20 and TA 2.40 came from a different
# parameterisation (Zhabinskaya & Benham 2011); Z-Hunt gives 1.40 and 2.50. A complete
# table from one source is used rather than a mixture.
Z_HUNT_DBZED: Dict[str, Tuple[float, float, float, float]] = {
    "AA": (4.40, 4.40, 6.20, 6.20), "AT": (6.20, 2.50, 6.20, 6.20),
    "AG": (3.40, 3.30, 5.20, 5.20), "AC": (5.20, 1.40, 5.20, 5.20),
    "TA": (2.50, 6.20, 6.20, 6.20), "TT": (4.40, 4.40, 6.20, 6.20),
    "TG": (1.40, 5.20, 5.20, 5.20), "TC": (3.30, 3.40, 5.20, 5.20),
    "GA": (3.30, 3.40, 5.20, 5.20), "GT": (5.20, 1.40, 5.20, 5.20),
    "GG": (2.40, 2.40, 4.00, 4.00), "GC": (4.20, 0.66, 4.00, 4.00),
    "CA": (1.40, 5.20, 5.20, 5.20), "CT": (3.40, 3.30, 5.20, 5.20),
    "CG": (0.66, 4.20, 4.00, 4.00), "CC": (2.40, 2.40, 4.00, 4.00),
}
Z_HUNT_SRC = ("Ho, Ellison, Quigley & Rich, EMBO J 5:2737, 1986, as implemented in "
              "Z-Hunt (zhun3.c, dbzed[4][16])")
# Window range for the best Z tract, in dinucleotides. A stated definition, not a sourced
# parameter: Z-Hunt itself takes the range as a command-line argument.
Z_MIN_DINUC, Z_MAX_DINUC = 6, 12
# B-Z junction cost. Z-Hunt's per-window energy has no junction term, so this is not
# used by the Z-Hunt channels. The value is now verified in open text: Zhabinskaya &
# Benham, PLoS Comput Biol 7:e1001051 (2011), PMC3024258, Methods: "a nucleation energy
# for each run of Z-DNA is required to form the junctions between the B-form and the
# Z-form. This has been measured to be approximately 5.0 kcal/mol/junction", citing the
# measurements (Peck & Wang 1983 among them, whose own text is only page images).
Z_NUCLEATION_KCAL = 5.0
Z_JUNCTION_SRC = ("Zhabinskaya & Benham, PLoS Comput Biol 7:e1001051 (2011), PMC3024258: "
                  "'approximately 5.0 kcal/mol/junction', measured")

CANONICAL_HELICAL_PERIOD_BP = 10.5

# Cytosine N3 pKa in the i-motif hemiprotonated pair. i-motifs form below this.
IMOTIF_CYTOSINE_PKA = 4.6      # free cytidine N3; the duplex context shifts it upward


# --------------------------------------------------------------------------------------
# G-quadruplex
# --------------------------------------------------------------------------------------

def _run_scores(seq: str, plus: str, minus: str, cap: int = G4HUNTER_RUN_CAP) -> np.ndarray:
    """Per-base G4Hunter scores: +min(run,cap) on `plus` runs, -min(run,cap) on `minus`."""
    s = seq.upper()
    out = np.zeros(len(s), dtype=float)
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        j = i
        while j < n and s[j] == c:
            j += 1
        run = j - i
        if c == plus:
            out[i:j] = min(run, cap)
        elif c == minus:
            out[i:j] = -float(min(run, cap))
        i = j
    return out


def g4hunter_profile(seq: str, window: int = G4HUNTER_WINDOW_BP) -> np.ndarray:
    """Per-base G4Hunter score, windowed as in Bedrat et al. 2016.

    Positive scores indicate G-quadruplex propensity on the given strand, negative on the
    complement. Edges use the largest centred window that fits, so the profile is defined
    at every position rather than truncated.
    """
    base = _run_scores(seq, "G", "C")
    n = len(base)
    if n == 0:
        return base
    half = window // 2
    cs = np.concatenate([[0.0], np.cumsum(base)])
    lo = np.maximum(0, np.arange(n) - half)
    hi = np.minimum(n, np.arange(n) + half + 1)
    return (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)


@dataclass(frozen=True)
class G4Candidate:
    start: int
    end: int                      # exclusive
    n_tracts: int
    tract_lengths: Tuple[int, ...]
    loop_lengths: Tuple[int, ...]
    n_tetrads: int
    has_bulge: bool
    strand: str                   # "+" if G-rich on the given strand, "-" if C-rich
    g4hunter: float


_G_TRACT = re.compile(r"G{2,}")
_C_TRACT = re.compile(r"C{2,}")


def find_g4_candidates(seq: str, max_loop: int = 12,
                       min_tract: int = 2) -> List[G4Candidate]:
    """Intramolecular quadruplex candidates: four or more G tracts with short loops.

    Both strands are scanned, because a C-rich tract on the given strand is a G-rich
    tract on its complement and forms a quadruplex there. Loops longer than `max_loop`
    terminate a candidate, which is the standard constraint; bulges are recorded rather
    than excluded, since bulged quadruplexes are real and the plan asks for bulge
    descriptors.
    """
    s = seq.upper()
    out: List[G4Candidate] = []
    for strand, pattern in (("+", _G_TRACT), ("-", _C_TRACT)):
        tracts = [(m.start(), m.end()) for m in pattern.finditer(s)
                  if m.end() - m.start() >= min_tract]
        i = 0
        while i < len(tracts):
            group = [tracts[i]]
            j = i + 1
            while j < len(tracts):
                gap = tracts[j][0] - group[-1][1]
                if gap > max_loop:
                    break
                group.append(tracts[j])
                j += 1
            if len(group) >= 4:
                start, end = group[0][0], group[-1][1]
                tl = tuple(b - a for a, b in group)
                ll = tuple(group[k + 1][0] - group[k][1] for k in range(len(group) - 1))
                # Tetrads are limited by the shortest participating tract.
                n_tet = min(tl)
                out.append(G4Candidate(
                    start=start, end=end, n_tracts=len(group),
                    tract_lengths=tl, loop_lengths=ll, n_tetrads=int(n_tet),
                    has_bulge=bool(len(set(tl)) > 1),
                    strand=strand,
                    g4hunter=float(np.mean(_run_scores(s[start:end], "G", "C")))
                    if strand == "+" else
                    float(-np.mean(_run_scores(s[start:end], "G", "C"))),
                ))
                i = j
            else:
                i += 1
    return out


def g4_topology_probabilities(c: G4Candidate) -> Dict[str, float]:
    """Soft topology assignment from loop lengths.

    The empirical rules this encodes: a single short central loop with two longer flanking
    loops favours chair/basket forms; all-short loops favour parallel propeller; a long
    central loop favours antiparallel. These are the qualitative determinants reported in
    the quadruplex structural literature. The probabilities are a smooth encoding of that
    ordering rather than a fitted classifier, which is why the topology channels are
    marked provisional.
    """
    loops = c.loop_lengths[:3] if len(c.loop_lengths) >= 3 else c.loop_lengths
    if not loops:
        return {"parallel": 1 / 3, "antiparallel": 1 / 3, "hybrid": 1 / 3}
    short = sum(1 for x in loops if x <= 3)
    long_ = sum(1 for x in loops if x >= 5)
    w_par = 1.0 + 1.5 * short
    w_anti = 1.0 + 1.5 * long_
    w_hyb = 1.0 + 0.75 * len([x for x in loops if 3 < x < 5])
    tot = w_par + w_anti + w_hyb
    return {"parallel": w_par / tot, "antiparallel": w_anti / tot, "hybrid": w_hyb / tot}


def g4_stability_score(c: G4Candidate) -> float:
    """A dimensionless stability ordering for a quadruplex candidate.

    More tetrads is more stable, and longer loops are less stable; both are among the
    best-established qualitative determinants of quadruplex stability. This is an
    ordering, not a free energy in kcal/mol, which is exactly why the free-energy channels
    built on it are provisional.
    """
    loop_penalty = sum(math.log1p(max(0, x - 1)) for x in c.loop_lengths)
    return float(c.n_tetrads) - 0.5 * loop_penalty - (0.5 if c.has_bulge else 0.0)


# Relative cation ordering for quadruplex stabilisation. Potassium is the strongest
# stabiliser, sodium weaker, lithium is the standard non-stabilising control. The
# ordering is not in doubt; the magnitudes here are a monotone encoding of it, not
# measured offsets, so every channel using them is provisional.
CATION_G4_WEIGHT = {"K": 1.0, "Na": 0.6, "Li": 0.05}


# --------------------------------------------------------------------------------------
# i-motif
# --------------------------------------------------------------------------------------

def imotif_profile(seq: str, window: int = G4HUNTER_WINDOW_BP) -> np.ndarray:
    """C-rich counterpart of the G4Hunter profile: the i-motif-forming strand."""
    return -g4hunter_profile(seq, window=window)


def imotif_ph_factor(ph: float) -> float:
    """Fraction of cytosines protonated at N3, by Henderson-Hasselbalch.

    The hemiprotonated C:C+ pair is what holds an i-motif together, so formation is
    strongly pH dependent and falls away above the cytosine N3 pKa.
    """
    return 1.0 / (1.0 + 10.0 ** (ph - IMOTIF_CYTOSINE_PKA))


# --------------------------------------------------------------------------------------
# Z-DNA
# --------------------------------------------------------------------------------------

def z_dna_profile(seq: str) -> np.ndarray:
    """Per-step Z propensity: the dinucleotide's energy in its better Z register.

    min(AS, SA) from the Z-Hunt table, kcal/mol, one value per step (length n-1). This is
    context-free and register-free by construction, which makes it the natural per-step
    track; the alternation constraint between neighbours enters in `z_hunt_window_energy`.
    Unknown letters give NaN rather than a guess.
    """
    s = seq.upper()
    out = np.full(max(0, len(s) - 1), np.nan)
    for i in range(len(out)):
        row = Z_HUNT_DBZED.get(s[i:i + 2])
        if row is not None:
            out[i] = min(row[0], row[1])
    return out


def z_hunt_window_energy(dinucs: Sequence[str]) -> float:
    """Minimum B-to-Z energy of consecutive dinucleotides, exactly as Z-Hunt computes it.

    Z-Hunt enumerates every assignment of AS or SA to each dinucleotide -- 2^n of them --
    and keeps the lowest sum, charging rows 0/1 where the conformation alternates across
    the boundary with the previous dinucleotide and rows 3/2 where it does not. That
    minimum is a shortest path through a two-state chain, so it is computed here by
    dynamic programming in O(n): same answer, without the exponential enumeration.
    """
    if not dinucs:
        return float("nan")
    rows = [Z_HUNT_DBZED.get(d) for d in dinucs]
    if any(r is None for r in rows):
        return float("nan")
    as_e, sa_e = rows[0][0], rows[0][1]          # first dinucleotide: rows 0 and 1
    for r in rows[1:]:
        # AS after AS: previous ends syn, current starts anti -> alternates -> row 0
        # AS after SA: previous ends anti, current starts anti -> breaks   -> row 3
        # SA after SA: previous ends anti, current starts syn  -> alternates -> row 1
        # SA after AS: previous ends syn,  current starts syn  -> breaks   -> row 2
        new_as = min(as_e + r[0], sa_e + r[3])
        new_sa = min(sa_e + r[1], as_e + r[2])
        as_e, sa_e = new_as, new_sa
    return float(min(as_e, sa_e))


def z_dna_best_tract(seq: str, min_dinuc: int = Z_MIN_DINUC,
                     max_dinuc: int = Z_MAX_DINUC) -> Dict[str, float]:
    """The most Z-favourable window, scored by Z-Hunt's own energy model.

    Windows of `min_dinuc` to `max_dinuc` consecutive non-overlapping dinucleotides, in
    both reading frames, scored by `z_hunt_window_energy` and ranked by energy per
    dinucleotide. There is no junction term, because Z-Hunt's per-window minimum has
    none; the fixed window range is what stops the trivially short window from winning.
    """
    s = seq.upper()
    cands = []                                    # (energy per dinucleotide, start, end)
    for frame in (0, 1):
        din = [s[i:i + 2] for i in range(frame, len(s) - 1, 2)]
        for i in range(len(din)):
            for k in range(min_dinuc, max_dinuc + 1):
                if i + k > len(din):
                    break
                e = z_hunt_window_energy(din[i:i + k])
                if np.isfinite(e):
                    cands.append((e / k, frame + 2 * i, frame + 2 * (i + k) - 1))
    if not cands:
        return {"z_best_kcal": float("nan"), "z_best_start": -1.0,
                "z_best_end": -1.0, "z_best_len_bp": 0.0,
                "z_best_centre": float("nan")}
    best_e = min(c[0] for c in cands)
    # **Ties are resolved by averaging, not by search order.** In repetitive sequence many
    # windows share the best energy, and taking the first one found picks a different
    # window on the reverse complement -- which broke the antisymmetry of the centre
    # offset (forward -11 against reverse +9 on a test sequence). The set of tied-optimal
    # windows maps onto its own mirror image under reverse complement, so its mean centre
    # and mean length are exactly antisymmetric and symmetric respectively, by
    # construction rather than by luck.
    tied = [c for c in cands if c[0] <= best_e + 1e-9]
    centre = float(np.mean([0.5 * (a + b) for _, a, b in tied]))
    length = float(np.mean([b - a + 1 for _, a, b in tied]))
    rep = min(tied, key=lambda c: abs(0.5 * (c[1] + c[2]) - centre))
    return {"z_best_kcal": float(best_e / 2.0),         # per base pair
            "z_best_start": float(rep[1]), "z_best_end": float(rep[2]),
            "z_best_len_bp": length, "z_best_centre": centre}


# --------------------------------------------------------------------------------------
# Mirror repeats (triplex / H-DNA) and inverted repeats (cruciform)
# --------------------------------------------------------------------------------------

_PURINE = set("AG")
_PYRIMIDINE = set("CT")


def purine_fraction(seq: str) -> float:
    s = seq.upper()
    if not s:
        return float("nan")
    return sum(1 for c in s if c in _PURINE) / len(s)


def find_mirror_repeats(seq: str, min_arm: int = 8,
                        max_spacer: int = 8) -> List[Tuple[int, int, int]]:
    """Mirror repeats, which are the sequence requirement for intramolecular triplex.

    Returns (centre, arm_length, spacer). A mirror repeat reads the same forwards and
    backwards about its centre -- unlike an inverted repeat, it is not complemented. H-DNA
    additionally requires the tract be homopurine/homopyrimidine, which
    `triplex_candidates` applies on top of this.
    """
    s = seq.upper()
    n = len(s)
    out: List[Tuple[int, int, int]] = []
    for spacer in range(0, max_spacer + 1):
        for centre in range(n):
            lo = centre - 1
            hi = centre + spacer
            arm = 0
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                arm += 1
                lo -= 1
                hi += 1
            if arm >= min_arm:
                out.append((centre, arm, spacer))
    return out


def triplex_candidates(seq: str, min_arm: int = 8, max_spacer: int = 8,
                       min_purine: float = 0.9) -> List[Dict[str, float]]:
    """Mirror repeats that are also homopurine or homopyrimidine: H-DNA candidates."""
    s = seq.upper()
    out = []
    for centre, arm, spacer in find_mirror_repeats(s, min_arm, max_spacer):
        lo, hi = centre - arm, centre + spacer + arm
        tract = s[max(0, lo):min(len(s), hi)]
        pf = purine_fraction(tract)
        if pf >= min_purine or (1.0 - pf) >= min_purine:
            out.append({"centre": float(centre), "arm": float(arm),
                        "spacer": float(spacer), "purine_fraction": float(pf),
                        "length": float(len(tract))})
    return out


def find_inverted_repeats(seq: str, min_arm: int = 6,
                          max_spacer: int = 10) -> List[Dict[str, float]]:
    """Inverted repeats, the sequence requirement for cruciform extrusion.

    An inverted repeat has one arm equal to the reverse complement of the other, so each
    arm can pair with itself to extrude a hairpin on each strand.
    """
    s = seq.upper()
    n = len(s)
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    out: List[Dict[str, float]] = []
    for spacer in range(0, max_spacer + 1):
        for centre in range(n):
            lo = centre - 1
            hi = centre + spacer
            arm = 0
            gc = 0
            while lo >= 0 and hi < n and comp.get(s[lo]) == s[hi]:
                arm += 1
                if s[lo] in "GC":
                    gc += 1
                lo -= 1
                hi += 1
            if arm >= min_arm:
                out.append({"centre": float(centre), "arm": float(arm),
                            "spacer": float(spacer),
                            "stem_gc_fraction": float(gc / arm)})
    return out


# --------------------------------------------------------------------------------------
# R-loop forming sequences
# --------------------------------------------------------------------------------------

def gc_skew(seq: str) -> float:
    """(G - C) / (G + C). Positive skew on the non-template strand favours R-loops."""
    s = seq.upper()
    g, c = s.count("G"), s.count("C")
    if g + c == 0:
        return 0.0
    return (g - c) / (g + c)


def rloop_zones(seq: str, riz_window: int = 50, rez_window: int = 100,
                min_riz_g: float = 0.4) -> Dict[str, float]:
    """RIZ/REZ decomposition of an R-loop forming sequence.

    Following the quantitative RLFS model: initiation needs a G-cluster-rich zone (RIZ),
    and elongation needs a following G-rich zone (REZ) whose favourability tracks GC skew.
    Reported as the best initiation score found, the best elongation score, and their
    separation.
    """
    s = seq.upper()
    n = len(s)
    if n < riz_window:
        return {"riz_best": float("nan"), "rez_best": float("nan"),
                "rloop_score": float("nan"), "riz_position": -1.0}
    g = np.array([1.0 if ch == "G" else 0.0 for ch in s])
    cs = np.concatenate([[0.0], np.cumsum(g)])

    riz_scores = (cs[riz_window:] - cs[:-riz_window]) / riz_window
    riz_i = int(np.argmax(riz_scores))
    riz_best = float(riz_scores[riz_i])

    rez_best = float("nan")
    if riz_i + riz_window < n:
        tail = s[riz_i + riz_window:]
        if len(tail) >= 10:
            w = min(rez_window, len(tail))
            rez_best = float(gc_skew(tail[:w]))

    score = 0.0
    if riz_best >= min_riz_g and np.isfinite(rez_best):
        score = riz_best * max(0.0, rez_best)
    return {"riz_best": riz_best, "rez_best": rez_best,
            "rloop_score": float(score), "riz_position": float(riz_i)}


# --------------------------------------------------------------------------------------
# A-phased bending, repeats, single-strandedness
# --------------------------------------------------------------------------------------

_A_TRACT = re.compile(r"(?:A{3,}|T{3,})")


def a_phasing(seq: str, period_bp: float = CANONICAL_HELICAL_PERIOD_BP) -> Dict[str, float]:
    """A-tracts repeating in phase with the helical period produce macroscopic curvature.

    Returns the number of A-tracts, the mean spacing between their centres, and a phasing
    coherence: the magnitude of the unit-vector sum of the spacings mapped onto the helical
    period, which is 1 when every tract sits at the same helical phase and 0 when they are
    scattered.
    """
    s = seq.upper()
    centres = [(m.start() + m.end()) / 2.0 for m in _A_TRACT.finditer(s)]
    if len(centres) < 2:
        return {"a_tract_count": float(len(centres)), "a_tract_mean_spacing": float("nan"),
                "a_phasing_coherence": 0.0}
    spacings = np.diff(np.array(centres))
    phases = 2.0 * np.pi * spacings / period_bp
    coherence = float(np.abs(np.mean(np.exp(1j * phases))))
    return {"a_tract_count": float(len(centres)),
            "a_tract_mean_spacing": float(np.mean(spacings)),
            "a_phasing_coherence": coherence}


def repeat_descriptors(seq: str, max_unit: int = 6,
                       min_copies: int = 3) -> Dict[str, float]:
    """Tandem repeats, which drive slipped-strand structures and repeat instability."""
    s = seq.upper()
    n = len(s)
    best_copies = 0.0
    best_unit = 0.0
    total_repeat_bp = 0
    covered = np.zeros(n, dtype=bool)
    for unit in range(1, max_unit + 1):
        i = 0
        while i + unit <= n:
            u = s[i:i + unit]
            copies = 1
            j = i + unit
            while j + unit <= n and s[j:j + unit] == u:
                copies += 1
                j += unit
            if copies >= min_copies:
                covered[i:j] = True
                if copies > best_copies:
                    best_copies = float(copies)
                    best_unit = float(unit)
                i = j
            else:
                i += 1
    total_repeat_bp = int(covered.sum())
    return {"max_tandem_copies": best_copies, "max_tandem_unit_bp": best_unit,
            "tandem_repeat_fraction": float(total_repeat_bp / n) if n else float("nan")}


def at_fraction(seq: str) -> float:
    s = seq.upper()
    if not s:
        return float("nan")
    return sum(1 for c in s if c in "AT") / len(s)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

PH_VALUES: Tuple[float, ...] = (5.5, 6.5, 7.4)          # plan: i-motif and triplex at 3 pH
SUPERHELICAL_DENSITIES: Tuple[float, ...] = (-0.06, -0.03, 0.0)   # plan: cruciform at 3
Z_IONIC_STRENGTHS_M: Tuple[float, ...] = (0.01, 0.15, 0.50)       # plan: Z-DNA at 3


def noncanonical_channels(seq: str) -> Dict[str, float]:
    """Every Family F channel for one sequence, as a flat name -> value mapping.

    GLOBAL and WINDOW channels only; the per-base profiles are returned by
    `noncanonical_profiles` so callers that want tracks do not pay for scalars they
    will discard.
    """
    s = seq.upper()
    out: Dict[str, float] = {}

    # --- quadruplex -------------------------------------------------------------------
    prof = g4hunter_profile(s)
    cands = find_g4_candidates(s)
    out["F_g4hunter_mean"] = float(np.mean(prof)) if prof.size else float("nan")
    out["F_g4hunter_max"] = float(np.max(prof)) if prof.size else float("nan")
    out["F_g4hunter_min"] = float(np.min(prof)) if prof.size else float("nan")
    # Strand-free summary: the strongest quadruplex propensity on either strand.
    out["F_g4hunter_absmax"] = float(np.max(np.abs(prof))) if prof.size else float("nan")
    out["F_g4_candidate_count"] = float(len(cands))

    if cands:
        best = max(cands, key=g4_stability_score)
        stab = g4_stability_score(best)
        topo = g4_topology_probabilities(best)
        out["F_g4_best_stability"] = float(stab)
        out["F_g4_best_tetrads"] = float(best.n_tetrads)
        out["F_g4_best_n_tracts"] = float(best.n_tracts)
        out["F_g4_best_loop_mean"] = float(np.mean(best.loop_lengths))
        out["F_g4_best_loop_min"] = float(np.min(best.loop_lengths))
        out["F_g4_best_loop_max"] = float(np.max(best.loop_lengths))
        out["F_g4_best_has_bulge"] = 1.0 if best.has_bulge else 0.0
        out["F_g4_best_strand"] = 1.0 if best.strand == "+" else -1.0
        for k, v in topo.items():
            out[f"F_g4_topology_{k}"] = float(v)
        for cat, w in CATION_G4_WEIGHT.items():
            out[f"F_g4_stability_{cat}"] = float(stab * w)
        # Quadruplex-duplex competition: quadruplex stability against the duplex it
        # must displace, proxied by the GC content of the same tract.
        tract = s[best.start:best.end]
        gc = (tract.count("G") + tract.count("C")) / max(1, len(tract))
        out["F_g4_duplex_competition"] = float(stab - 4.0 * gc)
    else:
        for k in ("F_g4_best_stability", "F_g4_best_tetrads", "F_g4_best_n_tracts",
                  "F_g4_best_loop_mean", "F_g4_best_loop_min", "F_g4_best_loop_max",
                  "F_g4_best_has_bulge", "F_g4_best_strand",
                  "F_g4_duplex_competition"):
            out[k] = float("nan")
        for k in ("parallel", "antiparallel", "hybrid"):
            out[f"F_g4_topology_{k}"] = float("nan")
        for cat in CATION_G4_WEIGHT:
            out[f"F_g4_stability_{cat}"] = float("nan")

    # --- i-motif ----------------------------------------------------------------------
    im = imotif_profile(s)
    im_max = float(np.max(im)) if im.size else float("nan")
    out["F_imotif_propensity"] = im_max
    for ph in PH_VALUES:
        out[f"F_imotif_pH{ph:g}"] = float(im_max * imotif_ph_factor(ph))

    # --- Z-DNA ------------------------------------------------------------------------
    z = z_dna_best_tract(s)
    out["F_zdna_best_kcal_per_bp"] = z["z_best_kcal"]
    # Reported as displacement of the tract centre from the fragment centre, which
    # transforms antisymmetrically under reverse complement. Raw start/end indices do
    # not: they map i -> n-1-i, which is a coordinate change, not a parity, and a
    # channel declared with the wrong parity is the error PLAN.md calls unrecoverable.
    if np.isfinite(z.get("z_best_centre", float("nan"))):
        out["F_zdna_centre_offset_bp"] = float(z["z_best_centre"] - 0.5 * (len(s) - 1))
    else:
        out["F_zdna_centre_offset_bp"] = float("nan")
    out["F_zdna_tract_len_bp"] = z["z_best_len_bp"]
    out["F_zdna_junction_kcal"] = float(Z_NUCLEATION_KCAL)
    zp = z_dna_profile(s)
    out["F_zdna_mean_step_kcal"] = float(np.mean(zp)) if zp.size else float("nan")
    # Salt lowers the B-Z transition energy; the published values are low-salt, so the
    # ionic axis is a monotone scaling and is marked provisional.
    for ionic in Z_IONIC_STRENGTHS_M:
        out[f"F_zdna_kcal_I{ionic:g}"] = float(z["z_best_kcal"] * (1.0 - 0.3 * math.log10(
            max(ionic, 1e-3) / 0.01) / 2.0))

    # --- triplex / H-DNA --------------------------------------------------------------
    tri = triplex_candidates(s)
    out["F_triplex_candidate_count"] = float(len(tri))
    out["F_triplex_best_arm"] = float(max((t["arm"] for t in tri), default=float("nan")))
    out["F_triplex_max_purine_fraction"] = float(
        max((max(t["purine_fraction"], 1.0 - t["purine_fraction"]) for t in tri),
            default=float("nan")))
    base_tri = out["F_triplex_best_arm"]
    for ph in PH_VALUES:
        # Pyrimidine-motif triplex needs protonated cytosine, so it is pH dependent in
        # the same direction as the i-motif.
        out[f"F_triplex_pH{ph:g}"] = float(
            base_tri * imotif_ph_factor(ph)) if np.isfinite(base_tri) else float("nan")

    # --- cruciform --------------------------------------------------------------------
    inv = find_inverted_repeats(s)
    out["F_inverted_repeat_count"] = float(len(inv))
    best_arm = max((d["arm"] for d in inv), default=float("nan"))
    out["F_cruciform_best_arm"] = float(best_arm)
    out["F_cruciform_best_stem_gc"] = float(
        max(((d["stem_gc_fraction"] for d in inv)), default=float("nan")))
    out["F_cruciform_min_spacer"] = float(
        min((d["spacer"] for d in inv), default=float("nan")))
    for sigma in SUPERHELICAL_DENSITIES:
        # Negative supercoiling drives extrusion; the driving force scales with |sigma|.
        out[f"F_cruciform_sigma{sigma:g}"] = float(
            best_arm * abs(sigma) * 10.0) if np.isfinite(best_arm) else float("nan")
    # Branch migration needs sequence symmetry about the junction.
    out["F_branch_migration"] = float(
        np.mean([d["arm"] for d in inv])) if inv else float("nan")

    # --- R-loop -----------------------------------------------------------------------
    rl = rloop_zones(s)
    out["F_rloop_initiation"] = rl["riz_best"]
    out["F_rloop_elongation"] = rl["rez_best"]
    out["F_rloop_score"] = rl["rloop_score"]
    out["F_gc_skew"] = float(gc_skew(s))

    # --- A-phasing, repeats, single-strandedness --------------------------------------
    out.update({f"F_{k}": v for k, v in a_phasing(s).items()})
    out.update({f"F_{k}": v for k, v in repeat_descriptors(s).items()})
    out["F_at_fraction"] = float(at_fraction(s))
    # Single-strandedness is the expected fraction of base pairs open, which is what
    # KAS-seq and permanganate footprinting read. It was the AT fraction, flagged as a
    # proxy waiting on a Poland-Scheraga calculation; Family E now computes that exactly,
    # under the same negative supercoiling a transcribed locus carries.
    import thermodynamics as _thermo
    _p = _thermo.helix_coil_supercoiled(s, -0.06, 37.0, 0.15)["open_prob"]
    out["F_single_strandedness"] = float(np.nanmean(_p)) if np.isfinite(_p).any() \
        else float("nan")
    # Triplet repeat instability: tandem repeats of unit 3 are the unstable class.
    out["F_triplet_repeat_instability"] = float(
        repeat_descriptors(s, max_unit=3)["max_tandem_copies"]
        if repeat_descriptors(s, max_unit=3)["max_tandem_unit_bp"] == 3.0 else 0.0)
    # Transition cooperativity length: how far the G4Hunter profile stays correlated.
    out["F_transition_cooperativity_bp"] = _correlation_length(prof)

    return out


def _correlation_length(x: np.ndarray, max_lag: int = 40) -> float:
    """Lag at which the profile's autocorrelation first falls below 1/e."""
    if x.size < 4:
        return float("nan")
    y = x - x.mean()
    denom = float(np.dot(y, y))
    if denom <= 0:
        return 0.0
    for lag in range(1, min(max_lag, x.size - 1)):
        r = float(np.dot(y[:-lag], y[lag:])) / denom
        if r < math.exp(-1.0):
            return float(lag)
    return float(min(max_lag, x.size - 1))


def noncanonical_profiles(seq: str) -> Dict[str, np.ndarray]:
    """Per-base Family F tracks."""
    s = seq.upper()
    zp = z_dna_profile(s)
    # Put the per-step Z energy onto bases by averaging the two steps each base joins.
    z_bp = np.full(len(s), np.nan)
    if zp.size:
        z_bp[0] = zp[0]
        z_bp[-1] = zp[-1]
        if len(s) > 2:
            z_bp[1:-1] = 0.5 * (zp[:-1] + zp[1:])
    return {
        "F_g4hunter": g4hunter_profile(s),
        "F_imotif": imotif_profile(s),
        "F_zdna_step_kcal": z_bp,
    }


def window_channels(seq: str, ladder: Sequence[int] = WINDOW_LADDER_BP
                    ) -> Dict[str, np.ndarray]:
    """Family F quantities on the revision-2 window ladder.

    The plan's multiresolution section says these become sliding-window tracks at
    inference, so they are produced at every rung rather than only at one.
    """
    s = seq.upper()
    out: Dict[str, np.ndarray] = {}
    prof = g4hunter_profile(s)
    for w in ladder:
        if w > len(s):
            continue
        cs = np.concatenate([[0.0], np.cumsum(prof)])
        n_win = len(s) - w + 1
        out[f"F_g4hunter_w{w}"] = (cs[w:w + n_win] - cs[:n_win]) / w
        skew = np.empty(n_win)
        for i in range(n_win):
            skew[i] = gc_skew(s[i:i + w])
        out[f"F_gc_skew_w{w}"] = skew
    return out


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

# Many Family F channels are *definitions* rather than parameterised models: a count of
# G-quadruplex candidates, the arm length of the longest inverted repeat, the fraction of
# a fragment in tandem repeat. There is no publication to cite for the value of such a
# channel, because there is no fitted parameter in it -- only the stated criterion in the
# function that computes it. They still owe a declared provenance, so that "no source"
# always means an omission and never means "there is nothing to say". This marker is
# what they carry, and the registry contract test treats it as a source.
_DEFINITION = ("structural definition evaluated on the sequence; no fitted parameters, "
               "the criterion is stated in the computing function's docstring")


def _spec(name: str, parity: Parity, units: str, desc: str,
          provisional: bool = False, reason: str = "", source: str = _DEFINITION,
          resolution: Resolution = Resolution.GLOBAL,
          window_bp: Optional[int] = None,
          condition_dependent: bool = False) -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=resolution, parity=parity,
                       units=units, description=desc, window_bp=window_bp,
                       condition_dependent=condition_dependent,
                       provisional=provisional, provisional_reason=reason, source=source)


_G4HUNTER_SRC = "Bedrat, Lacroix & Mergny, Nucleic Acids Research 44:1746, 2016"
_Z_SRC = ("Zhabinskaya & Benham, PLOS Comput Biol 7:e1001051, 2011, citing "
          "Ho, Ellison, Quigley & Rich, EMBO J 5:2737, 1986")
_RLFS_SRC = "Wongsurawat et al., Nucleic Acids Research 40:e16, 2012"
_G4_PROV = ("stability is a dimensionless ordering from tetrad count and loop length, "
            "not a parameterised free energy; cation axis is a monotone encoding of the "
            "K > Na > Li ordering, not measured offsets")

_SPECS: List[ChannelSpec] = [
    # G4Hunter and quadruplex structure -- not provisional, these are as published.
    # G4Hunter negates on the complement by construction (a G run is a C run on the
    # other strand), so the mean is ANTISYMMETRIC, not symmetric.
    _spec("F_g4hunter_mean", Parity.ANTISYMMETRIC, "score",
          "mean G4Hunter score over the fragment", source=_G4HUNTER_SRC),
    # max and min do not each map to themselves under reverse complement; they map to
    # each other with a sign change (max_forward == -min_reverse). That is an exchange
    # between strands, which is what STRANDED means. Declaring either SYMMETRIC or
    # ANTISYMMETRIC would be wrong, and `tests/test_channels.py` checks the exchange
    # relation explicitly rather than skipping these.
    _spec("F_g4hunter_max", Parity.STRANDED, "score",
          "maximum windowed G4Hunter score; G-rich on the given strand",
          source=_G4HUNTER_SRC),
    _spec("F_g4hunter_min", Parity.STRANDED, "score",
          "minimum windowed G4Hunter score; C-rich on the given strand",
          source=_G4HUNTER_SRC),
    _spec("F_g4hunter_absmax", Parity.SYMMETRIC, "score",
          "strongest quadruplex propensity on either strand", source=_G4HUNTER_SRC),
    _spec("F_g4_candidate_count", Parity.SYMMETRIC, "count",
          "intramolecular quadruplex candidates on both strands"),
    _spec("F_g4_best_tetrads", Parity.SYMMETRIC, "count",
          "tetrad count of the most stable candidate"),
    _spec("F_g4_best_n_tracts", Parity.SYMMETRIC, "count", "G tracts in that candidate"),
    _spec("F_g4_best_loop_mean", Parity.SYMMETRIC, "bp", "mean loop length"),
    _spec("F_g4_best_loop_min", Parity.SYMMETRIC, "bp", "shortest loop"),
    _spec("F_g4_best_loop_max", Parity.SYMMETRIC, "bp", "longest loop"),
    _spec("F_g4_best_has_bulge", Parity.SYMMETRIC, "indicator",
          "unequal tract lengths, so the quadruplex is bulged"),
    _spec("F_g4_best_strand", Parity.ANTISYMMETRIC, "indicator",
          "+1 if the G-rich tract is on the given strand, -1 on the complement"),
    # Provisional: free energies and the cation axis.
    _spec("F_g4_best_stability", Parity.SYMMETRIC, "dimensionless",
          "quadruplex stability ordering", provisional=True, reason=_G4_PROV),
    _spec("F_g4_duplex_competition", Parity.SYMMETRIC, "dimensionless",
          "quadruplex stability against the duplex it displaces",
          provisional=True, reason=_G4_PROV),
    *[_spec(f"F_g4_topology_{k}", Parity.SYMMETRIC, "probability",
            f"soft probability of {k} topology", provisional=True,
            reason="loop-length heuristic, not a fitted classifier")
      for k in ("parallel", "antiparallel", "hybrid")],
    *[_spec(f"F_g4_stability_{c}", Parity.SYMMETRIC, "dimensionless",
            f"quadruplex stability under {c}+", provisional=True, reason=_G4_PROV,
            condition_dependent=True)
      for c in CATION_G4_WEIGHT],
    # i-motif
    # The i-motif forms on the C-rich strand, so this exchanges with the G-rich
    # strand's quadruplex score under reverse complement.
    _spec("F_imotif_propensity", Parity.STRANDED, "score",
          "maximum C-rich (i-motif-forming) windowed score", source=_G4HUNTER_SRC),
    *[_spec(f"F_imotif_pH{ph:g}", Parity.STRANDED, "score",
            f"i-motif propensity scaled by cytosine N3 protonation at pH {ph:g}",
            condition_dependent=True, provisional=True,
            reason="pH scaling uses the free-cytidine pKa; the context shift that lets "
                   "some i-motifs fold near neutral pH is not parameterised, and a search "
                   "of the literature found no closed-form model of the transitional pH "
                   "from C-tract and loop lengths -- only qualitative trends and machine-"
                   "learned or simulation-based predictors")
      for ph in PH_VALUES],
    # Z-DNA
    _spec("F_zdna_best_kcal_per_bp", Parity.SYMMETRIC, "kcal/mol/bp",
          "lowest per-base-pair B-to-Z energy over windows of 6 to 12 dinucleotides, in "
          "both reading frames, by Z-Hunt's own energy model (no junction term, as in "
          "Z-Hunt)", source=Z_HUNT_SRC),
    _spec("F_zdna_centre_offset_bp", Parity.ANTISYMMETRIC, "bp",
          "displacement of the most Z-favourable window's centre from the fragment "
          "centre, which is the boundary position in a strand-independent frame",
          source=Z_HUNT_SRC),
    _spec("F_zdna_tract_len_bp", Parity.SYMMETRIC, "bp", "length of that window",
          source=Z_HUNT_SRC),
    _spec("F_zdna_junction_kcal", Parity.SYMMETRIC, "kcal/mol",
          "B-Z junction nucleation cost per junction; a measured constant, identical "
          "for every fragment, so it is dropped as a training target by the constant-"
          "channel rule and kept as a declared physical parameter",
          source=Z_JUNCTION_SRC),
    _spec("F_zdna_mean_step_kcal", Parity.SYMMETRIC, "kcal/mol",
          "mean per-step B-to-Z propensity, min(AS, SA) per dinucleotide",
          source=Z_HUNT_SRC),
    *[_spec(f"F_zdna_kcal_I{i:g}", Parity.SYMMETRIC, "kcal/mol/bp",
            f"B-to-Z cost at ionic strength {i:g} M", condition_dependent=True,
            provisional=True,
            reason="salt scaling is a monotone placeholder. No parameterisation exists to "
                   "replace it: Zhabinskaya & Benham 2011 (PMC3024258) state that 'no "
                   "quantitative information is currently available regarding how these "
                   "free energies partition between entropy and enthalpy, nor about their "
                   "ionic strength dependences'")
      for i in Z_IONIC_STRENGTHS_M],
    # Triplex
    _spec("F_triplex_candidate_count", Parity.SYMMETRIC, "count",
          "homopurine/homopyrimidine mirror repeats"),
    _spec("F_triplex_best_arm", Parity.SYMMETRIC, "bp", "longest mirror-repeat arm"),
    _spec("F_triplex_max_purine_fraction", Parity.SYMMETRIC, "fraction",
          "purine or pyrimidine purity of the best tract"),
    *[_spec(f"F_triplex_pH{ph:g}", Parity.SYMMETRIC, "bp",
            f"pyrimidine-motif triplex propensity at pH {ph:g}",
            condition_dependent=True, provisional=True,
            reason="pH scaling shares the i-motif placeholder")
      for ph in PH_VALUES],
    # Cruciform
    _spec("F_inverted_repeat_count", Parity.SYMMETRIC, "count", "inverted repeats"),
    _spec("F_cruciform_best_arm", Parity.SYMMETRIC, "bp", "longest inverted-repeat arm"),
    _spec("F_cruciform_best_stem_gc", Parity.SYMMETRIC, "fraction",
          "GC fraction of the best stem"),
    _spec("F_cruciform_min_spacer", Parity.SYMMETRIC, "bp",
          "shortest spacer between inverted arms"),
    *[_spec(f"F_cruciform_sigma{s:g}", Parity.SYMMETRIC, "dimensionless",
            f"cruciform extrusion driving force at superhelical density {s:g}",
            condition_dependent=True, provisional=True,
            reason="extrusion free energy is not parameterised; this scales arm length "
                   "by |sigma|")
      for s in SUPERHELICAL_DENSITIES],
    _spec("F_branch_migration", Parity.SYMMETRIC, "bp",
          "mean inverted-repeat arm, as branch migration range"),
    # R-loop
    _spec("F_rloop_initiation", Parity.STRANDED, "fraction",
          "best RIZ G fraction", source=_RLFS_SRC),
    _spec("F_rloop_elongation", Parity.STRANDED, "skew",
          "GC skew of the REZ following the best RIZ", source=_RLFS_SRC),
    _spec("F_rloop_score", Parity.STRANDED, "dimensionless",
          "combined R-loop forming score", source=_RLFS_SRC),
    _spec("F_gc_skew", Parity.ANTISYMMETRIC, "skew", "(G-C)/(G+C) on the given strand"),
    # A-phasing, repeats, single-strandedness
    _spec("F_a_tract_count", Parity.SYMMETRIC, "count", "A-tracts of length 3 or more"),
    _spec("F_a_tract_mean_spacing", Parity.SYMMETRIC, "bp",
          "mean spacing between A-tract centres"),
    _spec("F_a_phasing_coherence", Parity.SYMMETRIC, "dimensionless",
          "coherence of A-tract spacing with the helical period"),
    _spec("F_max_tandem_copies", Parity.SYMMETRIC, "count", "largest tandem copy number"),
    _spec("F_max_tandem_unit_bp", Parity.SYMMETRIC, "bp", "its repeat unit length"),
    _spec("F_tandem_repeat_fraction", Parity.SYMMETRIC, "fraction",
          "fraction of the fragment inside a tandem repeat"),
    _spec("F_at_fraction", Parity.SYMMETRIC, "fraction", "A+T fraction"),
    _spec("F_single_strandedness", Parity.SYMMETRIC, "fraction",
          "expected fraction of base pairs open at 37 C, 0.15 M Na and superhelical "
          "density -0.06, from the exact helix-coil partition function in Family E",
          source="Zimm & Bragg 1959 with SantaLucia 1998 stacking and Benham 1993 "
                 "stress, as computed in thermodynamics.helix_coil_supercoiled"),
    _spec("F_triplet_repeat_instability", Parity.SYMMETRIC, "count",
          "tandem copies when the repeat unit is a triplet"),
    _spec("F_transition_cooperativity_bp", Parity.SYMMETRIC, "bp",
          "autocorrelation length of the G4Hunter profile"),
]

REGISTRY.extend(_SPECS)

# The per-base-pair profiles `noncanonical_profiles` returns were emitted without being
# declared, so nothing downstream could include them: the target-generation driver builds
# its column set from the registry. Parity measured on an exact palindrome, all three
# exact to 0.00e+00:
#
#   F_g4hunter        ANTISYMMETRIC -- the G-richness score of one strand is minus the
#                     C-richness score read from the other, which is the same sign flip
#                     already recorded for F_g4hunter_mean.
#   F_imotif          ANTISYMMETRIC -- for the same reason, being the C-run score. This
#                     one was expected to come out STRANDED and did not; the measurement
#                     decided it.
#   F_zdna_step_kcal  SYMMETRIC -- a dinucleotide transition energy.
_PROFILE_SPECS: List[ChannelSpec] = [
    ChannelSpec(name="F_g4hunter", family=FAMILY, resolution=Resolution.PER_BP,
                parity=Parity.ANTISYMMETRIC, units="score",
                description="G4Hunter score in a sliding window centred here; positive "
                            "is G-rich on this strand and negative C-rich",
                source=_G4HUNTER_SRC),
    ChannelSpec(name="F_imotif", family=FAMILY, resolution=Resolution.PER_BP,
                parity=Parity.ANTISYMMETRIC, units="score",
                description="i-motif propensity in a sliding window centred here, the "
                            "C-run counterpart of the G4Hunter score",
                source=_G4HUNTER_SRC),
    ChannelSpec(name="F_zdna_step_kcal", family=FAMILY, resolution=Resolution.PER_BP,
                parity=Parity.SYMMETRIC, units="kcal/mol",
                description="B-to-Z transition energy of this dinucleotide step",
                source=Z_HUNT_SRC),
]
REGISTRY.extend(_PROFILE_SPECS)



if __name__ == "__main__":
    import json
    import sys

    demo = sys.argv[1] if len(sys.argv) > 1 else (
        "GGGTTAGGGTTAGGGTTAGGGAAACGCGCGCGCGCGCTTTTTAAAAACCCTAACCCTAACCCTAACCC")
    ch = noncanonical_channels(demo)
    print(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                      for k, v in ch.items()}, indent=2))
    print(f"\nFamily F channels declared: {len(REGISTRY.by_family('F'))}")
    print(f"computed this call: {len(ch)}")
    print(f"provisional: {len([s for s in REGISTRY.by_family('F') if s.provisional])}")
