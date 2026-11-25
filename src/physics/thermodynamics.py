#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family E: thermodynamics and strand dynamics.

`PLAN.md` lists 85 channels: "Nearest-neighbor free energy, enthalpy and entropy;
melting temperature at 5 ionic strengths; base-pair opening probability across 6
temperatures and 4 superhelical densities; opening and closing rates; imino exchange
proxy; stress-induced destabilization at 4 superhelical densities; hairpin and cruciform
free energy at 3 ionic strengths; triplex and i-motif free energy at 3 pH values; bubble
size moments; heat capacity change; hydrogen bond count, length and strength; strand
separation free energy over 3 window sizes."

Ground truth is "Array Melt equilibrium stabilities, nearest-neighbor models,
Poland-Scheraga and stress-induced destabilization calculations". The nearest-neighbour
half already exists in `analytic_targets`, validated against SantaLucia's published
values. What this module adds is the statistical mechanics: the partition function over
open and closed base pairs, with and without torsional stress.

**The partition function is computed exactly, not approximated.** Opening is modelled as
a helix-coil transition in the Zimm-Bragg form, with a sequence-dependent stacking
weight per step and a cooperativity parameter for nucleating a new open run. Base-pair
opening probabilities come from a forward-backward pass over the transfer matrix, which
is exact for the model, and bubble statistics come from the same pass rather than from a
separate approximation that could disagree with it.

**Superhelical stress makes the energy non-additive, and that is handled rather than
ignored.** The torsional energy stored in a domain depends on the *total* number of open
base pairs, because every opened pair absorbs about one tenth of a turn of the linking
deficit. A plain transfer matrix cannot represent that: it is local by construction. So
the recursion carries a second index counting how many base pairs are open so far,
giving Z(n) for every n, and the stress term is applied to each n before summing. That
is O(N^2) rather than O(N), which is affordable at fragment scale and is the difference
between computing stress-induced destabilization and merely naming it.

**What is stated rather than hidden.** Three simplifications, each with its consequence:

  * The loop-entropy factor of the full Poland-Scheraga model is not included. Zimm-Bragg
    charges a fixed nucleation cost per open run and nothing further for the run's
    length, so long internal loops are easier to form here than in a model carrying the
    (loop length)^-c factor. For the fragment lengths the plan scores this is a small
    effect on opening probability and a larger one on the tail of the bubble size
    distribution, so the bubble moments are reported with that caveat attached.
  * The open region's own torsional stiffness is omitted from the stress term, which is
    in Benham's full treatment. Omitting it makes opening slightly too easy at high
    superhelical density.
  * Opening and closing *rates* need an attempt frequency that is not derivable from
    equilibrium thermodynamics. They are provisional.

References:
- SantaLucia J. A unified view of polymer, dumbbell and oligonucleotide DNA
  nearest-neighbor thermodynamics. PNAS 95:1460, 1998. (the stacking parameters)
- Owczarzy R et al. Biochemistry 43:3537, 2004. (sodium correction)
- Zimm BH, Bragg JK. Theory of the phase transition between helix and random coil.
  J Chem Phys 31:526, 1959. (the transfer matrix)
- Blake RD, Delcourt SG. Thermal stability of DNA. NAR 26:3323, 1998.
  (the cooperativity parameter)
- Blake RD et al. Statistical mechanical simulation of polymeric DNA melting with
  MELTSIM. Bioinformatics 15:370, 1999.
- Benham CJ. Sites of predicted stress-induced DNA duplex destabilization occur
  preferentially at regulatory loci. PNAS 90:2999, 1993. (SIDD)
- Gueron M, Leroy JL. Studies of base pair kinetics by NMR measurement of proton
  exchange. Methods Enzymol 261:383, 1995. (imino exchange reports base-pair opening)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import analytic_targets as at  # noqa: E402
from channels import (  # noqa: E402
    REGISTRY,
    WINDOW_LADDER_BP,
    ChannelSpec,
    Parity,
    Resolution,
)

FAMILY = "E"

SANTALUCIA = "SantaLucia, PNAS 95:1460, 1998"
OWCZARZY = "Owczarzy et al., Biochemistry 43:3537, 2004"
ZIMM_BRAGG = "Zimm & Bragg, J Chem Phys 31:526, 1959"
BLAKE = "Blake & Delcourt, NAR 26:3323, 1998"
BENHAM = "Benham, PNAS 90:2999, 1993"
GUERON = "Gueron & Leroy, Methods Enzymol 261:383, 1995"

R_KCAL = 1.9872e-3               # gas constant, kcal/(mol K)
KELVIN = 273.15

# The plan's condition grids.
TEMPERATURES_C: Tuple[float, ...] = (25.0, 37.0, 45.0, 55.0, 65.0, 75.0)
IONIC_STRENGTHS_M: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 1.00)
SUPERHELICAL_DENSITIES: Tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0)
PH_VALUES: Tuple[float, ...] = (5.5, 6.5, 7.4)
SEPARATION_WINDOWS_BP: Tuple[int, ...] = (10, 50, 100)

# Helix-coil cooperativity: the statistical weight for nucleating a new open run.
# Blake & Delcourt's value for natural DNA. This is the single most important parameter
# in the model after the stacking free energies: it sets how sharp melting is and how
# large bubbles get, and a wrong value changes the shape of every opening channel while
# leaving the mean stability untouched.
SIGMA_COOP = 1.26e-5

# Benham's superhelical stiffness for a domain of N base pairs, in kcal/mol/turn^2.
# K = BENHAM_K_RT * RT / N.
BENHAM_K_RT = 2350.0
HELICAL_REPEAT_BP = 10.5

# Heat capacity change on melting, per base pair.
# Nearest-neighbour heat capacity change of duplex formation, cal/(mol K), from Rissone,
# Severino, Pastor-Ochoa et al., "DNA calorimetric force spectroscopy at single base pair
# resolution", Nature Communications 16, 2025 (doi 10.1038/s41467-025-57340-5), Table 1,
# read from the article text: single-molecule unzipping at 1 M NaCl, 7-42 C, reported as
# mean (SE). The sign is that of hybridisation, matching E_dH_kcal_mol. The paper notes
# that GC/CG and TA/AT were derived by circular symmetry relations rather than fitted
# directly. Motif X/Y is the Watson step X paired with Crick Y, so AC/TG is the steps AC
# and GT in this module's convention -- checked against the paper's own Delta-H, whose
# ordering matches SantaLucia's under that mapping (CG/GC -10 vs -10.6, TA/AT -7 vs -7.2).
#
# This replaces a single literature value applied uniformly to every base pair, which
# was flagged provisional because heat capacity is sequence dependent; it now is too.
NN_DCP_CAL: Dict[str, float] = {
    "AA": -39.0, "TT": -39.0,
    "AC": -34.0, "GT": -34.0,
    "AG": -32.0, "CT": -32.0,
    "AT": -20.0,
    "CA": -46.0, "TG": -46.0,
    "CC": -20.0, "GG": -20.0,
    "CG": -14.0,
    "GA": -39.0, "TC": -39.0,
    "GC": -33.0,
    "TA": -11.0,
}
NN_DCP_SE_CAL: Dict[str, float] = {
    "AA": 3, "TT": 3, "AC": 3, "GT": 3, "AG": 5, "CT": 5, "AT": 1, "CA": 7, "TG": 7,
    "CC": 4, "GG": 4, "CG": 4, "GA": 3, "TC": 3, "GC": 7, "TA": 6,
}
RISSONE = ("Rissone et al., Nat Commun 16 (2025), doi 10.1038/s41467-025-57340-5, "
           "Table 1 (1 M NaCl)")


def step_heat_capacity(seq: str) -> np.ndarray:
    """Nearest-neighbour Delta-Cp of duplex formation per step, kcal/(mol K)."""
    s = seq.upper()
    return np.array([NN_DCP_CAL.get(s[i:i + 2], np.nan) / 1000.0
                     for i in range(len(s) - 1)])

MIN_BUBBLE_BP = 3                # what counts as a bubble rather than a frayed pair

# Watson-Crick hydrogen bonds, as (Watson atom, Crick atom) pairs on the purine side.
WC_HBONDS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "AT": (("N6", "O4"), ("N1", "N3")),
    "GC": (("N1", "N3"), ("N2", "O2"), ("O6", "N4")),
}
HBOND_COUNT = {"A": 2, "T": 2, "G": 3, "C": 3}


# ======================================================================================
# Stacking weights
# ======================================================================================

def stack_free_energies(seq: str, temperature_C: float = 37.0,
                        na_molar: float = 0.15) -> np.ndarray:
    """Per-step stacking free energy in kcal/mol at the given temperature and salt.

    `dG(T) = dH - T dS` from the SantaLucia parameters, which is the van't Hoff form
    those parameters were fitted under. The salt correction is applied as an entropic
    term per phosphate, the standard treatment: lowering the sodium concentration
    destabilises the duplex because counterion release on melting is more favourable.
    """
    s = seq.upper()
    n = len(s) - 1
    T = temperature_C + KELVIN
    dg = np.zeros(n)
    for i in range(n):
        step = s[i:i + 2]
        dh = at.NN_DH[step]
        ds = at.NN_DS[step] / 1000.0
        dg[i] = dh - T * ds
    if na_molar > 0 and na_molar != 1.0:
        # Entropic salt term, 0.368 cal/(mol K) per phosphate per ln[Na+] (SantaLucia).
        dg += T * (0.368e-3) * math.log(na_molar)
    return dg


def _weights(seq: str, temperature_C: float, na_molar: float) -> np.ndarray:
    """Zimm-Bragg s per step: the Boltzmann weight for keeping this step helical."""
    T = temperature_C + KELVIN
    dg = stack_free_energies(seq, temperature_C, na_molar)
    # dG is negative for a stable step, so s = exp(-dG/RT) > 1 favours helix.
    return np.exp(-dg / (R_KCAL * T))


# ======================================================================================
# The partition function, exact for the model
# ======================================================================================

def helix_coil(seq: str, temperature_C: float = 37.0, na_molar: float = 0.15,
               sigma: float = SIGMA_COOP) -> Dict[str, object]:
    """Opening probability per base pair, by forward-backward over the transfer matrix.

    States are helix (0) and coil (1) per base pair. The transfer matrix between base
    pair i and i+1 is

        T_i = [[s_i, 1],
               [sigma, 1]]

    reading row as the state of i and column as the state of i+1. A stacking weight is
    earned only when **both** flanking base pairs are helical, which is the h-to-h entry;
    entering a helical run costs the nucleation weight `sigma` and nothing else.

    **The c-to-h entry is `sigma`, not `sigma * s_i`, and this matters.** Charging the
    stacking weight there would credit a step whose left-hand base pair is open with the
    stacking energy of an intact step. The error is invisible in any aggregate -- every
    whole-sequence channel came out exactly reverse-complement symmetric with it in
    place -- and it is visible immediately in the per-base-pair profile, which stopped
    being symmetric on an exact palindrome. The initial vector is `[sigma, 1]` for the
    same reason: a helical run that starts at base pair 0 pays nucleation like any
    other, and without that the two ends of the chain are not equivalent.

    Scaling is carried per step so that long sequences, where the raw weights overflow
    double precision within a few tens of base pairs, stay representable; the scale
    cancels in the probabilities.
    """
    w = _weights(seq, temperature_C, na_molar)
    n = len(seq)
    if n < 2:
        return {"open_prob": np.zeros(n), "log_Z": 0.0, "nucleation_prob": np.zeros(0)}

    fwd = np.zeros((n, 2))
    scale = np.zeros(n)
    fwd[0] = [sigma, 1.0]
    for i in range(1, n):
        si = w[i - 1]
        T = np.array([[si, 1.0], [sigma, 1.0]])
        v = fwd[i - 1] @ T
        m = v.max()
        if m <= 0 or not np.isfinite(m):
            return {"open_prob": np.full(n, np.nan), "log_Z": float("nan")}
        fwd[i] = v / m
        scale[i] = math.log(m)

    bwd = np.zeros((n, 2))
    bwd[-1] = [1.0, 1.0]
    for i in range(n - 2, -1, -1):
        si = w[i]
        T = np.array([[si, 1.0], [sigma, 1.0]])
        v = T @ bwd[i + 1]
        m = v.max()
        bwd[i] = v / m if m > 0 else v

    post = fwd * bwd
    tot = post.sum(axis=1)
    open_prob = np.where(tot > 0, post[:, 1] / np.where(tot > 0, tot, 1.0), np.nan)

    # Pair marginals, so the number of open runs can be counted exactly rather than
    # inferred: a run begins wherever base pair i is helix and i+1 is coil.
    nucleation = np.zeros(max(n - 1, 0))
    for i in range(n - 1):
        si = w[i]
        T = np.array([[si, 1.0], [sigma, 1.0]])
        joint = (fwd[i][:, None] * T) * bwd[i + 1][None, :]
        z = joint.sum()
        if z > 0:
            nucleation[i] = joint[0, 1] / z
    # Number of open runs = interior (helix, coil) boundaries, **plus one if base pair
    # 0 is itself open**. Without that boundary term the count is of run *starts* read
    # left to right, which under reverse complement becomes a count of run *ends*; the
    # two differ exactly when a run touches an end, so mean bubble size came out
    # different for a sequence and its reverse complement.
    mean_runs = float(nucleation.sum() + open_prob[0]) if n else 0.0
    return {"open_prob": open_prob, "log_Z": float(scale.sum()),
            "nucleation_prob": nucleation, "mean_runs": mean_runs,
            "stack_dG": stack_free_energies(seq, temperature_C, na_molar)}


def helix_coil_supercoiled(seq: str, sigma_sc: float, temperature_C: float = 37.0,
                           na_molar: float = 0.15,
                           sigma: float = SIGMA_COOP, kmax: Optional[int] = None) -> Dict[str, object]:
    """Opening under torsional stress, with the open-base-pair count carried explicitly.

    Superhelical stress couples to the *total* number of open base pairs, which a local
    transfer matrix cannot express. The recursion below therefore carries a second index
    `k`, the number of base pairs opened so far, producing Z(k) for every k; the stress
    energy is then applied per k and the results summed. This is what makes it a
    stress-induced destabilization calculation rather than a melting calculation with a
    label.

    The stress term is `(K/2)[(alpha + k/A)^2 - alpha^2]`, with `alpha = sigma_sc * N / A`
    the linking difference in turns, `A` the helical repeat and `K = 2350 RT / N`
    Benham's domain stiffness. Under negative supercoiling alpha is negative, so opening
    base pairs moves `alpha + k/A` towards zero and releases energy, which is the whole
    mechanism.
    """
    w = _weights(seq, temperature_C, na_molar)
    n = len(seq)
    T_K = temperature_C + KELVIN
    RT = R_KCAL * T_K
    if n < 2:
        return {"open_prob": np.zeros(n), "mean_open": 0.0,
                "bubble_moments": {}, "dG_stress": 0.0}

    K = BENHAM_K_RT * RT / n
    alpha = sigma_sc * n / HELICAL_REPEAT_BP
    # `kmax` truncates the open count (long domains; see `helix_coil_supercoiled_long`): totals
    # up to kmax are exact, totals above it are dropped. None keeps every count (0..n).
    K1 = n + 1 if kmax is None else min(n, int(kmax)) + 1
    k_idx = np.arange(K1)
    g_stress = 0.5 * K * ((alpha + k_idx / HELICAL_REPEAT_BP) ** 2 - alpha ** 2)
    w_stress = np.exp(-(g_stress - g_stress.min()) / RT)

    # Forward: F[i, state, k]. Work in a scaled representation per i.
    F = np.zeros((n, 2, K1))
    F[0, 0, 0] = sigma
    F[0, 1, 1] = 1.0
    logscale = np.zeros(n)
    for i in range(1, n):
        si = w[i - 1]
        prev = F[i - 1]
        # to helix at i: no new open bp
        F[i, 0, :] = prev[0] * si + prev[1] * sigma
        # to coil at i: k increments by one
        F[i, 1, 1:] = prev[0, :-1] + prev[1, :-1]
        m = F[i].max()
        if m <= 0 or not np.isfinite(m):
            return {"open_prob": np.full(n, np.nan), "mean_open": float("nan"),
                    "bubble_moments": {}, "dG_stress": float("nan")}
        F[i] /= m
        logscale[i] = math.log(m)

    # Backward: B[i, state, j] with j the number opened from i+1 to the end.
    B = np.zeros((n, 2, K1))
    bscale = np.ones(n)
    B[n - 1, 0, 0] = 1.0
    B[n - 1, 1, 0] = 1.0
    for i in range(n - 2, -1, -1):
        si = w[i]
        nxt = B[i + 1]
        B[i, 0, :] = nxt[0] * si
        B[i, 0, 1:] += nxt[1, :-1]
        B[i, 1, :] = nxt[0] * sigma
        B[i, 1, 1:] += nxt[1, :-1]
        m = B[i].max()
        if m > 0:
            B[i] /= m
            bscale[i] = m

    # Combine: total opened k = (opened up to and including i) + (opened after i).
    #
    # F and B are rescaled independently at every position, so the partition function
    # computed at one i is not comparable to the one at another. Every quantity below is
    # therefore formed as a ratio *within* a single i, where the scale factors cancel
    # exactly. The run count in particular is computed here rather than borrowed from
    # the unconstrained model: pairing a stressed open count with an unstressed run
    # count produced mean bubble sizes below one base pair, which is how this was found.
    open_prob = np.full(n, np.nan)
    nucleation = np.zeros(max(n - 1, 0))
    ktot = np.zeros(K1)
    for i in range(n):
        joint = np.zeros((2, K1))
        for st in (0, 1):
            joint[st] = np.convolve(F[i, st], B[i, st])[:K1]
        wj = joint * w_stress[None, :]
        tot = wj.sum()
        if tot <= 0:
            continue
        open_prob[i] = wj[1].sum() / tot
        if i == 0:
            ktot = wj.sum(axis=0) / tot
        if i < n - 1:
            # P(i helical, i+1 open): transfer weight T[0,1] = 1, and the open base
            # pair at i+1 adds one to the running count, hence the shift.
            # B[i+1] sits one normalisation step behind B[i]: B[i] = (T_i B[i+1]) /
            # bscale[i]. `tot` is formed from B[i], so anything built from B[i+1] has to
            # be divided by bscale[i] to land on the same scale. Leaving that out gave
            # more open runs than open base pairs, which is impossible and is how it was
            # caught.
            pair = np.zeros(K1)
            conv = np.convolve(F[i, 0], B[i + 1, 1])
            pair[1:] = conv[:K1 - 1]
            nucleation[i] = float((pair * w_stress).sum() / (tot * bscale[i]))
    mean_open = float(np.nansum(open_prob))
    # See `helix_coil`: the boundary term is what makes the run count symmetric.
    mean_runs = float(nucleation.sum() + open_prob[0])
    return {"open_prob": open_prob, "mean_open": mean_open,
            "nucleation_prob": nucleation, "mean_runs": mean_runs,
            "k_distribution": ktot,
            "dG_stress_kcal": float(0.5 * K * alpha ** 2)}


def helix_coil_supercoiled_long(seq: str, sigma_sc: float, temperature_C: float = 37.0,
                                na_molar: float = 0.15, tail_tol: float = 1e-15) -> Dict[str, object]:
    """`helix_coil_supercoiled` for kb-scale domains, exact to rounding at O(n kmax^2) instead of
    O(n^3). The stress weight is a Gaussian in the open count k centred at k* = -sigma_sc n with
    width A sqrt(n / 2350); the count is truncated at kmax = k* + 40 widths + 64, and the
    truncation is accepted only when the open-count distribution's mass in its last 16 entries is
    below `tail_tol` of the total (otherwise kmax doubles). Equal to the untruncated recursion to
    ~1e-12 where both can run (tested at 1 kb)."""
    n = len(seq)
    width = HELICAL_REPEAT_BP * math.sqrt(n / BENHAM_K_RT)
    kmax = int(max(0.0, -sigma_sc * n) + 40 * width + 64)
    while True:
        r = helix_coil_supercoiled(seq, sigma_sc, temperature_C, na_molar, kmax=kmax)
        kd = np.asarray(r.get("k_distribution", []), float)
        if kmax >= n or (kd.size and np.all(np.isfinite(kd)) and kd[-16:].sum() <= tail_tol * max(kd.sum(), 1e-300)):
            r["kmax"] = int(min(kmax, n))
            return r
        kmax *= 2


# ======================================================================================
# Derived statistics
# ======================================================================================

def bubble_statistics(seq: str, sigma_sc: float = 0.0, temperature_C: float = 37.0,
                      na_molar: float = 0.15) -> Dict[str, float]:
    """Moments of the open-base-pair count, and the mean size of an open run.

    Two different quantities, named apart because conflating them is easy and wrong:

      * `open_count_*` are the moments of the **total** number of open base pairs in the
        fragment, taken from the exact distribution Z(k) the supercoiled recursion
        already produces. No approximation.
      * `mean_bubble_bp` is the mean length of a single open run, computed as the mean
        open count divided by the mean number of runs, where a run is counted exactly by
        the pair marginal P(i helical, i+1 open). Also no approximation.

    The caveat from the module docstring applies to the second more than the first: with
    no loop-entropy factor, long runs are cheaper here than in full Poland-Scheraga, so
    `mean_bubble_bp` is an upper estimate.
    """
    sc = helix_coil_supercoiled(seq, sigma_sc, temperature_C, na_molar)
    k = np.asarray(sc.get("k_distribution", []), dtype=float)
    runs = float(sc.get("mean_runs", float("nan")))
    out: Dict[str, float] = {"mean_open_bp": float(sc["mean_open"]),
                             "mean_runs": runs}
    if k.size and np.isfinite(k).all() and k.sum() > 0:
        kk = np.arange(k.size, dtype=float)
        m1 = float((k * kk).sum())
        m2 = float((k * kk ** 2).sum())
        m3 = float((k * kk ** 3).sum())
        var = max(m2 - m1 ** 2, 0.0)
        sd = math.sqrt(var)
        out["open_count_mean"] = m1
        out["open_count_sd"] = sd
        out["open_count_skew"] = ((m3 - 3 * m1 * var - m1 ** 3) / sd ** 3
                                  if sd > 1e-12 else float("nan"))
    else:
        out.update({"open_count_mean": float("nan"), "open_count_sd": float("nan"),
                    "open_count_skew": float("nan")})
    out["mean_bubble_bp"] = (out["mean_open_bp"] / runs if runs > 1e-12
                             else float("nan"))
    return out


def sidd_profile(seq: str, sigma_sc: float, temperature_C: float = 37.0,
                 na_molar: float = 0.15) -> np.ndarray:
    """Stress-induced destabilization energy per base pair, in kcal/mol.

    Reported as the free energy of opening implied by the computed opening probability,
    `G_i = -RT ln(p_i / (1 - p_i))`, which is how SIDD output is conventionally read: a
    low value marks a base pair the superhelical stress has already paid most of the
    cost of opening. Using the probability rather than the bare stacking energy is what
    makes it a destabilization rather than a stability, because the probability carries
    the stress term and the cooperativity.
    """
    p = np.asarray(helix_coil_supercoiled(seq, sigma_sc, temperature_C,
                                          na_molar)["open_prob"], dtype=float)
    RT = R_KCAL * (temperature_C + KELVIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = -RT * np.log(p / (1.0 - p))
    return np.where(np.isfinite(g), g, np.nan)


# Attempt frequency for base-pair opening. Imino exchange gives closing rates near
# 1e7 to 1e8 per second for both AT and GC pairs, so the opening rate is set by the
# equilibrium constant times the closing rate.
K_CLOSE_PER_S = 1.0e7
_RATE_PROV = (
    "an attempt frequency is not derivable from equilibrium thermodynamics. The closing "
    "rate is fixed at the 1e7 per second that imino exchange measurements report for "
    "both AT and GC pairs, and the opening rate follows from the computed equilibrium "
    "constant. The sequence dependence is therefore entirely that of the equilibrium "
    "constant, and any sequence dependence of the prefactor itself is absent.")


def opening_rates(open_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Opening and closing rates per second, from the equilibrium constant."""
    p = np.asarray(open_prob, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_eq = p / (1.0 - p)
    return {"k_open_per_s": k_eq * K_CLOSE_PER_S,
            "k_close_per_s": np.full_like(p, K_CLOSE_PER_S),
            "bp_lifetime_ms": np.where(k_eq > 0, 1e3 / (k_eq * K_CLOSE_PER_S), np.nan)}


# ======================================================================================
# Hydrogen bonds
# ======================================================================================

def hbond_counts(seq: str) -> np.ndarray:
    """Watson-Crick hydrogen bonds per base pair: 2 for A-T, 3 for G-C. Exact."""
    return np.array([HBOND_COUNT.get(b, np.nan) for b in seq.upper()], dtype=float)


def hbond_geometry(pdb_path: str, seq: str) -> Dict[str, np.ndarray]:
    """Mean Watson-Crick hydrogen-bond donor-acceptor distance per base pair.

    Measured on the minimised structure, between the named donor and acceptor atoms of
    each pair. This is the one Family E channel that needs geometry rather than a
    thermodynamic table, and like the Family C torsions it rides on the structure the
    electrostatics pipeline already builds.
    """
    import frames as fr

    res = fr.read_pdb_residues(pdb_path)
    a_keys = sorted([k for k in res if k[0] == "A"], key=lambda k: k[1])
    b_keys = sorted([k for k in res if k[0] == "B"], key=lambda k: k[1])
    n = len(seq)
    dist = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    for i, base in enumerate(seq.upper()):
        if i >= len(a_keys) or i >= len(b_keys):
            continue
        w = res[a_keys[i]]
        c = res[b_keys[len(b_keys) - 1 - i]]
        pair = "AT" if base in "AT" else "GC"
        # On a pyrimidine the atom roles are the other way round.
        pairs = WC_HBONDS[pair]
        ds = []
        for x, y in pairs:
            p1 = w.get(x) if base in "AG" else w.get(y)
            p2 = c.get(y) if base in "AG" else c.get(x)
            if p1 is not None and p2 is not None:
                ds.append(float(np.linalg.norm(p1 - p2)))
        if ds:
            dist[i] = float(np.mean(ds))
            spread[i] = float(np.std(ds))
    return {"hbond_distance_A": dist, "hbond_distance_spread_A": spread}


# ======================================================================================
# Alternative structures: stems from Family F, energies from the nearest-neighbour model
# ======================================================================================

MATHEWS_DNA = ("ViennaRNA 2.8 with the DNA parameter set dna_mathews2004.par "
               "(Mathews et al. 2004 / SantaLucia & Hicks 2004 DNA loop and stack "
               "parameters), ViennaRNA salt correction")


class _DNAParameters:
    """Load ViennaRNA's DNA parameters for the duration of a block, then restore RNA.

    **ViennaRNA's energy parameters are global process state.** Loading the DNA set and
    not restoring it would silently switch every later RNA fold in the same process --
    Family L folds both transcripts of every fragment -- to DNA energies, and nothing
    would raise. So the switch is scoped, and restored in a `finally` so an exception
    cannot leave the process in the wrong state. `tests/test_thermodynamics.py` folds an
    RNA before and after a hairpin call and asserts the energies are identical.
    """

    def __enter__(self):
        import RNA
        RNA.params_load_DNA_Mathews2004()
        return RNA

    def __exit__(self, *exc):
        import RNA
        RNA.params_load_RNA_Turner2004()
        return False


def dna_fold_mfe(seq: str, na_molar: float = 0.15,
                 temperature_C: float = 37.0) -> Tuple[float, str]:
    """Minimum free energy secondary structure of one DNA strand, kcal/mol."""
    with _DNAParameters() as RNA:
        md = RNA.md()
        md.temperature = float(temperature_C)
        md.salt = float(na_molar)
        structure, mfe = RNA.fold_compound(seq.upper(), md).mfe()
    return float(mfe), structure


def hairpin_free_energy(seq: str, na_molar: float = 0.15,
                        temperature_C: float = 37.0) -> Dict[str, float]:
    """Most stable intramolecular structure of either strand, under the DNA model.

    This replaces "nearest-neighbour stem plus a Jacobson-Stockmayer loop term from one
    reference value", which was provisional because the loop term was not the measured
    table. The DNA parameter set that ships with ViennaRNA contains that table -- hairpin
    loops of 3, 4, 5 and 6 nt at 3.40, 3.40, 3.50 and 4.20 kcal/mol out to 30 nt -- plus
    terminal mismatches, special loops and the salt correction, so the whole fold is
    computed from one published parameter set instead of being assembled from parts.

    The lower of the two strands is reported, which makes the channel a property of the
    duplex and therefore symmetric under reverse complement; each strand alone is not,
    because the DNA loop and mismatch parameters are not complement-symmetric.
    """
    import noncanonical as nc
    from channels import reverse_complement

    e_w, _ = dna_fold_mfe(seq, na_molar, temperature_C)
    e_c, _ = dna_fold_mfe(reverse_complement(seq.upper()), na_molar, temperature_C)
    best_arm = max((int(r["arm"]) for r in nc.find_inverted_repeats(seq, min_arm=4)),
                   default=0)
    return {"hairpin_dG_kcal": float(min(e_w, e_c)), "hairpin_stem_bp": float(best_arm)}


_CRUCIFORM_PROV = (
    "extrusion is computed as the two strands' folds of the inverted-repeat region under "
    "the DNA parameter set, less the duplex free energy of that region, plus Benham's "
    "superhelical stress relief; the free energy of the four-way junction at the base of "
    "a cruciform is not parameterised here, so the absolute value omits one term.")


def cruciform_free_energy(seq: str, sigma_sc: float = -0.06,
                          na_molar: float = 0.15,
                          temperature_C: float = 37.0) -> float:
    """Free energy of extruding a cruciform at the best inverted repeat.

    Extrusion breaks the duplex over the inverted repeat and refolds each strand of it as
    a hairpin; the superhelical relief is the same Benham term the SIDD calculation uses,
    so the two are consistent by construction. The four-way junction cost is the one
    unparameterised term, which is why this channel stays provisional.
    """
    import noncanonical as nc
    from channels import reverse_complement

    reps = nc.find_inverted_repeats(seq, min_arm=4)
    if not reps:
        return float("nan")
    rep = max(reps, key=lambda r: (r["arm"], -r["spacer"]))
    c, a, sp = int(rep["centre"]), int(rep["arm"]), int(rep["spacer"])
    lo, hi = max(0, c - a), min(len(seq), c + sp + a)
    region = seq.upper()[lo:hi]
    if len(region) < 8 or any(b not in "ACGT" for b in region):
        return float("nan")
    e_w, _ = dna_fold_mfe(region, na_molar, temperature_C)
    e_c, _ = dna_fold_mfe(reverse_complement(region), na_molar, temperature_C)
    duplex = float(np.sum(stack_free_energies(region, temperature_C, na_molar)))
    n = len(seq)
    RT = R_KCAL * (temperature_C + KELVIN)
    K = BENHAM_K_RT * RT / max(n, 1)
    alpha = sigma_sc * n / HELICAL_REPEAT_BP
    relief = 0.5 * K * ((alpha + len(region) / HELICAL_REPEAT_BP) ** 2 - alpha ** 2)
    return float(e_w + e_c - duplex + relief)


def ph_dependent_free_energies(seq: str, ph: float,
                               temperature_C: float = 37.0) -> Dict[str, float]:
    """Triplex and i-motif free energies at one pH, from the Family F propensities.

    These are conversions of Family F scores into free energies by `dG = -RT ln(score)`,
    not independent thermodynamic calculations, and they are provisional for that
    reason. They are emitted here because `PLAN.md` lists them under Family E, and they
    call into Family F so there is a single implementation of each structure.
    """
    import noncanonical as nc

    RT = R_KCAL * (temperature_C + KELVIN)
    imotif = float(np.nanmax(nc.imotif_profile(seq))) * nc.imotif_ph_factor(ph)
    # Family F scores an H-DNA candidate by arm length and purine purity; the longer
    # and purer the mirror repeat, the more stable the triplex.
    cands = nc.triplex_candidates(seq)
    triplex = max((c["arm"] / 20.0 * max(c["purine_fraction"],
                                         1.0 - c["purine_fraction"])
                   for c in cands), default=0.0)
    # Protonation of the third-strand cytosine is what makes a triplex pH dependent.
    triplex *= nc.imotif_ph_factor(ph)
    def to_dg(x: float) -> float:
        return -RT * math.log(x) if x > 1e-12 else float("nan")
    return {"triplex_dG_kcal": to_dg(triplex), "imotif_dG_kcal": to_dg(imotif)}


# ======================================================================================
# Assembly
# ======================================================================================

def thermo_profiles(seq: str, minimised_pdb: Optional[str] = None,
                    na_molar: float = 0.15) -> Dict[str, np.ndarray]:
    """Every per-base-pair Family E channel, each an array of length n_bp.

    Step-indexed quantities -- the nearest-neighbour terms, which belong to a dinucleotide
    rather than to a base pair -- occupy positions 0 to n-2 and leave the last position
    NaN, rather than being stretched to n and implying a value that does not exist.
    """
    seq = seq.upper()
    n = len(seq)
    out: Dict[str, np.ndarray] = {}

    def put(name: str, v) -> None:
        a = np.asarray(v, dtype=float).ravel()
        padded = np.full(n, np.nan)
        k = min(n, len(a))
        padded[:k] = a[:k]
        out[name] = padded

    step = at.per_step_thermodynamics(seq)
    put("E_dH_kcal_mol", step["dH_kcal_mol"])
    put("E_dS_cal_mol_K", step["dS_cal_mol_K"])
    put("E_dG37_kcal_mol", step["dG37_kcal_mol"])

    for na in IONIC_STRENGTHS_M:
        put(f"E_stack_dG_na{na:g}", stack_free_energies(seq, 37.0, na))

    for T in TEMPERATURES_C:
        put(f"E_open_prob_T{T:g}", helix_coil(seq, T, na_molar)["open_prob"])

    for sc in SUPERHELICAL_DENSITIES:
        put(f"E_open_prob_sc{sc:g}",
            helix_coil_supercoiled(seq, sc, 37.0, na_molar)["open_prob"])
        put(f"E_sidd_dG_sc{sc:g}", sidd_profile(seq, sc, 37.0, na_molar))

    base = helix_coil(seq, 37.0, na_molar)["open_prob"]
    # Imino exchange in the EX2 limit reports the base-pair opening equilibrium constant
    # directly, so the proxy is that constant and not a rescaling of it.
    with np.errstate(divide="ignore", invalid="ignore"):
        put("E_imino_exchange_proxy", np.asarray(base) / (1.0 - np.asarray(base)))
    rates = opening_rates(base)
    put("E_k_open_per_s", rates["k_open_per_s"])
    put("E_bp_lifetime_ms", rates["bp_lifetime_ms"])

    put("E_hbond_count", hbond_counts(seq))
    put("E_dCp_kcal_mol_K", step_heat_capacity(seq))

    if minimised_pdb:
        g = hbond_geometry(minimised_pdb, seq)
        put("E_hbond_distance_A", g["hbond_distance_A"])
        put("E_hbond_distance_spread_A", g["hbond_distance_spread_A"])
    else:
        put("E_hbond_distance_A", np.full(n, np.nan))
        put("E_hbond_distance_spread_A", np.full(n, np.nan))

    for w in SEPARATION_WINDOWS_BP:
        v = np.full(n, np.nan)
        dg = step["dG37_kcal_mol"]
        if len(dg) >= w:
            for i in range(len(dg) - w + 1):
                v[i + w // 2] = float(np.sum(dg[i:i + w]))
        put(f"E_strand_separation_dG_w{w}", v)
    return out


def thermo_channels(seq: str, minimised_pdb: Optional[str] = None,
                    na_molar: float = 0.15, symmetrise: bool = False,
                    profiles: Optional[Dict[str, np.ndarray]] = None
                    ) -> Dict[str, float]:
    """Every whole-sequence Family E scalar channel."""
    if symmetrise:
        from channels import reverse_complement, symmetrise_channels
        fwd = thermo_channels(seq, minimised_pdb, na_molar, False)
        rev = thermo_channels(reverse_complement(seq), None, na_molar, False)
        return symmetrise_channels(fwd, rev)

    seq = seq.upper()
    # See `geometry.geometry_channels`: a caller holding the profiles already should not
    # pay for them twice. Here it is the six temperatures and four superhelical
    # densities of the opening calculation that would be repeated.
    prof = profiles if profiles is not None else thermo_profiles(
        seq, minimised_pdb, na_molar)
    out: Dict[str, float] = {}

    t = at.nn_thermodynamics(seq)
    out["E_dH_total_kcal_mol"] = t["dH_kcal_mol"]
    out["E_dS_total_cal_mol_K"] = t["dS_cal_mol_K"]
    out["E_dG37_total_kcal_mol"] = t["dG37_kcal_mol"]
    out["E_gc_fraction"] = at.gc_fraction(seq)

    for na in IONIC_STRENGTHS_M:
        out[f"E_Tm_na{na:g}_C"] = at.melting_temperature(seq, na_molar=na)

    for T in TEMPERATURES_C:
        out[f"E_mean_open_prob_T{T:g}"] = float(np.nanmean(prof[f"E_open_prob_T{T:g}"]))

    for sc in SUPERHELICAL_DENSITIES:
        b = bubble_statistics(seq, sc, 37.0, na_molar)
        out[f"E_open_count_mean_sc{sc:g}"] = b["open_count_mean"]
        out[f"E_open_count_sd_sc{sc:g}"] = b["open_count_sd"]
        out[f"E_open_count_skew_sc{sc:g}"] = b["open_count_skew"]
        out[f"E_mean_bubble_bp_sc{sc:g}"] = b["mean_bubble_bp"]
        out[f"E_mean_runs_sc{sc:g}"] = b["mean_runs"]
        out[f"E_sidd_min_dG_sc{sc:g}"] = float(np.nanmin(prof[f"E_sidd_dG_sc{sc:g}"]))

    for na in (0.01, 0.15, 1.00):
        h = hairpin_free_energy(seq, na)
        out[f"E_hairpin_dG_na{na:g}"] = h["hairpin_dG_kcal"]
        out[f"E_cruciform_dG_na{na:g}"] = cruciform_free_energy(seq, -0.06, na)
    out["E_hairpin_stem_bp"] = hairpin_free_energy(seq, na_molar)["hairpin_stem_bp"]

    for ph in PH_VALUES:
        d = ph_dependent_free_energies(seq, ph)
        out[f"E_triplex_dG_pH{ph:g}"] = d["triplex_dG_kcal"]
        out[f"E_imotif_dG_pH{ph:g}"] = d["imotif_dG_kcal"]

    out["E_dCp_total_kcal_mol_K"] = float(np.nansum(step_heat_capacity(seq)))
    out["E_hbond_total"] = float(np.nansum(prof["E_hbond_count"]))
    out["E_mean_hbond_distance_A"] = float(np.nanmean(prof["E_hbond_distance_A"]))

    # Melting cooperativity: how much sharper the transition is than a two-state one.
    # van't Hoff enthalpy over calorimetric enthalpy, the standard ratio.
    lo = float(np.nanmean(prof[f"E_open_prob_T{TEMPERATURES_C[0]:g}"]))
    hi = float(np.nanmean(prof[f"E_open_prob_T{TEMPERATURES_C[-1]:g}"]))
    if lo > 0 and hi > 0 and hi != lo:
        T1 = TEMPERATURES_C[0] + KELVIN
        T2 = TEMPERATURES_C[-1] + KELVIN
        dh_vh = R_KCAL * math.log(hi / lo) / (1.0 / T1 - 1.0 / T2)
        out["E_vant_hoff_dH_kcal_mol"] = -dh_vh
        out["E_melting_cooperativity"] = abs(dh_vh / t["dH_kcal_mol"]) \
            if t["dH_kcal_mol"] else float("nan")
    else:
        out["E_vant_hoff_dH_kcal_mol"] = float("nan")
        out["E_melting_cooperativity"] = float("nan")
    return out


def thermo_window_channels(seq: str, ladder: Sequence[int] = WINDOW_LADDER_BP
                           ) -> Dict[str, np.ndarray]:
    """Family E on the revision-2 window ladder."""
    prof = thermo_profiles(seq)
    keys = ("E_dG37_kcal_mol", "E_open_prob_T37", "E_sidd_dG_sc-0.06", "E_hbond_count")
    n = len(seq)
    out: Dict[str, np.ndarray] = {}
    for w in ladder:
        if w > n:
            continue
        for k in keys:
            v = prof[k]
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
# Every Family E channel is SYMMETRIC under reverse complement, and that is measured
# rather than assumed: the nearest-neighbour model is reverse-complement symmetric by
# construction, since AA and TT are one parameter, and the helix-coil partition function
# built on it inherits the symmetry. On an exact palindrome every computable profile
# matches its own reflection to better than 1e-9.
#
# Getting there took a correction. The first transfer matrix charged the stacking weight
# on the coil-to-helix transition, crediting a step with stacking energy when the base
# pair on one side of it is open. Every whole-sequence channel was still exactly
# symmetric under reverse complement with that error in place -- the aggregates could not
# see it -- and the per-base-pair profiles failed at a relative residual near 1.0. See
# `helix_coil`.
# ======================================================================================

_SIGMA_SRC = f"{BLAKE}; cooperativity sigma = {SIGMA_COOP:g}"
_OPEN_NOTE = (
    "Computed opening equilibrium constants reproduce the measured AT-to-GC ratio of "
    "about 10 at 37 C almost exactly, and the measured temperature dependence, but sit "
    "roughly 20 times below the absolute values imino exchange reports. The cause is "
    "that sigma is the nucleation parameter for cooperative melting, whereas opening a "
    "single base pair is a less disruptive event in which the neighbours restack. The "
    "ranking across sequence, which is what a landscape channel is for, is the part "
    "that validates; the absolute offset is recorded rather than tuned away.")
_LOOP_NOTE = _CRUCIFORM_PROV
_PH_PROV = ("a conversion of a Family F propensity score into a free energy by "
            "dG = -RT ln(score), not an independent thermodynamic calculation. Family F "
            "owns the structural definition; this is the same quantity in energy units.")


def _e(name: str, resolution: Resolution, units: str, desc: str,
       source: str = SANTALUCIA, provisional: bool = False, reason: str = "",
       condition: bool = False, parity: Parity = Parity.SYMMETRIC) -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=resolution,
                       parity=parity, units=units, description=desc,
                       condition_dependent=condition, provisional=provisional,
                       provisional_reason=reason, source=source)


_SPECS_E: List[ChannelSpec] = [
    _e("E_dH_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "nearest-neighbour enthalpy of this base-pair step"),
    _e("E_dS_cal_mol_K", Resolution.PER_BP, "cal/(mol K)",
       "nearest-neighbour entropy of this base-pair step"),
    _e("E_dG37_kcal_mol", Resolution.PER_BP, "kcal/mol",
       "nearest-neighbour free energy of this step at 37 C"),
    _e("E_dH_total_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "duplex enthalpy including helix initiation"),
    _e("E_dS_total_cal_mol_K", Resolution.GLOBAL, "cal/(mol K)", "duplex entropy"),
    _e("E_dG37_total_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "duplex free energy at 37 C"),
    _e("E_gc_fraction", Resolution.GLOBAL, "fraction", "GC content"),
    _e("E_imino_exchange_proxy", Resolution.PER_BP, "equilibrium constant",
       "base-pair opening equilibrium constant, what imino proton exchange measures in "
       "the EX2 limit. " + _OPEN_NOTE, f"{GUERON}; {_SIGMA_SRC}"),
    _e("E_k_open_per_s", Resolution.PER_BP, "per second", "base-pair opening rate",
       GUERON, provisional=True, reason=_RATE_PROV),
    _e("E_bp_lifetime_ms", Resolution.PER_BP, "milliseconds",
       "mean closed lifetime of this base pair", GUERON,
       provisional=True, reason=_RATE_PROV),
    _e("E_hbond_count", Resolution.PER_BP, "count",
       "Watson-Crick hydrogen bonds, 2 for A-T and 3 for G-C"),
    _e("E_hbond_total", Resolution.GLOBAL, "count", "total Watson-Crick hydrogen bonds"),
    _e("E_hbond_distance_A", Resolution.PER_BP, "angstrom",
       "mean Watson-Crick donor-acceptor distance, from the minimised structure",
       "cgNA+ geometry relaxed by restrained minimisation"),
    _e("E_hbond_distance_spread_A", Resolution.PER_BP, "angstrom",
       "spread of the donor-acceptor distances within this base pair",
       "cgNA+ geometry relaxed by restrained minimisation"),
    _e("E_mean_hbond_distance_A", Resolution.GLOBAL, "angstrom",
       "mean Watson-Crick donor-acceptor distance",
       "cgNA+ geometry relaxed by restrained minimisation"),
    _e("E_dCp_kcal_mol_K", Resolution.PER_BP, "kcal/(mol K)",
       "nearest-neighbour heat capacity change of duplex formation for this step, same "
       "sign convention as E_dH_kcal_mol", RISSONE),
    _e("E_dCp_total_kcal_mol_K", Resolution.GLOBAL, "kcal/(mol K)",
       "summed nearest-neighbour heat capacity change of duplex formation", RISSONE),
    _e("E_hairpin_stem_bp", Resolution.GLOBAL, "bp",
       "stem length of the most stable intramolecular hairpin"),
    _e("E_vant_hoff_dH_kcal_mol", Resolution.GLOBAL, "kcal/mol",
       "van't Hoff enthalpy from the temperature dependence of mean opening",
       _SIGMA_SRC),
    _e("E_melting_cooperativity", Resolution.GLOBAL, "ratio",
       "van't Hoff over calorimetric enthalpy; below one means the transition is not "
       "two-state", _SIGMA_SRC),
]

for _na in IONIC_STRENGTHS_M:
    _SPECS_E.append(_e(f"E_stack_dG_na{_na:g}", Resolution.PER_BP, "kcal/mol",
                       f"step stacking free energy at {_na:g} M sodium",
                       OWCZARZY, condition=True))
    _SPECS_E.append(_e(f"E_Tm_na{_na:g}_C", Resolution.GLOBAL, "celsius",
                       f"melting temperature at {_na:g} M sodium", OWCZARZY,
                       condition=True))
for _T in TEMPERATURES_C:
    _SPECS_E.append(_e(f"E_open_prob_T{_T:g}", Resolution.PER_BP, "probability",
                       f"base-pair opening probability at {_T:g} C. " + _OPEN_NOTE,
                       _SIGMA_SRC, condition=True))
    _SPECS_E.append(_e(f"E_mean_open_prob_T{_T:g}", Resolution.GLOBAL, "probability",
                       f"mean opening probability at {_T:g} C", _SIGMA_SRC,
                       condition=True))
for _sc in SUPERHELICAL_DENSITIES:
    _SPECS_E.append(_e(f"E_open_prob_sc{_sc:g}", Resolution.PER_BP, "probability",
                       f"opening probability at superhelical density {_sc:g}",
                       BENHAM, condition=True))
    _SPECS_E.append(_e(f"E_sidd_dG_sc{_sc:g}", Resolution.PER_BP, "kcal/mol",
                       f"stress-induced destabilization energy at sigma {_sc:g}",
                       BENHAM, condition=True))
    _SPECS_E.append(_e(f"E_sidd_min_dG_sc{_sc:g}", Resolution.GLOBAL, "kcal/mol",
                       f"most destabilized base pair at sigma {_sc:g}", BENHAM,
                       condition=True))
    for _q, _u, _d in (("open_count_mean", "bp", "mean number of open base pairs"),
                       ("open_count_sd", "bp", "spread of the open base-pair count"),
                       ("open_count_skew", "dimensionless",
                        "skew of the open base-pair count"),
                       ("mean_bubble_bp", "bp", "mean length of an open run"),
                       ("mean_runs", "count", "mean number of open runs")):
        _SPECS_E.append(_e(f"E_{_q}_sc{_sc:g}", Resolution.GLOBAL, _u,
                           f"{_d} at superhelical density {_sc:g}", BENHAM,
                           condition=True))
for _w in SEPARATION_WINDOWS_BP:
    _SPECS_E.append(_e(f"E_strand_separation_dG_w{_w}", Resolution.PER_BP, "kcal/mol",
                       f"free energy of separating a {_w} bp window"))
for _na in (0.01, 0.15, 1.00):
    _SPECS_E.append(_e(f"E_hairpin_dG_na{_na:g}", Resolution.GLOBAL, "kcal/mol",
                       f"minimum free energy fold of the more structured strand at "
                       f"{_na:g} M sodium, DNA parameters", MATHEWS_DNA, condition=True))
    _SPECS_E.append(_e(f"E_cruciform_dG_na{_na:g}", Resolution.GLOBAL, "kcal/mol",
                       f"cruciform extrusion at {_na:g} M sodium and sigma -0.06",
                       BENHAM, provisional=True, reason=_LOOP_NOTE, condition=True))
for _ph in PH_VALUES:
    # STRANDED, and measured so. An i-motif is built from the cytosines of one strand,
    # and a triplex from one strand's purine tract, so reverse complement does not leave
    # either unchanged -- it exchanges them for the partner strand's value. Family F
    # declares the same structures STRANDED, and the two families must not disagree
    # about the same molecule.
    _SPECS_E.append(_e(f"E_triplex_dG_pH{_ph:g}", Resolution.GLOBAL, "kcal/mol",
                       f"intramolecular triplex free energy at pH {_ph:g}, on this "
                       "strand's purine tract",
                       "Family F propensity", provisional=True, reason=_PH_PROV,
                       condition=True, parity=Parity.STRANDED))
    _SPECS_E.append(_e(f"E_imotif_dG_pH{_ph:g}", Resolution.GLOBAL, "kcal/mol",
                       f"i-motif free energy at pH {_ph:g}, on this strand's cytosines",
                       "Family F propensity", provisional=True, reason=_PH_PROV,
                       condition=True, parity=Parity.STRANDED))

REGISTRY.extend(_SPECS_E)


# --- measured index maps ------------------------------------------------------------
# The nearest-neighbour terms belong to a dinucleotide step, so they reflect as
# i -> n-2-i; measured exact on random palindromes. The window sums over steps are placed
# at the window centre and reflect with no shift for the even window sizes used here.
from channels import declare_index_maps as _dim  # noqa: E402
_dim(_SPECS_E, step=["E_dH_kcal_mol", "E_dS_cal_mol_K", "E_dG37_kcal_mol",
                     "E_dCp_kcal_mol_K"]
     + [f"E_stack_dG_na{_na:g}" for _na in IONIC_STRENGTHS_M])
