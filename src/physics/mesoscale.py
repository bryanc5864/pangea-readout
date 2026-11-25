#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family D: mesoscale mechanics.

`PLAN.md` lists 90 channels: cyclizability at 4 loop lengths; isotropic and anisotropic
bending stiffness with preferred bend direction over 6 window sizes; intrinsic curvature
magnitude and direction; torsional and stretch moduli with twist-stretch coupling;
twist-writhe partition at 5 superhelical densities; nucleosome formation free energy,
rotational phasing and translational positioning at 3 ionic strengths; loop closure free
energy at 5 lengths; persistence length across 5 ionic strengths; kink propensity;
response to applied tension at 4 forces; bubble nucleation under torsional stress.

Ground truth is "loop-seq, in vitro nucleosome occupancy, coarse-grained mechanics". The
experimental halves supervise at the delta-learning stage; what is computed here is the
coarse-grained mechanics, which comes from the cgNA+ stiffness matrix. This family is
therefore the one place in the analytic set where the *stiffness* rather than the ground
state is doing the work, and it sits on the plan's window ladder by construction, since
almost nothing here is defined at a point.

**Coordinate convention, established against the library rather than assumed.** With
the flags `analytic_targets.cgna_ground_state` uses -- which are the constructor
defaults -- `CGNAPlusParams` returns rotational coordinates as **rotation vectors**,
Omega = theta * n_hat in radians, and translations in nanometres expressed in the
preceding base-pair frame. The default `euler_definition=True` is what makes this a
rotation vector rather than a Cayley vector; `remove_factor_five=True` removes the
shipped factor of 10 on the way, and `group_split=True` is what puts the translation in
the preceding frame rather than the mid-step frame.

This was originally recorded here as the Cayley convention with a mid-step translation,
and both halves were wrong. The error is quiet because it is third order: a poly-A tract
returns a twist coordinate of 0.6028, which is 34.54 degrees read as a rotation vector
and 33.56 degrees read as a Cayley vector, and the rise coordinate of 0.334 nm is
unaffected either way. Per step it is a few percent; composed along a chain it reached
32 degrees of base-pair frame error over 33 bp and 2 angstrom of origin error, which
corrupts every direction-valued channel -- bend direction, nucleosome rotational
phasing, the curvature vector -- while leaving the scalar magnitudes looking plausible.
`tests/test_mesoscale.py` now pins `helical_path` against `CGNAPlusConf`, the library's
own chain builder, so the reconstruction is checked against the authority instead of
against a remembered convention.

**The stiffness caveat the cgNA+ authors state**, repeated here because it bites exactly
this module: the returned stiffness is the Hessian with respect to Lie-group tangent
fluctuations, not with respect to additive differences of the stored coordinates.
Fluctuation covariances taken as the inverse of that Hessian are therefore tangent-space
covariances, in radians for the rotational block, which is what the mechanics below
needs. Ground-state *differences* are a different object and are never fed into the
quadratic form here.

The decisive check that the whole chain is right is that the computed B-DNA bending
persistence length lands near the measured 50 nm. That is asserted in `tests/`, and it
is what makes the rest of the family trustworthy rather than merely plausible.

References for the closed-form mechanics:
- Marko JF, Siggia ED. Stretching DNA. Macromolecules 28:8759, 1995. (force-extension)
- Odijk T, J Polym Sci 15:477, 1977; Skolnick J, Fixman M, Macromolecules 10:944, 1977.
  (electrostatic persistence length)
- Shimada J, Yamakawa H. Ring-closure probabilities for twisted wormlike chains.
  Macromolecules 17:689, 1984. (cyclization)
- Luger K et al. Crystal structure of the nucleosome core particle at 2.8 A resolution.
  Nature 389:251, 1997. (superhelix geometry)
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
)

FAMILY = "D"

# --- physical constants and measured geometry -----------------------------------------
KT_PN_NM = 4.114                 # kT at 25 C, in pN nm
BJERRUM_NM = 0.714               # Bjerrum length in water at 25 C, nm
PHOSPHATE_SPACING_NM = 0.17      # axial charge spacing on B-DNA, nm (one per 1.7 A)
MANNING_XI = 4.2                 # Manning parameter for B-DNA

# Nucleosome core particle superhelix (Luger et al. 1997).
NUCLEOSOME_RADIUS_NM = 4.19
NUCLEOSOME_PITCH_NM = 2.59
NUCLEOSOME_BP = 147
NUCLEOSOME_TURNS = 1.65

# Loop lengths and forces the plan asks for.
CYCLIZATION_LENGTHS_BP: Tuple[int, ...] = (63, 73, 84, 94)
LOOP_CLOSURE_LENGTHS_BP: Tuple[int, ...] = (50, 75, 100, 150, 200)
TENSION_FORCES_PN: Tuple[float, ...] = (0.1, 0.5, 2.0, 10.0)
SUPERHELICAL_DENSITIES: Tuple[float, ...] = (-0.06, -0.045, -0.03, -0.015, 0.0)
IONIC_STRENGTHS_M: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.25)
NUCLEOSOME_IONIC_M: Tuple[float, ...] = (0.05, 0.10, 0.15)

MARKO_SIGGIA = "Marko & Siggia, Macromolecules 28:8759, 1995"
OSF = "Odijk 1977; Skolnick & Fixman 1977"
SY = "Shimada & Yamakawa, Macromolecules 17:689, 1984"
LUGER = "Luger et al., Nature 389:251, 1997"
CGNA = "cgNA+ (Sharma, Patelli, De Bruin & Maddocks)"


# ======================================================================================
# Extracting the step mechanics from cgNA+
# ======================================================================================

def step_blocks(gs_result: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Ground-state step coordinates and the step-block covariance.

    cgNA+ interleaves intra-base-pair (B), phosphate (W, C) and inter-base-pair step (S)
    entries. Only the S entries are base-pair steps, and only those carry the bending,
    twisting and stretching mechanics this family is about. The covariance is taken as
    the inverse of the full stiffness and then restricted to the step rows, so the
    coupling to the intra-base-pair and phosphate degrees of freedom is integrated out
    rather than ignored -- marginalising is the physically correct reduction, and simply
    inverting the step sub-block instead would give the conditional covariance, which is
    a different and stiffer quantity.
    """
    names = list(gs_result["param_names"])
    gs = np.asarray(gs_result["gs"], dtype=float)
    K = np.asarray(gs_result["stiffness"], dtype=float)

    step_idx = [i for i, n in enumerate(names) if n.startswith("S")]
    if not step_idx:
        raise ValueError("no inter-base-pair step entries in the cgNA+ parameter set")

    # Each entry is a 6-vector, laid out contiguously in the stiffness.
    rows: List[int] = []
    for i in step_idx:
        rows.extend(range(6 * i, 6 * i + 6))
    rows_arr = np.asarray(rows, dtype=int)

    # Marginal covariance of the steps: invert the whole Hessian, then restrict.
    cov_full = np.linalg.inv(K)
    cov = cov_full[np.ix_(rows_arr, rows_arr)]

    return {
        "step_gs": gs[step_idx],              # (n_steps, 6) rotation vector (rad) | nm
        "step_cov": cov,                      # (6 n_steps, 6 n_steps) tangent space
        "n_steps": np.int64(len(step_idx)),
    }


def step_blocks_banded(gs_result: Dict[str, object]) -> Dict[str, np.ndarray]:
    """`step_blocks` for long sequences: the same marginal step covariance, but only its 6x6
    diagonal blocks (all `per_step_fluctuations` reads), by selected inversion of the banded
    stiffness (`banded_inverse`; equal to the dense route to ~5e-14 relative, tested). Takes a
    cgNA+ result whose "stiffness" may be sparse."""
    import banded_inverse as BI
    names = list(gs_result["param_names"])
    gs = np.asarray(gs_result["gs"], dtype=float)
    step_idx = [i for i, n in enumerate(names) if n.startswith("S")]
    if not step_idx:
        raise ValueError("no inter-base-pair step entries in the cgNA+ parameter set")
    blocks = BI.diagonal_blocks_of_inverse(gs_result["stiffness"], [6 * i for i in step_idx], 6)
    return {"step_gs": gs[step_idx],
            "step_cov_diag": np.stack([blocks[6 * i] for i in step_idx]),
            "n_steps": np.int64(len(step_idx))}


def per_step_fluctuations(blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Variances and couplings per step, in tangent (radian / nm) coordinates."""
    if "step_cov_diag" in blocks:                 # long sequences: diagonal blocks only
        D = blocks["step_cov_diag"]
        return {"tilt_var": D[:, 0, 0].copy(), "roll_var": D[:, 1, 1].copy(), "twist_var": D[:, 2, 2].copy(),
                "rise_var": D[:, 5, 5].copy(), "twist_rise_cov": D[:, 2, 5].copy(),
                "tilt_roll_cov": D[:, 0, 1].copy(), "bend_var": D[:, 0, 0] + D[:, 1, 1]}
    cov = blocks["step_cov"]
    n = int(blocks["n_steps"])
    var = np.diag(cov).reshape(n, 6)
    tilt_v, roll_v, twist_v = var[:, 0], var[:, 1], var[:, 2]
    rise_v = var[:, 5]
    twist_rise_cov = np.array([cov[6 * i + 2, 6 * i + 5] for i in range(n)])
    tilt_roll_cov = np.array([cov[6 * i + 0, 6 * i + 1] for i in range(n)])
    return {
        "tilt_var": tilt_v, "roll_var": roll_v, "twist_var": twist_v,
        "rise_var": rise_v, "twist_rise_cov": twist_rise_cov,
        "tilt_roll_cov": tilt_roll_cov,
        "bend_var": tilt_v + roll_v,
    }


def elastic_moduli(blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Persistence length, torsional modulus, stretch modulus, twist-stretch coupling.

    For a discrete elastic chain with rise `a` and isotropic bending fluctuation
    <theta^2> per step, the tangent-tangent correlation decays as exp(-a/l_p) with
    <theta^2> = 2 a / l_p in three dimensions, so l_p = 2a / <theta^2>. The torsional
    modulus follows from the twist fluctuation as C = a / <phi^2> and the stretch
    modulus from the rise fluctuation as S = a / <da^2>, both in units of kT.
    """
    fl = per_step_fluctuations(blocks)
    gs = blocks["step_gs"]
    a_nm = float(np.mean(gs[:, 5]))                   # rise, nm

    bend_var = float(np.mean(fl["bend_var"]))
    twist_var = float(np.mean(fl["twist_var"]))
    rise_var = float(np.mean(fl["rise_var"]))
    tr_cov = float(np.mean(fl["twist_rise_cov"]))

    l_p_nm = 2.0 * a_nm / bend_var if bend_var > 0 else float("nan")
    C_nm = a_nm / twist_var if twist_var > 0 else float("nan")
    S_pn = KT_PN_NM * a_nm / rise_var if rise_var > 0 else float("nan")
    # Twist-stretch coupling, dimensionless correlation of the two.
    denom = math.sqrt(twist_var * rise_var) if twist_var > 0 and rise_var > 0 else 0.0
    g = tr_cov / denom if denom > 0 else float("nan")

    return {
        "rise_nm": a_nm,
        "persistence_length_nm": l_p_nm,
        "torsional_modulus_nm": C_nm,
        "stretch_modulus_pN": S_pn,
        "twist_stretch_coupling": g,
        "mean_bend_variance_rad2": bend_var,
        "mean_twist_variance_rad2": twist_var,
    }


# ======================================================================================
# Intrinsic curvature from the ground state
# ======================================================================================

def _rotvec_to_rotmat(v: np.ndarray) -> np.ndarray:
    """Rotation vector Omega = theta * n_hat (radians) -> rotation matrix, by Rodrigues.

    **This is the convention `CGNAPlusParams` actually returns**, and it was wrong here
    until it was checked against the library rather than assumed. The constructor flag
    `euler_definition` defaults to *True*, and it converts the shipped scaled Cayley
    vector eta = 10 tan(theta/2) n_hat into the rotation vector Omega = theta n_hat;
    `remove_factor_five`, also on by default, is what takes eta to the geometric Cayley
    vector c = 2 tan(theta/2) n_hat on the way. Only with `euler_definition=False` is the
    stored coordinate a Cayley vector.

    The two differ by theta^3/12 to leading order, which is about 3 percent at a B-DNA
    twist of 34 degrees -- small enough per step to leave the rise and the persistence
    length looking right, and large enough to accumulate into tens of degrees of frame
    error over a few tens of base pairs. `tests/test_mesoscale.py` now pins the
    reconstruction against `CGNAPlusConf`, which is the library's own authoritative
    chain builder, so this cannot drift again.
    """
    theta = float(np.linalg.norm(v))
    hat = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if theta < 1e-12:
        return np.eye(3) + hat
    return (np.eye(3) + (math.sin(theta) / theta) * hat
            + ((1.0 - math.cos(theta)) / (theta * theta)) * (hat @ hat))


def helical_path(blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Trace the ground-state duplex axis, step by step, in its own frame.

    The junction convention is the library's, read out of
    `cgnaplusparams/utils/se3_methods.py` rather than guessed: each step contributes the
    SE(3) element `[[exp(Omega^), w], [0, 1]]` and poses compose on the right, so

        R_{i+1} = R_i exp(Omega_i^),   r_{i+1} = r_i + R_i w_i.

    The translation is therefore expressed in the **preceding base-pair frame**, not the
    mid-step frame, because `group_split` is on by default. Using the mid-step form here
    was the second half of the same error as the Cayley/rotation-vector confusion above.
    """
    gs = blocks["step_gs"]
    n = gs.shape[0]
    frames = np.zeros((n + 1, 3, 3))
    origins = np.zeros((n + 1, 3))
    frames[0] = np.eye(3)
    for i in range(n):
        origins[i + 1] = origins[i] + frames[i] @ gs[i, 3:]
        frames[i + 1] = frames[i] @ _rotvec_to_rotmat(gs[i, :3])
    return {"origins": origins, "frames": frames}


def intrinsic_curvature(blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Magnitude and direction of the ground-state bend.

    Measured as the shortening of the end-to-end distance relative to the contour
    length, which is what macroscopic curvature does to a duplex, plus the net bend
    angle between the first and last base-pair normal.
    """
    path = helical_path(blocks)
    o = path["origins"]
    contour = float(np.sum(np.linalg.norm(np.diff(o, axis=0), axis=1)))
    end_to_end = float(np.linalg.norm(o[-1] - o[0]))
    shortening = 1.0 - end_to_end / contour if contour > 0 else float("nan")

    z0 = path["frames"][0][:, 2]
    z1 = path["frames"][-1][:, 2]
    cosang = float(np.clip(z0 @ z1, -1.0, 1.0))
    net_bend = math.degrees(math.acos(cosang))

    # Direction of the bend in the first base pair's frame: which way the helix turns.
    axis = np.cross(z0, z1)
    nrm = float(np.linalg.norm(axis))
    if nrm > 1e-12:
        axis = axis / nrm
        bend_dir = math.degrees(math.atan2(float(axis @ path["frames"][0][:, 1]),
                                           float(axis @ path["frames"][0][:, 0])))
    else:
        bend_dir = float("nan")

    return {
        "contour_length_nm": contour,
        "end_to_end_nm": end_to_end,
        "curvature_shortening": shortening,
        "net_bend_deg": net_bend,
        "bend_direction_deg": bend_dir,
    }


# ======================================================================================
# Window-resolved bending stiffness
# ======================================================================================

def windowed_bending(blocks: Dict[str, np.ndarray],
                     ladder: Sequence[int] = WINDOW_LADDER_BP
                     ) -> Dict[str, np.ndarray]:
    """Isotropic and anisotropic bending stiffness on the plan's window ladder.

    Bending stiffness is the plan's own example of a quantity defined over an extent.
    At each rung the tilt and roll variances are summed across the window, giving the
    window's total bending compliance; the anisotropy is the ratio of the two principal
    compliances, and the preferred direction is the angle of the softer one.
    """
    fl = per_step_fluctuations(blocks)
    n = int(blocks["n_steps"])
    a_nm = float(np.mean(blocks["step_gs"][:, 5]))
    out: Dict[str, np.ndarray] = {}

    tilt, roll, cross = fl["tilt_var"], fl["roll_var"], fl["tilt_roll_cov"]
    for w_bp in ladder:
        w = min(w_bp, n)
        if w < 2:
            continue
        n_win = n - w + 1
        lp = np.empty(n_win)
        aniso = np.empty(n_win)
        direction = np.empty(n_win)
        for i in range(n_win):
            t = float(tilt[i:i + w].sum())
            r = float(roll[i:i + w].sum())
            c = float(cross[i:i + w].sum())
            # Principal compliances of the 2x2 bending covariance.
            tr, det = t + r, t * r - c * c
            disc = max(0.0, 0.25 * tr * tr - det)
            lam1 = 0.5 * tr + math.sqrt(disc)
            lam2 = max(1e-12, 0.5 * tr - math.sqrt(disc))
            lp[i] = 2.0 * a_nm * w / tr if tr > 0 else np.nan
            aniso[i] = lam1 / lam2
            direction[i] = math.degrees(0.5 * math.atan2(2.0 * c, t - r))
        out[f"D_bend_lp_w{w_bp}"] = lp
        out[f"D_bend_anisotropy_w{w_bp}"] = aniso
        out[f"D_bend_direction_w{w_bp}"] = direction
    return out


# ======================================================================================
# Closed-form mechanics
# ======================================================================================

def osf_persistence_length(l_p0_nm: float, ionic_strength_M: float) -> float:
    """Total persistence length = intrinsic + electrostatic (Odijk-Skolnick-Fixman).

    l_OSF = l_B / (4 kappa^2 A^2) with A the axial spacing between effective charges.
    Manning condensation reduces the bare phosphate charge, so the effective spacing is
    the bare spacing divided by the fraction of charge that survives condensation.
    """
    kappa_inv_nm = 0.304 / math.sqrt(max(ionic_strength_M, 1e-6))   # Debye length, nm
    kappa = 1.0 / kappa_inv_nm
    eff_spacing = PHOSPHATE_SPACING_NM * MANNING_XI       # effective, after condensation
    l_osf = BJERRUM_NM / (4.0 * kappa**2 * eff_spacing**2)
    return l_p0_nm + l_osf


def marko_siggia_extension(force_pN: float, l_p_nm: float) -> float:
    """Fractional extension of a worm-like chain under tension.

    Inverts the Marko-Siggia interpolation f l_p / kT = x + 1/(4(1-x)^2) - 1/4
    numerically, which is exact for the formula rather than using its small- or
    large-force limit.
    """
    target = force_pN * l_p_nm / KT_PN_NM
    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = mid + 1.0 / (4.0 * (1.0 - mid) ** 2) - 0.25
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def twist_writhe_partition(sigma: float, l_p_nm: float, C_nm: float) -> Dict[str, float]:
    """Partition of excess linking number between twist and writhe.

    For a long molecule the partition minimises the total elastic energy and settles at
    a ratio fixed by the torsional and bending moduli: the softer mode absorbs more. The
    standard long-chain result gives the twist fraction as C/(C + 4 l_p) under the
    approximation that writhe is accommodated by plectonemic bending.
    """
    if not (np.isfinite(l_p_nm) and np.isfinite(C_nm)) or (C_nm + 4.0 * l_p_nm) <= 0:
        return {"twist_fraction": float("nan"), "writhe_fraction": float("nan"),
                "delta_twist": float("nan"), "writhe": float("nan")}
    f_tw = C_nm / (C_nm + 4.0 * l_p_nm)
    return {
        "twist_fraction": float(f_tw),
        "writhe_fraction": float(1.0 - f_tw),
        "delta_twist": float(f_tw * sigma),
        "writhe": float((1.0 - f_tw) * sigma),
    }


def loop_closure_free_energy(n_bp: int, l_p_nm: float, rise_nm: float,
                             net_bend_deg: float = 0.0) -> float:
    """Free energy to bring the two ends of an n_bp segment together, in kT.

    The bending cost of forcing a chain of contour length L into a closed loop of the
    same length is the Shimada-Yamakawa elastic term 2 pi^2 l_p / L, which dominates for
    loops short compared with the persistence length -- exactly the regime loop-seq
    probes. Intrinsic curvature already bends the molecule the right way, so it is
    credited against the required bend.
    """
    L = n_bp * rise_nm
    if L <= 0 or not np.isfinite(l_p_nm):
        return float("nan")
    elastic = 2.0 * math.pi**2 * l_p_nm / L
    # Credit intrinsic curvature: a molecule already bent by `net_bend_deg` needs that
    # much less imposed bend. Capped at the full circle so the credit cannot go negative.
    credit = min(1.0, math.radians(abs(net_bend_deg)) / (2.0 * math.pi))
    return float(elastic * (1.0 - credit) ** 2)


def cyclizability(n_bp: int, l_p_nm: float, rise_nm: float, C_nm: float,
                  net_bend_deg: float = 0.0) -> float:
    """log10 J-factor proxy: ring closure probability including the twist register.

    Beyond the bending term, closure requires the two ends to meet in torsional register,
    which costs energy whenever the helical repeat does not divide the loop length. That
    oscillating twist penalty is what makes measured cyclizability oscillate with a ~10.5
    bp period, so it is the part a sequence model has to reproduce.
    """
    bend = loop_closure_free_energy(n_bp, l_p_nm, rise_nm, net_bend_deg)
    if not np.isfinite(bend):
        return float("nan")
    L = n_bp * rise_nm
    # Twist register: residual turns needed to close an integer linking number.
    turns = n_bp / 10.5
    residual = turns - round(turns)
    twist_pen = 2.0 * math.pi**2 * C_nm * residual**2 / L if L > 0 else 0.0
    # Entropic prefactor of ring closure for a stiff chain.
    log_j = -(bend + twist_pen) / math.log(10.0) - 1.5 * math.log10(max(L, 1e-9))
    return float(log_j)


def nucleosome_free_energy(blocks: Dict[str, np.ndarray], l_p_nm: float,
                           ionic_strength_M: float,
                           n_register: int = 20) -> Dict[str, float]:
    """Elastic cost of wrapping the fragment onto the nucleosome superhelix.

    The histone octamer forces DNA onto a left-handed superhelix of radius 4.19 nm and
    pitch 2.59 nm for about 1.65 turns. The elastic cost is the bending energy of that
    imposed curvature, reduced where the sequence is intrinsically bent the right way and
    where it is locally soft. Rotational phasing is found by sweeping the register: the
    minor groove must face the octamer at the right positions, so rotating the sequence
    against the superhelix changes the cost with the helical period.
    """
    fl = per_step_fluctuations(blocks)
    n = int(blocks["n_steps"])
    a_nm = float(np.mean(blocks["step_gs"][:, 5]))

    # Superhelical curvature: kappa = R / (R^2 + (p/2pi)^2).
    p_red = NUCLEOSOME_PITCH_NM / (2.0 * math.pi)
    kappa = NUCLEOSOME_RADIUS_NM / (NUCLEOSOME_RADIUS_NM**2 + p_red**2)
    bend_per_step = kappa * a_nm                      # radians of bend demanded per step

    # Local bending stiffness per step, in kT per radian^2, from the fluctuations.
    k_tilt = 1.0 / np.maximum(fl["tilt_var"], 1e-12)
    k_roll = 1.0 / np.maximum(fl["roll_var"], 1e-12)

    usable = min(n, NUCLEOSOME_BP - 1)
    if usable < 10:
        return {"nucleosome_dG_kT": float("nan"), "rotational_phase_deg": float("nan"),
                "phasing_amplitude_kT": float("nan"),
                "translational_position_bp": float("nan")}

    energies = np.empty(n_register)
    for r in range(n_register):
        phase = 2.0 * math.pi * r / n_register
        # The superhelix bends into the minor groove on a ~10.5 bp period; which of tilt
        # or roll absorbs the bend rotates with the register.
        ang = phase + 2.0 * math.pi * np.arange(usable) / 10.5
        k_eff = k_roll[:usable] * np.cos(ang) ** 2 + k_tilt[:usable] * np.sin(ang) ** 2
        energies[r] = 0.5 * float(np.sum(k_eff)) * bend_per_step**2

    best = int(np.argmin(energies))
    dG = float(energies[best])

    # Salt screens the DNA-histone electrostatics, so wrapping is easier at higher salt.
    # Scaled by the electrostatic persistence length, which carries the same dependence.
    l_osf = osf_persistence_length(0.0, ionic_strength_M)
    dG_salt = dG * (1.0 + l_osf / max(l_p_nm, 1e-9))

    # Translational positioning: where along the fragment the cheapest 147 bp sits.
    if n > usable:
        k_mean = 0.5 * (k_tilt + k_roll)
        cs = np.concatenate([[0.0], np.cumsum(k_mean)])
        win = cs[usable:] - cs[:-usable]
        trans = float(np.argmin(win))
    else:
        trans = 0.0

    return {
        "nucleosome_dG_kT": dG_salt,
        "rotational_phase_deg": float(360.0 * best / n_register),
        "phasing_amplitude_kT": float(energies.max() - energies.min()),
        "translational_position_bp": trans,
    }


def kink_propensity(blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Where the duplex is softest to a large bend, and how soft.

    A kink nucleates where the harmonic cost of a large bend is lowest, so the softest
    step sets the propensity and the spread across steps says how localised it is.
    """
    fl = per_step_fluctuations(blocks)
    bend_k = 1.0 / np.maximum(fl["bend_var"], 1e-12)
    return {
        "kink_softest_kT_rad2": float(bend_k.min()),
        "kink_position_bp": float(int(np.argmin(bend_k))),
        "kink_heterogeneity": float(bend_k.std() / bend_k.mean())
        if bend_k.mean() > 0 else float("nan"),
    }


# ======================================================================================
# Assembly
# ======================================================================================

def mesoscale_channels(seq: str, gs_result: Optional[Dict[str, object]] = None,
                       symmetrise: bool = False) -> Dict[str, float]:
    """Every Family D scalar channel for one sequence.

    With `symmetrise=True` the channels are averaged with the reverse-complement
    computation according to their declared parity. The cgNA+ stiffness is not exactly
    reverse-complement symmetric (see `channels.symmetrise_channels`), the physical
    duplex is, and PANGEA enforces the symmetry architecturally, so production targets
    are generated symmetrised. It costs one extra cgNA+ evaluation, about 15 ms.
    """
    if symmetrise:
        from channels import reverse_complement, symmetrise_channels
        fwd = mesoscale_channels(seq, gs_result, symmetrise=False)
        rev = mesoscale_channels(reverse_complement(seq), None, symmetrise=False)
        return symmetrise_channels(fwd, rev)
    if gs_result is None:
        import analytic_targets as at
        gs_result = at.cgna_ground_state(seq, include_stiffness=True)

    blocks = step_blocks(gs_result)
    mod = elastic_moduli(blocks)
    curv = intrinsic_curvature(blocks)
    kink = kink_propensity(blocks)

    out: Dict[str, float] = {}
    out["D_persistence_length_nm"] = mod["persistence_length_nm"]
    out["D_torsional_modulus_nm"] = mod["torsional_modulus_nm"]
    out["D_stretch_modulus_pN"] = mod["stretch_modulus_pN"]
    out["D_twist_stretch_coupling"] = mod["twist_stretch_coupling"]
    out["D_rise_nm"] = mod["rise_nm"]
    out["D_mean_bend_variance_rad2"] = mod["mean_bend_variance_rad2"]
    out["D_mean_twist_variance_rad2"] = mod["mean_twist_variance_rad2"]

    out["D_contour_length_nm"] = curv["contour_length_nm"]
    out["D_end_to_end_nm"] = curv["end_to_end_nm"]
    out["D_curvature_shortening"] = curv["curvature_shortening"]
    out["D_net_bend_deg"] = curv["net_bend_deg"]
    out["D_bend_direction_deg"] = curv["bend_direction_deg"]

    out.update({f"D_{k}": v for k, v in kink.items()})

    lp0 = mod["persistence_length_nm"]
    rise = mod["rise_nm"]
    C = mod["torsional_modulus_nm"]

    # Persistence length across ionic strength.
    for I in IONIC_STRENGTHS_M:
        out[f"D_persistence_length_I{I:g}"] = osf_persistence_length(lp0, I)

    # Cyclizability and loop closure.
    for n_bp in CYCLIZATION_LENGTHS_BP:
        out[f"D_cyclizability_L{n_bp}"] = cyclizability(
            n_bp, lp0, rise, C, curv["net_bend_deg"])
    for n_bp in LOOP_CLOSURE_LENGTHS_BP:
        out[f"D_loop_closure_L{n_bp}_kT"] = loop_closure_free_energy(
            n_bp, lp0, rise, curv["net_bend_deg"])

    # Tension response.
    for f in TENSION_FORCES_PN:
        out[f"D_extension_f{f:g}pN"] = marko_siggia_extension(f, lp0)

    # Twist-writhe partition.
    for s in SUPERHELICAL_DENSITIES:
        p = twist_writhe_partition(s, lp0, C)
        out[f"D_delta_twist_sigma{s:g}"] = p["delta_twist"]
        out[f"D_writhe_sigma{s:g}"] = p["writhe"]
    out["D_twist_fraction"] = twist_writhe_partition(
        -0.06, lp0, C)["twist_fraction"]

    # Nucleosome.
    for I in NUCLEOSOME_IONIC_M:
        nuc = nucleosome_free_energy(blocks, lp0, I)
        out[f"D_nucleosome_dG_I{I:g}_kT"] = nuc["nucleosome_dG_kT"]
        if I == NUCLEOSOME_IONIC_M[-1]:
            out["D_nucleosome_rotational_phase_deg"] = nuc["rotational_phase_deg"]
            out["D_nucleosome_phasing_amplitude_kT"] = nuc["phasing_amplitude_kT"]
            out["D_nucleosome_translational_bp"] = nuc["translational_position_bp"]

    # Bubble nucleation under torsional stress: the softest twist step is where a bubble
    # opens first, and negative supercoiling drives it.
    fl = per_step_fluctuations(blocks)
    out["D_bubble_softest_twist_rad2"] = float(fl["twist_var"].max())
    out["D_bubble_position_bp"] = float(int(np.argmax(fl["twist_var"])))

    return out


def mesoscale_window_channels(seq: str, gs_result: Optional[Dict[str, object]] = None,
                              ladder: Sequence[int] = WINDOW_LADDER_BP
                              ) -> Dict[str, np.ndarray]:
    """Family D on the revision-2 window ladder."""
    if gs_result is None:
        import analytic_targets as at
        gs_result = at.cgna_ground_state(seq, include_stiffness=True)
    return windowed_bending(step_blocks(gs_result), ladder)


# ======================================================================================
# Registry
# ======================================================================================

def _d(name: str, parity: Parity, units: str, desc: str, source: str = CGNA,
       provisional: bool = False, reason: str = "",
       condition: bool = False) -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=Resolution.GLOBAL,
                       parity=parity, units=units, description=desc,
                       condition_dependent=condition, provisional=provisional,
                       provisional_reason=reason, source=source)


_NUC_PROV = ("the wrapping energy is the elastic term only; the histone-DNA contact "
             "energetics and the salt scaling are not parameterised from measurement")

_SPECS: List[ChannelSpec] = [
    _d("D_persistence_length_nm", Parity.SYMMETRIC, "nm",
       "intrinsic bending persistence length from the cgNA+ stiffness"),
    _d("D_torsional_modulus_nm", Parity.SYMMETRIC, "nm", "torsional modulus C in kT nm"),
    _d("D_stretch_modulus_pN", Parity.SYMMETRIC, "pN", "stretch modulus"),
    _d("D_twist_stretch_coupling", Parity.SYMMETRIC, "dimensionless",
       "correlation between twist and rise fluctuations"),
    _d("D_rise_nm", Parity.SYMMETRIC, "nm", "mean rise per base-pair step"),
    _d("D_mean_bend_variance_rad2", Parity.SYMMETRIC, "rad^2",
       "mean tilt plus roll variance per step"),
    _d("D_mean_twist_variance_rad2", Parity.SYMMETRIC, "rad^2",
       "mean twist variance per step"),
    _d("D_contour_length_nm", Parity.SYMMETRIC, "nm", "contour length of the duplex axis"),
    _d("D_end_to_end_nm", Parity.SYMMETRIC, "nm", "ground-state end-to-end distance"),
    _d("D_curvature_shortening", Parity.SYMMETRIC, "fraction",
       "fractional shortening caused by intrinsic curvature"),
    _d("D_net_bend_deg", Parity.SYMMETRIC, "degrees",
       "angle between the first and last base-pair normal"),
    _d("D_bend_direction_deg", Parity.ANTISYMMETRIC, "degrees",
       "azimuth of the bend in the first base pair's frame"),
    _d("D_kink_softest_kT_rad2", Parity.SYMMETRIC, "kT/rad^2",
       "bending stiffness of the softest step"),
    _d("D_kink_position_bp", Parity.ANTISYMMETRIC, "bp", "where that step sits"),
    _d("D_kink_heterogeneity", Parity.SYMMETRIC, "dimensionless",
       "coefficient of variation of per-step bending stiffness"),
    _d("D_twist_fraction", Parity.SYMMETRIC, "fraction",
       "share of excess linking number taken up as twist"),
    _d("D_bubble_softest_twist_rad2", Parity.SYMMETRIC, "rad^2",
       "largest per-step twist variance, where a bubble nucleates first"),
    _d("D_bubble_position_bp", Parity.ANTISYMMETRIC, "bp", "where that step sits"),
    _d("D_nucleosome_rotational_phase_deg", Parity.SYMMETRIC, "degrees",
       "register minimising the wrapping energy", source=LUGER,
       provisional=True, reason=_NUC_PROV),
    _d("D_nucleosome_phasing_amplitude_kT", Parity.SYMMETRIC, "kT",
       "spread of wrapping energy across register", source=LUGER,
       provisional=True, reason=_NUC_PROV),
    _d("D_nucleosome_translational_bp", Parity.ANTISYMMETRIC, "bp",
       "start of the cheapest 147 bp to wrap", source=LUGER,
       provisional=True, reason=_NUC_PROV),
]
_SPECS += [_d(f"D_persistence_length_I{I:g}", Parity.SYMMETRIC, "nm",
              f"total persistence length at {I:g} M monovalent salt",
              source=OSF, condition=True) for I in IONIC_STRENGTHS_M]
_SPECS += [_d(f"D_cyclizability_L{n}", Parity.SYMMETRIC, "log10 J",
              f"ring closure propensity for a {n} bp loop", source=SY,
              provisional=True,
              reason="a Shimada-Yamakawa elastic estimate with a twist-register term, "
                     "not the full ring-closure integral; loop-seq supervises the "
                     "residual at the delta-learning stage")
           for n in CYCLIZATION_LENGTHS_BP]
_SPECS += [_d(f"D_loop_closure_L{n}_kT", Parity.SYMMETRIC, "kT",
              f"bending free energy to close a {n} bp loop", source=SY)
           for n in LOOP_CLOSURE_LENGTHS_BP]
_SPECS += [_d(f"D_extension_f{f:g}pN", Parity.SYMMETRIC, "fraction",
              f"fractional extension under {f:g} pN", source=MARKO_SIGGIA,
              condition=True) for f in TENSION_FORCES_PN]
_SPECS += [_d(f"D_delta_twist_sigma{s:g}", Parity.SYMMETRIC, "dimensionless",
              f"twist share of excess linking number at sigma {s:g}", condition=True)
           for s in SUPERHELICAL_DENSITIES]
_SPECS += [_d(f"D_writhe_sigma{s:g}", Parity.SYMMETRIC, "dimensionless",
              f"writhe share at sigma {s:g}", condition=True)
           for s in SUPERHELICAL_DENSITIES]
_SPECS += [_d(f"D_nucleosome_dG_I{I:g}_kT", Parity.SYMMETRIC, "kT",
              f"nucleosome wrapping free energy at {I:g} M salt", source=LUGER,
              condition=True, provisional=True, reason=_NUC_PROV)
           for I in NUCLEOSOME_IONIC_M]

REGISTRY.extend(_SPECS)


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    demo = sys.argv[1] if len(sys.argv) > 1 else "ACGTACGTAGCTAGCTAGCTTACGATCGATCGGCTAGCTAGCATCGATCGATCGATTACGA"
    ch = mesoscale_channels(demo)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else None)
                      for k, v in ch.items()}, indent=2))
    print(f"\nFamily D channels declared: {len(REGISTRY.by_family('D'))}")
    print(f"computed: {len(ch)}")
    print(f"persistence length: {ch['D_persistence_length_nm']:.1f} nm "
          f"(measured B-DNA is about 50 nm)")


# --- measured index maps ------------------------------------------------------------
from channels import declare_index_maps as _dim  # noqa: E402
_dim(_SPECS, circular=[n for n in ("D_bend_direction_deg",
                                    "D_nucleosome_rotational_phase_deg")
                       if n in {s.name for s in _SPECS}])
