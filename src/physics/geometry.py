#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family C: geometry and local stiffness.

`PLAN.md` lists 215 channels: "12 rigid-base coordinate means; the full 12 by 12
stiffness matrix upper triangle per step; groove widths, depths and their fluctuations;
backbone torsions, sugar pucker phase and amplitude per strand with variances; BI and
BII populations; the same geometry re-predicted under high salt, methylation and
elevated temperature; local helical axis curvature and direction; phosphate position
descriptors; sugar pucker state probabilities; stretch and overtwist coupling;
Watson-Crick to Hoogsteen opening propensity; conformational substate occupancies."

Ground truth is "cgNA+ closed form, ABC and hexamer-scale MD libraries, cross-checked
against Deep DNAshape". What is computed here is the cgNA+ closed form. The MD libraries
supervise at the delta-learning stage and are not a substitute for them, so where cgNA+
genuinely cannot reach a quantity this module says so with `provisional=True` rather than
emitting a plausible-looking number.

**What cgNA+ can and cannot reach, decided by reading the model rather than by hope.**

cgNA+ is a Gaussian over rigid-base and rigid-phosphate coordinates. That it carries
*explicit phosphate degrees of freedom* -- the thing cgNA+ added over cgDNA -- is what
makes most of this family reachable at all:

  reachable directly, because the phosphate is an independent rigid body
    groove widths and depths and their fluctuations, and phosphate position
    descriptors. These read the phosphate pose itself, so they carry no reconstruction
    error at all, and the groove widths land on the measured B-DNA values.

  reachable only after a restrained minimisation, which is the finding that shaped this
  module
    the backbone torsions alpha, beta, gamma, delta, epsilon and zeta, and therefore
    epsilon-minus-zeta and BI/BII. Every one of these spans the junction between two
    rigid bodies that cgNA+ does not covalently constrain, and the junction does not
    close: in the raw reconstruction the C3'-O3' bond comes out at 1.641 angstrom
    against 1.43, and O5'-C5' at 1.223 against 1.44, which throws the torsions about
    those axes off by up to 63 degrees. Across a Gaussian ensemble it is far worse,
    C3'-O3' ranging from 0.64 to 3.44 angstrom, because the sugar and the phosphate
    fluctuate independently in the model while being covalently bonded in the molecule.
    `minimize.restrained_minimise` repairs it exactly -- 1.423 and 1.420 angstrom, with
    the restrained atoms moving only 0.34 angstrom RMS, so the sequence dependence
    survives -- and the torsions then land within 1.6 to 13 degrees of canonical B-DNA.
    So the torsions are production channels read off the minimised structure, which the
    electrostatics pipeline already builds for every fragment, and they are free there.
    Their *variances* are not: that would need the ensemble minimised snapshot by
    snapshot, so the variance channels are declared provisional.

  not reachable, because the sugar ring is rigid in the model
    sugar pucker phase and amplitude, and pucker state probabilities. The sugar atoms
    sit at fixed positions in the base frame, so any pucker computed from an
    unminimised cgNA+ reconstruction is the template's pucker repeated, carrying no
    sequence dependence whatever, and after minimisation it is the force field's pucker
    rather than a cgNA+ prediction. These are declared provisional.

  not reachable, because the model is a single Gaussian
    true conformational substate occupancies, and Watson-Crick to Hoogsteen opening,
    which is a roughly 180 degree glycosidic flip into a second basin that a unimodal
    quadratic form does not contain. What is computed instead is a harmonic proxy, and
    it is labelled a harmonic proxy.

  not reachable, because there is one parameter set
    re-prediction under high salt and elevated temperature. The shipped parameter sets
    are fit at a single ionic strength and temperature, so a "high salt geometry" from
    cgNA+ would be the same geometry with a new label on it.

  reachable, and this one is real
    re-prediction under methylation. `Di_hmethyl_methylated-hemi_combine` is a fitted
    parameter set over an extended alphabet -- M/N for 5-methylcytosine paired with
    guanine, H/K for 5-hydroxymethylcytosine -- and it reproduces the unmodified
    parameter set exactly on unmodified sequence, which is checked in `tests/`.

**Fluctuations come from sampling the model, not from a hand-rolled Jacobian.** Because
`group_split` is on, fluctuations are Lie-group tangent vectors and the correct
composition is x = gs o exp(delta), not gs + delta. `CGNAPlusConf.from_params(p,
excess=delta)` performs exactly that composition, so drawing delta ~ N(0, K^-1) and
reconstructing through the library gives fluctuation statistics that are exact within
the Gaussian model and that cannot disagree with the library's own convention.

**Groove offsets were measured, not assumed.** Scanning the phosphate index offset from
-14 to +14 on a mixed 33mer gives two local minima of the Watson-Crick P-P distance, at
offset -4 and offset +4. Those are the minor and major grooves, and they land on the
measured B-DNA values -- 11.6 A P-P, so 5.8 A width, against a measured minor groove
width near 5.7 A; and 18.2 A P-P, so 12.4 A width, against a measured major groove width
near 11.7 A. `tests/` pins both, so an upstream change to the phosphate indexing shows up
as a failure rather than as a quietly shifted channel.

References:
- Sharma, Patelli, De Bruin, Maddocks. cgNA+. (the model and its parameter sets)
- El Hassan & Calladine. Two distinct modes of protein-induced bend. JMB 282:331, 1998.
  (groove width convention: P-P distance less 5.8 A)
- Hartmann, Piazzola, Lavery. BI-BII transitions in B-DNA. NAR 21:561, 1993.
  (epsilon - zeta as the BI/BII discriminator, BI < 0 < BII)
- Altona & Sundaralingam. Conformational analysis of the sugar ring. JACS 94:8205, 1972.
  (pseudorotation phase and amplitude, used only to say what is not computed)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from channels import (  # noqa: E402
    REGISTRY,
    WINDOW_LADDER_BP,
    ChannelSpec,
    Parity,
    Resolution,
)

FAMILY = "C"

CGNA = "cgNA+ (Sharma, Patelli, De Bruin & Maddocks)"
ELHASSAN = "El Hassan & Calladine, JMB 282:331, 1998"
HARTMANN = "Hartmann, Piazzola & Lavery, NAR 21:561, 1993"
ALTONA = "Altona & Sundaralingam, JACS 94:8205, 1972"

# The cgNA+ rigid-base coordinate names, in the order the 6-vectors store them.
INTRA_NAMES: Tuple[str, ...] = ("buckle", "propeller", "opening",
                                "shear", "stretch", "stagger")
INTER_NAMES: Tuple[str, ...] = ("tilt", "roll", "twist", "shift", "slide", "rise")

# Under reverse complement, in the 3DNA convention, the antisymmetric rigid-base
# coordinates are exactly these; the rest are unchanged. Measured, not assumed: see
# tests/test_geometry.py::test_declared_parities_hold_on_an_exact_palindrome.
ANTISYMMETRIC_COORDS = frozenset({"buckle", "shear", "tilt", "shift"})

PARAM_SET_METHYL = "Di_hmethyl_methylated-hemi_combine"

# Phosphate index offsets of the two groove minima, found by scanning (see docstring).
MINOR_OFFSET = -4
MAJOR_OFFSET = +4
PHOSPHATE_VDW_A = 5.8            # twice the phosphate radius, El Hassan & Calladine

N_ENSEMBLE_DEFAULT = 200
# Relaxed snapshots per fragment on the ensemble subset (see minimised_ensemble_stats).
ENSEMBLE_MIN_SNAPSHOTS = 16
AXIS_SMOOTH_BP = 11              # one helical turn, to remove the per-step zigzag

# Groove-floor atoms, by base letter. These are the atoms a protein reads from each
# groove and they define where the groove floor is.
MINOR_FLOOR = {"A": ("N3", "C2"), "G": ("N3", "N2"), "C": ("O2",), "T": ("O2",)}
MAJOR_FLOOR = {"A": ("N7", "N6"), "G": ("N7", "O6"), "C": ("N4",), "T": ("O4", "C7")}

# The modified alphabet of `Di_hmethyl_methylated-hemi_combine`: M is 5-methylcytosine
# and N the guanine paired to it; H is 5-hydroxymethylcytosine and K its partner.
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G",
              "M": "N", "N": "M", "H": "K", "K": "H"}

# Which unmodified base supplies the all-atom template for a modified letter.
#
# This is exact for everything that depends only on where the phosphate sits -- groove
# widths, which are phosphorus-phosphorus distances, and the phosphate position
# descriptors -- because the modification changes the parameter set and so the pose,
# and the pose is what is used. It is *approximate* for major-groove depth, which
# measures to the nearest groove-floor atom and therefore loses the 5-methyl group that
# sits in the major groove; the methylated depth will be overstated by roughly the
# length of that methyl. Minor-groove depth is unaffected, the methyl being on the
# major-groove face.
BASE_ALIAS = {"M": "C", "N": "G", "H": "C", "K": "G"}


def parent_base(letter: str) -> str:
    """The unmodified base whose atom template stands in for a modified letter."""
    return BASE_ALIAS.get(letter, letter)

# Backbone torsions as (frame, atom, residue offset) quadruples, in chain order.
#
# `frame` says which rigid body the atom rides on, and that is the whole reason these
# channels exist: an atom on "base" is fixed in the base frame and carries no sequence
# dependence of its own, while an atom on "phos" rides the independent phosphate rigid
# body that cgNA+ adds. Every torsion below mixes the two, which is what makes all six
# sequence-dependent and computable from a Gaussian over rigid bodies.
#
# The indexing follows `cgna_atomic.build_duplex`: phosphate frame i carries the P that
# links residue i-1 to residue i, so a residue's own P and O5' come from frame i while
# its O3' comes from frame i+1. Getting that one offset wrong silently turns epsilon and
# zeta -- and therefore BI/BII -- into noise, so it is written out rather than inlined.
TORSION_DEFS: Dict[str, Tuple[Tuple[str, str, int], ...]] = {
    "alpha":   (("phos", "O3'", 0), ("phos", "P", 0),
                ("phos", "O5'", 0), ("base", "C5'", 0)),
    "beta":    (("phos", "P", 0), ("phos", "O5'", 0),
                ("base", "C5'", 0), ("base", "C4'", 0)),
    "gamma":   (("phos", "O5'", 0), ("base", "C5'", 0),
                ("base", "C4'", 0), ("base", "C3'", 0)),
    "delta":   (("base", "C5'", 0), ("base", "C4'", 0),
                ("base", "C3'", 0), ("phos", "O3'", 1)),
    "epsilon": (("base", "C4'", 0), ("base", "C3'", 0),
                ("phos", "O3'", 1), ("phos", "P", 1)),
    "zeta":    (("base", "C3'", 0), ("phos", "O3'", 1),
                ("phos", "P", 1), ("phos", "O5'", 1)),
}
TORSION_NAMES: Tuple[str, ...] = tuple(TORSION_DEFS)

BASE_ATOMS_NEEDED: Tuple[str, ...] = ("C1'", "C3'", "C4'", "C5'", "O4'")
PHOS_ATOMS_NEEDED: Tuple[str, ...] = ("P", "O5'", "O3'", "OP1", "OP2")


# ======================================================================================
# The cgNA+ coordinate vector, split into its four kinds of entry
# ======================================================================================

def cgna_blocks(gs_result: Dict[str, object]) -> Dict[str, object]:
    """Split the cgNA+ ground state and stiffness into intra, inter and phosphate parts.

    cgNA+ interleaves the entries as B0 C0 S0 W1 B1 C1 S1 W2 ...: `B` is intra-base-pair,
    `S` is the inter-base-pair step, and `W`/`C` are the Watson and Crick phosphates.
    `param_names` is carried through the whole module so the layout is never guessed.

    The covariance is the inverse of the *full* stiffness, restricted afterwards to
    whichever rows are wanted. That is the marginal covariance, which integrates the
    other degrees of freedom out; inverting a sub-block of the stiffness instead would
    give the conditional covariance, a different and stiffer object.
    """
    names = list(gs_result["param_names"])
    gs = np.asarray(gs_result["gs"], dtype=float)
    out: Dict[str, object] = {"param_names": names, "gs": gs}

    for kind in ("B", "S", "W", "C"):
        idx = [i for i, n in enumerate(names) if n.startswith(kind)]
        out[f"{kind}_idx"] = np.asarray(idx, dtype=int)
        out[f"{kind}_gs"] = gs[idx] if idx else np.zeros((0, 6))

    if "stiffness" in gs_result:
        K = np.asarray(gs_result["stiffness"], dtype=float)
        out["stiffness"] = K
        out["cov"] = np.linalg.inv(K)
    return out


def _rows(idx: Sequence[int]) -> np.ndarray:
    """Entry indices -> the coordinate rows they occupy in the stiffness."""
    r: List[int] = []
    for i in idx:
        r.extend(range(6 * int(i), 6 * int(i) + 6))
    return np.asarray(r, dtype=int)


def to_mid_frame(g: np.ndarray) -> np.ndarray:
    """Re-express the translational half of each coordinate in the mid-step frame.

    **Why this is not cosmetic.** With `group_split` on, cgNA+ gives translations in the
    *preceding* rigid body's frame: the junction is r_{i+1} = r_i + R_i w_ref. Every
    published rigid-base geometry table, and every statement of how these coordinates
    behave under reverse complement, uses the **mid-step** frame instead, in which
    r_{i+1} = r_i + R_i exp(Omega/2) w_mid. The two are related by
    w_mid = exp(Omega/2)^T w_ref.

    Measured on an exact palindrome, where parity has to hold exactly or not at all:

        coordinate   reference frame        mid-step frame
        shift        8.0e-01 (fails)        2.8e-13 (antisymmetric)
        slide        5.3e-01 (fails)        1.6e-13 (symmetric)
        rise         1.8e-02 (fails)        2.9e-14 (symmetric)
        shear        1.5e-01 (fails)        9.1e-14 (antisymmetric)
        stretch      4.4e-01 (fails)        1.2e-13 (symmetric)
        stagger      1.6e-01 (fails)        3.1e-13 (symmetric)

    So reporting the raw cgNA+ translations would have shipped six channels that look
    like shift, slide and rise, are quoted in angstrom like shift, slide and rise, and
    silently violate the reverse-complement symmetry PANGEA enforces architecturally.
    The rotational coordinates are unaffected, which is why this hid: half the vector
    was already exactly right.

    `mesoscale.helical_path` deliberately does **not** do this, and must not: it is
    reconstructing the chain, and the chain composes in the reference frame. Same
    numbers, two conventions, each correct for one job.
    """
    g = np.asarray(g, dtype=float)
    out = np.array(g)
    for i in range(len(g)):
        theta = float(np.linalg.norm(g[i, :3]))
        hat = np.array([[0.0, -g[i, 2], g[i, 1]],
                        [g[i, 2], 0.0, -g[i, 0]],
                        [-g[i, 1], g[i, 0], 0.0]]) * 0.5
        if theta < 1e-12:
            R = np.eye(3) + hat
        else:
            h = 0.5 * theta
            R = (np.eye(3) + (math.sin(h) / h) * hat
                 + ((1.0 - math.cos(h)) / (h * h)) * (hat @ hat))
        out[i, 3:] = R.T @ g[i, 3:]
    return out


def rigid_base_coordinates(blocks: Dict[str, object]) -> Dict[str, np.ndarray]:
    """The 12 rigid-base coordinate profiles: 6 intra-base-pair and 6 inter-base-pair.

    Rotations are returned in degrees and translations in angstrom, in the mid-step
    frame, which is the convention every published DNA geometry table uses rather than
    the radians, nanometres and preceding-frame translations cgNA+ stores. `mesoscale`
    keeps the native convention because it feeds the coordinates into a chain
    reconstruction; this family reports them, so it converts. See `to_mid_frame`.
    """
    out: Dict[str, np.ndarray] = {}
    for kind, labels in (("B", INTRA_NAMES), ("S", INTER_NAMES)):
        g = np.asarray(blocks[f"{kind}_gs"], dtype=float)
        if g.size == 0:
            continue
        m = to_mid_frame(g)
        for j, lab in enumerate(labels):
            out[lab] = np.degrees(m[:, j]) if j < 3 else m[:, j] * 10.0
    return out


def step_stiffness_12x12(blocks: Dict[str, object]) -> np.ndarray:
    """The 12 by 12 marginal stiffness of each base-pair step, as `PLAN.md` asks for.

    The 12 coordinates are the 6 intra-base-pair coordinates of base pair i followed by
    the 6 inter-base-pair coordinates of the step from i to i+1. The marginal covariance
    of that pair of entries is extracted from the full inverse and then inverted, so the
    result is the stiffness a reader of this step alone would infer, with every other
    degree of freedom integrated out.

    Returned in the native tangent-space units: rotations in radians, translations in
    nanometres, so the matrix stays a proper quadratic form in kT. Converting the units
    of a stiffness matrix is not a per-entry rescale of the coordinates and doing it
    carelessly is how a stiffness silently stops being symmetric-positive-definite.
    """
    cov = np.asarray(blocks["cov"], dtype=float)
    B, S = np.asarray(blocks["B_idx"]), np.asarray(blocks["S_idx"])
    n = len(S)
    out = np.zeros((n, 12, 12))
    for i in range(n):
        rows = np.concatenate([_rows([B[i]]), _rows([S[i]])])
        out[i] = np.linalg.inv(cov[np.ix_(rows, rows)])
    return out


def upper_triangle_names() -> List[str]:
    """The 78 names of the 12 by 12 upper triangle, in row-major order."""
    labels = [f"intra_{a}" for a in INTRA_NAMES] + [f"inter_{a}" for a in INTER_NAMES]
    return [f"{labels[i]}__{labels[j]}" for i in range(12) for j in range(i, 12)]


def step_stiffness_profiles(blocks: Dict[str, object]) -> Dict[str, np.ndarray]:
    """The upper triangle of the per-step 12 by 12 stiffness, as 78 named profiles."""
    K = step_stiffness_12x12(blocks)
    iu = np.triu_indices(12)
    names = upper_triangle_names()
    return {f"C_K_{nm}": K[:, i, j] for nm, i, j in zip(names, iu[0], iu[1])}


def configurational_entropy(blocks: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Gaussian configurational entropy per step, in units of k_B.

    For a multivariate Gaussian, S = (1/2) ln det(2 pi e Sigma), which is exactly
    defined for the model as fitted and needs no extra assumption. It is the honest
    version of "conformational substate occupancy": it says how much configuration space
    a step explores, without pretending the single Gaussian resolves distinct substates.
    """
    cov = np.asarray(blocks["cov"], dtype=float)
    B, S = np.asarray(blocks["B_idx"]), np.asarray(blocks["S_idx"])
    n = len(S)
    ent = np.zeros(n)
    const = 0.5 * 12 * math.log(2.0 * math.pi * math.e)
    for i in range(n):
        rows = np.concatenate([_rows([B[i]]), _rows([S[i]])])
        sign, logdet = np.linalg.slogdet(cov[np.ix_(rows, rows)])
        ent[i] = const + 0.5 * logdet if sign > 0 else float("nan")
    return {"config_entropy_kB": ent}


# ======================================================================================
# Poses and the Gaussian ensemble
# ======================================================================================

def _cgna_params(seq: str, parameter_set: Optional[str] = None,
                 include_stiffness: bool = True):
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams
    kw = dict(aligned_strands=True, include_stiffness=include_stiffness)
    if parameter_set is not None:
        kw["parameter_set_name"] = parameter_set
    return CGNAPlusParams(seq.upper(), **kw)


def sample_excess(stiffness: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Draw tangent-space fluctuations delta ~ N(0, K^-1), shaped (n_samples, ndof, 6).

    Sampling through the Cholesky factor of the stiffness avoids ever forming the
    covariance: if K = L L^T then solving L^T y = z with z standard normal gives
    y ~ N(0, K^-1). The stiffness is banded with bandwidth 41, so this is cheap even
    for long fragments.
    """
    import scipy.linalg as sla

    L = sla.cholesky(np.asarray(stiffness, dtype=float), lower=True)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((L.shape[0], int(n_samples)))
    y = sla.solve_triangular(L, z, lower=True, trans="T")
    return y.T.reshape(int(n_samples), -1, 6)


def duplex_poses(seq: str, excess: Optional[np.ndarray] = None,
                 parameter_set: Optional[str] = None,
                 params=None) -> Dict[str, np.ndarray]:
    """Base and phosphate poses for the ground state, or for an ensemble.

    Everything is returned in **angstrom**, with the Crick flip undone, which is the
    frame the all-atom templates are expressed in. Shapes are (n_snapshots, n, 4, 4);
    the ground state is a single snapshot.

    `excess` is composed by the library through `CGNAPlusConf.from_params`, which applies
    the Lie-group composition x = gs o exp(delta) that `group_split` implies. Adding
    `delta` to the stored coordinates instead would be wrong, and wrong in a way that
    grows with the size of the fluctuation.
    """
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusConf

    p = params if params is not None else _cgna_params(
        seq, parameter_set, include_stiffness=excess is not None)
    conf = CGNAPlusConf.from_params(p, excess=excess)

    def grab(name: str) -> np.ndarray:
        a = np.array(np.asarray(getattr(conf, name), dtype=float))
        while a.ndim > 4:
            a = a[0]
        if a.ndim == 3:
            a = a[None]
        return a

    wb, cb = grab("watson_base_poses"), grab("crick_base_poses")
    wp, cp = grab("watson_phosphate_poses"), grab("crick_phosphate_poses")
    bp = grab("bp_poses")

    # Detect nm vs angstrom from the base-base spacing rather than trusting a flag.
    d = np.linalg.norm(np.diff(wb[0, :, :3, 3], axis=0), axis=1)
    med = float(np.median(d))
    if 0.1 < med < 1.0:
        scale = 10.0
    elif 1.0 <= med < 10.0:
        scale = 1.0
    else:
        raise RuntimeError(f"cannot infer pose units: median base spacing {med}")
    for arr in (wb, cb, wp, cp, bp):
        arr[..., :3, 3] *= scale

    # Undo the aligned-strand flip so Crick templates place into their own base frame.
    dots = float(np.mean([wb[0, i, :3, 2] @ cb[0, i, :3, 2] for i in range(wb.shape[1])]))
    if dots > 0.0:
        cb = cb @ ca.CRICK_FLIP
        cp = cp @ ca.CRICK_FLIP

    return {"watson_base": wb, "crick_base": cb,
            "watson_phos": wp, "crick_phos": cp, "bp": bp}


def _place(poses: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Apply (n_snap, n, 4, 4) poses to (n, k, 3) local coordinates -> (n_snap, n, k, 3).

    Vectorised over snapshots and residues, because the ensemble is the expensive part
    of this family and a Python loop over 200 snapshots times a few hundred residues is
    the difference between seconds and minutes per fragment.
    """
    R, t = poses[..., :3, :3], poses[..., :3, 3]
    return np.einsum("snij,nkj->snki", R, local) + t[:, :, None, :]


def _local_stack(letters: Sequence[str], atom_names: Sequence[str],
                 templates, phosphate: bool = False) -> np.ndarray:
    """Template coordinates for one atom set, stacked per residue -> (n, k, 3)."""
    out = np.full((len(letters), len(atom_names), 3), np.nan)
    for i, L in enumerate(letters):
        src = templates.phosphate_ideal if phosphate else templates.all_atoms(parent_base(L))
        for j, nm in enumerate(atom_names):
            if nm in src:
                out[i, j] = src[nm]
    return out


def strand_atoms(letters: Sequence[str], base_poses: np.ndarray,
                 phos_poses: np.ndarray, templates,
                 base_names: Sequence[str], phos_names: Sequence[str]
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Place one strand's base-frame and phosphate-frame atoms for every snapshot.

    Returns two dicts keyed by atom name, each of shape (n_snap, n, 3). Phosphate-frame
    atoms are indexed by the residue whose **5' phosphate** they belong to, matching
    `cgna_atomic.build_duplex`: phosphate frame i carries the P that links residue i-1 to
    residue i, so residue 0 has no phosphate and its entries are NaN.
    """
    bl = _local_stack(letters, base_names, templates)
    pl = _local_stack(letters, phos_names, templates, phosphate=True)
    bpos = _place(base_poses, bl)
    ppos = _place(phos_poses, pl)
    # Residue 0 carries no 5' phosphate; the library pads that slot with zeros.
    ppos[:, 0, :, :] = np.nan
    return ({nm: bpos[:, :, j] for j, nm in enumerate(base_names)},
            {nm: ppos[:, :, j] for j, nm in enumerate(phos_names)})


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
              p3: np.ndarray) -> np.ndarray:
    """Signed dihedral in degrees for stacked point sets, by the standard construction."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1 = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1n = np.divide(b1, n1, out=np.full_like(b1, np.nan), where=n1 > 0)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


# ======================================================================================
# Backbone torsions and BI/BII
# ======================================================================================

def backbone_torsions(base: Dict[str, np.ndarray], phos: Dict[str, np.ndarray]
                      ) -> Dict[str, np.ndarray]:
    """The six backbone torsions per residue, in degrees, shaped (n_snap, n).

    Entries that would need an atom off the end of the strand come back NaN rather than
    wrapped or clipped, so a downstream mean over them is forced to be explicit about
    the ends instead of quietly averaging in an invented value.
    """
    n = next(iter(base.values())).shape[1]
    src = {"base": base, "phos": phos}
    out: Dict[str, np.ndarray] = {}
    for tname, quad in TORSION_DEFS.items():
        pts = []
        for frame, atom, off in quad:
            a = src[frame][atom]
            shifted = np.full_like(a, np.nan)
            if off == 0:
                shifted = a
            elif off > 0:
                shifted[:, :n - off] = a[:, off:]
            else:
                shifted[:, -off:] = a[:, :n + off]
            pts.append(shifted)
        out[tname] = _dihedral(*pts)
    return out


def _wrap180(x: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-180, 180]."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def bi_bii(torsions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """epsilon - zeta, and the BI population, per residue.

    Hartmann, Piazzola & Lavery define the BI and BII backbone substates by the sign of
    epsilon - zeta: negative, near -90 degrees, is BI; positive, near +90, is BII.

    **Two caveats travel with this number.** First, it is computed from the minimised
    structure, because the raw reconstruction breaks the C3'-O3' bond that epsilon turns
    about. Second, and this is why `BI_fraction` is provisional: a BI/BII *population*
    needs an ensemble, cgNA+ offers a single Gaussian whose epsilon-minus-zeta
    distribution is unimodal where the real one is bimodal, and minimising every
    snapshot of that ensemble is not affordable. So what is returned from a single
    structure is the sign and magnitude of epsilon - zeta, which is a genuine
    sequence-dependent descriptor, and the population is reported only when an ensemble
    of structures is supplied.
    """
    d = _wrap180(torsions["epsilon"] - torsions["zeta"])
    with np.errstate(invalid="ignore"):
        bi = np.mean(d < 0.0, axis=0) if d.shape[0] > 1 else np.where(
            np.isnan(d[0]), np.nan, (d[0] < 0.0).astype(float))
    return {"eps_minus_zeta": d, "BI_fraction": bi}


# ======================================================================================
# Grooves
# ======================================================================================

def _nanmean0(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=0)


def _nanstd0(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(a, axis=0)


def groove_geometry(p_watson: np.ndarray, p_crick: np.ndarray,
                    floor_minor: np.ndarray, floor_major: np.ndarray,
                    offsets: Tuple[int, int] = None
                    ) -> Dict[str, np.ndarray]:
    """Groove widths and depths per position, shaped (n_snap, n).

    **Width** is the Watson-Crick phosphorus-phosphorus distance at the groove's index
    offset, less 5.8 angstrom for the two phosphate radii, which is the El Hassan and
    Calladine convention every measured groove width is quoted in. The offsets -4 and +4
    are not a convention taken on trust: they are the two local minima of a scan over
    phosphate index offset, and they reproduce the measured B-DNA widths.

    **Depth** is the distance from the midpoint of that phosphorus-phosphorus chord to
    the nearest groove-floor atom of the base pairs the chord spans. The floor atom sets
    are the atoms a protein actually reads from each groove -- N3 and C2 of adenine, N3
    and N2 of guanine, O2 of the pyrimidines in the minor groove; N7 and N6 of adenine,
    N7 and O6 of guanine, N4 of cytosine, O4 and the C7 methyl of thymine in the major.
    This is a stated geometric definition computed from the atoms, not the Curves
    surface-based definition, and the two are not interchangeable to the decimal.

    All index arithmetic is in cgNA+ base-pair indexing, in which Crick position k is
    paired with Watson position k.
    """
    n = p_watson.shape[1]
    out: Dict[str, np.ndarray] = {}
    # `offsets` (minor, major) defaults to the B-DNA minima. Other duplex classes have
    # their own minima of the same scan -- A-form RNA and DNA-RNA hybrids at -3 and +6
    # (Family K, `duplex_classes`) -- and must be measured with them.
    minor_off, major_off = offsets if offsets is not None else (MINOR_OFFSET, MAJOR_OFFSET)
    for label, off, floor in (("minor", minor_off, floor_minor),
                              ("major", major_off, floor_major)):
        width = np.full(p_watson.shape[:2], np.nan)
        depth = np.full(p_watson.shape[:2], np.nan)
        for i in range(n):
            j = i + off
            if j < 0 or j >= n:
                continue
            a, b = p_watson[:, i], p_crick[:, j]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            width[:, i] = np.linalg.norm(a - b, axis=-1) - PHOSPHATE_VDW_A
            mid = 0.5 * (a + b)
            lo, hi = min(i, j), max(i, j)
            span = floor[:, lo:hi + 1]                       # (n_snap, m, k, 3)
            if span.size == 0:
                continue
            d = np.linalg.norm(span - mid[:, None, None, :], axis=-1)
            with np.errstate(invalid="ignore"):
                depth[:, i] = np.nanmin(d.reshape(d.shape[0], -1), axis=1)
        out[f"{label}_width_A"] = width
        out[f"{label}_depth_A"] = depth
    return out


# ======================================================================================
# Helical axis and phosphate position descriptors
# ======================================================================================

def _smooth(path: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along the base-pair axis, with shrinking ends."""
    n = path.shape[-2]
    w = max(1, min(int(window), n))
    half = w // 2
    out = np.empty_like(path)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[..., i, :] = path[..., lo:hi, :].mean(axis=-2)
    return out


def helical_axis(bp_poses: np.ndarray,
                 smooth_bp: int = AXIS_SMOOTH_BP) -> Dict[str, np.ndarray]:
    """Local helical axis, its curvature and the direction of the bend.

    Base-pair origins in B-form DNA sit close to the helical axis but zigzag around it
    with the helical period, so differentiating the raw origin path measures the zigzag
    rather than the axis. Smoothing over one helical turn (11 bp) removes the period and
    leaves the axis, which is the same reason a helical-axis algorithm averages over a
    turn. The window is a declared parameter, not a hidden constant.

    Curvature is returned in degrees per base pair, and the bend direction as an angle
    in the local base-pair frame, measured from the x axis towards the y axis, so that
    "which way does this locus bend" is answerable and not just "how much".
    """
    org = bp_poses[..., :3, 3]
    axis = _smooth(org, smooth_bp)
    t = np.gradient(axis, axis=-2)
    nrm = np.linalg.norm(t, axis=-1, keepdims=True)
    t = np.divide(t, nrm, out=np.full_like(t, np.nan), where=nrm > 1e-12)

    n = axis.shape[-2]
    curv = np.full(axis.shape[:-1], np.nan)
    direction = np.full(axis.shape[:-1], np.nan)
    for i in range(1, n - 1):
        cosang = np.clip(np.sum(t[..., i - 1, :] * t[..., i + 1, :], axis=-1), -1.0, 1.0)
        curv[..., i] = np.degrees(np.arccos(cosang)) / 2.0
        b = np.cross(t[..., i - 1, :], t[..., i + 1, :])
        bn = np.linalg.norm(b, axis=-1, keepdims=True)
        b = np.divide(b, bn, out=np.full_like(b, np.nan), where=bn > 1e-12)
        ex, ey = bp_poses[..., i, :3, 0], bp_poses[..., i, :3, 1]
        direction[..., i] = np.degrees(np.arctan2(np.sum(b * ey, axis=-1),
                                                  np.sum(b * ex, axis=-1)))
    return {"axis": axis, "tangent": t,
            "curvature_deg_per_bp": curv, "bend_direction_deg": direction}


def phosphate_descriptors(p_pos: np.ndarray, bp_poses: np.ndarray,
                          axis: np.ndarray, tangent: np.ndarray
                          ) -> Dict[str, np.ndarray]:
    """Where each phosphate sits relative to the local helical axis.

    Three numbers that between them locate a phosphate completely in cylindrical
    coordinates about the axis: the perpendicular distance to the axis (radius), the
    azimuth measured in the base-pair frame, and the offset along the axis from the
    base pair's own origin. These are the descriptors a groove-binding or
    backbone-contacting protein is sensitive to, and they are the natural output of a
    model that treats the phosphate as its own rigid body.
    """
    rel = p_pos - axis
    along = np.sum(rel * tangent, axis=-1)
    perp = rel - along[..., None] * tangent
    radius = np.linalg.norm(perp, axis=-1)
    ex, ey = bp_poses[..., :3, 0], bp_poses[..., :3, 1]
    azim = np.degrees(np.arctan2(np.sum(perp * ey, axis=-1),
                                 np.sum(perp * ex, axis=-1)))
    return {"radius_A": radius, "azimuth_deg": azim, "axial_offset_A": along}


# ======================================================================================
# The Hoogsteen proxy
# ======================================================================================

HOOGSTEEN_OPENING_DEG = 60.0


def hoogsteen_proxy(blocks: Dict[str, object]) -> np.ndarray:
    """Harmonic free-energy cost, in kT, of reaching a large base-pair opening angle.

    **This is a proxy and the name of the channel says so.** A Watson-Crick to Hoogsteen
    transition is a roughly 180 degree rotation of the purine about its glycosidic bond
    into a second free-energy basin, accompanied by a shear of about 2 angstrom. A single
    Gaussian has no second basin, so no quadratic form can give the real barrier or the
    real population. What it can give is the local stiffness against opening, which does
    order base pairs by how easily they distort towards an open state, and which is the
    quantity the MD and NMR ground truth would be regressed against.

    Computed as (1/2) k_open (theta - theta_0)^2 with k_open the marginal stiffness of
    the intra-base-pair opening coordinate and theta a fixed 60 degree reference.
    """
    cov = np.asarray(blocks["cov"], dtype=float)
    B = np.asarray(blocks["B_idx"])
    gs = np.asarray(blocks["B_gs"], dtype=float)
    target = math.radians(HOOGSTEEN_OPENING_DEG)
    out = np.zeros(len(B))
    for i, bi in enumerate(B):
        r = _rows([bi])
        k = float(np.linalg.inv(cov[np.ix_(r, r)])[2, 2])     # index 2 = opening
        out[i] = 0.5 * k * (target - float(gs[i, 2])) ** 2
    return out


# ======================================================================================
# Assembly
# ======================================================================================

def circ_mean(a: np.ndarray, axis=None) -> np.ndarray:
    """Circular mean in degrees. A linear mean of angles near the wrap is meaningless."""
    import warnings
    r = np.radians(np.asarray(a, dtype=float))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.degrees(np.arctan2(np.nanmean(np.sin(r), axis=axis),
                                     np.nanmean(np.cos(r), axis=axis)))


def circ_sd(a: np.ndarray, axis=None) -> np.ndarray:
    """Circular standard deviation in degrees, sqrt(-2 ln R) with R the mean resultant."""
    import warnings
    r = np.radians(np.asarray(a, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R = np.abs(np.nanmean(np.cos(r), axis=axis) + 1j * np.nanmean(np.sin(r), axis=axis))
        R = np.clip(R, 1e-12, 1.0)
        return np.degrees(np.sqrt(-2.0 * np.log(R)))


# ======================================================================================
# Torsions from a built structure
# ======================================================================================

def torsions_from_structure(pdb_path: str, n_bp: int) -> Dict[str, np.ndarray]:
    """The six backbone torsions per strand, read off an all-atom structure.

    This is the production route for the torsion channels, and it takes a **minimised**
    structure: see the module docstring for the measured reason. Chain A is Watson in
    5'->3' order and chain B is Crick in 5'->3' order, matching
    `cgna_atomic.build_duplex`; the Crick profile is reversed on the way out so both
    strands are reported in cgNA+ base-pair indexing and a position i refers to the same
    base pair on either strand.
    """
    import frames as fr

    res = fr.read_pdb_residues(pdb_path)
    out: Dict[str, np.ndarray] = {}
    for chain, tag, reverse in (("A", "W", False), ("B", "C", True)):
        keys = sorted([k for k in res if k[0] == chain], key=lambda k: k[1])
        m = len(keys)

        def at(i: int, name: str):
            if i < 0 or i >= m:
                return None
            return res[keys[i]].get(name)

        prof = {t: np.full(m, np.nan) for t in TORSION_NAMES}
        for i in range(m):
            quads = {
                "alpha":   (at(i - 1, "O3\'"), at(i, "P"), at(i, "O5\'"), at(i, "C5\'")),
                "beta":    (at(i, "P"), at(i, "O5\'"), at(i, "C5\'"), at(i, "C4\'")),
                "gamma":   (at(i, "O5\'"), at(i, "C5\'"), at(i, "C4\'"), at(i, "C3\'")),
                "delta":   (at(i, "C5\'"), at(i, "C4\'"), at(i, "C3\'"), at(i, "O3\'")),
                "epsilon": (at(i, "C4\'"), at(i, "C3\'"), at(i, "O3\'"), at(i + 1, "P")),
                "zeta":    (at(i, "C3\'"), at(i, "O3\'"), at(i + 1, "P"),
                            at(i + 1, "O5\'")),
            }
            for t, q in quads.items():
                if all(x is not None for x in q):
                    prof[t][i] = float(_dihedral(*[np.asarray(x)[None, None] for x in q])[0, 0])
        for t, v in prof.items():
            padded = np.full(n_bp, np.nan)
            k = min(n_bp, m)
            padded[:k] = v[:k]
            out[f"{t}_{tag}"] = padded[::-1] if reverse else padded
    return out


def sugar_pucker_from_structure(pdb_path: str, n_bp: int) -> Dict[str, np.ndarray]:
    """Pseudorotation phase and amplitude of every deoxyribose, per strand.

    Altona & Sundaralingam 1972: from the five endocyclic torsions nu0..nu4,

        P = atan2((nu4 + nu1) - (nu3 + nu0), 2 nu2 (sin 36 + sin 72)),
        amplitude = nu2 / cos P.

    **Reachable only on the minimised structure, and that is the point.** In the raw
    cgNA+ reconstruction the sugar is a rigid template in the base frame, so its pucker
    is the template's, identical everywhere. The restrained minimisation holds the bases
    and phosphates at their sequence-dependent cgNA+ positions and leaves the sugar free,
    so the ring relaxes into whatever that backbone geometry demands. Measured: every
    sugar C2'-endo, amplitude about 34 degrees, against the B-DNA 35-40; and the phase
    separates by sequence, A-tracts at 140 degrees against GC-rich DNA at 158. That is
    force-field pucker under cgNA+ restraints, the same class of ground truth as the
    molecular dynamics libraries PLAN.md names for this family.

    Reported in cgNA+ base-pair indexing, the Crick profile reversed, as for torsions.
    """
    import frames as fr

    res = fr.read_pdb_residues(pdb_path)
    s36, s72 = math.sin(math.radians(36.0)), math.sin(math.radians(72.0))
    out: Dict[str, np.ndarray] = {}
    for chain, tag, reverse in (("A", "W", False), ("B", "C", True)):
        keys = sorted([k for k in res if k[0] == chain], key=lambda k: k[1])
        phase = np.full(n_bp, np.nan)
        amp = np.full(n_bp, np.nan)
        for i, k in enumerate(keys[:n_bp]):
            r = res[k]
            if not all(a in r for a in ("C1'", "C2'", "C3'", "C4'", "O4'")):
                continue

            def tor(a, b, c, d):
                return float(_dihedral(*[np.asarray(r[x])[None, None]
                                         for x in (a, b, c, d)])[0, 0])
            v0 = tor("C4'", "O4'", "C1'", "C2'")
            v1 = tor("O4'", "C1'", "C2'", "C3'")
            v2 = tor("C1'", "C2'", "C3'", "C4'")
            v3 = tor("C2'", "C3'", "C4'", "O4'")
            v4 = tor("C3'", "C4'", "O4'", "C1'")
            P = math.degrees(math.atan2((v4 + v1) - (v3 + v0), 2.0 * v2 * (s36 + s72)))
            c = math.cos(math.radians(P))
            phase[i] = P % 360.0
            amp[i] = v2 / c if abs(c) > 1e-6 else np.nan
        out[f"phase_{tag}"] = phase[::-1].copy() if reverse else phase
        out[f"amplitude_{tag}"] = amp[::-1].copy() if reverse else amp
    return out


def build_minimised_structure(seq: str, workdir: str,
                              parameter_set: Optional[str] = None,
                              restraint_k: float = 500.0,
                              max_iterations: int = 400) -> Dict[str, object]:
    """Build the all-atom duplex and relax the sugar-phosphate junctions.

    The electrostatics pipeline in `PLAN.md` already performs exactly this step for
    every fragment before PDB2PQR and APBS, so in production this structure is already
    in hand and the torsion channels cost nothing extra. This function exists so Family
    C can be computed standalone, and so the test suite can check the repair.
    """
    import os

    import cgna_atomic as ca
    import minimize as mz

    os.makedirs(workdir, exist_ok=True)
    gs = ca.CGNAGroundState(seq.upper(), parameter_set=parameter_set)
    atoms = ca.build_duplex(seq.upper(), gs=gs)
    raw = os.path.join(workdir, "raw.pdb")
    mini = os.path.join(workdir, "min.pdb")
    ca.write_pdb(atoms, raw, f"PANGEA Family C: {seq.upper()[:40]}")
    report = mz.restrained_minimise(raw, mini, restraint_k=restraint_k,
                                    max_iterations=max_iterations)
    return {"raw_pdb": raw, "minimised_pdb": mini, "report": report}


# ======================================================================================
# Assembly
# ======================================================================================

def geometry_state(seq: str, n_ensemble: int = N_ENSEMBLE_DEFAULT,
                   seed: int = 0, parameter_set: Optional[str] = None,
                   minimised_pdb: Optional[str] = None,
                   min_ensemble: Optional[Dict[str, np.ndarray]] = None,
                   groove_offsets: Optional[Tuple[int, int]] = None
                   ) -> Dict[str, object]:
    """Everything Family C is computed from, for one sequence.

    The ground state and the ensemble are built separately on purpose: ground-state
    quantities are reported from the ground state, which is exact, and only the
    *fluctuations* come from sampling, so sampling error lands on a standard deviation
    rather than on a mean.

    `minimised_pdb` supplies the relaxed structure the torsion channels need. Without
    it those channels are absent rather than approximated, because the module docstring
    records by measurement that the unminimised values are wrong by up to 63 degrees.
    """
    import cgna_atomic as ca

    seq = seq.upper()
    n = len(seq)
    params = _cgna_params(seq, parameter_set, include_stiffness=True)
    stiff = params.stiffmat
    stiff = stiff.toarray() if hasattr(stiff, "toarray") else np.asarray(stiff)
    blocks = cgna_blocks({"param_names": list(params.param_names),
                          "gs": np.asarray(params.gs, dtype=float),
                          "stiffness": stiff})

    templates = ca.NucleotideTemplates()
    w_letters = list(seq)
    c_letters = [COMPLEMENT[c] for c in seq][::-1]

    def floor_stack(letters, poses_arr, table, reverse):
        k = max(len(v) for v in table.values())
        local = np.full((len(letters), k, 3), np.nan)
        for i, L in enumerate(letters):
            tpl = templates.all_atoms(parent_base(L))
            for j, nm in enumerate(table[parent_base(L)]):
                if nm in tpl:
                    local[i, j] = tpl[nm]
        placed = _place(poses_arr, local)
        return placed[:, ::-1] if reverse else placed

    def assemble(excess):
        poses = duplex_poses(seq, excess=excess, params=params)
        wb, cb = poses["watson_base"], poses["crick_base"]
        wp, cp = poses["watson_phos"], poses["crick_phos"]
        _, w_phos = strand_atoms(w_letters, wb, wp, templates,
                                 BASE_ATOMS_NEEDED, PHOS_ATOMS_NEEDED)
        _, c_phos = strand_atoms(c_letters, cb[:, ::-1], cp[:, ::-1], templates,
                                 BASE_ATOMS_NEEDED, PHOS_ATOMS_NEEDED)

        fmin = np.concatenate([floor_stack(w_letters, wb, MINOR_FLOOR, False),
                               floor_stack(c_letters, cb[:, ::-1], MINOR_FLOOR, True)],
                              axis=2)
        fmaj = np.concatenate([floor_stack(w_letters, wb, MAJOR_FLOOR, False),
                               floor_stack(c_letters, cb[:, ::-1], MAJOR_FLOOR, True)],
                              axis=2)

        pw = w_phos["P"]
        pc = c_phos["P"][:, ::-1]        # back into cgNA+ base-pair indexing
        ax = helical_axis(poses["bp"])
        return {"poses": poses,
                "grooves": groove_geometry(pw, pc, fmin, fmaj, groove_offsets),
                "axis": ax,
                "phos_W": phosphate_descriptors(pw, poses["bp"], ax["axis"],
                                                ax["tangent"]),
                "phos_C": phosphate_descriptors(pc, poses["bp"], ax["axis"],
                                                ax["tangent"])}

    ground = assemble(None)
    ens = assemble(sample_excess(stiff, n_ensemble, seed)) if n_ensemble else None

    torsions = torsions_from_structure(minimised_pdb, n) if minimised_pdb else None
    pucker = sugar_pucker_from_structure(minimised_pdb, n) if minimised_pdb else None

    return {"sequence": seq, "n_bp": n, "blocks": blocks, "params": params,
            "ground": ground, "ensemble": ens, "torsions": torsions, "pucker": pucker,
            "min_ensemble": min_ensemble,
            "n_ensemble": int(n_ensemble or 0)}


# ======================================================================================
# Conditions: methylation is real, salt and temperature are not available
# ======================================================================================

def methylate_cpg(seq: str) -> str:
    """Fully methylate every CpG, in the cgNA+ modified alphabet.

    A CpG step is self-complementary, so full methylation modifies both strands. In the
    `Di_hmethyl_methylated-hemi_combine` alphabet that is written C -> M for the
    5-methylcytosine and G -> N for the guanine whose partner is methylated, giving
    CG -> MN. This is the only condition axis in Family C backed by a fitted parameter
    set rather than by an assumption.

    CpGs inside the terminal dinucleotide are left unmethylated. That parameter set
    carries `stiff_end5` and `stiff_end3` blocks for the 16 unmodified dinucleotides
    only, so a modified base in an end dinucleotide raises a `KeyError` on a key such as
    'GM' rather than returning a wrong answer. Since the plan scores only the central
    window of every fragment and discards the ends as edge artifacts, nothing that is
    reported is affected.
    """
    s = list(seq.upper())
    n = len(s)
    i = 0
    while i < n - 1:
        if s[i] == "C" and s[i + 1] == "G" and i >= 2 and i + 1 <= n - 3:
            s[i], s[i + 1] = "M", "N"
            i += 2
        else:
            i += 1
    return "".join(s)


CONDITION_DELTA_NAMES: Tuple[str, ...] = (
    "dtwist_deg", "droll_deg", "drise_A", "dminor_width_A", "dmajor_width_A")

_NO_PARAMETER_SET = (
    "cgNA+ ships one parameter set, fit at a single ionic strength and temperature, so "
    "a re-predicted geometry under this condition would be the unmodified geometry "
    "with a new label on it. The value is NaN rather than a plausible number, and the "
    "ABC and hexamer-scale MD libraries PLAN.md names as Family C ground truth are what "
    "would supply it.")


def condition_deltas(seq: str, n_ensemble: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    """Change in the main geometry channels under each condition axis.

    Methylation is computed: the methylated and unmethylated sequences are both run
    through the *same* methylation-capable parameter set, so the difference is the
    modification and not a change of parameter set. High salt and elevated temperature
    return NaN, for the reason in `_NO_PARAMETER_SET`.
    """
    base = geometry_state(seq, n_ensemble=n_ensemble, parameter_set=PARAM_SET_METHYL)
    meth = geometry_state(methylate_cpg(seq), n_ensemble=n_ensemble,
                          parameter_set=PARAM_SET_METHYL)

    def summary(st):
        c = rigid_base_coordinates(st["blocks"])
        g = st["ground"]["grooves"]
        return {"dtwist_deg": c["twist"], "droll_deg": c["roll"], "drise_A": c["rise"],
                "dminor_width_A": g["minor_width_A"][0],
                "dmajor_width_A": g["major_width_A"][0]}

    a, b = summary(base), summary(meth)
    out = {"methylation": {}}
    for k in CONDITION_DELTA_NAMES:
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        m = min(len(x), len(y))
        d = np.full(len(np.asarray(a[k])), np.nan)
        d[:m] = y[:m] - x[:m]
        out["methylation"][k] = d
    n = len(seq)
    for cond in ("high_salt", "high_temperature"):
        out[cond] = {k: np.full(n, np.nan) for k in CONDITION_DELTA_NAMES}
    return out


# ======================================================================================
# Sugar pucker: stated, and stated to be unavailable
# ======================================================================================

_PUCKER_UNAVAILABLE = (
    "cgNA+ treats the sugar ring as rigid in the base frame, so a pucker computed from "
    "an unminimised reconstruction is the template's pucker repeated with no sequence "
    "dependence, and after the restrained minimisation it is the force field's pucker "
    "rather than a cgNA+ prediction. Pseudorotation phase and amplitude need the "
    "hexamer-scale MD libraries PLAN.md names as Family C ground truth.")

PUCKER_CHANNELS: Tuple[str, ...] = (
    "C_sugar_pucker_phase_W", "C_sugar_pucker_phase_C",
    "C_sugar_pucker_amplitude_W", "C_sugar_pucker_amplitude_C",
    "C_sugar_C2endo_prob_W", "C_sugar_C2endo_prob_C",
    "C_sugar_C3endo_prob_W", "C_sugar_C3endo_prob_C")


# ======================================================================================
# Profiles, channels, windows
# ======================================================================================

def geometry_profiles(seq: str, state: Optional[Dict[str, object]] = None,
                      n_ensemble: int = N_ENSEMBLE_DEFAULT,
                      minimised_pdb: Optional[str] = None,
                      with_conditions: bool = True) -> Dict[str, np.ndarray]:
    """Every per-base-pair Family C channel, each an array of length n_bp.

    Positions that a channel is not defined at -- the ends, where a step or a torsion
    runs off the strand -- are NaN. They are never filled in: a padded end value is
    indistinguishable from a measurement once it reaches a training set.
    """
    seq = seq.upper()
    n = len(seq)
    st = state if state is not None else geometry_state(
        seq, n_ensemble=n_ensemble, minimised_pdb=minimised_pdb)
    blocks = st["blocks"]
    out: Dict[str, np.ndarray] = {}

    def put(name: str, v: np.ndarray) -> None:
        a = np.asarray(v, dtype=float).ravel()
        padded = np.full(n, np.nan)
        padded[:min(n, len(a))] = a[:min(n, len(a))]
        out[name] = padded

    for lab, v in rigid_base_coordinates(blocks).items():
        put(f"C_{lab}", v)
    for name, v in step_stiffness_profiles(blocks).items():
        put(name, v)
    put("C_config_entropy_kB", configurational_entropy(blocks)["config_entropy_kB"])
    put("C_hoogsteen_opening_proxy_kT", hoogsteen_proxy(blocks))

    g, e = st["ground"], st["ensemble"]
    for key in ("minor_width_A", "major_width_A", "minor_depth_A", "major_depth_A"):
        put(f"C_{key}", g["grooves"][key][0])
        put(f"C_{key}_sd", _nanstd0(e["grooves"][key]) if e else np.full(n, np.nan))

    put("C_axis_curvature_deg_per_bp", g["axis"]["curvature_deg_per_bp"][0])
    put("C_axis_bend_direction_deg", g["axis"]["bend_direction_deg"][0])
    put("C_axis_curvature_sd", circ_sd(e["axis"]["curvature_deg_per_bp"], axis=0)
        if e else np.full(n, np.nan))
    put("C_axis_bend_direction_sd", circ_sd(e["axis"]["bend_direction_deg"], axis=0)
        if e else np.full(n, np.nan))

    for tag in ("W", "C"):
        d = g[f"phos_{tag}"]
        ed = e[f"phos_{tag}"] if e else None
        for key, circular in (("radius_A", False), ("azimuth_deg", True),
                              ("axial_offset_A", False)):
            put(f"C_phosphate_{key}_{tag}", d[key][0])
            if ed is None:
                put(f"C_phosphate_{key}_{tag}_sd", np.full(n, np.nan))
            else:
                put(f"C_phosphate_{key}_{tag}_sd",
                    circ_sd(ed[key], axis=0) if circular else _nanstd0(ed[key]))

    tors = st.get("torsions")
    me = st.get("min_ensemble") or {}
    for t in TORSION_NAMES:
        for tag in ("W", "C"):
            put(f"C_{t}_{tag}", tors[f"{t}_{tag}"] if tors else np.full(n, np.nan))
            put(f"C_{t}_{tag}_sd", me.get(f"{t}_{tag}_sd", np.full(n, np.nan)))
    for tag in ("W", "C"):
        if tors:
            put(f"C_eps_minus_zeta_{tag}",
                _wrap180(tors[f"epsilon_{tag}"] - tors[f"zeta_{tag}"]))
        else:
            put(f"C_eps_minus_zeta_{tag}", np.full(n, np.nan))
        # A population, from relaxed ensemble snapshots where they exist; NaN otherwise.
        # It used to be a single structure's 0/1 indicator under a "fraction" name.
        put(f"C_BI_fraction_{tag}", me.get(f"BI_fraction_{tag}", np.full(n, np.nan)))

    pk = st.get("pucker")
    for tag in ("W", "C"):
        put(f"C_sugar_pucker_phase_{tag}",
            pk[f"phase_{tag}"] if pk else np.full(n, np.nan))
        put(f"C_sugar_pucker_amplitude_{tag}",
            pk[f"amplitude_{tag}"] if pk else np.full(n, np.nan))
        # A state probability needs an ensemble of relaxed structures; one structure
        # gives a state, which here is C2'-endo for 98-100 percent of sugars.
        put(f"C_sugar_C2endo_prob_{tag}", me.get(f"C2endo_prob_{tag}", np.full(n, np.nan)))
        put(f"C_sugar_C3endo_prob_{tag}", me.get(f"C3endo_prob_{tag}", np.full(n, np.nan)))

    if with_conditions:
        cd = condition_deltas(seq, n_ensemble=0)
        for cond, prefix in (("methylation", "meth"), ("high_salt", "salt"),
                             ("high_temperature", "temp")):
            for k, v in cd[cond].items():
                put(f"C_{prefix}_{k}", v)
    else:
        for prefix in ("meth", "salt", "temp"):
            for k in CONDITION_DELTA_NAMES:
                put(f"C_{prefix}_{k}", np.full(n, np.nan))
    return out


def _coupling(blocks: Dict[str, object]) -> Dict[str, float]:
    """Stretch and overtwist couplings, from the marginal step covariance.

    Reported as correlation coefficients rather than raw covariances, so they do not
    depend on the unit convention of the two coordinates being coupled, and so the sign
    -- which is the part that carries the physics -- is unambiguous.
    """
    cov = np.asarray(blocks["cov"], dtype=float)
    S = np.asarray(blocks["S_idx"])
    pairs = {"twist_rise": (2, 5), "twist_roll": (1, 2), "slide_rise": (4, 5)}
    acc = {k: [] for k in pairs}
    overtwist = []
    for s in S:
        r = _rows([s])
        c = cov[np.ix_(r, r)]
        for k, (i, j) in pairs.items():
            den = math.sqrt(c[i, i] * c[j, j])
            acc[k].append(c[i, j] / den if den > 0 else float("nan"))
        overtwist.append(c[2, 2])
    out = {f"C_{k}_coupling": float(np.nanmean(v)) for k, v in acc.items()}
    out["C_overtwist_compliance_rad2"] = float(np.nanmean(overtwist))
    return out


def geometry_channels(seq: str, state: Optional[Dict[str, object]] = None,
                      n_ensemble: int = N_ENSEMBLE_DEFAULT,
                      minimised_pdb: Optional[str] = None,
                      symmetrise: bool = False,
                      profiles: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, float]:
    """Every whole-sequence Family C scalar channel.

    With `symmetrise=True` the channels are averaged with the reverse-complement
    computation by declared parity, for the reason recorded in
    `channels.symmetrise_channels`: the cgNA+ stiffness is not exactly
    reverse-complement symmetric, the physical duplex is, and PANGEA enforces the
    symmetry architecturally, so an unsymmetrised target would leave an irreducible
    error floor that reads as model error.
    """
    if symmetrise:
        from channels import reverse_complement, symmetrise_channels
        fwd = geometry_channels(seq, state, n_ensemble, minimised_pdb, False)
        rev = geometry_channels(reverse_complement(seq), None, n_ensemble, None, False)
        return symmetrise_channels(fwd, rev)

    seq = seq.upper()
    st = state if state is not None else geometry_state(
        seq, n_ensemble=n_ensemble, minimised_pdb=minimised_pdb)
    # `profiles` lets a caller that already has them avoid recomputing. That matters:
    # the per-base-pair pass includes the methylation condition delta, which is two more
    # cgNA+ evaluations, so computing it twice costs about 1.3 s per 61 bp fragment --
    # a fifth of the whole analytic stack.
    prof = profiles if profiles is not None else geometry_profiles(seq, state=st)
    blocks = st["blocks"]
    out: Dict[str, float] = {}

    for lab in INTRA_NAMES + INTER_NAMES:
        v = prof[f"C_{lab}"]
        out[f"C_mean_{lab}"] = float(circ_mean(v) if lab in
                                     ("buckle", "propeller", "opening", "tilt",
                                      "roll", "twist") else np.nanmean(v))

    for key in ("minor_width_A", "major_width_A", "minor_depth_A", "major_depth_A"):
        out[f"C_mean_{key}"] = float(np.nanmean(prof[f"C_{key}"]))
    out["C_min_minor_width_A"] = float(np.nanmin(prof["C_minor_width_A"]))
    out["C_max_minor_width_A"] = float(np.nanmax(prof["C_minor_width_A"]))
    out["C_mean_minor_width_sd_A"] = float(np.nanmean(prof["C_minor_width_A_sd"]))
    out["C_mean_major_width_sd_A"] = float(np.nanmean(prof["C_major_width_A_sd"]))

    curv = prof["C_axis_curvature_deg_per_bp"]
    out["C_total_curvature_deg"] = float(np.nansum(curv))
    out["C_max_curvature_deg_per_bp"] = float(np.nanmax(curv))
    out["C_mean_bend_direction_deg"] = float(circ_mean(prof["C_axis_bend_direction_deg"]))

    for tag in ("W", "C"):
        out[f"C_mean_phosphate_radius_{tag}"] = float(
            np.nanmean(prof[f"C_phosphate_radius_A_{tag}"]))
        out[f"C_mean_phosphate_azimuth_{tag}"] = float(
            circ_mean(prof[f"C_phosphate_azimuth_deg_{tag}"]))
        out[f"C_mean_BI_fraction_{tag}"] = float(np.nanmean(prof[f"C_BI_fraction_{tag}"]))

    for t in TORSION_NAMES:
        both = np.concatenate([prof[f"C_{t}_W"], prof[f"C_{t}_C"]])
        out[f"C_mean_{t}"] = float(circ_mean(both))

    out["C_total_config_entropy_kB"] = float(np.nansum(prof["C_config_entropy_kB"]))
    out.update(_coupling(blocks))
    out["C_mean_hoogsteen_proxy_kT"] = float(
        np.nanmean(prof["C_hoogsteen_opening_proxy_kT"]))
    out["C_max_hoogsteen_proxy_kT"] = float(
        np.nanmax(prof["C_hoogsteen_opening_proxy_kT"]))

    out["C_meth_mean_dtwist_deg"] = float(np.nanmean(prof["C_meth_dtwist_deg"]))
    out["C_meth_mean_dminor_width_A"] = float(np.nanmean(prof["C_meth_dminor_width_A"]))
    out["C_meth_mean_dmajor_width_A"] = float(np.nanmean(prof["C_meth_dmajor_width_A"]))
    return out


def geometry_window_channels(seq: str, state: Optional[Dict[str, object]] = None,
                             ladder: Sequence[int] = WINDOW_LADDER_BP,
                             n_ensemble: int = N_ENSEMBLE_DEFAULT,
                             minimised_pdb: Optional[str] = None
                             ) -> Dict[str, np.ndarray]:
    """Family C on the revision-2 window ladder.

    Revision 2 makes resolution the fifth axis, and the point of the ladder for this
    family is that a groove width or a twist averaged over 10 bp and over 500 bp are
    different observables, not the same observable at different precision: the first
    tracks a protein-sized footprint and the second tracks the composition of a locus.
    Windows longer than the sequence are skipped rather than silently truncated.
    """
    prof = geometry_profiles(seq, state=state, n_ensemble=n_ensemble,
                             minimised_pdb=minimised_pdb)
    keys = ("C_twist", "C_roll", "C_rise", "C_minor_width_A", "C_major_width_A",
            "C_axis_curvature_deg_per_bp", "C_config_entropy_kB")
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
                    means[i + w // 2] = np.nanmean(seg)
            out[f"{k}_w{w}"] = means
    return out


# ======================================================================================
# Registry
#
# Every parity below was measured on an exact palindrome before it was declared, never
# recalled. The measurements are reproduced in `tests/test_geometry.py`, and the ones
# that came out differently from the obvious guess are:
#
#   the 12 rigid-base coordinates obey the 3DNA rules exactly (1e-13) -- but only once
#   the translations are converted to the mid-step frame; in cgNA+'s native
#   preceding-frame convention shift is off by 80 percent. See `to_mid_frame`.
#
#   groove widths and depths are SYMMETRIC under an index shift of exactly the groove's
#   own phosphate offset, +4 for the minor groove and -4 for the major, holding to
#   1e-14. A groove is a property of a pair of phosphates on opposite strands, so
#   reverse complement moves it along the sequence as well as reflecting it. Declared
#   COMPOSITE per base pair; the whole-sequence means are unaffected and stay SYMMETRIC.
#
#   phosphate descriptors exchange strands with no index shift at all, to 1e-14, with
#   radius SYMMETRIC and azimuth and axial offset ANTISYMMETRIC under the exchange.
#   Declared STRANDED, with the exchange rule asserted in the tests.
#
#   all 78 entries of the per-step stiffness are COMPOSITE. The first guess was that
#   only the 36 cross intra-by-inter entries would be, and the pure blocks would carry a
#   simple parity; measurement says otherwise, with median residuals of 0.48, 0.65 and
#   0.52 for the three blocks where the coordinates themselves reach 1e-13. See
#   `_CROSS_BLOCK` for the two causes. The declaration follows the measurement.
# ======================================================================================

def _parity_of(coord: str) -> Parity:
    return (Parity.ANTISYMMETRIC if coord in ANTISYMMETRIC_COORDS
            else Parity.SYMMETRIC)


def _combine(a: Parity, b: Parity) -> Parity:
    anti = (a is Parity.ANTISYMMETRIC) ^ (b is Parity.ANTISYMMETRIC)
    return Parity.ANTISYMMETRIC if anti else Parity.SYMMETRIC


def _c(name: str, resolution: Resolution, parity: Parity, units: str, desc: str,
       source: str = CGNA, provisional: bool = False, reason: str = "",
       condition: bool = False, window_bp: Optional[int] = None) -> ChannelSpec:
    return ChannelSpec(name=name, family=FAMILY, resolution=resolution, parity=parity,
                       units=units, description=desc, window_bp=window_bp,
                       condition_dependent=condition, provisional=provisional,
                       provisional_reason=reason, source=source)


_GROOVE_SHIFT = ("a groove is a property of two phosphates on opposite strands, so "
                 "reverse complement shifts its index by the groove's own offset as "
                 "well as reflecting the sequence; measured exact to 1e-14")
# Measured, not assumed: on an exact palindrome every one of the 78 entries fails a
# simple parity, under either index map and either sign. Median residual 0.48 for the
# intra-intra block, 0.65 for intra-inter, 0.52 for inter-inter, against the 1e-13 the
# 12 coordinates themselves achieve. Two independent causes, both real:
_CROSS_BLOCK = (
    "the per-step 12 by 12 is not a reverse-complement-covariant object, for two "
    "measured reasons. First, its 12 coordinates are the intra-base-pair coordinates of "
    "base pair i together with the step from i to i+1 -- the standard rigid-base "
    "grouping PLAN.md asks for -- but reverse complement sends intra index i to n-1-i "
    "and step index i to n-2-i, so the block lands on a neighbouring block rather than "
    "on itself; the covariant object is the 18 by 18 over (intra_i, step_i, "
    "intra_{i+1}). Second, the stiffness is a quadratic form in the native cgNA+ "
    "coordinates, whose translational half is expressed in the preceding base-pair "
    "frame, and that convention does not obey the parity rules either (see "
    "to_mid_frame); carrying the stiffness into the mid-step frame needs a congruence "
    "transform by the Jacobian of that change of variables, which is not applied here. "
    "Both are recorded in DECISIONS.md as open items. The channels are emitted and "
    "usable; what is withheld is a parity claim about them.")
_ENSEMBLE_SD = ("standard deviation over draws from the fitted cgNA+ Gaussian; the "
                "model's own spread, not a measured one")
_TORSION_SRC = "cgNA+ phosphate frames, relaxed by restrained minimisation"
_TORSION_SD_PROV = (
    "a torsion variance needs the restrained minimisation applied to every ensemble "
    "snapshot, not just the ground state. Without it the raw reconstruction breaks the "
    "C3'-O3' bond across the ensemble, spanning 0.64 to 3.44 angstrom, and the "
    "resulting torsion spread of over 140 degrees is an artifact of unconstrained "
    "rigid bodies rather than a fluctuation.")
_BI_PROV = (
    "cgNA+ is a single Gaussian whose epsilon-minus-zeta distribution is unimodal where "
    "the real one is bimodal, so a BI/BII population from one minimised structure is a "
    "sign, not an occupancy. The ABC and hexamer-scale MD libraries PLAN.md names as "
    "Family C ground truth are what would supply the population.")
_HOOGSTEEN_PROV = (
    "a harmonic proxy: a Watson-Crick to Hoogsteen transition is a roughly 180 degree "
    "glycosidic flip into a second free-energy basin, and a single Gaussian has no "
    "second basin, so this is the local stiffness against opening rather than a barrier "
    "or a population.")

_PUCKER_SRC = (f"{ALTONA}, on the AMBER OL15 restrained minimisation of the cgNA+ "
               "structure")
_ENSEMBLE_SRC = ("cgNA+ Gaussian ensemble snapshots, each relaxed by AMBER OL15 restrained "
                 "minimisation (geometry.minimised_ensemble_stats)")

_SPECS: List[ChannelSpec] = []

# --- the 12 rigid-base coordinates, per base pair and as whole-sequence means --------
for _grp, _labels, _unit_rot, _unit_tr in (
        ("intra-base-pair", INTRA_NAMES, "degrees", "angstrom"),
        ("inter-base-pair step", INTER_NAMES, "degrees", "angstrom")):
    for _j, _lab in enumerate(_labels):
        _u = _unit_rot if _j < 3 else _unit_tr
        _SPECS.append(_c(f"C_{_lab}", Resolution.PER_BP, _parity_of(_lab), _u,
                         f"{_grp} {_lab}, mid-step frame", ELHASSAN))
        _SPECS.append(_c(f"C_mean_{_lab}", Resolution.GLOBAL, _parity_of(_lab), _u,
                         f"sequence mean {_grp} {_lab}", ELHASSAN))

# --- the 78 upper-triangle entries of the per-step 12 by 12 stiffness ----------------
_LABELS12 = [("intra", a) for a in INTRA_NAMES] + [("inter", a) for a in INTER_NAMES]
for _i in range(12):
    for _j in range(_i, 12):
        _ka, _kb = _LABELS12[_i], _LABELS12[_j]
        _nm = f"C_K_{_ka[0]}_{_ka[1]}__{_kb[0]}_{_kb[1]}"
        _SPECS.append(_c(_nm, Resolution.PER_BP, Parity.COMPOSITE,
                         "kT per unit squared",
                         f"marginal stiffness coupling {_ka[1]} to {_kb[1]} "
                         f"({_ka[0]} by {_kb[0]}); {_CROSS_BLOCK}"))

# --- grooves --------------------------------------------------------------------------
for _g in ("minor", "major"):
    for _q, _desc in (("width", "phosphorus-phosphorus distance less 5.8 angstrom"),
                      ("depth", "distance from the P-P chord midpoint to the nearest "
                                "groove-floor atom")):
        _SPECS.append(_c(f"C_{_g}_{_q}_A", Resolution.PER_BP, Parity.COMPOSITE,
                         "angstrom", f"{_g} groove {_q}: {_desc}. {_GROOVE_SHIFT}",
                         ELHASSAN))
        _SPECS.append(_c(f"C_{_g}_{_q}_A_sd", Resolution.PER_BP, Parity.COMPOSITE,
                         "angstrom", f"{_g} groove {_q} fluctuation. {_ENSEMBLE_SD}",
                         ELHASSAN))
        _SPECS.append(_c(f"C_mean_{_g}_{_q}_A", Resolution.GLOBAL, Parity.SYMMETRIC,
                         "angstrom", f"sequence mean {_g} groove {_q}", ELHASSAN))
_SPECS += [
    _c("C_min_minor_width_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       "narrowest point of the minor groove, the feature minor-groove binders read",
       ELHASSAN),
    _c("C_max_minor_width_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       "widest point of the minor groove", ELHASSAN),
    _c("C_mean_minor_width_sd_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       f"mean minor groove width fluctuation. {_ENSEMBLE_SD}", ELHASSAN),
    _c("C_mean_major_width_sd_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       f"mean major groove width fluctuation. {_ENSEMBLE_SD}", ELHASSAN),
]

# --- helical axis ----------------------------------------------------------------------
_SPECS += [
    _c("C_axis_curvature_deg_per_bp", Resolution.PER_BP, Parity.SYMMETRIC,
       "degrees per bp", "local helical axis curvature, axis smoothed over one turn"),
    _c("C_axis_bend_direction_deg", Resolution.PER_BP, Parity.COMPOSITE, "degrees",
       "direction of the local bend in the base-pair frame; an azimuth measured in a "
       "frame that itself reflects under reverse complement, so the map is a reflection "
       "of the angle and not a sign on the channel"),
    _c("C_axis_curvature_sd", Resolution.PER_BP, Parity.SYMMETRIC, "degrees per bp",
       f"curvature fluctuation. {_ENSEMBLE_SD}"),
    _c("C_axis_bend_direction_sd", Resolution.PER_BP, Parity.SYMMETRIC, "degrees",
       f"circular spread of the bend direction. {_ENSEMBLE_SD}"),
    _c("C_total_curvature_deg", Resolution.GLOBAL, Parity.SYMMETRIC, "degrees",
       "integrated helical axis curvature"),
    _c("C_max_curvature_deg_per_bp", Resolution.GLOBAL, Parity.SYMMETRIC,
       "degrees per bp", "sharpest local bend"),
    _c("C_mean_bend_direction_deg", Resolution.GLOBAL, Parity.COMPOSITE, "degrees",
       "circular mean bend direction"),
]

# --- phosphate position descriptors ----------------------------------------------------
for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
    for _key, _u, _d in (("radius_A", "angstrom", "distance to the local helical axis"),
                         ("azimuth_deg", "degrees", "azimuth in the base-pair frame"),
                         ("axial_offset_A", "angstrom",
                          "offset along the axis from the base-pair origin")):
        _SPECS.append(_c(f"C_phosphate_{_key}_{_tag}", Resolution.PER_BP,
                         Parity.STRANDED, _u, f"{_strand} phosphate {_d}"))
        _SPECS.append(_c(f"C_phosphate_{_key}_{_tag}_sd", Resolution.PER_BP,
                         Parity.STRANDED, _u,
                         f"{_strand} phosphate {_d} fluctuation. {_ENSEMBLE_SD}"))
    _SPECS.append(_c(f"C_mean_phosphate_radius_{_tag}", Resolution.GLOBAL,
                     Parity.STRANDED, "angstrom",
                     f"mean {_strand} phosphate radius"))
    _SPECS.append(_c(f"C_mean_phosphate_azimuth_{_tag}", Resolution.GLOBAL,
                     Parity.STRANDED, "degrees",
                     f"circular mean {_strand} phosphate azimuth"))

# --- backbone torsions, BI/BII, pucker --------------------------------------------------
for _t in TORSION_NAMES:
    for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
        _SPECS.append(_c(f"C_{_t}_{_tag}", Resolution.PER_BP, Parity.STRANDED, "degrees",
                         f"{_strand} backbone torsion {_t}, from the minimised structure",
                         _TORSION_SRC))
        _SPECS.append(_c(f"C_{_t}_{_tag}_sd", Resolution.PER_BP, Parity.STRANDED,
                         "degrees",
                         f"{_strand} {_t} circular spread over {ENSEMBLE_MIN_SNAPSHOTS} "
                         "relaxed cgNA+ ensemble snapshots; computed on the plan's 10 "
                         "percent ensemble subset and NaN elsewhere", _ENSEMBLE_SRC))
    _SPECS.append(_c(f"C_mean_{_t}", Resolution.GLOBAL, Parity.SYMMETRIC, "degrees",
                     f"circular mean backbone torsion {_t}, both strands", _TORSION_SRC))
for _tag, _strand in (("W", "Watson"), ("C", "Crick")):
    _SPECS.append(_c(f"C_eps_minus_zeta_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                     "degrees", f"{_strand} epsilon minus zeta, the BI/BII discriminator",
                     HARTMANN))
    _SPECS.append(_c(f"C_BI_fraction_{_tag}", Resolution.PER_BP, Parity.STRANDED,
                     "probability",
                     f"{_strand} BI population: fraction of {ENSEMBLE_MIN_SNAPSHOTS} relaxed "
                     "ensemble snapshots with epsilon - zeta < 0, so resolved in steps of "
                     f"1/{ENSEMBLE_MIN_SNAPSHOTS}; 10 percent ensemble subset, NaN elsewhere",
                     f"{HARTMANN}; {_ENSEMBLE_SRC}"))
    _SPECS.append(_c(f"C_mean_BI_fraction_{_tag}", Resolution.GLOBAL, Parity.STRANDED,
                     "probability", f"mean {_strand} BI population over the fragment",
                     f"{HARTMANN}; {_ENSEMBLE_SRC}"))

_PUCKER_PROB_PROV = (
    "a state probability needs an ensemble of relaxed structures; a single minimised "
    "structure gives a state, C2'-endo for 98-100 percent of sugars, which as a "
    "probability would be a near-constant indicator. Computable from the minimised "
    "cgNA+ ensemble on the plan's 10 percent ensemble subset, which has not been run.")
for _nm in PUCKER_CHANNELS:
    _is_prob = "_prob_" in _nm
    _SPECS.append(_c(_nm, Resolution.PER_BP, Parity.STRANDED,
                     "probability" if _is_prob else "degrees",
                     ("pseudorotation phase of this deoxyribose" if "phase" in _nm else
                      "pseudorotation amplitude of this deoxyribose" if "amplitude" in _nm
                      else "fraction of relaxed ensemble snapshots with this deoxyribose "
                           "in the named pucker state; 10 percent ensemble subset, NaN "
                           "elsewhere"),
                     _ENSEMBLE_SRC if _is_prob else _PUCKER_SRC))

# --- entropy, couplings, Hoogsteen ------------------------------------------------------
_SPECS += [
    _c("C_config_entropy_kB", Resolution.PER_BP, Parity.COMPOSITE, "k_B",
       "Gaussian configurational entropy of the 12 coordinates of this step; "
       + _CROSS_BLOCK),
    _c("C_total_config_entropy_kB", Resolution.GLOBAL, Parity.SYMMETRIC, "k_B",
       "summed configurational entropy"),
    _c("C_twist_rise_coupling", Resolution.GLOBAL, Parity.SYMMETRIC, "correlation",
       "twist-stretch coupling, as a correlation so its sign is unit-independent"),
    _c("C_twist_roll_coupling", Resolution.GLOBAL, Parity.SYMMETRIC, "correlation",
       "overtwist-roll coupling"),
    _c("C_slide_rise_coupling", Resolution.GLOBAL, Parity.SYMMETRIC, "correlation",
       "slide-rise coupling"),
    _c("C_overtwist_compliance_rad2", Resolution.GLOBAL, Parity.SYMMETRIC, "rad^2",
       "mean twist variance, the compliance to over- and undertwisting"),
    _c("C_hoogsteen_opening_proxy_kT", Resolution.PER_BP, Parity.SYMMETRIC, "kT",
       "harmonic cost of reaching a 60 degree base-pair opening", CGNA,
       provisional=True, reason=_HOOGSTEEN_PROV),
    _c("C_mean_hoogsteen_proxy_kT", Resolution.GLOBAL, Parity.SYMMETRIC, "kT",
       "mean opening proxy", CGNA, provisional=True, reason=_HOOGSTEEN_PROV),
    _c("C_max_hoogsteen_proxy_kT", Resolution.GLOBAL, Parity.SYMMETRIC, "kT",
       "least openable base pair", CGNA, provisional=True, reason=_HOOGSTEEN_PROV),
]

# --- condition axis ----------------------------------------------------------------------
_COND_UNITS = {"dtwist_deg": "degrees", "droll_deg": "degrees", "drise_A": "angstrom",
               "dminor_width_A": "angstrom", "dmajor_width_A": "angstrom"}
_COND_PARITY = {"dtwist_deg": Parity.SYMMETRIC, "droll_deg": Parity.SYMMETRIC,
                "drise_A": Parity.SYMMETRIC, "dminor_width_A": Parity.COMPOSITE,
                "dmajor_width_A": Parity.COMPOSITE}
for _pfx, _cond, _prov in (("meth", "full CpG methylation", False),
                           ("salt", "elevated ionic strength", True),
                           ("temp", "elevated temperature", True)):
    for _k in CONDITION_DELTA_NAMES:
        _SPECS.append(_c(f"C_{_pfx}_{_k}", Resolution.PER_BP, _COND_PARITY[_k],
                         _COND_UNITS[_k], f"change in {_k} under {_cond}",
                         CGNA if not _prov else "", condition=True,
                         provisional=_prov, reason=_NO_PARAMETER_SET if _prov else ""))
_SPECS += [
    _c("C_meth_mean_dtwist_deg", Resolution.GLOBAL, Parity.SYMMETRIC, "degrees",
       "mean twist change under full CpG methylation", CGNA, condition=True),
    _c("C_meth_mean_dminor_width_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       "mean minor groove width change under full CpG methylation", CGNA,
       condition=True),
    _c("C_meth_mean_dmajor_width_A", Resolution.GLOBAL, Parity.SYMMETRIC, "angstrom",
       "mean major groove width change under full CpG methylation", CGNA,
       condition=True),
]

REGISTRY.extend(_SPECS)


# --- measured index maps ------------------------------------------------------------
# Measured on random exact palindromes (results/rc_maps.json, tests/test_rc_maps.py):
#
#   step-indexed, i -> n-2-i: the six inter-base-pair coordinates and every condition
#   delta of them.
#
#   groove channels are exactly SYMMETRIC under a shift of the groove's own phosphate
#   offset, +4 for the minor groove and -4 for the major, holding to 1e-12. They were
#   declared COMPOSITE when the only way to express that was "not a plain sign"; with an
#   explicit index map they are ordinary symmetric channels, and the model can now
#   enforce their equivariance instead of passing them through.
#
#   circular: every angle that can wrap at +-180 degrees. Backbone beta and epsilon sit
#   within a few degrees of the wrap in B-DNA, so a Gaussian on raw degrees would read
#   -179 and +179 as 358 apart.
_STEP_C = [f"C_{c}" for c in INTER_NAMES] + [
    f"C_{p}_{k}" for p in ("meth", "salt", "temp")
    for k in ("dtwist_deg", "droll_deg", "drise_A")]
_GROOVE_C = {}
for _g, _sh in (("minor", +4), ("major", -4)):
    for _q in ("width", "depth"):
        _GROOVE_C[f"C_{_g}_{_q}_A"] = _sh
        _GROOVE_C[f"C_{_g}_{_q}_A_sd"] = _sh
    for _p in ("meth", "salt", "temp"):
        _GROOVE_C[f"C_{_p}_d{_g}_width_A"] = _sh
_CIRC_C = ([f"C_{t}_{tag}" for t in TORSION_NAMES for tag in ("W", "C")]
           + [f"C_mean_{t}" for t in TORSION_NAMES]
           + ["C_eps_minus_zeta_W", "C_eps_minus_zeta_C",
              "C_phosphate_azimuth_deg_W", "C_phosphate_azimuth_deg_C",
              "C_mean_phosphate_azimuth_W", "C_mean_phosphate_azimuth_C",
              "C_axis_bend_direction_deg", "C_mean_bend_direction_deg",
              "C_sugar_pucker_phase_W", "C_sugar_pucker_phase_C"])
from channels import declare_index_maps as _dim  # noqa: E402
_dim(_SPECS, step=_STEP_C, shifts=_GROOVE_C, circular=_CIRC_C,
     parity={n: Parity.SYMMETRIC for n in _GROOVE_C})


# ======================================================================================
# The minimised ensemble: backbone fluctuations, BI/BII populations, pucker states
# ======================================================================================



def minimised_ensemble_stats(seq: str, workdir: str,
                             n_snapshots: int = ENSEMBLE_MIN_SNAPSHOTS,
                             seed: int = 0) -> Dict[str, np.ndarray]:
    """Torsion spreads, BI fraction and pucker-state fractions over relaxed snapshots.

    **Why every snapshot is minimised.** The cgNA+ Gaussian samples rigid bases and rigid
    phosphates independently, and does not constrain the covalent link between them:
    across a raw ensemble the C3'-O3' bond spans 0.64 to 3.44 angstrom, and the torsion
    spread that comes out is over 140 degrees -- an artifact of unconstrained rigid
    bodies, not a fluctuation (see the module docstring). Relaxing each snapshot with the
    same restrained minimisation the ground state gets repairs the junctions while keeping
    each snapshot's sequence-dependent base and phosphate positions, so the spread across
    snapshots is a physical one.

    This is the expensive part of Family C -- two minimisations per snapshot -- so it
    runs on the plan's ensemble subset, "about 10 percent of fragments", and the channels
    are NaN elsewhere, which the training loss masks rather than imputes.

    **What these spreads are, and are not.** Every snapshot is relaxed from the canonical
    alpha/gamma start, so the spreads are fluctuations *within* the dominant backbone
    basin. Rare basin changes -- the alpha/gamma crankshaft transitions molecular
    dynamics sees at a few percent -- are not sampled by minimisation at all, so they are
    absent here rather than present at the wrong rate. Before this, a plain start
    produced alpha and gamma spreads near 60 degrees, which were minimiser traps.

    Minimisation also quenches the backbone's own thermal motion, so what remains is the
    part of each torsion's fluctuation that is slaved to the thermally sampled base and
    phosphate positions. Measured medians on the pilot: alpha 4.2, beta 7.7, gamma 3.0,
    delta 9.9, epsilon 7.3, zeta 10.5 degrees, with epsilon and zeta tails to 29-37 where
    BI/BII switching occurs -- smaller than molecular dynamics standard deviations, and to
    be read for their sequence dependence rather than their absolute size.
    """
    import os
    import shutil

    import cgna_atomic as ca
    import minimize as mz

    seq = seq.upper()
    n = len(seq)
    params = _cgna_params(seq, None, include_stiffness=True)
    stiff = params.stiffmat
    stiff = stiff.toarray() if hasattr(stiff, "toarray") else np.asarray(stiff)
    excess = sample_excess(stiff, n_snapshots, seed)
    templates = ca.NucleotideTemplates()

    tors = {f"{t}_{tag}": [] for t in TORSION_NAMES for tag in ("W", "C")}
    phases = {"W": [], "C": []}
    for k in range(n_snapshots):
        wd = os.path.join(workdir, f"snap{k:02d}")
        os.makedirs(wd, exist_ok=True)
        try:
            gs = ca.CGNAGroundState(seq, excess=excess[k])
            atoms = ca.build_duplex(seq, templates=templates, gs=gs)
            raw, mini = os.path.join(wd, "raw.pdb"), os.path.join(wd, "min.pdb")
            ca.write_pdb(atoms, raw, "PANGEA Family C ensemble snapshot")
            # Canonical start only. The plain start is the one that falls into the
            # alpha/gamma crankshaft trap; on the ground state the canonical start won by
            # 367 and 532 kJ/mol on the test duplexes, and the generator records which
            # start wins for every fragment, so this shortcut is checked on the corpus.
            mz.restrained_minimise(raw, mini, restraint_k=500.0, max_iterations=400,
                                   starts=("canonical",))
            t = torsions_from_structure(mini, n)
            for key in tors:
                tors[key].append(t[key])
            pk = sugar_pucker_from_structure(mini, n)
            for tag in ("W", "C"):
                phases[tag].append(pk[f"phase_{tag}"])
        except Exception:                                          # noqa: BLE001
            continue
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    out: Dict[str, np.ndarray] = {"n_snapshots_ok": np.array(
        float(len(phases["W"])))}
    if len(phases["W"]) < 4:
        return out
    for key, arrs in tors.items():
        a = np.stack(arrs)
        out[f"{key}_sd"] = circ_sd(a, axis=0)
    for tag in ("W", "C"):
        a = np.stack(tors[f"epsilon_{tag}"]) - np.stack(tors[f"zeta_{tag}"])
        d = _wrap180(a)
        with np.errstate(invalid="ignore"):
            frac = np.mean(d < 0.0, axis=0)
        frac[np.all(np.isnan(d), axis=0)] = np.nan
        out[f"BI_fraction_{tag}"] = frac
        P = np.stack(phases[tag])
        fin = np.isfinite(P)
        with np.errstate(invalid="ignore"):
            c2 = np.sum(fin & (P >= 108.0) & (P <= 216.0), axis=0) / fin.sum(axis=0)
            c3 = np.sum(fin & ((P <= 54.0) | (P >= 342.0)), axis=0) / fin.sum(axis=0)
        out[f"C2endo_prob_{tag}"] = c2
        out[f"C3endo_prob_{tag}"] = c3
    return out
