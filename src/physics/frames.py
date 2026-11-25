#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Base and base-pair reference frames read back out of an all-atom structure.

Cylindrical sampling has to be done in the base-pair frame of the structure the solver
actually saw, not in the cgNA+ frames the structure was built from, because the
restrained minimisation moves atoms.  This module re-fits base frames from a PDB using
the same canonical base geometry cgNA+ uses, then builds the mid-frame for each pair.

The mid-frame construction is the standard one: origin at the midpoint of the two base
origins, orientation halfway between the two triads along the geodesic in SO(3), with
the Crick triad flipped by diag(1,-1,-1) so both point along the duplex.  In that frame
+x points into the major groove and -x into the minor groove, which is what makes an
angular sector index physically meaningful.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgna_atomic import (  # noqa: E402
    CRICK_FLIP, add_cgnaplus_to_path, apply_se3, invert_se3, kabsch_se3,
)

RESNAME_TO_LETTER = {"DA": "A", "DT": "T", "DG": "G", "DC": "C",
                     "DA5": "A", "DA3": "A", "DT5": "T", "DT3": "T",
                     "DG5": "G", "DG3": "G", "DC5": "C", "DC3": "C",
                     # ribonucleotides (Family K's A-form classes); uracil is fitted on the
                     # thymine ideal base, whose methyl it lacks and which the fit then skips
                     "A": "A", "C": "C", "G": "G", "U": "T",
                     "A5": "A", "A3": "A", "C5": "C", "C3": "C", "G5": "G", "G3": "G",
                     "U5": "T", "U3": "T", "RA": "A", "RC": "C", "RG": "G", "RU": "T"}
ATOM_ALIASES = {"C5M": ("C5M", "C7")}


def read_pdb_residues(path: str) -> Dict[Tuple[str, int], Dict[str, np.ndarray]]:
    """{(chain, resseq): {atom_name: xyz}} in angstrom, from a PDB file."""
    out: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    resname: Dict[Tuple[str, int], str] = {}
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            rn = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                seq = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            key = (chain, seq)
            out.setdefault(key, {})[name] = np.array([x, y, z])
            resname[key] = rn
    for key in out:
        out[key]["_resname"] = resname[key]  # type: ignore[assignment]
    return out


def fit_base_frames(residues: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                    chain: str,
                    convention: str = "tsukuba",
                    cgna_root: Optional[str] = None) -> Tuple[List[int], np.ndarray]:
    """Fit a base frame per residue of *chain*. Returns (resseq list, (n,4,4) frames)."""
    add_cgnaplus_to_path(cgna_root) if cgna_root else add_cgnaplus_to_path()
    from cgnaplusparams.input.canonical_definitions import (  # noqa: E402
        TSUKUBA_BASES, CURVESPLUS_BASES,
    )
    ideal_all = {"tsukuba": TSUKUBA_BASES, "curvesplus": CURVESPLUS_BASES}[convention]

    keys = sorted([k for k in residues if k[0] == chain], key=lambda k: k[1])
    seqs, frames = [], []
    for key in keys:
        res = residues[key]
        letter = RESNAME_TO_LETTER.get(str(res.get("_resname", "")), None)
        if letter is None:
            continue
        ideal = ideal_all.get(letter)
        if ideal is None:
            continue
        I, A = [], []
        for nm, pos in ideal.items():
            for alias in ATOM_ALIASES.get(nm, (nm,)):
                if alias in res:
                    I.append(pos)
                    A.append(res[alias])
                    break
        if len(I) < 3:
            continue
        seqs.append(key[1])
        frames.append(kabsch_se3(np.asarray(I), np.asarray(A)))
    return seqs, np.asarray(frames) if frames else np.zeros((0, 4, 4))


def _rotation_geodesic_mid(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation halfway from Ra to Rb along the SO(3) geodesic."""
    R = Ra.T @ Rb
    # Principal matrix square root of a rotation via axis-angle.
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return Ra.copy()
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        # theta near pi: recover the axis from the symmetric part.
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = axis / norm
    h = theta / 2.0
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_half = np.eye(3) + np.sin(h) * K + (1 - np.cos(h)) * (K @ K)
    return Ra @ R_half


def base_pair_frames(watson_frames: np.ndarray, crick_frames: np.ndarray,
                     crick_is_flipped: bool = False) -> np.ndarray:
    """Mid-frames for paired bases. watson_frames[i] pairs with crick_frames[i].

    crick_frames must already be ordered to pair positionally with watson_frames.
    """
    n = min(len(watson_frames), len(crick_frames))
    out = np.zeros((n, 4, 4))
    for i in range(n):
        tw = watson_frames[i]
        tc = crick_frames[i] if crick_is_flipped else crick_frames[i] @ CRICK_FLIP
        R = _rotation_geodesic_mid(tw[:3, :3], tc[:3, :3])
        out[i] = np.eye(4)
        out[i][:3, :3] = R
        out[i][:3, 3] = 0.5 * (tw[:3, 3] + tc[:3, 3])
    return out


def bp_frames_from_pdb(path: str, convention: str = "tsukuba"
                       ) -> Tuple[np.ndarray, List[int]]:
    """Base-pair frames for a two-chain duplex PDB written by cgna_atomic.write_pdb.

    Chain A is the Watson strand 5'->3'; chain B is the Crick strand 5'->3', so Crick
    residue j pairs with Watson residue n+1-j.
    """
    residues = read_pdb_residues(path)
    w_seqs, w_frames = fit_base_frames(residues, "A", convention)
    c_seqs, c_frames = fit_base_frames(residues, "B", convention)
    if len(w_frames) == 0 or len(c_frames) == 0:
        raise RuntimeError(f"no base frames fitted from {path}")
    # Pair Watson i (0-based) with Crick reversed i.
    c_rev = c_frames[::-1]
    n = min(len(w_frames), len(c_rev))
    bp = base_pair_frames(w_frames[:n], c_rev[:n], crick_is_flipped=False)
    return bp, w_seqs[:n]


def helical_axis_points(bp_frames: np.ndarray) -> np.ndarray:
    """Base-pair origins, which trace the (curved) helical axis."""
    return bp_frames[:, :3, 3]


def groove_direction_check(bp_frames: np.ndarray,
                           residues: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                           chain_a: str = "A", chain_b: str = "B") -> Dict[str, float]:
    """Verify that +x of the base-pair frame points into the major groove.

    The phosphate groups of the two strands sit on the minor-groove side of the base
    pair.  Projecting the vector from the base-pair origin to the mean phosphate
    position onto the frame's x axis must therefore give a negative number.  A positive
    value means the frame convention is inverted and every angular sector index would be
    rotated by 180 degrees, which would silently swap the groove channels.
    """
    import numpy as _np

    keys_a = sorted([k for k in residues if k[0] == chain_a], key=lambda k: k[1])
    keys_b = sorted([k for k in residues if k[0] == chain_b], key=lambda k: k[1])
    n = min(len(bp_frames), len(keys_a), len(keys_b))
    projections = []
    for i in range(n):
        ra = residues[keys_a[i]]
        rb = residues[keys_b[len(keys_b) - 1 - i]]
        ps = [r["P"] for r in (ra, rb) if "P" in r]
        if not ps:
            continue
        mid_p = _np.mean(ps, axis=0)
        tau = bp_frames[i]
        v = mid_p - tau[:3, 3]
        projections.append(float(_np.dot(v, tau[:3, 0])))
    if not projections:
        return {}
    arr = _np.asarray(projections)
    return {
        "mean_x_projection_of_phosphates": float(arr.mean()),
        "fraction_negative": float((arr < 0).mean()),
        "major_groove_is_plus_x": bool(arr.mean() < 0),
    }
