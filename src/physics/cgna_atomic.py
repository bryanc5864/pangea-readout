#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Reconstruct all-atom duplex DNA coordinates from a cgNA+ ground state.

The plan requires that every window handed to the Poisson-Boltzmann solver be built on
a *sequence-dependent* conformation, because an ideal fiber model makes the computed
potential a near-deterministic function of base composition and the whole electrostatics
family collapses into GC content.  cgNA+ supplies that sequence dependence as rigid-body
frames: one SE(3) frame per base and one per phosphate group, on both strands.

This module turns those frames into atoms:

1. Build a per-nucleotide all-atom template expressed in that nucleotide's own base
   frame.  The templates come from the cgNA+ base-pair dictionary (full nucleotides,
   base + sugar + phosphate + hydrogens) and are re-expressed in the base frame by
   inverting the same Kabsch fit cgNA+ uses to read frames out of a PDB.  Using the
   package's own convention is what keeps placement and fitting consistent.
2. Place each template into the cgNA+ base frame for its position.
3. Overwrite the phosphate group atoms (P, OP1, OP2, O5', O3') using the cgNA+
   *phosphate* frames, which carry the sequence-dependent backbone information that a
   base-slaved template cannot.  O3' belongs to the 5' neighbour in the strand.
4. Write a PDB.  Step 3 leaves the sugar slightly inconsistent with the phosphate it
   now connects to; that is repaired by the restrained minimisation in `minimize.py`,
   which restrains bases and phosphates to these positions and lets the sugar relax.

Unit convention and the Crick-frame flip are both *detected at run time* rather than
assumed, and `roundtrip_check` re-fits frames from the built structure and compares
them with the cgNA+ input.  A silent convention error here would corrupt every
downstream channel while still producing plausible-looking numbers.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Locating the vendored cgNA+ package
# --------------------------------------------------------------------------------------

DEFAULT_CGNA_ROOT = "/home/bjc75/scratch/pangea/ext/cgNAplusParams"


def add_cgnaplus_to_path(root: str = DEFAULT_CGNA_ROOT) -> str:
    """Put the vendored cgNA+ checkout on sys.path without running code from its cwd."""
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "cgnaplusparams")):
        raise FileNotFoundError(f"cgNAplusParams not found under {root}")
    for p in (root, os.path.join(root, "cgnaplusparams", "SO3"),
              os.path.join(root, "cgnaplusparams", "pyConDec")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------

CRICK_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
RESNAME = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
# The cgNA+ canonical base definition calls thymine's methyl carbon C5M; the base-pair
# dictionary calls it C7.
ATOM_ALIASES = {"C5M": ("C5M", "C7")}
PHOSPHATE_ATOMS = ("P", "OP1", "OP2", "O5'", "O3'")


def kabsch_se3(ideal: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """SE(3) transform taking *ideal* onto *actual*; both (n,3) in the same units.

    Reimplements cgNA+'s `_kabsch_se3` with numpy only so this module does not depend on
    scipy's Rotation at import time. Returns tau with actual ~= tau[:3,:3] @ ideal + tau[:3,3].
    """
    ideal = np.asarray(ideal, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ci, ca = ideal.mean(axis=0), actual.mean(axis=0)
    H = (ideal - ci).T @ (actual - ca)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    T = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    tau = np.eye(4)
    tau[:3, :3] = T
    tau[:3, 3] = ca - T @ ci
    return tau


def apply_se3(tau: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return pts @ tau[:3, :3].T + tau[:3, 3]


def invert_se3(tau: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = tau[:3, :3].T
    out[:3, 3] = -tau[:3, :3].T @ tau[:3, 3]
    return out


def _match_atoms(ideal: Dict[str, np.ndarray], actual: Dict[str, np.ndarray]):
    """Pair ideal and actual coordinates by atom name, honouring aliases."""
    I, A = [], []
    for name, pos in ideal.items():
        for alias in ATOM_ALIASES.get(name, (name,)):
            if alias in actual:
                I.append(pos)
                A.append(actual[alias])
                break
    return np.asarray(I), np.asarray(A)


# --------------------------------------------------------------------------------------
# Nucleotide templates in the base frame
# --------------------------------------------------------------------------------------

class NucleotideTemplates:
    """All-atom nucleotide templates expressed in each nucleotide's own base frame.

    Coordinates are held in angstrom. `scale` converts to whatever units the cgNA+ poses
    use at placement time.
    """

    def __init__(self, cgna_root: str = DEFAULT_CGNA_ROOT, convention: str = "tsukuba"):
        add_cgnaplus_to_path(cgna_root)
        from cgnaplusparams.input.canonical_definitions import (  # noqa: E402
            TSUKUBA_BASES, CURVESPLUS_BASES, PHOSPHATE,
        )
        self.convention = convention
        self.ideal_bases = {"tsukuba": TSUKUBA_BASES, "curvesplus": CURVESPLUS_BASES}[convention]
        self.phosphate_ideal = {k: np.asarray(v, dtype=float) for k, v in PHOSPHATE.items()}

        bpdict_path = os.path.join(
            cgna_root, "cgnaplusparams", "output", "database", "bpdicts"
        )
        with open(bpdict_path) as f:
            bpdicts = json.load(f)

        self.templates: Dict[str, Dict[str, np.ndarray]] = {}
        self._build_templates(bpdicts)

    def _build_templates(self, bpdicts: dict) -> None:
        """For each base letter, express its full nucleotide in its own base frame."""
        for bp_key, entry in bpdicts.items():
            for side in ("resA", "resB"):
                res = entry[side]
                letter = res["resname"][-1]          # DA -> A
                if letter in self.templates:
                    continue
                atoms = {a["name"]: np.asarray(a["pos"], dtype=float) for a in res["atoms"]}
                ideal = self.ideal_bases.get(letter)
                if ideal is None:
                    continue
                I, A = _match_atoms(ideal, atoms)
                if len(I) < 3:
                    raise RuntimeError(f"too few matching base atoms for {letter}")
                tau = kabsch_se3(I, A)               # base frame of this nucleotide
                inv = invert_se3(tau)
                self.templates[letter] = {
                    name: apply_se3(inv, pos)[0] for name, pos in atoms.items()
                }
        missing = set("ACGT") - set(self.templates)
        if missing:
            raise RuntimeError(f"no template built for {sorted(missing)}")

    def base_atoms(self, letter: str) -> Dict[str, np.ndarray]:
        """Template atoms excluding the phosphate group (those come from phosphate frames)."""
        return {k: v for k, v in self.templates[letter].items() if k not in PHOSPHATE_ATOMS}

    def all_atoms(self, letter: str) -> Dict[str, np.ndarray]:
        return dict(self.templates[letter])


# --------------------------------------------------------------------------------------
# cgNA+ ground state
# --------------------------------------------------------------------------------------

class CGNAGroundState:
    """Thin wrapper over cgNA+ that exposes base and phosphate frames in angstrom.

    Detects whether the underlying package returns translations in nm or angstrom, and
    whether Crick frames are stored with the diag(1,-1,-1) flip applied.
    """

    def __init__(self, sequence: str, cgna_root: str = DEFAULT_CGNA_ROOT,
                 parameter_set: Optional[str] = None,
                 excess: Optional[np.ndarray] = None):
        add_cgnaplus_to_path(cgna_root)
        from cgnaplusparams import CGNAPlusConf, CGNAPlusParams  # noqa: E402

        self.sequence = sequence.upper()
        # CGNAPlusConf requires the aligned-strand convention, in which Crick frames are
        # already rotated to point along the duplex with Watson.  _detect_crick_flip
        # verifies that rather than trusting the flag.
        # `excess` places one snapshot of the Gaussian ensemble instead of the ground
        # state: a tangent-space fluctuation, composed by the library as gs o exp(excess),
        # which is the composition `group_split` implies. Adding it to the coordinates
        # instead would be wrong by an amount that grows with the fluctuation.
        kwargs = {"aligned_strands": True,
                  "include_stiffness": excess is not None}
        if parameter_set is not None:
            kwargs["parameter_set_name"] = parameter_set
        self.params = CGNAPlusParams(self.sequence, **kwargs)
        self.conf = CGNAPlusConf.from_params(self.params, excess=excess)

        wb = np.asarray(self.conf.watson_base_poses, dtype=float)
        cb = np.asarray(self.conf.crick_base_poses, dtype=float)
        wp = np.asarray(self.conf.watson_phosphate_poses, dtype=float)
        cp = np.asarray(self.conf.crick_phosphate_poses, dtype=float)
        # Snapshot axis may be present; take the single ground-state snapshot.
        wb, cb, wp, cp = (self._squeeze(x) for x in (wb, cb, wp, cp))

        self.unit_scale = self._detect_unit_scale(wb)
        for arr in (wb, cb, wp, cp):
            arr[:, :3, 3] *= self.unit_scale

        self.watson_bases, self.crick_bases = wb, cb
        self.watson_phosphates, self.crick_phosphates = wp, cp
        self.crick_flipped = self._detect_crick_flip(wb, cb)

    @staticmethod
    def _squeeze(arr: np.ndarray) -> np.ndarray:
        while arr.ndim > 3:
            arr = arr[0]
        return np.array(arr, dtype=float)

    @staticmethod
    def _detect_unit_scale(watson_bases: np.ndarray) -> float:
        """Return the factor converting pose translations to angstrom.

        Consecutive base frames along a duplex sit about 3.4 angstrom (0.34 nm) apart.
        """
        if len(watson_bases) < 2:
            return 1.0
        d = np.linalg.norm(np.diff(watson_bases[:, :3, 3], axis=0), axis=1)
        med = float(np.median(d))
        if 0.1 < med < 1.0:
            return 10.0          # nanometres
        if 1.0 <= med < 10.0:
            return 1.0           # angstrom
        raise RuntimeError(f"cannot infer units: median base-base spacing {med}")

    @staticmethod
    def _detect_crick_flip(watson_bases: np.ndarray, crick_bases: np.ndarray) -> bool:
        """True when Crick frames are stored already flipped to align with Watson.

        In a Watson-Crick pair the two base planes are nearly antiparallel.  If the
        stored Crick frame has the diag(1,-1,-1) flip applied, its third axis points the
        same way as Watson's; if not, it points opposite.
        """
        n = min(len(watson_bases), len(crick_bases))
        dots = [
            float(np.dot(watson_bases[i, :3, 2], crick_bases[i, :3, 2])) for i in range(n)
        ]
        return float(np.mean(dots)) > 0.0


# --------------------------------------------------------------------------------------
# Structure assembly
# --------------------------------------------------------------------------------------

class Atom:
    __slots__ = ("name", "resname", "chain", "resseq", "xyz", "element")

    def __init__(self, name, resname, chain, resseq, xyz, element):
        self.name, self.resname, self.chain = name, resname, chain
        self.resseq, self.xyz, self.element = resseq, np.asarray(xyz, float), element


def _element_of(atom_name: str) -> str:
    stripped = atom_name.lstrip("0123456789")
    if stripped.startswith(("H",)):
        return "H"
    return stripped[0] if stripped else "C"


def build_duplex(sequence: str,
                 templates: Optional[NucleotideTemplates] = None,
                 gs: Optional[CGNAGroundState] = None,
                 cgna_root: str = DEFAULT_CGNA_ROOT,
                 use_phosphate_frames: bool = True,
                 keep_hydrogens: bool = False) -> List[Atom]:
    """Build an all-atom duplex for *sequence* from its cgNA+ ground state.

    The Watson strand is chain A written 5'->3'; the Crick strand is chain B, also
    written 5'->3', which is the reverse complement.  Terminal 5' phosphates are dropped
    so the structure is a standard 5'-OH duplex.
    """
    sequence = sequence.upper()
    templates = templates or NucleotideTemplates(cgna_root)
    gs = gs or CGNAGroundState(sequence, cgna_root)
    n = len(sequence)

    wb, cb = gs.watson_bases, gs.crick_bases
    wp, cp = gs.watson_phosphates, gs.crick_phosphates
    # Atoms are placed in each base's own (unflipped) frame.  In the Tsukuba standard
    # reference frame the Watson frame and the flipped Crick frame of an ideal
    # Watson-Crick pair nearly coincide, which is what makes the intra-base-pair
    # coordinates small deviations.  So when cgNA+ hands back aligned (flipped) Crick
    # frames, the flip has to be undone before placing atoms; placing a template into
    # the aligned frame stacks the Crick base on top of its own partner.
    if gs.crick_flipped:
        cb = cb @ CRICK_FLIP
        cp = cp @ CRICK_FLIP

    scale = 1.0  # templates and poses are both in angstrom by this point
    atoms: List[Atom] = []

    def place_strand(letters: List[str], base_poses: np.ndarray, phos_poses: np.ndarray,
                     chain: str) -> None:
        """letters, base_poses and phos_poses are all in 5'->3' order for this strand."""
        m = len(letters)
        # Phosphate frame i carries the P that links residue i-1 (O3') to residue i (O5').
        phos_atoms: List[Dict[str, np.ndarray]] = []
        for i in range(m):
            if phos_poses is None or i >= len(phos_poses) or not np.all(np.isfinite(phos_poses[i])):
                phos_atoms.append({})
                continue
            tau = phos_poses[i]
            phos_atoms.append(
                {name: apply_se3(tau, pos * scale)[0]
                 for name, pos in templates.phosphate_ideal.items()}
            )

        for i, letter in enumerate(letters):
            tau = base_poses[i]
            local = templates.all_atoms(letter) if not use_phosphate_frames \
                else templates.base_atoms(letter)
            placed = {name: apply_se3(tau, pos)[0] for name, pos in local.items()}

            if use_phosphate_frames:
                pa = phos_atoms[i] if i < len(phos_atoms) else {}
                if i > 0 and pa:
                    # 5' phosphate of this residue: P, OP1, OP2 and O5' belong here.
                    for nm in ("P", "OP1", "OP2", "O5'"):
                        if nm in pa:
                            placed[nm] = pa[nm]
                elif "O5'" in templates.all_atoms(letter):
                    placed["O5'"] = apply_se3(tau, templates.all_atoms(letter)["O5'"])[0]
                # O3' of residue i is defined by the phosphate frame of residue i+1.
                nxt = phos_atoms[i + 1] if i + 1 < len(phos_atoms) else {}
                if nxt and "O3'" in nxt:
                    placed["O3'"] = nxt["O3'"]
                elif "O3'" in templates.all_atoms(letter):
                    placed["O3'"] = apply_se3(tau, templates.all_atoms(letter)["O3'"])[0]

            for name, xyz in placed.items():
                if not keep_hydrogens and _element_of(name) == "H":
                    continue
                atoms.append(Atom(name, RESNAME[letter], chain, i + 1, xyz, _element_of(name)))

    watson_letters = list(sequence)
    place_strand(watson_letters, wb, wp, "A")

    # Crick strand: cgNA+ indexes it against the Watson positions (3'->5' along Watson),
    # so reverse to get 5'->3' for the Crick chain itself.
    crick_letters = [COMPLEMENT[c] for c in sequence][::-1]
    place_strand(crick_letters, cb[::-1], cp[::-1], "B")
    return atoms


def write_pdb(atoms: List[Atom], path: str, remark: str = "") -> str:
    lines = []
    if remark:
        for line in remark.splitlines():
            lines.append(f"REMARK   1 {line[:68]}")
    serial = 0
    last_chain = None
    for a in atoms:
        if last_chain is not None and a.chain != last_chain:
            serial += 1
            lines.append(f"TER   {serial:5d}")
        serial += 1
        name = a.name
        # PDB atom-name column rules: elements of one letter start in column 14.
        if len(name) < 4 and len(a.element) == 1:
            name = f" {name:<3s}"
        else:
            name = f"{name:<4s}"
        x, y, z = a.xyz
        # PDB columns: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc, 18-20 resName,
        # 22 chainID, 23-26 resSeq, 27 iCode, 31-38/39-46/47-54 xyz, 77-78 element.
        lines.append(
            f"ATOM  {serial:5d} {name} {a.resname:>3s} {a.chain}{a.resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {a.element:>2s}"
        )
        last_chain = a.chain
    serial += 1
    lines.append(f"TER   {serial:5d}")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------

def geometry_report(atoms: List[Atom]) -> Dict[str, float]:
    """Bond-length sanity checks that catch a frame-convention error immediately."""
    by_res: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for a in atoms:
        by_res.setdefault((a.chain, a.resseq), {})[a.name] = a.xyz

    po5, po3, c1n = [], [], []
    for (chain, i), res in by_res.items():
        if "P" in res and "O5'" in res:
            po5.append(np.linalg.norm(res["P"] - res["O5'"]))
        nxt = by_res.get((chain, i + 1))
        if nxt and "O3'" in res and "P" in nxt:
            po3.append(np.linalg.norm(res["O3'"] - nxt["P"]))
        gly = "N9" if "N9" in res else ("N1" if "N1" in res else None)
        if gly and "C1'" in res:
            c1n.append(np.linalg.norm(res["C1'"] - res[gly]))

    def stat(v, label):
        return {} if not v else {
            f"{label}_mean": float(np.mean(v)),
            f"{label}_sd": float(np.std(v)),
            f"{label}_min": float(np.min(v)),
            f"{label}_max": float(np.max(v)),
        }

    out = {"n_atoms": float(len(atoms)), "n_residues": float(len(by_res))}
    out.update(stat(po5, "P_O5prime"))
    out.update(stat(po3, "O3prime_P"))
    out.update(stat(c1n, "C1prime_N_glycosidic"))
    return out


def base_pair_report(atoms: List[Atom], n_bp: int) -> Dict[str, float]:
    """Watson-Crick pair geometry, the sharpest test of the frame conventions.

    Chain A residue i pairs with chain B residue n_bp+1-i.  In B-DNA the C1'-C1'
    distance across a pair is close to 10.5 angstrom and the N1-N9 distance close to
    8.9 angstrom.  If the Crick frame flip is handled the wrong way round the two bases
    land on top of each other and these distances collapse toward zero, which no
    bond-length check would catch.
    """
    by_res: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    resname: Dict[Tuple[str, int], str] = {}
    for a in atoms:
        by_res.setdefault((a.chain, a.resseq), {})[a.name] = a.xyz
        resname[(a.chain, a.resseq)] = a.resname

    c1c1, nn = [], []
    for i in range(1, n_bp + 1):
        ra = by_res.get(("A", i))
        rb = by_res.get(("B", n_bp + 1 - i))
        if not ra or not rb:
            continue
        if "C1'" in ra and "C1'" in rb:
            c1c1.append(np.linalg.norm(ra["C1'"] - rb["C1'"]))
        ga = "N9" if resname[("A", i)] in ("DA", "DG") else "N1"
        gb = "N9" if resname[("B", n_bp + 1 - i)] in ("DA", "DG") else "N1"
        if ga in ra and gb in rb:
            nn.append(np.linalg.norm(ra[ga] - rb[gb]))

    out: Dict[str, float] = {"n_pairs_checked": float(len(c1c1))}
    if c1c1:
        out.update({
            "C1prime_C1prime_mean": float(np.mean(c1c1)),
            "C1prime_C1prime_sd": float(np.std(c1c1)),
            "C1prime_C1prime_min": float(np.min(c1c1)),
            "C1prime_C1prime_max": float(np.max(c1c1)),
        })
    if nn:
        out.update({
            "glycosidic_N_N_mean": float(np.mean(nn)),
            "glycosidic_N_N_sd": float(np.std(nn)),
        })
    return out


def min_heavy_distance(atoms: List[Atom]) -> float:
    """Smallest distance between atoms of different residues, a clash detector."""
    xyz = np.array([a.xyz for a in atoms])
    key = np.array([hash((a.chain, a.resseq)) for a in atoms])
    best = np.inf
    for i in range(0, len(xyz), 256):
        block = xyz[i:i + 256]
        d = np.linalg.norm(block[:, None, :] - xyz[None, :, :], axis=-1)
        same = key[i:i + 256][:, None] == key[None, :]
        d[same] = np.inf
        best = min(best, float(d.min()))
    return best


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", default="GCGCAAATTTGCGCATATAGCGCAAATTTGCGC")
    ap.add_argument("--out", default="/tmp/cgna_test.pdb")
    ap.add_argument("--cgna-root", default=DEFAULT_CGNA_ROOT)
    ap.add_argument("--keep-hydrogens", action="store_true")
    a = ap.parse_args()

    gs = CGNAGroundState(a.sequence, a.cgna_root)
    print(f"sequence      : {a.sequence} ({len(a.sequence)} bp)")
    print(f"unit scale    : {gs.unit_scale} (pose translations -> angstrom)")
    print(f"crick flipped : {gs.crick_flipped}")
    tmpl = NucleotideTemplates(a.cgna_root)
    print(f"templates     : {sorted(tmpl.templates)} "
          f"({{{', '.join(f'{k}:{len(v)}' for k, v in sorted(tmpl.templates.items()))}}} atoms)")
    atoms = build_duplex(a.sequence, tmpl, gs, keep_hydrogens=a.keep_hydrogens)
    write_pdb(atoms, a.out, remark=f"cgNA+ ground state for {a.sequence}")
    rep = geometry_report(atoms)
    rep.update(base_pair_report(atoms, len(a.sequence)))
    for k, v in rep.items():
        print(f"  {k:28s} {v:10.3f}")
    print(f"  {'min_interresidue_distance':28s} {min_heavy_distance(atoms):10.3f}")
    print(f"wrote {a.out}")
