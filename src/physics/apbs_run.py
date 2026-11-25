#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Poisson-Boltzmann ground truth for a DNA fragment: PQR, APBS, cylindrical sampling.

Pipeline for one fragment, following the plan's electrostatics specification:

  PDB (cgNA+ ground state, minimised)
    -> PDB2PQR with an AMBER parameter set          -> .pqr
    -> APBS nonlinear Poisson-Boltzmann, two-level focusing
         coarse grid near 1.0 A over the fragment plus solvent padding
         fine grid near 0.35 A over the central 21 bp
    -> potential and mobile-ion density maps (.dx)
    -> sampling on cylindrical shells in the base-pair frame
         6 angular sectors x 6 radial offsets, per base pair
    -> counterion atmosphere by integrating ion densities per phosphate

Only the central 21 bp of a 61 bp fragment are scored, because fragment ends carry large
spurious potentials and a model trained on them learns edge artifacts.

Ionic conditions are defined in IONIC_CONDITIONS.  Magnesium is treated at mean-field
level, which is known to be poor for divalents (ion size, ion-ion correlation and
site-specific binding are all outside the theory), so those channels are labelled
approximate and are excluded from any claim that needs quantitative divalent accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------


@dataclass
class Ion:
    charge: float
    concentration_M: float
    radius_A: float


@dataclass
class IonicCondition:
    name: str
    ions: List[Ion]
    temperature_K: float = 298.15
    approximate: bool = False       # True for divalent conditions
    note: str = ""
    # APBS builds one ion-accessible region from the LARGEST ion radius in the input, so
    # with chloride listed every condition used 1.67 A and KCl was numerically identical
    # to NaCl at the same concentration (solvation energy equal to 0.01 kJ/mol, DECISIONS
    # 2026-11-21). When set, every ion line is written with this radius: the radius of
    # the condition's counterion, the ion the atmosphere next to a polyanion is made of.
    accessibility_radius_A: Optional[float] = None

    @property
    def ion_accessibility_radius_A(self) -> float:
        """The single radius APBS uses for this condition's ion-accessible region."""
        if self.accessibility_radius_A is not None:
            return float(self.accessibility_radius_A)
        return max(float(i.radius_A) for i in self.ions)

    @property
    def ionic_strength_M(self) -> float:
        return 0.5 * sum(i.concentration_M * i.charge**2 for i in self.ions)


# Pauling/Shannon-like radii as used in APBS examples: Na+ 1.16, K+ 1.52, Mg2+ 0.86,
# Cl- 1.67 angstrom.
def _nacl(c: float, T: float = 298.15) -> List[Ion]:
    return [Ion(+1.0, c, 1.16), Ion(-1.0, c, 1.67)]


IONIC_CONDITIONS: List[IonicCondition] = [
    IonicCondition("NaCl_010mM", _nacl(0.010)),
    IonicCondition("NaCl_025mM", _nacl(0.025)),
    IonicCondition("NaCl_050mM", _nacl(0.050)),
    IonicCondition("NaCl_100mM", _nacl(0.100)),
    IonicCondition("NaCl_150mM", _nacl(0.150), note="physiological reference"),
    IonicCondition("NaCl_250mM", _nacl(0.250)),
    IonicCondition("NaCl_500mM", _nacl(0.500)),
    IonicCondition(
        "KCl_150mM",
        [Ion(+1.0, 0.150, 1.52), Ion(-1.0, 0.150, 1.67)],
        note="cation identity axis: K+ against Na+ at matched ionic strength",
    ),
    IonicCondition(
        "NaCl_150mM_MgCl2_2mM",
        [Ion(+1.0, 0.150, 1.16), Ion(+2.0, 0.002, 0.86), Ion(-1.0, 0.154, 1.67)],
        approximate=True,
        note="mean-field PB treats divalents poorly; approximate, excluded from "
             "quantitative divalent claims",
    ),
]

# Production ion exclusion radius: the conventional ~2 A Stern layer of nucleic-acid PB work
# (APBS and DelPhi practice) for the NaCl series and the Na/Mg mixture, and 2.36 A for KCl
# -- the Shannon K+ - Na+ radius difference (0.36 A) added -- so KCl differs from NaCl 150 mM
# in ion size only and the mixture in valence only. The bare Na+ radius (1.16 A) used first
# let ions approach closer than mean-field PB supports and moved the minor-groove potential
# 1 A out from -4.22 to -3.12 kT/e (DECISIONS.md, 2026-11-21). A copy, so every earlier study
# stays reproducible from IONIC_CONDITIONS.
PRODUCTION_ACCESSIBILITY_A = {"KCl_150mM": 2.36}
PRODUCTION_DEFAULT_ACCESSIBILITY_A = 2.0
PRODUCTION_CONDITIONS: List[IonicCondition] = [
    IonicCondition(c.name, c.ions, c.temperature_K,
                   approximate=c.approximate or c.name == "KCl_150mM",
                   note=c.note + ("; ion identity enters mean-field PB only through the "
                                  "exclusion radius, so approximate" if c.name == "KCl_150mM" else ""),
                   accessibility_radius_A=PRODUCTION_ACCESSIBILITY_A.get(c.name, PRODUCTION_DEFAULT_ACCESSIBILITY_A))
    for c in IONIC_CONDITIONS]
PRODUCTION_CONDITIONS_BY_NAME = {c.name: c for c in PRODUCTION_CONDITIONS}
CONDITIONS_BY_NAME = {c.name: c for c in IONIC_CONDITIONS}


def debye_length_A(ionic_strength_M: float, temperature_K: float = 298.15) -> float:
    """Debye screening length in angstrom for a monovalent-dominated aqueous solution.

    lambda_D = 3.04 A / sqrt(I[M]) at 25 C.  The coarse-grid padding has to be a fixed
    number of screening lengths, not a fixed distance: at 150 mM a 50 A pad is 6.4
    screening lengths and the solve converges, while at 25 mM the same 50 A is only 2.6
    and the nonlinear solver aborts.
    """
    if ionic_strength_M <= 0:
        return float("inf")
    base = 3.04 / (ionic_strength_M ** 0.5)
    return base * (temperature_K / 298.15) ** 0.5

# Sampling geometry
N_ANGULAR_SECTORS = 6
RADIAL_OFFSETS_A = (5.0, 7.0, 9.0, 11.0, 14.0, 18.0)
SCORED_BP = 21          # central base pairs scored per fragment
FRAGMENT_BP = 61        # fragment length


# --------------------------------------------------------------------------------------
# External tools
# --------------------------------------------------------------------------------------


def which_or_die(prog: str) -> str:
    p = shutil.which(prog)
    if p is None:
        raise RuntimeError(f"{prog} not found on PATH")
    return p


def run_pdb2pqr(pdb_in: str, pqr_out: str, forcefield: str = "AMBER",
                keep_chain: bool = True, log: Optional[str] = None) -> str:
    """Assign AMBER charges and radii. DNA has no titratable groups at neutral pH, so
    no pKa calculation is requested; hydrogens are added by the force field rules."""
    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    if exe is None:
        raise RuntimeError("pdb2pqr not found on PATH")
    cmd = [exe, f"--ff={forcefield}", "--nodebump", "--noopt"]
    if keep_chain:
        cmd.append("--keep-chain")
    cmd += [pdb_in, pqr_out]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if log:
        with open(log, "w") as f:
            f.write(res.stdout + "\n" + res.stderr)
    if res.returncode != 0 or not os.path.exists(pqr_out):
        raise RuntimeError(f"pdb2pqr failed ({res.returncode}): {res.stderr[-2000:]}")
    return pqr_out


def sanitise_pqr(path: str, min_radius_A: float = 1.0,
                 out_path: Optional[str] = None) -> dict:
    """Give every atom a positive radius, and report what had to be changed.

    PDB2PQR's AMBER parameter set assigns radius 0.000 to the terminal hydroxyl
    hydrogens H5T and H3T.  A charged atom with zero radius sits inside the solute with
    no dielectric boundary around it, which makes the potential singular at that point;
    the nonlinear solver then reports "Detected large potential values" after trapping
    millions of exponential overflows, and aborts.  Terminal hydroxyls are far outside
    the scored central 21 bp, so assigning them a small physical hydrogen radius cannot
    affect the scored channels, but leaving them at zero makes the solve fail outright.

    The count and identity of the patched atoms are returned so the change is auditable
    rather than silent.
    """
    out_path = out_path or path
    lines, patched = [], []
    n_rewritten = 0
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # APBS reads PQR by whitespace, but PDB2PQR writes coordinates in fixed 8.3f
                # columns: a coordinate <= -100.000 fills its column and fuses with the one
                # before it ("Valist_readPQR: Error parsing atom", seen on bent cgNA+
                # ensemble snapshots). Coordinates are therefore re-read from their columns
                # (31-54, as PDB2PQR writes them) and every line is rewritten with explicit
                # separators.
                fixed = _respace_pqr_line(line)
                if fixed is not None and fixed != line:
                    n_rewritten += 1
                    line = fixed
                parts = line.split()
                try:
                    radius = float(parts[-1])
                except (ValueError, IndexError):
                    lines.append(line)
                    continue
                if radius <= 0.0:
                    name = parts[2] if len(parts) > 2 else "?"
                    patched.append(name)
                    tail = line.rstrip("\n")
                    idx = tail.rfind(parts[-1])
                    line = tail[:idx] + f"{min_radius_A:.4f}" + "\n"
            lines.append(line)
    with open(out_path, "w") as f:
        f.writelines(lines)
    from collections import Counter

    return {
        "n_zero_radius_patched": len(patched),
        "patched_atom_names": dict(Counter(patched)),
        "min_radius_assigned_A": min_radius_A,
        "n_lines_respaced": n_rewritten,
    }


def _respace_pqr_line(line: str) -> Optional[str]:
    """A PDB2PQR ATOM line rewritten with whitespace between every field, coordinates read
    from columns 31-54. Returns None if the line does not have that layout."""
    try:
        head = line[:30]
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        q, r = line[54:].split()[:2]
        q, r = float(q), float(r)
    except (ValueError, IndexError):
        return None
    rec, serial = head[:6].strip(), head[6:11].strip()
    name, resn = head[12:16].strip(), head[17:20].strip()
    chain, resseq = head[21:22].strip(), head[22:26].strip()
    fields = [rec, serial, name, resn] + ([chain] if chain else []) + [resseq]
    return (" ".join(fields) + f" {x:.3f} {y:.3f} {z:.3f} {q:.4f} {r:.4f}\n")


def read_pqr(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Return coordinates (A), charges (e), radii (A) and per-atom records."""
    xyz, q, r, recs = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            # PQR is whitespace-delimited after the field columns; parse from the right
            # because atom/residue names can run together in fixed-column files.
            parts = line.split()
            try:
                charge, radius = float(parts[-2]), float(parts[-1])
                z, y, x = float(parts[-3]), float(parts[-4]), float(parts[-5])
            except (ValueError, IndexError):
                continue
            xyz.append((x, y, z))
            q.append(charge)
            r.append(radius)
            rec = {
                "name": parts[2] if len(parts) > 2 else "",
                "resname": parts[3] if len(parts) > 3 else "",
            }
            # pdb2pqr keeps the chain ID: record, serial, name, resname, chain, resseq,
            # x, y, z, q, r (11 fields); without a chain there are 10.
            if len(parts) == 11:
                rec["chain"], rec["resseq"] = parts[4], parts[5]
            elif len(parts) == 10:
                rec["chain"], rec["resseq"] = "", parts[4]
            recs.append(rec)
    return np.asarray(xyz), np.asarray(q), np.asarray(r), recs


# --------------------------------------------------------------------------------------
# APBS input generation
# --------------------------------------------------------------------------------------


def _grid_dims(length_A: float, spacing_A: float) -> int:
    """Smallest APBS-legal dimension (c*2^(l+1)+1, l=4) covering length at spacing."""
    n_needed = int(np.ceil(length_A / spacing_A)) + 1
    n = 33
    while n < n_needed:
        # legal dims: 2^(l+1)*c + 1 with l=4 -> 32c+1
        n += 32
    return n


def grid_spec(coarse_extent_A, fine_extent_A, fine_spacing_A: float = 0.35,
              coarse_spacing_A: float = 1.0, exact_fine_spacing: bool = False):
    """(cglen, fglen, dime) exactly as `write_apbs_input` writes them.

    `dime` serves both focusing levels, so when the coarse box needs more points than the
    fine box, the fine level silently becomes finer than `fine_spacing_A` (0.31 instead of
    0.34 A along the axis for a 61 bp solute at 150 mM; about 0.2 A at 10 mM). Labels at
    different conditions or solute lengths then carry different discretisations, which a
    model would learn as salt dependence, and the convergence study compared lengths at
    different resolutions (DECISIONS.md, 2026-11-21). `exact_fine_spacing=True` widens the
    fine box instead, to (dime - 1) * fine_spacing_A (never beyond the coarse box), so the
    fine level is solved at the specified spacing on every axis.
    """
    cg = [max(32.0, e) for e in coarse_extent_A]
    fg = [max(16.0, e) for e in fine_extent_A]
    dime = [max(_grid_dims(fg[i], fine_spacing_A),
                _grid_dims(cg[i], coarse_spacing_A)) for i in range(3)]
    if exact_fine_spacing:
        fg = [min(cg[i], (dime[i] - 1) * fine_spacing_A) for i in range(3)]
    return cg, fg, dime


# Coarse-level spacing; 1.0 A in production. Overridable for solver studies only
# (PANGEA_COARSE_SPACING_A), e.g. the long-solute high-salt divergence (DECISIONS.md).
COARSE_SPACING_A = float(os.environ.get("PANGEA_COARSE_SPACING_A", "1.0"))


def write_apbs_input(
    pqr_basename: str,
    name: str,
    workdir: str,
    condition: IonicCondition,
    center: Sequence[float],
    coarse_extent_A: Sequence[float],
    fine_extent_A: Sequence[float],
    fine_spacing_A: float = 0.35,
    coarse_spacing_A: Optional[float] = None,
    write_maps: bool = True,
    bcfl: str = "sdh",
    solute_dielectric: float = 2.0,
    solvent_dielectric: float = 78.54,
    sdie_probe_A: float = 1.4,
    reference: bool = False,
    exact_fine_spacing: bool = False,
    per_atom_energies: bool = False,
    calc_energy: bool = True,
    reference_maps: bool = False,
    reference_equation: str = "lpbe",
    write_ndens: bool = True,
) -> str:
    """Write a focused nonlinear PB input file (mg-auto, npbe) into *workdir*.

    `reference=True` writes the matching reference calculation for a solvation free
    energy instead: the same grid (dime, cglen, fglen, centre, charge discretisation),
    the solute in a uniform dielectric equal to its own, no mobile ions, linearised
    equation, no maps. G(solvated) - G(reference) on identical grids cancels the grid
    self-energy, which is the standard APBS construction of a solvation energy.

    `reference_equation="npbe"` states the same reference problem (no mobile ions, so the
    equation is linear) to APBS's Newton multigrid path instead of its linear one. The
    solution is the same -- identical to the printed digits on a 129 x 65 x 65 test grid --
    but the linear path solves its coarsest level with a banded direct factorisation whose
    cost grows as (nx ny)^2 nz: on a cruciform's 1089 x 289 x 257 grid (coarsest level
    137 x 37 x 33) that took over an hour and 27 GB, against minutes for the Newton path.
    `write_ndens=False` leaves out the ion-density map for callers that never read it.

    `mg-auto` runs the coarse and focused fine levels itself from `cglen` and `fglen`,
    which is both the supported way to focus and much cheaper than forcing one shared
    `dime` across two hand-written levels.

    Grid dimensions are chosen per axis, anisotropically, so a 61 bp duplex -- roughly
    200 A long and 20 A wide -- does not pay for a cubic mesh. Demanding 0.35 A over the
    long axis costs about 44 million points for no benefit; the scored region is the
    central 21 bp out to an 18 A radius.

    **`dime` has to satisfy both levels, and originally it did not.** `mg-auto` uses one
    `dime` for the coarse level and the focused fine level alike, so sizing it from the
    fine box alone silently coarsens the coarse grid as the fragment grows:

        fragment   coarse box   coarse spacing (fine-sized dime)
          21 bp      191 A          0.60 A
          61 bp      327 A          1.02 A
          93 bp      436 A          1.36 A
         121 bp      531 A          1.66 A

    `PLAN.md` specifies "a coarse grid near 1 angstrom over the full fragment plus
    solvent padding", so everything past about 61 bp was off spec. It showed up as
    solver failures rather than as quiet inaccuracy: the plan-mandated convergence study
    solved 121 bp fragments fine at 10 mM and aborted on every one of them at 50 mM and
    100 mM with an overflow. The nonlinear Boltzmann source term grows with ionic
    strength, and on a 1.7 A grid next to a phosphate it overflows. A 93 bp solve had
    failed the same way earlier and been recorded as an open item without the cause
    being found.

    So the dimension per axis is now the larger of what the fine box needs at
    `fine_spacing_A` and what the coarse box needs at `coarse_spacing_A`, which makes
    the grid obey the plan at every fragment length instead of only short ones.

    All paths written into the file are basenames, because APBS is invoked with
    `cwd=workdir`.
    """
    if coarse_spacing_A is None:
        coarse_spacing_A = COARSE_SPACING_A
    cg, fg, dime = grid_spec(coarse_extent_A, fine_extent_A, fine_spacing_A, coarse_spacing_A,
                             exact_fine_spacing=exact_fine_spacing)

    ion_lines = "\n".join(
        f"    ion charge {i.charge:+.1f} conc {i.concentration_M:.6f} "
        f"radius {condition.ion_accessibility_radius_A if condition.accessibility_radius_A is not None else i.radius_A:.3f}"
        for i in condition.ions
    )
    equation = "npbe"
    if reference:
        if reference_equation not in ("lpbe", "npbe"):
            raise ValueError(f"reference_equation must be lpbe or npbe, not {reference_equation!r}")
        ion_lines, equation, write_maps = "", reference_equation, False
        solvent_dielectric = solute_dielectric
    writes = ""
    if write_maps:
        writes = f"    write pot dx {name}_pot\n" + (f"    write ndens dx {name}_ndens\n" if write_ndens else "")
    elif reference and reference_maps:
        # the reference potential, for solvation from the maps (1/2 q (phi_solv - phi_ref)
        # at each atom) where APBS's own energy evaluation fails
        writes = f"    write pot dx {name}_pot\n"

    body = f"""read
    mol pqr {pqr_basename}
end

# Focused nonlinear Poisson-Boltzmann.
# Coarse level spans the whole fragment plus solvent padding; the fine level covers the
# central scored region at {fine_spacing_A} A and takes its boundary from the coarse solution.
elec name fine
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {cg[0]:.3f} {cg[1]:.3f} {cg[2]:.3f}
    fglen {fg[0]:.3f} {fg[1]:.3f} {fg[2]:.3f}
    cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    mol 1
    {equation}
    bcfl {bcfl}
{ion_lines}
    pdie {solute_dielectric:.3f}
    sdie {solvent_dielectric:.3f}
    srfm smol
    chgm spl2
    sdens 10.0
    srad {sdie_probe_A:.3f}
    swin 0.3
    temp {condition.temperature_K:.2f}
    calcenergy {("comps" if per_atom_energies else "total") if calc_energy else "no"}
    calcforce no
{writes}end

print elecEnergy fine end
quit
"""
    path = os.path.join(workdir, name + ".in")
    with open(path, "w") as f:
        f.write(body)
    return path


def run_apbs(apbs_in: str, workdir: str, log: Optional[str] = None,
             timeout_s: int = 3600) -> str:
    exe = which_or_die("apbs")
    res = subprocess.run(
        [exe, os.path.basename(apbs_in)], cwd=workdir,
        capture_output=True, text=True, timeout=timeout_s,
    )
    out = res.stdout + "\n" + res.stderr
    if log:
        with open(log, "w") as f:
            f.write(out)
    if res.returncode != 0:
        raise RuntimeError(f"apbs failed ({res.returncode}): {out[-3000:]}")
    return out


def parse_atom_energies(apbs_stdout: str) -> Optional[np.ndarray]:
    """Per-atom energies (kJ/mol, PQR order) of the last level that printed them.

    With `calcenergy comps` APBS reports each atom's share of the electrostatic energy,
    1/2 q_i phi(r_i). Differences between a solvated and a reference solve on the same grid
    give each atom's share of the solvation energy, 1/2 q_i phi_reaction(r_i), which can be
    summed over any region -- the scored base pairs -- without the atoms that sit next to
    the focusing boundary, whose values APBS itself flags as unreliable.
    """
    blocks = apbs_stdout.split("Per-atom energies:")
    if len(blocks) < 2:
        return None
    vals = re.findall(r"^\s*Atom\s+\d+:\s*([-\d.Ee+]+)\s*kJ/mol", blocks[-1], flags=re.M)
    return np.asarray([float(v) for v in vals], dtype=float) if vals else None


def parse_elec_energy(apbs_stdout: str) -> Optional[float]:
    m = re.findall(r"Global net ELEC energy\s*=\s*([-\d.Ee+]+)\s*kJ/mol", apbs_stdout)
    if m:
        return float(m[-1])
    m = re.findall(r"Total electrostatic energy\s*=\s*([-\d.Ee+]+)\s*kJ/mol", apbs_stdout)
    return float(m[-1]) if m else None


# --------------------------------------------------------------------------------------
# OpenDX maps
# --------------------------------------------------------------------------------------


class DXMap:
    """A uniform-grid OpenDX scalar field with trilinear interpolation."""

    def __init__(self, origin: np.ndarray, delta: np.ndarray, data: np.ndarray):
        self.origin, self.delta, self.data = origin, delta, data
        self.shape = np.array(data.shape)

    @classmethod
    def read(cls, path: str) -> "DXMap":
        origin = delta = None
        deltas, counts, values = [], None, []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("#") or not s:
                    continue
                if s.startswith("object 1"):
                    counts = [int(x) for x in s.split()[-3:]]
                elif s.startswith("origin"):
                    origin = np.array([float(x) for x in s.split()[1:4]])
                elif s.startswith("delta"):
                    deltas.append([float(x) for x in s.split()[1:4]])
                elif s.startswith(("object", "attribute", "component")):
                    continue
                else:
                    values.extend(float(x) for x in s.split())
        if counts is None or origin is None or len(deltas) < 3:
            raise ValueError(f"malformed dx file: {path}")
        delta = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])
        data = np.asarray(values, dtype=float)
        n = int(np.prod(counts))
        if data.size < n:
            raise ValueError(f"dx file truncated: {data.size} < {n}")
        return cls(origin, delta, data[:n].reshape(counts))

    def interpolate(self, pts: np.ndarray) -> np.ndarray:
        """Trilinear interpolation; points outside the grid return NaN."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        g = (pts - self.origin) / self.delta
        i0 = np.floor(g).astype(int)
        f = g - i0
        out = np.full(len(pts), np.nan)
        ok = np.all((i0 >= 0) & (i0 < self.shape - 1), axis=1)
        if not ok.any():
            return out
        i0o, fo = i0[ok], f[ok]
        acc = np.zeros(len(i0o))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (
                        (fo[:, 0] if dx else 1 - fo[:, 0])
                        * (fo[:, 1] if dy else 1 - fo[:, 1])
                        * (fo[:, 2] if dz else 1 - fo[:, 2])
                    )
                    acc += w * self.data[
                        i0o[:, 0] + dx, i0o[:, 1] + dy, i0o[:, 2] + dz
                    ]
        out[ok] = acc
        return out


# --------------------------------------------------------------------------------------
# Cylindrical sampling in the base-pair frame
# --------------------------------------------------------------------------------------


def shell_points(bp_frames: np.ndarray,
                 radial_offsets_A: Sequence[float] = RADIAL_OFFSETS_A,
                 n_sectors: int = N_ANGULAR_SECTORS) -> Tuple[np.ndarray, np.ndarray]:
    """Sampling points on cylindrical shells around each base pair.

    Angles are measured in the base-pair frame's x-y plane, with 0 along +x.  In the
    Tsukuba/3DNA convention +x points into the major groove, so sector 0 samples the
    major groove and sector 3 (180 degrees) the minor groove.  Tying the angle to the
    base-pair frame rather than a laboratory axis is what makes the channel meaningful
    along a curved, sequence-dependent duplex.

    Returns points with shape (n_bp, n_sectors, n_radii, 3) and the angles used.
    """
    angles = np.arange(n_sectors) * (2 * np.pi / n_sectors)
    radii = np.asarray(radial_offsets_A, dtype=float)
    n_bp = len(bp_frames)
    pts = np.zeros((n_bp, n_sectors, len(radii), 3))
    for i in range(n_bp):
        tau = bp_frames[i]
        T, o = tau[:3, :3], tau[:3, 3]
        ex, ey = T[:, 0], T[:, 1]
        for a, ang in enumerate(angles):
            direction = np.cos(ang) * ex + np.sin(ang) * ey
            for r, rad in enumerate(radii):
                pts[i, a, r] = o + rad * direction
    return pts, angles


def solvent_accessible_mask(points: np.ndarray, atom_xyz: np.ndarray,
                            atom_radii: np.ndarray, probe_A: float = 1.4,
                            block: int = 4096) -> np.ndarray:
    """True where a point lies in solvent, outside every atom's probe-inflated radius.

    Cylindrical shells at fixed radii from the helical axis unavoidably intersect the
    sugar-phosphate backbone: the phosphates sit near 9 A, which is one of the sampling
    radii, and at the two backbone sectors a sample can land on a phosphorus atom.
    Inside the solute the potential is dominated by the near-singular field of a buried
    point charge in a dielectric of 2, which produced -161 kT/e at r = 9 A and even
    changed sign at r = 5 A.  Those values describe the grid, not the molecule.

    Masking them is not a cosmetic choice: a channel that mixes solvent potential with
    buried self-energy is not the physical observable the plan defines, and its variance
    would be dominated by whether a sample happened to fall inside an atom.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    ok = np.ones(len(pts), dtype=bool)
    inflated = np.asarray(atom_radii, dtype=float) + probe_A
    for i in range(0, len(pts), block):
        chunk = pts[i:i + block]
        d = np.linalg.norm(chunk[:, None, :] - atom_xyz[None, :, :], axis=-1)
        ok[i:i + block] = np.all(d > inflated[None, :], axis=1)
    return ok.reshape(np.asarray(points).shape[:-1])


def sample_potential(pot: DXMap, bp_frames: np.ndarray,
                     radial_offsets_A: Sequence[float] = RADIAL_OFFSETS_A,
                     n_sectors: int = N_ANGULAR_SECTORS,
                     atom_xyz: Optional[np.ndarray] = None,
                     atom_radii: Optional[np.ndarray] = None,
                     probe_A: float = 1.4) -> Dict[str, np.ndarray]:
    """Potential on cylindrical shells, with buried sample points masked to NaN.

    When atom coordinates and radii are supplied, points that are not solvent accessible
    are set to NaN and a boolean validity mask is returned alongside.  The mask is part
    of the channel definition: it is sequence dependent, because which sectors the
    backbone occupies depends on the local geometry, so downstream code must carry it
    rather than silently averaging over NaNs.
    """
    pts, angles = shell_points(bp_frames, radial_offsets_A, n_sectors)
    flat = pts.reshape(-1, 3)
    vals = pot.interpolate(flat).reshape(pts.shape[:3])
    out: Dict[str, np.ndarray] = {"points": pts, "angles": angles, "potential": vals}
    if atom_xyz is not None and atom_radii is not None:
        mask = solvent_accessible_mask(pts, atom_xyz, atom_radii, probe_A)
        vals = np.where(mask, vals, np.nan)
        out["potential"] = vals
        out["solvent_accessible"] = mask
        out["accessible_fraction"] = float(mask.mean())
    return out


SURFACE_OFFSETS_A = (0.0, 1.0, 2.0, 4.0, 7.0, 11.0)


def surface_referenced_points(bp_frames: np.ndarray, atom_xyz: np.ndarray,
                              atom_radii: np.ndarray,
                              offsets_A: Sequence[float] = SURFACE_OFFSETS_A,
                              n_sectors: int = N_ANGULAR_SECTORS,
                              probe_A: float = 1.4,
                              march_max_A: float = 40.0,
                              step_A: float = 0.25
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sampling points at fixed offsets outward from the molecular surface.

    Why this exists alongside the axis-referenced shells: a cylindrical shell at a fixed
    radius from the helical axis passes through the sugar-phosphate backbone at the two
    backbone sectors, so with the buried points correctly masked only about 59 percent of
    the plan's 6 sector by 6 radius grid is defined, and whole channels are empty for
    every sequence.  A channel that is structurally undefined cannot be predicted or
    compared across species.

    Marching outward along each sector direction to the first solvent-accessible point
    gives a surface crossing, and sampling at fixed offsets beyond it yields a channel
    that is always defined and means the same thing for every sequence: "the potential
    this far outside the molecular surface, in this groove".  The surface radius itself
    is returned, because it is a shape observable in its own right - it is the local
    groove depth seen along that direction.
    """
    angles = np.arange(n_sectors) * (2 * np.pi / n_sectors)
    offsets = np.asarray(offsets_A, dtype=float)
    n_bp = len(bp_frames)
    pts = np.full((n_bp, n_sectors, len(offsets), 3), np.nan)
    surface_r = np.full((n_bp, n_sectors), np.nan)

    march = np.arange(0.0, march_max_A, step_A)
    inflated = np.asarray(atom_radii, dtype=float) + probe_A

    for i in range(n_bp):
        tau = bp_frames[i]
        T, o = tau[:3, :3], tau[:3, 3]
        ex, ey = T[:, 0], T[:, 1]
        for a, ang in enumerate(angles):
            direction = np.cos(ang) * ex + np.sin(ang) * ey
            probe_pts = o[None, :] + march[:, None] * direction[None, :]
            d = np.linalg.norm(probe_pts[:, None, :] - atom_xyz[None, :, :], axis=-1)
            free = np.all(d > inflated[None, :], axis=1)
            # First radius from which the ray stays in solvent all the way out.
            idx = None
            for k in range(len(march)):
                if free[k:].all():
                    idx = k
                    break
            if idx is None:
                continue
            r0 = float(march[idx])
            surface_r[i, a] = r0
            pts[i, a] = o[None, :] + (r0 + offsets)[:, None] * direction[None, :]
    return pts, surface_r, angles


def sample_potential_surface(pot: DXMap, bp_frames: np.ndarray,
                             atom_xyz: np.ndarray, atom_radii: np.ndarray,
                             offsets_A: Sequence[float] = SURFACE_OFFSETS_A,
                             n_sectors: int = N_ANGULAR_SECTORS,
                             probe_A: float = 1.4) -> Dict[str, np.ndarray]:
    pts, surface_r, angles = surface_referenced_points(
        bp_frames, atom_xyz, atom_radii, offsets_A, n_sectors, probe_A)
    flat = pts.reshape(-1, 3)
    ok = np.all(np.isfinite(flat), axis=1)
    vals = np.full(len(flat), np.nan)
    if ok.any():
        vals[ok] = pot.interpolate(flat[ok])
    vals = vals.reshape(pts.shape[:3])
    return {
        "potential_surface": vals,
        "surface_radius_A": surface_r,
        "offsets_A": np.asarray(offsets_A, dtype=float),
        "angles": angles,
        "defined_fraction": float(np.isfinite(vals).mean()),
    }


def counterion_atmosphere(ndens: DXMap, phosphate_xyz: np.ndarray,
                          radius_A: float = 7.0, n_shell: int = 200,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Mean mobile-ion number density in a sphere around each phosphate.

    A full Manning-style condensed fraction needs a volume integral of the excess ion
    density; this reports the sampled mean density within `radius_A`, which is the
    quantity the integral is built from and is what the encoder is asked to predict.
    """
    rng = rng or np.random.default_rng(0)
    # Low-discrepancy-ish sphere sample, fixed seed so the channel is reproducible.
    u = rng.uniform(-1, 1, n_shell)
    theta = rng.uniform(0, 2 * np.pi, n_shell)
    rad = radius_A * rng.uniform(0, 1, n_shell) ** (1 / 3)
    s = np.sqrt(1 - u**2)
    offs = np.stack([rad * s * np.cos(theta), rad * s * np.sin(theta), rad * u], axis=1)
    out = np.zeros(len(phosphate_xyz))
    for i, p in enumerate(phosphate_xyz):
        vals = ndens.interpolate(p[None, :] + offs)
        out[i] = np.nanmean(vals)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-conditions", action="store_true")
    a = ap.parse_args()
    if a.list_conditions:
        for c in IONIC_CONDITIONS:
            flag = " [approximate]" if c.approximate else ""
            print(f"{c.name:24s} I={c.ionic_strength_M*1000:7.2f} mM  T={c.temperature_K:.2f}K{flag}")
            if c.note:
                print(f"{'':24s} {c.note}")
