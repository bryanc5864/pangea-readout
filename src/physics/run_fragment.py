#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""End-to-end electrostatics ground truth for one DNA fragment.

    sequence
      -> cgNA+ sequence-dependent ground state        (cgna_atomic)
      -> all-atom duplex                              (cgna_atomic)
      -> restrained minimisation, AMBER OL15          (minimize)
      -> base-pair frames re-fitted from the minimised structure  (frames)
      -> PQR with AMBER charges and radii             (apbs_run)
      -> nonlinear Poisson-Boltzmann, two-level focusing, per ionic condition (apbs_run)
      -> potential on cylindrical shells + counterion atmosphere, central 21 bp only

Writes one .npz of channels plus a .json record per fragment.

`--grid-jitter` re-solves the same fragment with the grid centre displaced by a fraction
of a grid spacing.  The spread across jitters is the solver's own numerical noise, and
the composition gate needs it: "the potential separates composition-matched sequences"
only means something measured against that floor.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apbs_run  # noqa: E402
import cgna_atomic  # noqa: E402
import frames as frames_mod  # noqa: E402
import electrostatics_channels as ec_mod  # noqa: E402

SCORED_BP = apbs_run.SCORED_BP
RADIAL_OFFSETS_A = apbs_run.RADIAL_OFFSETS_A

# Production solver geometry, validated in results/solver_probe_v1.json and
# results/salt_padding_probe.json.
#
# Coarse-grid padding. The floor of 50 A is what solver_probe_v1 verified at 61 bp;
# 30 A aborts. results/salt_padding_probe.json then checked whether low salt needs more,
# by sweeping padding in Debye screening lengths at 25 and 10 mM: every configuration
# converged, including 50 A at 25 mM, so salt is NOT the constraint the earlier job
# failures suggested (those used 30 A, not 50 A -- see DECISIONS.md). The mild Debye
# scaling is kept only so that every production condition sits at or above a padding
# that was actually tested: 4.0 lambda_D gives 122 A at 10 mM and 77 A at 25 mM, both
# verified, and falls below the 50 A floor from 100 mM upward, where 50 A is verified.
MIN_TRANSVERSE_PAD_A = 50.0
N_DEBYE_PAD = 4.0  # see DECISIONS.md; not Debye physics, just a tested-envelope floor
AXIAL_PAD_A = 40.0
FINE_SPACING_A = 0.35


def central_slice(n_bp: int, scored: int = SCORED_BP) -> slice:
    """Indices of the central `scored` base pairs of an n_bp fragment."""
    if n_bp <= scored:
        return slice(0, n_bp)
    start = (n_bp - scored) // 2
    return slice(start, start + scored)


def perturb_pdb(pdb_in: str, pdb_out: str, sd_A: float, seed: int) -> None:
    """Copy a PDB with every coordinate displaced by N(0, sd_A).

    Used only to measure how sensitive the minimised structure is to tiny differences in
    its input -- the kind rounding produces on different hardware. Fixed-column PDB
    format, so the 0.001 A precision of the file is the smallest perturbation that
    registers.
    """
    rng = np.random.default_rng(seed)
    out = []
    with open(pdb_in) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                xyz = xyz + rng.normal(0.0, sd_A, 3)
                line = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
            out.append(line)
    with open(pdb_out, "w") as f:
        f.writelines(out)


def trim_pdb(pdb_in: str, pdb_out: str, n_bp: int, solute_bp: int) -> str:
    """Write the central `solute_bp` base pairs of an n_bp duplex PDB, coordinates untouched.

    Watson (chain A) residues lo+1..lo+solute_bp and their Crick (chain B) partners
    n+1-j; the phosphate at each new 5' end is dropped so the cut ends are 5'-OH termini,
    like every built fragment's. No re-minimisation: the kept atoms keep exactly the
    conformation they had in full context.
    """
    lo = (n_bp - solute_bp) // 2
    hi = lo + solute_bp
    keep_w = set(range(lo + 1, hi + 1))
    keep_c = set(range(n_bp + 1 - hi, n_bp + 1 - lo))
    first = {"A": lo + 1, "B": n_bp + 1 - hi}
    out = []
    with open(pdb_in) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain, res = line[21], int(line[22:26])
                name = line[12:16].strip()
                if (chain == "A" and res not in keep_w) or (chain == "B" and res not in keep_c):
                    continue
                if res == first.get(chain) and name in ("P", "OP1", "OP2"):
                    continue
                out.append(line)
            elif line.startswith(("TER", "END")):
                continue
    out.append("END\n")
    with open(pdb_out, "w") as f:
        f.writelines(out)
    return pdb_out


def ensemble_excess(sequence: str, snapshot: int, seed_key: str, n_snapshots: int = 8):
    """Tangent displacement of one cgNA+ Gaussian snapshot (1-based), seeded by `seed_key`.

    PLAN.md: "For about 10 percent of fragments, draw 8 configurations from the cgNA+
    Gaussian and solve each, supervising both the ensemble mean and the spread." The
    snapshots are delta ~ N(0, K^-1) from the fragment's own stiffness, composed by cgNA+
    as gs o exp(delta) (geometry.sample_excess); the seed is a hash of the fragment id, so a
    fragment's eight configurations are the same on every rerun.
    """
    import hashlib
    import geometry as G
    pr = G._cgna_params(sequence, include_stiffness=True)
    K = pr.stiffmat
    K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
    seed = int(hashlib.sha1(f"ensemble|{seed_key}".encode()).hexdigest()[:12], 16)
    return G.sample_excess(K, n_snapshots, seed=seed)[int(snapshot) - 1]


def build_structure(sequence: str, workdir: str, restraint_k: float = 500.0,
                    minimise: bool = True, cgna_root: str = cgna_atomic.DEFAULT_CGNA_ROOT,
                    min_iterations: int = 200, min_tolerance: Optional[float] = None,
                    perturb_A: float = 0.0, perturb_seed: int = 0,
                    snapshot: int = 0, snapshot_key: str = ""
                    ) -> Dict[str, object]:
    """cgNA+ ground state (or one ensemble snapshot) -> all-atom PDB -> restrained minimisation."""
    raw_pdb = os.path.join(workdir, "raw.pdb")
    excess = ensemble_excess(sequence, snapshot, snapshot_key or sequence) if snapshot > 0 else None
    gs = cgna_atomic.CGNAGroundState(sequence, cgna_root, excess=excess)
    templates = cgna_atomic.NucleotideTemplates(cgna_root)
    atoms = cgna_atomic.build_duplex(sequence, templates, gs)
    cgna_atomic.write_pdb(atoms, raw_pdb, remark=f"cgNA+ ground state {sequence}")
    if perturb_A > 0:
        perturb_pdb(raw_pdb, raw_pdb, perturb_A, perturb_seed)
    report = {
        "unit_scale": gs.unit_scale,
        "crick_flipped": bool(gs.crick_flipped),
        "geometry_raw": cgna_atomic.geometry_report(atoms),
    }
    pdb = raw_pdb
    if minimise:
        import minimize as minimise_mod

        min_pdb = os.path.join(workdir, "min.pdb")
        report["minimisation"] = minimise_mod.restrained_minimise(
            raw_pdb, min_pdb, restraint_k=restraint_k, max_iterations=min_iterations,
            tolerance=min_tolerance,
        )
        pdb = min_pdb
    report["perturb_A"] = perturb_A
    report["snapshot"] = int(snapshot)
    report["pdb"] = pdb
    return report


def _solve_by_length(pdb: str, workdir: str, conditions, bp_frames: np.ndarray, scored: slice,
                     solute_bp_by_condition: Optional[Dict[str, int]] = None, **kw) -> Dict[str, object]:
    """`solve_conditions` with a solute length per condition (the plan's screening-length rule).

    The structure is built and minimised once, at full length; a condition with a shorter
    solute is solved on that structure trimmed (`trim_pdb`), so every condition scores the
    same physical base pairs in the same conformation. Outputs are merged back into the
    order of `conditions`; scalar bookkeeping comes from the full-length group if any.
    """
    n = len(bp_frames)
    want = {c.name: int((solute_bp_by_condition or {}).get(c.name, n)) for c in conditions}
    groups: Dict[int, list] = {}
    for c in conditions:
        groups.setdefault(min(want[c.name], n), []).append(c)
    if len(groups) == 1 and n in groups:
        return solve_conditions(pdb, workdir, conditions, bp_frames, scored, **kw)
    parts = {}
    for L, conds in sorted(groups.items(), reverse=True):
        sub = os.path.join(workdir, f"L{L}")
        os.makedirs(sub, exist_ok=True)
        pdb_L = pdb if L >= n else trim_pdb(pdb, os.path.join(sub, f"solute_L{L}.pdb"), n, L)
        parts[L] = (conds, solve_conditions(pdb_L, sub, conds, bp_frames, scored, **kw))
    order = {c.name: i for i, c in enumerate(conditions)}
    first = parts[max(parts)][1]
    out: Dict[str, object] = {k: v for k, v in first.items() if not isinstance(v, np.ndarray)
                              and k not in ("conditions", "elec_energy_kJ_mol", "coarse_box_A",
                                            "debye_length_A", "transverse_pad_A")}
    per_cond = {}
    for L, (conds, o) in parts.items():
        for j, c in enumerate(conds):
            per_cond[c.name] = (o, j)
    names = [c.name for c in conditions]
    out["conditions"] = names
    stack_keys = [k for k, v in first.items() if isinstance(v, np.ndarray) and v.ndim >= 1
                  and len(v) == len(parts[max(parts)][0]) and k not in ("angles", "surface_offsets_A")]
    full_P = np.asarray(first.get("counterion", np.zeros((1, 0)))).shape[-1]
    for k in stack_keys:
        rows = []
        for nm in names:
            v = np.asarray(per_cond[nm][0][k])[per_cond[nm][1]]
            if k == "counterion" and v.shape[-1] != full_P:
                # per-phosphate, and a trimmed solute has fewer phosphates: NaN, not a
                # misaligned row. Family A's per-strand charge comes from the phosphate
                # cells (ext_condensed_fraction), which are mapped by base pair.
                v = np.full(full_P, np.nan)
            rows.append(v)
        out[k] = np.stack(rows)
    for k in ("angles", "surface_offsets_A"):
        if k in first:
            out[k] = first[k]
    for k in ("coarse_box_A", "debye_length_A", "transverse_pad_A"):
        out[k] = [per_cond[nm][0].get(k, [None] * 99)[per_cond[nm][1]] for nm in names]
    for k in ("accessible_fraction", "surface_defined_fraction"):
        out[k] = float(np.mean([parts[L][1][k] for L in parts]))
    for k in ("ext_manning_xi", "ext_surface_charge_e_per_A2", "ext_rise_A"):
        if k in first:
            out[k] = first[k]                       # geometry only: the full-length group's
    return out


def solve_conditions(pdb: str, workdir: str, conditions: Sequence[apbs_run.IonicCondition],
                     bp_frames: np.ndarray, scored: slice,
                     grid_jitter: float = 0.0, jitter_seed: int = 0,
                     keep_maps: bool = False, extended: bool = False,
                     reference_solve: bool = False,
                     exact_fine_spacing: bool = False,
                     pqr_in: Optional[str] = None,
                     grid_extent: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """PQR once, then one nonlinear PB solve per ionic condition.

    `pqr_in`: use this PQR instead of running pdb2pqr on `pdb` (a molecule with non-standard
    residues, e.g. Family N's modified nucleotides); `pdb` still supplies nothing else. `grid_extent`:
    size the coarse grid from this molecular extent instead of the PQR's own, so two closely
    related molecules (modified and unmodified) are solved on identical grids (the grids are centred
    on the scored base pairs, which `bp_frames` fixes)."""
    pqr = os.path.join(workdir, "frag.pqr")
    if pqr_in:
        shutil.copy(pqr_in, pqr)
    else:
        apbs_run.run_pdb2pqr(pdb, pqr, log=os.path.join(workdir, "pdb2pqr.log"))
    sanitation = apbs_run.sanitise_pqr(pqr)
    xyz, q, r, recs = apbs_run.read_pqr(pqr)

    centre = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    if grid_extent is not None:
        extent = np.asarray(grid_extent, float)
    # Coarse-grid padding, chosen from the measured sweep in results/solver_probe_v1.json
    # rather than from a rule of thumb.  The coarse box carries the Debye-Huckel
    # boundary condition; a 61 bp duplex carries -120 e and its potential decays slowly.
    # At 30 A of transverse padding the 61 bp solve aborts outright at both grid
    # spacings ("Detected large potential values", after trapping millions of Boltzmann
    # exponential overflows), and even where it does converge it is roughly nine times
    # slower.  At 50 A it converges in about 35 s and the sampled channels are unchanged
    # (rmse 0.0005 of a channel standard deviation against the 30 A result at 41 bp).
    # The long axis needs less padding per unit length because the duplex is already
    # long there.
    long_axis = int(np.argmax(extent))
    scored_pts = bp_frames[scored][:, :3, 3]
    fine_centre = scored_pts.mean(axis=0) if len(scored_pts) else centre
    # The fine box only has to contain the scored base pairs out to the largest sampling
    # radius, with a margin so the focused boundary is not sitting on sampled points.
    margin = 2.0 * (max(RADIAL_OFFSETS_A) + 6.0)
    fine_base = (scored_pts.max(axis=0) - scored_pts.min(axis=0) + margin) \
        if len(scored_pts) else extent + margin

    if grid_jitter:
        rng = np.random.default_rng(jitter_seed)
        fine_centre = fine_centre + rng.uniform(-grid_jitter, grid_jitter, 3)

    # atoms of the scored base pairs (both nucleotides), for per-base-pair solvation
    _owner = {}
    _nbp = len(bp_frames)
    for _i, _rec in enumerate(recs):
        _ch, _rs = _rec.get("chain", ""), _rec.get("resseq", "")
        try:
            _rs = int(_rs)
        except (TypeError, ValueError):
            continue
        _b = _rs - 1 if _ch == "A" else (_nbp - _rs if _ch == "B" else None)
        if _b is not None and scored.start <= _b < scored.stop:
            _owner.setdefault(_b - scored.start, []).append(_i)
    scored_atoms = np.array(sorted(i for v in _owner.values() for i in v), dtype=int)
    _pos = {a: j for j, a in enumerate(scored_atoms)}
    scored_owner = {k: np.array([_pos[i] for i in v], dtype=int) for k, v in _owner.items()}

    phosphate_idx = [i for i, rec in enumerate(recs) if rec["name"] == "P"]
    phosphate_xyz = xyz[phosphate_idx] if phosphate_idx else np.zeros((0, 3))

    out: Dict[str, object] = {"n_atoms": int(len(xyz)),
                              "total_charge": float(q.sum()),
                              "n_phosphates": int(len(phosphate_xyz)),
                              "pqr_sanitation": sanitation,
                              "min_radius_A": float(r.min()),
                              "fine_box_A": [float(x) for x in fine_base],
                              "coarse_box_A": []}
    pot_stack, ndens_stack, energies, mask_stack = [], [], [], []
    surf_stack, surf_r_stack = [], []
    ext_stack: Dict[str, dict] = {}            # key -> {condition index: value}
    cell_cache = ec_mod.CellCache()
    ref_by_grid: Dict[tuple, Optional[float]] = {}
    ref_energies: List[Optional[float]] = []
    failed_conditions: Dict[str, str] = {}
    energy_skipped: List[str] = []
    meta_by_cond: Dict[int, tuple] = {}
    for ci, cond in enumerate(conditions):
      try:
          lam = apbs_run.debye_length_A(cond.ionic_strength_M, cond.temperature_K)
          tpad = max(MIN_TRANSVERSE_PAD_A, N_DEBYE_PAD * lam)
          pad = np.full(3, tpad)
          pad[long_axis] = AXIAL_PAD_A
          coarse = extent + 2.0 * pad
          fine = np.minimum(fine_base, coarse)
          prefix = os.path.join(workdir, cond.name)
          apbs_in = apbs_run.write_apbs_input(
              os.path.basename(pqr), cond.name, workdir, cond,
              fine_centre, coarse, fine, fine_spacing_A=FINE_SPACING_A,
              exact_fine_spacing=exact_fine_spacing, per_atom_energies=reference_solve,
          )
          energy_ok = True
          try:
              stdout = apbs_run.run_apbs(apbs_in, workdir,
                                         log=os.path.join(workdir, f"{cond.name}.log"))
          except RuntimeError as exc:
              # The mobile-ion term of the energy evaluation overflows on some 121 bp low-salt
              # and divalent solves ("Detected large potential values in energy evaluation")
              # after the potential itself has been solved; the potential is kept by solving
              # again without the energy (DECISIONS.md, 2026-11-21), and that condition's
              # solvation is NaN rather than the whole fragment being lost.
              if "energy evaluation" not in str(exc):
                  raise
              apbs_in = apbs_run.write_apbs_input(
                  os.path.basename(pqr), cond.name, workdir, cond,
                  fine_centre, coarse, fine, fine_spacing_A=FINE_SPACING_A,
                  exact_fine_spacing=exact_fine_spacing, calc_energy=False)
              stdout = apbs_run.run_apbs(apbs_in, workdir,
                                         log=os.path.join(workdir, f"{cond.name}.log"))
              energy_ok = False
              energy_skipped.append(cond.name)
          energies.append(apbs_run.parse_elec_energy(stdout) if energy_ok else None)
          atom_E = apbs_run.parse_atom_energies(stdout) if (reference_solve and energy_ok) else None
          meta_by_cond[ci] = ([round(float(x), 1) for x in coarse], round(lam, 2), round(tpad, 1))

          pot = apbs_run.DXMap.read(prefix + "_pot.dx")
          sampled = apbs_run.sample_potential(
              pot, bp_frames[scored], atom_xyz=xyz, atom_radii=r,
          )
          pot_stack.append(sampled["potential"])
          mask_stack.append(sampled["solvent_accessible"])

          surf = apbs_run.sample_potential_surface(pot, bp_frames[scored], xyz, r)
          surf_stack.append(surf["potential_surface"])
          surf_r_stack.append(surf["surface_radius_A"])

          if extended:
              ext = ec_mod.extract_condition(pot, bp_frames[scored], surf["surface_radius_A"],
                                             xyz, r, recs, cond, len(bp_frames), scored,
                                             cell_cache)
              for k, v in ext.items():
                  ext_stack.setdefault(k, {})[ci] = v
          if reference_solve:
              # One reference per distinct grid: the self-energy it cancels is a property of
              # the grid, which conditions from 50 mM up share (DECISIONS.md, Family A).
              _, fg_used, dime = apbs_run.grid_spec(coarse, fine, FINE_SPACING_A,
                                                    exact_fine_spacing=exact_fine_spacing)
              gkey = tuple(dime) + tuple(np.round(fg_used, 3))
              if gkey not in ref_by_grid:
                  ref_in = apbs_run.write_apbs_input(
                      os.path.basename(pqr), cond.name + "_ref", workdir, cond,
                      fine_centre, coarse, fine, fine_spacing_A=FINE_SPACING_A, reference=True,
                      exact_fine_spacing=exact_fine_spacing, per_atom_energies=True,
                      reference_maps=True)
                  try:
                      so = apbs_run.run_apbs(ref_in, workdir,
                                             log=os.path.join(workdir, f"{cond.name}_ref.log"))
                      rp = os.path.join(workdir, cond.name + "_ref_pot.dx")
                      phi_ref = apbs_run.DXMap.read(rp).interpolate(xyz[scored_atoms]) if os.path.exists(rp) else None
                      if os.path.exists(rp):
                          os.remove(rp)
                      ref_by_grid[gkey] = (apbs_run.parse_elec_energy(so), apbs_run.parse_atom_energies(so), phi_ref)
                  except RuntimeError:
                      ref_by_grid[gkey] = (None, None, None)
              ref_energies.append(ref_by_grid[gkey][0])
              # Charge-potential solvation energy, 1/2 q_i (phi_solv - phi_ref) at each atom,
              # summed per scored base pair (both nucleotides, all atoms). The nonlinear
              # total energy is NOT used: wherever the fine box cuts the solute, phosphates
              # sit on the focusing boundary and the functional's ionic term overflows
              # (96 percent of 61 bp fragments at 150 mM; DECISIONS.md, 2026-11-21). The
              # scored atoms are far from that boundary. For NPBE this form omits the mobile
              # ions' entropic term, as MM-PBSA does; recorded as approximate.
              ra = ref_by_grid[gkey][1]
              bp_solv = np.full(scored.stop - scored.start, np.nan)
              if atom_E is not None and ra is not None and len(atom_E) == len(ra) == len(recs):
                  dE = atom_E - ra
                  nbp = len(bp_frames)
                  owner = {}
                  for i, rec in enumerate(recs):
                      ch, rs = rec.get("chain", ""), rec.get("resseq", "")
                      try:
                          rs = int(rs)
                      except (TypeError, ValueError):
                          continue
                      b = rs - 1 if ch == "A" else (nbp - rs if ch == "B" else None)
                      if b is not None and scored.start <= b < scored.stop:
                          owner.setdefault(b - scored.start, []).append(i)
                  for k, idx in owner.items():
                      bp_solv[k] = float(dE[idx].sum())
              ext_stack.setdefault("bp_solvation_kJ_mol", {})[ci] = bp_solv
              # The same charge-potential solvation from the maps: 1/2 q_i (phi_solv - phi_ref)
              # at each scored atom, both potentials on the same grid (the grid self-energy
              # cancels as in APBS's own per-atom form). Defined where the energy evaluation
              # fails (121 bp at 10-25 mM, the Na/Mg mixture), and beside it elsewhere so the
              # two forms are cross-checked on every fragment.
              phi_ref = ref_by_grid[gkey][2]
              bp_map = np.full(scored.stop - scored.start, np.nan)
              if phi_ref is not None and len(scored_atoms):
                  phi_s = pot.interpolate(xyz[scored_atoms])
                  kT_kJ = 8.314462618e-3 * cond.temperature_K
                  dG = 0.5 * q[scored_atoms] * (phi_s - phi_ref) * kT_kJ
                  for k, idx_local in scored_owner.items():
                      v = dG[idx_local]
                      if np.isfinite(v).all():
                          bp_map[k] = float(v.sum())
              ext_stack.setdefault("bp_solvation_map_kJ_mol", {})[ci] = bp_map

          ndens_path = prefix + "_ndens.dx"
          if os.path.exists(ndens_path) and len(phosphate_xyz):
              nd = apbs_run.DXMap.read(ndens_path)
              ndens_stack.append(apbs_run.counterion_atmosphere(nd, phosphate_xyz))
          else:
              ndens_stack.append(np.full(len(phosphate_xyz), np.nan))

          if not keep_maps:
              for suffix in ("_pot.dx", "_ndens.dx", "_charge.dx"):
                  p = prefix + suffix
                  if os.path.exists(p):
                      os.remove(p)

      except RuntimeError as exc:
        # a condition that cannot be solved at all leaves NaN, the others are kept
        failed_conditions[cond.name] = str(exc)[-300:]
        for st in (pot_stack, mask_stack, surf_stack, surf_r_stack, ndens_stack):
            st.append(None)
        energies.append(None)
        if reference_solve:
            ref_energies.append(None)

    if failed_conditions and len(failed_conditions) == len(conditions):
        raise RuntimeError("every condition failed: " + "; ".join(failed_conditions.values())[:1000])

    def _fill(stack):
        """Replace failed-condition placeholders by NaN shaped like a solved condition."""
        tmpl = next((x for x in stack if x is not None), None)
        if tmpl is None:
            return stack
        return [np.full(np.shape(tmpl), np.nan) if x is None else x for x in stack]
    pot_stack, mask_stack = _fill(pot_stack), _fill(mask_stack)
    surf_stack, surf_r_stack, ndens_stack = _fill(surf_stack), _fill(surf_r_stack), _fill(ndens_stack)
    for k in list(ext_stack):
        ext_stack[k] = _fill([ext_stack[k].get(i) for i in range(len(conditions))])
    out["coarse_box_A"] = [meta_by_cond.get(i, (None,))[0] for i in range(len(conditions))]
    out["debye_length_A"] = [meta_by_cond[i][1] if i in meta_by_cond else None for i in range(len(conditions))]
    out["transverse_pad_A"] = [meta_by_cond[i][2] if i in meta_by_cond else None for i in range(len(conditions))]
    out["failed_conditions"] = failed_conditions
    out["energy_skipped_conditions"] = energy_skipped
    out["conditions"] = [c.name for c in conditions]
    for k, v in ext_stack.items():
        out["ext_" + k] = np.asarray(v)              # (cond, bp, ...)
    if extended and surf_r_stack:
        geo = ec_mod.manning_and_surface_charge(bp_frames[scored], surf_r_stack[0])
        for k, v in geo.items():
            out["ext_" + k] = np.asarray(v)          # (bp,), geometry only
    if reference_solve:
        out["elec_energy_ref_kJ_mol"] = np.asarray(
            [np.nan if e is None else e for e in ref_energies], dtype=float)
    out["potential"] = np.asarray(pot_stack)         # (cond, bp, sector, radius)
    out["counterion"] = np.asarray(ndens_stack)      # (cond, phosphate)
    out["elec_energy_kJ_mol"] = np.asarray(
        [e if e is not None else np.nan for e in energies]
    )
    out["angles"] = sampled["angles"]
    out["solvent_accessible"] = np.asarray(mask_stack)
    out["accessible_fraction"] = float(np.mean(mask_stack)) if mask_stack else float("nan")
    out["potential_surface"] = np.asarray(surf_stack)
    out["surface_radius_A"] = np.asarray(surf_r_stack)
    out["surface_defined_fraction"] = (
        float(np.isfinite(np.asarray(surf_stack)).mean()) if surf_stack else float("nan"))
    out["surface_offsets_A"] = surf["offsets_A"]
    return out


def run_one(sequence: str, frag_id: str, outdir: str,
            conditions: Optional[Sequence[apbs_run.IonicCondition]] = None,
            workroot: Optional[str] = None, keep_work: bool = False,
            minimise: bool = True, grid_jitter: float = 0.0,
            jitter_seed: int = 0, keep_maps: bool = False,
            min_iterations: int = 200, min_tolerance: Optional[float] = None,
            perturb_A: float = 0.0, extended: bool = False,
            reference_solve: bool = False,
            exact_fine_spacing: bool = False,
            solute_bp_by_condition: Optional[Dict[str, int]] = None,
            snapshot: int = 0) -> Dict[str, object]:
    conditions = list(conditions or apbs_run.IONIC_CONDITIONS)
    os.makedirs(outdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{frag_id}_", dir=workroot)
    t0 = time.time()
    record: Dict[str, object] = {
        "frag_id": frag_id, "sequence": sequence, "length": len(sequence),
        "conditions": [c.name for c in conditions],
        "grid_jitter": grid_jitter, "jitter_seed": jitter_seed,
    }
    try:
        built = build_structure(sequence, work, minimise=minimise,
                                min_iterations=min_iterations, min_tolerance=min_tolerance,
                                perturb_A=perturb_A, perturb_seed=jitter_seed,
                                snapshot=snapshot, snapshot_key=frag_id)
        record.update({k: v for k, v in built.items() if k != "pdb"})
        pdb = str(built["pdb"])

        bp, _ = frames_mod.bp_frames_from_pdb(pdb)
        residues = frames_mod.read_pdb_residues(pdb)
        record["groove_check"] = frames_mod.groove_direction_check(bp, residues)
        record["n_bp_frames"] = int(len(bp))
        sl = central_slice(len(bp))
        record["scored_bp"] = [sl.start, sl.stop]

        solved = _solve_by_length(pdb, work, conditions, bp, sl, solute_bp_by_condition,
                                  grid_jitter=grid_jitter, jitter_seed=jitter_seed,
                                  keep_maps=keep_maps, extended=extended,
                                  reference_solve=reference_solve,
                                  exact_fine_spacing=exact_fine_spacing)
        record["solute_bp_by_condition"] = {c.name: int((solute_bp_by_condition or {}).get(c.name, len(bp)))
                                            for c in conditions}
        extra = {k: v for k, v in solved.items()
                 if k.startswith("ext_") or k == "elec_energy_ref_kJ_mol"}
        np.savez_compressed(
            os.path.join(outdir, f"{frag_id}.npz"),
            potential=solved["potential"],
            counterion=solved["counterion"],
            solvent_accessible=solved["solvent_accessible"],
            potential_surface=solved["potential_surface"],
            surface_radius_A=solved["surface_radius_A"],
            surface_offsets_A=solved["surface_offsets_A"],
            elec_energy_kJ_mol=solved["elec_energy_kJ_mol"],
            angles=solved["angles"],
            radii=np.asarray(apbs_run.RADIAL_OFFSETS_A),
            bp_origins=bp[sl][:, :3, 3],
            conditions=np.asarray(solved["conditions"]),
            sequence=np.asarray([sequence]),
            **extra,
        )
        record.update({
            "n_atoms": solved["n_atoms"], "total_charge": solved["total_charge"],
            "n_phosphates": solved["n_phosphates"],
            "pqr_sanitation": solved["pqr_sanitation"],
            "accessible_fraction": solved["accessible_fraction"],
            "surface_defined_fraction": solved["surface_defined_fraction"],
            "min_radius_A": solved["min_radius_A"],
            "fine_box_A": solved["fine_box_A"],
            "coarse_box_A": solved["coarse_box_A"],
            "debye_length_A": solved.get("debye_length_A"),
            "transverse_pad_A": solved.get("transverse_pad_A"),
            "elec_energy_kJ_mol": [None if np.isnan(e) else float(e)
                                   for e in solved["elec_energy_kJ_mol"]],
            "failed_conditions": solved.get("failed_conditions", {}),
            "energy_skipped_conditions": solved.get("energy_skipped_conditions", []),
            "status": "ok",
        })
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[-4000:]
    finally:
        record["seconds"] = round(time.time() - t0, 2)
        if keep_work:
            record["workdir"] = work
        else:
            shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(outdir, f"{frag_id}.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence")
    ap.add_argument("--frag-id", default="frag")
    ap.add_argument("--library", help="TSV library; with --rows, run these fragments")
    ap.add_argument("--rows", help="row selection, e.g. 0-99 (0-based, inclusive)")
    ap.add_argument("--stratum", help="restrict library rows to this stratum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workroot", default=None)
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition names; default all")
    ap.add_argument("--no-minimise", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--keep-maps", action="store_true")
    ap.add_argument("--grid-jitter", type=float, default=0.0)
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--min-iterations", type=int, default=200,
                    help="minimiser iteration cap (200 = the construction as first run)")
    ap.add_argument("--min-tolerance", type=float, default=None,
                    help="minimiser force tolerance, kJ/mol/nm (default: OpenMM's 10)")
    ap.add_argument("--solute-bp-json", default=None,
                    help="per-condition solute lengths from the convergence study, {condition: bp}")
    ap.add_argument("--resume", action="store_true", help="skip fragments whose .npz exists")
    ap.add_argument("--snapshot", type=int, default=0,
                    help="cgNA+ ensemble snapshot 1-8 instead of the ground state (outputs named "
                         "<frag_id>__s<k>)")
    ap.add_argument("--solute-bp", default=None,
                    help="per-condition solute length, e.g. NaCl_500mM:41,NaCl_250mM:41; the "
                         "fragment is built once and trimmed for those conditions")
    ap.add_argument("--exact-fine-spacing", action="store_true",
                    help="solve the fine level at exactly 0.35 A on every axis (production)")
    ap.add_argument("--production-conditions", action="store_true",
                    help="use PRODUCTION_CONDITIONS (counterion accessibility radius)")
    ap.add_argument("--extended", action="store_true",
                    help="extract every map-derived Family A channel (production)")
    ap.add_argument("--reference-solve", action="store_true",
                    help="also solve the uniform-dielectric reference for solvation energies")
    ap.add_argument("--perturb-A", type=float, default=0.0,
                    help="displace the raw build by N(0, this), seeded by --jitter-seed, so "
                         "jitter replicates also sample the minimiser's path sensitivity")
    a = ap.parse_args()

    conds = None
    table = (apbs_run.PRODUCTION_CONDITIONS_BY_NAME if a.production_conditions
             else apbs_run.CONDITIONS_BY_NAME)
    if a.conditions:
        names = [s.strip() for s in a.conditions.split(",") if s.strip()]
        conds = [table[n] for n in names]
    elif a.production_conditions:
        conds = list(apbs_run.PRODUCTION_CONDITIONS)

    jobs: List[tuple] = []
    row_conds: Dict[str, List[str]] = {}
    if a.solute_bp_json:
        import json as _json
        a.solute_bp = ",".join(f"{k}:{int(v)}" for k, v in _json.load(open(a.solute_bp_json)).items())
    if a.sequence:
        jobs.append((a.frag_id, a.sequence))
    elif a.library:
        import csv as _csv

        with open(a.library) as f:
            rows = list(_csv.DictReader(f, delimiter="\t"))
        if a.stratum:
            rows = [r for r in rows if r["stratum"] == a.stratum]
        if a.rows:
            lo, _, hi = a.rows.partition("-")
            rows = rows[int(lo): int(hi) + 1 if hi else int(lo) + 1]
        # A production manifest (data/family_a_manifest.py) carries each fragment's own
        # conditions; a plain library does not, and every row gets --conditions.
        jobs = [(r["frag_id"], r["sequence"]) for r in rows]
        row_conds = {r["frag_id"]: [c for c in r["conditions"].split(",") if c]
                     for r in rows if r.get("conditions")}
    else:
        print("need --sequence or --library", file=sys.stderr)
        return 2

    n_ok = 0
    if a.snapshot:
        jobs = [(f"{fid}__s{a.snapshot}", seq) for fid, seq in jobs]
        row_conds = {f"{k}__s{a.snapshot}": v for k, v in row_conds.items()}
    for fid, seq in jobs:
        if a.resume and os.path.exists(os.path.join(a.out, f"{fid}.npz")):
            n_ok += 1
            continue
        fconds = conds
        if fid in row_conds:
            fconds = [table[c] for c in row_conds[fid] if c in table]
        rec = run_one(seq, fid, a.out, fconds, a.workroot, a.keep_work,
                      not a.no_minimise, a.grid_jitter, a.jitter_seed, a.keep_maps,
                      min_iterations=a.min_iterations, min_tolerance=a.min_tolerance,
                      perturb_A=a.perturb_A, extended=a.extended,
                      reference_solve=a.reference_solve,
                      exact_fine_spacing=a.exact_fine_spacing,
                      solute_bp_by_condition=({k: int(v) for k, v in (x.split(":") for x in a.solute_bp.split(","))}
                                              if a.solute_bp else None),
                      snapshot=a.snapshot)
        n_ok += rec["status"] == "ok"
        print(f"{fid} {rec['status']} {rec['seconds']}s"
              + ("" if rec["status"] == "ok" else f" :: {rec.get('error')}"), flush=True)
    print(f"{n_ok}/{len(jobs)} fragments solved")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
