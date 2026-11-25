#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family A channel declarations, and assembly from the solver's per-fragment outputs.

Unlike the other families, Family A is not computed from the sequence inside
`landscape.compute_landscape`: it comes from APBS solves (`run_fragment.py --extended
--reference-solve --production-conditions --exact-fine-spacing`), one `.npz` per fragment,
and is assembled here into channels on the fragment's per-base-pair axis. Values exist on
the scored central 21 bp only; everything else is NaN, as is any condition a fragment was
not solved at (condition anchoring solves the full set at two conditions and a subset at
the rest, PLAN.md).

Reverse-complement structure, used exactly as the model's strand tie expects. In the
base-pair frame +x points into the major groove, +y toward strand I and +z along the
helix; reverse complementing flips y and z, so sector s maps to sector 6 - s (0 and 3,
the grooves, map to themselves), Watson maps to Crick, and the tangential and axial
field components change sign. Channels whose partner is another channel (sectors 4 and
5, Crick) are stored in their own frame -- reversed along the fragment -- so that
A_s5(x) = A_s1(rc x) and A_C(x) = A_W(rc x) position by position, the form of
L_m(x) = L_p(rc x) the tie implements; groove-centred quantities carry SYMMETRIC or
ANTISYMMETRIC parity.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from channels import REGISTRY, ChannelSpec, Parity, Resolution

FAMILY = "A"
CONDITIONS = (("NaCl_010mM", "c010"), ("NaCl_025mM", "c025"), ("NaCl_050mM", "c050"),
              ("NaCl_100mM", "c100"), ("NaCl_150mM", "c150"), ("NaCl_250mM", "c250"),
              ("NaCl_500mM", "c500"), ("KCl_150mM", "k150"), ("NaCl_150mM_MgCl2_2mM", "mg150"))
TAG = dict(CONDITIONS)
NACL_SERIES = tuple(c for c, _ in CONDITIONS if c.startswith("NaCl_") and c.endswith("mM")
                    and "Mg" not in c)
OFFSETS = (0, 1, 2, 4, 7, 11)
LONG_WINDOWS = (10, 25, 50, 100)
OWN_FRAME_SECTORS = (4, 5)                     # partners of 2 and 1
GROOVES = (("maj", 0), ("min", 1))             # index into the extractor's groove axis
VAL = {1.0: 0, -1.0: 1, 2.0: 2}                # extractor's valence axis (+1, -1, +2)
IONS = (("Na", "NaCl_150mM", 1.0), ("K", "KCl_150mM", 1.0), ("Mg", "NaCl_150mM_MgCl2_2mM", 2.0),
        ("Cl", "NaCl_150mM", -1.0))
SRC = "APBS nonlinear Poisson-Boltzmann on cgNA+ ground-state conformations (run_fragment.py)"
SCORED = 21
ENSEMBLE_PER_BP = tuple([f"A_s{s}_pot_d1_c150" for s in range(6)] +
                        [f"A_{g}_{q}_c150" for g in ("maj", "min") for q in ("Er", "probe")] +
                        ["A_W_condensed_c150", "A_C_condensed_c150", "A_bpsolv_c150",
                         "A_grooveasym_c150"])
MIN_SNAPSHOTS = 4


def _spec(name, res, parity, units, desc, cond=False):
    REGISTRY.register(ChannelSpec(name=name, family=FAMILY, resolution=res, parity=parity,
                                  units=units, description=desc, source=SRC,
                                  condition_dependent=cond))


def _sector_parity(s: int) -> Parity:
    return Parity.SYMMETRIC if s in (0, 3) else Parity.STRANDED


def _declare() -> None:
    PB, GL = Resolution.PER_BP, Resolution.GLOBAL
    for cname, t in CONDITIONS:
        for s in range(6):
            for d in OFFSETS:
                _spec(f"A_s{s}_pot_d{d}_{t}", PB, _sector_parity(s), "kT/e",
                      f"potential {d} A outside the surface, sector {s}, {cname}"
                      + (" (own frame)" if s in OWN_FRAME_SECTORS else ""), True)
        for g, _ in GROOVES:
            _spec(f"A_{g}_Er_{t}", PB, Parity.SYMMETRIC, "kT/(e A)", f"radial field, {g} groove, {cname}", True)
            _spec(f"A_{g}_Et_{t}", PB, Parity.ANTISYMMETRIC, "kT/(e A)", f"tangential field, {g} groove, {cname}", True)
            _spec(f"A_{g}_Ez_{t}", PB, Parity.ANTISYMMETRIC, "kT/(e A)", f"axial field, {g} groove, {cname}", True)
            _spec(f"A_{g}_dErdr_{t}", PB, Parity.SYMMETRIC, "kT/(e A^2)", f"radial field gradient, {g} groove, {cname}", True)
            _spec(f"A_{g}_probe_{t}", PB, Parity.SYMMETRIC, "kT", f"best +1 e probe energy, {g} groove, {cname}", True)
        _spec(f"A_W_condensed_{t}", PB, Parity.STRANDED, "e", f"condensed counterion charge, Watson phosphate, {cname}", True)
        _spec(f"A_C_condensed_{t}", PB, Parity.STRANDED, "e", f"condensed counterion charge, Crick phosphate, {cname} (own frame)", True)
        _spec(f"A_strandasym_{t}", PB, Parity.ANTISYMMETRIC, "kT/e", f"Watson minus Crick phosphate-cell potential, {cname}", True)
        _spec(f"A_grooveasym_{t}", PB, Parity.SYMMETRIC, "kT/e", f"major minus minor groove potential 1 A out, {cname}", True)
        _spec(f"A_bpsolv_{t}", PB, Parity.SYMMETRIC, "kJ/mol",
              f"electrostatic solvation energy of this base pair (charge-potential form, both "
              f"nucleotides), {cname}", True)
        _spec(f"A_solvation_{t}", GL, Parity.SYMMETRIC, "kJ/mol",
              f"electrostatic solvation energy of the scored 21 bp in their full context "
              f"(charge-potential form), {cname}", True)
    for ion, cname, z in IONS:
        for site in ("maj", "min", "bb"):
            _spec(f"A_{ion}_{site}_occ", PB, Parity.SYMMETRIC, "log10",
                  f"{ion} enrichment over bulk at {site} sites ({cname})", True)
    # Conformational ensemble (10 percent of fragments x 8 cgNA+ snapshots at 150 mM): mean and
    # spread across snapshots of a core subset; suffixes keep the strand-tie prefixes intact.
    parity = {n: sp for n, sp in ((f"A_s{s}_pot_d1_c150", _sector_parity(s)) for s in range(6))}
    parity.update({f"A_{g}_{q}_c150": Parity.SYMMETRIC for g, _ in GROOVES for q in ("Er", "probe")})
    parity.update({"A_W_condensed_c150": Parity.STRANDED, "A_C_condensed_c150": Parity.STRANDED,
                   "A_bpsolv_c150": Parity.SYMMETRIC, "A_grooveasym_c150": Parity.SYMMETRIC})
    for base in ENSEMBLE_PER_BP:
        for stat, word in (("ensmean", "mean"), ("enssd", "spread")):
            _spec(f"{base}_{stat}", PB, parity[base], "as the channel",
                  f"{word} of {base} across 8 cgNA+ ensemble snapshots"
                  + (" (own frame)" if base.startswith(("A_s4_", "A_s5_", "A_C_")) else ""), True)
    for stat, word in (("ensmean", "mean"), ("enssd", "spread")):
        _spec(f"A_solvation_c150_{stat}", GL, Parity.SYMMETRIC, "kJ/mol",
              f"{word} of A_solvation_c150 across 8 cgNA+ ensemble snapshots", True)
    # Window-level electrostatics on the long-fragment set only (PLAN.md: "the electrostatic
    # contribution to persistence length and the charge density integrated over 10 to 100 bp
    # windows ... need fragments long enough to contain the window"; NaN elsewhere, masked):
    # src/data/generate_family_a_longset_shards.py
    for w in LONG_WINDOWS:
        for cname, t in CONDITIONS:
            _spec(f"A_lw_qnet_w{w}_{t}", PB, Parity.SYMMETRIC, "e",
                  f"net charge (phosphates plus condensed counterions) of the {w} bp window centred here, {cname}", True)
            _spec(f"A_lw_lpel_w{w}_{t}", PB, Parity.SYMMETRIC, "nm",
                  f"electrostatic persistence length (Odijk finite-length OSF) of the {w} bp window centred here, {cname}", True)
    _spec("A_manning_xi", PB, Parity.SYMMETRIC, "1", "local Manning charge density parameter")
    _spec("A_surface_charge", PB, Parity.SYMMETRIC, "e/A^2", "local surface charge density")
    for s in range(6):
        _spec(f"A_s{s}_saltslope", PB, _sector_parity(s), "kT/e per ln(M)",
              f"slope of the sector-{s} potential 1 A out against ln ionic strength (NaCl series)"
              + (" (own frame)" if s in OWN_FRAME_SECTORS else ""), True)


def assemble(z: Dict[str, np.ndarray], n_bp: int = 61,
             ionic_strength: Optional[Dict[str, float]] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Family A channels for one fragment from its solver output (an opened .npz)."""
    conds = [str(c) for c in z["conditions"]]
    lo = (n_bp - SCORED) // 2
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}

    def put(name, v, own=False):
        a = np.full(n_bp, np.nan)
        v = np.asarray(v, float)
        a[lo:lo + len(v)] = v
        prof[name] = a[::-1].copy() if own else a

    have = {c: k for k, c in enumerate(conds)}
    E = z.get("elec_energy_kJ_mol"); R = z.get("elec_energy_ref_kJ_mol")
    for cname, t in CONDITIONS:
        k = have.get(cname)
        for s in range(6):
            for j, d in enumerate(OFFSETS):
                v = z["potential_surface"][k][:, s, j] if k is not None else np.full(SCORED, np.nan)
                put(f"A_s{s}_pot_d{d}_{t}", v, own=s in OWN_FRAME_SECTORS)
        for g, gi in GROOVES:
            if k is not None and "ext_field" in z:
                put(f"A_{g}_Er_{t}", z["ext_field"][k][:, gi, 0])
                put(f"A_{g}_Et_{t}", z["ext_field"][k][:, gi, 1])
                put(f"A_{g}_Ez_{t}", z["ext_field"][k][:, gi, 2])
                put(f"A_{g}_dErdr_{t}", z["ext_dEr_dr"][k][:, gi])
                put(f"A_{g}_probe_{t}", z["ext_probe_kT"][k][:, gi])
            else:
                for q in ("Er", "Et", "Ez", "dErdr", "probe"):
                    put(f"A_{g}_{q}_{t}", np.full(SCORED, np.nan))
        if k is not None and "ext_condensed_fraction" in z:
            put(f"A_W_condensed_{t}", z["ext_condensed_fraction"][k][:, 0])
            put(f"A_C_condensed_{t}", z["ext_condensed_fraction"][k][:, 1], own=True)
            pp = z["ext_phosphate_potential"][k]
            put(f"A_strandasym_{t}", pp[:, 0] - pp[:, 1])
        else:
            for q, own in (("W_condensed", False), ("C_condensed", True), ("strandasym", False)):
                put(f"A_{q}_{t}", np.full(SCORED, np.nan), own=own)
        if k is not None:
            put(f"A_grooveasym_{t}", z["potential_surface"][k][:, 0, 1] - z["potential_surface"][k][:, 3, 1])
        else:
            put(f"A_grooveasym_{t}", np.full(SCORED, np.nan))
        bs = z.get("ext_bp_solvation_kJ_mol")
        v = bs[k] if (k is not None and bs is not None) else np.full(SCORED, np.nan)
        # where APBS's per-atom energy is unavailable (its energy evaluation overflows at 121
        # bp and 10-25 mM and with Mg2+), the same charge-potential form from the maps
        bm = z.get("ext_bp_solvation_map_kJ_mol")
        if k is not None and bm is not None and not np.isfinite(np.asarray(v, float)).any():
            v = bm[k]
        put(f"A_bpsolv_{t}", v)
        glob[f"A_solvation_{t}"] = float(np.sum(v)) if np.all(np.isfinite(v)) else float("nan")
    for ion, cname, zv in IONS:
        k = have.get(cname)
        for site in ("maj", "min", "bb"):
            if k is None or "ext_groove_log10_enrichment" not in z:
                v = np.full(SCORED, np.nan)
            elif site == "bb":
                v = np.nanmean(z["ext_backbone_log10_enrichment"][k][:, :, VAL[zv]], axis=1)
            else:
                v = z["ext_groove_log10_enrichment"][k][:, 0 if site == "maj" else 1, VAL[zv]]
            put(f"A_{ion}_{site}_occ", v)
    for key, name in (("ext_manning_xi", "A_manning_xi"), ("ext_surface_charge_e_per_A2", "A_surface_charge")):
        put(name, z[key] if key in z else np.full(SCORED, np.nan))
    # salt response over the NaCl series present
    import apbs_run
    xs, ks = [], []
    for c in NACL_SERIES:
        if c in have:
            xs.append(np.log(apbs_run.CONDITIONS_BY_NAME[c].ionic_strength_M)); ks.append(have[c])
    for s in range(6):
        if len(ks) >= 3:
            Y = np.stack([z["potential_surface"][k][:, s, 1] for k in ks])        # (cond, bp)
            x = np.asarray(xs); xc = x - x.mean()
            slope = (xc[:, None] * (Y - np.nanmean(Y, 0))).sum(0) / (xc ** 2).sum()
        else:
            slope = np.full(SCORED, np.nan)
        put(f"A_s{s}_saltslope", slope, own=s in OWN_FRAME_SECTORS)
    # window-level channels exist on the long-fragment set only (its own assembler)
    for w in LONG_WINDOWS:
        for _, t in CONDITIONS:
            prof.setdefault(f"A_lw_qnet_w{w}_{t}", np.full(n_bp, np.nan))
            prof.setdefault(f"A_lw_lpel_w{w}_{t}", np.full(n_bp, np.nan))
    # ensemble channels come from snapshot solves (assemble_ensemble); a single solve has none
    for base in ENSEMBLE_PER_BP:
        for stat in ("ensmean", "enssd"):
            prof.setdefault(f"{base}_{stat}", np.full(n_bp, np.nan))
    for stat in ("ensmean", "enssd"):
        glob.setdefault(f"A_solvation_c150_{stat}", float("nan"))
    return prof, glob


_declare()


def assemble_ensemble(snapshots, n_bp: int = 61) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Ensemble mean and spread of the core subset from several snapshot solves (opened .npz).

    Each snapshot is assembled exactly like a ground-state solve (`assemble`), so own-frame
    storage and NaN outside the scored 21 bp carry over; fewer than MIN_SNAPSHOTS gives NaN.
    """
    parts = [assemble(z, n_bp=n_bp) for z in snapshots]
    prof: Dict[str, np.ndarray] = {}
    glob: Dict[str, float] = {}
    ok = len(parts) >= MIN_SNAPSHOTS
    for base in ENSEMBLE_PER_BP:
        st = np.stack([p[base] for p, _ in parts]) if parts else np.full((1, n_bp), np.nan)
        with np.errstate(all="ignore"):
            prof[f"{base}_ensmean"] = np.nanmean(st, axis=0) if ok else np.full(n_bp, np.nan)
            prof[f"{base}_enssd"] = np.nanstd(st, axis=0, ddof=1) if ok else np.full(n_bp, np.nan)
    v = np.array([g.get("A_solvation_c150", np.nan) for _, g in parts], dtype=float)
    glob["A_solvation_c150_ensmean"] = float(np.nanmean(v)) if ok else float("nan")
    glob["A_solvation_c150_enssd"] = float(np.nanstd(v, ddof=1)) if ok else float("nan")
    return prof, glob
