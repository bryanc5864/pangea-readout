#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""One place that computes the whole landscape for a sequence.

Each family module owns its physics and its channel declarations, but each exposes a
slightly different entry point: Family L is computed once per strand, Family D takes an
optional precomputed cgNA+ ground state, Family C takes an ensemble size and a minimised
structure. Until now every caller had to know all of that.

That cost something real. Families F and L emitted per-base-pair profiles that were never
declared in the registry, and nothing noticed, because no single caller ever compared the
full emitted set against the full declared set. The target-generation driver builds its
column set from the registry, so those tracks would have been silently dropped from the
training data. `tests/test_landscape.py` now asserts emitted == declared across every
family at once, which is a check that only exists because this module exists.

**The minimised structure is built once and shared.** Family C's backbone torsions and
Family E's hydrogen-bond distances both need it, it costs about 4.85 s per 61 bp fragment
against 1.52 s for everything else combined, and building it twice would make it 76
percent of a 90 percent cost.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from channels import REGISTRY, Parity, Resolution

# **These imports are eager on purpose, and the reason is a bug this caused.**
#
# Each family module registers its channels as a side effect of being imported. When
# this module imported them lazily inside `compute_landscape`, `channel_names` consulted
# an empty registry if it was called first -- which is exactly what the target generator
# does, since it needs the column set before it computes anything. The generator
# therefore built a dataset with **zero channels**, wrote it to disk, and reported
# success. Nothing raised.
#
# Importing them here means the registry is complete the moment `landscape` is
# importable, so there is no order in which a caller can observe it half-filled.
# `tests/test_landscape.py::test_importing_landscape_alone_populates_the_registry`
# pins that, and the generator refuses to write an empty channel set as a second line
# of defence.
import electronic as _electronic          # noqa: F401,E402  (Family B)
import geometry as _geometry              # noqa: F401,E402  (Family C)
import mesoscale as _mesoscale            # noqa: F401,E402  (Family D)
import modifications as _modifications    # noqa: F401,E402  (Family H)
import noncanonical as _noncanonical      # noqa: F401,E402  (Family F)
import rna_structure as _rna_structure    # noqa: F401,E402  (Family L)
import solvation as _solvation            # noqa: F401,E402  (Family G)
import thermodynamics as _thermodynamics  # noqa: F401,E402  (Family E)
import duplex_classes as _duplex_classes  # noqa: F401,E402  (Family K)
import single_strand as _single_strand    # noqa: F401,E402  (Family J)
import single_strand_oxdna as _single_strand_oxdna  # noqa: F401,E402  (Family J, oxDNA2 runs: side shards)
import rna_modification as _rna_modification  # noqa: F401,E402  (Family N)
import protein_readout as _protein_readout    # noqa: F401,E402  (Family I, held apart)
import structure_folds as _structure_folds    # noqa: F401,E402  (Family M quadruplexes, held apart)
import structure_triplex as _structure_triplex  # noqa: F401,E402  (Family M triplexes)
import structure_imotif as _structure_imotif    # noqa: F401,E402  (Family M i-motifs)
import structure_cruciform as _structure_cruciform  # noqa: F401,E402  (Family M cruciforms)
import qc_channels as _qc_channels            # noqa: F401,E402  (Family B quantum chemistry, "Bq")
import context_libraries as _context_libraries  # noqa: F401,E402  (Gmd, Kmd, Gdk, Bt: per-context libraries)
import rloop_energetics as _rloop_energetics  # noqa: F401,E402  (Fr)
import reaction_rates as _reaction_rates        # noqa: F401,E402  (Gr: hydrolytic damage rates)

# G and H joined on the second generation pass. They were complete and tested earlier but
# held out, because adding a family changes the channel set and the first target array
# was running: a task starting after the edit would have written a different layout from
# one that started before it.
FAMILIES: Tuple[str, ...] = ("B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "N")


def channel_names(resolution: Resolution,
                  families: Sequence[str] = FAMILIES) -> List[str]:
    """The declared channel set at one resolution, in a fixed order.

    Sorted by name so the column order of a generated dataset is reproducible and does
    not depend on module import order. A training set whose columns move between runs is
    a silent correctness hazard.
    """
    fam = set(families)
    return sorted(s.name for s in REGISTRY.specs()
                  if s.resolution is resolution and s.family in fam)


def compute_landscape(seq: str, minimised_pdb: Optional[str] = None,
                      n_ensemble: int = 200, seed: int = 0,
                      families: Sequence[str] = FAMILIES,
                      min_ensemble: Optional[Dict[str, np.ndarray]] = None,
                      context: Optional[Dict[str, float]] = None
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Every declared channel for one sequence, as (per-base-pair, whole-sequence).

    `minimised_pdb` is optional; without it the channels that need an all-atom relaxed
    structure come back NaN rather than approximated, for the reason recorded in
    `geometry`: the unminimised torsions are wrong by up to 63 degrees.
    """
    seq = seq.upper()
    n = len(seq)
    fam = set(families)
    profiles: Dict[str, np.ndarray] = {}
    globals_: Dict[str, float] = {}

    if "B" in fam:
        B = globals()["_electronic"]
        profiles.update(B.electronic_profiles(seq))
        globals_.update(B.electronic_channels(seq))

    if "C" in fam:
        G = globals()["_geometry"]
        st = G.geometry_state(seq, n_ensemble=n_ensemble, seed=seed,
                              minimised_pdb=minimised_pdb, min_ensemble=min_ensemble)
        c_prof = G.geometry_profiles(seq, state=st)
        profiles.update(c_prof)
        globals_.update(G.geometry_channels(seq, state=st, profiles=c_prof))

    if "D" in fam:
        D = globals()["_mesoscale"]
        globals_.update(D.mesoscale_channels(seq))

    if "E" in fam:
        E = globals()["_thermodynamics"]
        e_prof = E.thermo_profiles(seq, minimised_pdb=minimised_pdb)
        profiles.update(e_prof)
        globals_.update(E.thermo_channels(seq, minimised_pdb=minimised_pdb,
                                          profiles=e_prof))

    if "F" in fam:
        F = globals()["_noncanonical"]
        profiles.update(F.noncanonical_profiles(seq))
        globals_.update(F.noncanonical_channels(seq))

    if "G" in fam:
        G_ = globals()["_solvation"]
        if minimised_pdb:
            vip = profiles.get("B_vip_eV")
            g_prof = G_.solvation_profiles(seq, minimised_pdb, vip_profile=vip)
            profiles.update(g_prof)
            globals_.update(G_.solvation_channels(seq, minimised_pdb, profiles=g_prof,
                                                  vip_profile=vip))
        else:
            # Every Family G channel is an accessibility on the relaxed structure, so
            # without one they are emitted as NaN -- declared columns must always appear,
            # and a surface area guessed from the unrelaxed reconstruction would be wrong
            # smoothly (see `solvation`).
            for spec in REGISTRY.by_family("G"):
                if spec.resolution is Resolution.PER_BP:
                    profiles[spec.name] = np.full(n, np.nan)
                else:
                    globals_[spec.name] = float("nan")

    if "H" in fam:
        H_ = globals()["_modifications"]
        h_resp = H_.modification_response(seq)
        profiles.update(H_.modification_profiles(seq, response=h_resp))
        globals_.update(H_.modification_channels(seq, response=h_resp))

    if "J" in fam:
        J = globals()["_single_strand"]
        j_prof, j_glob = J.single_strand_channels(seq)
        profiles.update(j_prof)
        globals_.update(j_glob)

    if "I" in fam:
        # Held apart: only ever requested explicitly (never in FAMILIES), and it reads the
        # fragment's own nucleosome free energy from `context` (the main shard's Family D).
        I_ = globals()["_protein_readout"]
        i_prof, i_glob = I_.readout_channels(seq, context)
        profiles.update(i_prof)
        globals_.update(i_glob)

    if "Bq" in fam:
        # Quantum-chemistry tier mapped onto every step by its tetramer context (a table
        # lookup); requested explicitly, since the main shards predate the table.
        Q = globals()["_qc_channels"]
        q_prof, q_glob = Q.qc_channels(seq)
        profiles.update(q_prof)
        globals_.update(q_glob)

    if "M" in fam:
        # Held apart like Family I: explicit 3D folds (minimisation and PB per model), only
        # ever requested explicitly.
        # (production runs these per fold type and packs them, `pack_family_m.py`; here the
        # fragment is its own window for the cruciform, so short flanks leave it unmodelled)
        for fn in (lambda: globals()["_structure_folds"].fold_channels(seq),
                   lambda: globals()["_structure_triplex"].triplex_channels(seq),
                   lambda: globals()["_structure_imotif"].imotif_channels(seq),
                   lambda: globals()["_structure_cruciform"].cruciform_channels(seq, seq, 0)):
            m_prof, m_glob = fn()
            profiles.update(m_prof)
            globals_.update(m_glob)

    if "N" in fam:
        N = globals()["_rna_modification"]
        n_prof, n_glob = N.modification_channels(seq)
        profiles.update(n_prof)
        globals_.update(n_glob)

    if "K" in fam:
        K = globals()["_duplex_classes"]
        k_prof, k_glob = K.duplex_class_channels(seq)
        profiles.update(k_prof)
        globals_.update(k_glob)

    if "L" in fam:
        L = globals()["_rna_structure"]
        for strand in ("+", "-"):
            profiles.update(L.rna_profiles(seq, strand))
            globals_.update(L.rna_channels(seq, strand))

    # Pad every profile to the sequence length, so the array shape is a property of the
    # fragment rather than of which channel it is.
    for k, v in list(profiles.items()):
        a = np.asarray(v, dtype=float).ravel()
        if a.shape[0] != n:
            padded = np.full(n, np.nan)
            padded[:min(n, a.shape[0])] = a[:min(n, a.shape[0])]
            profiles[k] = padded
        else:
            profiles[k] = a
    return profiles, globals_


def landscape_arrays(seq: str, per_bp_names: Sequence[str],
                     global_names: Sequence[str], **kw
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """`compute_landscape` packed into two dense arrays with a fixed column order.

    Returns (n_per_bp_channels, n_bp) and (n_global_channels,). A channel the family did
    not emit becomes a column of NaN rather than a missing key, so the array shape is
    identical for every fragment and a shard can be concatenated without reconciliation.
    """
    prof, glob = compute_landscape(seq, **kw)
    n = len(seq)
    P = np.full((len(per_bp_names), n), np.nan, dtype=np.float32)
    for i, name in enumerate(per_bp_names):
        v = prof.get(name)
        if v is not None:
            P[i] = np.asarray(v, dtype=np.float32)
    Gv = np.full(len(global_names), np.nan, dtype=np.float32)
    for i, name in enumerate(global_names):
        v = glob.get(name)
        if v is not None:
            Gv[i] = np.float32(v)
    return P, Gv
