#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""All-atom RNA duplexes and DNA-RNA hybrids from cgNA+ A-form ground states (Family K).

PLAN.md, Family K: "electrostatics solved on A-form conformations". cgNA+ supplies the rigid-
base and phosphate frames of the RNA duplex (cgRNA+ OL3) and the hybrid (cgHYB+) as Family K
already uses them; the DNA builder (`cgna_atomic.build_duplex`) only has deoxynucleotide
templates. RNA templates are built the same way the DNA ones are: each nucleotide of a high-
resolution RNA duplex (PDB 466D, 1.16 A; interior Watson-Crick-paired A2, G3, C4 of chain A and
U13 of chain B) is expressed in its own base frame (Kabsch fit of the ideal Tsukuba base atoms;
uracil fitted on thymine's ring atoms without the methyl). The sugar keeps the template's
C3'-endo pucker and 2'-OH; the phosphate group comes from cgNA+'s phosphate frames, exactly as
for DNA. 'T' in the cgNA+ sequence of an RNA strand is uracil.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

import cgna_atomic as ca

HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "rna_templates_v1", "466D.pdb")
TEMPLATE_PDB = os.environ.get("PANGEA_RNA_TEMPLATE",
                              _REL if os.path.exists(_REL)
                              else "/home/bjc75/other/biophyschem/data/rna_templates_v1/466D.pdb")
TEMPLATE_RESIDUES = {"A": ("A", 2), "G": ("A", 3), "C": ("A", 4), "U": ("B", 13)}
RNA_RESNAME = {"A": "A", "C": "C", "G": "G", "T": "U", "U": "U"}
PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "O5'", "O3'"}


def _read_residue(path: str, chain: str, resseq: int) -> Dict[str, np.ndarray]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") and line[21] == chain and int(line[22:26]) == resseq:
                if line[16] not in (" ", "A"):          # first alternate location only
                    continue
                nm = line[12:16].strip()
                nm = {"O1P": "OP1", "O2P": "OP2"}.get(nm, nm)
                out[nm] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


class RNATemplates:
    """Same interface as `cgna_atomic.NucleotideTemplates`, for ribonucleotides."""

    def __init__(self, convention: str = "tsukuba", pdb: str = TEMPLATE_PDB):
        dna = ca.NucleotideTemplates(convention=convention)
        self.phosphate_ideal = dna.phosphate_ideal
        ideal = dna.ideal_bases
        self.templates: Dict[str, Dict[str, np.ndarray]] = {}
        for letter, (ch, rs) in TEMPLATE_RESIDUES.items():
            atoms = _read_residue(pdb, ch, rs)
            ref = ideal["T"] if letter == "U" else ideal[letter]
            I, A = [], []
            for nm, pos in ref.items():
                if nm in ("C5M", "C7"):
                    continue
                if nm in atoms:
                    I.append(pos); A.append(atoms[nm])
            if len(I) < 3:
                raise RuntimeError(f"cannot fit a base frame for template {letter}")
            tau = ca.kabsch_se3(np.asarray(I, float), np.asarray(A, float))
            inv = ca.invert_se3(tau)
            self.templates[letter] = {nm: ca.apply_se3(inv, x)[0] for nm, x in atoms.items()}
        self.templates["T"] = self.templates["U"]

    def base_atoms(self, letter: str) -> Dict[str, np.ndarray]:
        return {k: v for k, v in self.templates[letter].items() if k not in PHOSPHATE_ATOMS}

    def all_atoms(self, letter: str) -> Dict[str, np.ndarray]:
        return dict(self.templates[letter])


def build_mixed_duplex(sequence: str, gs: "ca.CGNAGroundState", watson_templates, crick_templates,
                       watson_resnames: Dict[str, str], crick_resnames: Dict[str, str]) -> List["ca.Atom"]:
    """`cgna_atomic.build_duplex` with a template set and residue names per strand."""
    sequence = sequence.upper()
    wb, cb = gs.watson_bases, gs.crick_bases
    wp, cp = gs.watson_phosphates, gs.crick_phosphates
    if gs.crick_flipped:
        cb = cb @ ca.CRICK_FLIP
        cp = cp @ ca.CRICK_FLIP
    atoms: List[ca.Atom] = []

    def place(letters, base_poses, phos_poses, chain, templates, resnames):
        m = len(letters)
        phos = []
        for i in range(m):
            if phos_poses is None or i >= len(phos_poses) or not np.all(np.isfinite(phos_poses[i])):
                phos.append({}); continue
            tau = phos_poses[i]
            phos.append({nm: ca.apply_se3(tau, pos)[0] for nm, pos in templates.phosphate_ideal.items()})
        for i, letter in enumerate(letters):
            tau = base_poses[i]
            placed = {nm: ca.apply_se3(tau, pos)[0] for nm, pos in templates.base_atoms(letter).items()}
            pa = phos[i] if i < len(phos) else {}
            if i > 0 and pa:
                for nm in ("P", "OP1", "OP2", "O5'"):
                    if nm in pa:
                        placed[nm] = pa[nm]
            elif "O5'" in templates.all_atoms(letter):
                placed["O5'"] = ca.apply_se3(tau, templates.all_atoms(letter)["O5'"])[0]
            nxt = phos[i + 1] if i + 1 < len(phos) else {}
            if nxt and "O3'" in nxt:
                placed["O3'"] = nxt["O3'"]
            elif "O3'" in templates.all_atoms(letter):
                placed["O3'"] = ca.apply_se3(tau, templates.all_atoms(letter)["O3'"])[0]
            for nm, xyz in placed.items():
                if ca._element_of(nm) == "H":
                    continue
                atoms.append(ca.Atom(nm, resnames[letter], chain, i + 1, xyz, ca._element_of(nm)))

    place(list(sequence), wb, wp, "A", watson_templates, watson_resnames)
    crick = [ca.COMPLEMENT[c] for c in sequence][::-1]
    place(crick, cb[::-1], cp[::-1], "B", crick_templates, crick_resnames)
    return atoms


def build_rna_duplex(sequence: str, parameter_set: str) -> List["ca.Atom"]:
    gs = ca.CGNAGroundState(sequence, parameter_set=parameter_set)
    t = RNATemplates()
    return build_mixed_duplex(sequence, gs, t, t, RNA_RESNAME, RNA_RESNAME)


def build_hybrid(sequence: str, parameter_set: str) -> List["ca.Atom"]:
    """DNA Watson strand (the given sequence) paired with an RNA Crick strand."""
    gs = ca.CGNAGroundState(sequence, parameter_set=parameter_set)
    return build_mixed_duplex(sequence, gs, ca.NucleotideTemplates(), RNATemplates(),
                              ca.RESNAME, RNA_RESNAME)
