#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M, cruciforms: an inverted repeat extruded as two hairpins on a solved four-way junction.

Family F's best inverted repeat (arm a, spacer s, centre c) in the fragment's long window: the
top strand's left arm pairs with its right arm as hairpin T (loop: the spacer), the bottom
strand's likewise as hairpin B, and the flanks re-pair as duplexes L and R, the four helices
meeting at a four-way junction. The junction is a solved one (data/junction_templates_v1: 1DCW,
the four-way junction of an inverted repeat, and 1L4J; stacked-X). Strands map as A = top strand
into hairpin T, B = top strand out of it, C = bottom strand into hairpin B, D = bottom strand
out, so the template's arms A/D, A/B, B/C, C/D become L, T, R, B. Each helix is the cgNA+
duplex of its own sequence, superposed on its template arm by the sugar atoms of the two
junction-proximal base pairs; the hairpin loops are built from sequence (`loop_builder`; a
spacer under MIN_LOOP nt takes the innermost stem pairs into the loop); relaxation restrains every
placed heavy atom except the sugar-phosphate of the nucleotides at the four strand crossings,
whose joins it closes.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import g4_build as B          # noqa: E402
import g4_templates as T      # noqa: E402

_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data", "junction_templates_v1")
TEMPLATE_DIR = os.environ.get("PANGEA_JUNCTION_TEMPLATES", _REL if os.path.isdir(_REL)
                              else "/home/bjc75/other/biophyschem/data/junction_templates_v1")
TEMPLATES = ("1DCW", "1L4J")
COMP = str.maketrans("ACGT", "TGCA")
MIN_LOOP, MIN_STEM, MIN_FLANK = 3, 4, 3
SUGAR = ("C1'", "C4'", "O4'", "C3'")


def rc(s):
    return s.translate(COMP)[::-1]


def select(window: str, frag_offset: int, frag_len: int) -> Optional[dict]:
    """The modelled inverted repeat in the long window (positions in window coordinates)."""
    import noncanonical as NC
    w = window.upper()
    frag = w[frag_offset:frag_offset + frag_len]
    irs = NC.find_inverted_repeats(frag)
    if not irs:
        return None
    best = max(irs, key=lambda d: (d["arm"], -d["spacer"]))
    a, s = int(best["arm"]), int(best["spacer"])
    c = frag_offset + int(best["centre"])                 # first spacer position, window coordinates
    k = max(0, -(-(MIN_LOOP - s) // 2))                   # stem pairs taken into the loop
    a2 = a - k
    if a2 < MIN_STEM or c - a < MIN_FLANK or len(w) - (c + s + a) < MIN_FLANK:
        return None
    left = (c - a, c - k)                                 # stem T 5' arm
    loop = (c - k, c + s + k)
    right = (c + s + k, c + s + a)                        # stem T 3' arm
    if rc(w[left[0]:left[1]]) != w[right[0]:right[1]]:
        return None
    return {"window": w, "arm": a2, "loop": loop, "left": left, "right": right, "spacer": s, "k": k}


def _arm_pairs(res, chain_x, chain_y, first_of_3prime: bool):
    """Junction-proximal two base pairs of the arm formed by chains x and y in a template, as
    [(x key, y key), ...] proximal first. x's segment in this arm: its 3' part if first_of_3prime
    (proximal = its first residue there), else its 5' part (proximal = its last residue)."""
    pairs = []
    xs = sorted([k for k in res if k[0] == chain_x], key=lambda k: k[1])
    ys = [k for k in res if k[0] == chain_y]
    for kx in xs:
        bx = res[kx]["resname"].strip()[-1]
        for ky in ys:
            by = res[ky]["resname"].strip()[-1]
            if {bx, by} not in ({"A", "T"}, {"G", "C"}):
                continue
            na = "N1" if bx in "AG" else "N3"; nb = "N1" if by in "AG" else "N3"
            if na in res[kx]["atoms"] and nb in res[ky]["atoms"] and \
                    np.linalg.norm(res[kx]["atoms"][na] - res[ky]["atoms"][nb]) < 3.2:
                pairs.append((kx, ky))
    pairs.sort(key=lambda p: p[0][1])
    return pairs[:2] if first_of_3prime else pairs[::-1][:2]


def _duplex(seq: str):
    """cgNA+ duplex atoms grouped by (chain, resseq): chain A = seq 5'->3', chain B = rc(seq)."""
    import cgna_atomic as ca
    atoms = ca.build_duplex(seq)
    out: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for at in atoms:
        out.setdefault((at.chain, at.resseq), {})[at.name] = at.xyz
    return out


def _place(arm_atoms, arm_keys, tpl_res, tpl_keys):
    """Superpose arm residues (arm_keys) on template residues (tpl_keys) by sugar atoms."""
    P, Q = [], []
    for ka, kt in zip(arm_keys, tpl_keys):
        for n in SUGAR:
            if n in arm_atoms[ka] and n in tpl_res[kt]["atoms"]:
                P.append(arm_atoms[ka][n]); Q.append(tpl_res[kt]["atoms"][n])
    R, t, rmsd = B.kabsch(np.array(P), np.array(Q))
    return {k: {n: R @ x + t for n, x in v.items()} for k, v in arm_atoms.items()}, rmsd


def build_model(sel: dict, template: str, work: str, seed: int = 0) -> Tuple[str, dict]:
    import loop_builder as LB
    w = sel["window"]; n = len(w)
    (l0, l1), (p0, p1), (r0, r1) = sel["left"], sel["loop"], sel["right"]
    atoms, _ = T.read_pdb(os.path.join(TEMPLATE_DIR, template + ".pdb"))
    tres = T.residues(atoms)
    # template arms (proximal two pairs): L = A 5' part with D, T = A 3' part with B,
    # R = B 3' part with C, B = C 3' part with D
    arm_tpl = {"L": _arm_pairs(tres, "A", "D", False), "T": _arm_pairs(tres, "A", "B", True),
               "R": _arm_pairs(tres, "B", "C", True), "B": _arm_pairs(tres, "C", "D", True)}
    # our helices, each built with its Watson strand running away from the junction except L
    seqs = {"L": w[0:l0], "T": w[l0:l1], "R": w[r1:n], "B": rc(w[r0:r1])}
    if any(len(v) < 2 for v in arm_tpl.values()):
        raise ValueError("template arm pairs not found")
    placed, fit = {}, {}
    for arm, sq in seqs.items():
        d = _duplex(sq)
        L = len(sq)
        if arm == "L":           # proximal = the 3' end of chain A (Watson runs toward the junction)
            keys = [("A", L), ("B", 1)], [("A", L - 1), ("B", 2)]
        else:                    # proximal = the 5' end of chain A
            keys = [("A", 1), ("B", L)], [("A", 2), ("B", L - 1)]
        ak = [keys[0][0], keys[0][1], keys[1][0], keys[1][1]]
        tk = [arm_tpl[arm][0][0], arm_tpl[arm][0][1], arm_tpl[arm][1][0], arm_tpl[arm][1][1]]
        placed[arm], fit[arm] = _place(d, ak, tres, tk)
    # strands in sequence order: top = L.A + T.A + loop T + T.B + R.A ; bottom = R.B + B.A + loop B + B.B + L.B
    def chain_res(arm, ch):
        ks = sorted([k for k in placed[arm] if k[0] == ch], key=lambda k: k[1])
        letters = seqs[arm] if ch == "A" else rc(seqs[arm])
        return [(B.LETTER_TO_RESNAME[b], {nm: x for nm, x in placed[arm][k].items()}) for k, b in zip(ks, letters)]
    top = chain_res("L", "A") + chain_res("T", "A")
    miss_top = {len(top): [B.LETTER_TO_RESNAME[x] for x in w[p0:p1]]}
    top += chain_res("T", "B") + chain_res("R", "A")
    bot = chain_res("R", "B") + chain_res("B", "A")
    miss_bot = {len(bot): [B.LETTER_TO_RESNAME[x] for x in rc(w[p0:p1])]}
    bot += chain_res("B", "B") + chain_res("L", "B")
    for strand in (top, bot):                   # 5' termini carry no phosphate
        strand[0] = (strand[0][0], {k: v for k, v in strand[0][1].items() if k not in B.PHOSPHATE_GROUP})
    # hairpin loops: built and annealed on a small fragment (the stem's distal STEM_END pairs and the
    # loop), then grafted in -- annealing the whole molecule (~7,700 atoms in GBn2 on a CPU) would
    # take hours per model
    loopT = _hairpin_loop(top, len(seqs["L"]) + len(seqs["T"]), [B.LETTER_TO_RESNAME[x] for x in w[p0:p1]], work, "T", seed)
    loopB = _hairpin_loop(bot, len(seqs["R"]) + len(seqs["B"]), [B.LETTER_TO_RESNAME[x] for x in rc(w[p0:p1])], work, "B", seed)
    nT = len(seqs["L"]) + len(seqs["T"]); nB = len(seqs["R"]) + len(seqs["B"])
    top_full = top[:nT] + loopT + top[nT:]
    bot_full = bot[:nB] + loopB + bot[nB:]
    # junction joins: the residues either side of each strand crossing keep a free backbone
    jt = [nT - len(seqs["T"]) - 1, nT - len(seqs["T"]),                       # L | T (chain A, 0-based)
          nT + len(loopT) + len(seqs["T"]) - 1, nT + len(loopT) + len(seqs["T"])]
    jb = [nB - len(seqs["B"]) - 1, nB - len(seqs["B"]),
          nB + len(loopB) + len(seqs["B"]) - 1, nB + len(loopB) + len(seqs["B"])]
    out = os.path.join(work, "model.pdb")
    info = _minimise_full(top_full, bot_full, {("A", i + 1) for i in jt} | {("B", i + 1) for i in jb}, out)
    free = [("A", i + 1) for i in jt] + [("B", i + 1) for i in jb]
    # window position of every residue, both chains (top: index; bottom: complement position)
    top_pos = list(range(0, l1)) + list(range(p0, p1)) + list(range(r0, n))
    bot_pos = list(range(n - 1, r0 - 1, -1)) + list(range(p1 - 1, p0 - 1, -1)) + list(range(l1 - 1, -1, -1))
    info.update(template=template, arm_fit_A={k: round(v, 3) for k, v in fit.items()},
                top_positions=top_pos, bottom_positions=bot_pos, junction_residues=free)
    return out, info


STEM_END = 4


def _hairpin_loop(strand, n_before, loop_resnames, work, tag, seed):
    """Loop residues (resname, heavy atoms) built between strand[n_before-1] and strand[n_before]
    on the hairpin's distal STEM_END pairs, annealed there with the stem restrained."""
    import loop_builder as LB
    from openmm import app, unit
    five = strand[n_before - STEM_END:n_before]; three = strand[n_before:n_before + STEM_END]
    mini = five + three
    # the fragment's 5' nucleotide becomes a terminus: no 5' phosphate in the AMBER templates
    mini = [(mini[0][0], {k: v for k, v in mini[0][1].items() if k not in B.PHOSPHATE_GROUP})] + mini[1:]
    gp = LB.write_gapped_pdb(mini, {STEM_END: loop_resnames}, os.path.join(work, f"hp{tag}_gap.pdb"))
    known = list(range(1, STEM_END + 1)) + list(range(STEM_END + len(loop_resnames) + 1, 2 * STEM_END + len(loop_resnames) + 1))
    out = os.path.join(work, f"hp{tag}.pdb")
    LB.fill_and_relax(gp, out, known, seed=seed)
    pdb = app.PDBFile(out)
    X = pdb.positions.value_in_unit(unit.angstrom)
    res = list(pdb.topology.residues())[STEM_END:STEM_END + len(loop_resnames)]
    return [(r.name, {a.name: np.array([X[a.index][0], X[a.index][1], X[a.index][2]]) for a in r.atoms()
                      if a.element.symbol != "H"}) for r in res]


def _minimise_full(top, bot, free_bb, out_pdb):
    """Whole model: AMBER OL15 + GBn2 minimisation, every heavy atom restrained (2000 kJ/mol/nm^2)
    except the sugar-phosphate of the junction-crossing residues, whose joins it closes."""
    import openmm
    import loop_builder as LB
    from openmm import app, unit
    from pdbfixer import PDBFixer
    gp, _ = LB.write_gapped_pdb_multi({"A": (top, {}), "B": (bot, {})}, out_pdb + ".in.pdb")
    fx = PDBFixer(filename=gp); fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14/DNA.OL15.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=2.0 * unit.nanometer,
                             constraints=None)
    pos = np.array(fx.positions.value_in_unit(unit.nanometer))
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", 2000.0)
    for p_ in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p_)
    for a_ in fx.topology.atoms():
        key = (a_.residue.chain.id, int(a_.residue.id))
        if a_.element.symbol != "H" and not (key in free_bb and a_.name in LB.BACKBONE):
            rest.addParticle(a_.index, pos[a_.index])
    system.addForce(rest)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"),
                         {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")})
    ctx.setPositions(fx.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 5.0, 5000)
    st = ctx.getState(getPositions=True, getEnergy=True)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(fx.topology, st.getPositions(), fh, keepIds=True)
    return {"energy_kJ_mol": st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)}
