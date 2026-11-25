#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family W: the window-ladder and wide-branch targets of PLAN.md, on 10,240 bp genomic fragments.

The plan's second output resolution is "a window ladder at 10, 20, 50, 100, 200 and 500 base pairs,
carrying everything defined over an extent", and its wide branch "sees 5 to 10 kb and feeds only the
mesoscale channels that genuinely need it: persistence length over long windows, nucleosome free
energy, supercoiling response, R-loop formation". The 61 bp library cannot carry either, so this
family computes them on long fragments, with the same models as Families D, E and F:

* Mechanics (Family D's model, cgNA+): the marginal step covariance of the whole fragment, by exact
  banded selected inversion (`mesoscale.step_blocks_banded`); bending stiffness, anisotropy and
  direction over the ladder and over 1, 2 and 5 kb windows (`mesoscale.windowed_bending`); the
  nucleosome wrapping free energy of every 147 bp window at Family D's three ionic strengths, with
  its rotational phase and phasing amplitude (`mesoscale.nucleosome_free_energy`); cyclizability at
  Family D's four loop lengths and loop-closure free energy at its five lengths for every window
  (`mesoscale.cyclizability`, `mesoscale.loop_closure_free_energy`); the fragment's moduli and its
  twist-writhe partition at Family D's five superhelical densities. Computed on both strands and
  combined by each quantity's Family D parity (cgNA+ is not exactly reverse-complement symmetric).
* Supercoiling response (Family E's model): the fragment as one topological domain under Family E's
  four superhelical densities, per base pair opening probability and SIDD free energy, and the
  domain's open-count moments (`thermodynamics.helix_coil_supercoiled_long`, exact truncation).
* R-loop formation: the statistical-mechanical model of Stolz et al. (PNAS 116:6260, 2019), as the
  authors implement it (github.com/chedinlab/rlooper, Rloop_equilibrium_model.cpp): one R-loop of any
  length >= 2 bp anywhere in the domain, or none, with energy
      G = a + sum B(steps) + 2 pi^2 C K (alpha + m A)^2 / (4 pi^2 C + K m),  ground K alpha^2 / 2,
  B the RNA:DNA minus DNA:DNA nearest-neighbour difference (Huppert 2008 as tabulated by the
  authors), a = 10.5 kcal/mol and A = 1/10.5 turn per bp (the paper), C = 1.8 and K = 2200 RT / N
  (the authors' code; the SI was not retrievable), T = 310 K, the fragment as the domain (the code's
  "N auto"). Either strand may be the non-template (RNA-like) strand, so both are computed.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

import mesoscale as D
import thermodynamics as E
from channels import WINDOW_LADDER_BP

FAMILY = "W"
LENGTH_BP = 10240
WIDE_WINDOWS_BP: Tuple[int, ...] = (1000, 2000, 5000)
BEND_WINDOWS_BP: Tuple[int, ...] = tuple(WINDOW_LADDER_BP) + WIDE_WINDOWS_BP
SIDD_SIGMAS: Tuple[float, ...] = tuple(E.SUPERHELICAL_DENSITIES)
RLOOP_SIGMAS: Tuple[float, ...] = (-0.07, -0.06, -0.04, -0.02)
RLOOP = {"a": 10.5, "A": 1 / 10.5, "C": 1.8, "K_RT": 2200.0, "T": 310.0, "R": 0.0019858775, "minlength": 2}
# RNA 5'->3' dinucleotide (RNA:DNA minus DNA:DNA), kcal/mol (rlooper, after Huppert 2008)
RLOOP_B = {"GG": -0.36, "GC": -0.16, "GA": -0.10, "GU": -0.06, "CG": 0.97, "CC": 0.34, "CA": 0.45, "CU": 0.38,
           "AG": -0.12, "AC": -0.16, "AA": 0.60, "AU": -0.12, "UG": 0.45, "UC": 0.50, "UA": 0.28, "UU": 0.80}
COMP = str.maketrans("ACGT", "TGCA")


def _rc(s: str) -> str:
    return s.translate(COMP)[::-1]


# --------------------------------------------------------------------------------------
# R-loops
# --------------------------------------------------------------------------------------
def rloop_steps(nontemplate: str) -> np.ndarray:
    """B per dinucleotide step of the RNA that copies `nontemplate` (5'->3'), kcal/mol."""
    r = nontemplate.upper().replace("T", "U")
    return np.array([RLOOP_B[r[j:j + 2]] for j in range(len(r) - 1)], float)


def rloop_model(nontemplate: str, sigma: float, N: int = None) -> Dict[str, object]:
    """Per-bp R-loop probability, formation probability and moments for one strand as non-template."""
    P = RLOOP; n = len(nontemplate); N = N or n
    RT = P["R"] * P["T"]; A = P["A"]; C = P["C"]; K = P["K_RT"] * RT / N
    alpha = N * sigma * A
    B = rloop_steps(nontemplate)
    cB = np.concatenate([[0.0], np.cumsum(B)])          # cB[k] = sum of the first k steps
    m = np.arange(n + 1, dtype=float)
    gsc = 2 * math.pi ** 2 * C * K * (alpha + m * A) ** 2 / (4 * math.pi ** 2 * C + K * m)
    g0 = 0.5 * K * alpha ** 2
    mins = P["minlength"]
    # reference energy: the lowest state (log-space accumulation)
    gref = g0
    for s0 in range(0, n - mins + 1):
        mm = np.arange(mins, n - s0 + 1)
        g = P["a"] + (cB[s0 + mm - 1] - cB[s0]) + gsc[mm]
        gref = min(gref, float(g.min()))
    z0 = math.exp(-(g0 - gref) / RT)
    prob = np.zeros(n); Z = z0; m1 = 0.0; gsum = 0.0
    for s0 in range(0, n - mins + 1):
        mm = np.arange(mins, n - s0 + 1)
        g = P["a"] + (cB[s0 + mm - 1] - cB[s0]) + gsc[mm]
        w = np.exp(-(g - gref) / RT)
        Z += w.sum(); m1 += float((w * mm).sum()); gsum += float((w * g).sum())
        # bp s0 + j (j = 0..) is covered by every R-loop of length > j
        rev = np.cumsum(w[::-1])[::-1]                  # rev[t] = sum_{m >= mins + t} w
        cover = np.empty(n - s0); cover[:mins] = rev[0]
        if n - s0 > mins:
            cover[mins:] = rev[1:n - s0 - mins + 1]
        prob[s0:] += cover
    prob /= Z
    p_form = 1.0 - z0 / Z
    return {"prob": prob, "p_form": float(p_form), "B": B,
            "mean_length_bp": float(m1 / (Z - z0)) if Z > z0 else float("nan"),
            "mean_dG_kcal": float(gsum / (Z - z0) - g0) if Z > z0 else float("nan")}


# --------------------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------------------
def _blocks(seq: str) -> Dict[str, np.ndarray]:
    import cgna_atomic as ca
    ca.add_cgnaplus_to_path()
    from cgnaplusparams import CGNAPlusParams
    p = CGNAPlusParams(seq.upper(), aligned_strands=True, include_stiffness=True)
    return D.step_blocks_banded({"param_names": list(p.param_names), "gs": np.asarray(p.gs), "stiffness": p.stiffmat})


def _window_mech(bl: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Every mechanics window quantity, indexed by window start (in steps)."""
    fl = D.per_step_fluctuations(bl)
    gs = bl["step_gs"]; ns = int(bl["n_steps"])
    rise = gs[:, 5]
    out: Dict[str, np.ndarray] = {}
    wb = D.windowed_bending(bl, BEND_WINDOWS_BP)
    for w in BEND_WINDOWS_BP:
        for k in ("lp", "anisotropy", "direction"):
            out[f"bend_{k}_w{w}"] = wb[f"D_bend_{k}_w{w}"]
    path = D.helical_path(bl)
    z = path["frames"][:, :, 2]                          # base-pair normals, (ns + 1, 3)
    cs = lambda v: np.concatenate([[0.0], np.cumsum(v)])  # noqa: E731
    c_bend, c_twist, c_rise = cs(fl["bend_var"]), cs(fl["twist_var"]), cs(rise)

    def wmean(c, w):
        return (c[w:] - c[:-w]) / w
    for nb in D.CYCLIZATION_LENGTHS_BP + D.LOOP_CLOSURE_LENGTHS_BP:
        w = nb - 1
        if w > ns:
            continue
        a_w = wmean(c_rise, w); bv = wmean(c_bend, w); tv = wmean(c_twist, w)
        lp = 2 * a_w / bv; Cm = a_w / tv
        cosang = np.clip(np.einsum("ij,ij->i", z[:ns - w + 1], z[w:]), -1, 1)
        net = np.degrees(np.arccos(cosang))
        if nb in D.CYCLIZATION_LENGTHS_BP:
            out[f"cyc_L{nb}"] = np.array([D.cyclizability(nb, lp[i], a_w[i], Cm[i], net[i]) for i in range(len(lp))])
        if nb in D.LOOP_CLOSURE_LENGTHS_BP:
            out[f"loopG_L{nb}"] = np.array([D.loop_closure_free_energy(nb, lp[i], a_w[i], net[i]) for i in range(len(lp))])
    # nucleosome: every 147 bp window (146 steps), its own persistence length
    w = D.NUCLEOSOME_BP - 1
    if ns >= w:
        I_ref = D.NUCLEOSOME_IONIC_M[-1]
        a_w = wmean(c_rise, w); bv = wmean(c_bend, w); lp = 2 * a_w / bv
        dg = {I: np.empty(ns - w + 1) for I in D.NUCLEOSOME_IONIC_M}
        ph = np.empty(ns - w + 1); amp = np.empty(ns - w + 1)
        f_ref = 1.0 + D.osf_persistence_length(0.0, I_ref) / np.maximum(lp, 1e-9)
        for s in range(ns - w + 1):
            sub = {"step_gs": gs[s:s + w], "step_cov_diag": bl["step_cov_diag"][s:s + w], "n_steps": np.int64(w)}
            r = D.nucleosome_free_energy(sub, float(lp[s]), I_ref)
            raw = r["nucleosome_dG_kT"] / f_ref[s]
            for I in D.NUCLEOSOME_IONIC_M:
                dg[I][s] = raw * (1.0 + D.osf_persistence_length(0.0, I) / max(float(lp[s]), 1e-9))
            ph[s] = r["rotational_phase_deg"]; amp[s] = r["phasing_amplitude_kT"]
        for I in D.NUCLEOSOME_IONIC_M:
            out[f"nuc_dG_I{I:g}"] = dg[I]
        out["nuc_phase"] = ph; out["nuc_amplitude"] = amp
    mod = D.elastic_moduli(bl)
    return out, mod


# parity of each mechanics window quantity (Family D's declarations); angle periods for circular means
_ANGLE = {"bend_direction": 180.0, "nuc_phase": 360.0}
_ANTI = ("bend_direction",)


def _combine(fwd: np.ndarray, rev: np.ndarray, key: str, w: int) -> np.ndarray:
    """Average forward and reverse-complement window values (window starts s and ns - w - s)."""
    rv = rev[::-1]                                        # rev at ns - w - s, for s = 0..
    base = key.split("_w")[0] if key.startswith("bend_") else key
    sign = -1.0 if any(base.startswith(a) for a in _ANTI) else 1.0
    per = next((p for k, p in _ANGLE.items() if base.startswith(k)), None)
    if per is None:
        return 0.5 * (fwd + sign * rv)
    t = 2 * np.pi / per
    a, b = np.radians(fwd) * (360.0 / per), np.radians(sign * rv) * (360.0 / per)
    m = np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))
    out = np.degrees(m) * (per / 360.0)
    if per == 360.0:
        out = np.mod(out, 360.0)
    return out


def _window_steps(key: str) -> int:
    if "_w" in key and key.startswith("bend_"):
        return int(key.split("_w")[1])
    if key.startswith(("cyc_L", "loopG_L")):
        return int(key.split("_L")[1]) - 1
    if key.startswith("nuc_"):
        return D.NUCLEOSOME_BP - 1
    raise KeyError(key)


def long_channels(seq: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, int]]:
    """(tracks, scalars, rc_shift). Tracks have one value per base pair (window quantities at the
    window centre, NaN where no window fits); rc_shift is each track's index shift under reverse
    complement in the channels.py convention (i <-> n-1-i+rc_shift): -1 for windows of an odd number
    of steps, else 0. Tracks ending _W/_C are a strand pair that exchange under reverse complement."""
    s = seq.upper(); n = len(s)
    tracks: Dict[str, np.ndarray] = {}; shift: Dict[str, int] = {}; sc: Dict[str, float] = {}
    bf, br = _blocks(s), _blocks(_rc(s))
    (wf, mf), (wr, mr) = _window_mech(bf), _window_mech(br)
    for key in wf:
        w = _window_steps(key)
        v = _combine(wf[key], wr[key], key, w)
        name = f"W_{key}"
        if key.startswith("bend_direction"):
            # the soft bending axis has period 180 degrees; stored doubled (period 360) so the
            # trainer's (sin, cos) representation of circular channels is continuous
            v = (2.0 * v + 180.0) % 360.0 - 180.0
            name = f"W_bend_axis2_w{w}"
        t = np.full(n, np.nan); c0 = w // 2
        t[c0:c0 + len(v)] = v
        tracks[name] = t; shift[name] = -(w % 2)     # index i <-> n-1-i+shift (channels.py)
    for k in ("persistence_length_nm", "torsional_modulus_nm", "stretch_modulus_pN", "rise_nm", "twist_stretch_coupling"):
        sc[f"W_domain_{k}"] = 0.5 * (mf[k] + mr[k])
    for sg in D.SUPERHELICAL_DENSITIES:
        p = D.twist_writhe_partition(sg, sc["W_domain_persistence_length_nm"], sc["W_domain_torsional_modulus_nm"])
        sc[f"W_delta_twist_sigma{sg:g}"] = p["delta_twist"]; sc[f"W_writhe_sigma{sg:g}"] = p["writhe"]
    # supercoiling response: the fragment as one domain (Family E's model, exactly)
    RT37 = E.R_KCAL * (37.0 + E.KELVIN)
    for sg in SIDD_SIGMAS:
        r = E.helix_coil_supercoiled_long(s, sg, 37.0, 0.15)
        p = np.asarray(r["open_prob"], float)
        tracks[f"W_sidd_open_sc{sg:g}"] = p; shift[f"W_sidd_open_sc{sg:g}"] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tracks[f"W_sidd_dG_sc{sg:g}"] = -RT37 * np.log(p / (1.0 - p))
        shift[f"W_sidd_dG_sc{sg:g}"] = 0
        kd = np.asarray(r["k_distribution"], float); kk = np.arange(kd.size)
        m1 = float((kd * kk).sum()); m2 = float((kd * kk ** 2).sum())
        sc[f"W_sidd_mean_open_sc{sg:g}"] = m1; sc[f"W_sidd_open_sd_sc{sg:g}"] = math.sqrt(max(m2 - m1 ** 2, 0.0))
        sc[f"W_sidd_mean_runs_sc{sg:g}"] = float(r["mean_runs"])
    # R-loops, each strand as the non-template strand; C tracks mapped back to top-strand indices
    for tag, strand in (("W", s), ("C", _rc(s))):
        for sg in RLOOP_SIGMAS:
            r = rloop_model(strand, sg)
            pr = r["prob"] if tag == "W" else r["prob"][::-1]
            tracks[f"W_rloop_p_sc{sg:g}_{tag}"] = pr; shift[f"W_rloop_p_sc{sg:g}_{tag}"] = 0
            sc[f"W_rloop_pform_sc{sg:g}_{tag}"] = r["p_form"]
            sc[f"W_rloop_mean_len_sc{sg:g}_{tag}"] = r["mean_length_bp"]
            sc[f"W_rloop_mean_dG_sc{sg:g}_{tag}"] = r["mean_dG_kcal"]
        B = rloop_steps(strand)
        t = np.full(n, np.nan)
        if tag == "W":
            t[:n - 1] = B                                  # step j at its 5' base pair
        else:
            t[1:] = B[::-1]                                # bottom-strand step mapped to its top-strand 3' end
        tracks[f"W_rloop_B_{tag}"] = t; shift[f"W_rloop_B_{tag}"] = 0      # W at the 5' end, C at the 3' end: mirror pair
    return tracks, sc, shift


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
SRC_RLOOP = ("Stolz et al., PNAS 116:6260 (2019), model as implemented in github.com/chedinlab/rlooper "
             "(Rloop_equilibrium_model.cpp); B table after Huppert, Mol BioSyst 4:686 (2008)")


def _declare() -> None:
    from channels import REGISTRY, ChannelSpec, Parity, Resolution
    WIN, PB, GL = Resolution.WINDOW, Resolution.PER_BP, Resolution.GLOBAL
    S, A, ST = Parity.SYMMETRIC, Parity.ANTISYMMETRIC, Parity.STRANDED
    d = {sp.name: sp for sp in REGISTRY.specs() if sp.family == "D"} if hasattr(REGISTRY, "specs") else {}
    nuc_reason = d.get("D_nucleosome_dG_I0.15_kT").provisional_reason if "D_nucleosome_dG_I0.15_kT" in d else "as Family D's nucleosome channels"
    cyc_reason = d.get("D_cyclizability_L94").provisional_reason if "D_cyclizability_L94" in d else "as Family D's cyclizability channels"
    reg = lambda **k: REGISTRY.register(ChannelSpec(family=FAMILY, **k))  # noqa: E731
    for w in BEND_WINDOWS_BP:
        sh = -(w % 2)
        reg(name=f"W_bend_lp_w{w}", resolution=WIN, window_bp=w, parity=S, units="nm", rc_shift=sh, source=D.CGNA,
            description=f"bending persistence length of the {w}-step window centred here (marginal cgNA+ covariance of the whole fragment)")
        reg(name=f"W_bend_anisotropy_w{w}", resolution=WIN, window_bp=w, parity=S, units="ratio", rc_shift=sh, source=D.CGNA,
            description=f"ratio of the principal bending compliances over that {w}-step window")
        reg(name=f"W_bend_axis2_w{w}", resolution=WIN, window_bp=w, parity=A, units="degrees", rc_shift=sh, circular=True,
            source=D.CGNA, description=f"twice the angle of the softer bending axis over that {w}-step window (period 360)")
    for I in D.NUCLEOSOME_IONIC_M:
        reg(name=f"W_nuc_dG_I{I:g}", resolution=WIN, window_bp=D.NUCLEOSOME_BP, parity=S, units="kT", source=D.LUGER,
            condition_dependent=True, provisional=True, provisional_reason=nuc_reason,
            description=f"elastic nucleosome wrapping free energy of the 147 bp window centred here, I = {I:g} M")
    reg(name="W_nuc_phase", resolution=WIN, window_bp=D.NUCLEOSOME_BP, parity=S, units="degrees", circular=True, source=D.LUGER,
        provisional=True, provisional_reason=nuc_reason, description="rotational phase of that window's best register")
    reg(name="W_nuc_amplitude", resolution=WIN, window_bp=D.NUCLEOSOME_BP, parity=S, units="kT", source=D.LUGER,
        provisional=True, provisional_reason=nuc_reason, description="rotational phasing amplitude of that window")
    for nb in D.CYCLIZATION_LENGTHS_BP:
        reg(name=f"W_cyc_L{nb}", resolution=WIN, window_bp=nb, parity=S, units="log10 J", rc_shift=-((nb - 1) % 2),
            source=D.SY, provisional=True, provisional_reason=cyc_reason,
            description=f"cyclizability of the {nb} bp window centred here (Family D's model on the window)")
    for nb in D.LOOP_CLOSURE_LENGTHS_BP:
        reg(name=f"W_loopG_L{nb}", resolution=WIN, window_bp=nb, parity=S, units="kT", rc_shift=-((nb - 1) % 2), source=D.SY,
            description=f"loop-closure free energy of the {nb} bp window centred here")
    for sg in SIDD_SIGMAS:
        reg(name=f"W_sidd_open_sc{sg:g}", resolution=PB, parity=S, units="probability", source=E.BENHAM, condition_dependent=True,
            description=f"opening probability with the 10,240 bp fragment as one domain at superhelical density {sg:g}")
        reg(name=f"W_sidd_dG_sc{sg:g}", resolution=PB, parity=S, units="kcal/mol", source=E.BENHAM, condition_dependent=True,
            description=f"stress-induced destabilisation free energy, that domain, sigma {sg:g}")
        for k, u in (("mean_open", "bp"), ("open_sd", "bp"), ("mean_runs", "count")):
            reg(name=f"W_sidd_{k}_sc{sg:g}", resolution=GL, parity=S, units=u, source=E.BENHAM, condition_dependent=True,
                description=f"open-count {k.replace('_', ' ')} of the domain at sigma {sg:g}")
    for sg in RLOOP_SIGMAS:
        for t in ("W", "C"):
            strand = "top (Watson)" if t == "W" else "bottom (Crick)"
            reg(name=f"W_rloop_p_sc{sg:g}_{t}", resolution=PB, parity=ST, units="probability", source=SRC_RLOOP, condition_dependent=True,
                description=f"probability this base pair is in an R-loop, {strand} strand non-template, domain = fragment, sigma {sg:g}")
            for k, u in (("pform", "probability"), ("mean_len", "bp"), ("mean_dG", "kcal/mol")):
                reg(name=f"W_rloop_{k}_sc{sg:g}_{t}", resolution=GL, parity=ST, units=u, source=SRC_RLOOP, condition_dependent=True,
                    description=f"R-loop {k} for the domain, {strand} strand non-template, sigma {sg:g}")
    for t in ("W", "C"):
        reg(name=f"W_rloop_B_{t}", resolution=PB, parity=ST, units="kcal/mol", source=SRC_RLOOP,
            description=f"RNA:DNA minus DNA:DNA step free energy with the {'top' if t == 'W' else 'bottom'} strand as non-template")
    for k, u in (("persistence_length_nm", "nm"), ("torsional_modulus_nm", "nm"), ("stretch_modulus_pN", "pN"),
                 ("rise_nm", "nm"), ("twist_stretch_coupling", "dimensionless")):
        reg(name=f"W_domain_{k}", resolution=GL, parity=S, units=u, source=D.CGNA, description=f"{k.replace('_', ' ')} of the 10,240 bp fragment")
    for sg in D.SUPERHELICAL_DENSITIES:
        reg(name=f"W_delta_twist_sigma{sg:g}", resolution=GL, parity=S, units="dimensionless", source=D.CGNA, condition_dependent=True,
            description=f"twist share of the linking difference over the domain at sigma {sg:g}")
        reg(name=f"W_writhe_sigma{sg:g}", resolution=GL, parity=S, units="dimensionless", source=D.CGNA, condition_dependent=True,
            description=f"writhe share of the linking difference over the domain at sigma {sg:g}")


_declare()
