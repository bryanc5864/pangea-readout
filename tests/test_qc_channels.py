# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family B quantum-chemistry channels: tetramer lookup, orientation and reverse complement."""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
import qc_channels as Q  # noqa: E402
from channels import REGISTRY, Resolution  # noqa: E402

COMP = str.maketrans("ACGT", "TGCA")


def _table(seed=0):
    """A synthetic table over the 136 RC-unique 4-mers, with bp1 != bp2 so swaps show."""
    rng = np.random.default_rng(seed)
    tab, seen = {}, set()
    for t in itertools.product("ACGT", repeat=4):
        c = "".join(t); rc = c.translate(COMP)[::-1]
        if rc in seen:
            continue
        seen.add(c)
        rec = {}
        for g, _ in Q.GEOMS:
            rec[f"stack_{g}"] = rng.normal(-10, 2)
            rec[f"hb_bp1_{g}"], rec[f"hb_bp2_{g}"] = rng.normal(-20, 5, 2)
            for c_ in Q.SAPT:
                rec[f"stack_sapt_{c_}_{g}"] = rng.normal()
                rec[f"hb_bp1_sapt_{c_}_{g}"], rec[f"hb_bp2_sapt_{c_}_{g}"] = rng.normal(size=2)
            for q in ("homo", "lumo", "gap", "dipole"):
                rec[f"{q}_{g}"] = rng.normal()
        rec["vip"], rec["vea"] = rng.normal(7, 0.3), rng.normal(0, 0.3)
        for _, tmpl in Q.BP_KEYS:                      # Stage B and anything else per base pair
            for bp in ("bp1", "bp2"):
                rec.setdefault(tmpl.format(bp=bp), rng.normal())
        for k in Q.STEP:
            rec.setdefault(k[3:], rng.normal())
        tab[c] = rec
    return tab


def test_136_contexts_and_declared_channels():
    tab = _table()
    assert len(tab) == 136
    p, g = Q.qc_channels("ACGTTGCAAGCT" * 5 + "A", table=tab)
    assert set(p) == {s.name for s in REGISTRY.by_family("Bq") if s.resolution is Resolution.PER_BP}
    assert set(g) == {s.name for s in REGISTRY.by_family("Bq") if s.resolution is Resolution.GLOBAL}


def test_lookup_orientation_and_edges():
    tab = _table(1)
    seq = "GATTACAGGCTTACG"
    p, _ = Q.qc_channels(seq, table=tab)
    n = len(seq)
    assert np.isnan(p["Bq_stack_gs"][[0, n - 2, n - 1]]).all()      # no flank / no step
    for i in range(1, n - 2):
        ctx = seq[i - 1:i + 3]
        rec = tab.get(ctx) or tab[ctx.translate(COMP)[::-1]]
        assert np.isclose(p["Bq_stack_gs"][i], rec["stack_gs"])
    # H-bond of a base pair: mean of its two contexts, with bp1/bp2 swapped for stored RCs
    i = 5
    est = []
    for step, which in ((i - 1, "second"), (i, "first")):
        ctx = seq[step - 1:step + 3]
        assert ctx != ctx.translate(COMP)[::-1]            # (no palindrome at this position)
        if ctx in tab:
            est.append(tab[ctx]["hb_bp2_gs" if which == "second" else "hb_bp1_gs"])
        else:
            rec = tab[ctx.translate(COMP)[::-1]]
            est.append(rec["hb_bp1_gs" if which == "second" else "hb_bp2_gs"])
    assert np.isclose(p["Bq_hb_gs"][i], np.mean(est))


def test_reverse_complement_maps_steps_and_pairs():
    tab = _table(2)
    rng = np.random.default_rng(3)
    seq = "".join(rng.choice(list("ACGT"), 61))
    rc = seq.translate(COMP)[::-1]
    p, g = Q.qc_channels(seq, table=tab)
    q, h = Q.qc_channels(rc, table=tab)
    n = len(seq)
    for k in Q.STEP:                                   # step channel: i -> n-2-i
        a = p[k][: n - 1]; b = q[k][: n - 1][::-1]
        np.testing.assert_allclose(a, b, equal_nan=True)
    for k in Q.BP:                                     # base-pair channel: i -> n-1-i
        np.testing.assert_allclose(p[k], q[k][::-1], equal_nan=True)
    for k in Q.GLOBAL:
        assert np.isclose(g[k], h[k], equal_nan=True)


def test_without_a_table_everything_is_nan():
    p, g = Q.qc_channels("ACGT" * 15 + "A", table={})
    assert all(np.isnan(v).all() for v in p.values()) and all(np.isnan(v) for v in g.values())
