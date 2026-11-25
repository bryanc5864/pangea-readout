# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Family M: selection of the modelled quadruplex, channel bookkeeping, one full model."""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics", "fold"))

import g4_select as S  # noqa: E402
import structure_folds as M  # noqa: E402
from channels import REGISTRY, Resolution  # noqa: E402

PU22 = "TGAGGGTGGGTAGGGTGGGTAA"                     # c-MYC Pu22 (PDB 1XAV), loops 1, 2, 1


def _embed(core: str, n: int = 61, seed: int = 3) -> str:
    rng = np.random.default_rng(seed)
    pad = "".join(rng.choice(list("AT"), n - len(core)))   # A/T flanks add no G tracts
    return pad[: (n - len(core)) // 2] + core + pad[(n - len(core)) // 2:]


def test_selection_of_pu22_and_its_reverse_complement():
    s = _embed(PU22)
    m = S.select(s)
    assert m is not None and m.reason == "ok" and m.strand == "+"
    assert m.loops == ["T", "TA", "T"]
    rc = s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    m2 = S.select(rc)
    assert m2.reason == "ok" and m2.strand == "-" and m2.loops == ["T", "TA", "T"]
    # the same nucleotides, seen from either orientation of the fragment
    assert m2.fragment_positions(len(s)) == sorted(len(s) - 1 - p for p in m.fragment_positions(len(s)))


def test_long_loops_and_short_tracts_are_not_modelled():
    assert S.select(_embed("GGGTTTTTTTTGGGTGGGTGGG")).reason.startswith("loop longer")
    assert S.select(_embed("GGTTGGTGGTGG")).reason.startswith("fewer than four")


def test_v3_selection_two_and_three_tetrads():
    m3 = S.select_v3(_embed(PU22))
    assert m3.reason == "ok" and m3.n_tet == 3 and m3.loops == ["T", "TA", "T"]
    m2 = S.select_v3(_embed("GGTTGGTGTGGTTGG"))            # thrombin-binding aptamer: two tetrads
    assert m2.reason == "ok" and m2.n_tet == 2 and m2.loops == ["TT", "TGT", "TT"]
    assert S.select_v3("AT" * 30 + "A") is None


def test_declared_channels_match_emitted_for_an_unmodelled_fragment():
    prof, glob = M.fold_channels("AT" * 30 + "A")
    assert set(prof) == set(M.channel_names_per_bp()) and set(glob) == set(M.channel_names_global())
    for X, _ in M.CLASSES:
        assert glob[f"M_g4{X}_modelled"] == 0.0 and np.all(prof[f"M_g4{X}_in_model"] == 0)


def test_other_folds_unmodelled_fragment_and_declarations():
    import structure_triplex as TX
    import structure_imotif as IM
    import structure_cruciform as CR
    s = "AT" * 30 + "A"
    for mod, fn in ((TX, lambda: TX.triplex_channels(s)), (IM, lambda: IM.imotif_channels(s)),
                    (CR, lambda: CR.cruciform_channels(s, s, 0))):
        prof, glob = fn()
        assert set(prof) == set(mod.channel_names_per_bp()) and set(glob) == set(mod.channel_names_global())
        mk = [k for k in glob if k.endswith("_modelled")][0]
        assert glob[mk] == 0.0
    names = M.channel_names_per_bp() + M.channel_names_global() + TX.channel_names_per_bp() + \
        TX.channel_names_global() + IM.channel_names_per_bp() + IM.channel_names_global() + \
        CR.channel_names_per_bp() + CR.channel_names_global()
    assert len(names) == len(set(names))
    assert all(REGISTRY[n].family == "M" for n in names)


@pytest.mark.skipif(shutil.which("apbs") is None, reason="APBS not installed")
def test_one_static_parallel_model_of_pu22(tmp_path):
    s = _embed(PU22)
    prof, glob = M.fold_channels(s, n_models=1, workroot=str(tmp_path), md=False)
    assert glob["M_g4p_modelled"] == 1.0 and glob["M_g4p_n_tetrads"] == 3.0
    assert int(prof["M_g4p_in_model"].sum()) == 16
    for k in ("M_g4p_phi_sites_K150", "M_g4p_phi_groove_d1_K150", "M_g4p_ion_binding_K150",
              "M_g4p_solvation_Na150", "M_g4p_selectivity_KNa", "M_g4p_dipole", "M_g4p_rg"):
        assert np.isfinite(glob[k]), k
    assert glob["M_g4p_phi_groove_d1_K150"] < 0
    assert glob["M_g4p_phi_sites_K150"] < glob["M_g4p_phi_groove_d1_K150"]
    assert glob["M_g4p_selectivity_KNa"] < 0
    on = prof["M_g4p_in_model"] == 1
    assert np.isfinite(prof["M_g4p_res_phi_d1_K150"][on]).all()
    assert np.isnan(prof["M_g4p_res_phi_d1_K150"][~on]).all()
