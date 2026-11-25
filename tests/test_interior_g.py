# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The interior Family G recomputation reproduces the stored definitions on all positions."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data import InMemoryLandscape  # noqa: E402


def _arrays():
    rng = np.random.default_rng(0)
    pb = ["G_radical_target_W", "G_radical_target_C", "G_sasa_total_W", "X_other"]
    gl = ["G_max_cleavage_W", "G_cleavage_heterogeneity_W", "G_max_cleavage_C",
          "G_mean_sasa_total_W", "G_mean_radical_target_C", "X_global"]
    P = rng.random((5, len(pb), 20)).astype(np.float32)
    P[:, 0, 0] = 9.0                                   # a huge terminal value on W
    G = np.zeros((5, len(gl)), dtype=np.float32)
    return P, G, pb, gl


def test_trim_zero_equals_the_stored_definitions():
    P, G, pb, gl = _arrays()
    InMemoryLandscape._interior_family_g(P, G, pb, gl, 0, False)
    assert np.allclose(G[:, 0], P[:, 0].max(1))
    assert np.allclose(G[:, 1], P[:, 0].std(1))
    assert np.allclose(G[:, 2], P[:, 1].max(1))
    assert np.allclose(G[:, 3], P[:, 2].mean(1))
    assert np.allclose(G[:, 4], P[:, 1].mean(1))
    assert np.all(G[:, 5] == 0)                        # other families untouched


def test_interior_trim_removes_terminal_values():
    P, G, pb, gl = _arrays()
    InMemoryLandscape._interior_family_g(P, G, pb, gl, 2, False)
    assert np.all(G[:, 0] < 1.0)                       # the terminal 9.0 is gone
    assert np.allclose(G[:, 0], P[:, 0, 2:18].max(1))
