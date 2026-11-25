# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Auxiliary channels land on the main layout with the main dataset's own transforms."""
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
from data import AuxLandscape, InMemoryLandscape  # noqa: E402


def test_aux_mapping_reproduces_the_main_transforms_including_circular():
    rng = np.random.default_rng(0)
    n = 500
    ang = rng.uniform(-180, 180, n); pos = 10 ** rng.uniform(-3, 2, n); lin = rng.normal(3, 2, n)
    raw = np.stack([ang, pos, lin], axis=1).astype(np.float32)
    info = {"names": ["X_angle", "X_pos", "X_lin"], "parity": ["symmetric"] * 3,
            "family": ["C"] * 3, "units": [""] * 3, "rc_shift": [0] * 3,
            "circular": [True, False, False]}
    main, new, logged, dropped = InMemoryLandscape._transform(raw, info, 2.0, 1e-3, False)
    cols, aux = AuxLandscape._map(raw, info["names"], new, positional=False)
    assert list(cols) == list(range(len(new["names"])))       # every main column mapped
    assert np.allclose(aux.astype(np.float32), main.astype(np.float32), atol=2e-3)
    assert new["names"][:2] == ["X_angle__sin", "X_angle__cos"] and "X_pos" in logged
