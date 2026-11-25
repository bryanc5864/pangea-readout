# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The featurizer's windowing: coverage, exactness on 61-bp inputs, and stitching."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training environment not active")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
sys.path.insert(0, os.path.join(ROOT, "src", "readout"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pangea as P  # noqa: E402
import featurize as FZ  # noqa: E402
from test_pangea import _synthetic_plan  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = P.PANGEA(_synthetic_plan(), width=32, n_blocks=2, rank=8, window_ladder=(10,)).eval()
    m.window_readout = "pooled"
    return m


def _rand(n, L, seed=0):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), L)) for _ in range(n)]


@pytest.mark.parametrize("L", [61, 62, 92, 121, 170, 230])
def test_tiles_cover_every_position_and_end_flush(L):
    st = FZ.tiles(L, 31)
    cov = np.zeros(L, int)
    for s in st:
        cov[s:s + FZ.WINDOW] += 1
    assert cov.min() >= 1 and st[0] == 0 and st[-1] + FZ.WINDOW == L


def test_a_61bp_sequence_is_featurized_exactly_as_one_window(model):
    seqs = _rand(5, 61)
    pb, gl = FZ.featurize(model, seqs, 31, "", "", False, 64, "cpu")
    with torch.no_grad():
        out = model(P.one_hot(seqs))
    assert np.allclose(pb.astype(np.float32), out["per_bp"][0].numpy(), atol=2e-3)
    assert np.allclose(gl.astype(np.float32), out["global"][0].numpy(), atol=2e-3)


def test_stitching_is_a_weighted_average_of_the_covering_windows(model):
    seqs = _rand(3, 92)                                     # windows at 0 and 31
    pb, _ = FZ.featurize(model, seqs, 31, "", "", False, 64, "cpu")
    with torch.no_grad():
        a = model(P.one_hot([s[0:61] for s in seqs]))["per_bp"][0].numpy()
        b = model(P.one_hot([s[31:92] for s in seqs]))["per_bp"][0].numpy()
    w = 1.0 - np.abs(np.arange(61) - 30) / 31.0
    # position 10: only window 0 covers it
    assert np.allclose(pb[:, :, 10].astype(np.float32), a[:, :, 10], atol=2e-3)
    # position 45: both windows (index 45 in window 0, 14 in window 1)
    exp = (a[:, :, 45] * w[45] + b[:, :, 14] * w[14]) / (w[45] + w[14])
    assert np.allclose(pb[:, :, 45].astype(np.float32), exp, atol=2e-3)


def test_short_sequences_use_flanks_and_keep_only_their_own_positions(model):
    seqs = _rand(2, 50)
    left, right = "A" * 25, "C" * 25
    pb, _ = FZ.featurize(model, seqs, 31, left, right, False, 64, "cpu")
    assert pb.shape[-1] == 50
    full, _ = FZ.featurize(model, seqs, 31, left, right, True, 64, "cpu")
    assert full.shape[-1] == 61 and np.allclose(full[:, :, 6:56], pb)
