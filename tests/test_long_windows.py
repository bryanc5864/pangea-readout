# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Long-window channels: loading, alignment checks, and an end-to-end training run."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data import InMemoryLandscape, LongWindowSet  # noqa: E402

L, F = 61, 30


def _attrs(h, key, names, fam, parity="symmetric"):
    st = h5py.string_dtype(); n = len(names)
    h.attrs[f"{key}_names"] = np.array(names, dtype=st)
    h.attrs[f"{key}_family"] = np.array([fam] * n, dtype=st)
    h.attrs[f"{key}_parity"] = np.array([parity] * n, dtype=st)
    h.attrs[f"{key}_units"] = np.array(["u"] * n, dtype=st)
    h.attrs[f"{key}_provisional"] = np.zeros(n, bool)
    h.attrs[f"{key}_rc_shift"] = np.zeros(n, np.int64)
    h.attrs[f"{key}_circular"] = np.zeros(n, bool)


def _corpus(tmp, n_per=24, misalign=False):
    rng = np.random.default_rng(0)
    main, long_ = tmp / "main", tmp / "long"
    main.mkdir(); long_.mkdir()
    st = h5py.string_dtype()
    for s in range(2):
        seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(n_per)]
        gc = np.array([[x.count("G") + x.count("C") for x in seqs]], float).T / L
        pb = rng.normal(size=(n_per, 2, L)) + gc[:, :, None]
        gl = rng.normal(size=(n_per, 2)) + gc
        with h5py.File(main / f"shard_{s:04d}.h5", "w") as h:
            h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=st)
            h.create_dataset("frag_id", data=np.array([f"f{s}_{i}" for i in range(n_per)], dtype=object), dtype=st)
            h.create_dataset("stratum", data=np.array(["panel"] * n_per, dtype=object), dtype=st)
            h.create_dataset("per_bp", data=pb.astype(np.float32)); h.create_dataset("global", data=gl.astype(np.float32))
            _attrs(h, "per_bp", ["B_x", "C_y"], "B"); _attrs(h, "global", ["B_gx", "C_gy"], "C")
        wins = ["".join(rng.choice(list("ACGT"), F)) + x + "".join(rng.choice(list("ACGT"), F)) for x in seqs]
        if misalign and s == 1:
            wins[0] = wins[0][:F] + "A" * L + wins[0][F + L:]
        pbl = np.full((n_per, 2, 21), np.nan, np.float32)
        lab = rng.random(n_per) < 0.75                       # anchoring: some fragments unsolved
        pbl[lab] = (rng.normal(size=(lab.sum(), 2, 21)) + 3 * gc[lab, :, None]).astype(np.float32)
        gll = np.full((n_per, 1), np.nan, np.float32); gll[lab, 0] = (5 * gc[lab, 0]).astype(np.float32)
        with h5py.File(long_ / f"shard_{s:04d}.h5", "w") as h:
            h.create_dataset("window", data=np.array(wins, dtype=object), dtype=st)
            h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=st)
            h.create_dataset("per_bp_central", data=pbl); h.create_dataset("global", data=gll)
            _attrs(h, "per_bp", ["A_s0_pot_d0_c010", "A_maj_Er_c010"], "A"); _attrs(h, "global", ["A_solvation_c010"], "A")
            h.attrs["center_offset"] = F + (L - 21) // 2; h.attrs["window_length"] = L + 2 * F
    return main, long_


def test_long_set_loads_and_checks_alignment(tmp_path):
    main, long_ = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path), verbose=False)
    Ls = LongWindowSet(str(long_), ds, verbose=False)
    assert Ls.window_length == 121 and Ls.center == 50 and Ls.n_central == 21
    assert Ls.per_bp_np.shape == (48, 2, 21) and 0 < Ls.has_label.sum() < 48
    assert Ls.pb["names"] == ["A_s0_pot_d0_c010", "A_maj_Er_c010"]


def test_window_centre_mismatch_is_refused(tmp_path):
    main, long_ = _corpus(tmp_path, misalign=True)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path), verbose=False)
    with pytest.raises(ValueError, match="centre"):
        LongWindowSet(str(long_), ds, verbose=False)


def test_training_runs_with_long_windows(tmp_path):
    main, long_ = _corpus(tmp_path, n_per=40)
    out = tmp_path / "run"
    cmd = [sys.executable, "-W", "ignore", os.path.join(ROOT, "src", "model", "train.py"),
           "--shards", str(main / "shard_*.h5"), "--cache-dir", str(tmp_path), "--out", str(out),
           "--long-shards", str(long_), "--long-per-step", "8", "--width", "16", "--blocks", "2",
           "--rank", "8", "--epochs", "3", "--batch-size", "16", "--test-fraction", "0.1",
           "--library", str(tmp_path / "none.tsv"), "--windows", "10"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    fm = json.load(open(out / "final_metrics.json"))
    assert "A" in fm["test"]["per_bp_median_r2_by_family"]      # long-window family scored
    assert "B" in fm["test"]["per_bp_median_r2_by_family"]      # main family still scored
    cfg = json.load(open(out / "config.json"))
    assert cfg["per_bp_names"][-2:] == ["A_s0_pot_d0_c010", "A_maj_Er_c010"]
