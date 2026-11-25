# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Side shards (families added after the corpus) merged per fragment by InMemoryLandscape."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
# src/data (a package) and src/model/data.py share the name "data"; drop whichever is
# cached so this file's own import resolves through its own sys.path entry.
for _k in [k for k in list(sys.modules) if k == "data" or k.startswith("data.")]:
    del sys.modules[_k]
from data import InMemoryLandscape  # noqa: E402

L = 61


def _write(path, seqs, pb, gl, pb_names, gl_names, fam, main):
    st = h5py.string_dtype()
    with h5py.File(path, "w") as h:
        h.create_dataset("sequence", data=np.array(seqs, dtype=object), dtype=st)
        h.create_dataset("frag_id", data=np.array([f"f{i}" for i in range(len(seqs))],
                                                  dtype=object), dtype=st)
        h.create_dataset("per_bp", data=pb.astype(np.float32))
        h.create_dataset("global", data=gl.astype(np.float32))
        if main:
            h.create_dataset("stratum", data=np.array(["panel"] * len(seqs), dtype=object),
                             dtype=st)
        for key, names in (("per_bp", pb_names), ("global", gl_names)):
            n = len(names)
            h.attrs[f"{key}_names"] = np.array(names, dtype=st)
            h.attrs[f"{key}_family"] = np.array([fam] * n, dtype=st)
            h.attrs[f"{key}_parity"] = np.array(["stranded" if fam == "J" else "symmetric"] * n,
                                                dtype=st)
            h.attrs[f"{key}_units"] = np.array(["u"] * n, dtype=st)
            h.attrs[f"{key}_provisional"] = np.zeros(n, bool)
            h.attrs[f"{key}_rc_shift"] = np.zeros(n, np.int64)
            h.attrs[f"{key}_circular"] = np.zeros(n, bool)


def _corpus(tmp_path, misalign=False, bad_side_layout=False):
    rng = np.random.default_rng(0)
    main, side = tmp_path / "main", tmp_path / "side"
    main.mkdir(); side.mkdir()
    expect = []
    for s in range(2):
        seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(8)]
        _write(main / f"shard_{s:04d}.h5", seqs, rng.normal(size=(8, 3, L)),
               rng.normal(size=(8, 2)), ["B_a", "B_b", "B_c"], ["B_g1", "B_g2"], "B", True)
        side_pb, side_gl = rng.normal(size=(8, 2, L)), rng.normal(size=(8, 1))
        side_seqs = list(reversed(seqs)) if (misalign and s == 1) else seqs
        names = ["J_p_x", "J_m_y"] if (bad_side_layout and s == 1) else ["J_p_x", "J_m_x"]
        _write(side / f"shard_{s:04d}.h5", side_seqs, side_pb, side_gl, names, ["J_p_g"],
               "J", False)
        expect.append((seqs, side_pb, side_gl))
    return main, side, expect


def test_side_channels_are_merged_per_fragment(tmp_path):
    main, side, expect = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                           extra_dirs=[str(side)], verbose=False)
    assert ds.pb["names"] == ["B_a", "B_b", "B_c", "J_p_x", "J_m_x"]
    assert ds.gl["names"] == ["B_g1", "B_g2", "J_p_g"]
    seqs = [s for e in expect for s in e[0]]
    assert ds.sequences == seqs
    side_pb = np.concatenate([e[1] for e in expect]).astype(np.float32)
    side_gl = np.concatenate([e[2] for e in expect]).astype(np.float32)
    # stored standardised in float16: undo the per-channel standardisation to compare
    mp, sp = np.asarray(ds.pb["mean"]), np.asarray(ds.pb["std"])
    mg, sg = np.asarray(ds.gl["mean"]), np.asarray(ds.gl["std"])
    got_pb = ds.per_bp_np[:, 3:5, :].astype(np.float64) * sp[3:5, None] + mp[3:5, None]
    got_gl = ds.global_np[:, 2].astype(np.float64) * sg[2] + mg[2]
    np.testing.assert_allclose(got_pb, side_pb, atol=5e-3)
    np.testing.assert_allclose(got_gl, side_gl[:, 0], atol=5e-3)
    # and the main channels are untouched by the merge
    got_main = ds.per_bp_np[:, 0, :].astype(np.float64) * sp[0] + mp[0]
    assert np.isfinite(got_main).all()
    # a second construction reads the cache and gives the same arrays
    ds2 = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                            extra_dirs=[str(side)], verbose=False)
    np.testing.assert_array_equal(ds2.per_bp_np, ds.per_bp_np)


def test_misaligned_side_shard_is_refused(tmp_path):
    main, side, _ = _corpus(tmp_path, misalign=True)
    with pytest.raises(ValueError, match="not aligned"):
        InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                          extra_dirs=[str(side)], verbose=False)


def test_side_layout_change_is_refused(tmp_path):
    main, side, _ = _corpus(tmp_path, bad_side_layout=True)
    with pytest.raises(ValueError, match="layout differs"):
        InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                          extra_dirs=[str(side)], verbose=False)


def test_auxiliary_side_shards_are_merged_and_mapped(tmp_path):
    from data import AuxLandscape
    main, side, _ = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                           extra_dirs=[str(side)], verbose=False)
    rng = np.random.default_rng(5)
    aux, aux_side = tmp_path / "aux", tmp_path / "aux_side"
    aux.mkdir(); aux_side.mkdir()
    seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(6)]
    # the auxiliary windows carry one main per-bp channel and no globals of their own ...
    _write(aux / "shard_0000.h5", seqs, rng.normal(size=(6, 1, L)), np.zeros((6, 0)),
           ["B_a"], [], "B", False)
    # ... and a side shard adds Family J per-bp and whole-sequence channels
    jpb, jgl = rng.normal(size=(6, 2, L)), rng.normal(size=(6, 1))
    _write(aux_side / "shard_0000.h5", seqs, jpb, jgl, ["J_p_x", "J_m_x"], ["J_p_g"], "J", False)
    a = AuxLandscape(str(aux), ds, verbose=False, extra_dirs=[str(aux_side)])
    names_pb = [ds.pb["names"][j] for j in a.pb_cols]
    assert names_pb == ["B_a", "J_p_x", "J_m_x"]
    assert [ds.gl["names"][j] for j in a.gl_cols] == ["J_p_g"]
    k = names_pb.index("J_m_x"); j = int(a.pb_cols[k])
    want = (jpb[:, 1] - ds.pb["mean"][j]) / ds.pb["std"][j]
    np.testing.assert_allclose(a.per_bp_np[:, k].astype(np.float64), want, atol=5e-3)
    jg = int(a.gl_cols[0])
    np.testing.assert_allclose(a.global_np[:, 0].astype(np.float64),
                               (jgl[:, 0] - ds.gl["mean"][jg]) / ds.gl["std"][jg], atol=5e-3)


def test_misaligned_auxiliary_side_shard_is_refused(tmp_path):
    from data import AuxLandscape
    main, side, _ = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                           extra_dirs=[str(side)], verbose=False)
    rng = np.random.default_rng(6)
    aux, aux_side = tmp_path / "aux", tmp_path / "aux_side"
    aux.mkdir(); aux_side.mkdir()
    seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(4)]
    _write(aux / "shard_0000.h5", seqs, rng.normal(size=(4, 1, L)), np.zeros((4, 0)),
           ["B_a"], [], "B", False)
    _write(aux_side / "shard_0000.h5", list(reversed(seqs)), rng.normal(size=(4, 2, L)),
           rng.normal(size=(4, 1)), ["J_p_x", "J_m_x"], ["J_p_g"], "J", False)
    with pytest.raises(ValueError, match="not aligned"):
        AuxLandscape(str(aux), ds, verbose=False, extra_dirs=[str(aux_side)])


def test_two_auxiliary_sets_with_a_family_on_one_only(tmp_path):
    """Genome-wide windows (B + side J) and quadruplex windows (B + side J + side M-like
    channel): one loader, united layout, NaN where a set lacks a family."""
    from data import AuxLandscape
    main, side, _ = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                           extra_dirs=[str(side)], verbose=False)
    rng = np.random.default_rng(8)
    a1, a2, sj, sg = (tmp_path / x for x in ("aux1", "aux2", "side_j", "side_g"))
    for d in (a1, a2, sj, sg):
        d.mkdir()
    s1 = ["".join(rng.choice(list("ACGT"), L)) for _ in range(5)]
    s2 = ["".join(rng.choice(list("ACGT"), L)) for _ in range(3)]
    _write(a1 / "shard_0000.h5", s1, rng.normal(size=(5, 1, L)), np.zeros((5, 0)), ["B_a"], [], "B", False)
    _write(a2 / "fold_0000.h5", s2, rng.normal(size=(3, 1, L)), np.zeros((3, 0)), ["B_a"], [], "B", False)
    j1, j2 = rng.normal(size=(5, 1)), rng.normal(size=(3, 1))
    _write(sj / "shard_0000.h5", s1, np.zeros((5, 0, L)), j1, [], ["J_p_g"], "J", False)
    _write(sj / "fold_0000.h5", s2, np.zeros((3, 0, L)), j2, [], ["J_p_g"], "J", False)
    g2 = rng.normal(size=(3, 1))
    _write(sg / "fold_0000.h5", s2, np.zeros((3, 0, L)), g2, [], ["B_g1"], "B", False)
    a = AuxLandscape(f"{a1},{a2}", ds, verbose=False, extra_dirs=[str(sj), str(sg)],
                     optional_extra=True)
    assert a.sequences == s1 + s2
    gl_names = [ds.gl["names"][j] for j in a.gl_cols]
    assert sorted(gl_names) == ["B_g1", "J_p_g"]
    kj, kb = gl_names.index("J_p_g"), gl_names.index("B_g1")
    jj, jb = int(a.gl_cols[kj]), int(a.gl_cols[kb])
    np.testing.assert_allclose(a.global_np[:, kj].astype(np.float64),
                               (np.concatenate([j1, j2])[:, 0] - ds.gl["mean"][jj]) / ds.gl["std"][jj], atol=5e-3)
    got_b = a.global_np[:, kb].astype(np.float64)
    assert np.isnan(got_b[:5]).all()                       # the genome-wide set lacks it
    np.testing.assert_allclose(got_b[5:], (g2[:, 0] - ds.gl["mean"][jb]) / ds.gl["std"][jb], atol=5e-3)
    # without optional_extra a missing side shard is an error
    with pytest.raises(FileNotFoundError):
        AuxLandscape(f"{a1},{a2}", ds, verbose=False, extra_dirs=[str(sj), str(sg)])


def test_auxiliary_holdout_is_deterministic_and_aligned(tmp_path):
    from data import AuxLandscape
    main, side, _ = _corpus(tmp_path)
    ds = InMemoryLandscape(str(main / "shard_*.h5"), cache_dir=str(tmp_path),
                           extra_dirs=[str(side)], verbose=False)
    rng = np.random.default_rng(9)
    aux = tmp_path / "auxh"; aux.mkdir()
    seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(40)]
    vals = rng.normal(size=(40, 1, L))
    _write(aux / "shard_0000.h5", seqs, vals, np.zeros((40, 0)), ["B_a"], [], "B", False)
    full = AuxLandscape(str(aux), ds, verbose=False)
    a = AuxLandscape(str(aux), ds, verbose=False, holdout={"f": 0.3})
    b = AuxLandscape(str(aux), ds, verbose=False, holdout={"f": 0.3})
    assert a.holdout_ids == b.holdout_ids and 0 < len(a.holdout_ids) < 40
    assert len(a.sequences) + len(a.holdout_ids) == 40
    kept = [i for i in range(40) if f"f{i}" not in set(a.holdout_ids)]
    assert a.sequences == [seqs[i] for i in kept]
    np.testing.assert_array_equal(a.per_bp_np, full.per_bp_np[kept])
    assert AuxLandscape(str(aux), ds, verbose=False, holdout={"zz": 0.9}).holdout_ids == []
