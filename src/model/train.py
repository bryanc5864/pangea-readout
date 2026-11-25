#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Stage-1 pretraining: fit PANGEA to the analytic landscape.

`PLAN.md`: "Pretraining on analytically computable targets... This stage teaches the
trunk the local sequence-to-physics map before any expensive label is touched."

Every stage-1 target is a deterministic function of the 61 bp input -- cgNA+, nearest
neighbour models, ViennaRNA and the quantum chemistry tables all take the fragment and
nothing else -- so the ceiling for nearly every channel is R-squared 1. The exceptions are
the ensemble standard deviations, which carry Monte Carlo noise from a finite number of
draws. Anything well short of the ceiling is a modelling shortfall to find, not a limit to
accept.

**What changed from the first run, each for a measured reason** (see DECISIONS.md):

  * The equivariant merge reflects each channel by its measured index map. The first run
    flipped everything as if base-pair-indexed and averaged 47 step and groove channels
    with neighbouring positions; `C_twist` reached R-squared 0.09.
  * Log-variances are merged symmetrically; they had been sign-flipped for antisymmetric
    channels.
  * Constant channels are dropped, circular ones become (sin, cos), heavy-tailed positive
    ones become log10 -- decided from the data and recorded in every run.
  * The loss is a vectorised beta-NLL with families balanced explicitly. The stacked
    family weights of the first run were ill-posed on a negative loss and drifted
    together toward their clamp.
  * The data live on the GPU. The first run spent 606 s an epoch reading HDF5 over NFS
    and looping over channels in Python.

The validation split is by sequence similarity, using groups computed once and cached.
On this library 3.5 percent of fragments sit in near-duplicate groups -- tiling neighbours
and the designed hexamer library -- which a random split would leak.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import InMemoryLandscape, load_bpp, pooled_window_targets  # noqa: E402
from pangea import (PANGEA, ChannelPlan, LandscapeLoss, MIN_PAIR_SEPARATION,  # noqa: E402
                    model_kwargs_from_args)


# ======================================================================================
# Splits
# ======================================================================================

def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """A deterministic random split. Optimistic on a tiled library; kept to measure that."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * val_fraction))
    return perm[k:], perm[:k]


def similarity_split_from_labels(labels: np.ndarray, val_fraction: float, seed: int,
                                 force_train: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Assign whole similarity groups to one side until validation is large enough. Groups in
    `force_train` are never drawn for the held-out side."""
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    want = int(round(len(labels) * val_fraction))
    taken, chosen = 0, []
    for j in order:
        if taken >= want:
            break
        if force_train and int(uniq[j]) in force_train:
            continue
        chosen.append(uniq[j])
        taken += int(counts[j])
    is_val = np.isin(labels, np.asarray(chosen))
    report = {"split": "similarity", "n_groups": int(len(uniq)),
              "largest_group": int(counts.max()) if len(counts) else 0,
              "n_val": int(is_val.sum()), "n_train": int((~is_val).sum())}
    return np.flatnonzero(~is_val), np.flatnonzero(is_val), report


def similarity_split(seqs: Sequence[str], val_fraction: float, seed: int,
                     threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Group near-duplicates by MinHash and split by group (see `splits.py`)."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
    import splits as sp
    labels, info = sp.group_by_similarity(list(seqs), threshold=threshold, seed=seed)
    tr, va, rep = similarity_split_from_labels(labels, val_fraction, seed)
    rep.update({k: v for k, v in info.items() if isinstance(v, (int, float, str))})
    rep["threshold"] = threshold
    return tr, va, rep


def three_way_split(n: int, labels: Optional[np.ndarray], val_fraction: float,
                    test_fraction: float, seed: int, kind: str, force_train: Optional[set] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train / validation / test, with the test side carved first and never consulted.

    Validation picks the best epoch, so R-squared reported on it carries a selection
    optimism; PLAN.md's internal validation needs held-out fragments, so the headline
    numbers come from a test side that no training or selection decision ever sees.
    Under the similarity split whole groups go to one side, so near-duplicates cannot
    straddle test and train. With test_fraction 0 this reduces to the two-way split.
    """
    if test_fraction <= 0:
        if kind == "similarity":
            tr, va, rep = similarity_split_from_labels(labels, val_fraction, seed, force_train)
        else:
            tr, va = split_indices(n, val_fraction, seed)
            rep = {"split": "random", "n_train": int(len(tr)), "n_val": int(len(va))}
        rep["n_test"] = 0
        return tr, va, np.array([], dtype=np.int64), rep
    if kind == "similarity":
        rest, te, rep0 = similarity_split_from_labels(labels, test_fraction, seed + 1000, force_train)
        tr_r, va_r, _ = similarity_split_from_labels(
            np.asarray(labels)[rest], val_fraction / (1.0 - test_fraction), seed, force_train)
        tr, va = rest[tr_r], rest[va_r]
        rep = {"split": "similarity", "n_groups": rep0["n_groups"],
               "largest_group": rep0["largest_group"]}
    else:
        rng = np.random.default_rng(seed + 1000)
        perm = rng.permutation(n)
        k_te = int(round(n * test_fraction))
        te, rest = perm[:k_te], perm[k_te:]
        k_va = int(round(n * val_fraction))
        va, tr = rest[:k_va], rest[k_va:]
        rep = {"split": "random"}
    rep.update({"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))})
    assert not (set(tr.tolist()) & set(te.tolist())) and not (set(va.tolist()) & set(te.tolist()))
    return np.sort(tr), np.sort(va), np.sort(te), rep


# Base pairs excluded from each end for the interior R-squared (see evaluate_full).
INTERIOR_TRIM = 5

# Targets are standardised to unit variance over the corpus, so an evaluation side whose
# variance for a channel falls below this has, in effect, a constant target there.
MIN_EVAL_VARIANCE = 1e-8

# Nominal central coverage levels for the calibration curves.
CAL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.683, 0.7, 0.8, 0.9, 0.954, 0.99)


# Set by main() when Family W is appended: (per-bp width, global width) of the full plan. Results of
# an outermost evaluate / evaluate_full on main fragments are padded with NaN to it, since Family W's
# columns have no labels there (they are scored on Family W fragments by evaluate_wide).
_PLAN_WIDTH: Optional[Tuple[int, int]] = None


def _pad_to_plan(res):
    if _PLAN_WIDTH is None:
        return res
    npb, ngl = _PLAN_WIDTH

    def pad(a, n):
        a = np.asarray(a)
        if a.shape[-1] >= n:
            return a
        return np.concatenate([a, np.full(a.shape[:-1] + (n - a.shape[-1],), np.nan)], -1)
    out = {}
    for key, v in res.items():
        n = ngl if key == "global" else npb
        if key in ("per_bp", "global") or key.startswith("window_"):
            out[key] = {f: pad(a, n) for f, a in v.items()} if isinstance(v, dict) else pad(v, n)
        else:
            out[key] = v
    return out


@torch.no_grad()
def evaluate_full(model, X, Ypb, Ygl, idx, batch: int, windows: Sequence[int],
                  levels: Sequence[float] = CAL_LEVELS, long=None,
                  pb_col0: int = 0, gl_col0: int = 0,
                  pos: Optional[Tuple[int, int]] = None, _inner: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    if not _inner:
        return _pad_to_plan(evaluate_full(model, X, Ypb, Ygl, idx, batch, windows, levels, long,
                                          pb_col0, gl_col0, pos, _inner=True))
    """Per-channel R-squared and calibration of the predicted variances.

    Calibration: for each nominal central level p, the fraction of held-out targets
    with |y - mu| <= z_p * sigma, z_p the two-sided Gaussian quantile. A calibrated
    channel tracks the diagonal; above it the variances are too wide, below too narrow.
    Also the mean squared standardised error, which is 1 when calibrated.
    """
    if long is not None:
        return _with_long(evaluate_full, model, X, Ypb, Ygl, idx, batch, windows, long,
                          levels=levels, _inner=True)
    from statistics import NormalDist
    zq = torch.tensor([NormalDist().inv_cdf(0.5 + p / 2.0) for p in levels],
                      device=X.device)
    model.eval()
    acc: Dict[str, List[torch.Tensor]] = {}
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        x = X[b].float()
        tgt = {"per_bp": Ypb[b].float(), "global": Ygl[b].float()}
        for w in windows:
            tgt[f"window_{w}"] = pooled_window_targets(tgt["per_bp"], w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out = model(x)
        for key, t in tgt.items():
            m, lv = _slice_out(out[key][0].float(), out[key][1].float(), key, t,
                               pb_col0, gl_col0, pos)
            valid = torch.isfinite(t)
            tz = torch.where(valid, t, torch.zeros_like(t))
            mz = torch.where(valid, m, torch.zeros_like(m))
            dims = (0, 2) if t.dim() == 3 else (0,)
            z = (tz - mz).abs() * torch.exp(-0.5 * lv)
            z = torch.where(valid, z, torch.full_like(z, float("inf")))
            cover = torch.stack([(z <= q).sum(dims) for q in zq]).float()
            z2 = torch.where(valid, z, torch.zeros_like(z)) ** 2
            parts = [((mz - tz) ** 2).sum(dims), tz.sum(dims), (tz ** 2).sum(dims),
                     valid.sum(dims).float(), cover, z2.sum(dims)]
            if t.dim() == 3:
                # Interior positions only (INTERIOR_TRIM bp from each end excluded): the
                # free ends of a 61 bp fragment carry cgNA+'s end parameterisation, which
                # can dominate a channel's variance (methylation twist deltas reach -38.8
                # degrees at the terminal steps against +-5 in the interior).
                k0, k1 = INTERIOR_TRIM, t.shape[-1] - INTERIOR_TRIM
                vi = valid[..., k0:k1]
                ti, mi = tz[..., k0:k1], mz[..., k0:k1]
                parts += [((mi - ti) ** 2).sum(dims), ti.sum(dims), (ti ** 2).sum(dims),
                          vi.sum(dims).float()]
            if key not in acc:
                acc[key] = parts
            else:
                for a, p_ in zip(acc[key], parts):
                    a += p_
    model.train()
    res: Dict[str, Dict[str, np.ndarray]] = {}

    def _r2(sse, s, sq, n):
        n_ = n.clamp(min=1.0)
        var = sq / n_ - (s / n_) ** 2
        r_ = 1.0 - sse / n_ / var.clamp(min=1e-12)
        return torch.where(var > MIN_EVAL_VARIANCE, r_,
                           torch.full_like(r_, float("nan"))).cpu().numpy()

    for key, parts in acc.items():
        sse, s, sq, n, cover, z2 = parts[:6]
        n_ = n.clamp(min=1.0)
        res[key] = {"r2": _r2(sse, s, sq, n),
                    "coverage": (cover / n_).cpu().numpy(),
                    "mean_z2": (z2 / n_).cpu().numpy(),
                    "n": n.cpu().numpy()}
        if len(parts) > 6:
            res[key]["r2_interior"] = _r2(*parts[6:10])
    return res


def _slice_out(m, lv, key, t, pb_col0, gl_col0, pos):
    """Model outputs restricted to the columns (and positions) a target block covers.

    The plan holds the main channels first and any long-window channels after them, so
    main targets read columns [0, C_main) and long-window targets [C_main, C_main + C_long)
    on the central positions of their windows.
    """
    if key == "per_bp":
        m = m[:, pb_col0:pb_col0 + t.shape[1]]
        lv = lv[:, pb_col0:pb_col0 + t.shape[1]] if lv is not None else None
        if pos is not None:
            m = m[..., pos[0]:pos[1]]
            lv = lv[..., pos[0]:pos[1]] if lv is not None else None
    elif key == "global":
        m = m[:, gl_col0:gl_col0 + t.shape[1]]
        lv = lv[:, gl_col0:gl_col0 + t.shape[1]] if lv is not None else None
    else:                                               # pooled windows: main channels only
        m = m[:, :t.shape[1]]
        lv = lv[:, :t.shape[1]] if lv is not None else None
    return m, lv


class LongTensors:
    """The long-window set on the device, with where its columns sit in the plan."""

    def __init__(self, L, device, C_main: int, G_main: int):
        self.X = torch.from_numpy(L.one_hot()).to(device)
        self.Ypb = torch.from_numpy(L.per_bp_np).to(device)
        self.Ygl = torch.from_numpy(L.global_np).to(device)
        self.pos = (L.center, L.center + L.n_central)
        self.C_main, self.G_main = C_main, G_main
        self.has = torch.as_tensor(L.has_label, device=device)


def _with_long(fn, model, X, Ypb, Ygl, idx, batch, windows, long, **kw):
    """Evaluate the main fragments and the long windows, merged channel-wise into the plan."""
    kw["_inner"] = True
    main = fn(model, X, Ypb, Ygl, idx, batch, windows, **kw)
    li = idx[long.has[idx]] if len(idx) else idx
    if len(li) == 0:
        return main
    lres = fn(model, long.X, long.Ypb, long.Ygl, li, batch, (), pb_col0=long.C_main,
              gl_col0=long.G_main, pos=long.pos, **kw)
    out = {}
    for key, v in main.items():
        lv = lres.get(key)
        if key.startswith("window_"):
            n_extra = long.Ypb.shape[1]
            if isinstance(v, dict):
                out[key] = {f: np.concatenate([a, np.full(a.shape[:-1] + (n_extra,), np.nan)], -1)
                            for f, a in v.items()}
            else:
                out[key] = np.concatenate([v, np.full(n_extra, np.nan)])
        elif isinstance(v, dict):
            merged = {}
            for f, a in v.items():
                b = lv.get(f, lv.get("r2")) if lv is not None else None
                merged[f] = np.concatenate([a, b], -1) if b is not None else a
            out[key] = merged
        else:
            out[key] = np.concatenate([v, lv], -1) if lv is not None else v
    return out


def atomic_save(obj, path: str) -> None:
    """torch.save to a temporary file, then rename over the target.

    A 1B-parameter checkpoint is about 16 GB; a preemption or walltime kill in the middle
    of writing it in place would leave a corrupt file that every later link of a resume
    chain fails to load. The rename is atomic on the same filesystem.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


DEFAULT_LIBRARY = "/home/bjc75/scratch/pangea/fragments/library_v2.tsv"


def library_metadata(path: str, sequences: Sequence[str]) -> Dict[str, np.ndarray]:
    """Species and clade for each fragment, joined on sequence from the library TSV.

    Synthetic strata (hexamers, composition controls, extremes, random) have no species;
    they come back as empty strings.
    """
    import csv
    by_seq: Dict[str, Tuple[str, str, str]] = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sp = row.get("source", "") or ""
            if row.get("stratum") == "mpra":
                sp = row.get("clade", "") or ""          # mpra rows: clade column is species
            by_seq[row["sequence"].upper()] = (row.get("stratum", ""), sp,
                                               row.get("clade", "") or "")
    st, sp, cl = [], [], []
    for q in sequences:
        a, b, c = by_seq.get(q.upper(), ("", "", ""))
        st.append(a); sp.append(b); cl.append(c)
    return {"stratum": np.asarray(st), "species": np.asarray(sp), "clade": np.asarray(cl)}


CLADE_OF_SPECIES = {
    "Homo sapiens": "animal", "Mus musculus": "animal", "Danio rerio": "animal",
    "Drosophila melanogaster": "animal", "Caenorhabditis elegans": "animal",
    "Arabidopsis thaliana": "plant", "Zea mays": "plant", "Sorghum bicolor": "plant",
    "Saccharomyces cerevisiae": "fungus", "Plasmodium falciparum": "protist",
    "Dictyostelium discoideum": "protist", "Escherichia coli": "bacterium",
    "Mycobacterium tuberculosis": "bacterium", "Streptomyces coelicolor": "bacterium",
    "Thermus thermophilus": "bacterium", "Methanocaldococcus jannaschii": "archaeon"}


def train_filter_mask(spec: Optional[str], meta: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """'species:Homo sapiens' or 'clade:bacterium,plant' -> boolean mask over fragments."""
    if not spec:
        return None
    key, _, vals = spec.partition(":")
    # Underscores stand for spaces, so a species filter survives unquoted shell expansion
    # in the Slurm scripts ("species:Homo_sapiens").
    wanted = {v.strip().replace("_", " ") for v in vals.split(",") if v.strip()}
    if key == "species":
        return np.isin(meta["species"], list(wanted))
    if key == "clade":
        clade = np.asarray([CLADE_OF_SPECIES.get(x, "") for x in meta["species"]])
        return np.isin(clade, list(wanted))
    raise ValueError(f"unknown train filter {spec!r}; use species:... or clade:...")


RARE_EVENT_MODAL_FRACTION = 0.99


@torch.no_grad()
def rare_event_channels(Y: torch.Tensor, idx: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Channels whose modal target value covers >= 99 percent of training-side values.

    A property of the targets alone, decided before any model is scored, and applied the
    same way to every run. For such a channel R-squared is set by the one percent of
    events, so it is reported separately with an event-detection AUROC.
    Returns (is_rare, modal_value) per channel.
    """
    n_ch = Y.shape[1]
    rare = np.zeros(n_ch, dtype=bool)
    modes = np.full(n_ch, np.nan, dtype=np.float32)
    sub = idx if len(idx) <= 60000 else idx[torch.randperm(len(idx), device=idx.device)[:60000]]
    for j in range(n_ch):
        v = Y[sub, j].reshape(-1).float()
        v = v[torch.isfinite(v)]
        if v.numel() < 10:
            continue
        vals, counts = torch.unique(v, return_counts=True)
        k = int(torch.argmax(counts))
        frac = float(counts[k]) / v.numel()
        modes[j] = float(vals[k])
        rare[j] = frac >= RARE_EVENT_MODAL_FRACTION
    return rare, modes


def _auroc(score: np.ndarray, label: np.ndarray) -> Optional[float]:
    """Mann-Whitney AUROC; None when one class is absent."""
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def rare_event_auroc(model, X, Y_key: str, Y: torch.Tensor, idx: torch.Tensor,
                     channels: np.ndarray, modes: np.ndarray, windows,
                     batch: int = 4096) -> Dict[int, Optional[float]]:
    """AUROC of |prediction - mode| for separating events (target != mode) from the rest."""
    out: Dict[int, Optional[float]] = {}
    if not len(channels):
        return out
    preds, tgts = [], []
    model.eval()
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            o = model(X[b].float())
        preds.append(o[Y_key][0].float()[:, channels].cpu())
        tgts.append(Y[b][:, channels].float().cpu())
    model.train()
    P_, T_ = torch.cat(preds).numpy(), torch.cat(tgts).numpy()
    for k, j in enumerate(channels):
        p_, t_ = P_[:, k].reshape(-1), T_[:, k].reshape(-1)
        ok = np.isfinite(t_)
        out[int(j)] = _auroc(np.abs(p_[ok] - modes[j]), t_[ok] != modes[j])
    return out


@torch.no_grad()
def evaluate_wide(model, WS, idx, w_pb0: int, w_gl0: int, device, batch: int = 2) -> dict:
    """Per-channel R-squared of Family W's tracks (over every labelled position) and scalars on a
    set of Family W fragments, in standardised units."""
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    nT, nS = WS.tracks_np.shape[1], WS.scalars_np.shape[1]
    sse_t, sst_t, n_t = np.zeros(nT), np.zeros(nT), np.zeros(nT)
    ys, ps = [], []
    sums = np.zeros(nT); sq = np.zeros(nT)
    pred_t = []
    for k0 in range(0, len(idx), batch):
        b = np.asarray(idx[k0:k0 + batch])
        xw = torch.from_numpy(WS.one_hot(b)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            o = model(xw)
        pm = o["per_bp"][0][:, w_pb0:w_pb0 + nT].float().cpu().numpy()
        gm = o["global"][0][:, w_gl0:w_gl0 + nS].float().cpu().numpy()
        yt = WS.tracks_np[b].astype(np.float32)
        ok = np.isfinite(yt)
        sse_t += np.where(ok, (pm - np.nan_to_num(yt)) ** 2, 0).sum((0, 2))
        sums += np.where(ok, yt, 0).sum((0, 2)); sq += np.where(ok, yt ** 2, 0).sum((0, 2)); n_t += ok.sum((0, 2))
        ys.append(WS.scalars_np[b].astype(np.float32)); ps.append(gm)
    var_t = sq / np.maximum(n_t, 1) - (sums / np.maximum(n_t, 1)) ** 2
    r2_t = 1.0 - (sse_t / np.maximum(n_t, 1)) / np.maximum(var_t, 1e-12)
    Y, Pp = np.concatenate(ys), np.concatenate(ps)
    r2_s = []
    for j in range(nS):
        ok = np.isfinite(Y[:, j])
        r2_s.append(float(1.0 - np.mean((Pp[ok, j] - Y[ok, j]) ** 2) / max(np.var(Y[ok, j]), 1e-12)) if ok.sum() > 2 else float("nan"))
    model.train()
    return {"n_fragments": int(len(idx)),
            "tracks": {n_: float(v) for n_, v in zip(WS.pb["names"], r2_t)},
            "scalars": {n_: v for n_, v in zip(WS.gl["names"], r2_s)},
            "median_track_r2": float(np.nanmedian(r2_t)), "median_scalar_r2": float(np.nanmedian(r2_s)) if r2_s else float("nan")}


@torch.no_grad()
def evaluate_alongset(model, AL, idx, c_main: int, device, batch: int = 16) -> dict:
    """Per-channel R-squared of Family A long-set columns on their labelled central stretch."""
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    cols = [c_main + c for c in AL.cols]; nC = len(cols)
    sse, s1, s2, n = (np.zeros(nC) for _ in range(4))
    for k0 in range(0, len(idx), batch):
        b = np.asarray(idx[k0:k0 + batch])
        x = torch.from_numpy(AL.one_hot(b)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            o = model(x)
        pm = o["per_bp"][0][:, cols, AL.start:AL.start + AL.stretch].float().cpu().numpy()
        y = AL.per_bp_np[b].astype(np.float32); ok = np.isfinite(y)
        sse += np.where(ok, (pm - np.nan_to_num(y)) ** 2, 0).sum((0, 2))
        s1 += np.where(ok, y, 0).sum((0, 2)); s2 += np.where(ok, y ** 2, 0).sum((0, 2)); n += ok.sum((0, 2))
    var = s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2
    r2 = 1.0 - (sse / np.maximum(n, 1)) / np.maximum(var, 1e-12)
    r2 = np.where(n > 2, r2, np.nan)
    model.train()
    return {"n_segments": int(len(idx)), "median_r2": float(np.nanmedian(r2)) if np.isfinite(r2).any() else float("nan"),
            "r2": {plan_name: float(v) for plan_name, v in zip([str(c) for c in cols], r2)}}


def _final_evaluation(args, model_cls_kwargs, plan, X, Ypb, Ygl, idx_sets, device, long=None, wide=None, alongset=None):
    """Load the best EMA weights and evaluate them on every held-out side."""
    path = os.path.join(args.out, "best_ema.pt")
    if not os.path.exists(path):
        return
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PANGEA(plan, **model_cls_kwargs).to(device)
    model.load_state_dict(ck["model"])
    out = {"levels": list(CAL_LEVELS), "best_epoch": ck.get("epoch"),
           "rare_event_rule": f"modal training-side value covers >= "
                              f"{RARE_EVENT_MODAL_FRACTION:.0%} of values"}
    tr = idx_sets.get("train")
    rare = {}
    if tr is not None and len(tr):
        for key, Y in (("per_bp", Ypb), ("global", Ygl)):
            rare[key] = rare_event_channels(Y, tr)
        names = {"per_bp": plan.per_bp_names, "global": plan.global_names}
        out["rare_event_channels"] = {k: [names[k][j] for j in np.flatnonzero(r[0])]
                                      for k, r in rare.items()}
    arrays = {}
    for side, idx in idx_sets.items():
        if side not in ("val", "test") or idx is None or len(idx) == 0:
            continue
        res = evaluate_full(model, X, Ypb, Ygl, idx, args.eval_batch_size, args.windows,
                            long=long)
        rec = {"n_fragments": int(len(idx))}
        rec.update(summarise(res["per_bp"]["r2"], plan.per_bp_names, plan.per_bp_family,
                             "per_bp"))
        rec.update(summarise(res["global"]["r2"], plan.global_names, plan.global_family,
                             "global"))
        for key in ("per_bp", "global"):
            if key in rare:
                is_rare, modes = rare[key]
                r2v = res[key]["r2"]
                if len(r2v) > len(is_rare):              # long-window channels: never rare-flagged
                    is_rare = np.concatenate([is_rare, np.zeros(len(r2v) - len(is_rare), bool)])
                keep = np.isfinite(r2v) & ~is_rare
                rec[f"{key}_median_r2_excluding_rare_events"] = (
                    float(np.median(r2v[keep])) if keep.any() else None)
                Y = Ypb if key == "per_bp" else Ygl
                au = rare_event_auroc(model, X, key, Y, idx, np.flatnonzero(is_rare), modes,
                                      args.windows, args.eval_batch_size)
                nm = plan.per_bp_names if key == "per_bp" else plan.global_names
                rec[f"{key}_rare_events"] = {nm[j]: {"r2": float(r2v[j]), "auroc": a_}
                                             for j, a_ in au.items()}
        for key, r in res.items():
            if key.startswith("window_"):
                fin = r["r2"][np.isfinite(r["r2"])]
                rec[f"{key}_median_r2"] = float(np.median(fin)) if fin.size else None
            if "r2_interior" in r:
                ri = r["r2_interior"]
                fam = plan.per_bp_family
                rec[f"{key}_median_r2_interior"] = (float(np.nanmedian(ri))
                                                    if np.isfinite(ri).any() else None)
                rec[f"{key}_median_r2_interior_by_family"] = {
                    f_: float(np.nanmedian(ri[[i for i, x in enumerate(fam) if x == f_]]))
                    for f_ in sorted(set(fam))
                    if np.isfinite(ri[[i for i, x in enumerate(fam) if x == f_]]).any()}
            cov = r["coverage"]
            rec[f"{key}_calibration_median_coverage"] = [
                float(np.nanmedian(cov[i])) for i in range(cov.shape[0])]
            rec[f"{key}_median_mean_z2"] = float(np.nanmedian(r["mean_z2"]))
            for f_, v in r.items():
                arrays[f"{side}__{key}__{f_}"] = v
        out[side] = rec
    # Windows under the pooled readout as well (window means = window average of the
    # per-bp means), which beat the dedicated window heads on validation at every ladder
    # rung and is the primary window readout; both are reported.
    model.window_readout = "pooled"
    for side in ("val", "test"):
        idx = idx_sets.get(side)
        if side not in out or idx is None or len(idx) == 0:
            continue
        res_p = evaluate_full(model, X, Ypb, Ygl, idx, args.eval_batch_size, args.windows,
                              long=long)
        for key, r in res_p.items():
            if key.startswith("window_"):
                fin = r["r2"][np.isfinite(r["r2"])]
                out[side][f"{key}_median_r2_pooled_readout"] = (float(np.median(fin))
                                                                 if fin.size else None)
                arrays[f"{side}__{key}__r2_pooled_readout"] = r["r2"]
    model.window_readout = "head"
    sp = idx_sets.get("species")
    te = idx_sets.get("test")
    if sp is not None and te is not None and len(te):
        te_np_ = te.cpu().numpy()
        out["test_by_species"] = {}
        for name in sorted(set(sp[te_np_]) - {""}):
            sel = te_np_[sp[te_np_] == name]
            if len(sel) < 100:
                continue
            r = evaluate_full(model, X, Ypb, Ygl, torch.as_tensor(sel, device=device),
                              args.eval_batch_size, args.windows, long=long)
            rec = {"n_fragments": int(len(sel)),
                   "in_training_filter": bool(idx_sets.get("species_trained", {}).get(name, True))}
            rec.update(summarise(r["per_bp"]["r2"], plan.per_bp_names, plan.per_bp_family,
                                 "per_bp"))
            rec.update(summarise(r["global"]["r2"], plan.global_names, plan.global_family,
                                 "global"))
            out["test_by_species"][name] = {k: v for k, v in rec.items()
                                            if not k.endswith("_worst")}
            arrays[f"test_species__{name}__per_bp__r2"] = r["per_bp"]["r2"]
            arrays[f"test_species__{name}__global__r2"] = r["global"]["r2"]
    if alongset is not None:
        AL_, c_main_, a_sides = alongset
        res_ = {sd: evaluate_alongset(model, AL_, ix, c_main_, device) for sd, ix in a_sides.items() if len(ix)}
        for sd, r_ in res_.items():          # column indices -> channel names
            r_["r2"] = {plan.per_bp_names[int(k)]: v for k, v in r_["r2"].items()}
        out["family_a_longset"] = res_
    if wide is not None:
        WS_, w_pb0_, w_gl0_, w_sides = wide
        out["family_w"] = {sd: evaluate_wide(model, WS_, ix, w_pb0_, w_gl0_, device)
                           for sd, ix in w_sides.items() if len(ix)}
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(args.out, "final_metrics.npz"),
             per_bp_names=np.asarray(plan.per_bp_names, dtype=object),
             global_names=np.asarray(plan.global_names, dtype=object), **arrays)
    for side in ("val", "test"):
        if side in out:
            r = out[side]
            fmt = lambda v: "nan" if v is None else f"{v:.4f}"            # noqa: E731
            print(f"final[{side}] per-bp median R2 {fmt(r['per_bp_median_r2'])}  "
                  f"global median R2 {fmt(r['global_median_r2'])}  "
                  f"(n={r['n_fragments']})", flush=True)


# ======================================================================================
# Evaluation
# ======================================================================================

@torch.no_grad()
def evaluate(model, X, Ypb, Ygl, idx, batch: int, windows: Sequence[int], long=None,
             pb_col0: int = 0, gl_col0: int = 0, pos: Optional[Tuple[int, int]] = None,
             _inner: bool = False) -> Dict[str, np.ndarray]:
    """Per-channel R-squared on held-out fragments, accumulated on the device."""
    if not _inner:
        return _pad_to_plan(evaluate(model, X, Ypb, Ygl, idx, batch, windows, long, pb_col0, gl_col0, pos, _inner=True))
    if long is not None:
        return _with_long(evaluate, model, X, Ypb, Ygl, idx, batch, windows, long, _inner=True)
    model.eval()
    acc: Dict[str, List[torch.Tensor]] = {}
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        x = X[b].float()
        tgt = {"per_bp": Ypb[b].float(), "global": Ygl[b].float()}
        for w in windows:
            tgt[f"window_{w}"] = pooled_window_targets(tgt["per_bp"], w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out = model(x)
        for key, t in tgt.items():
            m, _ = _slice_out(out[key][0].float(), None, key, t, pb_col0, gl_col0, pos)
            valid = torch.isfinite(t)
            tz = torch.where(valid, t, torch.zeros_like(t))
            mz = torch.where(valid, m, torch.zeros_like(m))
            dims = (0, 2) if t.dim() == 3 else (0,)
            parts = [((mz - tz) ** 2).sum(dims), tz.sum(dims), (tz ** 2).sum(dims),
                     valid.sum(dims).float()]
            if key not in acc:
                acc[key] = parts
            else:
                for a, p in zip(acc[key], parts):
                    a += p
    model.train()
    r2 = {}
    for key, (sse, s, sq, n) in acc.items():
        n = n.clamp(min=1.0)
        var = sq / n - (s / n) ** 2
        r = 1.0 - sse / n / var.clamp(min=1e-12)
        # R-squared is undefined where the evaluation side has no target variance -- a
        # rare-event channel with no events in the sample scored -6e10 before this.
        r2[key] = torch.where(var > MIN_EVAL_VARIANCE, r, torch.full_like(r, float("nan"))
                              ).cpu().numpy()
    return r2


def summarise(r2: np.ndarray, names: Sequence[str], families: Sequence[str],
              label: str, k_worst: int = 10) -> Dict[str, object]:
    r2 = np.asarray(r2, dtype=float)
    finite = np.isfinite(r2)
    out: Dict[str, object] = {
        f"{label}_n": int(len(names)),
        f"{label}_median_r2": float(np.median(r2[finite])) if finite.any() else None,
        f"{label}_mean_r2": float(np.mean(np.clip(r2[finite], -1, 1))) if finite.any() else None,
        f"{label}_frac_above_0.9": float(np.mean(r2[finite] > 0.9)) if finite.any() else None,
        f"{label}_frac_above_0.99": float(np.mean(r2[finite] > 0.99)) if finite.any() else None,
    }
    fam = {}
    for f in sorted(set(families)):
        m = np.array([x == f for x in families]) & finite
        if m.any():
            fam[f] = round(float(np.median(r2[m])), 4)
    out[f"{label}_median_r2_by_family"] = fam
    order = np.argsort(np.where(finite, r2, np.inf))
    out[f"{label}_worst"] = [{"channel": names[i], "r2": round(float(r2[i]), 4)}
                             for i in order[:k_worst] if finite[i]]
    return out


# ======================================================================================
# Training
# ======================================================================================


def bpp_loss(logits: torch.Tensor, q: torch.Tensor, n_real: Optional[int] = None) -> torch.Tensor:
    """Binary cross-entropy of the pair stack's base-pair logits against Vienna's matrices.

    `logits` (2B, N, N): rows [:B] the forward pass, whose own transcript is the plus
    strand (target channel 0, forward frame); rows [B:] the reverse pass, whose own
    transcript is the minus strand (target channel 1, stored in that same frame). `q` is
    the uint8 upper triangle, p = (q/255)^2. Pairs closer than the minimum hairpin are
    excluded (they cannot pair). Summed over partners and averaged over positions, so it
    sits on the per-base scale of the per-bp channels. With `n_real`, only the first
    n_real samples of each half have matrices (auxiliary rows without them are skipped).
    """
    B2, n, _ = logits.shape
    B = B2 // 2
    iu = torch.triu_indices(n, n, 1, device=logits.device)
    keep = (iu[1] - iu[0]) >= MIN_PAIR_SEPARATION
    iu = iu[:, keep]
    p = (q.float() / 255.0) ** 2                                    # (B, 2, T)
    p = p[:, :, keep]
    pred = torch.cat([logits[:B, iu[0], iu[1]], logits[B:, iu[0], iu[1]]])  # (2B, T')
    tgt = torch.cat([p[:, 0], p[:, 1]])
    if n_real is not None:
        sel = torch.cat([torch.arange(n_real, device=logits.device),
                         B + torch.arange(n_real, device=logits.device)])
        pred, tgt = pred[sel], tgt[sel]
    bce = F.binary_cross_entropy_with_logits(pred.float(), tgt, reduction="sum")
    return 2.0 * bce / (pred.shape[0] * n)       # each pair counts for both its bases

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-dilation", type=int, default=32)
    ap.add_argument("--attn-layers", type=int, default=0)
    ap.add_argument("--attn-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--global-readout", choices=("linear", "attn_mlp"), default="linear",
                    help="linear: low-rank map of the mean pool (the first design); "
                         "attn_mlp: mean, max and attention pools through an MLP")
    ap.add_argument("--global-queries", type=int, default=4)
    ap.add_argument("--pair-bias", action="store_true",
                    help="bias every attention head with a 2-D convolution over the "
                         "base-pair map (needs --attn-layers > 0)")
    ap.add_argument("--pair-channels", type=int, default=32)
    ap.add_argument("--pair-layers", type=int, default=4)
    ap.add_argument("--pair-module", choices=("conv", "axial"), default="conv",
                    help="conv: a static 2-D convolution over the base-pair map; axial: a "
                         "pair representation updated by row/column attention, which also "
                         "feeds a per-position summary to the trunk (PairStack)")
    ap.add_argument("--pair-heads", type=int, default=4)
    ap.add_argument("--strand-tie", action="store_true",
                    help="read every Family L minus-strand output from the reverse pass's "
                         "plus-strand output (exact: L_m(x) = L_p(rc x), same frame)")
    ap.add_argument("--wc-tie-global", nargs="*", default=[],
                    help="whole-sequence W/C stems whose Crick value is read from the reverse "
                         "pass's Watson value (only stems verified by rc_consistency.py)")
    ap.add_argument("--pair-bpp", action="store_true",
                    help="give the convolutional pair bias a base-pair head (needed for "
                         "--bpp-dir with --pair-module conv)")
    ap.add_argument("--bpp-dir", default=None,
                    help="per-shard base-pair probability matrices (src/data/generate_bpp.py) "
                         "supervising the axial pair stack")
    ap.add_argument("--aux-bpp-dir", default=None,
                    help="the same for the auxiliary shards")
    ap.add_argument("--bpp-weight", type=float, default=1.0,
                    help="weight of the base-pair loss (binary cross-entropy summed over "
                         "partners, averaged over positions)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="fragments per evaluation forward pass (lower it for the largest "
                         "models; it does not change any metric)")
    ap.add_argument("--ckpt-every-minutes", type=float, default=0.0,
                    help="0: checkpoint every epoch; otherwise at most this often (the "
                         "last epoch is always checkpointed)")
    ap.add_argument("--aux-shards", default=None,
                    help="auxiliary shards carrying a subset of the channels (e.g. Family L "
                         "only), mixed into every step with all other columns NaN")
    ap.add_argument("--long-shards", default=None,
                    help="long-window shards (data.LongWindowSet), aligned to the main shards; "
                         "their channels are appended to the plan and trained on windows")
    ap.add_argument("--long-per-step", type=int, default=0,
                    help="long windows per optimiser step, in their own forward pass")
    ap.add_argument("--long-weight", type=float, default=1.0)
    ap.add_argument("--wide-shards", default=None,
                    help="Family W shards (data.WideSet): 10,240 bp fragments with window-ladder and "
                         "wide-branch tracks; switches the wide branch on and joins the split groups")
    ap.add_argument("--wide-per-step", type=int, default=2,
                    help="Family W fragments per optimiser step, in their own forward pass")
    ap.add_argument("--wide-weight", type=float, default=1.0)
    ap.add_argument("--alongset-shards", default=None,
                    help="Family A long-set shards (data.ALongSet: 205 bp segments, central 105 bp labelled); "
                         "needs --long-shards, whose Family A columns they fill")
    ap.add_argument("--alongset-per-step", type=int, default=16)
    ap.add_argument("--alongset-weight", type=float, default=1.0)
    ap.add_argument("--aux-per-step", type=int, default=0,
                    help="auxiliary fragments added to each optimiser step")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="split each optimiser step's batch into this many micro-batches "
                         "(same step, less memory)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="beta-NLL exponent; 0 is the plain Gaussian NLL")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--windows", type=int, nargs="*", default=[10, 20, 50])
    ap.add_argument("--consistency-weight", type=float, default=0.1)
    ap.add_argument("--split", choices=["similarity", "random"], default="similarity")
    ap.add_argument("--allow-provisional", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--aux-extra-dirs", nargs="*", default=[],
                    help="side-shard directories aligned to the auxiliary shards (families "
                         "labelled on the auxiliary windows after they were made)")
    ap.add_argument("--rare-beta", type=float, default=None,
                    help="beta for rare-event channels (modal value >= 99 percent of training "
                         "targets); default: the global --beta for every channel")
    ap.add_argument("--aux-holdout", nargs="*", default=[],
                    help="PREFIX:FRACTION, e.g. fold_g4_:0.1 -- auxiliary windows kept out of "
                         "training by a hash of their id, for scoring that window set")
    ap.add_argument("--aux-extra-optional", action="store_true",
                    help="a side directory may lack some auxiliary shards (a family labelled "
                         "on one auxiliary set only); those channels are NaN there")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="side-shard directories of families added after the corpus "
                         "(src/data/generate_family_shards.py), merged per fragment")
    ap.add_argument("--g-interior-trim", type=int, default=0,
                    help="recompute Family G's whole-sequence strand statistics over positions "
                         "[k, n-k) (target revision v2.1: removes the terminal-residue artifact; "
                         "DECISIONS.md)")
    ap.add_argument("--library", default=DEFAULT_LIBRARY,
                    help="library TSV, for species and clade of each fragment")
    ap.add_argument("--train-filter", default=None,
                    help="train (and select) only on fragments matching 'species:A,B' or "
                         "'clade:bacterium,plant'; the test side stays over every species, "
                         "so held-out species are scored zero-shot by the same protocol")
    ap.add_argument("--test-fraction", type=float, default=0.0,
                    help="held-out test side, carved before validation and never used "
                         "for training or model selection; 0 keeps the two-way split")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed of the train/val/test split (default: --seed). Fixing it while "
                         "--seed varies measures initialisation and batch-order variance on "
                         "an identical split")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-equivariance", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    # Data parallelism, only when launched by torchrun (WORLD_SIZE > 1). Every rank holds
    # the whole dataset on its own GPU and trains on its slice of each global batch; the
    # single-process path below is unchanged.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # RANK and LOCAL_RANK are read only under torchrun. The Slurm scripts pass the model's
    # low-rank head size in an environment variable that is also called RANK, and reading
    # it in a single-process run made rank 0 look like rank 64 -- which would have skipped
    # every evaluation, history line and checkpoint without an error.
    rank = int(os.environ.get("RANK", "0")) if world > 1 else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if world > 1 else 0
    is_main = rank == 0
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if args.batch_size % world:
            raise ValueError(f"batch size {args.batch_size} not divisible by {world} ranks")
    cuda_ok = torch.cuda.is_available()
    print(f"torch {torch.__version__} (bundled CUDA {torch.version.cuda}); "
          f"cuda available: {cuda_ok}", flush=True)
    if cuda_ok:
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif args.require_cuda:
        raise RuntimeError("CUDA unavailable and --require-cuda was given")
    device = (f"cuda:{local_rank}" if world > 1 else "cuda") if cuda_ok else "cpu"

    t0 = time.time()
    ds = InMemoryLandscape(args.shards, cache_dir=args.cache_dir,
                           drop_provisional=not args.allow_provisional,
                           max_shards=args.max_shards, interior_trim_g=args.g_interior_trim,
                           extra_dirs=args.extra_dirs)
    X = torch.from_numpy(ds.one_hot()).to(device)
    Ypb = torch.from_numpy(ds.per_bp_np).to(device)
    Ygl = torch.from_numpy(ds.global_np).to(device)
    aux = None
    Bm = Ba = None
    if args.bpp_dir:
        if args.pair_module != "axial" and not args.pair_bpp:
            raise SystemExit("--bpp-dir needs --pair-module axial or --pair-bpp")
        Bm = torch.from_numpy(load_bpp(args.bpp_dir, ds.paths, ds.sequences)).to(device)
        print(f"base-pair probabilities: {tuple(Bm.shape)} for the main fragments", flush=True)
    if args.aux_shards and args.aux_per_step > 0:
        from data import AuxLandscape
        ho = {k: float(v) for k, v in (x.split(":") for x in args.aux_holdout)} if args.aux_holdout else None
        aux = AuxLandscape(args.aux_shards, ds, extra_dirs=args.aux_extra_dirs or None,
                           optional_extra=args.aux_extra_optional, holdout=ho)
        if aux.holdout_ids:
            with open(os.path.join(args.out, "aux_holdout_ids.txt"), "w") as f:
                f.write("\n".join(aux.holdout_ids) + "\n")
            print(f"  auxiliary hold-out: {len(aux.holdout_ids)} windows kept out of training", flush=True)
        if args.aux_bpp_dir:
            Ba = torch.from_numpy(load_bpp(args.aux_bpp_dir, aux.paths, aux.sequences)).to(device)
        Xa = torch.from_numpy(aux.one_hot()).to(device)
        Ya_pb = torch.from_numpy(aux.per_bp_np).to(device)
        Ya_gl = torch.from_numpy(aux.global_np).to(device)
        aux_pb_cols = torch.as_tensor(aux.pb_cols, device=device)
        aux_gl_cols = torch.as_tensor(aux.gl_cols, device=device)
    LT = None
    pb_info, gl_info = ds.pb, ds.gl
    if args.long_shards:
        from data import LongWindowSet, extend_channel_info
        Lset = LongWindowSet(args.long_shards, ds)
        LT = LongTensors(Lset, device, Ypb.shape[1], Ygl.shape[1])
        pb_info = extend_channel_info(ds.pb, Lset.pb)
        gl_info = extend_channel_info(ds.gl, Lset.gl)
        n_long_pb, n_long_gl = LT.Ypb.shape[1], LT.Ygl.shape[1]
    AL = None
    if args.alongset_shards:
        if LT is None:
            raise ValueError("--alongset-shards needs --long-shards (their Family A columns)")
        from data import ALongSet
        AL = ALongSet(args.alongset_shards, Lset.pb)
    WS = None
    if args.wide_shards:
        from data import WideSet, extend_channel_info
        WS = WideSet(args.wide_shards, drop_provisional=not args.allow_provisional)
        w_pb0, w_gl0 = len(pb_info["names"]), len(gl_info["names"])
        pb_info = extend_channel_info(pb_info, WS.pb)
        gl_info = extend_channel_info(gl_info, WS.gl)
    N = X.shape[0]
    print(f"{N} fragments on {device} in {time.time()-t0:.0f}s; "
          f"{Ypb.shape[1]} per-bp and {Ygl.shape[1]} global target columns", flush=True)
    print("transforms: " + json.dumps({k: (len(v) if isinstance(v, list) else v)
                                       for k, v in ds.transforms.items()}), flush=True)

    groups_main = ds.groups
    if WS is not None and args.split == "similarity":
        groups_main, w_groups, wrep = WS.join_groups(ds)
        print("Family W grouping: " + json.dumps(wrep), flush=True)
    tr_np, va_np, te_np, split_report = three_way_split(
        N, groups_main if args.split == "similarity" else None, args.val_fraction,
        args.test_fraction, args.seed if args.split_seed is None else args.split_seed,
        args.split, force_train=(WS.force_train_groups if WS is not None else None))
    if WS is not None:
        import hashlib
        side = np.full(len(WS.sequences), -1, dtype=np.int64)
        g2side = {}
        if args.split == "similarity":
            for arr, sd in ((tr_np, 0), (va_np, 1), (te_np, 2)):
                for i in arr:
                    g2side[int(groups_main[i])] = sd
        for k in range(len(WS.sequences)):
            g = int(w_groups[k]) if args.split == "similarity" else None
            if g is not None and g in g2side:
                side[k] = g2side[g]             # with the library fragments it contains
            else:
                u = int(hashlib.sha1(WS.frag_ids[k].encode()).hexdigest()[:12], 16) / float(1 << 48)
                side[k] = 2 if u < args.test_fraction else (1 if u < args.test_fraction + args.val_fraction else 0)
        w_tr, w_va, w_te = (np.flatnonzero(side == sd) for sd in (0, 1, 2))
        split_report["family_w"] = {"train": int(len(w_tr)), "val": int(len(w_va)), "test": int(len(w_te))}
    if AL is not None:
        from data import main_frag_ids
        al_tr, al_va, al_te = AL.sides(main_frag_ids(ds), tr_np, va_np, te_np)
        split_report["family_a_longset"] = {"train": int(len(al_tr)), "val": int(len(al_va)), "test": int(len(al_te))}
    meta = library_metadata(args.library, ds.sequences) if os.path.exists(args.library) \
        else None
    if args.train_filter:
        if meta is None:
            raise FileNotFoundError(f"--train-filter needs the library at {args.library}")
        keep = train_filter_mask(args.train_filter, meta)
        tr_np, va_np = tr_np[keep[tr_np]], va_np[keep[va_np]]
        split_report.update({"train_filter": args.train_filter,
                             "n_train": int(len(tr_np)), "n_val": int(len(va_np))})
        if len(tr_np) == 0 or len(va_np) == 0:
            raise ValueError(f"train filter {args.train_filter!r} left nothing to train on")
    print("split: " + json.dumps(split_report), flush=True)
    os.makedirs(args.out, exist_ok=True)
    if is_main:
        np.savez(os.path.join(args.out, "split_indices.npz"), train=tr_np, val=va_np,
                 test=te_np)
    tr_idx = torch.as_tensor(tr_np, device=device)
    va_idx = torch.as_tensor(va_np, device=device)
    te_idx = torch.as_tensor(te_np, device=device) if len(te_np) else None

    plan = ChannelPlan.from_metadata(pb_info["names"], gl_info["names"], pb_info["parity"],
                                     gl_info["parity"], pb_info["family"], gl_info["family"],
                                     pb_info["rc_shift"])
    model_kwargs = model_kwargs_from_args(vars(args))
    if WS is not None:
        global _PLAN_WIDTH
        _PLAN_WIDTH = (len(plan.per_bp_names), len(plan.global_names))
        model_kwargs["use_wide_branch"] = True
        rungs = {}
        for j, w in WS.window_of.items():
            rungs.setdefault(int(w), []).append(w_pb0 + j)
        model_kwargs["pb_window_rungs"] = rungs
    model = PANGEA(plan, **model_kwargs).to(device)
    beta_pb = beta_gl = None
    if args.rare_beta is not None:
        # the trainer's own rare-event rule (modal value >= 99 percent), on the training side
        rp, _ = rare_event_channels(Ypb, tr_idx); rg, _ = rare_event_channels(Ygl, tr_idx)
        n_pb_main, n_gl_main = Ypb.shape[1], Ygl.shape[1]
        beta_pb = [args.rare_beta if (j < n_pb_main and rp[j]) else args.beta
                   for j in range(len(pb_info["family"]))]
        beta_gl = [args.rare_beta if (j < n_gl_main and rg[j]) else args.beta
                   for j in range(len(gl_info["family"]))]
        print(f"rare-event beta {args.rare_beta}: {int(rp.sum())} per-bp and {int(rg.sum())} "
              f"whole-sequence channels", flush=True)
    loss_fn = LandscapeLoss(pb_info["family"], gl_info["family"], beta=args.beta,
                            beta_per_bp=beta_pb, beta_global=beta_gl).to(device)
    if LT is not None:
        long_tr = tr_idx[LT.has[tr_idx]]
        if len(long_tr) == 0:
            raise ValueError("no labelled long window on the training side")
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # Unused parameters exist by design (FiLM layers when no condition is given).
        fwd_model = DDP(model, device_ids=[local_rank] if cuda_ok else None,
                        find_unused_parameters=True)
    else:
        fwd_model = model
    print(f"receptive field {model.trunk.receptive_field()} bp, "
          f"{model.n_parameters()/1e6:.2f}M parameters, attention layers "
          f"{args.attn_layers}, rank {args.rank}", flush=True)

    steps_per_epoch = max(1, len(tr_np) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_epochs * steps_per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p))) * 0.98 + 0.02
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start_epoch, step = 0, 0
    ckpt = os.path.join(args.out, "checkpoint.pt")
    if args.resume and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        if ck.get("per_bp_names") != plan.per_bp_names or \
                ck.get("global_names") != plan.global_names:
            raise RuntimeError("checkpoint was written for a different channel set")
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, step = int(ck["epoch"]) + 1, int(ck["step"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=step - 1)
        resumed_best = ck.get("best_score")
        if resumed_best is None:
            # Checkpoints written before the best score was stored: recover it from the
            # run's own history, which records the same score for every evaluation.
            hist = os.path.join(args.out, "history.jsonl")
            if os.path.exists(hist):
                scores = []
                for line in open(hist):
                    r = json.loads(line)
                    if r.get("per_bp_median_r2") is not None:
                        scores.append((r.get("per_bp_median_r2") or 0)
                                      + (r.get("global_median_r2") or 0))
                resumed_best = max(scores) if scores else None
        # With periodic checkpoints the best model can be newer than the checkpoint, so
        # the best score is the larger of the two records; and history lines written
        # after the checkpoint are dropped, since those epochs are about to be re-run.
        bsj = os.path.join(args.out, "best_score.json")
        if os.path.exists(bsj):
            sc = json.load(open(bsj)).get("score")
            if sc is not None:
                resumed_best = sc if resumed_best is None else max(resumed_best, sc)
        hist = os.path.join(args.out, "history.jsonl")
        if is_main and os.path.exists(hist):
            lines = [l for l in open(hist) if l.strip()]
            kept = [l for l in lines if json.loads(l).get("epoch", -1) < start_epoch]
            if len(kept) != len(lines):
                with open(hist, "w") as fh:
                    fh.writelines(kept)
        print(f"resumed at epoch {start_epoch}, step {step}, best score "
              f"{resumed_best}", flush=True)
    else:
        resumed_best = None
    idx_sets = {"val": va_idx, "test": te_idx, "train": tr_idx}
    if meta is not None:
        idx_sets["species"] = meta["species"]
        trained = set(meta["species"][tr_np].tolist())
        idx_sets["species_trained"] = {n: (n in trained) for n in set(meta["species"])}
    if start_epoch >= args.epochs:
        print("nothing to do", flush=True)
        if is_main and not os.path.exists(os.path.join(args.out, "final_metrics.json")):
            _final_evaluation(args, model_kwargs, plan, X, Ypb, Ygl, idx_sets, device, long=LT,
                              wide=(WS, w_pb0, w_gl0, {"val": w_va, "test": w_te}) if WS is not None else None,
                              alongset=(AL, LT.C_main, {"val": al_va, "test": al_te}) if AL is not None else None)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return 0

    with open(os.path.join(args.out, "config.json") if is_main else os.devnull, "w") as f:
        json.dump({"args": vars(args), "n_parameters": model.n_parameters(),
                   "split": split_report, "transforms": ds.transforms,
                   "per_bp_names": plan.per_bp_names,
                   "global_names": plan.global_names}, f, indent=1)

    # **The best score survives a resume.** It used to restart at -1e9 in every new
    # process, so after a preemption or a walltime resume the first evaluation always
    # overwrote best_ema.pt, even when an earlier epoch had scored higher.
    best = resumed_best if resumed_best is not None else -1e9
    last_ckpt = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        if world > 1:
            g = torch.Generator().manual_seed(args.seed * 100003 + epoch)
            perm = tr_idx[torch.randperm(len(tr_idx), generator=g).to(device)]
        else:
            perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        run = torch.zeros((), device=device)
        run_bpp = torch.zeros((), device=device)
        per_rank = args.batch_size // world
        accum = max(1, args.accum_steps)
        for k in range(steps_per_epoch):
            b_all = perm[k * args.batch_size + rank * per_rank:
                         k * args.batch_size + (rank + 1) * per_rank]
            opt.zero_grad(set_to_none=True)
            chunks = b_all.chunk(accum)
            for mi, b in enumerate(chunks):
                x = X[b].float()
                tgt = {"per_bp": Ypb[b].float(), "global": Ygl[b].float()}
                if aux is not None:
                    na = max(1, args.aux_per_step // len(chunks))
                    ai = torch.randint(0, Xa.shape[0], (na,), device=device)
                    pb_a = torch.full((na,) + tuple(Ypb.shape[1:]), float("nan"),
                                      device=device)
                    pb_a[:, aux_pb_cols] = Ya_pb[ai].float()
                    gl_a = torch.full((na, Ygl.shape[1]), float("nan"), device=device)
                    gl_a[:, aux_gl_cols] = Ya_gl[ai].float()
                    x = torch.cat([x, Xa[ai].float()])
                    tgt = {"per_bp": torch.cat([tgt["per_bp"], pb_a]),
                           "global": torch.cat([tgt["global"], gl_a])}
                bpp_t = None
                if Bm is not None:
                    bpp_t = Bm[b]
                    if aux is not None:
                        bpp_t = torch.cat([bpp_t, Ba[ai] if Ba is not None else
                                           torch.full((len(ai),) + tuple(Bm.shape[1:]), 255,
                                                      dtype=torch.uint8, device=device)])
                if LT is not None:                         # long-window columns: NaN here
                    nb_ = tgt["per_bp"].shape[0]
                    tgt["per_bp"] = torch.cat([tgt["per_bp"], torch.full(
                        (nb_, n_long_pb, tgt["per_bp"].shape[-1]), float("nan"), device=device)], 1)
                    tgt["global"] = torch.cat([tgt["global"], torch.full(
                        (nb_, n_long_gl), float("nan"), device=device)], 1)
                if WS is not None:                         # Family W columns: NaN here
                    nb_ = tgt["per_bp"].shape[0]
                    tgt["per_bp"] = torch.cat([tgt["per_bp"], torch.full(
                        (nb_, WS.tracks_np.shape[1], tgt["per_bp"].shape[-1]), float("nan"), device=device)], 1)
                    tgt["global"] = torch.cat([tgt["global"], torch.full(
                        (nb_, WS.scalars_np.shape[1]), float("nan"), device=device)], 1)
                for w in args.windows:
                    tgt[f"window_{w}"] = pooled_window_targets(tgt["per_bp"], w)
                last = mi == len(chunks) - 1
                sync = (fwd_model.no_sync() if (world > 1 and not last)
                        else contextlib.nullcontext())
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_ok):
                        out = fwd_model(x, equivariant=not args.no_equivariance,
                                        return_bpp=bpp_t is not None)
                    bpp_logits = out.pop("bpp_logits", None)
                    loss, _ = loss_fn(out, tgt)
                    if bpp_t is not None:
                        lb = bpp_loss(bpp_logits, bpp_t, n_real=len(b) if Ba is None else None)
                        loss = loss + args.bpp_weight * lb
                        run_bpp += lb.detach() / len(chunks)
                    if LT is not None and args.long_per_step > 0:
                        nl = max(1, args.long_per_step // len(chunks))
                        li = long_tr[torch.randint(0, len(long_tr), (nl,), device=device)]
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_ok):
                            outl = fwd_model(LT.X[li].float(), equivariant=not args.no_equivariance)
                        outl.pop("bpp_logits", None)
                        p0, p1 = LT.pos
                        tl_pb = torch.full((nl, len(plan.per_bp_names), p1 - p0), float("nan"),
                                           device=device)
                        tl_pb[:, LT.C_main:] = LT.Ypb[li].float()
                        tl_gl = torch.full((nl, len(plan.global_names)), float("nan"), device=device)
                        tl_gl[:, LT.G_main:] = LT.Ygl[li].float()
                        ol = {"per_bp": (outl["per_bp"][0][..., p0:p1], outl["per_bp"][1][..., p0:p1]),
                              "global": outl["global"]}
                        lossl, _ = loss_fn(ol, {"per_bp": tl_pb, "global": tl_gl})
                        loss = loss + args.long_weight * lossl
                    if AL is not None and args.alongset_per_step > 0 and len(al_tr):
                        na_ = max(1, args.alongset_per_step // len(chunks))
                        si = np.random.choice(al_tr, na_)
                        xs = torch.from_numpy(AL.one_hot(si)).to(device)
                        ts = torch.full((na_, len(plan.per_bp_names), AL.length), float("nan"), device=device)
                        cols = torch.as_tensor([LT.C_main + c for c in AL.cols], device=device)
                        ts[:, cols, AL.start:AL.start + AL.stretch] = torch.from_numpy(AL.per_bp_np[si]).to(device).float()
                        tg = torch.full((na_, len(plan.global_names)), float("nan"), device=device)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_ok):
                            outs = fwd_model(xs, equivariant=not args.no_equivariance)
                        outs.pop("bpp_logits", None)
                        losss, _ = loss_fn({"per_bp": outs["per_bp"], "global": outs["global"]}, {"per_bp": ts, "global": tg})
                        loss = loss + args.alongset_weight * losss
                    if WS is not None and args.wide_per_step > 0 and len(w_tr):
                        nw = max(1, args.wide_per_step // len(chunks))
                        wi = np.random.choice(w_tr, nw)
                        xw = torch.from_numpy(WS.one_hot(wi)).to(device)
                        tw_pb = torch.full((nw, len(plan.per_bp_names), WS.length), float("nan"), device=device)
                        tw_pb[:, w_pb0:w_pb0 + WS.tracks_np.shape[1]] = torch.from_numpy(WS.tracks_np[wi]).to(device).float()
                        tw_gl = torch.full((nw, len(plan.global_names)), float("nan"), device=device)
                        tw_gl[:, w_gl0:w_gl0 + WS.scalars_np.shape[1]] = torch.from_numpy(WS.scalars_np[wi]).to(device).float()
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_ok):
                            outw = fwd_model(xw, equivariant=not args.no_equivariance)
                        outw.pop("bpp_logits", None)
                        lossw, _ = loss_fn({"per_bp": outw["per_bp"], "global": outw["global"]},
                                           {"per_bp": tw_pb, "global": tw_gl})
                        loss = loss + args.wide_weight * lossw
                    if args.consistency_weight > 0 and args.windows:
                        pb_mean = out["per_bp"][0].float()
                        cons = torch.stack([
                            F.mse_loss(out[f"window_{w}"][0].float(),
                                       pooled_window_targets(pb_mean, w).detach().nan_to_num())
                            for w in args.windows]).mean()
                        loss = loss + args.consistency_weight * cons
                    (loss / len(chunks)).backward()
                run += loss.detach() / len(chunks)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update_parameters(model)
            step += 1
        train_loss = float(run) / steps_per_epoch
        rec: Dict[str, object] = {"epoch": epoch, "step": step, "train_loss": train_loss,
                                  **({"train_bpp_loss": float(run_bpp) / steps_per_epoch}
                                     if Bm is not None else {}),
                                  "lr": opt.param_groups[0]["lr"],
                                  "seconds": round(time.time() - t_ep, 1)}
        if world > 1:
            dist.all_reduce(run)
            run /= world
        if not is_main:
            if world > 1:
                dist.barrier()
            continue
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            r2 = evaluate(ema.module, X, Ypb, Ygl, va_idx, args.eval_batch_size,
                          args.windows, long=LT)
            rec.update(summarise(r2["per_bp"], plan.per_bp_names, plan.per_bp_family,
                                 "per_bp"))
            rec.update(summarise(r2["global"], plan.global_names, plan.global_family,
                                 "global"))
            for w in args.windows:
                fin = r2[f"window_{w}"][np.isfinite(r2[f"window_{w}"])]
                rec[f"window_{w}_median_r2"] = float(np.median(fin)) if fin.size else None
            score = (rec["per_bp_median_r2"] or 0) + (rec["global_median_r2"] or 0)
            if score > best:
                best = score
                np.savez(os.path.join(args.out, "best_r2.npz"), **r2,
                         per_bp_names=np.asarray(plan.per_bp_names, dtype=object),
                         global_names=np.asarray(plan.global_names, dtype=object))
                atomic_save({"model": ema.module.state_dict(), "plan": plan.__dict__,
                             "args": vars(args), "epoch": epoch, "score": score},
                            os.path.join(args.out, "best_ema.pt"))
                with open(os.path.join(args.out, "best_score.json"), "w") as fbs:
                    json.dump({"epoch": epoch, "score": score}, fbs)
        print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_worst")}),
              flush=True)
        with open(os.path.join(args.out, "history.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        due = (args.ckpt_every_minutes <= 0 or epoch == args.epochs - 1
               or time.time() - last_ckpt >= 60.0 * args.ckpt_every_minutes)
        if due:
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                         "optimizer": opt.state_dict(), "epoch": epoch, "step": step,
                         "best_score": best,
                         "per_bp_names": plan.per_bp_names,
                         "global_names": plan.global_names}, ckpt)
            last_ckpt = time.time()
        if world > 1:
            dist.barrier()                     # matches the non-main ranks' barrier
    if world > 1:
        dist.barrier()
        if not is_main:
            dist.destroy_process_group()
            return 0
    print("training complete", flush=True)
    _final_evaluation(args, model_kwargs, plan, X, Ypb, Ygl, idx_sets, device, long=LT,
                      wide=(WS, w_pb0, w_gl0, {"val": w_va, "test": w_te}) if WS is not None else None,
                      alongset=(AL, LT.C_main, {"val": al_va, "test": al_te}) if AL is not None else None)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
