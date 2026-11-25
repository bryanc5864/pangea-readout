#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Reading the generated landscape shards into training batches.

**The channel layout comes from the dataset, not from the physics.** Every shard stores
its own channel names, families, parities and units as HDF5 attributes. A training job
reads those rather than rebuilding them from the channel registry, so a change to the
physics between target generation and training cannot silently re-map columns. The
training environment does not even have the physics stack installed, which makes the
separation structural rather than a matter of discipline.

**Window targets are derived, and that is the plan's inter-scale relation, not a
shortcut.** `PLAN.md`: "Where an exact relation between scales exists, such as integrated
charge over a window being the sum of its per-base contributions, it is imposed as a
consistency loss rather than left to chance, so the scales agree with each other by
construction." For a mean-poolable channel the window value *is* the pooled per-base
value, so the window target is that pool, computed NaN-aware so a window overlapping an
undefined position is not quietly dragged toward zero.

Channels that are not mean-poolable -- persistence length over a window is not the mean
of per-base persistence lengths -- are a different case, and the plan treats them as
"predicted independently and the approximate relation used as a diagnostic". Those live
in the whole-sequence head here, because the current 61 bp fragments cannot carry a
genuine window-resolved version of them.
"""
from __future__ import annotations

import glob
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

BASES = "ACGT"
_BASE_INDEX = {b: i for i, b in enumerate(BASES)}


def _decode(arr) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


@dataclass
class ChannelMeta:
    per_bp_names: List[str]
    global_names: List[str]
    per_bp_parity: List[str]
    global_parity: List[str]
    per_bp_family: List[str]
    global_family: List[str]
    per_bp_provisional: np.ndarray
    global_provisional: np.ndarray

    @staticmethod
    def from_file(path: str) -> "ChannelMeta":
        with h5py.File(path, "r") as h:
            a = h.attrs
            return ChannelMeta(
                _decode(a["per_bp_names"]), _decode(a["global_names"]),
                _decode(a["per_bp_parity"]), _decode(a["global_parity"]),
                _decode(a["per_bp_family"]), _decode(a["global_family"]),
                np.asarray(a["per_bp_provisional"], dtype=bool),
                np.asarray(a["global_provisional"], dtype=bool))


class LandscapeShards(Dataset):
    """All generated shards as one indexable dataset.

    Shards are opened lazily per worker: an h5py handle cannot be shared across a fork,
    and sharing one is the standard way to get silent data corruption in a DataLoader
    with `num_workers > 0`.

    `drop_provisional` is on by default. A provisional channel is one whose *parameters*
    do not trace to a published source -- `channels.ChannelSpec` says the trainer is
    expected to refuse them unless explicitly allowed -- so training on them by default
    would make the project's own rule advisory.
    """

    def __init__(self, pattern: str, drop_provisional: bool = True,
                 standardise: bool = True, max_shards: Optional[int] = None):
        self.paths = sorted(glob.glob(pattern))
        if max_shards:
            self.paths = self.paths[:max_shards]
        if not self.paths:
            raise FileNotFoundError(f"no shards matched {pattern}")
        self.meta = ChannelMeta.from_file(self.paths[0])

        counts, self.offsets = [], []
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as h:
                n = int(h.attrs["n_fragments"])
                if _decode(h.attrs["per_bp_names"]) != self.meta.per_bp_names:
                    raise ValueError(
                        f"{p} has a different channel layout from {self.paths[0]}; "
                        "shards from different generator versions cannot be mixed")
            counts.append(n)
            self.offsets.append(total)
            total += n
        self.counts = counts
        self.total = total
        self._handles: Dict[int, h5py.File] = {}

        keep_pb = ~self.meta.per_bp_provisional if drop_provisional else \
            np.ones(len(self.meta.per_bp_names), dtype=bool)
        keep_gl = ~self.meta.global_provisional if drop_provisional else \
            np.ones(len(self.meta.global_names), dtype=bool)
        self.keep_pb = np.flatnonzero(keep_pb)
        self.keep_gl = np.flatnonzero(keep_gl)

        self.per_bp_names = [self.meta.per_bp_names[i] for i in self.keep_pb]
        self.global_names = [self.meta.global_names[i] for i in self.keep_gl]
        self.per_bp_parity = [self.meta.per_bp_parity[i] for i in self.keep_pb]
        self.global_parity = [self.meta.global_parity[i] for i in self.keep_gl]
        self.per_bp_family = [self.meta.per_bp_family[i] for i in self.keep_pb]
        self.global_family = [self.meta.global_family[i] for i in self.keep_gl]

        self.standardise = standardise
        self.pb_mean = self.pb_std = self.gl_mean = self.gl_std = None

    def __len__(self) -> int:
        return self.total

    def _handle(self, k: int) -> h5py.File:
        h = self._handles.get(k)
        if h is None:
            h = h5py.File(self.paths[k], "r")
            self._handles[k] = h
        return h

    def _locate(self, i: int) -> Tuple[int, int]:
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return k, i - self.offsets[k]

    def fit_standardiser(self, n_sample: int = 4000, seed: int = 0) -> None:
        """Per-channel mean and standard deviation, from a NaN-aware sample.

        Channels span many orders of magnitude -- kcal/mol beside probabilities beside
        angstrom -- so an unstandardised Gaussian NLL would be dominated by whichever
        channel happens to have the largest units. A channel with no finite values in
        the sample gets mean 0 and scale 1, so it contributes nothing rather than
        producing a division by zero.
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.total, size=min(n_sample, self.total), replace=False)
        P = np.stack([self[int(i)]["per_bp"].numpy() for i in idx])
        G = np.stack([self[int(i)]["global"].numpy() for i in idx])
        with np.errstate(invalid="ignore"):
            self.pb_mean = np.nan_to_num(np.nanmean(P, axis=(0, 2)))
            self.pb_std = np.nan_to_num(np.nanstd(P, axis=(0, 2)), nan=1.0)
            self.gl_mean = np.nan_to_num(np.nanmean(G, axis=0))
            self.gl_std = np.nan_to_num(np.nanstd(G, axis=0), nan=1.0)
        self.pb_std[self.pb_std < 1e-8] = 1.0
        self.gl_std[self.gl_std < 1e-8] = 1.0

    def all_sequences(self) -> List[str]:
        """Every fragment's sequence, in dataset order.

        Read straight from the shards rather than through `__getitem__`, which would
        decode and standardise the full target arrays for a job that only needs the
        strings. The similarity split needs all of them before training starts.
        """
        out: List[str] = []
        for k in range(len(self.paths)):
            out.extend(_decode(self._handle(k)["sequence"]))
        return out

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        k, j = self._locate(i)
        h = self._handle(k)
        seq = h["sequence"][j]
        seq = seq.decode() if isinstance(seq, bytes) else str(seq)
        P = np.asarray(h["per_bp"][j][self.keep_pb], dtype=np.float32)
        G = np.asarray(h["global"][j][self.keep_gl], dtype=np.float32)
        if self.standardise and self.pb_mean is not None:
            P = (P - self.pb_mean[:, None]) / self.pb_std[:, None]
            G = (G - self.gl_mean) / self.gl_std

        x = np.zeros((4, len(seq)), dtype=np.float32)
        for t, b in enumerate(seq.upper()):
            c = _BASE_INDEX.get(b)
            if c is not None:
                x[c, t] = 1.0
        return {"x": torch.from_numpy(x), "per_bp": torch.from_numpy(P),
                "global": torch.from_numpy(G), "index": torch.tensor(i)}


def pooled_window_targets(per_bp: torch.Tensor, window: int) -> torch.Tensor:
    """NaN-aware centred moving average of the per-base targets.

    This is the window target for every mean-poolable channel, and it is the exact
    inter-scale relation rather than an approximation of one. Positions are kept NaN
    where the whole window is undefined, so an end-of-fragment window does not become a
    spurious zero.
    """
    valid = torch.isfinite(per_bp)
    filled = torch.where(valid, per_bp, torch.zeros_like(per_bp))
    k = min(int(window), per_bp.shape[-1])
    pad_l, pad_r = (k - 1) // 2, k - 1 - (k - 1) // 2
    num = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(filled, (pad_l, pad_r), mode="constant", value=0.0),
        kernel_size=k, stride=1) * k
    den = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(valid.float(), (pad_l, pad_r), mode="constant",
                                value=0.0),
        kernel_size=k, stride=1) * k
    out = num / den.clamp(min=1.0)
    return torch.where(den > 0, out, torch.full_like(out, float("nan")))


# ======================================================================================
# The whole dataset in memory, transformed once
# ======================================================================================

class InMemoryLandscape:
    """Every fragment's targets as dense tensors, ready to live on the GPU.

    **Why this replaced the shard reader for training.** The first full run took 606 s an
    epoch on a B200 for a 2M-parameter model. Two causes, both avoidable: each batch
    gathered 128 random rows from 458 gzip-compressed HDF5 files over NFS, and the loss
    looped over about a thousand channels in Python with a host synchronisation each. The
    whole corpus is under 5 GB once standardised to float16, which is small next to a
    B200's 183 GB, so it is loaded once and indexed on the device.

    **Target transforms, each decided from the data and recorded.**

      * Channels that are constant or almost entirely NaN across the corpus are dropped.
        A constant channel has no information to learn, and it actively harmed the first
        run: the NLL rewards collapsing the variance on a constant without bound, which
        drove the training loss to -31, and its R-squared came out at -13 million.
      * Circular channels -- declared in the registry -- become (sin, cos). A Gaussian on
        raw degrees is wrong at the wrap, where -179 and +179 are two degrees apart.
      * Strictly positive channels spanning more than two decades become log10. Base-pair
        opening probabilities run from 1e-8 to 1e-2; on a linear scale the fit is a
        contest over the handful of frayed ends, while the physically meaningful scale is
        the free energy, which is the logarithm. Which channels were transformed is
        written into every run's record, so a reported R-squared always says what scale
        it was computed on.
      * Everything is then standardised per channel.

    The transformed, standardised arrays are cached to one uncompressed file, so the
    scaling study's many runs load in seconds rather than re-reading 512 shards.
    """

    VERSION = 3

    def __init__(self, pattern: str, cache_dir: Optional[str] = None,
                 drop_provisional: bool = True, log_decades: float = 2.0,
                 min_finite_frac: float = 1e-3, max_shards: Optional[int] = None,
                 similarity_threshold: float = 0.5, verbose: bool = True,
                 interior_trim_g: int = 0, extra_dirs: Optional[Sequence[str]] = None):
        self.interior_trim_g = int(interior_trim_g)
        self.extra_dirs = [os.path.abspath(d) for d in (extra_dirs or [])]
        self.paths, self.cache = self.cache_file(pattern, cache_dir, drop_provisional,
                                                 log_decades, min_finite_frac, max_shards,
                                                 similarity_threshold, interior_trim_g,
                                                 self.extra_dirs)
        if os.path.exists(self.cache):
            self._load_cache(verbose)
        else:
            self._build(drop_provisional, log_decades, min_finite_frac,
                        similarity_threshold, verbose)
            self._save_cache(verbose)

    @classmethod
    def cache_file(cls, pattern, cache_dir=None, drop_provisional=True, log_decades=2.0,
                   min_finite_frac=1e-3, max_shards=None, similarity_threshold=0.5,
                   interior_trim_g=0, extra_dirs=None):
        """(shard paths, cache path) for these options -- without loading anything."""
        import hashlib
        paths = sorted(glob.glob(pattern))
        if max_shards:
            paths = paths[:max_shards]
        if not paths:
            raise FileNotFoundError(f"no shards matched {pattern}")
        key = hashlib.sha1(("|".join(os.path.basename(p) for p in paths)
                            + f"|v{cls.VERSION}|{drop_provisional}|{log_decades}"
                            f"|{min_finite_frac}|{similarity_threshold}"
                            + (f"|gtrim{int(interior_trim_g)}" if interior_trim_g
                               else "")
                            + "".join(f"|extra:{os.path.basename(os.path.abspath(d))}"
                                      for d in (extra_dirs or []))).encode()
                           ).hexdigest()[:16]
        cache_dir = cache_dir or os.path.dirname(os.path.abspath(paths[0]))
        return paths, os.path.join(cache_dir, f"landscape_cache_{key}.npz")

    # ----------------------------------------------------------------------------------
    def _build(self, drop_provisional, log_decades, min_finite_frac, sim_thr, verbose):
        import time
        t0 = time.time()
        with h5py.File(self.paths[0], "r") as h:
            a = h.attrs
            meta = {k: (_decode(a[k]) if a[k].dtype.kind in "OSU" else
                        np.asarray(a[k]).tolist()) for k in a.keys()
                    if k.endswith(("_names", "_parity", "_family", "_units",
                                   "_provisional", "_rc_shift", "_circular"))}
        # Side shards (families added after the corpus; `generate_family_shards.py`): their
        # channel metadata is appended to the main shards' in a fixed directory order.
        # Each shard is checked against the layout of its own kind -- main shards against
        # the first main shard, side shards against the first side shard of their
        # directory -- never against the merged list.
        main_layout = list(meta["per_bp_names"]), list(meta["global_names"])
        side_layout = {}
        for d in self.extra_dirs:
            with h5py.File(os.path.join(d, os.path.basename(self.paths[0])), "r") as h:
                side_layout[d] = (_decode(h.attrs["per_bp_names"]),
                                  _decode(h.attrs["global_names"]))
        for d in self.extra_dirs:
            with h5py.File(os.path.join(d, os.path.basename(self.paths[0])), "r") as h:
                a = h.attrs
                for k in list(meta):
                    if k in a:
                        v = (_decode(a[k]) if a[k].dtype.kind in "OSU" else
                             np.asarray(a[k]).tolist())
                        meta[k] = list(meta[k]) + list(v)
                    else:
                        raise ValueError(f"side shard in {d} lacks attribute {k}")
        for need in ("per_bp_rc_shift", "per_bp_circular", "global_circular"):
            if need not in meta:
                raise ValueError(
                    f"shards lack '{need}'; run src/data/patch_shard_metadata.py first, "
                    "so the trainer does not reflect step-indexed channels as if they "
                    "were base-pair-indexed")

        def keep_mask(key):
            prov = np.asarray(meta[f"{key}_provisional"], dtype=bool)
            return ~prov if drop_provisional else np.ones_like(prov)

        kp, kg = np.flatnonzero(keep_mask("per_bp")), np.flatnonzero(keep_mask("global"))
        P_parts, G_parts, seqs, strata = [], [], [], []
        for i, p in enumerate(self.paths):
            with h5py.File(p, "r") as h:
                if (_decode(h.attrs["per_bp_names"]), _decode(h.attrs["global_names"])) \
                        != main_layout:
                    raise ValueError(f"{p}: channel layout differs from {self.paths[0]}")
                pb_all = np.asarray(h["per_bp"][:], dtype=np.float32)
                gl_all = np.asarray(h["global"][:], dtype=np.float32)
                shard_seqs = _decode(h["sequence"][:])
                for d in self.extra_dirs:
                    with h5py.File(os.path.join(d, os.path.basename(p)), "r") as e:
                        if (_decode(e.attrs["per_bp_names"]),
                                _decode(e.attrs["global_names"])) != side_layout[d]:
                            raise ValueError(f"side shard {d}/{os.path.basename(p)}: channel "
                                             f"layout differs from that directory's first shard")
                        if _decode(e["sequence"][:]) != shard_seqs:
                            raise ValueError(f"side shard {d}/{os.path.basename(p)} is not "
                                             f"aligned with {p}")
                        pb_all = np.concatenate([pb_all, np.asarray(e["per_bp"][:], np.float32)], 1)
                        gl_all = np.concatenate([gl_all, np.asarray(e["global"][:], np.float32)], 1)
                P_parts.append(pb_all[:, kp, :])
                G_parts.append(gl_all[:, kg])
                seqs.extend(shard_seqs)
                strata.extend(_decode(h["stratum"][:]))
            if verbose and (i + 1) % 64 == 0:
                print(f"  read {i+1}/{len(self.paths)} shards  "
                      f"{time.time()-t0:.0f}s", flush=True)
        P = np.concatenate(P_parts, axis=0); del P_parts
        G = np.concatenate(G_parts, axis=0); del G_parts
        if self.interior_trim_g:
            self._interior_family_g(P, G, [meta["per_bp_names"][j] for j in kp],
                                    [meta["global_names"][j] for j in kg],
                                    self.interior_trim_g, verbose)

        def sel(key, idx):
            return {f: [meta[f"{key}_{f}"][j] for j in idx]
                    for f in ("names", "parity", "family", "units", "rc_shift",
                              "circular")}
        pb, gl = sel("per_bp", kp), sel("global", kg)

        P, pb, pb_log, pb_drop = self._transform(P, pb, log_decades, min_finite_frac,
                                                 positional=True)
        G, gl, gl_log, gl_drop = self._transform(G, gl, log_decades, min_finite_frac,
                                                 positional=False)
        self.per_bp_np, self.global_np = P, G
        self.pb, self.gl = pb, gl
        self.transforms = {"per_bp_log10": pb_log, "global_log10": gl_log,
                           "per_bp_dropped": pb_drop, "global_dropped": gl_drop,
                           "family_g_interior_trim": self.interior_trim_g}
        self.sequences, self.strata = seqs, strata

        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
        import splits as sp
        if verbose:
            print(f"  grouping {len(seqs)} sequences by similarity ...", flush=True)
        labels, info = sp.group_by_similarity(seqs, threshold=sim_thr, seed=0)
        self.groups = np.asarray(labels, dtype=np.int64)
        self.group_info = {k: v for k, v in info.items()
                           if isinstance(v, (int, float, str))}
        if verbose:
            print(f"  built in {time.time()-t0:.0f}s: {P.shape[0]} fragments, "
                  f"{P.shape[1]} per-bp and {G.shape[1]} global channels", flush=True)

    # ----------------------------------------------------------------------------------
    G_TRACKS = ("sasa_total", "sasa_sugar", "sasa_phosphate", "sasa_base", "radical_target",
                "major_edge", "minor_edge")

    @staticmethod
    def _interior_family_g(P, G, pb_names, gl_names, trim, verbose):
        """Recompute Family G's whole-sequence strand statistics over interior positions.

        The minimised duplex's terminal residues are not treated alike on the two strands:
        C(x) and W(rc x) -- the same molecule -- disagree by up to 6.6 channel spreads at
        the two terminal positions of each end and agree to 0.02-0.19 inside
        (`src/eval/rc_consistency.py`, DECISIONS.md). A whole-sequence maximum picks the
        most exposed residue, which is terminal, so the stored max-cleavage and
        heterogeneity channels mostly measure that artifact. Same definitions as
        `solvation.solvation_channels` (nanmean, nanstd, nanmax of the per-bp track), over
        positions [trim, n - trim) instead of all, in raw units before any transform.
        """
        import warnings
        pi = {n: i for i, n in enumerate(pb_names)}
        gi = {n: i for i, n in enumerate(gl_names)}
        L = P.shape[-1]
        changed = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for tag in ("W", "C"):
                for key in InMemoryLandscape.G_TRACKS:
                    src, dst = f"G_{key}_{tag}", f"G_mean_{key}_{tag}"
                    if src in pi and dst in gi:
                        G[:, gi[dst]] = np.nanmean(P[:, pi[src], trim:L - trim], axis=1)
                        changed.append(dst)
                src = f"G_radical_target_{tag}"
                if src not in pi:
                    continue
                inner = P[:, pi[src], trim:L - trim]
                for dst, fn in ((f"G_cleavage_heterogeneity_{tag}", np.nanstd),
                                (f"G_max_cleavage_{tag}", np.nanmax)):
                    if dst in gi:
                        G[:, gi[dst]] = fn(inner, axis=1)
                        changed.append(dst)
        if verbose:
            print(f"  Family G whole-sequence statistics over positions [{trim}, {L - trim}): "
                  f"{len(changed)} channels recomputed", flush=True)
        return changed

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _transform(X, info, log_decades, min_finite_frac, positional):
        """Drop dead channels, expand circular ones, log heavy-tailed ones, standardise."""
        axes = (0, 2) if positional else (0,)
        cols, new = [], {k: [] for k in info}
        new["mean"], new["std"], new["transform"] = [], [], []
        logged, dropped = [], []
        for j, name in enumerate(info["names"]):
            x = X[:, j]
            finite = np.isfinite(x)
            frac = float(finite.mean())
            if frac < min_finite_frac:
                dropped.append((name, "almost entirely NaN"))
                continue
            vals = x[finite]
            if float(np.std(vals)) < 1e-9 * max(1.0, float(np.abs(vals).mean())):
                dropped.append((name, "constant across the corpus"))
                continue
            parts = []
            if info["circular"][j]:
                rad = np.deg2rad(x)
                parts = [("__sin", np.sin(rad), "sin"), ("__cos", np.cos(rad), "cos")]
            else:
                lo, hi = np.percentile(vals, [0.5, 99.5])
                if float(vals.min()) > 0 and lo > 0 and hi / lo > 10 ** log_decades:
                    parts = [("", np.log10(np.where(finite, x, np.nan)), "log10")]
                    logged.append(name)
                else:
                    parts = [("", x, "identity")]
            for suffix, arr, how in parts:
                arr = arr.astype(np.float32)
                m = float(np.nanmean(arr)); sd = float(np.nanstd(arr))
                sd = sd if sd > 1e-12 else 1.0
                cols.append(((arr - m) / sd).astype(np.float16))
                new["names"].append(name + suffix)
                for k in ("parity", "family", "units", "rc_shift", "circular"):
                    new[k].append(info[k][j])
                # An antisymmetric angle maps theta -> -theta, under which sin changes
                # sign and cos does not; inheriting ANTISYMMETRIC for both would force
                # the model to predict cos = 0.
                if how == "cos" and info["parity"][j] == "antisymmetric":
                    new["parity"][-1] = "symmetric"
                new["mean"].append(m); new["std"].append(sd); new["transform"].append(how)
        out = np.stack(cols, axis=1)
        return out, new, logged, dropped

    # ----------------------------------------------------------------------------------
    def _save_cache(self, verbose):
        import json
        os.makedirs(os.path.dirname(os.path.abspath(self.cache)), exist_ok=True)
        np.savez(self.cache, per_bp=self.per_bp_np, global_=self.global_np,
                 groups=self.groups,
                 sequences=np.asarray(self.sequences, dtype=object),
                 strata=np.asarray(self.strata, dtype=object),
                 meta=json.dumps({"pb": self.pb, "gl": self.gl,
                                  "transforms": self.transforms,
                                  "group_info": self.group_info}))
        if verbose:
            print(f"  cached to {self.cache}", flush=True)

    def _load_cache(self, verbose):
        import json, time
        t0 = time.time()
        z = np.load(self.cache, allow_pickle=True)
        self.per_bp_np, self.global_np = z["per_bp"], z["global_"]
        self.groups = z["groups"]
        self.sequences = [str(s) for s in z["sequences"]]
        self.strata = [str(s) for s in z["strata"]]
        m = json.loads(str(z["meta"]))
        self.pb, self.gl = m["pb"], m["gl"]
        self.transforms, self.group_info = m["transforms"], m["group_info"]
        if verbose:
            print(f"  loaded cache {os.path.basename(self.cache)} in "
                  f"{time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.per_bp_np.shape[0])

    def one_hot(self) -> np.ndarray:
        L = len(self.sequences[0])
        x = np.zeros((len(self.sequences), 4, L), dtype=np.uint8)
        lut = np.full(256, 255, dtype=np.uint8)
        for k, b in enumerate("ACGT"):
            lut[ord(b)] = k
            lut[ord(b.lower())] = k
        for i, s in enumerate(self.sequences):
            codes = lut[np.frombuffer(s.encode(), dtype=np.uint8)]
            ok = codes < 4
            x[i, codes[ok], np.flatnonzero(ok)] = 1
        return x


class AuxLandscape:
    """Auxiliary fragments carrying a subset of the channels, on the main dataset's scale.

    Built for the Family L data-scaling experiment: windows with only Family L targets
    (`src/data/generate_aux_targets.py`). Each auxiliary channel is mapped to the main
    dataset's output column of the same name and passed through that column's own
    transform -- log10 where the main dataset logs it, then the main dataset's mean and
    standard deviation -- so auxiliary targets are on exactly the scale the model is
    trained and scored on. The trainer places them into the full layout with every other
    column NaN, which the loss masks.
    """

    def __init__(self, pattern: str, main: "InMemoryLandscape", verbose: bool = True,
                 extra_dirs: Optional[Sequence[str]] = None, optional_extra: bool = False,
                 holdout: Optional[Dict[str, float]] = None):
        # `holdout` {frag_id prefix: fraction}: windows of that prefix whose sha1(frag_id) falls
        # below the fraction are kept out of training (listed in self.holdout_ids) so a window
        # set can also be scored on windows the model never saw.
        # `pattern`: a directory (its shard_*.h5), a glob, or several of either separated by
        # commas (auxiliary sets with different channel content, e.g. the genome-wide windows
        # and the quadruplex windows). Passing directories rather than globs matters: the
        # Slurm scripts expand their extra arguments unquoted, and a glob there was expanded
        # by the shell into 256 separate arguments.
        paths = []
        for pat in [x for x in str(pattern).split(",") if x]:
            if os.path.isdir(pat):
                pat = os.path.join(pat, "*.h5")
            got = sorted(glob.glob(pat))
            if not got:
                raise FileNotFoundError(f"no auxiliary shards match {pat}")
            paths += got
        self.paths = paths
        # Per shard: its own channels plus those of every side shard (same windows, same
        # order, verified by sequence). With `optional_extra`, a side directory may lack a
        # shard (a family labelled on one auxiliary set only) and those channels are NaN for
        # its windows. Channel layouts are united across shards in order of first appearance.
        blocks, seqs, fids = [], [], []
        layout_of = {}
        for p in paths:
            with h5py.File(p, "r") as h:
                names_pb = _decode(h.attrs["per_bp_names"])
                names_gl = _decode(h.attrs["global_names"])
                Pp = h["per_bp"][:].astype(np.float32); Gp = h["global"][:].astype(np.float32)
                sq = [s.decode() if isinstance(s, bytes) else s for s in h["sequence"][:]]
                fi = ([s.decode() if isinstance(s, bytes) else s for s in h["frag_id"][:]]
                      if "frag_id" in h else [f"{os.path.basename(p)}:{i}" for i in range(len(sq))])
            key = (os.path.dirname(p), "base")
            if layout_of.setdefault(key, (names_pb, names_gl)) != (names_pb, names_gl):
                raise ValueError(f"{p}: channel layout differs from the first shard")
            for d in (extra_dirs or ()):
                q = os.path.join(d, os.path.basename(p))
                if not os.path.exists(q):
                    if optional_extra:
                        continue
                    raise FileNotFoundError(f"no side shard for {p} in {d}")
                with h5py.File(q, "r") as h2:
                    s2 = [s.decode() if isinstance(s, bytes) else s for s in h2["sequence"][:]]
                    if s2 != sq:
                        raise ValueError(f"{q}: side shard not aligned with {p}")
                    spb, sgl = _decode(h2.attrs["per_bp_names"]), _decode(h2.attrs["global_names"])
                    if layout_of.setdefault((d, "side"), (spb, sgl)) != (spb, sgl):
                        raise ValueError(f"{q}: side shard layout differs within {d}")
                    names_pb = names_pb + spb
                    names_gl = names_gl + sgl
                    Pp = np.concatenate([Pp, h2["per_bp"][:].astype(np.float32)], axis=1)
                    Gp = np.concatenate([Gp, h2["global"][:].astype(np.float32)], axis=1)
            blocks.append((names_pb, names_gl, Pp, Gp)); seqs += sq; fids += fi
        pbn, gln = [], []
        for npb, ngl, _, _ in blocks:
            pbn += [x for x in npb if x not in pbn]
            gln += [x for x in ngl if x not in gln]
        n_tot, Lbp = len(seqs), blocks[0][2].shape[2]
        P = np.full((n_tot, len(pbn), Lbp), np.nan, dtype=np.float32)
        G = np.full((n_tot, len(gln)), np.nan, dtype=np.float32)
        off = 0
        for npb, ngl, Pp, Gp in blocks:
            k = Pp.shape[0]
            P[off:off + k, [pbn.index(x) for x in npb]] = Pp
            G[off:off + k, [gln.index(x) for x in ngl]] = Gp
            off += k
        del blocks
        self.holdout_ids: List[str] = []
        if holdout:
            import hashlib
            keep = np.ones(len(seqs), dtype=bool)
            for i, f in enumerate(fids):
                for pre, frac in holdout.items():
                    if f.startswith(pre) and int(hashlib.sha1(f.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac:
                        keep[i] = False
            self.holdout_ids = [f for f, k in zip(fids, keep) if not k]
            P, G = P[keep], G[keep]
            seqs = [x for x, k in zip(seqs, keep) if k]
            fids = [x for x, k in zip(fids, keep) if k]
        self.frag_ids = fids
        self.sequences = seqs
        self.pb_cols, Ypb = self._map(P, pbn, main.pb, positional=True)
        self.gl_cols, Ygl = self._map(G, gln, main.gl, positional=False)
        self.per_bp_np, self.global_np = Ypb, Ygl
        if verbose:
            print(f"  auxiliary: {len(seqs)} fragments; {len(self.pb_cols)} per-bp and "
                  f"{len(self.gl_cols)} global columns mapped onto the main layout", flush=True)

    @staticmethod
    def _map(X, names, info, positional):
        cols, out = [], []
        for j, name in enumerate(info["names"]):
            base = name.split("__")[0]
            if base not in names:
                continue
            how = info["transform"][j]
            x = X[:, names.index(base)].astype(np.float64)
            if how in ("sin", "cos"):
                # Circular channels are stored in degrees and expanded by the main dataset
                # into sine and cosine columns (`_transform`); the auxiliary values get the
                # same expansion before the column's own standardisation.
                x = (np.sin if how == "sin" else np.cos)(np.deg2rad(x))
            elif how == "log10":
                with np.errstate(invalid="ignore", divide="ignore"):
                    x = np.log10(np.where(x > 0, x, np.nan))
            out.append(((x - info["mean"][j]) / info["std"][j]).astype(np.float16))
            cols.append(j)
        Y = np.stack(out, axis=1) if out else np.zeros((len(X), 0) + X.shape[2:], np.float16)
        return np.asarray(cols, dtype=np.int64), Y

    def one_hot(self) -> np.ndarray:
        # Same encoding and dtype as InMemoryLandscape.one_hot, since the trainer
        # concatenates the two.
        return InMemoryLandscape.one_hot(self)


def load_bpp(bpp_dir: str, shard_paths: Sequence[str], sequences: Sequence[str]) -> np.ndarray:
    """Base-pair probability matrices aligned to a dataset's fragment order.

    `src/data/generate_bpp.py` writes one `<shard>.bpp.npz` per target shard. They are
    concatenated in the dataset's own shard order, and each shard's block is checked
    against the sha1 of the dataset's sequences for that block, so a matrix can never be
    attached to the wrong fragment. Returns uint8 (n_fragments, 2, n*(n-1)/2):
    channel 0 the plus-strand transcript (forward frame), channel 1 the minus-strand
    transcript (its own frame), upper triangles quantised as round(255*sqrt(p)).
    """
    import hashlib
    parts, off = [], 0
    for path in shard_paths:
        f = os.path.join(bpp_dir, os.path.basename(path).replace(".h5", ".bpp.npz"))
        if not os.path.exists(f):
            raise FileNotFoundError(f"no base-pair probabilities for {path} ({f})")
        z = np.load(f)
        q = z["bpp"]
        block = sequences[off:off + q.shape[0]]
        if hashlib.sha1("\n".join(block).encode()).hexdigest() != str(z["seq_sha1"]):
            raise ValueError(f"{f} does not match the dataset's fragments at {off}")
        parts.append(q); off += q.shape[0]
    if off != len(sequences):
        raise ValueError(f"base-pair probabilities cover {off} of {len(sequences)} fragments")
    return np.concatenate(parts)


def landscape_from_args(args: dict, verbose: bool = False) -> "InMemoryLandscape":
    """The training dataset exactly as a run built it, from its stored arguments."""
    return InMemoryLandscape(args["shards"], cache_dir=args.get("cache_dir"),
                             drop_provisional=not args.get("allow_provisional", False),
                             max_shards=args.get("max_shards"), verbose=verbose,
                             interior_trim_g=args.get("g_interior_trim") or 0,
                             extra_dirs=args.get("extra_dirs") or None)


def channel_meta_from_args(args: dict) -> Tuple[dict, dict]:
    """A run's channel metadata (names, parity, family, rc_shift, transforms) read from
    its dataset cache's small `meta` entry only -- what an encoder needs to be rebuilt
    for inference, without loading the targets."""
    import json
    _, path = InMemoryLandscape.cache_file(
        args["shards"], args.get("cache_dir"),
        drop_provisional=not args.get("allow_provisional", False),
        max_shards=args.get("max_shards"), interior_trim_g=args.get("g_interior_trim") or 0,
        extra_dirs=args.get("extra_dirs") or None)
    m = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
    return m["pb"], m["gl"]


class LongWindowSet:
    """Channels that need more context than the 61 bp fragment carries, on long windows.

    Low-salt electrostatics (Family A) depends on sequence beyond the fragment, and the
    encoder must see whatever sequence a label depends on, so those labels live on windows
    of `window_length` bp centred on each corpus fragment (`src/data/extend_windows.py`).
    Shards are aligned row for row with the main shards -- so every window inherits its
    fragment's train / validation / test assignment and the similarity split holds -- and
    carry per-bp values for the central `n_central` positions only (where the solver
    scores), plus whole-window values. Each window's centre is checked against its main
    fragment. Channels are standardised on their own finite values (identity transform)
    and appended to the model's channel plan after the main channels.
    """

    def __init__(self, pattern: str, main: "InMemoryLandscape", verbose: bool = True):
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "shard_*.h5")
        paths = sorted(glob.glob(pattern))
        if [os.path.basename(p) for p in paths] != [os.path.basename(p) for p in main.paths]:
            raise ValueError("long-window shards must be aligned one to one with the main shards")
        W, P, G, meta = [], [], [], None
        row = 0
        for p in paths:
            with h5py.File(p, "r") as h:
                a = h.attrs
                m = {k: (_decode(a[k]) if a[k].dtype.kind in "OSU" else np.asarray(a[k]).tolist())
                     for k in a.keys() if k.endswith(("_names", "_parity", "_family", "_units",
                                                      "_provisional", "_rc_shift", "_circular"))}
                m["center_offset"] = int(a["center_offset"]); m["window_length"] = int(a["window_length"])
                if meta is None:
                    meta = m
                elif any(m[k] != meta[k] for k in ("per_bp_names", "global_names", "center_offset",
                                                   "window_length")):
                    raise ValueError(f"{p}: layout differs from the first long-window shard")
                frag = _decode(h["sequence"][:]); win = _decode(h["window"][:])
                n = len(frag)
                if frag != main.sequences[row:row + n]:
                    raise ValueError(f"{p}: fragments not aligned with the main dataset")
                L = len(frag[0]); off = (meta["window_length"] - L) // 2
                if any(w[off:off + L] != f for w, f in zip(win, frag)):
                    raise ValueError(f"{p}: a window's centre differs from its fragment")
                W += win; P.append(np.asarray(h["per_bp_central"][:], np.float32))
                G.append(np.asarray(h["global"][:], np.float32)); row += n
        if row != len(main.sequences):
            raise ValueError("long-window shards cover a different number of fragments")
        self.windows = W
        self.window_length = meta["window_length"]
        self.center = meta["center_offset"]
        P = np.concatenate(P); G = np.concatenate(G)
        self.n_central = P.shape[-1]
        self.pb, self.gl = {}, {}
        for key, X, dst in (("per_bp", P, self.pb), ("global", G, self.gl)):
            for f in ("names", "parity", "family", "units", "rc_shift", "circular"):
                dst[f] = list(meta.get(f"{key}_{f}", [0] * len(meta[f"{key}_names"])))
            dst["mean"], dst["std"], dst["transform"], dst["n_finite"] = [], [], [], []
            for j in range(X.shape[1]):
                v = X[:, j]
                fin = np.isfinite(v)
                mu = float(v[fin].mean()) if fin.any() else 0.0
                sd = float(v[fin].std()) if fin.sum() > 1 else 1.0
                sd = sd if sd > 1e-12 else 1.0
                X[:, j] = (v - mu) / sd
                dst["mean"].append(mu); dst["std"].append(sd); dst["transform"].append("identity")
                dst["n_finite"].append(int(fin.sum()))
        self.per_bp_np = P.astype(np.float16)
        self.global_np = G.astype(np.float16)
        self.has_label = (np.isfinite(self.per_bp_np).reshape(len(W), -1).any(1)
                          | np.isfinite(self.global_np).any(1))
        if verbose:
            print(f"  long windows: {len(W)} of {self.window_length} bp ({int(self.has_label.sum())} "
                  f"labelled); {P.shape[1]} per-bp (central {self.n_central}) and {G.shape[1]} "
                  f"whole-window channels appended to the plan", flush=True)

    def one_hot(self) -> np.ndarray:
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        out = np.zeros((len(self.windows), 4, self.window_length), dtype=np.float16)
        for i, s in enumerate(self.windows):
            for j, c in enumerate(s):
                k = idx.get(c)
                if k is not None:
                    out[i, k, j] = 1.0
        return out


def extend_channel_info(main_info: Dict[str, list], extra: Dict[str, list]) -> Dict[str, list]:
    """Main channel metadata with a long-window block appended (names must not collide)."""
    clash = set(main_info["names"]) & set(extra["names"])
    if clash:
        raise ValueError(f"long-window channels collide with main channels: {sorted(clash)[:5]}")
    return {k: list(main_info[k]) + list(extra.get(k, [None] * len(extra["names"])))
            for k in main_info}


# ======================================================================================
# Family W: 10,240 bp fragments for the window ladder and the wide branch
# ======================================================================================

class WideSet:
    """Family W targets (`src/physics/longrange.py`, `src/data/pack_family_w.py`) on 10,240 bp
    genomic fragments: per-position tracks (window-ladder and wide-branch quantities) and whole-
    fragment scalars. Both are appended to the model's channel plan after the main (and long-window)
    channels; on a Family W fragment every other column is NaN and masked, and vice versa.

    Channel metadata (parity, units, circular, provisional) comes from the registry, the reverse-
    complement index shifts from the shards (they must agree). Targets get the main dataset's
    transforms (`InMemoryLandscape._transform`: dead channels dropped, circular ones expanded to sin
    and cos, heavy-tailed ones logged, all standardised), decided on Family W's own values. Tracks
    stay in host memory (float16, ~1.2 GB per 1,000 fragments) and go to the device per batch.

    Leakage: `contains` lists, per fragment, the library fragments it holds a contiguous near-copy
    of (`sample_long_fragments.py`); `join_groups` puts each long fragment in one similarity group
    with all of them before the split, so whole groups still go to one side.
    """

    HUB_TILES = 3

    def __init__(self, pattern: str, verbose: bool = True, drop_provisional: bool = True,
                 log_decades: float = 3.0, min_finite_frac: float = 0.0):
        import sys as _sys
        self.force_train_groups = set()
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "famW_*.h5")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no Family W shards match {pattern}")
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "physics"))
        import longrange  # noqa: F401  (registers Family W)
        from channels import REGISTRY
        spec = {s.name: s for s in REGISTRY.by_family("W")}
        seqs, T, S, contains, sources, fids = [], [], [], [], [], []
        tn = sn = rcs = None
        for p in paths:
            with h5py.File(p, "r") as h:
                a_tn, a_sn = _decode(h.attrs["track_names"]), _decode(h.attrs["scalar_names"])
                a_rc = [int(x) for x in h.attrs["rc_shift"]]
                if tn is None:
                    tn, sn, rcs = a_tn, a_sn, a_rc
                elif (a_tn, a_sn, a_rc) != (tn, sn, rcs):
                    raise ValueError(f"{p}: channel layout differs from {paths[0]}")
                seqs += [x.decode() if isinstance(x, bytes) else str(x) for x in h["sequence"][:]]
                T.append(np.asarray(h["tracks"][:], np.float16))      # float16 in host memory (~1.2 GB per 1,000)
                S.append(np.asarray(h["scalars"][:], np.float32))
                contains += [[c for c in str(x).split(";") if c] for x in _decode(h["contains"][:])]
                sources += _decode(h["source"][:]); fids += _decode(h["frag_id"][:])
        for n_, r_ in zip(tn, rcs):
            if n_ not in spec:
                raise KeyError(f"Family W track {n_} is not registered")
            if spec[n_].rc_shift != r_:
                raise ValueError(f"{n_}: shard rc_shift {r_} != registry {spec[n_].rc_shift}")
        missing = [n_ for n_ in sn if n_ not in spec]
        if missing:
            raise KeyError(f"Family W scalars not registered: {missing[:5]}")
        Tm, Sm = np.concatenate(T), np.concatenate(S)
        self.sequences, self.contains, self.sources, self.frag_ids = seqs, contains, sources, fids
        self.length = Tm.shape[-1]

        def info(names, rc):
            return {"names": list(names), "parity": [spec[n_].parity.value for n_ in names],
                    "family": ["W"] * len(names), "units": [spec[n_].units for n_ in names],
                    "rc_shift": list(rc), "circular": [bool(spec[n_].circular) for n_ in names]}
        keep_t = [j for j, n_ in enumerate(tn) if not (drop_provisional and spec[n_].provisional)]
        keep_s = [j for j, n_ in enumerate(sn) if not (drop_provisional and spec[n_].provisional)]
        self.dropped_provisional = [tn[j] for j in range(len(tn)) if j not in keep_t] + \
                                   [sn[j] for j in range(len(sn)) if j not in keep_s]
        it, is_ = info([tn[j] for j in keep_t], [rcs[j] for j in keep_t]), info([sn[j] for j in keep_s], [0] * len(keep_s))
        Xt, self.pb, lt, dt = InMemoryLandscape._transform(Tm[:, keep_t], it, log_decades, min_finite_frac, True)
        Xs, self.gl, ls_, ds_ = InMemoryLandscape._transform(Sm[:, keep_s], is_, log_decades, min_finite_frac, False)
        self.tracks_np = Xt.astype(np.float16)                 # (n, C, L)
        self.scalars_np = Xs.astype(np.float16)                # (n, S)
        # window-ladder tracks: which columns a pooled head at which width should produce
        self.window_of = {}
        for j, n_ in enumerate(self.pb["names"]):
            base = n_.split("__")[0]
            sp = spec.get(base)
            if sp is not None and sp.window_bp is not None:
                self.window_of[j] = int(sp.window_bp)
        self.transforms = {"tracks_logged": lt, "tracks_dropped": dt, "scalars_logged": ls_, "scalars_dropped": ds_}
        if verbose:
            print(f"  Family W: {len(seqs)} fragments of {self.length} bp from {len(set(sources))} genomes; "
                  f"{Xt.shape[1]} track and {Xs.shape[1]} scalar columns "
                  f"({len(self.window_of)} from pooled window heads); provisional dropped: {len(self.dropped_provisional)}", flush=True)

    def join_groups(self, main: "InMemoryLandscape") -> Tuple[np.ndarray, np.ndarray, dict]:
        """(main labels, W labels): the main similarity groups with every long fragment joined to the
        groups of the library fragments it contains (union-find); W fragments that contain none get
        their own new group."""
        fid_main = []
        for p in main.paths:
            with h5py.File(p, "r") as h:
                fid_main += _decode(h["frag_id"][:])
        if len(fid_main) != len(main.sequences):
            raise ValueError("main shards' frag_id count differs from the dataset")
        pos = {f: i for i, f in enumerate(fid_main)}
        lab = np.asarray(main.groups, dtype=np.int64)
        # Repeat hubs: a library fragment held by >= HUB_TILES tiles is a near-copy of a repeated
        # element (or a low-complexity tract) and would chain most tiles into one group (measured: one
        # component of 12,725 of 21,391 tiles; 41 tiles at most without these links). Hubs do not
        # link tiles; their similarity groups are kept on the training side instead.
        n_tiles = {}
        for cs in self.contains:
            for c in cs:
                n_tiles[c] = n_tiles.get(c, 0) + 1
        hubs = {c for c, n_ in n_tiles.items() if n_ >= self.HUB_TILES and c in pos}
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        base = int(lab.max()) + 1 if len(lab) else 0
        wl = np.empty(len(self.sequences), dtype=np.int64)
        n_linked = 0
        for k, cs in enumerate(self.contains):
            node = base + k
            find(node)
            idx = [pos[c] for c in cs if c in pos and c not in hubs]
            n_linked += bool(idx)
            for i in idx:
                ra, rb = find(node), find(int(lab[i]))
                if ra != rb:
                    parent[ra] = rb
        new_main = np.array([find(int(g)) for g in lab], dtype=np.int64)
        for k in range(len(self.sequences)):
            wl[k] = find(base + k)
        comp = np.unique(np.concatenate([new_main, wl]), return_counts=True)[1]
        force_train = {int(new_main[pos[c]]) for c in hubs}
        rep = {"w_fragments": int(len(wl)), "w_linked_to_library": int(n_linked),
               "largest_joined_group": int(comp.max()) if len(comp) else 0,
               "repeat_hub_fragments": int(len(hubs)), "groups_kept_in_training": int(len(force_train)),
               "main_fragments_kept_in_training": int(np.isin(new_main, list(force_train)).sum()),
               "main_groups_before": int(len(np.unique(lab))), "main_groups_after": int(len(np.unique(new_main)))}
        self.force_train_groups = force_train
        return new_main, wl, rep

    def one_hot(self, idx: np.ndarray) -> np.ndarray:
        L = self.length
        out = np.zeros((len(idx), 4, L), dtype=np.float32)
        lut = np.full(256, -1, dtype=np.int64)
        for j, b in enumerate("ACGT"):
            lut[ord(b)] = j
        for r, i in enumerate(idx):
            a = lut[np.frombuffer(self.sequences[int(i)].encode(), dtype=np.uint8)]
            ok = a >= 0
            out[r, a[ok], np.flatnonzero(ok)] = 1.0
        return out


class ALongSet:
    """Family A's long-fragment set (`generate_family_a_longset_shards.py`): 205 bp genomic segments,
    each centred on a corpus fragment, with Family A channels (and the window-level A_lw channels) on
    their central 105 bp from tiled production solves. Every column must already be in the plan (the
    Family A long-window channels appended by `LongWindowSet`) and is standardised with that column's
    own mean and standard deviation, so these labels are on exactly the scale the model is trained and
    scored on; A_lw columns, NaN on the 121 bp windows, are standardised on their own values here.
    A segment inherits its fragment's train / validation / test side (no leakage by construction)."""

    def __init__(self, pattern: str, long_info: Dict[str, list], verbose: bool = True):
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "longset_*.h5")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no Family A long-set shards match {pattern}")
        seqs, P, fids, names, start = [], [], [], None, None
        for p in paths:
            with h5py.File(p, "r") as h:
                n_ = _decode(h.attrs["per_bp_names"])
                if names is None:
                    names, start = n_, int(h.attrs["stretch_start"])
                elif n_ != names:
                    raise ValueError(f"{p}: channel layout differs")
                seqs += _decode(h["sequence"][:]); fids += _decode(h["frag_id"][:])
                P.append(np.asarray(h["per_bp"][:], np.float32))
        P = np.concatenate(P)
        col = {n_: j for j, n_ in enumerate(long_info["names"])}
        keep = [j for j, n_ in enumerate(names) if n_ in col]
        self.dropped = [n_ for n_ in names if n_ not in col]
        self.cols = [col[names[j]] for j in keep]           # positions inside the long-window block
        X = P[:, keep]
        for k, c in enumerate(self.cols):
            mu, sd = long_info["mean"][c], long_info["std"][c]
            v = X[:, k]
            fin = np.isfinite(v)
            if not np.isfinite(mu) or long_info["n_finite"][c] == 0:      # e.g. A_lw: NaN on every 121 bp window
                mu = float(v[fin].mean()) if fin.any() else 0.0
                sd = float(v[fin].std()) if fin.sum() > 1 else 1.0
            X[:, k] = (v - mu) / (sd if sd > 1e-12 else 1.0)
        self.per_bp_np = X.astype(np.float16)
        self.sequences, self.frag_ids, self.start = seqs, fids, start
        self.length, self.stretch = len(seqs[0]), X.shape[-1]
        if verbose:
            print(f"  Family A long set: {len(seqs)} segments of {self.length} bp, {len(self.cols)} columns on the "
                  f"central {self.stretch} bp; not in the plan: {len(self.dropped)}", flush=True)

    def sides(self, main_frag_ids: Sequence[str], tr, va, te) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        side = {}
        for arr, s_ in ((tr, 0), (va, 1), (te, 2)):
            for i in np.asarray(arr):
                side[main_frag_ids[int(i)]] = s_
        lab = np.array([side.get(f, -1) for f in self.frag_ids])
        if (lab < 0).any():
            raise ValueError(f"{int((lab < 0).sum())} long-set segments have no main fragment")
        return tuple(np.flatnonzero(lab == s_) for s_ in (0, 1, 2))

    def one_hot(self, idx) -> np.ndarray:
        out = np.zeros((len(idx), 4, self.length), dtype=np.float32)
        lut = np.full(256, -1, dtype=np.int64)
        for j, b in enumerate("ACGT"):
            lut[ord(b)] = j
        for r, i in enumerate(idx):
            a = lut[np.frombuffer(self.sequences[int(i)].encode(), dtype=np.uint8)]
            ok = a >= 0
            out[r, a[ok], np.flatnonzero(ok)] = 1.0
        return out


def main_frag_ids(main: "InMemoryLandscape") -> List[str]:
    """frag_id of every main-dataset fragment, in the dataset's order (read from its shards)."""
    out: List[str] = []
    for p in main.paths:
        with h5py.File(p, "r") as h:
            out += _decode(h["frag_id"][:])
    if len(out) != len(main.sequences):
        raise ValueError("main shards' frag_id count differs from the dataset")
    return out
