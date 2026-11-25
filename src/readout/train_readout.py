# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""READOUT heads (Aim 2): how much of an assay a physics bottleneck reaches.

Arms, all with the same head architecture and the same training budget (PLAN.md Aim 2):
  seq        one-hot sequence (4 channels)
  phys       the encoder's per-bp landscape (all channels, standardised)
  phys:k     the landscape through a learned linear projection to k channels (the
             bottleneck width sweep, k in 4 ... 256)
  seq+phys   both, concatenated
  decode     the decoder test: predict each base from the landscape alone; reports per-base
             accuracy and bits recovered out of the entropy of the bases (2 bits for
             uniform sequence); decode:k through a k-channel projection
The head: [optional 1x1 projection] -> conv stem (k=7) -> dilated residual conv blocks ->
mean and max pooling -> MLP. Same width, depth and optimiser for every arm; only the input
layer differs. AdamW, early stopping on the validation metric, fixed epoch budget.
Splits are leakage-controlled per dataset (`DATASETS`): a provided test split when the
dataset has one, otherwise similarity-grouped (MinHash, as for the encoder's corpus).
Writes results/readout/<dataset>__<arm>__<encoder>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "model")); sys.path.insert(0, os.path.join(SRC, "data"))

# Fixed, named physical channel sets (no learned projection): candidates for a physics
# bottleneck that keeps full resolution. Their decodability is measured, not assumed.
SELECTIONS = {
    "shape4": ["C_minor_width_A", "C_propeller", "C_roll", "C_twist"],
    "shape13": ["C_shift", "C_slide", "C_rise", "C_tilt", "C_roll", "C_twist", "C_shear",
                "C_stretch", "C_stagger", "C_buckle", "C_propeller", "C_opening",
                "C_minor_width_A"],
}


def selection_indices(name: str, per_bp_names):
    """Channel indices for a named selection, or `family:X` for every channel of family X."""
    if name.startswith("family:"):
        fam = name.split(":", 1)[1]
        idx = [i for i, n in enumerate(per_bp_names) if n.startswith(fam + "_")]
    else:
        want = SELECTIONS[name]
        missing = [w for w in want if w not in per_bp_names]
        if missing:
            raise SystemExit(f"selection {name}: channels not in this landscape: {missing}")
        idx = [per_bp_names.index(w) for w in want]
    if not idx:
        raise SystemExit(f"selection {name} is empty")
    return idx


DATASETS = {
    # name: task, target column, how to split, optional group column for reporting
    "loopseq": {"task": "regression", "target": "target", "split": "similarity", "by": "library"},
    "splice121": {"task": "classification", "n_classes": 3, "target": "target",
                  "split": "provided", "by": None},
    # two assay systems (tobacco leaf, maize protoplast) whose readouts correlate at only
    # R2 0.14-0.40 (Jores et al. 2021): one output per system on a shared trunk, targets
    # standardised per system, scored per system -- as the source paper modelled them.
    "jores": {"task": "regression", "target": "target", "split": "provided", "by": "subset",
              "condition": "subset"},
    # the plan's positive control; chromosomes XV and XVI are the test side
    "nucleosome_invitro": {"task": "regression", "target": "target", "split": "provided",
                           "by": "library"},
}


def _npz_shape(path: str, key: str):
    """Shape and dtype of one array inside an .npz, read from its header only."""
    import zipfile
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as fh:
        version = np.lib.format.read_magic(fh)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _, dtype = reader(fh)
    return shape, dtype


def load_features(prefix: str):
    """Concatenate a featurisation's chunks into one preallocated array (peak memory = the
    total plus one chunk, not twice the total: the plant set is 14 GB in float16)."""
    files = sorted(glob.glob(prefix + ".chunk*.npz"))
    if not files:
        raise FileNotFoundError(prefix)
    shapes = [_npz_shape(f, "per_bp") for f in files]
    n = sum(sh[0][0] for sh in shapes)
    pb = np.empty((n,) + tuple(shapes[0][0][1:]), dtype=shapes[0][1])
    seqs, cols, off = [], None, 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        part = z["per_bp"]; pb[off:off + len(part)] = part; off += len(part); del part
        seqs.extend(str(s) for s in z["sequences"])
        if cols is None:
            cols = {k: list(v) for k, v in m["columns"].items()}
        else:
            for k, v in m["columns"].items():
                cols[k].extend(v)
    return pb, seqs, cols, m


def one_hot_np(seqs):
    L = len(seqs[0]); x = np.zeros((len(seqs), 4, L), dtype=np.float32)
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            k = lut.get(c)
            if k is not None:
                x[i, k, j] = 1.0
    return x


def make_split(cfg, seqs, cols, seed=0):
    n = len(seqs); rng = np.random.default_rng(seed)
    if cfg["split"] == "provided":
        sp = np.asarray(cols["split"])
        test = np.flatnonzero(sp == "test")
        pool = np.flatnonzero(sp != "test")
        # validation from the training side, grouped so near-duplicates stay together
        import splits as S
        labels, _ = S.group_by_similarity([seqs[i] for i in pool], threshold=0.5, seed=seed)
        g = np.unique(labels); rng.shuffle(g)
        vg = set(g[: max(1, len(g) // 10)].tolist())
        val = pool[np.array([l in vg for l in labels])]
        train = pool[np.array([l not in vg for l in labels])]
        return train, val, test
    import splits as S
    labels, _ = S.group_by_similarity(seqs, threshold=0.5, seed=seed)
    labels = np.asarray(labels); g = np.unique(labels); rng.shuffle(g)
    nt, nv = len(g) // 10, len(g) // 10
    tg, vg = set(g[:nt].tolist()), set(g[nt:nt + nv].tolist())
    test = np.flatnonzero([l in tg for l in labels]); val = np.flatnonzero([l in vg for l in labels])
    train = np.setdiff1d(np.arange(n), np.concatenate([test, val]))
    return train, val, test


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average over positions (window w, centred, averaging only existing positions)."""
    if w <= 1:
        return x
    L = x.shape[-1]
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,), x.dtype), np.cumsum(x, axis=-1)], axis=-1)
    lo = np.clip(np.arange(L) - w // 2, 0, L); hi = np.clip(np.arange(L) - w // 2 + w, 0, L)
    return ((c[..., hi] - c[..., lo]) / (hi - lo)).astype(x.dtype)


def binned(x: np.ndarray, w: int) -> np.ndarray:
    """Average over non-overlapping w-bp bins, repeated back to full length so the head is
    unchanged: L/w distinct values per channel survive (the last bin may be shorter)."""
    if w <= 1:
        return x
    L = x.shape[-1]
    edges = list(range(0, L, w)) + [L]
    out = np.empty_like(x)
    for a, b in zip(edges[:-1], edges[1:]):
        out[..., a:b] = x[..., a:b].mean(axis=-1, keepdims=True)
    return out


class Batches:
    """Row access to one or more (n, channels, L) arrays, concatenated on channels per batch;
    `smooth_w` > 1 replaces the landscape by its moving average (an information-limited
    representation: base-level identity is averaged away, physics at that scale is kept)."""

    def __init__(self, *arrays, smooth_w: int = 1, smooth_from: int = 0, bin_w: int = 1):
        self.arrays, self.smooth_w, self.smooth_from = arrays, smooth_w, smooth_from
        self.bin_w = bin_w
        self.shape = (arrays[0].shape[0], sum(x.shape[1] for x in arrays), arrays[0].shape[2])

    def __getitem__(self, idx):
        parts = [x[idx].astype(np.float32) for x in self.arrays]
        if self.smooth_w > 1:
            parts = [p if i < self.smooth_from else smooth(p, self.smooth_w)
                     for i, p in enumerate(parts)]
        if self.bin_w > 1:
            parts = [binned(p, self.bin_w) for p in parts]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


class Head(nn.Module):
    def __init__(self, in_ch, n_out, k=None, width=96, n_blocks=3, per_position=False):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, k, 1) if k else None
        c0 = k or in_ch
        self.stem = nn.Conv1d(c0, width, 7, padding=3)
        # dilations double per block: receptive field 7 + 4 * (2**n_blocks - 1)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2 * d, dilation=d)
                                    for d in (2 ** i for i in range(n_blocks)))
        self.norms = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(n_blocks))
        self.per_position = per_position
        if per_position:
            self.out = nn.Conv1d(width, n_out, 1)
        else:
            self.mlp = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, n_out))

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        h = F.gelu(self.stem(x))
        for conv, norm in zip(self.blocks, self.norms):
            h = h + F.gelu(norm(conv(h)))
        if self.per_position:
            return self.out(h)
        return self.mlp(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


COND = None          # per-sample output column for conditioned regression (set in main)


def pick(out, b):
    """The prediction for each sample: its own system's output when conditioned."""
    if COND is None:
        return out[:, 0]
    return out[torch.arange(len(b)), torch.from_numpy(COND[b])]


def evaluate(model, X, y, idx, task, bs=1024):
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            o = model(torch.from_numpy(X[b]))
            preds.append(pick(o, b)[:, None] if task == "regression" else o)
    p = torch.cat(preds)
    if task == "regression":
        p = p[:, 0].numpy(); t = y[idx]
        r = float(np.corrcoef(p, t)[0, 1]); r2 = float(1 - np.mean((p - t) ** 2) / np.var(t))
        return {"pearson_r": r, "r2": r2}, r
    t = torch.from_numpy(y[idx]).long()
    acc = float((p.argmax(1) == t).float().mean())
    ce = float(F.cross_entropy(p, t))
    return {"accuracy": acc, "cross_entropy": ce}, acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="featurize.py output prefix")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--arm", required=True, help="seq | phys | phys:K | seq+phys | decode")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for the result file (tuning variants)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--outdir", default="results/readout")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    cfg = DATASETS[a.dataset]
    PB, seqs, cols, meta = load_features(a.features)
    tr, va, te = make_split(cfg, seqs, cols, a.seed)
    if a.max_train and len(tr) > a.max_train:
        tr = np.random.default_rng(a.seed).choice(tr, a.max_train, replace=False)
    S = one_hot_np(seqs)
    # The landscape stays float16 in memory (a float32 copy of the plant set alone is 29 GB);
    # batches are converted as they are drawn, and seq+phys is concatenated per batch.
    # NaN -> 0 in place, a block of rows at a time: np.nan_to_num on the whole float16
    # array peaked at 50 GB for a 14.7 GB plant-set array (measured), from its temporaries.
    for i in range(0, len(PB), 4096):
        blk = PB[i:i + 4096]
        blk[~np.isfinite(blk)] = 0
    arm = a.arm
    per_position = arm.startswith("decode")
    if arm == "seq":
        X, k = Batches(S), None
    elif arm == "phys":
        X, k = Batches(PB), None
    elif arm.startswith("phys:"):
        X, k = Batches(PB), int(arm.split(":")[1])
    elif arm == "seq+phys":
        X, k = Batches(S, PB), None
    elif arm == "decode":
        X, k = Batches(PB), None
    elif arm.startswith("decode:"):
        X, k = Batches(PB), int(arm.split(":")[1])  # how many landscape dimensions carry sequence
    elif arm.startswith("physavg:"):                # landscape smoothed over w bp
        X, k = Batches(PB, smooth_w=int(arm.split(":")[1])), None
    elif arm.startswith("decodeavg:"):              # bits of sequence left after smoothing
        X, k = Batches(PB, smooth_w=int(arm.split(":")[1])), None
    elif arm.startswith("physbin:") or arm.startswith("decodebin:"):
        # physbin:W[:k]: non-overlapping W-bp bin means (L/W values per channel), optionally
        # through a learned k-channel projection -- a bottleneck on numbers per bp, which
        # the decoder (decodebin:W[:k]) measures in bits of sequence that survive.
        parts_ = arm.split(":")
        X = Batches(PB, bin_w=int(parts_[1]))
        k = int(parts_[2]) if len(parts_) > 2 else None
    elif arm.startswith("physsel:") or arm.startswith("decodesel:"):
        # physsel:<selection>: fixed named channels at full resolution (physsel:shape4,
        # physsel:family:E, ...); decodesel:<selection> measures the bits they carry.
        sel = arm.split(":", 1)[1]
        idx = selection_indices(sel, meta["per_bp_names"])
        X, k = Batches(np.ascontiguousarray(PB[:, idx, :])), None
    elif arm == "global":                           # whole-sequence channels only
        G = np.nan_to_num(np.concatenate([np.load(f, allow_pickle=True)["glob"]
                                          for f in sorted(glob.glob(a.features + ".chunk*.npz"))]))
        X, k = Batches(np.repeat(G[:, :, None], 1, axis=2)), None
    elif arm == "decodeglobal":
        # How much per-base sequence the whole-sequence channels carry: the global vector
        # at every position plus a one-hot of the position, decoded base by base.
        G = np.nan_to_num(np.concatenate([np.load(f, allow_pickle=True)["glob"]
                                          for f in sorted(glob.glob(a.features + ".chunk*.npz"))]))
        L = S.shape[-1]
        Gr = np.repeat(G[:, :, None], L, axis=2).astype(np.float16)
        pos = np.broadcast_to(np.eye(L, dtype=np.float16)[None], (len(G), L, L))
        X, k = Batches(Gr, pos), None
    else:
        raise SystemExit(f"unknown arm {arm}")
    if per_position:
        y = S.argmax(1).astype(np.int64)          # (n, L) base identity
        task, n_out = "decode", 4
    else:
        y = np.asarray([float(v) for v in cols[cfg["target"]]], dtype=np.float64)
        if cfg["task"] != "regression":                # labels are stored as "0.0", "1.0", ...
            y = np.rint(y).astype(np.int64)
        task, n_out = cfg["task"], (1 if cfg["task"] == "regression" else cfg["n_classes"])
        if task == "regression":
            if cfg.get("condition"):
                cvals = np.asarray(cols[cfg["condition"]])
                levels = sorted(set(cvals.tolist()))
                cond = np.array([levels.index(v) for v in cvals], dtype=np.int64)
                n_out = len(levels)
                for c in range(n_out):                # standardise each system on its own
                    m = cond == c
                    mu, sd = y[np.intersect1d(tr, np.flatnonzero(m))].mean(), \
                        y[np.intersect1d(tr, np.flatnonzero(m))].std()
                    y[m] = (y[m] - mu) / sd
                y = y.astype(np.float32)
            else:
                mu, sd = y[tr].mean(), y[tr].std()
                y = ((y - mu) / sd).astype(np.float32)
    global COND
    if task == "regression" and cfg.get("condition"):
        COND = cond
    model = Head(X.shape[1], n_out, k=k, width=a.width, n_blocks=a.blocks,
                 per_position=per_position)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    best, best_state, bad, hist = -1e9, None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = np.random.default_rng(a.seed + ep).permutation(tr)
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            xb = torch.from_numpy(X[b]); out = model(xb)
            if task == "regression":
                loss = F.mse_loss(pick(out, b), torch.from_numpy(y[b]))
            elif task == "decode":
                loss = F.cross_entropy(out, torch.from_numpy(y[b]))
            else:
                loss = F.cross_entropy(out, torch.from_numpy(y[b]))
            opt.zero_grad(); loss.backward(); opt.step()
        if task == "decode":
            model.eval()
            with torch.no_grad():
                o = torch.cat([model(torch.from_numpy(X[va[s:s + 512]]))
                               for s in range(0, len(va), 512)])
            score = float((o.argmax(1) == torch.from_numpy(y[va])).float().mean())
        else:
            _, score = evaluate(model, X, y, va, task)
        hist.append({"epoch": ep, "val": score, "seconds": round(time.time() - t0)})
        print(f"epoch {ep}: val {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    res = {"dataset": a.dataset, "arm": arm, "encoder": meta.get("run"), "n_parameters": n_par,
           "head": {"width": a.width, "blocks": a.blocks, "lr": a.lr, "epochs": a.epochs,
                    "receptive_field_bp": 7 + 4 * (2 ** a.blocks - 1)}, "tag": a.tag,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "split": cfg["split"], "history": hist, "best_val": best}
    if task == "decode":
        model.eval()
        with torch.no_grad():
            o = torch.cat([model(torch.from_numpy(X[te[s:s + 512]]))
                           for s in range(0, len(te), 512)])
        t = torch.from_numpy(y[te])
        acc = float((o.argmax(1) == t).float().mean())
        ce_bits = float(F.cross_entropy(o, t)) / math.log(2)
        freq = np.bincount(y[te].ravel(), minlength=4) / y[te].size
        h_bits = float(-(freq[freq > 0] * np.log2(freq[freq > 0])).sum())
        res["test"] = {"per_base_accuracy": acc, "entropy_bits": h_bits,
                       "bits_recovered": h_bits - ce_bits, "cross_entropy_bits": ce_bits}
    else:
        res["test"], _ = evaluate(model, X, y, te, task)
        if cfg.get("by"):
            g = np.asarray(cols[cfg["by"]])
            res["test_by_" + cfg["by"]] = {
                v: evaluate(model, X, y, te[g[te] == v], task)[0]
                for v in sorted(set(g[te].tolist())) if (g[te] == v).sum() >= 50}
    print(json.dumps(res["test"]), flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    enc = os.path.basename(str(meta.get("run", "enc")).rstrip("/"))
    out = os.path.join(a.outdir, f"{a.dataset}__{arm.replace(':', '_').replace('+', '_')}__{enc}"
                       f"{'__' + a.tag if a.tag else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
