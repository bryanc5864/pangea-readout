#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Measure, for every per-base-pair channel, how it maps under reverse complement.

PANGEA enforces reverse-complement equivariance by merging its prediction on a sequence
with its prediction on the reverse complement, reflected back into the forward frame.
The first version reflected every per-base-pair channel with a plain flip, i -> n-1-i.
That is right for a quantity indexed by base pair and **wrong for one indexed by step**:
the step between base pairs i and i+1 maps to the step between n-2-i and n-1-i, so its
reflection is i -> n-2-i. The parity tests had measured that shift; the model did not
use it. The consequence was that the merge averaged every step quantity with its
neighbour's, and since twist alternates strongly from step to step the average destroys
it -- `C_twist` trained to R-squared 0.09.

So the map is measured here, for every channel, on random exact palindromes: for each
candidate shift s and sign, test v[i] == sign * v[n-1-i+s] to 1e-9. A random palindrome
admits only the true map exactly, so ambiguity shows up as no exact fit rather than as a
wrong answer. The result is written to `results/rc_maps.json` and the family modules
declare it; `tests/test_rc_maps.py` re-measures and asserts the declarations.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape as LS  # noqa: E402
from channels import REGISTRY, Resolution, reverse_complement  # noqa: E402

SHIFTS = (0, -1, 1, -2, 2, -3, 3, -4, 4)


def palindromes(n_pal: int = 4, half: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n_pal):
        h = "".join(rng.choice(list("ACGT"), half))
        yield h + reverse_complement(h)


def fit_map(v: np.ndarray):
    """(sign, shift, residual) of the best exact reflection, or None if none is exact."""
    n = len(v)
    best = None
    for s in SHIFTS:
        idx = np.arange(n)[::-1] + s
        ok = (idx >= 0) & (idx < n)
        t = np.full(n, np.nan)
        t[ok] = v[idx[ok]]
        m = np.isfinite(v) & np.isfinite(t)
        if m.sum() < 6:
            continue
        scale = max(np.abs(v[m]).max(), 1e-12)
        for sign in (1.0, -1.0):
            r = float(np.abs(v[m] - sign * t[m]).max() / scale)
            if best is None or r < best[2]:
                best = (sign, s, r)
    return best


def main() -> int:
    import geometry as G
    out = {}
    pals = list(palindromes())
    wd = os.environ.get("TMPDIR", "/tmp") + "/rcmap_min"
    built = G.build_minimised_structure(pals[0], wd)
    per_pal = []
    for k, p in enumerate(pals):
        prof, _ = LS.compute_landscape(p, n_ensemble=0,
                                       minimised_pdb=built["minimised_pdb"] if k == 0
                                       else None)
        per_pal.append(prof)

    for s in REGISTRY.specs():
        if s.resolution is not Resolution.PER_BP or s.family not in LS.FAMILIES:
            continue
        fits = []
        for prof in per_pal:
            v = np.asarray(prof.get(s.name, []), dtype=float)
            if v.size == 0 or np.isfinite(v).sum() < 6 or np.nanstd(v) < 1e-12:
                continue
            fits.append(fit_map(v))
        fits = [f for f in fits if f is not None]
        if not fits:
            out[s.name] = {"declared": s.parity.value, "measured": "undetermined"}
            continue
        signs = {f[0] for f in fits if f[2] < 1e-9}
        shifts = {f[1] for f in fits if f[2] < 1e-9}
        exact = all(f[2] < 1e-9 for f in fits) and len(signs) == 1 and len(shifts) == 1
        worst = max(f[2] for f in fits)
        out[s.name] = {
            "declared": s.parity.value,
            "measured": ("exact" if exact else "inexact"),
            "sign": (int(next(iter(signs))) if exact else None),
            "shift": (int(next(iter(shifts))) if exact else None),
            "best_shift_inexact": (None if exact else int(min(fits, key=lambda f: f[2])[1])),
            "worst_residual": worst,
        }
    os.makedirs("results", exist_ok=True)
    with open("results/rc_maps.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    from collections import Counter
    c = Counter((v["declared"], v["measured"], v.get("sign"), v.get("shift"))
                for v in out.values())
    print("declared parity, measured, sign, shift  ->  count")
    for k, n in sorted(c.items(), key=lambda t: (-t[1], str(t[0]))):
        print(f"  {k}  ->  {n}")
    bad = [k for k, v in out.items() if v["declared"] in ("symmetric", "antisymmetric")
           and v["measured"] == "exact" and
           ((v["sign"] == 1) != (v["declared"] == "symmetric"))]
    print("sign disagreements with the declaration:", bad[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
