#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The channel vocabulary: what every PANGEA output declares about itself.

`PLAN.md` revision 2 makes the landscape a five-axis tensor -- position, observable,
condition, molecule class, and **resolution**. Three properties have to travel with each
channel rather than live in someone's head, because each one is load-bearing and each
one fails silently if it is wrong.

**Resolution.** Revision 2: "Many of the quantities in the channel set are not defined at
a point at all. Persistence length, cyclizability, nucleosome formation free energy,
strand separation free energy, stress-induced destabilization and RNA folding free energy
are all properties of an extent, and forcing them into a per-base output either discards
them or misrepresents them." A channel declares PER_BP, WINDOW (with its ladder rung) or
GLOBAL, so the pyramid head can be built from the declaration instead of a hand-written
list that drifts.

**Parity under reverse complement.** Revision 2, and the traps table: "An antisymmetric
channel predicted with the wrong sign passes every correlation check." This is the one
error class that no validation number reveals, so parity is declared per channel and
checked as a unit test before anything downstream runs. SYMMETRIC means the channel is
unchanged when the duplex is read from the other strand; ANTISYMMETRIC means it changes
sign; STRANDED means the channel is defined per strand and the two swap.

COMPOSITE is the honest fourth answer: the channel does transform under reverse
complement, but not by a sign applied to itself alone, because reverse complement maps
it onto a *different* channel or a different index. Family C's per-step 12 by 12
stiffness is the case that forced this value to exist. Its 12 coordinates are the
intra-base-pair coordinates of base pair i together with the inter-base-pair step from
i to i+1, which is the standard rigid-base grouping and the one `PLAN.md` asks for --
but reverse complement sends intra index i to n-1-i and step index i to n-2-i, so the
pair (intra_i, step_i) lands on (intra_j, step_{j-1}), which is a different block. The
smallest reverse-complement-covariant object is the 18 by 18 marginal over
(intra_i, step_i, intra_{i+1}), with 171 entries rather than 78.

Declaring these COMPOSITE rather than guessing a sign is what keeps the parity test
meaningful: a test that passes because the tolerance was loosened tells you nothing, and
a sign asserted on a channel that has no sign is exactly the silent error the parity
machinery exists to catch. `check_parity` and `symmetrise_channels` both pass COMPOSITE
channels through untouched, and the family carrying them owes its own check.

**Provisional.** A channel whose ground truth is not yet rigorously parameterised must
not quietly become a training target. `provisional=True` says the values are a documented
placeholder, and the trainer is expected to refuse them unless explicitly allowed. The
project rule is that every number traces to code that ran; this flag marks the ones whose
*parameters* do not yet trace to a published source.

**Index map under reverse complement.** A parity says what happens to a channel's
*value*; `rc_shift` says what happens to its *index*. They are separate facts and both
are load-bearing. PANGEA's first equivariant merge reflected every per-base-pair channel
with a plain flip, i -> n-1-i, which is right for a base-pair quantity and wrong for a
step quantity, whose reflection is i -> n-2-i. The parity tests had measured that shift
and the model did not use it, so the merge averaged every step quantity with its
neighbour's -- and because twist alternates strongly from step to step, `C_twist` trained
to an R-squared of 0.09. Thirty-five channels were affected.

Nothing here computes physics. This is the declaration layer the families register into.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# The window ladder fixed by PLAN.md revision 2.
WINDOW_LADDER_BP: Tuple[int, ...] = (10, 20, 50, 100, 200, 500)


class Resolution(enum.Enum):
    """Which of the three output resolutions a channel lives at."""

    PER_BP = "per_bp"
    WINDOW = "window"
    GLOBAL = "global"


class Parity(enum.Enum):
    """Behaviour under reverse complement of the whole duplex."""

    SYMMETRIC = "symmetric"            # unchanged
    ANTISYMMETRIC = "antisymmetric"    # changes sign
    STRANDED = "stranded"              # defined per strand; the two strands exchange
    COMPOSITE = "composite"            # see below


@dataclass(frozen=True)
class ChannelSpec:
    """One output channel of the landscape."""

    name: str
    family: str                        # "F", "L", ... as lettered in PLAN.md
    resolution: Resolution
    parity: Parity
    units: str
    description: str
    window_bp: Optional[int] = None     # required iff resolution is WINDOW
    condition_dependent: bool = False
    provisional: bool = False
    provisional_reason: str = ""
    source: str = ""                    # the publication the parameters come from
    # How a PER_BP channel's *index* maps under reverse complement, measured rather than
    # assumed: position i maps to n-1-i+rc_shift. Zero for a quantity indexed by base
    # pair; -1 for one indexed by the step between base pairs i and i+1, which maps to
    # the step between n-2-i and n-1-i; +4 and -4 for the minor and major groove, which
    # are properties of a phosphate pair offset along the strand. See `rc_maps.json` and
    # tests/test_rc_maps.py, and the next field's note for why this exists.
    rc_shift: int = 0
    # True for an angle that wraps at +-180 degrees -- a backbone torsion, an azimuth, a
    # bend direction. A Gaussian on raw degrees is wrong for these near the wrap, where
    # -179 and +179 are two degrees apart and look 358 apart; the trainer represents them
    # as (sin, cos).
    circular: bool = False

    def __post_init__(self) -> None:
        if self.resolution is Resolution.WINDOW and self.window_bp is None:
            raise ValueError(f"{self.name}: WINDOW channel must declare window_bp")
        if self.resolution is not Resolution.WINDOW and self.window_bp is not None:
            raise ValueError(f"{self.name}: only WINDOW channels may set window_bp")
        if self.provisional and not self.provisional_reason:
            raise ValueError(f"{self.name}: provisional channels must say why")


class ChannelRegistry:
    """The set of channels a family exposes, with the checks that keep it honest."""

    def __init__(self) -> None:
        self._specs: Dict[str, ChannelSpec] = {}

    def register(self, spec: ChannelSpec) -> ChannelSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate channel name: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def extend(self, specs: Iterable[ChannelSpec]) -> None:
        for s in specs:
            self.register(s)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> ChannelSpec:
        return self._specs[name]

    def names(self) -> List[str]:
        return list(self._specs)

    def specs(self) -> List[ChannelSpec]:
        return list(self._specs.values())

    def by_family(self, family: str) -> List[ChannelSpec]:
        return [s for s in self._specs.values() if s.family == family]

    def by_resolution(self, resolution: Resolution) -> List[ChannelSpec]:
        return [s for s in self._specs.values() if s.resolution is resolution]

    def provisional(self) -> List[ChannelSpec]:
        return [s for s in self._specs.values() if s.provisional]

    def production(self) -> List[ChannelSpec]:
        """Channels fit to be training targets: everything not flagged provisional."""
        return [s for s in self._specs.values() if not s.provisional]

    def summary(self) -> Dict[str, object]:
        res: Dict[str, int] = {}
        par: Dict[str, int] = {}
        fam: Dict[str, int] = {}
        for s in self._specs.values():
            res[s.resolution.value] = res.get(s.resolution.value, 0) + 1
            par[s.parity.value] = par.get(s.parity.value, 0) + 1
            fam[s.family] = fam.get(s.family, 0) + 1
        return {
            "total": len(self._specs),
            "by_family": fam,
            "by_resolution": res,
            "by_parity": par,
            "provisional": len(self.provisional()),
            "production": len(self.production()),
        }


# The single registry every family adds to on import.
REGISTRY = ChannelRegistry()


def reverse_complement(seq: str) -> str:
    """Reverse complement, preserving case for ACGTN and passing anything else through."""
    table = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(table)[::-1]


def check_parity(values_fwd, values_rev, spec: ChannelSpec,
                 atol: float = 1e-8, rtol: float = 1e-6) -> Tuple[bool, float]:
    """Does a channel behave as it declared under reverse complement?

    `values_fwd` are the channel's values on a sequence; `values_rev` are its values on
    the reverse complement, already reversed back into the forward coordinate frame for
    PER_BP and WINDOW channels so the two are positionally comparable.

    Returns (passed, max_absolute_discrepancy). STRANDED channels are not checked here,
    because the check depends on how the family lays out its two strands; the family
    supplies its own.
    """
    import numpy as np

    a = np.asarray(values_fwd, dtype=float)
    b = np.asarray(values_rev, dtype=float)
    if a.shape != b.shape:
        return False, float("inf")
    if spec.parity is Parity.SYMMETRIC:
        target = b
    elif spec.parity is Parity.ANTISYMMETRIC:
        target = -b
    else:
        return True, 0.0          # STRANDED and COMPOSITE: the family checks its own
    m = np.isfinite(a) & np.isfinite(target)
    if not m.any():
        return True, 0.0
    diff = np.abs(a[m] - target[m])
    tol = atol + rtol * np.abs(target[m])
    return bool(np.all(diff <= tol)), float(diff.max())


def symmetrise_channels(forward: Dict[str, float], reverse: Dict[str, float],
                        registry: "ChannelRegistry" = None,
                        skip_units: Sequence[str] = ("bp index", "bp")
                        ) -> Dict[str, float]:
    """Average a channel set with its reverse-complement computation, by declared parity.

    **Why this exists.** `PLAN.md` requires reverse-complement equivariance to be
    enforced architecturally in PANGEA. Measurement shows the cgNA+ ground truth does
    not have it exactly: on an exact palindrome, where no index mapping is ambiguous,
    the ground-state rotational coordinates are mirror-symmetric to machine precision,
    but the marginal step covariances are not -- up to 46 percent per step in tilt,
    20 percent in roll, 3.6 percent in twist, averaging to about 0.1 percent in
    aggregate quantities like persistence length. The stiffness is correctly banded and
    the row extraction matches the parameter ordering, so this is a property of the
    parameter set rather than of this code.

    The physical molecule *is* symmetric: a duplex does not know which strand you chose
    to write down. So the asymmetry is ground-truth noise, and an architecturally
    equivariant model cannot fit it -- it would become an irreducible error floor that
    looks like model error but is not. Averaging the two orientations removes it and
    makes the target exactly symmetric, which is both physically correct and what the
    architecture can represent.

    Channels whose units are positional indices are passed through from the forward
    orientation, since an index does not average meaningfully across a frame flip.
    """
    reg = registry if registry is not None else REGISTRY
    out: Dict[str, float] = {}
    for name, fwd in forward.items():
        if name not in reg:
            out[name] = fwd
            continue
        spec = reg[name]
        rev = reverse.get(name)
        if (rev is None or spec.parity in (Parity.STRANDED, Parity.COMPOSITE)
                or spec.units in skip_units):
            out[name] = fwd
            continue
        sign = -1.0 if spec.parity is Parity.ANTISYMMETRIC else 1.0
        try:
            if not (np.isfinite(fwd) and np.isfinite(rev)):
                out[name] = fwd
                continue
        except TypeError:
            out[name] = fwd
            continue
        out[name] = 0.5 * (fwd + sign * rev)
    return out



def declare_index_maps(specs: Iterable[ChannelSpec], step: Iterable[str] = (),
                       shifts: Optional[Dict[str, int]] = None,
                       circular: Iterable[str] = (),
                       parity: Optional[Dict[str, Parity]] = None) -> None:
    """Set `rc_shift`, `circular` and, where measurement corrected it, `parity`.

    Families declare these by name, after construction, so the measured facts sit in one
    readable list per family rather than scattered through constructor calls. Every name
    must exist: a typo here would silently leave a step channel reflected as a base-pair
    channel, which is the exact bug this field exists to prevent.
    """
    by_name = {s.name: s for s in specs}
    step = set(step)
    shifts = dict(shifts or {})
    circ = set(circular)
    par = dict(parity or {})
    unknown = (step | set(shifts) | circ | set(par)) - set(by_name)
    if unknown:
        raise KeyError(f"declare_index_maps: no such channels {sorted(unknown)}")
    for name in step:
        shifts.setdefault(name, -1)
    for name, sh in shifts.items():
        object.__setattr__(by_name[name], "rc_shift", int(sh))
    for name in circ:
        object.__setattr__(by_name[name], "circular", True)
    for name, p in par.items():
        object.__setattr__(by_name[name], "parity", p)
