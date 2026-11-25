#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Build the stratified 61 bp fragment library that the solvers are run on.

The plan's sampling design, built for universality from the start:

  panel       windows from all fifteen panel genomes, which is what makes the
              universality certificate and the held-out-kingdom test possible
  hexamer     every one of the 4,096 hexamers placed in several fixed flanking
              contexts, which guarantees local coverage genomic sampling never gives
  extremes    composition extremes at both tails
  random      random sequence at target compositions, the off-manifold stratum
  compctrl    composition-matched control groups: sets of fragments with identical
              mononucleotide (and separately dinucleotide) composition but different
              order.  This stratum exists to run the plan's first gate, which is to
              confirm the computed potential separates sequences matched on base
              composition before anything is trained.
  mpra        the MPRA library sequences themselves (added once those data are in)
  annotated   genomic windows stratified by promoter / enhancer / intergenic / repeat
              (added once the annotations are in)

Every fragment carries its stratum, source, coordinates and composition so that any
later split can be made by sequence similarity, by composition or by species.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fasta import Fasta, gc_fraction, revcomp  # noqa: E402

FRAGMENT_BP = 61
BASES = "ACGT"


def frag_id(seq: str, stratum: str) -> str:
    h = hashlib.sha1(f"{stratum}:{seq}".encode()).hexdigest()[:16]
    return f"{stratum}_{h}"


def dinuc_counts(seq: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in range(len(seq) - 1):
        out[seq[i:i + 2]] = out.get(seq[i:i + 2], 0) + 1
    return out


def mono_counts(seq: str) -> Dict[str, int]:
    return {b: seq.count(b) for b in BASES}


# --------------------------------------------------------------------------------------
# Shuffles used by the composition control
# --------------------------------------------------------------------------------------


def mono_shuffle(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def dinuc_shuffle(seq: str, rng: random.Random, max_tries: int = 200) -> Optional[str]:
    """Altschul-Erikson dinucleotide-preserving shuffle.

    Preserving dinucleotide composition is the stricter control: mononucleotide shuffling
    leaves dinucleotide frequencies free, and stacking energetics and groove geometry are
    largely dinucleotide-driven, so a pipeline could pass a mononucleotide-matched test
    for the wrong reason.
    """
    target = dinuc_counts(seq)
    first, last = seq[0], seq[-1]
    for _ in range(max_tries):
        # Build the edge multigraph, pick a random last-edge spanning tree (Euler path).
        edges: Dict[str, List[str]] = {b: [] for b in BASES}
        for i in range(len(seq) - 1):
            edges[seq[i]].append(seq[i + 1])
        for b in edges:
            rng.shuffle(edges[b])

        # Last edge from each vertex must lead toward the terminal vertex (connectivity).
        last_edge: Dict[str, str] = {}
        ok = True
        for b in BASES:
            if b == last or not edges[b]:
                continue
            # Try to pick a last edge whose vertex can reach `last` using chosen edges.
            choices = list(range(len(edges[b])))
            rng.shuffle(choices)
            picked = None
            for idx in choices:
                last_edge[b] = edges[b][idx]
                if _reaches_all(last_edge, last):
                    picked = idx
                    break
                del last_edge[b]
            if picked is None:
                ok = False
                break
            edges[b].append(edges[b].pop(picked))  # move chosen edge to the end
        if not ok:
            continue

        out = [first]
        cursor = {b: 0 for b in BASES}
        cur = first
        for _ in range(len(seq) - 1):
            if cursor[cur] >= len(edges[cur]):
                ok = False
                break
            nxt = edges[cur][cursor[cur]]
            cursor[cur] += 1
            out.append(nxt)
            cur = nxt
        if not ok or len(out) != len(seq):
            continue
        cand = "".join(out)
        if cand[-1] == last and dinuc_counts(cand) == target:
            return cand
    return None


def _reaches_all(last_edge: Dict[str, str], terminal: str) -> bool:
    """Every vertex with a chosen last edge must reach `terminal` by following them."""
    for start in last_edge:
        seen, cur = set(), start
        while cur != terminal:
            if cur in seen or cur not in last_edge:
                return False
            seen.add(cur)
            cur = last_edge[cur]
    return True


# --------------------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------------------


def sample_genomic(fa: Fasta, n: int, rng: random.Random, species: str,
                   length: int = FRAGMENT_BP, max_tries_factor: int = 40
                   ) -> List[dict]:
    """Uniform random windows, rejecting any window containing a non-ACGT base."""
    names = [c for c in fa.names if fa.length(c) >= length + 2]
    if not names:
        return []
    weights = [fa.length(c) for c in names]
    total = sum(weights)
    out: List[dict] = []
    seen = set()
    tries = 0
    while len(out) < n and tries < n * max_tries_factor:
        tries += 1
        r = rng.random() * total
        acc = 0
        contig = names[-1]
        for c, w in zip(names, weights):
            acc += w
            if r <= acc:
                contig = c
                break
        start = rng.randrange(0, fa.length(contig) - length)
        seq = fa.fetch(contig, start, start + length).upper()
        if len(seq) != length or any(b not in BASES for b in seq):
            continue
        key = (contig, start)
        if key in seen:
            continue
        seen.add(key)
        strand = rng.choice("+-")
        if strand == "-":
            seq = revcomp(seq)
        out.append({
            "stratum": "panel",
            "sequence": seq,
            "source": species,
            "contig": contig,
            "start": start,
            "strand": strand,
        })
    return out


def make_hexamer_library(flank_contexts: Sequence[Tuple[str, str]],
                         length: int = FRAGMENT_BP) -> List[dict]:
    """Every hexamer centred in each fixed flanking context."""
    out: List[dict] = []
    for ctx_name, flank in flank_contexts:
        for i in range(4**6):
            hexamer = "".join(BASES[(i >> (2 * k)) & 3] for k in range(6))
            pad = length - 6
            left = pad // 2
            right = pad - left
            seq = (flank * ((left // len(flank)) + 1))[:left] + hexamer + \
                  (flank * ((right // len(flank)) + 1))[:right]
            out.append({
                "stratum": "hexamer",
                "sequence": seq,
                "source": f"hexamer:{ctx_name}",
                "contig": ctx_name,
                "start": left,
                "strand": "+",
                "hexamer": hexamer,
            })
    return out


def make_random(n_per_gc: int, gc_targets: Sequence[float], rng: random.Random,
                stratum: str = "random", length: int = FRAGMENT_BP) -> List[dict]:
    out: List[dict] = []
    for gc in gc_targets:
        for _ in range(n_per_gc):
            seq = "".join(
                rng.choice("GC") if rng.random() < gc else rng.choice("AT")
                for _ in range(length)
            )
            out.append({
                "stratum": stratum,
                "sequence": seq,
                "source": f"{stratum}:gc{gc:.2f}",
                "contig": f"gc{gc:.2f}",
                "start": 0,
                "strand": "+",
                "gc_target": gc,
            })
    return out


def make_composition_control(n_groups: int, group_size: int, rng: random.Random,
                             gc_targets: Sequence[float] = (0.25, 0.40, 0.50, 0.60, 0.75),
                             length: int = FRAGMENT_BP) -> List[dict]:
    """Groups of fragments sharing exact composition but differing in order.

    Each group contains one seed sequence, `group_size` mononucleotide-preserving
    shuffles and `group_size` dinucleotide-preserving shuffles.  Within a group every
    fragment has identical mononucleotide composition; the dinucleotide members also
    share dinucleotide composition.  Any channel that is a function of composition alone
    must be constant within a group.
    """
    out: List[dict] = []
    per_gc = max(1, n_groups // len(gc_targets))
    gidx = 0
    for gc in gc_targets:
        for _ in range(per_gc):
            seed = "".join(
                rng.choice("GC") if rng.random() < gc else rng.choice("AT")
                for _ in range(length)
            )
            group = f"cc{gidx:04d}"
            gidx += 1
            out.append({
                "stratum": "compctrl", "sequence": seed, "source": "compctrl:seed",
                "contig": group, "start": 0, "strand": "+",
                "control_group": group, "control_role": "seed", "gc_target": gc,
            })
            seen = {seed}
            for _ in range(group_size):
                for _try in range(50):
                    s = mono_shuffle(seed, rng)
                    if s not in seen:
                        break
                seen.add(s)
                out.append({
                    "stratum": "compctrl", "sequence": s, "source": "compctrl:mono",
                    "contig": group, "start": 0, "strand": "+",
                    "control_group": group, "control_role": "mono_shuffle",
                    "gc_target": gc,
                })
            for _ in range(group_size):
                s = None
                for _try in range(50):
                    cand = dinuc_shuffle(seed, rng)
                    if cand and cand not in seen:
                        s = cand
                        break
                if s is None:
                    continue
                seen.add(s)
                out.append({
                    "stratum": "compctrl", "sequence": s, "source": "compctrl:dinuc",
                    "contig": group, "start": 0, "strand": "+",
                    "control_group": group, "control_role": "dinuc_shuffle",
                    "gc_target": gc,
                })
    return out


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

FIELDS = [
    "frag_id", "stratum", "sequence", "length", "gc", "source", "contig", "start",
    "strand", "control_group", "control_role", "hexamer", "gc_target", "clade",
]


def finalise(records: Iterable[dict]) -> List[dict]:
    out = []
    for r in records:
        seq = r["sequence"]
        rec = {k: r.get(k, "") for k in FIELDS}
        rec["sequence"] = seq
        rec["length"] = len(seq)
        rec["gc"] = round(gc_fraction(seq), 6)
        rec["frag_id"] = frag_id(seq, r["stratum"])
        out.append(rec)
    return out


def write_library(records: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for r in records:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", help="panel_prepared.json from prepare_genomes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--per-species", type=int, default=4000)
    ap.add_argument("--hexamer-contexts", type=int, default=3)
    ap.add_argument("--random-per-gc", type=int, default=200)
    ap.add_argument("--control-groups", type=int, default=20)
    ap.add_argument("--control-size", type=int, default=6)
    ap.add_argument("--strata", default="panel,hexamer,extremes,random,compctrl")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    want = set(s.strip() for s in a.strata.split(",") if s.strip())
    records: List[dict] = []
    summary: Dict[str, int] = {}

    if "panel" in want:
        if not a.panel:
            print("--panel is required for the panel stratum", file=sys.stderr)
            return 2
        with open(a.panel) as f:
            panel = json.load(f)
        for g in panel["genomes"]:
            with Fasta(g["fasta"]) as fa:
                recs = sample_genomic(fa, a.per_species, rng, g["species"])
            for r in recs:
                r["clade"] = g["clade"]
            records += recs
            print(f"[panel] {g['species']:32s} {len(recs):6d} fragments", flush=True)

    if "hexamer" in want:
        contexts = [("polyA", "A"), ("alternatingGC", "GC"), ("mixed", "ACGT")]
        records += make_hexamer_library(contexts[: a.hexamer_contexts])

    if "extremes" in want:
        records += make_random(a.random_per_gc, (0.05, 0.10, 0.15, 0.85, 0.90, 0.95),
                               rng, stratum="extremes")

    if "random" in want:
        records += make_random(a.random_per_gc, (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
                               rng, stratum="random")

    if "compctrl" in want:
        records += make_composition_control(a.control_groups, a.control_size, rng)

    final = finalise(records)
    # Drop exact duplicate sequences within a stratum, keeping the first occurrence.
    seen, deduped = set(), []
    for r in final:
        if r["frag_id"] in seen:
            continue
        seen.add(r["frag_id"])
        deduped.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_library(deduped, a.out)
    for r in deduped:
        summary[r["stratum"]] = summary.get(r["stratum"], 0) + 1

    meta = {
        "seed": a.seed,
        "fragment_bp": FRAGMENT_BP,
        "n_fragments": len(deduped),
        "by_stratum": summary,
        "gc_min": min(r["gc"] for r in deduped),
        "gc_max": max(r["gc"] for r in deduped),
        "library": a.out,
    }
    with open(a.out.replace(".tsv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
