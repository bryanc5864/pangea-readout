#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Minimal FASTA indexing and random access, standard library only.

Writes a samtools-compatible .fai index and serves arbitrary subsequences without
loading a genome into memory.  Kept dependency-free so genome preparation does not wait
on the conda environment, and so the data stage has no moving parts that can silently
change version.
"""
from __future__ import annotations

import gzip
import os
import shutil
from typing import Dict, Iterator, List, Optional, Tuple

COMPLEMENT = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv", "TGCANtgcanYRMKSWVHDByrmkswvhdb")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def decompress(gz_path: str, out_path: str, overwrite: bool = False) -> str:
    if os.path.exists(out_path) and not overwrite:
        return out_path
    tmp = out_path + ".part"
    with gzip.open(gz_path, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 22)
    os.replace(tmp, out_path)
    return out_path


def build_fai(fasta_path: str, fai_path: Optional[str] = None) -> str:
    """Build a samtools-style .fai: name, length, offset, line_bases, line_width."""
    fai_path = fai_path or fasta_path + ".fai"
    records: List[Tuple[str, int, int, int, int]] = []
    with open(fasta_path, "rb") as f:
        name = None
        seq_len = 0
        offset = 0
        line_bases = 0
        line_width = 0
        ragged = False
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b">"):
                if name is not None:
                    records.append((name, seq_len, offset, line_bases, line_width))
                name = line[1:].split()[0].decode()
                seq_len = 0
                offset = f.tell()
                line_bases = 0
                line_width = 0
                ragged = False
            else:
                stripped = line.rstrip(b"\r\n")
                n = len(stripped)
                if line_bases == 0:
                    line_bases, line_width = n, len(line)
                elif n != line_bases and not ragged:
                    # Final short line of a record is normal; anything else is ragged
                    # and makes constant-stride indexing invalid.
                    nxt = f.readline()
                    f.seek(f.tell() - len(nxt) if nxt else f.tell())
                    if nxt and not nxt.startswith(b">"):
                        raise ValueError(
                            f"{fasta_path}: ragged line lengths in {name}, cannot index"
                        )
                    ragged = True
                seq_len += n
        if name is not None:
            records.append((name, seq_len, offset, line_bases, line_width))
    with open(fai_path, "w") as f:
        for r in records:
            f.write("\t".join(str(x) for x in r) + "\n")
    return fai_path


class Fasta:
    """Random access to an indexed FASTA."""

    def __init__(self, path: str):
        self.path = path
        fai = path + ".fai"
        if not os.path.exists(fai):
            build_fai(path, fai)
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        with open(fai) as f:
            for line in f:
                name, length, offset, lb, lw = line.rstrip("\n").split("\t")
                self.index[name] = (int(length), int(offset), int(lb), int(lw))
        self._fh = open(path, "rb")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Fasta":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def names(self) -> List[str]:
        return list(self.index)

    def length(self, name: str) -> int:
        return self.index[name][0]

    def total_length(self) -> int:
        return sum(v[0] for v in self.index.values())

    def fetch(self, name: str, start: int, end: int) -> str:
        """0-based, half-open [start, end)."""
        length, offset, lb, lw = self.index[name]
        start = max(0, start)
        end = min(length, end)
        if end <= start:
            return ""
        first = offset + (start // lb) * lw + (start % lb)
        last = offset + (end - 1) // lb * lw + ((end - 1) % lb) + 1
        self._fh.seek(first)
        raw = self._fh.read(last - first)
        return raw.replace(b"\n", b"").replace(b"\r", b"").decode()


def gc_fraction(seq: str) -> float:
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    return gc / (gc + at) if (gc + at) else 0.0
