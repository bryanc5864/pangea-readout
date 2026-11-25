#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Turn the downloaded experimental assays into sequence-and-target tables.

Only the assays that ship sequences can become supervised channels directly; the ones
distributing coordinates alone (G4-seq, in vitro nucleosome occupancy, CPD-seq,
dCPD-seq) need a genome lift first and are handled elsewhere.  This module covers the
three sequence-based ones:

  loopseq      intrinsic cyclizability C0 for the 50 bp variable region  (Family D)
  array_melt   hairpin dG37 and Tm with the dot-bracket target structure (Family E)
  orchid       the measured 256-tetramer and 64-trimer cleavage tables    (Family G)

Each ingester validates before it writes.  The checks are not decoration: the Array Melt
release contains a published file whose entropy column is computed with Tm in Celsius
rather than Kelvin, and an ingester that did not test dG = dH - T dS would absorb that
silently.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tarfile
import time
from typing import Dict, Iterable, List, Optional, Tuple

LOOPSEQ_ADAPTER_5 = "TTTCTTCACTTATCTCCCACCGTCC"
LOOPSEQ_ADAPTER_3 = "GGCAGAAGACAAGGGAACGAAATAG"
LOOPSEQ_VARIABLE_BP = 50
R_CAL = 1.9872
KELVIN = 273.15


# --------------------------------------------------------------------------------------
# loop-seq
# --------------------------------------------------------------------------------------


def ingest_loopseq(paths: Dict[str, str], out_dir: str) -> dict:
    """Extract the 50 bp variable region and C0 from the loop-seq supplementary tables."""
    rows: List[dict] = []
    report: Dict[str, object] = {"libraries": {}}
    for library, path in paths.items():
        if not os.path.exists(path):
            report["libraries"][library] = {"status": "missing", "path": path}
            continue
        n_in = n_adapter_ok = 0
        lib_rows = []
        with open(path) as f:
            raw_header = f.readline().rstrip("\n")
            first_row = f.readline().rstrip("\n")
            seq_i, c0_i, how = _locate_columns(raw_header, first_row)
            if seq_i is None or c0_i is None:
                report["libraries"][library] = {
                    "status": "could not locate columns",
                    "header": raw_header.split("\t")}
                continue
            f.seek(0)
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(seq_i, c0_i):
                    continue
                seq = parts[seq_i].strip().upper()
                try:
                    c0 = float(parts[c0_i])
                except ValueError:
                    continue
                n_in += 1
                has_adapters = (seq.startswith(LOOPSEQ_ADAPTER_5)
                                and seq.endswith(LOOPSEQ_ADAPTER_3))
                if has_adapters:
                    n_adapter_ok += 1
                    variable = seq[len(LOOPSEQ_ADAPTER_5):
                                   len(seq) - len(LOOPSEQ_ADAPTER_3)]
                else:
                    variable = seq
                if len(variable) != LOOPSEQ_VARIABLE_BP:
                    continue
                if any(b not in "ACGT" for b in variable):
                    continue
                lib_rows.append({
                    "dataset": "loopseq", "library": library,
                    "sequence": variable, "length": len(variable),
                    "target_name": "C0_intrinsic_cyclizability", "target": c0,
                })
        rows += lib_rows
        report["libraries"][library] = {
            "status": "ok", "rows_read": n_in,
            "rows_kept": len(lib_rows),
            "adapter_match_fraction": round(n_adapter_ok / n_in, 4) if n_in else None,
        }

    if rows:
        vals = [r["target"] for r in rows]
        report["n_rows"] = len(rows)
        report["target_min"] = min(vals)
        report["target_max"] = max(vals)
        report["target_mean"] = sum(vals) / len(vals)
        report["unique_sequences"] = len({r["sequence"] for r in rows})
        _write_tsv(rows, os.path.join(out_dir, "loopseq_c0.tsv"),
                   ["dataset", "library", "sequence", "length", "target_name", "target"])
        report["output"] = os.path.join(out_dir, "loopseq_c0.tsv")
    return report


# --------------------------------------------------------------------------------------
# Array Melt
# --------------------------------------------------------------------------------------


def ingest_array_melt(path: str, out_dir: str, tol_dg: float = 0.05,
                      tol_tm: float = 0.5) -> dict:
    """Ingest hairpin thermodynamics, checking dG and Tm against dH and dS.

    The release is known to contain a file whose dS column uses Tm in Celsius.  Rather
    than trusting the chosen file, both identities are tested row by row:
        dG37 = dH - 310.15 * dS
        Tm   = dH / dS - 273.15
    and the fraction of rows satisfying each is reported.  A low fraction means the file
    carries the bug and must not be used.
    """
    report: Dict[str, object] = {"path": path}
    if not os.path.exists(path):
        report["status"] = "missing"
        return report

    rows: List[dict] = []
    n = ok_dg = ok_tm = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"RefSeq", "dH", "dS", "Tm", "dG_37"} - set(reader.fieldnames or [])
        if missing:
            report["status"] = "unexpected columns"
            report["missing"] = sorted(missing)
            return report
        for rec in reader:
            try:
                dh = float(rec["dH"])
                ds = float(rec["dS"])
                tm = float(rec["Tm"])
                dg = float(rec["dG_37"])
            except (TypeError, ValueError):
                continue
            seq = (rec.get("RefSeq") or "").strip().upper()
            if not seq or any(b not in "ACGT" for b in seq):
                continue
            n += 1
            if abs(dg - (dh - 310.15 * ds)) < tol_dg:
                ok_dg += 1
            if ds != 0 and abs(tm - (dh / ds - KELVIN)) < tol_tm:
                ok_tm += 1
            rows.append({
                "dataset": "array_melt", "sequence": seq, "length": len(seq),
                "dH_kcal_mol": dh, "dS_kcal_mol_K": ds, "Tm_C": tm,
                "dG37_kcal_mol": dg,
                "target_struct": (rec.get("TargetStruct") or "").strip(),
                "series": (rec.get("Series") or "").strip(),
                "seqid": (rec.get("SEQID") or "").strip(),
            })

    report["n_rows"] = n
    report["fraction_dG_consistent"] = round(ok_dg / n, 5) if n else None
    report["fraction_Tm_consistent"] = round(ok_tm / n, 5) if n else None
    report["status"] = "ok" if n and ok_dg / n > 0.95 and ok_tm / n > 0.95 \
        else "INCONSISTENT: do not use this file"
    if rows:
        lens = [r["length"] for r in rows]
        report["length_min"], report["length_max"] = min(lens), max(lens)
        report["series_counts"] = _counts(r["series"] for r in rows)
        _write_tsv(rows, os.path.join(out_dir, "array_melt_thermo.tsv"),
                   ["dataset", "seqid", "sequence", "length", "target_struct", "series",
                    "dH_kcal_mol", "dS_kcal_mol_K", "Tm_C", "dG37_kcal_mol"])
        report["output"] = os.path.join(out_dir, "array_melt_thermo.tsv")
    return report


# --------------------------------------------------------------------------------------
# ORChID
# --------------------------------------------------------------------------------------


def ingest_orchid(tarball: str, out_dir: str) -> dict:
    """Extract the measured tetramer and trimer cleavage tables from the ORChID server code.

    The tables are embedded as literal data inside the Perl module rather than shipped
    as files.  The tetramer table has one row per tetramer with a mean and a standard
    deviation at each of the four positions; the trimer table is the three-position
    fallback used when a tetramer was not measured.

    Note the correction recorded in DECISIONS.md: ORChID is a tetramer model with a
    trimer fallback, not the 512-pentamer table the plan describes.
    """
    report: Dict[str, object] = {"tarball": tarball}
    if not os.path.exists(tarball):
        report["status"] = "missing"
        return report

    text = None
    with tarfile.open(tarball, "r:gz") as tf:
        names = tf.getnames()
        report["members"] = names
        for member in tf.getmembers():
            if member.name.endswith(".pm") or member.name.endswith(".pl"):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                content = fh.read().decode("utf-8", "replace")
                if re.search(r"\b[ACGT]{4}\b", content):
                    text = content if text is None else text + "\n" + content
    if text is None:
        report["status"] = "no perl source with sequence tables found"
        return report

    tetramers: Dict[str, List[float]] = {}
    trimers: Dict[str, List[float]] = {}
    for line in text.splitlines():
        # Rows look like: GAGA 8 0.742 0.058 1.117 0.025 0.240 0.391 0.246 0.288
        m = re.match(r"^\s*['\"]?([ACGT]{3,4})['\"]?[\s,]+(.*)$", line)
        if not m:
            continue
        word, rest = m.group(1), m.group(2)
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+|\b\d+\b", rest)]
        if len(word) == 4 and len(nums) >= 9:
            tetramers.setdefault(word, nums[:9])
        elif len(word) == 3 and len(nums) >= 7:
            trimers.setdefault(word, nums[:7])

    report["n_tetramers"] = len(tetramers)
    report["n_trimers"] = len(trimers)
    report["status"] = "ok" if len(tetramers) == 256 else (
        f"expected 256 tetramers, parsed {len(tetramers)}")

    if tetramers:
        rows = [{"tetramer": k, "n_instances": v[0],
                 **{f"mean_pos{i+1}": v[1 + 2 * i] for i in range(4)},
                 **{f"sd_pos{i+1}": v[2 + 2 * i] for i in range(4)}}
                for k, v in sorted(tetramers.items())]
        _write_tsv(rows, os.path.join(out_dir, "orchid_tetramer_256.tsv"),
                   ["tetramer", "n_instances"]
                   + [f"mean_pos{i+1}" for i in range(4)]
                   + [f"sd_pos{i+1}" for i in range(4)])
        report["tetramer_output"] = os.path.join(out_dir, "orchid_tetramer_256.tsv")
    if trimers:
        rows = [{"trimer": k, "n_instances": v[0],
                 **{f"mean_pos{i+1}": v[1 + 2 * i] for i in range(3)},
                 **{f"sd_pos{i+1}": v[2 + 2 * i] for i in range(3)}}
                for k, v in sorted(trimers.items())]
        _write_tsv(rows, os.path.join(out_dir, "orchid_trimer_64.tsv"),
                   ["trimer", "n_instances"]
                   + [f"mean_pos{i+1}" for i in range(3)]
                   + [f"sd_pos{i+1}" for i in range(3)])
        report["trimer_output"] = os.path.join(out_dir, "orchid_trimer_64.tsv")
    return report


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _locate_columns(raw_header: str, first_row: str
                    ) -> Tuple[Optional[int], Optional[int], str]:
    """Find the sequence and C0 columns, using the data rather than the header alone.

    The loop-seq tables are not consistently formatted: three libraries separate
    "Sequence #" and "Sequence" with a tab, while the nucleosomal library separates them
    with spaces, so its header has one field fewer than its data rows.  Matching on
    header text alone therefore either misses the sequence column or silently selects
    the index column, which would train a cyclizability channel against row numbers.

    The sequence column is located by looking for a long pure-ACGT field in the first
    data row, which is unambiguous.  The C0 column is matched by header name, counting
    from the right so a short header does not shift it.
    """
    row = first_row.split("\t")
    seq_i = None
    for i, field in enumerate(row):
        v = field.strip().upper()
        if len(v) >= 40 and all(b in "ACGT" for b in v):
            seq_i = i
            break

    header = [h.strip() for h in raw_header.split("\t")]
    c0_i = None
    if "C0" in header:
        # Align on the right: data rows may carry one extra leading field.
        offset = len(row) - len(header)
        c0_i = header.index("C0") + (offset if header.index("C0") > 0 else 0)
        if c0_i >= len(row):
            c0_i = None
    if c0_i is None:
        # Fall back to the documented layout: C0 follows the three raw cyclizability
        # measurements, so it is the fourth numeric field after the sequence.
        if seq_i is not None and seq_i + 4 < len(row):
            c0_i = seq_i + 4
    return seq_i, c0_i, "data-located sequence, header-aligned C0"


def _write_tsv(rows: List[dict], path: str, fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _counts(values: Iterable[str]) -> Dict[str, int]:
    from collections import Counter

    return dict(Counter(values).most_common())


def ingest_nucleosome_invitro(occ_path: str, genome_glob: str, out_dir: str,
                              window: int = 147, stride: int = 75) -> dict:
    """The plan's positive control: in vitro nucleosome occupancy (Kaplan et al. 2009, GEO
    GSM351491 normalised track), as READOUT rows.

    sacCer3 is tiled with `window` bp windows every `stride` bp on the 16 nuclear
    chromosomes; each row's target is the mean normalised occupancy over the central 21 bp
    (the map's own coordinates used directly: the calibration found offsets of 0-10 bp,
    `results/yeast_coord_calibration.json`). Rows need full occupancy coverage of that centre
    and an all-ACGT window. The stated baseline is GC: a 147 bp GC window alone correlates
    with this map at r = 0.78 (DECISIONS 2026-11-19), so the control is read against it.
    """
    import glob as _glob
    import gzip as _gzip
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fasta import Fasta as _Fasta
    import numpy as np
    occ: Dict[str, Dict[int, float]] = {}
    with _gzip.open(occ_path, "rt") as f:
        f.readline()
        for line in f:
            q = line.rstrip("\n").split("\t")
            if len(q) < 3:
                continue
            try:
                occ.setdefault(q[0], {})[int(q[1])] = float(q[2])
            except ValueError:
                continue
    fa_path = sorted(_glob.glob(genome_glob))[0]
    rows, n_skip = [], 0
    half, c0 = window // 2, 10
    with _Fasta(fa_path) as fa:
        contigs = [n for n in fa.names if n.startswith("NC_")][:16]
        for k, contig in enumerate(contigs):
            chrom = str(k + 1)
            track = occ.get(chrom, {})
            L = fa.length(contig)
            seq_all = fa.fetch(contig, 0, L).upper()
            for centre in range(half + 1, L - half, stride):          # 1-based centre
                vals = [track.get(p) for p in range(centre - c0, centre + c0 + 1)]
                if any(v is None for v in vals):
                    n_skip += 1
                    continue
                w = seq_all[centre - 1 - half: centre - 1 - half + window]
                if len(w) != window or any(b not in "ACGT" for b in w):
                    n_skip += 1
                    continue
                rows.append({"dataset": "nucleosome_invitro", "library": f"chr{chrom}",
                             "sequence": w, "length": window,
                             "target_name": "invitro_occupancy_central21", "target": float(np.mean(vals)),
                             "chromosome": chrom, "centre_1based": centre,
                             # whole chromosomes held out: windows overlap by 72 bp, so a
                             # random or similarity split would put neighbours on both sides
                             "split": "test" if chrom in ("15", "16") else "train"})
    out = os.path.join(out_dir, "nucleosome_invitro.tsv")
    _write_tsv(rows, out, ["dataset", "library", "sequence", "length", "target_name", "target",
                           "chromosome", "centre_1based", "split"])
    t = np.array([r["target"] for r in rows])
    gc = np.array([(r["sequence"].count("G") + r["sequence"].count("C")) / window for r in rows])
    return {"output": out, "rows": len(rows), "skipped": n_skip, "genome": os.path.basename(fa_path),
            "window": window, "stride": stride, "target_mean": float(t.mean()), "target_sd": float(t.std()),
            "gc_pearson_r": float(np.corrcoef(gc, t)[0, 1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assays", required=True, help="download root, e.g. data/scratch/assays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    want = set(a.only) if a.only else {"loopseq", "array_melt", "orchid"}
    report: Dict[str, object] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if "nucleosome_invitro" in want:
        report["nucleosome_invitro"] = ingest_nucleosome_invitro(
            os.path.join(a.assays, "nucleosome_invitro/InVitro_normalized.tab.gz"),
            "/home/bjc75/scratch/pangea/genomes/fa/GCF_000146045*.fa", a.out)

    if "loopseq" in want:
        report["loopseq"] = ingest_loopseq({
            "random": os.path.join(a.assays, "loopseq/random_library.txt"),
            "tiling": os.path.join(a.assays, "loopseq/tiling_library.txt"),
            "chrv": os.path.join(a.assays, "loopseq/chrv_library.txt"),
            "nucleosomal": os.path.join(a.assays, "loopseq/nucleosomal_library.txt"),
        }, a.out)
    if "array_melt" in want:
        report["array_melt"] = ingest_array_melt(
            os.path.join(a.assays, "array_melt/arr_v1_adjusted.csv"), a.out)
    if "orchid" in want:
        report["orchid"] = ingest_orchid(
            os.path.join(a.assays, "orchid/orchid2_server_predictions.tar.gz"), a.out)

    path = os.path.join(a.out, "ingest_report.json")
    # merge into an existing report: a run restricted with --only must not erase the record
    # of datasets it did not touch
    merged = {}
    if os.path.exists(path):
        try:
            merged = json.load(open(path))
        except ValueError:
            merged = {}
    merged.update(report)
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(json.dumps(report, indent=2)[:4000])
    print(f"\n-> {path}")
    bad = [k for k, v in report.items()
           if isinstance(v, dict) and str(v.get("status", "ok")).lower() not in ("ok", "none")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
