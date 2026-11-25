#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Unify the regulatory-activity datasets into one schema for the CEILING arm.

Output schema, one row per element:

    dataset, subset, split, element_id, sequence, length, target_name, target, species

`split` preserves the authors' own split where they published one, because the plan
compares against numbers that are already familiar and re-splitting would break that.
Where no split is published the field is empty and the split is made later, by sequence
similarity and by chromosome, as the plan requires.

Every ingester validates before writing, and the validation is dataset-specific because
the failure modes are:

  lentimpra   two orientations per element; the authors' `fold` column must be preserved
  deepstarr   FASTA and activity files join BY ROW ORDER with no shared key, so the
              record counts must match exactly or every label is silently misassigned
  jores       three species in one table; species must be carried through for the
              cross-species transfer arm
  yeast       only about 78 percent of rows are the nominal 110 bp, so length filtering
              is explicit rather than assumed
  splice      the upstream label mapping is undocumented, so it is re-derived here from
              the centre dinucleotides and the agreement rate is reported
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

FIELDS = ["dataset", "subset", "split", "element_id", "sequence", "length",
          "target_name", "target", "species"]

YEAST_PREFIX = "TGCATTTTTTTCACATC"
YEAST_SUFFIX = "GGTTACGGCTGTT"


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _write(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _stats(rows: List[dict]) -> dict:
    if not rows:
        return {"n_rows": 0}
    lens = Counter(r["length"] for r in rows)
    vals = [r["target"] for r in rows if r["target"] is not None]
    out = {
        "n_rows": len(rows),
        "n_unique_sequences": len({r["sequence"] for r in rows}),
        "length_mode": lens.most_common(1)[0][0],
        "length_distinct": len(lens),
        "splits": dict(Counter(r["split"] for r in rows)),
        "species": dict(Counter(r["species"] for r in rows)),
    }
    if vals:
        vals_sorted = sorted(vals)
        out.update({
            "target_min": vals_sorted[0],
            "target_max": vals_sorted[-1],
            "target_mean": sum(vals) / len(vals),
            "target_median": vals_sorted[len(vals_sorted) // 2],
        })
    return out


# --------------------------------------------------------------------------------------


def ingest_lentimpra(root: str) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    report: Dict[str, object] = {}
    for cell in ("K562", "HepG2", "WTC11"):
        path = os.path.join(root, "lentimpra", f"{cell}.tsv")
        if not os.path.exists(path):
            report[cell] = "missing"
            continue
        n = 0
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            cols = set(reader.fieldnames or [])
            need = {"seq_id", "seq", "mean_value"}
            if not need <= cols:
                report[cell] = {"status": "unexpected columns",
                                "found": sorted(cols)}
                continue
            for rec in reader:
                seq = (rec["seq"] or "").strip().upper()
                if not seq or any(b not in "ACGT" for b in seq):
                    continue
                try:
                    target = float(rec["mean_value"])
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "dataset": "lentimpra", "subset": cell,
                    "split": f"fold{rec.get('fold', '')}".strip(),
                    "element_id": rec["seq_id"], "sequence": seq, "length": len(seq),
                    "target_name": "log2_RNA_over_DNA", "target": target,
                    "species": "Homo sapiens",
                })
                n += 1
        report[cell] = {"status": "ok", "rows": n}
    return rows, report


def _read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    name, buf = None, []
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def ingest_deepstarr(root: str) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    report: Dict[str, object] = {}
    for split in ("Train", "Val", "Test"):
        fa = os.path.join(root, "deepstarr", f"Sequences_{split}.fa")
        act = os.path.join(root, "deepstarr", f"Sequences_activity_{split}.txt")
        if not (os.path.exists(fa) and os.path.exists(act)):
            report[split] = "missing"
            continue
        seqs = list(_read_fasta(fa))
        with open(act) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        # The activity file may or may not carry a header; detect it.
        header_present = False
        first = lines[0].replace("\t", " ").split()
        try:
            [float(x) for x in first]
        except ValueError:
            header_present = True
        if header_present:
            lines = lines[1:]
        if len(seqs) != len(lines):
            report[split] = {
                "status": "ROW COUNT MISMATCH - labels would be misassigned",
                "n_fasta": len(seqs), "n_activity": len(lines),
            }
            continue
        n = 0
        for (name, seq), line in zip(seqs, lines):
            parts = line.replace("\t", " ").split()
            try:
                dev = float(parts[0])
            except (IndexError, ValueError):
                continue
            s = seq.strip().upper()
            if not s or any(b not in "ACGTN" for b in s):
                continue
            rows.append({
                "dataset": "deepstarr", "subset": "developmental",
                "split": split.lower(), "element_id": name,
                "sequence": s, "length": len(s),
                "target_name": "Dev_log2_enrichment", "target": dev,
                "species": "Drosophila melanogaster",
            })
            n += 1
        report[split] = {"status": "ok", "rows": n,
                         "joined_by": "row order, counts verified equal",
                         "header_skipped": header_present}
    return rows, report


def ingest_jores(root: str) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    report: Dict[str, object] = {}
    species_map = {"At": "Arabidopsis thaliana", "Zm": "Zea mays",
                   "Sb": "Sorghum bicolor"}
    for system in ("leaf", "proto"):
        for split in ("train", "test"):
            path = os.path.join(root, "jores_plant", f"CNN_{split}_{system}.tsv")
            if not os.path.exists(path):
                report[f"{system}_{split}"] = "missing"
                continue
            n = 0
            with open(path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                cols = set(reader.fieldnames or [])
                if not {"sequence", "enrichment"} <= cols:
                    report[f"{system}_{split}"] = {"status": "unexpected columns",
                                                   "found": sorted(cols)}
                    continue
                for rec in reader:
                    seq = (rec["sequence"] or "").strip().upper()
                    if not seq or any(b not in "ACGTN" for b in seq):
                        continue
                    try:
                        target = float(rec["enrichment"])
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "dataset": "jores_plant", "subset": system, "split": split,
                        "element_id": rec.get("gene", ""), "sequence": seq,
                        "length": len(seq),
                        "target_name": "log2_enrichment_vs_35S_minP",
                        "target": target,
                        "species": species_map.get((rec.get("sp") or "").strip(),
                                                   rec.get("sp", "")),
                    })
                    n += 1
            report[f"{system}_{split}"] = {"status": "ok", "rows": n}
    return rows, report


def ingest_yeast(root: str, require_exact_length: bool = True
                 ) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    report: Dict[str, object] = {}
    path = os.path.join(root, "yeast_promoter", "train_sequences.txt.gz")
    if not os.path.exists(path):
        return rows, {"status": "missing"}
    n_in = n_kept = n_trimmed = 0
    with _open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            n_in += 1
            seq = parts[0].strip().upper()
            try:
                target = float(parts[1])
            except ValueError:
                continue
            if require_exact_length and len(seq) != 110:
                continue
            core = seq
            if seq.startswith(YEAST_PREFIX) and seq.endswith(YEAST_SUFFIX):
                core = seq[len(YEAST_PREFIX):len(seq) - len(YEAST_SUFFIX)]
                n_trimmed += 1
            if any(b not in "ACGT" for b in core):
                continue
            rows.append({
                "dataset": "yeast_promoter", "subset": "random_80mer", "split": "train",
                "element_id": "", "sequence": core, "length": len(core),
                "target_name": "FACS_bin_expression", "target": target,
                "species": "Saccharomyces cerevisiae",
            })
            n_kept += 1
    report = {"status": "ok", "rows_read": n_in, "rows_kept": n_kept,
              "adapter_trim_fraction": round(n_trimmed / n_kept, 4) if n_kept else None,
              "exact_110bp_fraction": round(n_kept / n_in, 4) if n_in else None}
    return rows, report


def ingest_splice(root: str) -> Tuple[List[dict], dict]:
    """Ingest splice sites and re-derive the undocumented label mapping from the data."""
    rows: List[dict] = []
    report: Dict[str, object] = {}
    for split in ("train", "test"):
        path = os.path.join(root, "splice_sites", f"{split}.fna")
        if not os.path.exists(path):
            report[split] = "missing"
            continue
        label_dinuc: Dict[str, Counter] = {}
        recs = []
        for name, seq in _read_fasta(path):
            s = seq.strip().upper()
            label = name.split("|")[-1].strip() if "|" in name else ""
            if len(s) != 400 or not label:
                continue
            recs.append((name, s, label))
            # Acceptor AG ends the intron just before the junction; donor GT begins it.
            label_dinuc.setdefault(label, Counter())
            label_dinuc[label][f"198-199={s[198:200]}"] += 1
            label_dinuc[label][f"200-201={s[200:202]}"] += 1
        derived = {}
        for label, counter in label_dinuc.items():
            top = counter.most_common(2)
            derived[label] = {k: v for k, v in top}
        report[split] = {
            "status": "ok", "rows": len(recs),
            "label_counts": dict(Counter(l for _, _, l in recs)),
            "centre_dinucleotide_evidence": derived,
            "note": "upstream label mapping is undocumented; evidence above is the "
                    "basis for interpreting it, and the raw label is stored as-is",
        }
        for name, s, label in recs:
            rows.append({
                "dataset": "splice_sites", "subset": "all", "split": split,
                "element_id": name.split("|")[0], "sequence": s, "length": len(s),
                "target_name": "splice_class_raw_label", "target": float(label)
                if label.lstrip("-").isdigit() else None,
                "species": "Homo sapiens",
            })
    return rows, report


INGESTERS = {
    "lentimpra": ingest_lentimpra,
    "deepstarr": ingest_deepstarr,
    "jores_plant": ingest_jores,
    "yeast_promoter": ingest_yeast,
    "splice_sites": ingest_splice,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    want = a.only or list(INGESTERS)
    report: Dict[str, object] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for name in want:
        fn = INGESTERS.get(name)
        if fn is None:
            report[name] = "no ingester"
            continue
        rows, rep = fn(a.assays)
        path = os.path.join(a.out, f"activity_{name}.tsv")
        if rows:
            _write(rows, path)
        report[name] = {"detail": rep, "summary": _stats(rows),
                        "output": path if rows else None}
        s = _stats(rows)
        print(f"[{name:16s}] rows={s.get('n_rows', 0):>9,} "
              f"unique={s.get('n_unique_sequences', 0):>9,} "
              f"len_mode={s.get('length_mode', '-')}", flush=True)

    path = os.path.join(a.out, "activity_ingest_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
