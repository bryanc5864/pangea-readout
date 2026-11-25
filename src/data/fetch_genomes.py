#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Download the fifteen-genome species panel used for the universality certificate.

For each species we resolve the current RefSeq reference assembly through the NCBI
Datasets API, locate its directory on the NCBI FTP mirror, download the genomic FASTA
and its md5 checksum, verify the checksum, and record everything in a manifest.

Usage:
    python fetch_genomes.py --out /path/to/genomes [--species "Homo sapiens" ...]
                            [--manifest-only] [--jobs 4]

The manifest (genomes_manifest.json) is the record of provenance: accession, assembly
name, URL, md5, file size, reported GC percent and genome length for every species.
No sequence file is ever edited by hand.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
UA = {"User-Agent": "pangea-genome-fetch/1.0"}

# The panel from the plan. clade is recorded for the held-out-kingdom split.
PANEL = [
    ("Homo sapiens", "animal"),
    ("Mus musculus", "animal"),
    ("Danio rerio", "animal"),
    ("Drosophila melanogaster", "animal"),
    ("Caenorhabditis elegans", "animal"),
    ("Arabidopsis thaliana", "plant"),
    ("Zea mays", "plant"),
    ("Saccharomyces cerevisiae", "fungus"),
    ("Escherichia coli", "bacterium"),
    ("Mycobacterium tuberculosis", "bacterium"),
    ("Streptomyces coelicolor", "bacterium"),
    ("Plasmodium falciparum", "protist"),
    ("Dictyostelium discoideum", "protist"),
    ("Methanocaldococcus jannaschii", "archaeon"),
    ("Thermus thermophilus", "bacterium"),
]


def fetch(url: str, tries: int = 4, timeout: int = 60) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last = e
            time.sleep(2 * 2**i)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def resolve_assembly(species: str) -> dict:
    """Current RefSeq reference assembly for a species, with assembly statistics."""
    quoted = urllib.parse.quote(species)
    for filt in ("&filters.reference_only=true", ""):
        url = (
            f"{API}/genome/taxon/{quoted}/dataset_report"
            f"?filters.assembly_source=refseq{filt}&page_size=20"
        )
        try:
            reports = json.loads(fetch(url)).get("reports", [])
        except RuntimeError:
            continue
        # Prefer the most complete assembly, then the longest.
        order = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
        reports = [r for r in reports if r.get("accession", "").startswith("GCF_")]
        if not reports:
            continue
        reports.sort(
            key=lambda r: (
                order.get(r.get("assembly_info", {}).get("assembly_level"), 9),
                -int(r.get("assembly_stats", {}).get("total_sequence_length") or 0),
            )
        )
        r = reports[0]
        stats = r.get("assembly_stats", {})
        return {
            "species": species,
            "accession": r["accession"],
            "organism_name": r.get("organism", {}).get("organism_name"),
            "assembly_level": r.get("assembly_info", {}).get("assembly_level"),
            "gc_percent": stats.get("gc_percent"),
            "total_sequence_length": int(stats.get("total_sequence_length") or 0),
        }
    raise RuntimeError(f"no RefSeq assembly found for {species}")


def ftp_dir(accession: str) -> str:
    """Resolve the NCBI FTP directory holding an assembly, e.g. GCF_000001405.40."""
    m = re.match(r"^(GC[AF])_(\d{3})(\d{3})(\d{3})\.(\d+)$", accession)
    if not m:
        raise ValueError(f"unexpected accession format: {accession}")
    pre, a, b, c, _ = m.groups()
    base = f"{FTP}/{pre}/{a}/{b}/{c}/"
    listing = fetch(base).decode("utf-8", "replace")
    # Directory names are "<accession>_<assembly name>"; take the one matching ours.
    names = re.findall(r'href="([^"]+)/"', listing)
    hits = [n for n in names if n.startswith(accession + "_")]
    if not hits:
        hits = [n for n in names if n.startswith(accession.split(".")[0])]
    if not hits:
        raise RuntimeError(f"no FTP directory for {accession} under {base}")
    hits.sort()
    return base + hits[-1] + "/"


def md5sum(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dest)


def get_species(species: str, clade: str, out: str, manifest_only: bool) -> dict:
    info = resolve_assembly(species)
    info["clade"] = clade
    d = ftp_dir(info["accession"])
    stem = d.rstrip("/").rsplit("/", 1)[-1]
    fna_url = f"{d}{stem}_genomic.fna.gz"
    info["assembly_dir"] = d
    info["fasta_url"] = fna_url
    dest = os.path.join(out, f"{stem}_genomic.fna.gz")
    info["path"] = dest

    # Expected md5 from the assembly's own checksum file.
    expected = None
    try:
        for line in fetch(d + "md5checksums.txt").decode().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].endswith(os.path.basename(dest)):
                expected = parts[0]
                break
    except RuntimeError:
        pass
    info["md5_expected"] = expected

    if manifest_only:
        info["status"] = "manifest-only"
        return info

    if os.path.exists(dest) and expected and md5sum(dest) == expected:
        info["status"] = "cached"
    else:
        download(fna_url, dest)
        info["status"] = "downloaded"
    info["bytes"] = os.path.getsize(dest)
    got = md5sum(dest)
    info["md5"] = got
    info["md5_ok"] = (expected is None) or (got == expected)
    if not info["md5_ok"]:
        raise RuntimeError(f"md5 mismatch for {species}: {got} != {expected}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    panel = PANEL if not a.species else [(s, dict(PANEL).get(s, "unknown")) for s in a.species]

    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {
            ex.submit(get_species, sp, cl, a.out, a.manifest_only): sp for sp, cl in panel
        }
        for fu in concurrent.futures.as_completed(futs):
            sp = futs[fu]
            try:
                r = fu.result()
                results.append(r)
                print(
                    f"[ok] {sp:32s} {r['accession']:20s} "
                    f"GC={r['gc_percent']}% len={r['total_sequence_length']:,} {r['status']}",
                    flush=True,
                )
            except Exception as e:
                errors.append({"species": sp, "error": f"{type(e).__name__}: {e}"})
                print(f"[FAIL] {sp}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    results.sort(key=lambda r: r["gc_percent"] if r["gc_percent"] is not None else 0)
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "NCBI RefSeq via Datasets API v2alpha",
        "n_species": len(results),
        "genomes": results,
        "errors": errors,
    }
    mpath = os.path.join(a.out, "genomes_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {mpath}")
    if results:
        gcs = [r["gc_percent"] for r in results if r["gc_percent"] is not None]
        print(f"GC range across panel: {min(gcs):.1f}% to {max(gcs):.1f}%")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
