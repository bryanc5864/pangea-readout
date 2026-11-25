#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Download the datasets listed in a manifest, verify, and record provenance.

Manifests live in configs/*.json.  Each file entry may carry `expect_bytes`, which is
checked after download: a size that does not match usually means a portal served an
interstitial or a login page rather than the data, which is a failure mode that
otherwise shows up much later as an unparseable file.

Writes <out>/download_manifest_<name>.json with the URL, byte count, sha256 and status
for every file, so any number traced back later has a provenance record.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "pangea-data-fetch/1.0"}


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str, tries: int = 3, timeout: int = 300) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last = e
            time.sleep(3 * 2**i)
    raise RuntimeError(f"download failed: {url}: {last}")


def fetch_file(ds_name: str, entry: dict, out_root: str, force: bool) -> dict:
    dest = os.path.join(out_root, entry["dest"])
    rec = {"dataset": ds_name, "url": entry["url"], "path": dest}
    expect = entry.get("expect_bytes")
    if os.path.exists(dest) and not force:
        size = os.path.getsize(dest)
        if expect is None or size == expect:
            rec.update({"status": "cached", "bytes": size, "sha256": sha256(dest)})
            return rec
    download(entry["url"], dest)
    size = os.path.getsize(dest)
    rec.update({"status": "downloaded", "bytes": size, "sha256": sha256(dest)})
    if expect is not None:
        rec["expect_bytes"] = expect
        rec["size_ok"] = size == expect
        if size != expect:
            rec["status"] = "size_mismatch"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    with open(a.manifest) as f:
        spec = json.load(f)
    datasets = spec["datasets"]
    if a.datasets:
        datasets = [d for d in datasets if d["name"] in set(a.datasets)]

    tasks = [(d["name"], e) for d in datasets for e in d["files"]]
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fetch_file, n, e, a.out, a.force): (n, e["dest"])
                for n, e in tasks}
        for fu in concurrent.futures.as_completed(futs):
            name, dest = futs[fu]
            try:
                r = fu.result()
                results.append(r)
                flag = "" if r["status"] != "size_mismatch" else "  <-- SIZE MISMATCH"
                print(f"[{r['status']:14s}] {name:20s} {dest:48s} "
                      f"{r['bytes']/1e6:9.2f} MB{flag}", flush=True)
            except Exception as e:
                errors.append({"dataset": name, "dest": dest,
                               "error": f"{type(e).__name__}: {e}"})
                print(f"[FAILED        ] {name:20s} {dest}: {e}",
                      file=sys.stderr, flush=True)

    tag = os.path.basename(a.manifest).replace(".json", "")
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": a.manifest,
        "n_files": len(results),
        "total_bytes": sum(r["bytes"] for r in results),
        "files": sorted(results, key=lambda r: (r["dataset"], r["path"])),
        "errors": errors,
        "dataset_metadata": {d["name"]: {k: v for k, v in d.items() if k != "files"}
                             for d in datasets},
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"download_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(results)} files, {out['total_bytes']/1e9:.2f} GB -> {path}")
    mism = [r for r in results if r["status"] == "size_mismatch"]
    if mism:
        print(f"{len(mism)} size mismatches: inspect before using those files")
    return 1 if errors or mism else 0


if __name__ == "__main__":
    sys.exit(main())
