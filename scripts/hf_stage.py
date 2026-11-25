# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Stage the PANGEA/READOUT Hugging Face dataset deposit (datasets/bryan7264/pangea-readout).

The staging folder mirrors the repo:
  targets/<name>/      symlinks to packed training targets (HDF5 shards), uploaded as-is
  libraries/           symlinks to the fragment libraries and the genome manifest
  raw/<name>/          uncompressed tar parts (<= 4 GiB) of solver outputs that are many small files,
                       plus <name>.index.tsv.gz listing every member, its size and its part
  code_snapshots/      tar parts of the frozen code snapshots each production run executed from
  code/                the git tree at HEAD (identical to the GitHub repository)
  MANIFEST.tsv         every staged file: path in repo, bytes, sha256 (tar parts), source
  EXCLUDED.tsv         what was left out and why

Every source is read once; tar parts are hashed while they are written. The run is restartable: a finished
archive leaves a state record and is skipped on the next run. Text files read into the archives and the code
tree are scanned for credentials; any hit aborts the run before anything is uploaded.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time

G = "/nfs/roberts/scratch/pi_skr2/bjc75/pangea"
S = "/home/bjc75/scratch/pangea"
R = "/home/bjc75/other/biophyschem"
PART_BYTES = 4 * 1024 ** 3
SCAN_MAX = 1 << 20  # text files up to 1 MiB are read whole and scanned

CACHE = re.compile(r"^landscape_cache_[0-9a-f]+\.npz$")
# credentials and personal mail domains; any hit stops the deposit
SECRET = re.compile(rb"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
                    rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----|[A-Za-z0-9._%+-]+@gmail\.com")

# packed training targets and libraries, uploaded file by file
ASIS = [(f"{G}/targets_v2", "targets/v2_main")] + \
       [(f"{G}/targets_v2_fam{f}", f"targets/v2_fam{f}") for f in ("FrGr", "I", "J", "J2", "K", "M", "N")] + [
    (f"{G}/famA_prod/long_shards", "targets/famA_long"),
    (f"{G}/famA_longset/shards", "targets/famA_longset"),
    (f"{G}/famW/shards", "targets/famW"),
    (f"{G}/aux_all", "targets/aux_all"),
    (f"{G}/aux_all_famJNK", "targets/aux_all_famJNK"),
    (f"{G}/aux_L", "targets/aux_L"),
    (f"{G}/bpp_auxAll", "targets/bpp_auxAll"),
    (f"{G}/bpp_auxL", "targets/bpp_auxL"),
    (f"{G}/bpp_v2", "targets/bpp_v2"),
    (f"{G}/fold_windows/base", "targets/fold_windows_base"),
    (f"{G}/fold_windows/famJNK", "targets/fold_windows_famJNK"),
    (f"{G}/rare_windows/base", "targets/rare_windows_base"),
    (f"{G}/rare_windows/famJNK", "targets/rare_windows_famJNK"),
    (f"{S}/fragments", "libraries/fragments"),
]
ASIS_FILES = [(f"{S}/genomes/genomes_manifest.json", "libraries/genomes_manifest.json")]

# solver outputs (many small files) -> raw/<name>/<name>.partNNN.tar; members keep the source folder name
TARS = [(f"{G}/{d}", d, ex) for d, ex in (
    ("famA_prod", ["long_shards"]), ("famA_extras", []), ("famA_longset", ["shards"]), ("famA_smoke", []),
    ("famJox", []), ("famK_prod", []), ("famM", []), ("famM3", []), ("famM_cr", []), ("famM_im", []),
    ("famM_tx", []), ("famN_es", []), ("famW", ["shards"]), ("fold_windows", ["base", "famJNK"]),
    ("rare_windows", ["base", "famJNK"]), ("isc", []), ("md_hydration", []), ("dock", []), ("qc", []),
    ("apbs", []), ("ensemble_replicate", []))] + [
    (f"{G}/archive_home/qc", "qc_hexamer_archive", []),
    (f"{S}/apbs", "apbs_home", []),
    (f"{S}/qc", "qc_home", []),
]
CODE_TARS = [(f"{G}/code", "scratch_code", []), (f"{S}/code", "home_code", [])]

EXCLUDED = [
    ("landscape_cache_*.npz (all folders)", "data-loader caches, rebuilt from the shards on first use"),
    (f"{G}/sizing", "a loader cache and symlinks to targets_v2"),
    (f"{G}/targets_v1", "superseded by targets_v2; no current script reads it"),
    (f"{S}/targets_pilot, {S}/targets_pilot_v2", "superseded pilots"),
    (f"{S}/genomes (FASTA)", "public NCBI assemblies; fetch_genomes.py re-downloads and md5-checks them "
                             "(accessions in libraries/genomes_manifest.json)"),
    (f"{S}/assays, {S}/processed", "third-party experimental datasets and their processed forms; not ours to "
                                   "redistribute; configs/*.json and src/data/fetch_datasets.py, ingest_*.py "
                                   "rebuild them"),
    (f"{S}/ext, {S}/ext_untrusted, {G}/ext_untrusted", "third-party downloads (parameter sets, PDB entries, tool "
                                                       "sources)"),
    (f"{G}/logs, {G}/tmp, {G}/runs, {G}/features", "job logs and scratch; empty folders"),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class HashWriter(io.RawIOBase):
    """Write-through file wrapper that sha256-hashes and counts what it writes."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def writable(self):
        return True

    def write(self, b):
        self.f.write(b)
        self.h.update(b)
        self.n += len(b)
        return len(b)


def walk(src, exclude_top=()):
    """Sorted (path, rel) for regular files and symlinks below src, skipping top-level excluded folders."""
    out = []
    for root, dirs, files in os.walk(src, followlinks=False):
        rel_root = os.path.relpath(root, src)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in exclude_top]
        links = [d for d in dirs if os.path.islink(os.path.join(root, d))]  # symlinked folders: store the link
        dirs[:] = sorted(d for d in dirs if d not in links)
        for f in sorted(files + links):
            p = os.path.join(root, f)
            out.append((p, os.path.normpath(os.path.join(rel_root, f))))
    return out


def scan(path, data, hits):
    if SECRET.search(data):
        hits.append(path)


def build_tar(src, name, exclude, outdir, state_dir, hits):
    state = os.path.join(state_dir, f"{outdir.replace('/', '_')}__{name}.json")
    if os.path.exists(state):
        rec = json.load(open(state))
        if all(os.path.exists(p["path"]) and os.path.getsize(p["path"]) == p["bytes"] for p in rec["parts"]):
            log(f"skip {name}: done earlier ({len(rec['parts'])} parts)")
            return rec
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):  # this archive's parts from an interrupted earlier attempt
        if f.startswith(f"{name}.part") or f == f"{name}.index.tsv.gz":
            os.remove(os.path.join(outdir, f))
    entries = [(p, r) for p, r in walk(src, exclude) if not CACHE.match(os.path.basename(p))]
    log(f"{name}: {len(entries)} entries from {src}")
    parts, index = [], []
    tf = hw = fh = None
    cur = 0

    def close():
        nonlocal tf, hw, fh
        if tf is None:
            return
        tf.close()
        fh.close()
        parts[-1].update(bytes=hw.n, sha256=hw.h.hexdigest())
        log(f"  part {len(parts) - 1}: {hw.n / 1e9:.2f} GB")
        tf = hw = fh = None

    for path, rel in entries:
        st = os.lstat(path)
        size = st.st_size if not os.path.islink(path) else 0
        if tf is None or (cur > 0 and cur + size + 1536 > PART_BYTES):
            close()
            pth = os.path.join(outdir, f"{name}.part{len(parts):03d}.tar")
            parts.append({"path": pth})
            fh = open(pth, "wb")
            hw = HashWriter(fh)
            tf = tarfile.open(fileobj=hw, mode="w|", format=tarfile.PAX_FORMAT)
            cur = 0
        ti = tf.gettarinfo(path, arcname=f"{name}/{rel}")
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        if ti.isreg():
            if size <= SCAN_MAX:
                with open(path, "rb") as f:
                    data = f.read()
                scan(path, data, hits)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            else:
                with open(path, "rb") as f:
                    tf.addfile(ti, f)
        else:
            tf.addfile(ti)
        cur += size + 512 + (-size % 512)
        index.append((f"{name}/{rel}", size, len(parts) - 1))
    close()
    idx = os.path.join(outdir, f"{name}.index.tsv.gz")
    with gzip.open(idx, "wt") as g:
        g.write("member\tbytes\tpart\n")
        for m, b, p in index:
            g.write(f"{m}\t{b}\t{p}\n")
    rec = {"name": name, "source": src, "excluded": exclude, "members": len(index), "parts": parts, "index": idx}
    json.dump(rec, open(state, "w"), indent=1)
    return rec


def write_card(stage, manifest, head):
    """Dataset card; every count and size in it is computed from the manifest."""
    units = {}
    for path, size, _, _ in manifest:
        top = path.split("/")
        key = "/".join(top[:2]) if top[0] in ("targets", "raw") and len(top) > 2 else top[0]
        n, b = units.get(key, (0, 0))
        units[key] = (n + 1, b + int(size))
    rows = "\n".join(f"| `{k}/` | {n:,} | {b / 1e9:.2f} |" for k, (n, b) in sorted(units.items()))
    tot_n, tot_b = len(manifest), sum(int(r[1]) for r in manifest)
    card = f"""---
license: mit
pretty_name: PANGEA/READOUT computed nucleic-acid biophysics labels
viewer: false
tags:
- biology
- dna
- rna
- nucleic-acids
- biophysics
- electrostatics
- molecular-dynamics
- quantum-chemistry
---

# PANGEA/READOUT: computed biophysical labels for nucleic acids

Training data for PANGEA, a sequence model of nucleic-acid biophysics, and READOUT, the measurement of how much
regulatory function that biophysics reaches. Code: [github.com/bryanc5864/pangea-readout](https://github.com/bryanc5864/pangea-readout)
(also mirrored here under `code/`, at commit `{head[:12]}`).

**The labels are computed physics, not experimental measurements**: Poisson-Boltzmann electrostatics (APBS),
secondary-structure and base-pair probabilities, coarse-grained (oxDNA2, cgNA+) and atomistic (OpenMM) simulation,
non-canonical structure building (quadruplex, triplex, i-motif, cruciform) and quantum chemistry (GFN2-xTB, DFT).
Experimental datasets are used only for the stage-2 correction and for evaluation, and are not redistributed here.

**State.** Label production was stopped on 2026-11-24 at 22:28 EST before every family had finished; the
per-family completeness at that moment is in `code/STATUS.md` and every decision is in `code/DECISIONS.md`.
No PANGEA model has been trained yet: training starts only once every family's labels exist. Models will be
published in a separate model repository.

## Layout

{tot_n:,} files, {tot_b / 1e9:.1f} GB. `MANIFEST.tsv` lists every file with its size, its source and, for tar
parts, its sha256; `EXCLUDED.tsv` lists what was left out and why.

| Folder | Files | GB |
|---|---:|---:|
{rows}

- `targets/`: packed training targets, HDF5 shards as written by the packers in `code/src/data/`
  (`generate_targets.py`, `pack_*.py`, `generate_family_*_shards.py`). `v2_main` is the main 61-bp fragment
  library; `v2_fam*` are per-family packs keyed to the same shards; `famW` holds the 10,240-bp windows.
- `libraries/`: the fragment libraries (`library_v2.tsv` is the current one) and the genome manifest (NCBI
  accessions and md5s of the fifteen reference assemblies the fragments were drawn from).
- `raw/<name>/`: solver outputs before packing, as uncompressed tar parts of at most 4 GiB. Extract with
  `cat raw/famA_prod/famA_prod.part*.tar | tar -xi` (or each part with `tar -xf`); members keep their source
  folder name. `<name>.index.tsv.gz` lists every member, its size and its part.
- `code_snapshots/`: the frozen code snapshots that each production array ran from (all but one early snapshot
  carry a `MANIFEST.sha256` of their files), so every label can be traced to the exact code that computed it.
- `code/`: the repository at commit `{head[:12]}`.

## Sources of sequence

Fragments are drawn from fifteen public reference genomes (NCBI RefSeq; see `libraries/genomes_manifest.json`),
from published MPRA libraries (DeepSTARR, lentiMPRA, Jores et al. plant promoters; sequences only, no
measurements), and from designed strata (every hexamer in fixed flanks, composition extremes, random and
composition-matched controls). Folded-structure templates are PDB entries (in `code/data/`).

## License

MIT, the same as the code.
"""
    open(os.path.join(stage, "README.md"), "w").write(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=f"{G}/hf_stage")
    ap.add_argument("--state", default=f"{G}/tmp/hf/stage_state")
    a = ap.parse_args()
    os.makedirs(a.state, exist_ok=True)
    hits, manifest = [], []

    # 1. code tree at HEAD (same as GitHub); scanned like the archives
    head = subprocess.check_output(["git", "-C", R, "rev-parse", "HEAD"], text=True).strip()
    code = os.path.join(a.stage, "code")
    if not os.path.exists(os.path.join(a.state, f"code_{head}.done")):
        subprocess.run(["rm", "-rf", code], check=True)
        os.makedirs(code)
        arch = subprocess.run(["git", "-C", R, "archive", "--format=tar", head], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", code], input=arch, check=True)
        open(os.path.join(a.state, f"code_{head}.done"), "w").write(head + "\n")
    for p, rel in walk(code):
        if os.path.islink(p):  # symlinks in the repo (logs/slurm -> scratch) are not files; upload_folder skips them
            continue
        size = os.lstat(p).st_size
        if os.path.isfile(p) and not os.path.islink(p) and size <= SCAN_MAX:
            scan(p, open(p, "rb").read(), hits)
        manifest.append((f"code/{rel}", size, "", f"git {head}:{rel}"))
    log(f"code tree at {head}: {len(manifest)} files")

    # 2. packed targets and libraries as symlinks
    for src, dest in ASIS:
        n = 0
        for p, rel in walk(src):
            if CACHE.match(os.path.basename(p)) or not os.path.isfile(p) or os.path.islink(p):
                continue
            d = os.path.join(a.stage, dest, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if not os.path.islink(d):
                os.symlink(p, d)
            manifest.append((f"{dest}/{rel}", os.path.getsize(p), "", p))
            n += 1
        log(f"{dest}: {n} files linked from {src}")
    for src, dest in ASIS_FILES:
        d = os.path.join(a.stage, dest)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not os.path.islink(d):
            os.symlink(src, d)
        manifest.append((dest, os.path.getsize(src), "", src))

    # 3. archives
    recs = []
    for group, specs in (("raw", TARS), ("code_snapshots", CODE_TARS)):
        for src, name, ex in specs:
            outdir = os.path.join(a.stage, group, name) if group == "raw" else os.path.join(a.stage, group)
            rec = build_tar(src, name, ex, outdir, a.state, hits)
            rec["group"] = group
            recs.append(rec)
            rel_dir = os.path.relpath(outdir, a.stage)
            for p in rec["parts"]:
                manifest.append((f"{rel_dir}/{os.path.basename(p['path'])}", p["bytes"], p["sha256"],
                                 f"tar of {src}" + (f" (without {', '.join(ex)})" if ex else "")))
            manifest.append((f"{rel_dir}/{os.path.basename(rec['index'])}", os.path.getsize(rec["index"]), "",
                             f"member index of {name}"))
            if hits:
                break
        if hits:
            break

    if hits:
        with open(os.path.join(a.state, "SECRET_HITS.txt"), "w") as f:
            f.write("\n".join(hits) + "\n")
        log(f"ABORT: possible credentials in {len(hits)} files; listed in {a.state}/SECRET_HITS.txt")
        sys.exit(3)

    with open(os.path.join(a.stage, "MANIFEST.tsv"), "w") as f:
        f.write("path_in_repo\tbytes\tsha256\tsource\n")
        for row in manifest:
            f.write("\t".join(str(x) for x in row) + "\n")
    with open(os.path.join(a.stage, "EXCLUDED.tsv"), "w") as f:
        f.write("source\treason\n")
        for row in EXCLUDED:
            f.write("\t".join(row) + "\n")
    write_card(a.stage, manifest, head)
    json.dump({"head": head, "archives": recs, "n_files": len(manifest),
               "bytes": sum(int(r[1]) for r in manifest)}, open(os.path.join(a.state, "summary.json"), "w"),
              indent=1)
    log(f"staged {len(manifest)} files, {sum(int(r[1]) for r in manifest) / 1e9:.1f} GB in {a.stage}")


if __name__ == "__main__":
    main()
