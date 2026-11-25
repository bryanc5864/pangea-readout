# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Upload the staged deposit (scripts/hf_stage.py) to the Hugging Face dataset repo, then verify it.

One upload_folder call per top-level unit (code, libraries, code_snapshots, each targets/<name>, each raw/<name>),
smallest first; a finished unit leaves a state record and is skipped on a rerun, and upload_folder itself resumes
a unit that was interrupted. The card, MANIFEST.tsv and EXCLUDED.tsv go last. Verification compares every
MANIFEST path and size with the repo's file listing, and the sha256 of every tar part with the Hub's LFS record.
The token is read from a mode-600 file, never from the command line.
"""
import argparse
import json
import os
import sys
import time

REPO = "bryan7264/pangea-readout"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def du(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))  # follows the symlinks to the real files
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.config/pangea_tokens/hf"))
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=open(a.token_file).read().strip())
    os.makedirs(a.state, exist_ok=True)

    units = ["code", "libraries", "code_snapshots"]
    for grp in ("targets", "raw"):
        units += [f"{grp}/{d}" for d in sorted(os.listdir(os.path.join(a.stage, grp)))]
    units.sort(key=lambda u: du(os.path.join(a.stage, u)))
    if not a.verify_only:
        for u in units:
            mark = os.path.join(a.state, u.replace("/", "__") + ".done")
            if os.path.exists(mark):
                log(f"skip {u}: uploaded earlier")
                continue
            for attempt in range(1, 6):
                try:
                    t0 = time.time()
                    ci = api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=os.path.join(a.stage, u),
                                           path_in_repo=u, commit_message=f"Add {u}")
                    log(f"{u}: done in {time.time() - t0:.0f} s ({getattr(ci, 'oid', ci)})")
                    open(mark, "w").write(f"{getattr(ci, 'oid', ci)}\n")
                    break
                except Exception as e:  # network or Hub errors: back off and resume
                    log(f"{u}: attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}")
                    if attempt == 5:
                        raise
                    time.sleep(120 * attempt)
        for f in ("README.md", "MANIFEST.tsv", "EXCLUDED.tsv"):
            api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj=os.path.join(a.stage, f),
                            path_in_repo=f, commit_message=f"Add {f}")
            log(f"{f}: uploaded")

    # verification against the manifest
    remote = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
        if hasattr(e, "size") and e.__class__.__name__ == "RepoFile":
            remote[e.path] = (e.size, (e.lfs.sha256 if getattr(e, "lfs", None) else None))
    missing, badsize, badsha, n = [], [], [], 0
    with open(os.path.join(a.stage, "MANIFEST.tsv")) as f:
        next(f)
        for line in f:
            path, size, sha, _ = line.rstrip("\n").split("\t", 3)
            n += 1
            if path not in remote:
                missing.append(path)
            elif remote[path][0] != int(size):
                badsize.append(path)
            elif sha and remote[path][1] and remote[path][1] != sha:
                badsha.append(path)
    rep = {"manifest_files": n, "remote_files": len(remote), "missing": missing[:50], "n_missing": len(missing),
           "size_mismatch": badsize[:50], "n_size_mismatch": len(badsize), "sha_mismatch": badsha,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rep, open(os.path.join(a.state, "verify.json"), "w"), indent=1)
    log(f"verify: {n} in manifest, {len(remote)} on the Hub, missing {len(missing)}, size mismatch {len(badsize)}, "
        f"sha mismatch {len(badsha)}")
    sys.exit(0 if not (missing or badsize or badsha) else 4)


if __name__ == "__main__":
    main()
