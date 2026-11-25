# PANGEA / READOUT

A biophysical landscape of nucleic acids, and a measurement of how much of regulatory
function it reaches.

The research plan of record is [`PLAN.md`](PLAN.md), recorded verbatim as supplied.
Current state is in [`STATUS.md`](STATUS.md); every autonomous decision, deviation and
correction is in [`DECISIONS.md`](DECISIONS.md).

This repository is at the **data stage**: ground-truth generation and dataset
acquisition. The encoder, the READOUT heads and the downstream measurements are not
built yet.

## Where everything is

| What | Where |
|---|---|
| Code, configs, tests, plan, decision log, status, summary results (JSON), structure templates | this GitHub repository |
| Computed labels, fragment libraries, candidate sets, packed targets, code snapshots that produced them | Hugging Face dataset [`bryan7264/pangea-readout`](https://huggingface.co/datasets/bryan7264/pangea-readout) |
| Trained models | none yet: no PANGEA model is trained until every family's labels exist |

The labels are computed physics (electrostatics, folding, coarse-grained and atomistic MD,
quantum chemistry), not experimental measurements. Third-party experimental datasets and
reference genomes are not redistributed; `configs/*.json` and `src/data/fetch_*.py`
fetch them from their sources and check them against published checksums.

## Environment

Yale Bouchet HPC (Slurm, account `pi_skr2`). All commands assume:

```bash
source scripts/env.sh          # Lmod + conda, and the project path variables
pangea_activate phys           # the solver environment
```

The `phys` environment is built once with:

```bash
bash scripts/make_env_phys.sh  # APBS 3.4.1, PDB2PQR, OpenMM, pdbfixer, MDAnalysis, SciPy stack
```

Quantum chemistry uses the cluster's own modules rather than conda. `xtb` needs
libraries from its Lmod dependency modules, so it is called through a wrapper with the
library path resolved once:

```bash
export XTB_BIN=$PWD/scripts/xtb_wrapper.sh
```

Storage: code and results live on the backed-up home fileset; bulk data lives in
`/home/bjc75/scratch/pangea` (symlinked as `data/scratch`). The conda package cache is
on node-local SSD because extracting it on NFS is pathologically slow.

## Reproducing the data stage

Each stage is one command and writes a provenance record.

### 1. Species panel

```bash
python src/data/fetch_genomes.py    --out data/scratch/genomes --jobs 5
python src/data/prepare_genomes.py  --manifest data/scratch/genomes/genomes_manifest.json \
                                    --out data/scratch/genomes/fa --jobs 8
```

Resolves the fifteen reference assemblies through the NCBI Datasets API, downloads them,
verifies each against the assembly's own md5, then decompresses and indexes them for
random access and measures composition. Measured GC spans 19.3 percent
(*Plasmodium falciparum*) to 72.0 percent (*Streptomyces coelicolor*).

### 2. Fragment library

```bash
python src/data/sample_fragments.py \
    --panel data/scratch/genomes/fa/panel_prepared.json \
    --out data/scratch/fragments/library_v1.tsv \
    --per-species 8000 --random-per-gc 500 --control-groups 50 --control-size 6
```

139,381 fragments of 61 bp across five strata: `panel`, `hexamer` (every 4,096 hexamer
in three fixed flanking contexts), `extremes`, `random`, and `compctrl`. The `compctrl`
stratum is 50 groups of composition-matched sequences used by the first gate; within a
group every fragment shares mononucleotide composition exactly, and the
dinucleotide-shuffled members share dinucleotide composition exactly as well.

### 3. Public datasets

```bash
python src/data/fetch_datasets.py --manifest configs/experimental_datasets.json \
    --out data/scratch/assays --jobs 4
python src/data/fetch_datasets.py --manifest configs/activity_datasets.json \
    --out data/scratch/assays --jobs 4
```

The manifests carry the verified URL, the expected byte count, the column layout, the
license status and, where the plan's description of a source proved wrong, a
`plan_correction` field. Downloads are checked against the expected size, because a
portal serving a login interstitial otherwise surfaces much later as an unparseable
file. A sha256 is recorded per file.

### 4. Electrostatics ground truth

One fragment, all nine ionic conditions:

```bash
python src/physics/run_fragment.py --sequence <61-mer> --frag-id demo \
    --out data/scratch/apbs/demo --workroot /tmp
```

A shard of the library on Slurm:

```bash
sbatch --array=0-21 scripts/slurm_apbs_array.sh \
    data/scratch/fragments/library_v1.tsv data/scratch/apbs/compctrl 30 compctrl
```

The pipeline is: cgNA+ sequence-dependent ground state, all-atom reconstruction,
restrained minimisation under AMBER OL15, base-pair frames re-fitted from the minimised
structure, PQR with AMBER charges and radii, focused nonlinear Poisson-Boltzmann in
APBS, then sampling on cylindrical shells in the base-pair frame (six angular sectors by
six radial offsets) over the central 21 bp only.

Before running at scale, size the solver from measurement rather than assumption:

```bash
python src/physics/solver_probe.py --solute-bp 41 61 --fine-spacing 0.5 0.35 \
    --transverse-pad 30 50 --out results/solver_probe_v1.json
```

The probe reports wall time *and* channel-level agreement against the most expensive
converged configuration, so cost and accuracy are traded off explicitly.

### 5. Quantum chemistry

```bash
bash scripts/run_qc_tier1.sh
```

Builds capped base-pair-dimer geometries for all 136 reverse-complement-canonical
tetramer contexts and runs the GFN2-xTB tier over them. Geometries come from cgNA+, so
the central step responds to its flanking bases before any quantum chemistry runs; a
fixed idealised step would make the result depend on sequence only through base identity.

Interaction energies are defined by difference:
`E_stack = E(dimer) - E(bp1) - E(bp2)` and `E_hbond = E(bp) - E(watson) - E(crick)`.
These are **total** interaction energies, not the electrostatic / exchange-repulsion /
induction / dispersion decomposition the plan calls for; that requires the
symmetry-adapted perturbation theory tier and is not implemented yet.

### 6. Assay ingestion and leakage-controlled splits

```bash
python src/data/ingest_assays.py   --assays data/scratch/assays --out data/scratch/processed
python src/data/ingest_activity.py --assays data/scratch/assays --out data/scratch/processed
python src/data/add_mpra_stratum.py --processed data/scratch/processed \
    --library data/scratch/fragments/library_v1.tsv \
    --out data/scratch/fragments/library_v2.tsv
python src/data/splits.py --table data/scratch/processed/activity_deepstarr.tsv \
    --out results/splits_deepstarr.json --n-folds 10
```

The ingesters validate before writing, and the validation is dataset-specific: the
loop-seq adapter layout is confirmed rather than assumed, both Array Melt thermodynamic
identities are tested row by row (one file in that release has an entropy column
computed with Tm in Celsius), and the DeepSTARR FASTA and activity files are joined by
row order only after their record counts are checked equal.

`splits.py` groups sequences by MinHash similarity and assigns whole groups to folds, so
near-duplicate elements cannot straddle a split boundary, then audits the largest
similarity that still crosses one. A random split of an MPRA library leaks, and the
resulting score measures memorisation while looking normal.

Coordinate-only assays need their mapping established rather than assumed:

```bash
python src/data/calibrate_yeast_coords.py \
    --occupancy data/scratch/assays/nucleosome_invitro/InVitro_normalized.tab.gz \
    --out results/yeast_coord_calibration.json
```

### 7. Analytic targets

```bash
python src/physics/analytic_targets.py --sequence <seq> --stiffness
```

Nearest-neighbour duplex thermodynamics (SantaLucia unified parameters with the
Owczarzy sodium correction) and the cgNA+ ground state and stiffness matrix. These are
the unlimited-quantity labels for the first training stage.

## Tests

```bash
python -m pytest tests/ -q
```

48 tests. They exist because the failure mode that matters here is a silent convention
error, not a crash. They cover the Watson-Crick pair geometry that detects a
Crick-frame flip handled the wrong way round, the groove direction that fixes what an
angular sector means, reverse-complement parity, exact trilinear interpolation of
OpenDX maps, electroneutrality of every ionic condition, the dinucleotide-preserving
shuffle, reproduction of all ten published nearest-neighbour free energies, and a
construction where planted near-duplicates must land in the same fold while unrelated
sequences must not be merged.

## Layout

```
PLAN.md  STATUS.md  DECISIONS.md  README.md  LICENSE
configs/     verified dataset manifests
src/data/    genome and dataset acquisition, fragment library
src/physics/ structure building, minimisation, frames, APBS, gates, solver probe
src/qc/      quantum chemistry fragment library and the xtb tier
scripts/     environment bootstrap, Slurm array job, xtb wrapper
tests/       unit tests
results/     machine-readable outputs; every reported number traces here
```

## License

MIT, Bryan Cheng, 2026. See [`LICENSE`](LICENSE).

Third-party data retains its own terms. Several sources carry no explicit license and
are used here for non-commercial academic research with citation; the splice-site
benchmark's parent repository is CC BY-NC-SA 4.0. Per-dataset license status is recorded
in the manifests under `configs/` and summarised in `DECISIONS.md`. No third-party data
is redistributed from this repository.
