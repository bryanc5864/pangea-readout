# STATUS

**Project:** PANGEA / READOUT — a universal biophysical landscape of nucleic acids
**Plan of record:** `PLAN.md` (user-supplied, recorded verbatim)
**Phase:** Build — stage-1 encoder and the scaling study
**Updated:** 2026-11-22

**2026-11-24 22:28: ALL PANGEA JOBS STOPPED on the user's order** (24 job IDs cancelled; see DECISIONS.md). Nothing is queued or running; the rows below show the state at the stop. Completed outputs are on disk.

## Label checklist (2026-11-24, 14:03) -- every family: production -> packer -> final inputs

No PANGEA model exists or trains until every row below is "packed" (user's instruction).
Remaining work ~283k CPU-h with all 43,607 cruciforms (14:03; scope settled by the plan: all candidates; DFT and
side-chain production included, pKa production not yet). Training start (`results/eta_2026-11-24_1403.json`,
`scripts/eta_labels.py`; our CPU share without week over the last 6-48 h): ~Dec 12 (Dec 6-15).

Queue routing (2026-11-24 09:15): the cluster admits only jobs of > 24 h to the week partition (submit filter:
"Jobs of 24 hours or less should be submitted to the day partition"). Listing week next to day had slipped past
that filter, so our <= 24 h tasks had been running in week; no pending job lists week any more (production arrays and
tests are day,scavenge; jobs already running there finish). The plan to give week to the DFT arrays is dropped.

| family | production | packer | state |
| --- | --- | --- | --- |
| A main + ensemble | 34152648 (+ final pass 34320113-17) | generate_family_a_long_shards.py (solve-merge fix 2026-11-24) | **solved** 191,344 / 191,344; packing 34549843 |
| A long set (205 bp tiles) | 34293939 | generate_family_a_longset_shards.py | **packed** 2,002 segments (34433091) |
| A extras (opening, dielectric) | 34298364 (subset, both salts) + 34311833 (150 mM) + leftovers waves (`famA_extras/sources.tsv`; w1 34478525) | pack_lookup_families.py --aextras ("Ax") | running, 18,430 / 191,344 at 03:50; timed-out rows rerun by `scripts/famAx_leftovers.py` waves (w2 34526398, 3,957 fragments; w3 34558351, 8,576; w4 34597879, 2,522) |
| K A-form electrostatics | 34284715 + remainder 34318664 | pack_lookup_families.py --kes ("Kes") | running, 105,798 / 574,032 at 03:50 |
| J oxDNA2 (v2: DH, H-bond, exposure terms) | 34288949 + remainder 34318878 -> famJox/solves_v2 | pack_family_jox.py (finalize now reads solves_v2) | running, 11,987 / 191,344 at 11:15; timed-out rows rerun in leftover waves (`famJox/sources.tsv`; w1 34558355, 447; w2 34597880, 425) |
| M quadruplex v3 (GPU build, then CPU solve) | 34349846 (122 x 600) -> slurm_family_m3_solve.sh | pack_family_m.py | build started 09:28; RTX PRO 6000 Blackwell nodes excluded (OpenCL fails there; 96 bad records removed, guard added); tasks 0-1 requeued; 17:30: OpenCL context leak found (task 2 saved nothing, tasks 3-4 failing; see DECISIONS); cause found 18:00: PDBFixer defaulted to CUDA and leaked a context per model; PDBFixer now on CPU (live + snapshot, 18:12); throttle 1 until a patched task shows flat GPU memory and saves records; rows of tasks 2-4 to a leftovers wave |
| M triplex / i-motif / cruciform | 34318914 / 34318915 / cruciforms: all 43,607 (stems >= 8 bp 34547759, 4,187 x 3; stems < 8 bp 34567521 + 34567522, 10,350 x 3; 12 h, 32 GB, snapshot famMcr_20261124_1000); cost test 34489354 (3 of 4 done: 1.5-2.3 h, <= 25.8 GB; stem 28 queued) | pack_family_m.py | queued (73 / 7,021; 0 / 25,868; 0 / 43,607); scope settled 14:01: all candidates (plan) |
| W long fragments | 34308662 + redo 34369810 | pack_family_w.py | **packed** 21,391 (34433091) |
| Fr R-loop energetics, Gr damage rates | done | pack_lookup_families.py | **packed** (targets_v2_famFrGr) |
| G MD hydration / docking / trimers | 34298686 (+ shards 8-12 rerun 34550015) / done / 34298869 | pack_lookup_families.py --md --dock --trimers | running / done / 10 of 68 |
| N modified-base electrostatics ("Nes") | 34436498 (16 tasks, 7 mods, same-protocol references incl. 2'-OH controls) | pack_lookup_families.py --nes | **solved** 224 / 224 (all channel values finite); packing 34549844 |
| N modified-stack DFT | 34319769 | Bq-style table | queued |
| B pKa shifts | ddX eta 0.3 lmax 12/14/16 tests 34325002-4 | Bq table | lmax 12 vs 14: shifts move 0.005-0.115 pKa units (done); lmax 16 unconverged with fast multipoles on (34325004), rerun with them off at 16 and 14 (34547318-9, queued) |
| I side-chain groove energies ("Isc", held apart) | v1 did not converge (240 starts, 34324960); v2 Monte Carlo minimisation, convergence test 34492018 (2 contexts x 7 residues x 2 seeds) | pack_lookup_families.py --isc (refuses an incomplete table) | v2 convergence test queued |
| 69 provisional channels | no published parameterisation (`results/provisional_channels.json`) | -- | excluded from training and headline numbers, as amendment A1 specifies (settled 14:01) |

## Current state (2026-11-22)

**No PANGEA model exists and none is training** (user's instruction): every earlier model and
model-derived artifact was deleted (112 GB); the final 2M-1B ladder is trained once, on every
family, when the labels are complete (`scripts/finalize_v6_inputs.sh`, then
`scripts/submit_final_ladder.sh`).

**Labels in production:** Family A on the plan's design (full corpus at 10, 50, 150, 500 mM; 15%
subsets elsewhere; ensemble); Family M (quadruplex windows); QC tiers (DFT/SAPT 3 geometries, Stage B,
pKa probe); Family J oxDNA2 (first 20%); Family K A-form electrostatics (probe). Ready: K, J2, N, M
(corpus) side shards; auxiliary windows with J/N/K; quadruplex and rare-event windows.

**Being built:** triplex, cruciform and i-motif builders; non-parallel G4; J electrostatics/hydration/
reactivity; K 2'-OH hydration; A dielectric response and base-pair opening; N regenerated with 2'-OMe.

## Current state (2026-11-21, 08:20)

**Release ladder v5** training on GPUs (2M-1B all running or chained; 10M at epoch 44: per-bp
0.9971, whole-sequence 0.9933 over 623 / 508 channels). Weakest: folding families J, L, N
(data-limited) and rare-event / circular channels.

**Final ladder v6 prepared** (`scripts/submit_final_ladder.sh`, pre-registered criteria in
DECISIONS): v5 + Family A (long windows, ensemble), Family M, Family B quantum chemistry
("Bq"), J revision 2, J/N/K on the 1.02M auxiliary windows, quadruplex windows and
rare-event windows. Launch order: inputs finish -> `scripts/finalize_v6_inputs.sh` -> merged
cache -> ladder.

**Labelling in flight:**
- Family A production (gate PASSED on the final construction): 746,070 solves + 153,424
  ensemble snapshot solves, split by memory (jobs 34118988/89, 34118990-97).
- QC tiers: DFT B3LYP-D3BJ/def2-TZVP + SAPT0 (stacking and H-bonds) at 3 geometries, 136
  contexts (34118606-08) -> 61 "Bq" channels by table lookup (code and tests done).
- Family M (quadruplexes): 799 corpus fragments (34117983) + 9,287 modelled quadruplex
  windows (34119213); validation vs solved structures (34116210).
- Auxiliary J/N/K labels on 1.02M windows (34115850); quadruplex windows (23,792) and
  rare-event windows (15 genomes) labelled with the cheap families (34118201/02).
- Electrostatics convergence study (documents residual flank error; held behind production).

**Aim 1/2 analyses added:** split conformal calibration (RESULTS 35), composition
residualisation (RESULTS 36), positive control (in vitro nucleosome occupancy) ingested with
a chromosome hold-out and READOUT arms queued, GC + 5-mer ridge primary gate running.

## Current state (2026-11-21, 06:50)

**v5 training** on GPUs (2M epoch 45, val per-bp 0.9934 over 623 channels; 10M/50M/200M started).

**Electrostatics construction final pending its gate:** converged minimiser, exclusion radius
2.0 A (KCl 2.36 A), exact 0.35 A fine spacing, extended extraction, per-base-pair
solvation (per-atom energies; the nonlinear total energy is broken on long solutes and not
used), reference solves, per-condition solute length. Gate on exactly this running
(`compctrl_v5`). Convergence conclusions verified to carry over from 1.16 to 2.0 A.

**Family A data path built and tested:** 121 bp windows (`extend_windows.py`, 0 centre
mismatches), manifest with per-fragment conditions, resumable production runner,
long-window shards and a long-window training path (`--long-shards`; 70 tests pass);
ensemble snapshots (`--snapshot`). End-to-end smoke test on real data queued. Proposed
production: full corpus at 50, 150, 500 mM; 15% subsets at 10, 25, 100, 250 mM, KCl,
Na/Mg; ensemble 10% x 8 at 150 mM (~23,000 CPU-h).

**Probes running:** size-modified PB (`smpbe`) on the long-solute overflow; Psi4 timings.

## Current state (2026-11-21, 05:30)

**Incident handled:** the home fileset (which includes `~/scratch` and `~/project`) hit its
500,000-file quota at ~04:45 and every job writing there failed. Files moved to group
scratch, `logs/slurm` and code snapshots now live there, failed jobs resubmitted
(DECISIONS.md, incident entry). Memory note saved.

**v5 release ladder** resubmitted on the merged cache (623 per-bp + 508 whole-sequence
channels incl. K, J, N); first links waiting for GPUs. `v4f_L200M` resumes from epoch 22.

**Electrostatics:** the nonlinear total energy is broken on 61 bp solutes (focusing
boundary through the helix); solvation redefined per base pair from per-atom energies
(the earlier solvation SNR retracted). Gate on the exact production construction queued.
Outer shells (7-11 A from the surface) look unconverged at 61 bp from moderate salt
(extrapolated ~0.26 SD at 150 mM); a whole-solute fine box at 0.7 A is being tested.

**Quantum chemistry:** Psi4 1.11 + simple-dftd3 installed; centred hexamer tier done; three
stacking geometries built from cgNA+'s softest step mode; DFT/SAPT timing probe queued.

## Current state (2026-11-21, 04:00)

**Families now implemented** (14 in the plan): B, C, D, E, F, G, H, L as before; **J**
(single-stranded DNA self-structure, 54 channels), **K** (RNA duplex and DNA-RNA hybrid,
410), **N** (RNA modification response: m6A, pseudouridine, inosine, 42) and **I**
(protein readout, held apart, 60) added today; **A** (electrostatics) declared (479
channels), extraction complete for every map-derived quantity, production pending the
convergence study. Not started: **M** (non-canonical folds as explicit geometry).

**Release ladder `v5`** (2M-1B on B-H, J, K, L, N; strand tie incl. K/J/N pairs; aux
windows; v2.1 targets) is queued behind the merged cache (K+J+N), which waits on the
Family N side shards. `v6` adds Family A.

**Electrostatics today:** composition gate FAIL as first computed (noise floor from an
older construction), diagnostic PASS with a matched floor, **binding PASS** on the
converged minimiser (91% of channels SNR > 5, median 28.8). Found and fixed: the
200-iteration minimiser (path noise), KCl identical to NaCl (APBS's single largest-radius
accessibility), grid spacing varying with solute length and condition, and a confounded
convergence protocol (re-minimised per length, per-length grids). 500 mM restored at
41 bp. Final gate on the full production construction and a fixed-grid convergence
validation running.

**Other fixes today:** side-shard merge layout check (blocked the first cache build);
hexamer QC tier one base off centre (rebuilt, re-running); Family I metadata builder.

## Current state (2026-11-20, 16:30)

**Plan revision 3**, amendments A1-A7 (`PLAN.md`); revision 2 archived verbatim.

**GPU partitions saturated** (every GPU type fully allocated; 180-340 higher-priority jobs
per partition; this account's fair-share is low after today's use). Queued, in priority
order: v4 ladder (2M, 50M, 200M with the strand tie), 1B last link, `v3_L10M_auxAll`,
`v4f_L200M` (final encoder: tie + whole-sequence auxiliary data + v2.1 targets), seed
replicates, the two axial base-pair arms (low priority). Three arms stopped early
(DECISIONS.md, with interim numbers).

**Results today (test side unless noted):**
- Strand tie: Family L per-bp 0.658 -> 0.784, whole-sequence 0.867 -> 0.893 (10M);
  Family F whole-sequence -0.0135 at 10M (four times the seed spread), being measured
  across capacity.
- Windows: pooled per-bp readout 0.94-0.97 -> 0.98-0.996 at every rung.
- Whole-sequence channels as per-bp functionals: not better (head 0.9992 vs 0.9939).
- Base-pair supervision: +0.14 for the axial stack (validation), none for the conv pair
  bias; not in the final encoder.
- 16-snapshot channels at their measured replicate ceilings (median gap 0.015).
- Ground truth: Family G W/C strand asymmetry at the fragment ends -> target revision
  v2.1 (interior statistics).

**Aim 2 started on CPU** (A7): `src/readout/` (featurizer with 61-bp windowing, readout
heads with the three arms, bottleneck sweep and decoder test; tests pass). Development
encoder `v3_L10M_tie`; loop-seq, splice-site and Jores plant features done or running; 30
readout jobs queued on CPU. Sourced ceiling: loop-seq technical replicates r = 0.869.

## Where the run is

The user supplied a finished plan and asked for the data stage first, so the
autonomous-researcher novelty loop is skipped (see DECISIONS.md). Work follows the order
the plan dictates for ground truth: build a sequence-dependent conformation, solve it,
and run the composition-matched control **before** committing large compute.

## Allocation, corrected

This session's interactive allocation is **cpu=1, mem=8G** (`AllocTRES=cpu=1,mem=8G`),
not the 48 cores and 368 GB of the node it sits on. `nproc` returns 1. All heavy work
goes through `sbatch` with explicit `--cpus-per-task` and `--mem`; running solvers in
this shell gets them killed by the cgroup OOM killer. See DECISIONS.md.

## Current step

Electrostatics pipeline validated end to end and running at scale on Slurm. The solver
configuration was chosen by measurement (`results/solver_probe_v1.json`): 61 bp solute,
0.35 A fine grid, 50 A transverse coarse padding, nonlinear PB, **34.9 s per solve**,
which sits inside the plan's 30 to 60 s budget.

## Next action

1. Fixed-grid convergence validation -> resubmit the study at all nine conditions on the
   fixed protocol -> production solute lengths, anchoring, low-salt long context.
2. Final gate and per-channel SNR (`channel_snr.py`) on the production construction.
3. Family N shards -> merged cache -> v5 ladder starts.
4. Family M design; Family A open items (opening, windowed persistence, ensemble).

## Done

### Environment
- [x] Environment surveyed and recorded (Slurm, modules, storage, network)
- [x] `PLAN.md` written verbatim from the user's document
- [x] conda env `phys`: APBS 3.4.1, PDB2PQR, OpenMM 8.6.1, pdbfixer, MDAnalysis,
      numpy 2.4.6, scipy 1.17.1, numba 0.68.1. Package cache moved to node-local SSD
      after three NFS-related build failures.
- [x] xtb 6.7.1 verified with a GFN2 single point; ORCA 6.1.1 present for the DFT tier

### Data acquired
- [x] Fifteen-genome panel, md5-verified, 3.04 GB. Reported GC 19.5 to 72.0 percent;
      measured GC 19.3 to 72.0 percent after fixing a biased estimator
- [x] Genomes decompressed and indexed for random access (9.6 GB)
- [x] Fragment library v1: **139,381** 61 bp fragments —
      panel 119,949 / hexamer 12,288 / extremes 3,000 / random 3,500 / compctrl 644
- [x] Composition controls verified exact: 50 groups, 0 mononucleotide mismatches,
      0 dinucleotide mismatches among seed and dinuc members
- [x] Activity datasets, 23 files, 0.91 GB: lentiMPRA (K562/HepG2/WTC11), DeepSTARR,
      Jores plant, yeast promoters, Sharpr-MPRA, splice sites
- [x] Experimental datasets, 27 files, 0.37 GB: loop-seq, ORChID (tables + hg17 wib),
      Array Melt, G4-seq (Marsico + Chambers), in vitro nucleosome occupancy,
      CPD-seq, dCPD-seq. All size-verified and spot-checked for parseability.
- [x] Every URL verified before download; provenance recorded with sha256 per file
- [x] Five factual errors in `PLAN.md` found and recorded (see DECISIONS.md)

### Code validated
- [x] cgNA+ ground state to all-atom duplex, with units and Crick-frame convention
      detected at run time rather than assumed
- [x] Restrained minimisation under AMBER OL15; restrained atoms held to 0.265 A RMSD,
      so sequence dependence survives the cleanup
- [x] Base-pair frames re-fitted from the minimised structure; groove direction
      confirmed (+x is the major groove, 91.8 percent of phosphates project negative)
- [x] 20 unit tests passing, including Watson-Crick pair geometry
      (C1'-C1' 10.685 +/- 0.070 A, glycosidic N-N 9.009 A), reverse-complement parity,
      and exact trilinear interpolation of OpenDX maps

## In flight

- [ ] Slurm 33899777: `compctrl` stratum, 644 fragments at 150 mM (the gate)
- [ ] Slurm 33899782: hexamer-context quantum chemistry, 2,080 contexts
- [ ] Slurm 33899816: lpbe against npbe across salt, which decides whether the
      high-salt conditions can be kept

## Done since the last revision

- [x] Grid-jitter noise floor measured: 0.0145 kT/e, 0.67 percent of channel magnitude,
      with positional signal 9.5 times larger
- [x] Yeast coordinates calibrated; no lift needed; GC explains R-squared 0.61 of in
      vitro nucleosome occupancy, which weakens the plan's positive control
- [x] S1-END-seq (GSE203632) and KAS-seq (GSE139420) located and downloaded; all three
      human coordinate-only assays confirmed hg19 from the data
- [x] Leakage-controlled splitting implemented and validated
- [x] MPRA stratum added: library v2 at 191,344 fragments

## Not started

- [ ] Quantum chemistry fragment library (xtb tier, ORCA DFT tier, SAPT subset)
- [ ] Analytic targets: cgNA+ geometry and stiffness, nearest-neighbour thermodynamics
- [ ] MPRA and annotation strata of the fragment library
- [ ] Genome lifts for the coordinate-only assays (G4-seq hg19, Kaplan sacCer2,
      CPD-seq sacCer3)
- [ ] PANGEA encoder; READOUT heads; CEILING; universality certificate; design aim

## Gates

| Gate | Status |
| --- | --- |
| Composition-matched control | pipeline ready, not yet run |
| Pre-training review | not reached |
| Post-results review | not reached |

## Open risk

The plan budgets 15,000 to 30,000 CPU-hours for 1.8 million nonlinear solves. That is
consistent with roughly 30 to 60 seconds per solve, so per-fragment cost has to be
measured before the full library is committed. If a solve is materially slower, the
fragment count or the condition count gets cut and the reduction is logged as a
deviation rather than quietly absorbed.

## Code map

```
src/data/fasta.py              stdlib FASTA indexing and random access
src/data/fetch_genomes.py      NCBI Datasets resolution + md5-verified download
src/data/prepare_genomes.py    decompress, index, measure composition
src/data/sample_fragments.py   stratified 61 bp library incl. composition controls
src/data/fetch_datasets.py     manifest-driven download with provenance recording
src/physics/cgna_atomic.py     cgNA+ ground state -> all-atom duplex
src/physics/minimize.py        restrained minimisation (AMBER OL15)
src/physics/frames.py          base and base-pair frames re-fitted from structure
src/physics/apbs_run.py        PQR, focused nonlinear PB, cylindrical sampling
src/physics/run_fragment.py    end-to-end per-fragment driver
src/physics/composition_gate.py  the plan's first gate
configs/activity_datasets.json      verified activity sources
configs/experimental_datasets.json  verified experimental sources + plan corrections
tests/test_physics.py          20 tests
scripts/env.sh                 Lmod + conda bootstrap
```

## Paths

- code and docs: `/home/bjc75/other/biophyschem`
- bulk data: `/home/bjc75/scratch/pangea` (also `data/scratch`)
- conda envs: `/home/bjc75/project/conda_envs`; package cache `/tmp/bjc75_conda_pkgs`

## Data stage ledger

| Item | State | Evidence |
| --- | --- | --- |
| Fifteen-genome panel | done | 19.3 to 72.0 percent GC measured, md5 verified |
| Fragment library | done | 139,381 fragments, five strata |
| Composition controls | done, exact | 0 mono and 0 dinuc mismatches in 50 groups |
| Experimental datasets | 7 of 8 | S1-END-seq / KAS-seq unverified |
| Activity datasets | done | 5 datasets, every sequence exactly its nominal length |
| Assay ingestion, sequence-based | done | loop-seq 197,151 rows at adapter match 1.000; Array Melt 100 percent thermodynamically self-consistent; ORChID 256 tetramers and 64 trimers |
| Activity ingestion | done | lentiMPRA 731,550 / DeepSTARR 484,052 / Jores 147,966 / yeast 5,511,002 / splice 30,000 |
| Analytic targets | done | 10 of 10 published nearest-neighbour free energies reproduced |
| Quantum chemistry, tetramer | done | 136 of 136, median within-step context spread 0.91 kcal/mol |
| Quantum chemistry, hexamer | queued | 2,080 contexts |
| Electrostatics config | done, measured | 34.9 s per solve at 0.35 A and 50 A padding |
| Electrostatics, gate stratum | queued | 644 fragments at 150 mM |
| Ionic conditions validated | 3 of 9 | **verified converging:** 10, 25, 150 mM. **verified diverging under npbe:** 250, 500 mM. **untested:** 50, 100 mM, KCl 150 mM, Na/Mg. A condition counts as validated only after it has completed a run. |
| Electrostatics, full library | not started | awaits the gate |
| Coordinate lifts | in progress | yeast calibration running; hg19 for G4-seq not fetched |
| MPRA stratum | code ready | not yet appended to the library |
| SAPT tier, transfer integrals | not started | the plan's expensive electronic-structure tier |
