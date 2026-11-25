# DECISIONS

Every autonomous decision, deviation and environment fact, with reasons. Append only.

## 2026-11-19 — Environment

Host: Yale Bouchet HPC (`bouchet.ycrc.yale.edu`), running on compute node `c1106u13n03`
in the `agent` partition (48 cores, 368 GB RAM, 7-day limit). Login node is `login1`.

- Scheduler: Slurm. Account `pi_skr2`, QOS `interactive,normal`.
- CPU partitions: `day` (1 day, 172 nodes), `week` (7 days, 128 cores/node), `mpi`,
  `bigmem` (4 TB), `priority`, `devel`. Large-scale embarrassingly parallel solver work
  goes to `day`/`week`/`mpi`.
- GPU partitions: `gpu_b200` (8x B200), `gpu_h200` (8x H200), `gpu_h100` (4x H100),
  `gpu` (l40s/a40/a5000), `gpu_rtx6000`, plus `gpu_devel`.
- Python 3.9.21 system; conda 26.9.4 via `module load miniconda/26.9.4`. Lmod must be
  initialised explicitly in non-interactive shells (`source /usr/share/lmod/lmod/init/bash`),
  which is why `scripts/env.sh` exists.
- Modules present and relevant: `xtb/6.7.1`, `ORCA/6.1.1`, `GROMACS/2026.4`,
  `OpenMM/8.4.0-CUDA-12.8.0`, `AmberTools/23.6`, `PyTorch/2.9.1-CUDA-12.8.0`,
  `CUDA/12.0-12.9`. APBS and PDB2PQR are **not** modules, so they come from conda-forge.
- Network: github.com, zenodo.org, ftp.ncbi.nlm.nih.gov, hgdownload.soe.ucsc.edu,
  ebi.ac.uk, ftp.ensembl.org, encodeproject.org and conda all reachable.

### Storage decision

| Fileset | Free | Backed up | Purge | Use |
| --- | --- | --- | --- | --- |
| `roberts:home` (125 GiB quota) | 64 GiB | yes | no | code, docs, results, figures, paper |
| `roberts:project` (4096 GiB group) | 382 GiB | yes | no | conda envs, final ground-truth tables |
| `roberts:scratch` (10240 GiB group) | 7.4 TiB | no | 30 days | genomes, raw solver output, work dirs |

Project code lives in `/home/bjc75/other/biophyschem` (home, backed up). Bulk data lives
in `/home/bjc75/scratch/pangea`, symlinked as `data/scratch`. Conda environments go to
`/home/bjc75/project/conda_envs` rather than `~/.conda/envs`, because home has only
~190k free inodes and conda environments hold 100k+ files each.

**Risk logged:** scratch is purged after 30 days without access and is not backed up.
Any ground-truth table that is expensive to regenerate gets copied to
`roberts:project` as soon as it is final.

## 2026-11-19 — Plan of record

The user supplied a complete research plan (PANGEA/READOUT) and instructed that it be
recorded verbatim and followed without cutting corners. It is stored as `PLAN.md`,
word for word. The only change made in transcription was re-rendering the tables, which
arrived as flattened cell-per-line text, into markdown tables. No wording was altered.

The autonomous-researcher-2 novelty loop (phases 1 to 4) is therefore **not run**: the
proposal, the prior-art analysis and the claim structure are given by the user. The
project enters at the equivalent of phase 6 (build), starting with data, as instructed.
`PLAN.md` carries its own prior-work table and reference list, which stand in for
PROPOSAL.md and the related-work backbone.

## 2026-11-19 — cgNA+ implementation choice

The plan requires sequence-dependent ground-state conformations before any solver runs,
and names cgNA+ as the source. Surveyed the available implementations:

- `eskoruppa/cgNAplusParams` — **chosen**. Python, stdlib+numpy/scipy/numba, exposes
  `watson_base_poses`, `crick_base_poses`, `watson_phosphate_poses`,
  `crick_phosphate_poses` as SE(3) frames, ships the cgDNA+ Curves BSTJ 10mus parameter
  set, and carries the Tsukuba and Curves+ ideal base geometries plus the phosphate
  group geometry and the Kabsch frame-fitting conventions needed to invert the fit and
  *place* atoms. Dependencies SO3 and pyConDec are git submodules and were initialised.
- `rahul2512/cgDNA_py` — cgDNA (no phosphates). Rejected: phosphates carry essentially
  all of the charge, so a base-only model cannot drive a Poisson-Boltzmann calculation.
- `rahul2512/cgNAplus_Sugar` — sugar/backbone reconstruction work from the cgNA+ group.
  Kept as a reference for the sugar-building step, not as the primary dependency.

Clones live in `/home/bjc75/scratch/pangea/ext/` and are treated as untrusted downloads:
they are never executed from their own directory, and our code imports them by path.

## 2026-11-19 — Species panel resolved

`src/data/fetch_genomes.py` resolves the fifteen-species panel through the NCBI Datasets
API v2alpha, restricted to RefSeq, preferring the most complete assembly. All fifteen
resolved. Reported GC spans **19.5 percent (Plasmodium falciparum GCF_000002765.6) to
72.0 percent (Streptomyces coelicolor GCF_000203835.1)**, which matches the span the plan
claims and confirms the panel does bracket the stated range. Every download is checked
against the assembly's own md5 and recorded in `genomes_manifest.json` with accession,
URL, md5, byte count, reported GC percent and genome length.

## 2026-11-19 — Three correctness bugs found and fixed in the structure builder

All three were caught by unit tests, and all three would have produced plausible-looking
numbers rather than crashes. They are recorded because the plan treats exactly this
class of silent error as the main risk.

1. **Crick frame flip applied the wrong way round.** `CGNAPlusConf` requires
   `aligned_strands=True`, in which Crick base frames arrive already rotated by
   diag(1,-1,-1) to point along the duplex with Watson. In the Tsukuba standard
   reference frame the Watson frame and the flipped Crick frame of an ideal
   Watson-Crick pair nearly *coincide*, which is what makes the intra-base-pair
   coordinates small deviations. Atoms must therefore be placed in the *unflipped*
   frame. Placing them in the aligned frame stacked each Crick base on top of its own
   partner: minimum inter-residue distance 0.066 angstrom, with every bond length still
   perfect. After the fix, C1'-C1' across pairs is 10.685 +/- 0.070 angstrom and
   glycosidic N-N is 9.009 angstrom, both canonical B-DNA, and the closest
   inter-residue contact is the 1.61 angstrom O3'-P bond.
   Guarded by `test_watson_crick_pair_geometry` and `test_no_atomic_clashes`.

2. **PDB writer omitted the altLoc column**, shifting resName, chainID and resSeq one
   column left. Every downstream parser that reads by column, including PDB2PQR, would
   have mis-assigned residues. Caught because frames could not be re-fitted from our own
   output. Guarded by `test_pdb_round_trip_frames`.

3. **GC estimator read a prefix rather than sampling.** Capping at 50 Mb and reading
   contigs in order measured only the start of human chromosome 1 and returned 47.3
   percent GC against a true 40.9 percent. The universality analysis regresses
   per-genome accuracy on composition, so this would have gone straight into the
   headline claim. Now sampled as random 100 kb windows across all contigs in
   proportion to length.

The groove convention was also confirmed rather than assumed: phosphates project onto
the negative x axis of the base-pair frame, so +x is the major groove and the angular
sector index means what the channel definition says it means
(`test_major_groove_is_plus_x`).

Test suite: 20 tests, all passing.

## 2026-11-19 — Conda on NFS, and where the package cache lives

Three environment builds failed for filesystem rather than dependency reasons.

- A `timeout 120` wrapper sent SIGTERM to conda mid-extraction and left a half-written
  `setuptools` in the package cache, after which every later solve failed with
  `CondaVerificationError`. Lesson: never wrap conda in `timeout`.
- Deleting that cache with `rm -rf` on the `roberts:project` NFS mount hung in
  uninterruptible I/O for 14 minutes and blocked the rest of the build script, so a
  later heredoc rewrite of the script never ran and a stale version was executed.
- Fix: `CONDA_PKGS_DIRS` now points at `/tmp/bjc75_conda_pkgs` on the node's local SSD
  (1.6 TB free), where extracting hundreds of thousands of small files is fast.
  Environments still live on `roberts:project` so compute nodes can see them. The old
  NFS cache is being removed by a detached job.

## 2026-11-19 — Ionic conditions fixed at nine

`PLAN.md` is internally inconsistent on this: the Family A channel description and the
compute budget both say nine ionic conditions (and 200,000 fragments x 9 = 1.8 million
solves), while the dataset table says "150k fragments, 7 conditions". Nine is used,
since two of the three statements agree and the channel count depends on it. The set is
seven NaCl concentrations from 10 mM to 500 mM, one KCl point at 150 mM for the cation
identity axis, and one Na/Mg mixture. The magnesium condition is flagged `approximate`
in code and is excluded from any claim needing quantitative divalent accuracy, as the
plan requires.

## 2026-11-19 — Corrections to PLAN.md found while verifying data sources

Dataset locations were verified by opening them, not by trusting the plan or memory.
Five statements in `PLAN.md` turned out to be wrong. They are recorded here, and
`configs/experimental_datasets.json` carries each correction in a `plan_correction`
field next to the affected dataset. The plan text itself is left untouched, because the
user asked for it verbatim.

1. **ORChID does not provide "all 512 pentamers".** ORChID is a sliding *tetramer*
   model: a measured 256-tetramer table plus a 64-trimer table, with pentamer
   predictions derived at run time. There are in any case 1024 pentamers, not 512. The
   512-entry pentamer table belongs to DNAshape (minor groove width), a different
   resource. This matters for the plan's prior-work table, which contrasts ORChID as a
   "pentamer lookup" against our approach, and for the Family G channel design.
   Also `orchid.bu.edu` is dead; the live host is `dna.bu.edu/orchid`. The ~30 Mb
   genome-scale map is UCSC hg17 track `encodeBu_ORChID1`, requiring the index table
   plus the binary `.wib`; decoded size is 29,948,050 single-nucleotide values, which
   matches the paper's figure.

2. **loop-seq has no GEO series.** The sequence-plus-cyclizability tables exist only as
   Nature Supplementary Datasets; SRA PRJNA667271 holds reads alone. The usable target
   column is `C0` (intrinsic cyclizability); the deposited `Sequence` field is the full
   100 bp construct, so the 50 bp variable region is characters 26 to 75.

3. **Array Melt's second author is Eesha Sharma**, not D. Sharma. Separately, the
   published `arr_v1_1M_n=27732.csv` has a real bug: its `dS` column stores dH/Tm with
   Tm in Celsius rather than Kelvin, on every row. We use the adjusted and raw v1 files,
   which are internally consistent (dG_37 = dH - 310.15 dS and Tm = dH/dS - 273.15 both
   hold exactly).

4. **dCPD-seq is Genome Research 2026**, not 2025. GEO release was 2025-10-01, which
   explains the plan's date.

5. **Zhang 2011 has no GEO accession**, so it cannot serve as an in vitro nucleosome
   source. Only Kaplan 2009 (GSE13622) and GSE15188 assemble nucleosomes on purified
   bulk genomic DNA; every later in vitro map uses SGD-plasmid libraries and carries
   library coverage bias. Since in vitro nucleosome occupancy is the plan's *positive
   control*, a library-biased substrate would corrupt the control, so Kaplan 2009 is
   used.

### Datasets that do not ship sequences

G4-seq, in vitro nucleosome occupancy, CPD-seq and dCPD-seq distribute coordinates and
counts only, so each needs a genome lift to become a supervised channel. G4-seq is hg19
for both studies (confirmed from maximum chr1 coordinate, not assumed); Kaplan 2009 is
on a May 2006 SGD build between sacCer1 and sacCer2; CPD-seq is sacCer3. The lifts and
their builds are tracked per dataset in the manifest.

### Licensing status, recorded rather than assumed

Open: Array Melt data (CC BY 4.0), DeepSTARR (CC BY 4.0), yeast promoters (CC BY 4.0),
lentiMPRA via human_legnet (MIT), dCPD-seq (CC BY 4.0). No declared license, citation
only: Jores plant MPRA, Sharpr-MPRA, ORChID, loop-seq, Kaplan 2009, CPD-seq. The splice
site benchmark's parent repository is CC BY-NC-SA 4.0, so it is treated as
non-commercial. All are used here for non-commercial academic research and cited; none
is redistributed.

### One verification still outstanding

S1-END-seq and KAS-seq (Family F) have not been verified yet and no accession is
recorded for them. Nothing will be downloaded for those channels until a URL is checked.

## 2026-11-19 — Solver configuration chosen by measurement

`results/solver_probe_v1.json`. Swept solute length, fine grid spacing, coarse padding
and equation on one sequence and one conformation, measuring wall time and channel-level
agreement together.

| solute bp | fine h (A) | transverse pad (A) | grid points | time | channel rmse / sd |
| --- | --- | --- | --- | --- | --- |
| 41 | 0.50 | 30 | 1.82M | 89.0 s | 0.305 |
| 41 | 0.50 | 50 | 1.82M | 9.4 s | 0.305 |
| 41 | 0.35 | 30 | 6.66M | 36.6 s | reference |
| 41 | 0.35 | 50 | 6.66M | 36.5 s | 0.0005 |
| 61 | 0.50 | 30 | 1.82M | 6.5 s | **aborted, overflow** |
| 61 | 0.50 | 50 | 1.82M | 9.8 s | 0.309 |
| 61 | 0.35 | 30 | 6.66M | 21.9 s | **aborted, overflow** |
| 61 | 0.35 | 50 | 6.66M | 34.9 s | 0.062 |

Conclusions, and the production configuration:

- **Coarse padding was the defect, not cost.** At 30 A transverse padding the 61 bp
  solve aborts at both spacings. Where it does converge (41 bp) it is nine times slower.
  50 A converges reliably and changes the sampled channels by 0.0005 of a channel
  standard deviation, so the padding is free in accuracy terms.
- **0.35 A is required.** Coarsening to 0.5 A costs 0.305 of a channel standard
  deviation, which is far too much for a target the encoder is meant to reproduce.
- **Cost is within the plan's budget.** 34.9 s per solve against the 30 to 60 s the
  plan's 15,000 to 30,000 CPU-hours for 1.8 million solves implies. The earlier
  impression that this was infeasible came from my own bug: `dime` was computed in a
  fixed axis order while the duplex's long axis is z, so the short axes were
  over-resolved and the long axis under-resolved.

Production: solute 61 bp (the full fragment, as the plan specifies), fine spacing
0.35 A, transverse padding 50 A, axial padding 40 A, nonlinear PB. Constants live in
`src/physics/run_fragment.py` so the configuration is in one place.

## 2026-11-19 — Deviation: surface-referenced sampling added alongside the plan's shells

`PLAN.md` defines Family A as "potential on cylindrical shells at 6 angular sectors and
6 radial offsets". Implemented literally, with radii measured from the helical axis,
this has a structural problem that only shows up once the numbers are inspected.

The sugar-phosphate backbone sits near 9 A from the axis, which is one of the sampling
radii. At the two backbone sectors the sample point lands inside a phosphate group,
where the potential is the near-singular field of a buried charge in a dielectric of 2.
Before masking, those channels read **-161 and -159 kT/e** at r = 9 A and went
**positive** at r = 5 A, while the solvent channels sat between -1 and -4 kT/e. Those
values describe the grid, not the molecule, and their variance would have been dominated
by whether a sample happened to fall inside an atom.

Two changes:

1. **Buried points are masked.** `solvent_accessible_mask` sets any sample that is not
   solvent accessible (within an atom's radius plus a 1.4 A probe) to NaN, and the
   boolean mask is stored with the channels. It is sequence dependent, so downstream
   code must carry it rather than average over it. With masking, only **59.1 percent**
   of the plan's 6 by 6 grid is defined, and whole channels (both backbone sectors at
   r <= 11 A, and the minor groove at r = 5 A) are empty for every sequence.
2. **Surface-referenced sampling added, and made primary.** For each base pair and
   sector, march outward to the first radius from which the ray stays in solvent, then
   sample at offsets 0, 1, 2, 4, 7 and 11 A beyond that surface. This is **100 percent
   defined**, and it means the same thing for every sequence: the potential this far
   outside the molecular surface, in this groove. The surface radius itself is kept as a
   channel, because it is the local groove depth along that direction.

A channel that is undefined for every sequence cannot be predicted, cannot be compared
across species, and cannot carry a universality claim, which is why this is treated as a
correction to the channel definition rather than a preference. The axis-referenced
shells are still computed and stored, so the plan's literal definition remains available.

Validation of the surface-referenced channels on one 61 bp fragment at 150 mM NaCl:

- **Surface radius reproduces the B-DNA cross-section.** 4.83 A at the major-groove
  sector, 5.83 A at the minor-groove sector, 9.33 and 9.35 A at the two intermediate
  sectors, 12.64 and 12.58 A at the two backbone sectors. The pseudo-twofold symmetry of
  the duplex shows up as the agreement between each pair, which is a check on the frame
  construction that no single-sector number would give.
- **Decay matches Debye-Huckel screening.** Mean potential falls from -3.37 kT/e at the
  surface to -0.779 kT/e at 11 A beyond it, a factor of 4.32. The Debye length at
  150 mM is 7.8 A, predicting exp(-11/7.8) = 0.243, a factor of 4.12. Agreement to
  within 5 percent, and nothing in the pipeline was tuned to produce it.
- **The minor groove is the most negative region**, -4.82 kT/e against -3.54 kT/e on the
  major-groove side, which is the long-established result that motivates using minor
  groove potential as a recognition feature at all.
- Sequence dependence is present and larger than in the axis-referenced version:
  per-base-pair standard deviation 0.285 kT/e at the surface against 0.173 kT/e at
  r = 7 A.

## 2026-11-19 — The raw APBS energy is not usable as a channel

`print elecEnergy` returns about -2.2e13 kJ/mol for a 61 bp duplex, which is dominated
by grid self-energy and carries no information about the sequence. The plan already
anticipates this ("Report relative to a fixed reference sequence so the uninformative
global offset does not dominate"). The raw value is stored for provenance but is not a
channel; the reference-sequence subtraction has not been implemented yet, so no energy
channel is currently produced.

## 2026-11-19 — Yeast coordinates need no lift, and the positive control is weaker than assumed

`results/yeast_coord_calibration.json`.

The in vitro nucleosome map was built on a May 2006 SGD assembly and ships no sequence,
so the offset against sacCer3 cannot be found by alignment. It was found by exploiting
the known GC-occupancy relationship: a trial offset is applied to the assay coordinates,
sacCer3 sequence is extracted, and the GC-occupancy correlation is measured. The curve
peaks at the correct alignment.

Result: the curve is smooth and single-peaked, and the best offsets are **0, +10, +5 and
0 bp** on chromosomes I, VI, III and IX. The penalty for simply using offset 0 is
negligible: median correlation **0.7800 at offset 0 against 0.7814 at the per-chromosome
optimum**, a difference of 0.0014. **Decision: use sacCer3 coordinates directly, no
lift.** The accumulated build differences are real but smaller than the 147 bp
nucleosome footprint, so they do not matter at this resolution.

**The finding that matters more.** G+C content in a 147 bp window correlates with in
vitro nucleosome occupancy at **r = 0.78 (median across four chromosomes), so R-squared
is about 0.61**. The plan states that "GC content alone explain[s] close to half the
variance" and uses in vitro nucleosome occupancy as the **positive control**, on the
reasoning that a physics atlas should nearly saturate it. Measured here, a single scalar
feature already reaches 61 percent of the variance.

Two consequences, both of which should be settled before the control is relied on:

1. **The positive control is close to uninformative as specified.** "The physics head
   comes close to the sequence model on in vitro nucleosome occupancy" would be
   satisfied by any model that has access to local GC, which every channel in the atlas
   does. Passing it would not show the landscape is sound. The control needs to be
   stated against the GC baseline rather than against the sequence model: the question
   worth asking is how much of the remaining 39 percent the physics reaches.
2. **The GC-dominance trap is more acute than the plan's framing suggests.** The plan
   already requires "GC plus 5-mer ridge at matched capacity as the primary gate on every
   dataset", which is the right control. This measurement says that gate will be the
   binding constraint on the positive control task, not a formality.

Caveat on comparability: this correlates 147 bp windowed GC against per-base-pair log2
occupancy at 20,000 sampled positions per chromosome. Published values depend on the
aggregation used (per nucleosome, per window, or genome-wide at coarser resolution), so
0.78 is not directly comparable to a specific published figure. What is not in doubt is
the implication for the control, since the comparison here is made the same way for both.

## 2026-11-19 — S1-END-seq and KAS-seq located, and all three human assays are hg19

The delegated search did not return these, so they were verified directly rather than
left as a gap.

- **S1-END-seq: GSE203632.** Confirmed by reading the GEO file list, which names
  `S1_END_seq_*` files explicitly, rather than by matching the series title (the title
  says "END-seq", which is why a title-based search missed it). Strand-separated bigWig
  per sample across five human lines. The bigWig magic was verified by range request as
  0x888FFC26, version 4.
- **KAS-seq: GSE139420.** Ships both bigWig signal and q < 0.01 BED peak calls; the BED
  peaks are the directly usable form.

**Genome build determined from the data, not assumed.** Parsing the S1-END-seq bigWig
chromosome B-tree gives chr1 = 249,250,621, chr2 = 243,199,373, chrX = 155,270,560 over
25 contigs, which is hg19 exactly. The KAS-seq peak set reaches chr1:249,240,722, which
fits hg19 (249,250,621) and overflows hg38 (248,956,422), so it is hg19 as well.

Together with G4-seq, which was independently confirmed as hg19, **all three human
coordinate-only assays are on hg19**. One genome download serves all of them, and no
liftOver is needed among them. This closes the last outstanding data gap: all eight
experimental datasets and all five activity datasets are now downloaded and verified.

## 2026-11-19 — Correction: the low-salt failures were my bug, not a Debye-length effect

`results/salt_padding_probe.json` refutes the explanation recorded earlier today, and the
earlier entry should be read with this one.

**What I claimed.** The jitter and gate jobs aborted at 25 mM, and I attributed it to the
Debye screening length: 50 A is 6.4 screening lengths at 150 mM but only 2.6 at 25 mM, so
the coarse-grid boundary condition was too close for low salt.

**What the data says.** Sweeping padding at 25 and 10 mM, **all twelve configurations
converged**, including the supposedly failing case of 50 A at 25 mM (37.5 s), and
including 10 mM at 79 A (41.8 s). Both boundary conditions, `sdh` and `mdh`, worked
throughout; `mdh` was consistently 15 to 35 percent slower with no convergence benefit.

**What actually happened.** The failing jobs did not use 50 A. The code at submission
computed `coarse = max(extent * 1.8, extent + 60)`, which for a 61 bp duplex of extent
(36.0, 34.8, 200.8) A gives a coarse box of (96.0, 94.8, 361.4) and therefore **30 A of
transverse padding**, not the 50 A I believed I had set. `solver_probe_v1.json` had
already recorded that 61 bp at 30 A aborts at both grid spacings. So the first probe had
identified the cause correctly, and I then reintroduced the same defect by writing a
formula that did not deliver the padding I intended, and misread the resulting failure as
new physics.

**Lesson applied.** The padding is now a named constant with the measured value, not a
formula whose output has to be inferred, and the per-condition padding actually used is
written into every fragment's record so a future failure can be diagnosed from the output
rather than reconstructed from the source.

**Production setting.** `N_DEBYE_PAD = 4.0` with a 50 A floor. This is not Debye physics;
it is bookkeeping that keeps every production condition at or above a padding that was
actually tested: 122 A at 10 mM and 77 A at 25 mM, both verified here, falling to the
50 A floor from 100 mM upward, where 50 A is verified. Padding beyond the convergence
threshold does not change the channels: `solver_probe_v1` measured 30 A against 50 A at
41 bp at 0.0005 of a channel standard deviation.

**Consequence for the plan.** None of the nine ionic conditions is blocked. The earlier
worry that low-salt conditions might be infeasible or need a different solver setup was
unfounded.

## 2026-11-19 — Second correction: the failing condition is 500 mM, not low salt

A clean four-condition run on an uncontended node settles it
(`/home/bjc75/scratch/pangea/apbs/smoke/lowsalt01.json`):

| Condition | Debye length | padding used | outcome |
| --- | --- | --- | --- |
| NaCl 10 mM | 30.4 A | 122 A | **converged** |
| NaCl 25 mM | 19.2 A | 77 A | **converged** |
| NaCl 150 mM | 7.9 A | 50 A | **converged** |
| NaCl 500 mM | 4.3 A | 50 A | **aborted**, exponential overflow |

So the direction of my first hypothesis was exactly backwards. Low salt is fine. **High
salt is the failing regime**, and at 500 mM the 50 A pad is 11.6 screening lengths, so
insufficient padding cannot be the explanation.

This is the second time in this session that I extrapolated a configuration validated at
150 mM to an untested regime and treated it as validated. Each of the nine ionic
conditions is now treated as unvalidated until it has actually completed a run, and the
padding and Debye length actually used are written into every fragment record so the
regime is visible in the output rather than inferred from source.

A probe over padding, grid spacing and equation at 250 and 500 mM is running to
establish the cause rather than guess at it. Until it resolves, **500 mM is excluded
from production** and the gate runs at 150 mM, which is verified end to end.

## 2026-11-19 — Node contention diagnosed, not misread as solver slowness

A 10 mM solve appeared to take 22 minutes against 54 s in the probe with identical grid
dimensions. The process was in state R at **4.8 percent CPU**, which over 22 minutes is
about 63 s of actual compute, matching the expected cost. The cause was my own hexamer
quantum chemistry job running 24 workers on the same 48-core node. Pausing its driver
with SIGSTOP (in-flight calculations finish, no new ones spawn, nothing is lost) took
the solver to 95.5 percent CPU immediately.

Worth recording because the symptom, a solve that is thirty times slower than measured,
looks exactly like a numerical problem and would have justified changing solver settings
that were in fact correct. The check that distinguished them was CPU time against wall
time, not any property of the solver.

## 2026-11-19 — Bug in the composition gate's channel filter

`channel_matrix` dropped any channel with fewer than 10 finite values. With fewer than
10 fragments that discards **every** channel, and the function returned an empty matrix
rather than an error, so the gate would have reported "0 channels" instead of failing.
The threshold is now a fraction of the fragments present (at least half, minimum three)
and an empty result raises.

Found by exercising the gate on the 8 jitter replicates rather than waiting for the real
run. That exercise also validated the analysis path on real solver output: 68 channels
retained at 150 mM (36 surface-referenced, all defined; 25 of 36 axis-referenced,
consistent with the 59 percent accessibility; 6 surface radii; 1 counterion), and on
identical sequences the variance decomposition returns 3.3e-05 between against 6.0e-05
within, which is pure noise on a null input.

## 2026-11-19 — Correction to the environment record: this session has 1 CPU and 8 GB

The environment entry at the top of this file lists the agent-partition node as "48
cores, 368 GB RAM". That is the **node** specification read from `sinfo`. My actual
allocation is:

    AllocTRES=cpu=1,mem=8G,node=1

`nproc` inside the allocation returns **1**, and it did so when I ran it; I read the
output as stray and did not follow it up. I never ran `scontrol show job` on my own job.

Consequences, all of them mine rather than the software's:

- The composition-gate workers were killed by the kernel, not by a numerical problem.
  The log is explicit: `oom-kill: constraint=CONSTRAINT_MEMCG,
  oom_memcg=/system.slice/slurmstepd.scope/job_33891243, task=apbs`, with
  `anon-rss:1522360kB`. Eight concurrent 1.5 GB solvers against an 8 GB cgroup.
  `apbs failed (-9)` is SIGKILL, which I initially read as another solver fault.
- The 10 mM solve that appeared to take 22 minutes at 4.8 percent CPU was not contention
  between my jobs for a large node. There is one core, and I had 24 quantum chemistry
  workers plus a solver on it.
- Moving the hexamer quantum chemistry off the cluster "because the node is mine for
  seven days" oversubscribed a single core 24-fold. That job was eventually killed by
  the OOM killer (exit 137) and its work is lost.
- The 24 xtb processes still visible after I paused their driver were zombies: a stopped
  parent cannot reap children.

**Rule adopted.** All heavy work goes through `sbatch` with explicit `--cpus-per-task`
and `--mem`. `scripts/slurm_apbs_array.sh` already requests 4 CPUs and 16 GB, which is
correct; the right action an hour ago was to wait for the queue rather than relocate the
work onto an allocation I had not checked. `scripts/run_gate_local.sh` is kept only
because it is useful inside a properly sized salloc, and now carries a warning.

No result in RESULTS.md is affected: every number there came from a run that completed
and wrote output. What was affected is throughput, several scheduling decisions, and the
diagnoses I built on top of them.

## 2026-11-19 — 500 mM: the nonlinear solver, not the box

`results/highsalt_probe.json`, read with the caveat that it ran while the OOM-thrashing
gate workers were alive, so an individual FAIL could be SIGKILL rather than overflow.
The load-bearing evidence is independent of that: **500 mM npbe also failed on clean,
uncontended cluster nodes** in the jitter array, and in the single-process `lowsalt01`
run where 10, 25 and 150 mM all converged.

| Condition | npbe, padding 20 / 30 / 50 / 100 A | lpbe, padding 50 A |
| --- | --- | --- |
| 500 mM | FAIL / FAIL / FAIL / FAIL | **converged, 23.7 s** |
| 250 mM | FAIL / FAIL / FAIL / converged | not tested |

Padding is ruled out: 500 mM fails across a fivefold range of box size, including boxes
far larger than the 11.6 screening lengths that 50 A already represents. Grid spacing is
ruled out: 0.50 A fails too. The linearised equation converges at the same box and
spacing where the nonlinear one does not, which localises the problem to the nonlinear
Boltzmann term. With a -120 e solute and a 4.3 A screening length that term is stiff,
and the multigrid iterate diverges, which is what the trapped exponential overflows
report.

**Not yet decided.** The physically principled option is to use the linearised equation
at high salt, because linearisation error grows with the potential magnitude and the
potential is smallest exactly where screening is strongest. That argument has to be
measured, not asserted: the test is to compare lpbe against npbe channels at 150 mM,
where both converge, and show the discrepancy is small and shrinking with salt. Until
that measurement exists, **500 mM and 250 mM are excluded from production** and the
plan's nine-condition set is reduced to the seven that converge with npbe. This is a
deviation from `PLAN.md` and will be reported as one.

## 2026-11-19 — lpbe substitution REJECTED on evidence; 500 mM dropped

`results/lpbe_vs_npbe.json`, job 33899816, 4 sequences per condition, clean allocation.

Discrepancy between the linearised and nonlinear solutions, as a fraction of the
channel's own standard deviation:

| Condition | lpbe vs npbe, rmse / sd | npbe convergent sequences |
| --- | --- | --- |
| 10 mM | 4.32 | 4 of 4 |
| 50 mM | 2.11 | 4 of 4 |
| 100 mM | 1.41 | 4 of 4 |
| 150 mM | 1.08 | 4 of 4 |
| 250 mM | 0.76 | 3 of 4 |
| 500 mM | no convergent pair | 0 of 4 |

The physical argument was right about the **direction** and wrong about the
**magnitude**. The discrepancy does fall monotonically with ionic strength, exactly as
screening predicts. But even at 250 mM it is 0.76 of a channel standard deviation, and
extrapolating the trend to 500 mM leaves roughly half a standard deviation.

For scale, within this same project: the coarse-padding choice was accepted at
rmse/sd = 0.0005, and a 0.5 A grid spacing was **rejected** at rmse/sd = 0.305. A
substitution costing 0.5 to 0.76 is therefore two to three times worse than an
approximation already rejected as far too coarse, and a thousand times worse than the
padding tolerance. Using lpbe at high salt would put a larger error into the training
target than any other approximation in the pipeline.

**Decision: lpbe is not substituted anywhere.** This is the outcome the check was
designed to permit, and the plausible-sounding argument is discarded rather than used.

**Consequence for the condition axis.** 500 mM is dropped from production. The plan's
nine conditions become eight, and 250 mM is kept but flagged: it converged for 3 of 4
sequences, so it is sequence-dependent and per-fragment convergence is recorded rather
than assumed. Reported as a deviation from `PLAN.md`.

**Conditions now actually validated under npbe**, each by a completed run:
10, 25, 50, 100 and 150 mM converge for every sequence tried; 250 mM for 3 of 4;
500 mM for none. This run supplied the first real evidence for 50 and 100 mM, which the
status table had correctly listed as untested.

---

# PLAN.md revision 2 (received 2026-11-19)

The user supplied a revised plan. It is recorded verbatim in `PLAN.md`; revision 1 is
archived at `PLAN_v1_archived.md`. A whitespace-normalised diff shows the revision is
confined to exactly two areas, and nothing else in the document changed.

## What revision 2 adds

**1. Multiresolution outputs (new, six paragraphs in Aim 1).** The landscape now has an
explicit resolution axis, making it five axes rather than four:

- per base pair: geometry, stiffness, shell potentials, site potentials, stacking,
  reactivity, hydration
- a window ladder at 10, 20, 50, 100, 200 and 500 bp: everything defined over an extent
  (persistence length, cyclizability, nucleosome free energy, strand separation,
  stress-induced destabilisation, RNA folding free energy)
- per sequence: total electrostatic solvation free energy, net charge, duplex melting
  temperature, overall fold free energy

Architecturally a pyramid, with **exact inter-scale relations imposed as consistency
losses** (integrated charge over a window = sum of per-base contributions) and emergent
ones (persistence length vs local stiffness) used only as diagnostics.

This is not cosmetic for the ground-truth stage. It says explicitly that fragment-level
labels become sliding-window tracks at inference, and that the extrapolation is valid
only where the quantity is local relative to its window -- "which is exactly what the
convergence study in the ground truth section measures, so the two decisions are made
together rather than separately." The convergence study is therefore load-bearing for
the architecture, not only for the solver budget.

**2. Electrostatics made tractable (new, five paragraphs).** Revision 1 said "Take a
61 bp fragment." Revision 2 replaces that with four mechanisms:

- **Condition-dependent fragment length.** Flank set by the Debye length: about 36 bp at
  10 mM, 16 at 50, 11 at 100, 9 at 150, 7 at 300, for ~2% residual contribution. With a
  21 bp scored window that is 93 / 53 / 43 / 39 / 35 bp solutes.
- **Condition anchoring with delta-learning.** Solve the full set at two bracketing
  anchor conditions, 10-20% at the rest, learn the condition residual. Stated as
  ~8 solves per fragment down to ~2.9, a factor near 2.8.
- **A mandated convergence study.** ~2,000 fragments at 121, 61, 41, 31 and 21 bp across
  every ionic condition, because linearised theory is only a guide in the nonlinear
  counterion-condensation regime. The plan estimates "tens of CPU-hours".
- **A separate long-fragment set** for the window-level electrostatic channels
  (electrostatic persistence length, charge integrated over 10-100 bp windows), which are
  not reconstructable from tiled local potentials.

It also adds an explicit warning against a shortcut I had not proposed but could have:
training a surrogate on very short fragments produces high correlation against targets
that are themselves unconverged, "which is fidelity to the wrong number and is invisible
to every validation split."

## Correction: my per-solve cost estimate was roughly 10x too high

Reported to the user as ~440 CPU-s per solve and ~199,000 CPU-h for library v2. That
figure took the ~110 s per-condition **pipeline** time (cgNA+ build, minimisation,
pdb2pqr, APBS, sampling) and multiplied it by 4 threads, treating the whole pipeline as
if it were 4-way-parallel solver time. It is not.

`results/solver_probe_v1.json` already contained the direct measurement, and I did not
use it. APBS alone, 61 bp solute, 0.35 A fine grid, 50 A transverse pad, NaCl_150mM:
**34.9 seconds**. At 41 bp: **36.6 s**.

The same file refutes a second belief. **Cost is nearly flat in transverse padding**
(36.6 s at 30 A vs 36.5 s at 50 A for the 41 bp solute) because `dime` is held fixed, so
padding changes the coarse *spacing*, not the grid point count. The cost driver is grid
points: 1.82M at 0.5 A fine vs 6.66M at 0.35 A fine. Padding was never the expensive
choice; I had assumed it was.

And the 30 A aborts were **solute-size specific**, not universal: at 61 bp, 30 A padding
returns -6 at both spacings, while at 41 bp it converges at both. Since revision 2 puts
most production conditions at 35-53 bp solutes, the padding floor needs re-deriving at
the new lengths rather than inherited from the 61 bp result.

**No new budget is being asserted from this.** I have twice extrapolated a validated
configuration into an untested regime and presented it as validated. The numbers above
are what was measured at one condition on one or two sequences. The replacement estimate
comes from `solver_cost_v2`, which measures the production configurations directly.

## Open inconsistencies in revision 2, recorded not silently resolved

These are internal to the supplied document. None is edited in `PLAN.md`.

1. **"Group G" vs Family I for protein readout.** The families table and the body both
   place protein readout at **Family I** ("Family I is held apart deliberately... Every
   headline number is reported for families A through H alone"). But the traps table says
   "Group G excluded from every headline number", and the dataset table says "Group G
   protein readout, held apart". Family G is Solvation, hydration and chemical
   reactivity. Read as **Family I**, a leftover label from an earlier draft. Excluding
   Family G instead would drop the hydroxyl-radical and photochemistry channels from
   every headline number, which contradicts their stated role as external validation.

2. **Fragment and condition counts disagree across three places.** The compute budget
   says "200,000 fragments across 9 ionic conditions... about 1.8 million solves". The
   dataset table says "150k fragments, 7 conditions". Family A says "9 ionic conditions".
   Our measured position is **eight** conditions, 500 mM having been dropped on evidence
   (see the lpbe section above), on a library of 191,344 fragments.

3. **"Stage one / stage two" is overloaded.** The Summary calls PANGEA stage one and
   READOUT stage two. Aim 1 separately calls pretraining and delta-learning the two
   training stages. These are different partitions of the work and the terms collide.
   This file uses **PANGEA-pretrain** and **PANGEA-delta** for the training stages, and
   **PANGEA** / **READOUT** for the two project stages.

4. **"the stronger of the two"** in Deliverables describes a set of three papers.

## Consequences for work already done

- `MIN_TRANSVERSE_PAD_A = 50.0` and the fixed 61 bp fragment are both superseded. Length
  becomes a function of condition; the padding floor is re-measured per length.
- `N_DEBYE_PAD = 4.0` was a tested-envelope floor for solvent padding. Revision 2 uses
  the Debye length for a different purpose -- **DNA flank length** -- so the two must not
  be conflated. They are now separate constants.
- Library v2 fragments are 61 bp. That serves the 10 mM end badly (needs ~93 bp) and
  wastes solver time at 150 mM and above (needs ~39 bp). Fragment generation has to
  become condition-aware, which the convergence study settles before it is rebuilt.
- The nine `IonicCondition` entries in `apbs_run.py` stay as definitions; eight are
  production.

---

# Measurement: the electrostatics budget was wrong by a factor of about 20

`results/solver_cost_v2.json`, job 33904468 on `scavenge`. Both beliefs the probe was
written to test turned out false, and both had been load-bearing.

## APBS does not use the threads it is given

| threads | wall seconds | grid points |
| --- | --- | --- |
| 1 | 23.0 | 6,661,697 |
| 2 | 23.0 | 6,661,697 |
| 4 | 21.5 | 6,661,697 |
| 8 | 21.3 | 6,661,697 |

Eight threads buy 7 percent. The solve is serial for practical purposes.

Every CPU-hour figure quoted before this took a wall time and multiplied it by the
allocation's thread count. That multiplication was invented. Combined with using the
~110 s **pipeline** time (cgNA+ build, minimisation, pdb2pqr, APBS, sampling) in place
of the ~22 s **solver** time, the published figure of 440 CPU-s per solve was about 20
times the measured 22 CPU-s.

## Solute length does not change solver cost

| solute | wall seconds | grid points | coarse spacing |
| --- | --- | --- | --- |
| 21 bp | 22.7 | 6,661,697 | 0.82 / 0.83 / 0.58 A |
| 31 bp | 22.5 | 6,661,697 | 0.85 / 0.87 / 0.71 A |
| 41 bp | 20.2 | 6,661,697 | 0.90 / 0.90 / 0.83 A |
| 61 bp | 21.1 | 6,661,697 | 0.96 / 0.95 / 1.08 A |
| 93 bp | 16.1 | 6,661,697 | 1.07 / 1.00 / 1.48 A -> **failed** |

Grid points are identical at every length, because `write_apbs_input` derives `dime`
from the **fine** box and `mg-auto` uses one `dime` for both levels. Lengthening the
solute only coarsens the coarse grid. Cost is set by the fine box, which the plan fixes
at 0.35 A over the central 21 bp out to an 18 A radial reach.

Confirmed by the spacing sweep: 0.5 A gives 1,815,937 points and 6.2 s. That is 3.67
times fewer points for 3.4 times less time, so cost is near-linear in grid points.

**Consequence for `PLAN.md` revision 2.** The revision calls condition-dependent
fragment length "the lever that makes the solver budget tractable". Measured, it is not
a cost lever at all under this focusing setup. It remains valuable for the reason the
plan gives second -- convergence at low salt -- and it buys a finer coarse grid for
free, which is an accuracy gain. But the budget does not come from it.

The levers that are real: **condition anchoring** (8 solves per fragment down to about
2.9, a factor of 2.76, exactly as the plan states) and **grid points**, which the plan
fixes by specification and should not be traded away -- 0.5 A spacing was already
rejected on accuracy at rmse/sd 0.305.

## The corrected budget

At 22 CPU-seconds per solve, single core, production configuration:

| scope | solves | CPU-hours | at the 1,500-core `day` cap |
| --- | --- | --- | --- |
| 191,344 fragments x 8 conditions | 1.53M | 9,355 | 6.2 h |
| the same, with condition anchoring | 0.55M | 3,391 | 2.3 h |
| 10% ensemble subset x 8 configs, anchored | 0.44M | 2,713 | 1.8 h |
| **total, anchored, with ensemble** | **1.0M** | **6,104** | **4.1 h** |

`PLAN.md` budgets 15,000 to 30,000 CPU-hours. The measured requirement is **under that
budget**, by a factor of two to five.

**The electrostatics family is not the bottleneck, and the claim that it was is
withdrawn.** Every statement made to the user about Family A being infeasible, about
40 percent of the landscape being out of reach, about weeks to months of solver time,
and about needing to shrink the stage-2 label set to make it affordable, was downstream
of a thread-count multiplication that was never measured.

## Open: the 93 bp solve failed

93 bp is the solute length the plan's Debye rule implies at 10 mM, which is exactly the
condition where the long fragment is needed. It failed at the padding the current rule
produces, with the long-axis coarse spacing at 1.48 A. Since cost is flat in solute
length, the fix costs nothing: raise `dime` for the long fragments, or raise the axial
padding. `context_convergence` (job 33904051) covers 121 bp and will show whether this
is a padding problem or a `dime` problem. Not resolved here, and no low-salt production
runs until it is.

---

# Families F, L and D implemented; the channel declaration layer

`src/physics/channels.py` is new and is the foundation revision 2 requires: every
channel declares its **resolution** (per-bp, a window-ladder rung, or global), its
**parity** under reverse complement, its units, its source, and whether it is
**provisional**. 134 tests pass across the suite.

| Family | Module | Channels declared | Provisional | Plan's count |
| --- | --- | --- | --- | --- |
| F non-canonical structure | `noncanonical.py` | 60 | 23 | 53 |
| L RNA secondary structure | `rna_structure.py` | 74 (37 per strand) | 6 | about 60 |
| D mesoscale mechanics | `mesoscale.py` | 52 | 7 | 90 |

Family D is short of the plan's 90 because the per-window bending channels live in
`mesoscale_window_channels` rather than as named scalars; they add 3 per rung across 6
rungs per window position.

## Validation against measurement, not against plausibility

Family D is the only analytic family whose outputs can be checked against textbook
numbers, so it validates the whole cgNA+ chain -- coordinate convention, unit
convention, the marginalisation of the stiffness, and the elastic relations.

| quantity | computed | measured B-DNA | verdict |
| --- | --- | --- | --- |
| bending persistence length | 47.1 nm | ~50 nm | agrees |
| stretch modulus | 1,652 pN | 1,000-1,500 pN | agrees |
| rise per step | 0.332 nm | 0.34 nm | agrees |
| torsional modulus | 39.9 nm | 75-110 nm | **does not agree** |
| twist-stretch coupling | +0.29 | measured sign is negative | **sign unresolved** |

The torsional modulus is carried as a strict `xfail` rather than hidden by a loose
tolerance, plus a second test pinning the computed value so a silent upstream change is
noticed. Whether 40 nm is a real local-versus-effective distinction -- single-molecule
measurements fold bend-twist coupling into an effective C -- or an error in this
reduction, is unresolved and must be before Family D twist channels are trusted.

## The coordinate convention, established by measurement

cgNA+ returns rotations as **unscaled Cayley vectors**, 2 tan(theta/2) along the axis
(`cgnaplusparams/SO3/so3/conversions.py`), and translations in **nanometres**. Checked
against known B-DNA: poly-A returns a twist Cayley coordinate of 0.6028, which is 33.6
degrees -- the reduced A-tract twist -- and a rise of 0.334, which is 3.34 A. Both are
asserted in `tests/test_mesoscale.py` so an upstream convention change cannot pass
silently.

## Finding: the cgNA+ stiffness is not reverse-complement symmetric

Measured on an **exact palindrome**, where RC(seq) == seq and no index mapping is
ambiguous:

- ground-state rotations mirror to machine precision, under the 3DNA parity in which
  tilt changes sign and roll and twist do not;
- marginal step covariances do **not**: up to 46% per step in tilt, 20% in roll, 3.6%
  in twist, averaging to about 0.1% in aggregate quantities such as persistence length.

Ruled out as causes: the stiffness is correctly banded (bandwidth 41, consistent with
nearest-neighbour coupling over the interlaced B/C/S/W layout), and the row extraction
matches the parameter ordering. So this is a property of the cgNA+ parameter set.

**Why it matters, and it is not bookkeeping.** `PLAN.md` enforces reverse-complement
equivariance *architecturally*. A target that is not exactly symmetric is therefore
something the model **cannot** fit: the residual becomes an irreducible error floor
that will look like model error and is not. The physical duplex is symmetric -- it does
not know which strand was written down -- so the asymmetry is ground-truth noise.

**Decision: production targets are symmetrised.** `channels.symmetrise_channels`
averages the two orientations by declared parity. Parity residuals fall to exactly
zero, and the persistence length moves from 47.064 to 47.088 nm, which is within the
noise of the quantity. Cost is one extra cgNA+ evaluation, about 15 ms.

A negative-control test asserts the *unsymmetrised* channels fail parity, so the
symmetrisation cannot silently become a no-op.

**This applies to Family C as well**, which is also cgNA+-derived, and to any family
built on the stiffness. Recorded here rather than fixed there, because Family C's
channel module is not yet written.

## Provisional parameters, recorded so they cannot become silent training targets

Each of these computes a propensity or an ordering where the plan asks for an absolute
free energy, and is flagged `provisional=True` with a reason in the registry:

- **Z-DNA**: only four dinucleotide B-to-Z energies could be sourced (CG 0.66, CA/TG
  1.20, TA 2.40 kcal/mol, low salt; nucleation about 5.0 kcal/mol per junction, from
  Zhabinskaya & Benham 2011 citing Ho, Ellison, Quigley & Rich, EMBO J 5:2737, 1986).
  The full ten-dinucleotide anti-syn / syn-anti table in that paper is published as an
  **image**, and a fetch could not read it. Non-alternating dinucleotides carry a
  documented high-energy default. Closing this needs the table from the PDF.
- **Quadruplex free energy and the K/Na/Li axis**: a dimensionless ordering from tetrad
  count and loop length, and a monotone encoding of the cation ordering. Not measured
  offsets.
- **Quadruplex topology probabilities**: a loop-length heuristic, not a fitted
  classifier.
- **i-motif and triplex pH scaling**: uses the free-cytidine N3 pKa; the duplex-context
  shift is not parameterised.
- **Nucleosome wrapping**: the elastic term only. Histone-DNA contact energetics and
  the salt scaling are not parameterised from measurement.
- **Cyclizability**: a Shimada-Yamakawa elastic estimate with a twist-register term,
  not the full ring-closure integral. loop-seq supervises the residual at
  delta-learning, which is what the plan intends.
- **Co-transcriptional folding**: a prefix-folding proxy, which `PLAN.md` itself calls
  a proxy.

## Dependencies added

ViennaRNA 2.8.0 and freesasa, into `${CONDA_ENVS_PATH}/phys`. ViennaRNA is the
nearest-neighbour RNA model with salt correction that the plan names as Family L's
ground truth, so Family L is almost entirely non-provisional. psi4 is still absent, so
the SAPT tier is not yet reachable.

---

## Family C, and three convention errors the parity machinery caught

*Session of 2026-11-19, continued. Everything below traces to code that ran; the
commands and their outputs are in the transcript.*

### 1. The cgNA+ rotational coordinate is a rotation vector, not a Cayley vector

`mesoscale.py` recorded, as an established measurement, that cgNA+ returns rotations as
Cayley vectors `2 tan(theta/2) n`. That was wrong, and so was the translation
convention recorded alongside it.

`CGNAPlusParams.__init__` defaults `euler_definition=True`, which converts the stored
Cayley vector into the rotation vector `Omega = theta n` in radians, and
`group_split=True`, which expresses translations in the **preceding** base-pair frame
rather than the mid-step frame. Asked of the library directly, by constructing the same
sequence twice with `euler_definition` on and off:

```
euler-flag vec : [-0.03618  0.02069  0.60784 -0.02275 -0.03561  0.33402]
cayley     vec : [-0.03734  0.02135  0.62737 -0.03667 -0.03259  0.33309]
2atan(|c|/2) = 0.60927 rad (34.908 deg);  |e| = 0.60927 rad (34.908 deg)
euler == theta*nhat ? True
euler == cayley     ? False
```

**Why it hid.** The two differ at third order, about 3 percent at a B-DNA twist of 34
degrees. The twist coordinate 0.6028 reads as 33.56 degrees under the Cayley reading and
34.54 degrees under the correct one, and both are inside the plausible range for an
A-tract. Rise was unaffected. Every scalar magnitude looked right.

**What it cost.** `helical_path` applied the Cayley formula and the mid-step translation
rule. Checked against `CGNAPlusConf`, the library's own chain builder:

| | before | after |
| --- | --- | --- |
| base-pair frame orientation error | mean 16.2 deg, max 32.4 deg | 0 (1e-9) |
| origin RMS deviation | 0.093 nm, max 0.200 nm | 0 (1e-9) |

End-to-end distance agreed to 0.03 percent either way, which is why no existing test
caught it: the sequence is nearly straight, so the errors cancelled in the one aggregate
that was being checked. What was corrupted was every **direction**-valued quantity --
bend direction, nucleosome rotational phasing, the curvature vector.

Persistence length moved 47.1 -> 49.2 nm, closer to the measured 50 nm.

**Fix.** `_cayley_to_rotmat` replaced by `_rotvec_to_rotmat` (Rodrigues); translations
applied in the preceding frame. `tests/test_mesoscale.py` now pins `helical_path`
against `CGNAPlusConf` to 1e-9, and asserts the coordinate convention by re-deriving it
from the explicitly-Cayley call rather than by asserting a magnitude.

**Rule taken from this.** A convention is checked against the authority that defines it,
not against a remembered docstring, and not against a magnitude that both candidates
satisfy.

### 2. Reported coordinates need the mid-step frame, or six channels silently break parity

Family C reports the 12 rigid-base coordinates. In cgNA+'s native preceding-frame
convention the translations do **not** obey the reverse-complement rules every published
table states. Measured on an exact palindrome, where parity must hold exactly:

| coordinate | reference frame | mid-step frame |
| --- | --- | --- |
| shift | 8.0e-01 (fails) | 2.8e-13 (antisymmetric) |
| slide | 5.3e-01 (fails) | 1.6e-13 (symmetric) |
| rise | 1.8e-02 (fails) | 2.9e-14 (symmetric) |
| shear | 1.5e-01 (fails) | 9.1e-14 (antisymmetric) |
| stretch | 4.4e-01 (fails) | 1.2e-13 (symmetric) |
| stagger | 1.6e-01 (fails) | 3.1e-13 (symmetric) |

The rotations are exact in either convention, which is how this hid: half the vector was
always right. `geometry.to_mid_frame` applies `w_mid = exp(Omega/2)^T w_ref`.
`mesoscale.helical_path` deliberately does not, and must not -- the chain composes in
the reference frame. Same numbers, two conventions, each correct for one job.

A negative-control test asserts the raw translations still fail, so the conversion
cannot become a no-op.

### 3. cgNA+ cannot reach backbone torsions without a minimisation

The plan asks Family C for backbone torsions and BI/BII populations. cgNA+ carries
explicit phosphate rigid bodies, so this looked directly reachable. It is not, and the
reason is measurable: cgNA+ does not constrain the covalent bond between the sugar,
which rides the base frame, and the phosphate group, which is its own rigid body.

| bond | expected | raw ground state | raw across the ensemble | after minimisation |
| --- | --- | --- | --- | --- |
| C3'-O3' | 1.43 A | 1.641 A | 0.64 to 3.44 A | 1.423 +- 0.002 A |
| O5'-C5' | 1.44 A | 1.223 A | 0.07 to 4.71 A | 1.420 +- 0.002 A |

Every torsion turns about one of those two bonds or involves an atom across them, so
all six are corrupted:

| torsion | raw | minimised | canonical B-DNA |
| --- | --- | --- | --- |
| alpha | -19.9 (off 43) | -54.2 (off 8.8) | -63 |
| beta | -164.1 (off 25) | 179.2 (off 8.2) | 171 |
| gamma | -9.3 (off 63) | 40.7 (off 13.3) | 54 |
| delta | 174.9 (off 52) | 134.8 (off 11.8) | 123 |
| epsilon | -167.5 (off 7.5) | 178.9 (off 6.1) | -175 |
| zeta | -125.2 (off 25) | -101.6 (off 1.6) | -100 |

`minimize.restrained_minimise` repairs it with the restrained atoms moving only 0.344 A
RMS, so the sequence dependence survives. **The electrostatics pipeline already runs
exactly this minimisation on every fragment**, so in production the torsion channels are
free: they are read off a structure that is built anyway.

What is *not* free is their variance, which would need every ensemble snapshot minimised.
Those 12 channels are provisional. So are the BI/BII populations, for the separate reason
that cgNA+ is a single Gaussian where the real epsilon-minus-zeta distribution is
bimodal; what is shipped from one structure is the sign, not an occupancy.

Sugar pucker is not reachable at all -- the ring is rigid in the base frame, so an
unminimised pucker is the template's and a minimised one is the force field's, neither a
cgNA+ prediction. Eight channels, provisional, NaN.

### 4. A fourth parity value was needed, and it is not an escape hatch

`Parity.COMPOSITE` added to `channels.py`: the channel transforms under reverse
complement, but not by a sign on itself, because the map sends it to a different index
or a different channel. Three groups measured into it:

- **Groove widths and depths.** SYMMETRIC under an index shift of exactly the groove's
  own phosphate offset, +4 minor and -4 major, holding to 1e-14. A groove is a property
  of two phosphates on opposite strands, so reverse complement moves it along the
  sequence as well as reflecting it. Whole-sequence means are unaffected and stay
  SYMMETRIC.
- **Phosphate descriptors.** STRANDED with *no* shift at all, to 1e-14: radius even,
  azimuth and axial offset odd under the strand exchange. The exchange rule is asserted.
- **All 78 entries of the per-step 12 by 12 stiffness.** The first guess was that only
  the 36 cross intra-by-inter entries would be composite. Measurement refuted it --
  median residuals 0.48 intra-intra, 0.65 intra-inter, 0.52 inter-inter, where the 12
  coordinates reach 1e-13 in the same computation. Two independent causes, both open
  items: (a) the (intra_i, step_i) grouping is not RC-covariant, since reverse
  complement sends intra i to n-1-i and step i to n-2-i, and the covariant object is the
  18 by 18 over (intra_i, step_i, intra_{i+1}); (b) the stiffness is a quadratic form in
  native coordinates whose translational half is in the preceding frame, which per
  finding 2 does not obey parity, and carrying it to the mid-step frame needs a
  congruence transform by the Jacobian of that change of variables. Neither is applied.
  The channels are emitted and usable; what is withheld is a parity claim.

The test asserts that *no* entry carries a simple parity and would fail if one did, so
COMPOSITE stays a measured property rather than a blanket.

### 5. Methylation is the one real condition axis in Family C

`Di_hmethyl_methylated-hemi_combine` is a fitted parameter set over an extended alphabet
(M/N for 5-methylcytosine and its partner, H/K for 5-hydroxymethylcytosine). It
reproduces the unmodified parameter set exactly on unmodified sequence, which is
asserted, so a methylation delta cannot be an artifact of switching parameter sets.
Full CpG methylation gives, interior mean: twist -0.272 deg, roll +0.212 deg, minor
groove width +0.130 A, major groove width +0.087 A -- small, and in the measured
direction.

Its end blocks cover only the 16 unmodified dinucleotides, so a modified base in a
terminal dinucleotide raises `KeyError: 'GM'`. `methylate_cpg` leaves terminal CpGs
alone; since the plan scores only central windows this affects nothing reported.

High salt and elevated temperature have no parameter set and emit **NaN**, flagged
provisional. A plausible number there would be a fabrication.

### Family C status

212 channels against the plan's 215. 175 production, 37 provisional.

| | | measured B-DNA |
| --- | --- | --- |
| minor groove width | 5.79 A | ~5.7 A |
| major groove width | 12.36 A | ~11.7 A |
| phosphate radius | 9.77 A | ~9.3 A |
| mean twist | 34.4 deg/step | 34.3 deg/step |
| BI population | 0.78 | 0.7-0.85 |
| minimised torsions | within 1.6-13 deg | canonical BI |

Groove offsets -4 and +4 were found by scanning phosphate index offset from -14 to +14
and taking the two local minima of the P-P distance, not assumed; the test re-runs the
scan. Full suite: **166 passed, 1 xfailed**.

### Open items added here

- The 18 by 18 RC-covariant stiffness object, and the mid-frame congruence transform of
  the stiffness. Either would turn 78 COMPOSITE channels into parity-carrying ones.
- Torsion variances, which need per-snapshot minimisation.
- `Parity.COMPOSITE` channels are passed through `symmetrise_channels` untouched. They
  need a family-supplied symmetrisation using their own index map before they are used
  as training targets.

### Queue

`pangea-conv` (33904051), `pangea-pb` (33899777) and `pangea-qc6` (33899782) have sat
PENDING on `day` behind Priority all session. The convergence study is plan-mandated and
blocks the 93 bp low-salt question, so it was resubmitted to the uncapped `scavenge`
partition as `pangea-convs` (33912642), with `--requeue` since scavenge preempts.

---

## Family E, and a transfer matrix that only the per-base-pair profile could falsify

*Session of 2026-11-19, continued.*

89 channels against the plan's 85. The nearest-neighbour half already existed in
`analytic_targets`, validated against SantaLucia's published values; what was added is
the statistical mechanics built on it.

### The partition function is exact, including under torsional stress

Opening is a helix-coil transition in Zimm-Bragg form with a sequence-dependent stacking
weight per step and Blake & Delcourt's cooperativity parameter. Opening probabilities
come from forward-backward over the transfer matrix, and bubble statistics come from the
same pass, so the two cannot disagree.

Superhelical stress makes the energy **non-additive**: the torsional term depends on the
total number of open base pairs, because each opened pair absorbs about a tenth of a turn
of linking deficit. A plain transfer matrix is local and cannot represent that. The
recursion therefore carries a second index counting base pairs opened so far, giving
Z(k) for every k, with the Benham stress term applied per k before summing. O(N^2)
instead of O(N), and it is the difference between computing stress-induced
destabilization and naming it.

The two implementations are checked against each other: with the domain stiffness set to
zero they agree to **7e-18** on both the opening profile and the run count. The right
limit is K = 0, not sigma = 0, and that distinction is itself physics -- a topologically
closed domain at zero linking difference still resists opening, because opening generates
positive supercoiling. Measured, that closed domain opens 27 percent less than a free one.

### The validation that was not fitted

| | computed at 37 C | measured by imino exchange |
| --- | --- | --- |
| AT / GC opening ratio | **10.05** | about 10 |
| AT opening constant | 4.5e-7 | 1e-5 to 1e-6 |
| GC opening constant | 4.5e-8 | 1e-6 to 1e-7 |

The ratio falls out of the SantaLucia stacking parameters and Blake's cooperativity with
nothing tuned, which validates the whole chain. The **absolute values sit about 20 times
low**, and that is recorded at the computed value rather than closed by adjusting sigma.
The cause is identified: sigma is the nucleation parameter for cooperative *melting*,
while opening a single base pair is a less disruptive event in which the neighbours
restack. Tuning it to match would be fitting to the validation. SIDD energies land at 2
to 20 kcal/mol, inside Benham's published range.

### Three bugs, each caught by a different test, none visible in any aggregate

**1. The transfer matrix charged stacking on nucleation.** The coil-to-helix entry was
`sigma * s` rather than `sigma`, which credits a step with stacking energy when the base
pair on one side of it is open. A stacking weight is earned only when *both* flanking
base pairs are helical.

What makes this worth recording is how it hid. **Every whole-sequence channel was exactly
reverse-complement symmetric with the error in place** -- to 0.00e+00, not approximately.
The aggregates could not see it. Only the per-base-pair profile on an exact palindrome
revealed it, at a relative residual near 1.0. The initial vector needed the same
treatment, `[sigma, 1]` rather than `[1, 1]`, so that a helical run starting at base pair
0 pays nucleation like any other and the two ends of the chain are equivalent.

Fixing it moved the absolute opening constants down by a factor of 6, which is how the
20-fold discrepancy above came to be measured honestly instead of accidentally agreeing.

**2. Run count and open count came from different models.** Mean bubble size is mean open
base pairs over mean open runs. The open count came from the stressed recursion and the
run count from the unstressed one, giving **0.72 base pairs per bubble** -- impossible,
since a run contains at least one open pair. Both now come from one partition function.

**3. A scaling mismatch inside the stressed recursion.** Forward and backward arrays are
rescaled independently at every position, so a quantity built from `B[i+1]` must be
divided by that position's backward scale factor before being compared with a
normalisation formed from `B[i]`. Leaving it out gave more open runs than open base
pairs. There is now a comment at the site saying why, because the same trap is one line
away in three places.

**4. The run count omitted a boundary term.** Counting only interior (helix, coil)
boundaries counts run *starts* read left to right; under reverse complement that becomes
a count of run *ends*, and the two differ exactly when a run touches an end. Mean bubble
size came out 1.61 forward and 3.11 on the reverse complement. Adding the probability
that base pair 0 is itself open makes the count symmetric.

### Parity

Every Family E channel is SYMMETRIC, measured on an exact palindrome, except the
pH-dependent ones. `E_imotif_dG_pH*` and `E_triplex_dG_pH*` are **STRANDED**: an i-motif
is built from one strand's cytosines and a triplex from one strand's purine tract, so
reverse complement exchanges them for the partner strand's value rather than leaving them
alone. Family F declares the same structures STRANDED and the two families must not
disagree about the same molecule. A negative-control test asserts they really do differ
under reverse complement, so the declaration cannot go stale.

### What is provisional, and why

- **Opening and closing rates** (2 channels). An attempt frequency is not derivable from
  equilibrium thermodynamics; the closing rate is pinned at the 1e7 per second imino
  exchange reports and all sequence dependence comes from the equilibrium constant.
- **Hairpin and cruciform** (6). The stem is exact nearest-neighbour; the loop initiation
  term is a Jacobson-Stockmayer extrapolation from one reference value rather than the
  measured DNA loop table.
- **Triplex and i-motif** (6). Conversions of Family F propensity scores into energies by
  `dG = -RT ln(score)`, not independent thermodynamic calculations.
- **Heat capacity** (2). One literature value applied uniformly, where the published
  range spans 0.05 to 0.13 kcal/mol/K per base pair.

### Stated simplifications

The loop-entropy factor of full Poland-Scheraga is absent, so long internal loops are
cheaper here than they should be; this affects the tail of the bubble size distribution
more than the opening probabilities. The open region's own torsional stiffness is omitted
from the stress term, making opening slightly too easy at high superhelical density.

### Status

Registry now holds **301 channels** across families C and E: 248 production, 53
provisional. Full suite **194 passed, 1 xfailed**.

`pangea-convs` (33912642) is RUNNING on scavenge. Its first condition block shows **121 bp
solves completing in 33 s each with no failures**, which already indicates the earlier
93 bp failure was specific to that run's padding or grid rather than a length limit.

---

## Family B tier 1, the landscape driver, and PANGEA

*Session of 2026-11-19, continued. The user authorised installing whatever packages
training needs.*

### Family B: 34 channels, and what is deliberately not declared

`results/qc_tetramer_gfn2.json` turned out to be **complete**: 136 contexts, all
converged, and the table is exactly keyed on `min(t, rc(t))`, so all 256 tetramers
resolve -- 136 directly and 120 through reverse complement -- with no gaps. Lookup swaps
the two base pairs of the central step when it resolves through the reverse complement,
since they exchange; the symmetric quantities would not reveal a missing swap.

`PLAN.md` budgets 98 channels for Family B. **34 are declared.** The SAPT decomposition
into electrostatic, exchange-repulsion, induction and dispersion terms, the DFT tiers,
transfer integrals, molecular electrostatic potentials at the hydrogen-bonding atoms,
pKa shifts and the polarizability tensor are **not declared at all**, rather than
declared and filled with NaN. The distinction is deliberate: declaring a channel and
leaving it empty is right when the quantity is defined but unavailable for a particular
sequence, and wrong when its ground truth has never been computed, because it puts a
column in the training set that no run could ever fill.

Sanity check on the tier that does exist: stacking energies span -19.6 to -12.4 kcal/mol
against a published range of -9.9 to -14.7, so GFN2 overbinds by roughly 5 kcal/mol.
That offset is stated on every Family B channel. The lowest ionization potential in a
test fragment falls on a GC-rich context, which is the expected place for a hole.

### A cross-family contract test, and the three things it found

Families C and E had per-family contract tests asserting emitted == declared. D, F and L
had none, and nothing compared the *full* emitted set against the *full* declared set.
`src/physics/landscape.py` now computes every family through one entry point and
`tests/test_landscape.py` checks the contract in both directions at once.

**1. The target generator wrote a dataset with zero channels and reported success.**
Family modules register their channels as an import side effect; `landscape` imported
them lazily inside `compute_landscape`, so a caller that asked for the column set
*first* -- exactly what the generator does -- saw an empty registry. The generator built
a well-formed HDF5 file containing nothing, printed "channels: 0 per base pair, 0 whole
sequence", and exited 0.

Fixed three ways, because one was not enough: the family imports are eager, the generator
refuses to write an empty channel set, and a test checks the registry is populated by
importing `landscape` alone **in a subprocess**. The subprocess matters -- inside a test
session every family has already been imported by other tests, which is precisely the
condition under which the bug is invisible.

**2. Seven per-base-pair tracks were emitted but never declared.** `F_g4hunter`,
`F_imotif`, `F_zdna_step_kcal`, and the plus- and minus-strand RNA unpaired-probability
and positional-entropy tracks. The generator builds its columns from the registry, so all
seven would have been silently absent from every training example.

Parity measured on an exact palindrome before declaring, all three Family F tracks exact
to 0.00e+00:

| channel | parity | residual |
| --- | --- | --- |
| F_g4hunter | ANTISYMMETRIC | 0.00e+00 |
| F_imotif | ANTISYMMETRIC | 0.00e+00 |
| F_zdna_step_kcal | SYMMETRIC | 0.00e+00 |
| L_{p,m}_unpaired_prob | STRANDED | coincide to 3.5e-05 on a palindrome |
| L_{p,m}_positional_entropy | STRANDED | coincide to 2.0e-03 |

`F_imotif` was expected to be STRANDED and is not; the measurement decided it.

**3. Families C and E each computed their profiles twice.** `geometry_channels` and
`thermo_channels` called their own profile functions internally even when the caller
already held them. 1.74 -> 1.41 s per fragment.

Two smaller quality gaps surfaced from the same test: 28 Family F production channels
carried no `source`, and one provisional channel's reason was four words. The 28 are
*definitions* rather than parameterised models -- a count of quadruplex candidates has no
publication to cite because it has no fitted parameter -- so they now carry an explicit
`_DEFINITION` marker. "No source" always means an omission now, never "nothing to say".

### Measured cost of stage-1 label generation

| | s/fragment at 61 bp | over 191,344 fragments |
| --- | --- | --- |
| analytic stack (B, C, D, E, F, L) | 1.41 | 75 CPU-h |
| restrained minimisation | 4.6 | 245 CPU-h |
| **total** | **6.0** | **320 CPU-h** |

About 14 minutes of wall clock at the 1,500-core cap. Label generation is not the
bottleneck and will not become one, so nothing was traded away: the Family C ensemble
runs at its full 200 draws and the minimisation runs on every fragment.

Running as `pangea-targets` (33926438), a 512-way array on `scavenge`. Sharded 512 ways
rather than 128 because scavenge preempts with REQUEUE and a requeued task restarts from
the beginning, so the shard size bounds what a preemption costs. Shards are **strided**,
not contiguous, so every shard sees every stratum.

### The training environment

Separate `train` conda environment, kept apart from `phys` on purpose: label generation
needs OpenMM, ViennaRNA and cgNA+, training needs none of them, and labels go to disk
between the two. The training job therefore cannot import the physics even by accident,
which is what makes "the channel layout comes from the dataset, not from the registry"
structural rather than a matter of discipline.

Two silent failures getting there, both worth remembering. `set -euo pipefail` combined
with sourcing lmod's init **aborted the install script while exiting 0** -- lmod returns
non-zero on paths that are not failures. And a shell-backgrounded `&` job did not survive
the tool call, also reporting success. Both produced "exit 0, nothing happened".

### PANGEA

`src/model/` was an empty directory at the start of this session. It now holds the
architecture, the data layer and the training loop.

**Reverse-complement equivariance is architectural.** The trunk runs twice with shared
weights, on the sequence and on its reverse complement, and the outputs merge by each
channel's *declared* parity: averaged for SYMMETRIC, differenced for ANTISYMMETRIC, and
passed through for STRANDED and COMPOSITE. This is exact to floating point for any
weights, at initialisation and after any number of gradient steps, which is what makes it
a property of the architecture rather than something a validation metric has to discover.
It costs a second forward pass. `tests/test_pangea.py` verifies it on an **untrained**
model, with a negative control that disabling the merge breaks it, so the test cannot
pass vacuously.

COMPOSITE channels are passed through rather than averaged, because they transform under
an index shift rather than a sign -- averaging them would be wrong, not conservative.

Also implemented: FiLM conditioning at every block (parameterised as 1 + delta so a zero
condition vector is exactly the identity), the low-rank output head that makes rank r the
measurement the plan asks for, heteroscedastic mean-and-variance outputs under a Gaussian
NLL, per-family loss weights learned by uncertainty weighting, the three-resolution
pyramid, and the inter-scale consistency loss. NaN targets are **masked, never imputed**:
a large part of the landscape is legitimately undefined at a given position, and filling
it with zeros would train the model to predict zero there.

Provisional channels are **excluded from training by default**. `channels.ChannelSpec`
says they must not quietly become targets, so `--allow-provisional` is what it takes.

### What the first train cannot be, and why

Every fragment in the library is exactly 61 bp. That caps the window ladder at the 10, 20
and 50 bp rungs, leaves 100, 200 and 500 untrainable, and means the wide 5-to-10 kb
branch has no context to see. The wide branch is implemented and **off by default** --
fed 61 bp it would train to nothing while still looking like a working component.

A longer-fragment library is needed for the full pyramid, and the plan already calls for
one for the integrated electrostatic channels, so it is wanted regardless.

### Status

Registry: **528 channels** over families B, C, D, E, F, L -- 226 per base pair, 302 whole
sequence, 435 production and 93 provisional. Test suite **209 passed, 1 xfailed**, plus
the PANGEA tests which need the training environment.

---

## The convergence study found a solver bug, and target generation found a throughput one

*Session of 2026-11-19, continued.*

### The 93 bp failure explained: one `dime`, two grid levels

`PLAN.md` specifies "a coarse grid near 1 angstrom over the full fragment plus solvent
padding and a fine grid near 0.35 angstroms over the central 21 bp". `mg-auto` uses a
**single `dime` for both levels**, and `write_apbs_input` sized it from the fine box
alone. The coarse grid therefore degraded as fragments grew:

| fragment | coarse box | coarse spacing | on spec? |
| --- | --- | --- | --- |
| 21 bp | 191 A | 0.60 A | yes |
| 61 bp | 327 A | 1.02 A | yes |
| 93 bp | 436 A | 1.36 A | **no** |
| 121 bp | 531 A | 1.66 A | **no** |

Everything past about 61 bp was off the plan's own specification, and the symptom was not
quiet inaccuracy but outright failure. The plan-mandated convergence study made it
unmistakable: 121 bp fragments solved cleanly at 10 mM, 24 of 24, and then **aborted on
every single one at 50 mM and 100 mM** with an overflow after 17 s. The nonlinear
Boltzmann source term scales with ionic strength, so on a 1.7 A grid beside a phosphate
it overflows; at 10 mM the ion density is low enough to survive.

A 93 bp solve had failed the same way earlier in the session and been recorded as an open
item. Worse, the shared-`dime` behaviour had already been *noticed* and filed as a cost
curiosity -- "grid points identical at 21 and 93 bp, so solute length is not a cost
lever" -- without connecting it to the failure. The observation was right and its
interpretation was wrong: it was a correctness bug wearing the costume of an accounting
result.

**Fix.** `dime` per axis is now the larger of what the fine box needs at 0.35 A and what
the coarse box needs at 1.0 A. Coarse spacing is 0.93 to 0.98 A from 21 bp to 201 bp.

**Verified on the exact failing cases**, both of which previously aborted at 17 s:

```
NaCl_100mM  121 bp -> OK  754.3 s
NaCl_050mM  121 bp -> OK  121.2 s
```

**Cost consequence, which is real.** Grid points go from 5.3M to 9.1M at 121 bp, and the
121 bp solve now takes about 121 s against the 22 s measured at 21 bp. So solute length
*is* a cost lever after all -- it simply was not one while the grid was wrong, because
the coarse level was being silently under-resolved. The earlier electrostatics budget was
computed from 21 bp solves at the correct grid and stands for the short, high-salt
conditions; the low-salt conditions, which need 93 bp fragments, are more expensive than
that budget assumed and the number needs recomputing from the convergence study's own
timings once it reruns.

The convergence study currently running (33912642) has valid data for 61, 41, 31 and 21
bp at every condition and for all five lengths at 10 mM, but its 121 bp reference is
missing at 50 mM and above, which is the reference the convergence curve is measured
against. It needs rerunning on the fixed grid.

### Target generation stalled, and the first diagnosis was wrong

The first 512-way target-generation array produced **zero shards in 45 minutes** at 256
concurrent tasks. Not slow: stalled. No task emitted even its first progress line, and
those print every 25 fragments, so each task had managed fewer than 25 fragments in 45
minutes against a measured 6.4 s each.

**First diagnosis, which was wrong.** `minimize.restrained_minimise` constructed
`app.ForceField("amber14/DNA.OL15.xml")` on every call, parsing the XML from disk once
per fragment, and with 256 tasks re-reading the same files from one NFS server that
looked like an obvious filesystem bottleneck. A per-process cache was added
(`minimize._force_field`) and measured: **4.84 s/fragment against 4.85 s before.** No
effect. The cache is kept because it is harmless and correct, but it was not the problem,
and the NFS story was a guess that happened to fit the symptom.

**Actual cause: OpenMM's CPU platform defaults `Threads` to the machine's physical core
count.** Asked directly, it reports 48 on these nodes. It reads that from the hardware,
not from the Slurm allocation or the cgroup, and -- this is the part that makes it easy
to miss -- `OMP_NUM_THREADS` does not control it. Only `OPENMM_CPU_THREADS` or the
platform property does, and the Slurm script set every *other* threading variable to 1.

So a 512-way array declaring `--cpus-per-task=1` had every task spawning 48 OpenMM
threads. Dozens of tasks per node, 48 threads each, contending for cores none of them
were allocated. The 100-fold slowdown was oversubscription, and the filesystem was a
bystander.

**Fix, in two places so the environment cannot be the only thing standing between this
and a repeat.** `restrained_minimise` now selects the platform explicitly and sets
`Threads` from `OPENMM_CPU_THREADS` with a default of 1, so one allocated core gets one
thread whatever the caller forgot to export; and the Slurm script exports
`OPENMM_CPU_THREADS=1` alongside the others. The array is also throttled to 64
concurrent rather than 256, which is prudence about shared infrastructure rather than a
fix for this bug.

**What this means for the earlier timing.** Every per-fragment measurement in this
session was taken on a login node where OpenMM helped itself to 48 threads, so
4.84 s/fragment is a 48-thread number, not a single-core one. The honest single-core cost
is being measured now, and the stage-1 label budget has to be restated from it. This is
the second time this session that a cost figure turned out to be a thread-count artefact;
the first was the APBS budget, where multiplying wall time by an allocation's thread
count inflated it 20-fold. The lesson did not transfer the first time because it was
recorded as an arithmetic error rather than as a class of error: **a time is not a
CPU-time until the thread count is known.**

### A note on the operating environment

Three separate silent failures this session traced to the same class of problem --
something reporting success while doing nothing:

- `set -euo pipefail` plus sourcing lmod's init aborted an install script **exiting 0**.
- A shell-backgrounded `&` job did not survive its tool call, also reporting success.
- The target generator wrote a dataset with **zero channels** and printed a success line.

Only the third was caught by a test, and only because a test was written for it
afterwards. The first two were caught by checking whether the artefact existed rather
than whether the command succeeded. That is the habit worth keeping: verify the output,
not the exit code.

---

## Model verification, and a loss that diverged on a family with nothing to fit

*Session of 2026-11-19, continued.*

### The training environment

`torch 2.15.0+cu132`, bundled **CUDA 13.2**, with h5py and pytest, in a `train`
environment separate from `phys`. Import of torch takes 47 s from NFS, which is a
one-time cost per job.

The bundled CUDA is 13.2 while the cluster's newest module is 12.8. That is not
necessarily a problem -- torch wheels carry their own runtime and what matters is the
driver -- but it is not verified either, so `train.py` now prints the torch version, the
bundled CUDA version, whether CUDA is available and which device, and takes
`--require-cuda` to make a CPU fallback an error instead of a slow success. A GPU probe
job (33929895) reports the driver version directly.

### `gpu_devel` allows one job per user

Three attempts to submit a GPU probe were rejected with `QOSMaxSubmitJobPerUserLimit`,
and the obvious explanation -- a 512-element array filling a submit quota -- was wrong.
`part_gpu_devel` has `MaxSubmitPU=1`, and an unrelated job of the user's was already
running there. `part_gpu` has `MaxJobsPU=12` and no submit limit. Recorded because the
error message points at a global quota and the cause was partition-specific.

### The model tests do not import the physics

`tests/test_pangea.py` builds its channel plan from explicit metadata rather than from
the registry, for the same reason the training job reads its layout from the dataset: the
training environment deliberately has no cgNA+, OpenMM or ViennaRNA, so a test that
imported them could only run in the other environment. A separate test asserts the real
registry exercises **all four** parity classes, skipping where the physics is
unavailable, so the equivariance merge has no untested branch in production.

The equivariance tests run on an **untrained** model with random weights, and again after
a gradient step, because that is what makes architectural equivariance a property of the
architecture rather than something that might be learned. Each has a negative control
asserting the same model fails with the merge disabled, so the test cannot pass for
trivial reasons -- a near-constant output would satisfy an equivariance check without the
merge doing any work.

### The loss diverged on a family with no valid targets

21 of 22 model tests passed first time. The failure was worth more than the passes.

The uncertainty weighting contributes `exp(-s) * L + s` per family, where `s` is the
learned log-variance and the `+ s` term is what stops every weight collapsing to zero.
A test that gave one family a perfectly-fit target -- `L` exactly 0 -- found that nothing
then balances `+ s`: the gradient with respect to `s` is a constant 1, `s` grows without
bound, and the reported weight came back **NaN**.

`L = 0` is not a contrived case. It is exactly what happens when a family has **no finite
targets at all**, which will occur in production for any family whose channels are all
filtered out as provisional, or for a family whose data is missing from a batch.

Two fixes:

- `gaussian_nll` now returns the count of finite targets alongside the value, and
  `forward` **skips a family with zero valid targets entirely** rather than letting it
  contribute a zero loss. A family with nothing to fit should not appear in the loss at
  all; conflating "fits perfectly" with "has no data" was the actual error.
- The log-variance is clamped where it is *used*, to a weight range of about 0.0025 to
  400, which covers the orders of magnitude the channel set differs by. Clamping at use
  rather than clipping the parameter keeps it differentiable.

The test was then rewritten into two: one checking a family with no valid targets is
absent from the report and leaves the parameters finite, and one checking a noisy family
really is down-weighted relative to a fittable one.

### Convergence study, rerun

`pangea-conv2` (33930809) runs the plan-mandated study on the corrected grid with the
(sequence, length) cells in a 32-way process pool. Serial is no longer viable: with the
grid fixed, a 121 bp solve costs about 121 s against 22 s at 21 bp, so the reference
length dominates and 24 sequences over five conditions would be most of a day on one
core.

### A pre-training gate

`src/data/qc_targets.py` checks the generated dataset on its own terms before anything
trains on it, and the training job refuses to proceed if it fails. It looks for channels
that are entirely NaN across the corpus, channels with no variance, channels undefined in
the *interior* of fragments rather than only at the ends, missing shards, disagreeing
channel layouts, generator failures, strata proportions, and duplicate sequences that
would leak between training and validation.

It exists because of the pattern this session kept producing: an install script exiting 0
with no environment, a backgrounded job reporting success after dying, and the target
generator writing a well-formed file containing zero channels. In each case the exit code
was clean and the artefact was empty. **The gate is the artefact, not the exit code.**

---

## The pipeline runs end to end, and the QC gate paid for itself on its first use

*Session of 2026-11-19, continued.*

### End-to-end smoke test on real generated data

A 96-fragment pilot through the whole chain -- generation, QC, dataset, model, loss,
metrics:

```
96 fragments from 1 shards
channels: 189 per base pair, 246 whole sequence (excluding provisional)
receptive field 19 bp, 0.06M parameters
epoch 0: per_bp_median_r2 -0.057  by family {B -0.040, C -0.060, E -0.063, F -0.049, L -0.027}
epoch 1: per_bp_median_r2 -0.052  global_median_r2 -0.047  (D -0.060)
```

R-squared near zero is the right answer for 96 fragments and a 0.06M-parameter model over
two epochs; negative means worse than predicting the channel mean, which is what an
almost-untrained model does. This was a plumbing test and the plumbing holds. Generation
measured 5.60 s/fragment on 96 fragments, against the 5.45 s predicted from six.

### The QC gate found two things on its first run

It exited non-zero, which is what it is for.

- **`E_strand_separation_dG_w100` is entirely NaN.** A 100 bp window cannot exist on a
  61 bp fragment. Real, but a property of the library rather than a defect in the
  channel, so the gate now recognises `_w<N>` channels with N greater than the fragment
  length and reports them as a named library limitation instead of failing. They are
  still *reported*, because such a column would otherwise sit in the training set as a
  permanent NaN with nothing fitting to it.
- **`E_strand_separation_dG_w50` flagged as "undefined in the interior".** A false
  positive in my own heuristic: a 50 bp window on a 61 bp fragment has only 12 valid
  centres and they are all central, so comparing the middle third against the maximum
  flags correct behaviour. Replaced with a **contiguity** test -- a hole is a position
  that is undefined with defined positions on both sides -- which is the property that
  actually separates a gap from a narrow band.

It also surfaced **15 constant production channels**. Two are constant by construction:
`D_writhe_sigma0` and `D_delta_twist_sigma0` are identically zero at zero superhelical
density, since there is no linking difference to partition. The rest are constant only
because 96 genomic-panel fragments happen to contain no quadruplex or triplex candidate,
which a larger sample fixes. Both cases matter for the same reason: a constant channel
has an undefined R-squared and will appear as an unexplained NaN in the metrics unless it
is known about first.

### The validation split was wrong, not merely imprecise

The split had been left random with a note that a similarity split would be better. That
understated it. `library_v2` tiles MPRA elements at a 40 bp step, so several 61 bp
fragments come from each 200 bp element and neighbouring tiles overlap by up to 21 bp.
A random split puts near-duplicates on both sides and the validation score then measures
memorisation while looking entirely normal -- which is one of the quiet failures
`PLAN.md` names explicitly, with the control being "splits by sequence similarity and by
chromosome, with the overlap quantified and reported".

`--split similarity` is now the **default**, grouping near-duplicates with the MinHash
machinery already in `src/data/splits.py` and assigning whole groups to one side. The run
logs the group count and the largest group, because a split whose groups are enormous is
not really a split. `--split random` is kept deliberately, so the leakage can be
*measured* by comparing the two rather than merely asserted.

On the pilot the split reports 96 groups of size 1 and zero confirmed pairs, which is
correct: those fragments are all from the genomic panel stratum. The tiled MPRA stratum
is 51,963 of the library's 191,344 fragments, and because shards are strided every shard
contains some, so the real split will have real groups.

### Status of the running work

- `pangea-targets` (33929651): 192 tasks running, 320 queued, about 60 minutes per shard
  at the contended rate of 9.8 s/fragment against 5.45 s measured single-core. Zero
  failures so far.
- `pangea-conv2` (33930809): the convergence study on the corrected grid, 32-way pool.
- `pangea-gpuprob` (33929895): pending. The installed torch bundles CUDA 13.2 and the
  cluster's newest module is 12.8, so whether CUDA works is a driver question and not yet
  answered. `train.py` prints the answer and `--require-cuda` turns a CPU fallback into
  an error, so the first real run cannot silently train on CPU.

---

## Family H, and the OOM that looked like a solver bug

*Session of 2026-11-19, continued.*

### Family H: 38 channels of the plan's 60, with the shortfall itemised

`PLAN.md` asks for 5 modifications by 12 observables. The shipped parameter set was
interrogated rather than assumed: its dinucleotide alphabet is **A, C, G, T, H, K, M, N**
-- that is, 5-methylcytosine (M/N) and 5-hydroxymethylcytosine (H/K) and nothing else.
**5fC, 5caC and N6-methyladenine have no parameters and are not declared at all.** Of the
twelve observables, minor groove potential needs a Poisson-Boltzmann solve (Family A),
stacking energy needs quantum chemistry on *modified* stacks which the canonical GFN2
tetramer table does not cover, melting temperature needs nearest-neighbour parameters for
modified base pairs that are not transcribed here, and hydration and photoproduct
propensity are Family G. `UNAVAILABLE_MODIFICATIONS` and `UNAVAILABLE_OBSERVABLES` name
every one, and a test asserts the absent modifications stay absent.

Measured response on a 61 bp fragment with 12 CpG sites:

| | d twist | d minor width | d persistence length | d nucleosome |
| --- | --- | --- | --- | --- |
| 5mC | -0.44 deg | +0.22 A | **+2.76 nm** | +2.23 kT |
| 5hmC | -0.49 deg | +0.29 A | **+1.15 nm** | +1.01 kT |

Methylation stiffens the duplex, which is the measured direction, and hydroxymethylation
stiffens it *less*, which is what a hydroxyl that hydrogen-bonds to water should do. Both
are asserted as directional comparisons rather than absolute values, because the ordering
is the physical statement while the magnitudes carry the parameter set's uncertainty.

Parity measured on an exact palindrome, after checking that a modified palindrome is
still a palindrome -- CpG is self-complementary and M pairs with N, so the symmetry
survives and there is something to measure. Step deltas SYMMETRIC under the step map to
1e-12; groove deltas SYMMETRIC under a shift of the groove's own phosphate offset, also
1e-12, with a negative control that they fail without the shift; the configurational
entropy delta COMPOSITE at 2e-1, which is the known cgNA+ asymmetry.

**Not yet in `landscape.FAMILIES`.** Including it would change the channel set and
invalidate the 512 target shards currently generating. It joins on the next pass.

### A bug that would have been silent in the other direction

`modification_response` obtained Family D's mechanics by calling
`analytic_targets.cgna_ground_state`, which defaults to the **canonical** parameter set.
On a modified sequence that raises `KeyError: 'TM'`, which is how it was found. The
dangerous direction is the reverse: computing the unmodified baseline with the default
set and the modified sequence with the modified set would have succeeded, and the
reported "modification response" would have been partly the difference between two
parameter sets.

Fixed by reusing the ground state the geometry call already produced -- correct, and one
fewer cgNA+ evaluation -- and by exposing `parameter_set` on `cgna_ground_state` so no
other caller can reach the default by omission.

### The convergence study was being killed, not failing to converge

The parallel rerun at 32 workers in 128 GB looked like a solver regression: 121 bp failed
everywhere, 61 bp failed too where it had previously succeeded, and the short solves took
130 s instead of 22 s. The recorded reasons settled it:

```
('NaCl_010mM', 121, 'rc=-9')            x23
('NaCl_010mM',  61, 'rc=-9')             x9
('NaCl_050mM', 121, 'overflow/abort')   x24
```

**`rc=-9` is SIGKILL**: the OOM killer. The corrected grid is 14.1M points at 121 bp and
APBS multigrid holds many work arrays per point, so 32 concurrent solves do not fit in
128 GB. Resubmitted at **8 workers in 192 GB**.

Two things were confirmed rather than assumed before blaming memory. First, the grid
change is modest -- 1.0x at 21 bp, 1.1x at 61 bp, 1.7x at 121 bp -- so it cannot explain
a 61 bp solve going from working to failing. Second, 21 bp and 61 bp both solve
**standalone** on the corrected grid, so there is no regression.

### Timings from a shared login node are not timings

The standalone check above reported 263 s for a 21 bp solve and 301 s for 61 bp, against
22 s measured earlier on a quiet node. The difference is entirely contention: it ran on
the login node while 192 array tasks and a 32-way solver pool were in flight.

The same pattern appeared twice more in the same hour. The target array's throughput went
10.4 -> 6.28 s/fragment as its start-up burst cleared, and the convergence study's first
condition showed every length at about 120 s before the second condition settled to 16-20
s once the pool pipelined.

That makes three distinct ways a measured time has misled this session: multiplying wall
time by an allocation's thread count (the 20-fold APBS error), measuring while a library
helped itself to 48 threads (the OpenMM minimisation), and measuring under contention
from one's own concurrent jobs (here). The rule that covers all three: **a measured
duration is only a cost when the thread count and the contention are both known, and the
first condition of a fresh worker pool is never representative.**

---

## Family G: most of chemical reactivity is an accessibility calculation

*Session of 2026-11-19, continued.*

78 channels against the plan's 85.

### The organising idea, and why it is testable

A chemical probe attacks a *named atom*. Whether it reacts is therefore governed by
whether solvent can reach that atom, which makes most of this family a geometric
calculation on the relaxed structure rather than a fitted sequence model:

| probe | attacked atom | what it does |
| --- | --- | --- |
| hydroxyl radical | C4', C5' | hydrogen abstraction from the deoxyribose |
| dimethyl sulfate | guanine N7 | methylation in the major groove |
| DEPC | purine N7 | carbethoxylation |
| KMnO4 | thymine C5-C6 | oxidation, only where the duplex is distorted |
| bisulfite | cytosine C6 | deamination, single-stranded only |
| DNase I | minor groove edge | cleavage, rate tracks groove width |

Each is a declared atom set and a stated geometric measure, so nothing in the list needs
a parameter.

**The premise is checked, not assumed.** Balasubramanian, Pogozelski and Tullius showed
hydroxyl radical strand-break yield tracks the accessible surface of the backbone
hydrogens, with C4' and C5' the preferred sites. Measured on a relaxed 61 bp duplex:

```
C5' 21.69   C4' 15.97   C2' 8.68   C3' 7.64   C1' 2.60      (mean A^2)
OP1 39.95   OP2 34.79   O3' 6.45   O5' 2.74
apolar 5049 A^2   polar 7276 A^2
```

C5' and C4' are the most exposed sugar carbons by a wide margin, which is the mechanism
the cleavage channel rests on. A test asserts they exceed C1' by more than threefold,
because if that failed the channel would simply fit ORChID badly for a reason nobody
could name.

**SASA is computed on the minimised structure, necessarily.** The raw cgNA+
reconstruction has C3'-O3' at 1.64 angstrom against 1.43, and across a Gaussian ensemble
from 0.64 to 3.44. A surface area computed from atoms at wrong distances is wrong
smoothly, which is the worst kind. A test asserts the function refuses a missing
structure rather than degrading.

### 8-oxoguanine reuses Family B

Guanine oxidation is governed by the ionization potential, and the GFN2 tetramer table
already carries a vertical ionization potential for every context. The susceptibility
channel is derived from it rather than modelled separately -- one implementation of one
quantity -- and it is **NaN without Family B** rather than guessed, which a test pins.
It is also NaN at every non-guanine position, which another test pins, because a
susceptibility at a thymine would be meaningless.

### Eleven observables are named as unavailable rather than approximated

Groove hydration occupancy, water residence time, spine-of-hydration continuity, water
bridge probability and interfacial water entropy all need explicit-solvent molecular
dynamics; a static structure has no residence time to report. Intercalator and
minor-groove-binder affinity need docking. Depurination, depyrimidination and alkylation
*rates*, and abasic site stability, need kinetic parameters that are not transcribed
here -- the N7 accessibility this module does compute is the structural half of the
alkylation problem, not the rate. All eleven are in `UNAVAILABLE` with reasons, and a
test requires each reason to exceed 40 characters and to say what it *requires*, after
the Family H experience of a reason reading only "Family G".

### What is provisional

The polar hydration term applies one coefficient to all nitrogen, oxygen and phosphorus
rather than Eisenberg and McLachlan's per-atom-type table, so its absolute scale is not
traceable even though the sequence ranking, which the surface area carries, is. The
dipyrimidine *classification* is exact and not provisional; the propensity weighting
across the four classes and the roughly twofold enhancement at methylated cytosine are
flat literature factors, so those channels are. The plan's ground truth for them is
CPD-seq and dCPD-seq, and this is the structural prior measurement would correct.

### Families G and H are not yet in the landscape

Both are complete and tested, and neither is in `landscape.FAMILIES`, because adding a
family changes the channel set and would invalidate the target shards now generating.
They join on the next pass. Family G also needs the minimised structure, which the
generator already builds per fragment and currently discards after Families C and E have
used it, so including G costs nothing extra.

### Why Families G and H wait for the array to finish

Adding a family to `landscape.FAMILIES` changes the channel set, and the target array is
running 192 tasks that each imported the module at task start. A task starting *after*
such an edit would emit more columns than one that started before it, producing shards
with mismatched layouts inside a single set.

`LandscapeShards` raises on exactly that mismatch, so it would be caught rather than
silently trained on -- but it would spoil the run. The channel set is therefore frozen
until the array drains, and Families G and H join on the next generation pass.

That pass regenerates everything rather than writing G and H into side shards keyed by
fragment id. The reason is cost: Family G needs the minimised structure, which is 4 s of
the 5.45 s per fragment, and Family H needs two further cgNA+ evaluations, so a
G-and-H-only pass would cost almost what a full pass costs. One pass, one channel set,
one layout.

---

## The convergence study answers its question with a condition-dependent reference

*Session of 2026-11-19, continued.*

At 8 workers in 192 GB the `rc=-9` SIGKILLs are gone, which confirms the earlier diagnosis
was memory. What remains is narrower and real:

```
NaCl_010mM   121 bp: 24/24 solved      median 253.7 s
NaCl_050mM   121 bp:  0/24             overflow/abort at 56.8 s
NaCl_100mM   121 bp:  0/24             overflow/abort
             61 / 41 / 31 / 21 bp: 24/24 at every condition
```

**A genuine nonlinear Poisson-Boltzmann overflow at 121 bp and 50 mM or above**, and the
physics names it: the Boltzmann source term is proportional to ion concentration, so the
low-salt case converges and the higher-salt case diverges. It aborts in 57 s, faster than
a success takes, which is itself the signature of a divergence rather than a timeout.

**The study does not need a 121 bp reference at those conditions, and this is not a
concession.** Revision 2 sets fragment length from the Debye length: the flank required
for about 2 percent residual contribution is 36 bp at 10 mM but only 16 at 50, 11 at 100,
9 at 150 and 7 at 300 -- fragments of 35 to 53 bp. A 61 bp reference exceeds every one of
those, and 61 bp solves 24 of 24 at every condition. At 10 mM, the one condition that
genuinely needs a long fragment, 121 bp solves 24 of 24.

So the reference length is **condition-dependent, exactly as the production fragment
length already is**. Applying the plan's own logic to the study's reference rather than
fixing it at the longest length is the consistent choice, and it makes the study
answerable as it stands rather than after a solver change.

### The open item this leaves, stated narrowly

Long fragments at high salt overflow the nonlinear solver. The standard remedies are to
solve the linearised equation first and use it as the initial guess for the nonlinear
one, or to add focusing levels so the coarse potential near the phosphates is better
resolved before the nonlinear term is evaluated. Neither is implemented.

This matters only for a use that wants a long *and* high-salt fragment, and the plan's
own Debye-length argument says it does not: at high salt a long fragment is "several
times larger than necessary". The item is recorded so that if some later channel does
need one -- the integrated window-level electrostatics keep a long-fragment set, though at
which conditions is not yet decided -- the cause and the remedy are already known rather
than rediscovered.

### A note on what the first diagnosis cost

This one failure mode was diagnosed three times before it was right: first as a padding
problem, then as the shared-`dime` coarse-grid bug, then as memory. The first was wrong,
the second was real and fixed a different set of failures, the third was real and fixed
the 10 mM case. Only after all three was the residual narrow enough to name correctly.

What made the difference each time was reading the recorded failure *reason* rather than
the pass/fail pattern: `rc=-9` is SIGKILL and means memory, `overflow/abort` means the
solver diverged, and the two had been lumped together as "FAIL" in the log line while the
JSON had been distinguishing them all along.

---

## Sequence overlap, quantified

*Session of 2026-11-19, continued.*

`PLAN.md` names similarity leakage as one of the ways the project fails quietly -- "MPRA
libraries and cross-species sets share sequence" -- with the control being "splits by
sequence similarity and by chromosome, **with the overlap quantified and reported**".
Quantified is the word that is easy to skip, so here is the measurement.

### What the QC gate found

Three sequences out of 147,700 appear twice. Not a deduplication bug: all three are a
genomic-panel Drosophila fragment coinciding with a DeepSTARR MPRA fragment, and
DeepSTARR enhancers *are* Drosophila genomic sequence, so the two strata reach the same
locus by different routes.

```
panel  Drosophila NT_033779.5             start=13503737
mpra   deepstarr chr2L_13503710_13503958  offset=160
```

The library's own dedup caught 37 such collisions and missed 3.

### The overlap, measured

MinHash over the 59,963 fragments that can collide -- the whole MPRA stratum plus the
Drosophila part of the genomic panel:

| | |
| --- | --- |
| near-duplicate groups of size 2 or more | 1,037 |
| largest group | 6 |
| confirmed near-duplicate pairs | 2,272 |
| **fragments in a multi-member group** | **3.53 percent** |
| groups spanning the panel/MPRA boundary | 33 (74 fragments) |
| **cross-stratum near-duplicate fragments** | **0.123 percent** |

**A correction to my own reasoning.** From the three exact duplicates I inferred that
there would be "many" near-duplicates across the panel/MPRA boundary. There are not:
0.123 percent. The inference from three exact collisions to a large hidden overlap was
wrong in magnitude, and the measurement is what said so.

What is real is the 3.53 percent in near-duplicate groups overall, which is almost
entirely *within* stratum. `library_v2` tiles elements with a 40 bp step, so several 61 bp
fragments come from each 200 bp element and neighbouring tiles overlap by up to 21 bp.
Under a random split those 3.53 percent leak between training and validation; under the
similarity split they do not. **That is the number that justifies making similarity the
default**, and it is now measured rather than argued.

The largest group is 6, so there are no pathological mega-groups to make the split
lopsided -- which was the other thing worth confirming before trusting it.

### The QC gate now separates blocking from reporting

Duplicates were failing the gate. They are now a **warning** below 0.1 percent of
fragments and a **problem** above it, with the mitigation and the canary interpretation
stated in the message.

The reasoning is about how gates survive: one that blocks on something already handled
downstream gets switched off, and one that stays silent about it is worse. Reporting it
with the mitigation named is the only version that survives contact with a deadline.

**And the claim the leniency rests on is now executable.** The gate says duplicates do not
leak "because the similarity split co-locates identical sequences". That is load-bearing,
so `tests/test_landscape.py` asserts it two ways: identical sequences must land in the
same MinHash group, and no group may appear on both sides of the split. Enough
plausible-sounding reasons have been wrong today -- the force-field cache, the NFS story,
the submit quota -- that a justification used to relax a safety gate has to be a test
rather than a sentence.

---

## The first training run, read for bugs rather than capacity

*Session of 2026-11-19, continued. The user asked to source the missing parameters, make
the model predict everything, and chase high performance while following the plan.*

### What five epochs on 171,202 fragments showed

Per-base-pair median R-squared went 0.105 -> 0.654 and global 0.405 -> 0.661 in five
epochs, still climbing steeply. Undertrained, certainly. But the *worst* channels pointed
at bugs, and every stage-1 target is a deterministic function of the 61 bp input -- the
ceiling is R-squared 1 for everything except the Monte Carlo ensemble spreads -- so
anything well short of it is a defect to find, not a limit to accept.

### Bug 1: the equivariant merge reflected step channels as if they were base pairs

`C_twist`, the most basic geometry channel, trained to **R-squared 0.09**. Twist is set
by local sequence and any convolution should fit it above 0.95.

The merge reflected every per-base-pair output with a plain flip, i -> n-1-i. That is
right for a base-pair quantity. A step quantity -- the step between base pairs i and i+1
-- maps to the step between n-2-i and n-1-i, so its reflection is i -> n-2-i. The model
was therefore averaging every step channel with its *neighbour's* value, and because twist
alternates strongly from step to step, the average destroys it.

The parity tests had measured exactly this shift and used it. The model did not. The
measurement and the architecture disagreed and nothing compared them.

`src/physics/measure_rc_maps.py` measured the map of every per-base-pair channel on random
exact palindromes, where a quantity either reflects exactly or does not:

| map | channels |
| --- | --- |
| step-indexed, shift -1 | **35** at first measurement: all Family B per-bp, the six inter-step coordinates, the nearest-neighbour terms |
| groove, shift +4 / -4 | 6 that had been declared COMPOSITE, which are exactly symmetric under the groove's own offset |
| base pair, shift 0 | the rest |
| sign disagreements with declared parity | **none** -- the bug was purely in the index map |

`ChannelSpec` gained `rc_shift`; every family declares its maps by name through
`declare_index_maps`, which refuses unknown names so a typo cannot silently leave a step
channel reflected as a base pair. `tests/test_rc_maps.py` re-measures and asserts, with a
negative control that the plain flip fails on `C_twist`. The grooves are now ordinary
SYMMETRIC channels with a declared shift, so the model enforces their equivariance instead
of passing them through. Window outputs add the offset an even-width centred window
introduces (`window_index_offset`: -1 for the plan's 10, 20 and 50 bp rungs), checked
against how window targets are actually pooled.

The synthetic plan in the model tests had every shift at zero, which is why those tests
passed while the real model was wrong. It now cycles through 0, -1, +4 and -4.

### Bug 2: log-variances were merged with the mean's sign rule

For an antisymmetric channel the merged log-variance was (fv - rv)/2, pinned near zero
whatever the model predicted. The variance of an antisymmetric quantity is symmetric; only
its mean changes sign. Log-variances are now always averaged, with a regression test.

### Bug 3: constant channels drove the loss to minus 31

`F_zdna_junction_kcal`, `D_writhe_sigma0` and `D_delta_twist_sigma0` are constant across
the corpus -- the last two identically zero by construction. Their R-squared came out at
**-13,566,655**, and a Gaussian NLL rewards collapsing the variance on a constant without
bound, which is most of why the training loss had fallen to -31. The in-memory dataset
now drops constant and almost-all-NaN channels and records which.

### Bug 4: the family weights were ill-posed

The loss stacked per-family Kendall-Gal weights, exp(-s) L + s, on a per-channel Gaussian
NLL. That form assumes L is non-negative. With L < 0 its gradient in s is always positive,
so every weight drifts toward the clamp together. Read off the checkpoint: all six family
weights between 4.07 and 4.78, carrying no per-family information, still falling.

A per-channel learned variance in a Gaussian likelihood *is* Kendall and Gal's uncertainty
weighting, one task per channel, so the plan's requirement is met without the extra layer.
It is removed; families are balanced explicitly, each contributing its mean channel loss
with equal weight. The loss is now **beta-NLL** (Seitzer et al., ICLR 2022), which removes
the documented incentive for a network to inflate the variance on a hard target and stop
fitting its mean -- which is what low R-squared on the hardest families looks like from
outside. beta = 0 is kept as an ablation run.

### Bug 5: 606 seconds an epoch on a B200 for a 2M-parameter model

Each batch gathered 128 random rows from 458 gzip-compressed HDF5 files over NFS, and the
loss looped over about a thousand channels in Python with a host synchronisation each. The
corpus is under 5 GB once standardised to float16, so it now lives on the GPU, the loss is
vectorised, the two equivariant passes are one batched pass, and the transformed dataset
is cached once for every run of the scaling study.

### Target representation

- **Circular channels** -- 30 of them: backbone torsions (beta and epsilon sit within a few
  degrees of +-180 in B-DNA), epsilon-minus-zeta, phosphate azimuths, bend directions,
  pucker phase -- are trained as (sin, cos). An antisymmetric angle splits: sin
  antisymmetric, cos symmetric.
- **Strictly positive channels spanning more than two decades** are trained as log10.
  Opening probabilities run from 1e-8 to 1e-2, and the physically meaningful scale is the
  free energy, which is the logarithm. Which channels were transformed is written into
  every run, so a reported R-squared always says what scale it was computed on.

### One parity error of mine, caught while declaring maps

`G_oxoguanine_susceptibility_eV` reports only *Watson-strand* guanines, so under reverse
complement it moves to the other strand's guanines. It is STRANDED, not SYMMETRIC as
first declared.

---

## Parameters sourced, and computation run, for the provisional channels

Provisional channels: **99 -> 61 of 644**.

| family | moved to production | how |
| --- | --- | --- |
| E | `E_dCp_kcal_mol_K`, `E_dCp_total_kcal_mol_K` | the ten nearest-neighbour heat capacity changes of Rissone et al., *Nat Commun* 16, 2025, Table 1, read from the article text. Sequence-dependent now; mean -29.5 cal/mol/K per step on a test sequence against the paper's stated average of -30(10). |
| E | `E_hairpin_dG_na*` (3) | ViennaRNA's complete DNA parameter set (`dna_mathews2004.par`), with the published hairpin loop table, mismatches, special loops and salt correction, instead of a nearest-neighbour stem plus one extrapolated loop constant. |
| F | `F_zdna_step_kcal`, `F_zdna_mean_step_kcal`, `F_zdna_best_kcal_per_bp` | the complete 16-dinucleotide table from the Z-Hunt source (`zhun3.c`), which implements Ho et al. 1986 |
| F | `F_single_strandedness` | Family E's exact helix-coil partition function, replacing the AT fraction |
| L | six co-transcriptional channels | PLAN.md asks for co-transcriptional folding *proxies*; the prefix-folding proxy is the specified channel |
| C | pucker phase and amplitude (4) | Altona-Sundaralingam on the minimised structure the pipeline already builds |
| C | torsion spreads (12), BI populations (4), pucker state probabilities (4) | 16 cgNA+ ensemble snapshots, each minimised, on the plan's 10 percent ensemble subset |

### The Z-DNA table corrected two of my four values

The earlier "four sourced values" were CA/TG 1.20 and TA 2.40. Z-Hunt gives **1.40** and
**2.50**; the earlier values came from a different parameterisation. One complete table
from one source is used rather than a mixture.

The table's semantics were checked against known Z-DNA structure rather than the array
comments: 5'-CG-3' in Z-DNA has C anti and G syn, and the anti-syn row gives CG 0.66; GC in
its syn-anti register gives 0.66; CA/TG gives 1.40. Z-Hunt enumerates all 2^n
assignments; that minimum is a shortest path through a two-state chain, computed here by
dynamic programming and **verified identical to the exhaustive enumeration on 300 random
windows**. It recovers the known hierarchy: (CG)n 0.33 < (CA)n 0.70 < (TA)n 1.25 <
A-tract 2.20 kcal/mol/bp.

**A wrong inference, caught by asking rather than asserting.** The Z-Hunt source contains
`initesum = 10.0` and `sigma = 10/RT`, which I read as two 5 kcal/mol B-Z junctions and was
about to cite as corroboration. Asked to quote how they are used: `initesum` is an initial
upper bound for the search that never binds, and `sigma` sits in a separate supercoiling
partition function the code does not explain. **Z-Hunt's per-window energy has no
junction term at all.** The faithful model therefore scores fixed windows by Z-Hunt's own
energies, which removes the one unverifiable parameter from the production channels. The
junction channel itself is now provisional: Peck & Wang 1983, which measures it, is on PMC
only as page images.

### Pucker is reachable after all, on the right structure

The earlier finding was that pucker is unreachable because cgNA+ holds the sugar rigid.
True of the raw reconstruction. But the restrained minimisation holds bases and
phosphates and leaves the sugar free, so the ring relaxes into what the sequence-dependent
backbone demands. Measured: every sugar C2'-endo, amplitude about 34 degrees against B-DNA
35-40, and phase separating by sequence -- A-tracts at **139.8** degrees against GC-rich DNA
at **158.0**. Force-field pucker under cgNA+ restraints is the same class of ground truth
as the MD libraries the plan names.

### ViennaRNA's parameters are global process state

Loading the DNA parameter set and not restoring it would silently switch Family L's RNA
folding to DNA energies, in the same process, with nothing raised. The DNA fold runs inside
a context manager that restores Turner 2004 in a `finally`; verified bit-identical RNA
energies before and after (-23.900 / -23.900).

### What stays provisional, each for a stated reason

61 channels: salt and temperature geometry (no parameter set exists), the Hoogsteen proxy
(a genuine second basin a Gaussian cannot contain), nucleosome contact energetics, the
cruciform four-way junction term, rates (an attempt frequency is not equilibrium
thermodynamics), and the channels whose ground truth is experimental -- quadruplex
stability and topology, CPD weighting, cyclizability. For those last, the plan's design is
an analytic prior at stage 1 and delta-learning on the measurement at stage 2, so fitting
the measurement into the stage-1 channel would blur the two stages rather than sharpen
either.

---

## A minimiser trap in 20 percent of the backbone, found through an implausible spread

### How it surfaced

The relaxed-ensemble pilot gave torsion spreads across 16 snapshots of 9-19 degrees for
beta, delta, epsilon and zeta -- the range molecular dynamics reports -- and **about 60
degrees for alpha and gamma**. That pointed back at the ground state, and the ground state
in the v1 shards shows the same thing: over 134,640 positions, **20.1 percent of alpha in
g+ and 20.1 percent of gamma in g-**, at the same nucleotides (correlation 0.999), evenly
along the fragment, so not an end effect. Canonical B-DNA has alpha in g- and gamma in
g+.

### Trap or preference: measured, not argued

The concerted alpha/gamma change is a crankshaft: it moves O5' and C5' while leaving the
phosphate and the base -- the restrained atoms -- where they are, so the minimiser could
settle in either basin. Same structure, same restraints, same force field, three
protocols (`scratchpad` experiment, recorded here):

| duplex | plain, 400 iterations | plain, to convergence | canonical start, then released |
| --- | --- | --- | --- |
| mixed | -52,235.4 kJ/mol, 14.2% non-canonical | identical | **-52,577.8**, 0% |
| random | -50,237.8 kJ/mol, 20.0% non-canonical | identical | **-50,763.6**, 0% |

The plain run had converged, so this is not an iteration cap. The canonical start holds
alpha and gamma within +-29 degrees of g-/g+ with a flat-bottomed restraint, then switches
it off and minimises again; energies are compared without the restraint term. It reaches
a minimum of the same energy function lower by 342 and 526 kJ/mol. **The non-canonical
states were traps.**

### The fix, and what it reaches

`minimize.restrained_minimise` now runs both starts and keeps the lower final energy,
recording both energies and which won. Through the production path: canonical kept on
both test duplexes, at 367 and 532 kJ/mol lower, no non-canonical alpha/gamma, restrained
atoms within 0.29 angstrom RMSD of their cgNA+ positions. The generator writes the audit
for every fragment (`min_start_kept`, `min_energy_kJ_mol`), so the protocol's effect is
measured on the corpus, not inferred from two duplexes.

The minimised structure feeds the Family C torsions and pucker, Family G accessibility and
the electrostatics, so **every v1 shard carries this artifact** in those channels; v2
regeneration replaces them. Ensemble snapshots use the canonical start only -- the plain
start is the one that traps -- so the ensemble spreads are fluctuations within the
dominant basin, and the description says so: rare crankshaft transitions are not sampled
by minimisation at all.

---

## The global readout could not represent an extreme or a location

The global head was a low-rank *linear* map of the mean-pooled trunk features. That
represents anything additive along the sequence and nothing that is a maximum, a minimum
or a position. After removing the constant channels, the worst global channels of the
first run were exactly those: `D_bubble_position_bp` 0.002, `F_g4_best_tetrads` 0.003,
`F_zdna_centre_offset_bp` 0.003, `L_p_min_hairpin_size` 0.013, `F_g4_best_loop_max`.

`GlobalReadout` pools mean, max and four learned attention queries -- attention values
carry a centred position u and u squared -- and passes them through a two-layer MLP before
the low-rank head. u maps to -u under reverse complement as a centre offset does, so the
existing merge keeps every global channel's parity; tested, with a negative control that
fails when the merge is switched off.

On a synthetic location task (regress where a planted GGGG sits), the same trunk reaches
a normalised MSE of **0.0002** with the new readout against **0.331** with the linear one.
The linear readout is kept as the baseline; `v2_S_gmlp` and `v2_M_gmlp` are the
ablation against `v2_S_base` and `v2_M`.

---

## Literature pass on the remaining provisional channels

Searched for published parameters behind the provisional channels whose reason was a
missing number rather than a missing physical model.

**Resolved: the B-Z junction energy.** Zhabinskaya & Benham, *PLoS Comput Biol* 7:e1001051
(2011), PMC3024258, Methods, in open text: "a nucleation energy for each run of Z-DNA is
required to form the junctions between the B-form and the Z-form. This has been measured
to be approximately 5.0 kcal/mol/junction". `F_zdna_junction_kcal` is production with that
source. It is constant, so the constant-channel rule still drops it as a target.

**Found, and recorded as an alternative rather than adopted: a second Z-DNA table.** The
same paper's Table 1 is a different parameterisation from Z-Hunt's (CA/TG 1.3 / 4.6 / 4.5
against Z-Hunt's 1.40 / 5.20; AA/TT 3.9 against 4.40; ZZ junctions 4.0 to 7.4). The
project keeps the single complete Z-Hunt table rather than mixing sources.

**Stays provisional, now on a published statement: Z-DNA salt scaling.** The same paper:
"no quantitative information is currently available regarding how these free energies
partition between entropy and enthalpy, nor about their ionic strength dependences."

**Stays provisional: i-motif pH.** No closed-form model of the transitional pH from C-tract
and loop lengths was found -- qualitative trends (longer C-tracts stabilise, longer loops
destabilise, loop sequence can dominate) and machine-learned or MD-based predictors only.

**Stays provisional, with the model now identified: cruciform extrusion.** Zhabinskaya &
Benham, *Nucleic Acids Res* (2013), doi 10.1093/nar/gkt733. The equations are images on
PMC; read directly from them:

    G_l  = n_at b_at + n_cg b_cg + 2k ln(n_l)                (eq. 2)
    G_ex = dH - T dS + G_l - 4 b_at - 2k ln(4)               (eq. 3)
    dH = 192.5 kcal/mol, dS = 0.565 kcal/mol/K, k = 2.44

with dH and dS from 2D-gel extrusion of an imperfect inverted repeat with a 4 A/T bp loop
(Benham, Savitt & Bauer, *J Mol Biol* 316:563, 2002): 17.35 kcal/mol at 310 K. Three things
keep it provisional. (1) b_at and b_cg are Benham's copolymeric energies at 0.01 M and
37 C (Benham, *J Mol Biol* 225:835, 1992); no open text found states their values (the NRC
1995 chapter and the 2004 *Genome Res* SIDD paper both defer to the 1992 paper), and the
salt dependence of the initiation term is not parameterised at all. (2) k = 2.44 is cited
to SantaLucia & Hicks 2004, whose loop extrapolation is 2.44 RT ln(n/x), but eq. 2 prints
2k ln(n_l) with no RT, so its units are ambiguous. (3) Benham's superhelical model is for
kilobase domains; a 61 bp fragment treated as its own domain holds a linking difference of
-0.35 turns at sigma = -0.06, so extruding a 30 bp cruciform overshoots to positive
supercoiling and the "relief" term becomes a large cost. That is why the current cruciform
channels read 100-130 kcal/mol, against 17 kcal/mol for Benham's measured extrusion, and
it is a statement about the definition, not the parameters: a cruciform channel for a
short fragment needs a stated embedding (the fragment inside a domain of given size and
superhelicity), which the plan does not yet specify. Excluded from training as provisional.

---

## An hour of 191 cores lost to a frozen snapshot that could not run

The v2 regeneration was launched from a frozen copy of `src/`, so that edits during a
multi-hour array could not give shards written by different code. The copy was verified
-- sha256 of every source file against the working tree -- but never *run*. Family B
reads `results/qc_tetramer_gfn2.json` by a path relative to its own file, which in the
copy pointed at a `results/` that did not exist, so **every fragment of every task
failed**. The generator caught each exception, counted it, and moved on; the progress
lines said "failures 100" after 100 fragments and nothing else. The array was cancelled
after about 50 minutes, roughly 150 CPU-hours, with no shard written.

Three changes, each closing one part of that path:

1. `scripts/freeze_code.sh` copies `src/` *and* the runtime data files read from outside
   it, writes the manifest over both, and then runs the generator from the snapshot on two
   fragments, refusing to publish the snapshot unless both succeed. Its first use caught
   a second bug the same day (a provenance edit referencing an out-of-scope variable).
2. The generator prints the first errors to stderr as they happen, and a task whose first
   five fragments all fail raises instead of writing a shard of NaN.
3. The array monitor reports any fragment error the moment it appears in a task's stderr,
   not a shard count at the end.

### Two settings changed for the relaunch

**Ensemble size 200 -> 3200.** The groove-width fluctuation channels scored R-squared
0.38-0.55 in the best model, the worst Family C channels. Measured on one fragment over
three seeds: at 200 samples the seed-to-seed scatter of `C_minor_width_A_sd` is 0.066
angstrom against an along-sequence spread of 0.079 -- a ratio of 0.84, so the R-squared
ceiling set by sampling noise alone is about 0.3. The model was already at the ceiling;
the targets were the limit. At 800 samples the ratio is 0.41-0.54; at 3200 it extrapolates
to about 0.2-0.27, a ceiling near 0.95. Cost: about 2 s per fragment on a compute node,
roughly 110 CPU-hours over the corpus, against the 860 the regeneration costs anyway.
Only Family C consumes the ensemble size (Family H runs with none).

**Provenance in every shard:** the ensemble size, ensemble fraction, seed, whether the
structure was minimised, and the path of the frozen code that wrote it.

---

## The first design ablations, on the v1 targets

All on the similarity split, validation side, read by `src/eval/compare_runs.py` from each
run's own history. The v1 targets carry the alpha/gamma trap artifacts and 200-sample
ensemble spreads, so absolute numbers will move on v2; the comparisons are like for like.

| comparison, matched epochs | per-bp median R2 | global median R2 |
| --- | --- | --- |
| S (2.0M), beta 0.5, similarity split, epoch 99 | 0.9883 | 0.9803 |
| same, **random split**, epoch 99 | 0.9883 | 0.9810 |
| same, **beta 0**, epoch 99 | 0.9733 | 0.9340 |

**Similarity leakage is negligible on this corpus.** The random split scores within 0.001
of the similarity split at every epoch from 9 to 99, which matches the measured overlap:
3.7 percent of fragments have any near-duplicate. This is the plan's "overlap quantified
and reported" for the internal split.

**beta-NLL is worth 0.046 global R-squared**, and it is the difference between learning
the location channels and not: at beta 0 the Z-tract centre offset scores 0.010 and the
bubble position 0.008 -- the network inflates the variance on hard targets and stops
fitting their means, which is exactly the failure beta-NLL was adopted to prevent.

**The rank-64 head was a bottleneck at small scale.** Rank 256 at the same width lifts
the share of per-bp channels above R-squared 0.99 from 0.44 to 0.66.

**What limits the best model (M_attn, 14.8M, epoch 28):** families B, C, D, E, F are at
per-bp medians of 0.997-0.9999. The rest is concentrated:

- Family L (RNA secondary structure): per-bp median 0.52, global 0.89. Positional
  entropy 0.31-0.35, unpaired probability 0.70, ensemble diversity and hairpin sizes
  0.2-0.27. A fold is a property of which distant positions pair.
- Ground-state backbone torsions 0.73-0.77: the v1 targets with the crankshaft traps.
- Groove-width spreads 0.38-0.55: at the sampling-noise ceiling (above).
- Five global channels near 0 whose modal value covers 99.0-99.5 percent of fragments
  (G4 tetrads, R-loop score, both G-quadruplex gains, triplet repeats): rare-event
  channels, for which R-squared is the wrong metric.

### Two architecture changes, each with its own ablation

- `GlobalReadout` (mean, max and attention pools through an MLP) -- `S_gmlp`, `M_gmlp`.
- `PairBias`: a dilated 2-D convolution over the base-pair map (base identities and
  signed separation) projected to one additive bias per attention head, aimed at Family
  L. Zero-initialised, so it starts as the baseline exactly -- `M_attn_pair`,
  `M_attn_pair_gmlp`.

### Two bugs caught before they could cost a run

1. **PyTorch's fused inference path returned NaN for a per-head 3-D additive mask** in
   eval mode, while the training path was correct -- it would have corrupted exactly the
   evaluation numbers. The biased attention is now computed explicitly from each layer's
   own weights, tested equal to the layer's output to 1e-5 at zero bias.
2. **The Slurm scripts pass the head rank in an environment variable named RANK**, which
   is also torchrun's process-rank variable. The new data-parallel code read it in
   single-process runs, which would have made rank 0 believe it was rank 64 and skip
   every evaluation, history line and checkpoint without an error. Process-rank variables
   are now read only when WORLD_SIZE > 1, and the multi-GPU script names its head rank
   HRANK. No job started in the window between the two edits.

Data parallelism (`torchrun`, one process per GPU, the whole dataset on each GPU, rank 0
evaluates and writes) is for the 200M and 1B rungs: extrapolating from L's 84 s/epoch at
49M on one GPU, 1B would need about a day for 100 epochs on a single device.

---

## The models are at the sampling-noise ceiling on every Monte Carlo channel

`src/eval/noise_ceiling.py` recomputed the landscape for 300 fragments under two seeds
(`results/noise_ceiling_v1.json`). Twelve per-bp and two global channels move with the
seed -- the ensemble spreads of groove widths and depths, phosphate radius, azimuth and
axial offset, and axis bend direction and curvature; every other channel is deterministic.
The ceiling is 1 - noise variance / corpus variance, on the training scale.

| channel (v1 targets, 200 samples) | ceiling | S 2.0M | M 10.1M | M_attn 14.8M | L 49.1M |
| --- | --- | --- | --- | --- | --- |
| C_minor_width_A_sd | 0.400 | 84% | 94% | 95% | 97% |
| C_major_width_A_sd | 0.562 | 92% | 98% | 98% | 99% |
| C_phosphate_radius_A_W_sd | 0.811 | 93% | 97% | 98% | 99% |
| C_major_depth_A_sd | 0.850 | 91% | 95% | 96% | 96% |
| C_axis_curvature_sd | 0.929 | 91% | 97% | 97% | 98% |
| C_minor_depth_A_sd | 0.945 | 97% | 99% | 99% | 99% |

(Percentages are R-squared as a share of the ceiling; all twelve are in
`results/noise_ceiling_v1.json` and reproduce from the runs' `best_r2.npz`.) From 10M
parameters up, every Monte Carlo channel is at 94-99 percent of what its targets allow:
the worst Family C channels were limited by target noise, not by the model, which is why
the v2 targets use 3200 samples. The two global mean-spread channels score 101-105 percent
of their estimated ceilings; a ceiling estimated from 300 pairs carries a few percent of
uncertainty itself, and these are reported with that caveat.

---

## Effective rank of the channel stack, first measurement (v1 targets)

`src/eval/effective_rank.py`, correlation matrix of the standardised targets, channels
finite on at least 90 percent of samples (`results/effective_rank_v1.json`):

| resolution | channels | participation ratio | components for 95% | for 99% |
| --- | --- | --- | --- | --- |
| per base pair (341,016 fragment positions) | 203 | 10.8 | 41 | 69 |
| whole sequence (191,344 fragments) | 238 | 5.6 | 41 | 70 |

By family, per-bp participation ratios: B 2.1, C 13.3, E 1.7, L 3.5; global: B 3.4, C 5.2,
D 3.7, E 1.8, F 8.4, L 3.0. The plan's working expectation was "an effective rank in the
tens", and that is what this is. Circular channels count as two coordinates (sin, cos).

**It explains the head-rank ablation.** Per-channel R-squared of 0.99 needs the head to
represent more than 99 percent of each channel's variance, and the per-bp stack needs 69
linear components for 99 percent overall; a rank-64 head therefore had to cap the share of
channels above 0.99, which is what the rank ablation showed (0.44 at rank 64, 0.66 at
full rank). The bottleneck sweep on v2 measures the saturating rank directly, which is the
plan's independent estimate of the same quantity.

---

## The design for the v2 experiments, and the experiments

Chosen from the v1 ablations at matched epochs (validation side, similarity split):

| change | where it acts | measured |
| --- | --- | --- |
| attention, 4 layers | non-local channels | Family L per-bp 0.378 -> 0.547 (M, epoch 57) |
| pair-map bias on attention | which positions can pair | Family L per-bp 0.544 -> 0.674, global 0.9920 -> 0.9946 (epoch 46) |
| attention/MLP global readout | extremes and locations | global median +0.005 to +0.011 at S and M |
| head rank at least the 99%-variance dimension | per-bp ceiling | channels above 0.99: 0.44 -> 0.66 (S) |
| beta-NLL 0.5 over beta 0 | variance inflation on hard targets | global +0.046; location channels 0.01 -> 0.8 |

A 49M convolution-only model beats the 15M attention model on per-bp channels (0.9987
against 0.9981) and loses badly on Family L (0.417 against 0.542), so the larger rungs keep
attention rather than trading it for width.

The v2 experiment set (`scripts/submit_v2_experiments.sh`), every run with a held-out test
side of 5 percent, similarity-grouped, that no training or selection decision sees:

1. **The scaling ladder at 2, 10, 50, 200 and 1,000 million parameters**, widths chosen on
   the real v2 channel plan (`src/model/size_ladder.py`), data-parallel at 200M (2 GPUs)
   and 1B (4 GPUs). Measured cost on one H200: 213M at 3.6 min/epoch, 876M at 12.4
   min/epoch and 124 GiB.
2. **The bottleneck sweep run to full width** (risk table) at the 10M trunk, rank 4 to
   full: the smallest saturating rank is the plan's independent estimate of intrinsic
   dimensionality, to set against the effective-rank spectrum.
3. **Universality at each rung up to 200M**: a held-out-kingdom encoder (bacteria and
   plants only, every other species scored zero-shot) and a human-only encoder, both
   scored per species on the same test side as the multi-clade ladder. The plan's
   prediction, stated in advance there and repeated here before any of these runs: per-
   channel accuracy improves with capacity while the universality gap widens.

---

## External validation, first results: measurements the solver never saw

### Array Melt: Family E's DNA fold against 27,732 measured hairpins

`src/eval/validate_array_melt.py` -> `results/validation_array_melt.json`. Ke et al. 2025
(*Nat Commun*, doi 10.1038/s41467-025-60455-4) measured two-state hairpin stabilities in
50 mM Na-HEPES pH 8.0 with 25 mM NaCl (bioRxiv text); total sodium is not exactly
determinable from that, so the comparison is reported in sodium-independent terms and at
two sodium levels. Scored: `thermodynamics.dna_fold_mfe` -- ViennaRNA with the Mathews
2004 DNA parameters, the function behind the hairpin channels -- at 37 C. Nothing is
fitted except one constant offset, reported as such.

| condition | Pearson r | Spearman | MAE | constant offset | MAE after offset | RMSE after offset |
| --- | --- | --- | --- | --- | --- | --- |
| 1 M Na | **0.684** | 0.696 | 2.02 | +1.94 | **0.71** | 0.94 |
| 75 mM Na | 0.638 | 0.642 | 0.98 | +0.71 | 0.74 | 0.99 |

By construct class at 1 M: Watson-Crick stems r = **0.893** (MAE 0.63), triloops 0.683,
mismatches 0.660, tetraloops 0.558, bulges 0.523. The physics is right about stems and
weakest on bulges and loop-sequence effects -- which are what the Array Melt authors'
newer parameters (dna24) improve; their own models, fitted to this dataset, reach MAE
0.18-0.56 kcal/mol. **dna24 is deliberately not adopted for the targets**: Array Melt
would then stop being an independent check of the physics, which is what the plan needs
external validation to be.

### ORChID: a null at aligned positions for the groove channels

`src/eval/validate_orchid.py` -> `results/validation_orchid_v1.json`: each channel
averaged over every corpus occurrence of each tetramer (64 shards), strand-correct, then
correlated with the measured cleavage at the same four positions. Primary metric fixed in
advance: aligned positions, no offset search.

| channel (v1) | r, all positions | by position 1-4 |
| --- | --- | --- |
| C_minor_width_A | +0.012 | +0.03, +0.10, +0.02, -0.38 |
| C_minor_depth_A | -0.007 | +0.01, -0.05, +0.05, +0.02 |
| C_major_width_A | -0.023 | -0.05, -0.09, -0.03, +0.35 |

A null at aligned positions, reported as such. It is a weak test of these channels: a
groove width is a base-pair quantity set by phosphates across about 10 bp, while ORChID
scores one nucleotide on one strand, so most of the width's dependence lies outside the
tetramer and averages out. The direct test is Family G's per-strand C4'/C5' accessibility
(the atoms the radical abstracts from), which exists only in the v2 shards.

---

## The home quota filled mid-regeneration, and two black-hole nodes

**What happened.** The project's "scratch" (`PANGEA_SCRATCH=/home/bjc75/scratch/pangea`)
is a directory inside the home directory, not the group scratch fileset, so everything
the project writes counts against the personal home quota of 125 GiB (93 percent used at
the start of the day). Around 23:00 it filled. From then on every write under /home
failed with `OSError: [Errno 122] Disk quota exceeded` -- including the error messages
themselves, which is why the first failed tasks had empty `.err` files and why a
truncated cache came out at exactly 750 MiB. Casualties: about 130 v2 array tasks that
computed for up to 70 minutes and could not write their shards, the ladder-sizing job,
and the last four v1 ablation runs (L, M_gmlp, M_attn_pair, M_attn_pair_gmlp), which
died writing checkpoints. Their histories up to that point are intact and their
comparisons are already recorded above; they are not rerun.

Separately, two nodes (a1128u41n01, a1124u28n01) failed every task within one to two
seconds with exit code 0:53 and no log file: 36 of the early failures. A node that fails
fast attracts more tasks, the classic black hole.

**Fixes.**
- Freed 13 GB by purging the pip download cache and 0.9 GB of the day's smoke outputs;
  writes were confirmed with a 1 GB probe before anything was released.
- v2 targets, dataset caches and all v3 run directories now live on the group scratch
  fileset (`/nfs/roberts/scratch/pi_skr2/bjc75/pangea`, 10 TB quota), reached through
  links from the old paths (`targets_v2`, `runs_v3`), so a 16 GB 1B checkpoint can never
  land in home. That fileset is purged after 30 days without access and is not backed
  up, so final artifacts must be copied to backed-up storage when the work concludes.
- The two nodes are excluded from every array; failed shards are recomputed by fill-in
  arrays from the same frozen snapshot. A second fill-in initially duplicated 58 shards
  of the first -- two tasks writing one `.partial` file could corrupt it -- and the
  duplicates were cancelled before any could reach its write.
- Checkpoints are now written atomically (temporary file, then rename), so a kill during
  a write leaves the previous checkpoint intact; the best score and the history are
  reconciled on resume.

### Loop-seq: Family D's cyclizability prior is a weak positive

`src/eval/validate_loopseq.py` -> `results/validation_loopseq.json`: Basu et al. 2021
intrinsic cyclizability C0, the random library (no genomic or nucleosomal selection),
4,000 sequences, deterministic subsample; nothing fitted.

| channel | Pearson r | Spearman |
| --- | --- | --- |
| D_cyclizability_L63 (and L73, L84, L94: identical) | +0.170 | +0.165 |
| D_net_bend_deg | +0.190 | +0.169 |
| D_curvature_shortening | +0.187 | +0.177 |
| D_persistence_length_nm | -0.022 | -0.044 |

The cgNA+ intrinsic curvature carries a real but small share of what loop-seq measures
(r squared about 3-4 percent), and the flexibility term carries none. This is why the
plan marks cyclizability as an analytic prior supervised on loop-seq at the delta-
learning stage: most of the signal has to come from there. The four loop lengths give
identical rankings because the Shimada-Yamakawa estimate is monotone in the same net
bend at every length, so they are one channel in four columns -- noted for the
channel-count-inflation risk, which the effective-rank analysis already accounts for.

### CPD-seq: deliberately not used as a validation of the current channels

The Family G photoproduct channels are exact dipyrimidine-class indicators with a flat
methylation factor; they carry no per-class or per-context propensity. Against naked-DNA
CPD-seq they could only show that dimers form at dipyrimidines, which is chemistry, not a
test of the model. The class and context propensities CPD-seq measures are what the
plan's delta-learning stage fits, so CPD-seq is kept for that.

---

## Methylation deltas at the fragment ends, and an interior R-squared

A per-channel screen of the first v2 shards (`src/eval/screen_targets.py`) flagged
`H_5mC_dtwist_deg` reaching -35.2 degrees and `H_5hmC_dtwist_deg` -38.8 -- against a bulk
of +-5 (0.1st to 99.9th percentile: -5.27 to +2.99). Every value beyond +-8 degrees sits at
the two terminal steps (positions 1-2 and 57-58), in CpG contexts such as `...CGGG` at the
3' end. The delta is clean by construction -- methylated and unmethylated sequence run
through the same methylated cgNA+ parameter set -- so these are cgNA+'s own end
parameterisation of a methylated terminal step in a free-ended 61-mer. They are 0.02
percent of values and carry about half of the channel's variance.

**The targets are not edited.** They are the faithful output of the defined computation,
and trimming them after seeing them would be choosing the data. Instead the final
evaluation now reports, for every per-bp and window channel, R-squared over the interior
positions (5 bp excluded at each end) beside the primary all-position number, per family,
so a headline is never silently dominated by end parameterisation -- for any family, not
just H. The other flags of the screen are expected: accessibility tails at the
more-exposed terminal nucleotides, circular means wrapping at +-180 degrees (trained as
sin/cos), the documented rare-event channels, and provisional channels (cruciform up to
430 kcal/mol, the Hoogsteen proxy) that are excluded from training.

---

## Conditions are channels at stage 1, not encoder inputs -- yet

PLAN.md describes a condition-conditioned encoder: ionic strength, temperature,
superhelicity, pH and methylation as inputs, so a locus can be queried under altered
conditions. The FiLM layers that take that condition vector are implemented and tested
(zero condition is exactly the identity), but the stage-1 training feeds no condition:
every condition-dependent quantity is enumerated as its own channel (the hairpin at 0.01,
0.15 and 1 M sodium; opening probability across temperatures and superhelical densities),
which is what the targets as generated support. Turning those enumerations into a
condition input -- the plan's anchoring and delta-learning scheme -- is a separate,
later step: it needs the targets reorganised as (fragment, condition) pairs and a
held-out-condition evaluation, and nothing in the scaling ladder depends on it. Stated
here so no result from the v3 runs is read as a test of the conditioning.

### The two-start minimisation, measured on half the corpus

Read from the per-fragment audit in 515 v2 shards (96,305 fragments): **0 fragment
failures**; ensemble subset 9.92 percent against 10 (standard error 0.1; the 7.8 percent
on the first 11 shards was the early-finisher bias, shards with fewer ensemble fragments
finishing first). The canonical start was kept for **99.927 percent** of fragments, at a
median energy **513 kJ/mol** below the plain start (5th-95th percentile 278-705). The
plain start found the lower minimum for 0.073 percent -- about 70 fragments -- which is
the case for running both starts rather than imposing the canonical one.

### ORChID on the full v2 corpus: the atom the radical attacks is what tracks cleavage

`results/validation_orchid_v2.json`, all 1024 v2 shards; agrees with the 11-shard preview
to two decimals.

| channel | pooled r (primary) | position 2 | position 3 |
| --- | --- | --- | --- |
| G_sasa_sugar W / C | +0.125 / +0.128 | +0.354 / +0.361 | +0.290 / +0.289 |
| G_probe_hydroxyl_radical W / C (= G_radical_target) | +0.089 / +0.110 | +0.346 / +0.393 | +0.230 / +0.225 |
| G_sasa_total W / C | -0.039 / -0.039 | -0.122 / -0.123 | -0.038 / -0.037 |
| C_minor_width_A | +0.010 | +0.096 | +0.022 |

The pooled number fixed in advance is weak, and it is reported as the primary result.
The structure inside it is the informative part: accessibility of the deoxyribose --
where the radical abstracts its hydrogen -- tracks measured cleavage at the central
positions, while total accessibility of the nucleotide does not. That is the chemical
specificity the mechanism predicts.

### Two scheduling fixes at the hand-off to training

The finalize job ran as designed: QC 0 problems over 1024 shards, the v2 cache built
once (191,344 fragments; 286 per-bp and 323 global columns, as sized), 28 runs submitted.
The 200M and 1B jobs then sat in `BadConstraints`: their constraint names only 80-180 GB
GPUs, none of which exist in the first partition of their list, and Slurm held them
rather than trying the others. Their partition lists were corrected in place and the
submission script fixed for reruns.

### Effective rank on the v2 targets: two new families, almost no new dimensions

`results/effective_rank_v2.json` (same method as v1; ensemble-subset channels excluded by
the 90 percent coverage rule):

| | v1 (B C D E F L) | v2 (+ G, H) |
| --- | --- | --- |
| per-bp channels | 203 | 265 |
| per-bp participation ratio | 10.8 | 11.9 |
| per-bp components for 95 / 99 percent | 41 / 69 | 42 / 72 |
| global channels | 238 | 306 |
| global participation ratio | 5.6 | 5.3 |
| global components for 95 / 99 percent | 41 / 70 | 49 / 85 |

Adding solvation and chemical reactivity (G) and modification response (H) -- 62 new
per-bp channels -- moved the 95 percent component count by one. Most of what they measure
is already carried by geometry and thermodynamics; by family, G's own per-bp participation
ratio is 6.1 and H's 4.4. This is the plan's channel-count-inflation risk answered by
measurement: 571 training channels, effective ranks in the tens, reported next to the
count as the risk table requires.

### Walltimes sized for the fastest GPUs; continuation links added

The first v3 run (2M) trained cleanly -- epoch 1 per-bp median 0.205, every family
learning -- but at 93 s/epoch on an A40, against 12 s for the v1 small model on a B200,
because the scheduler places runs on whatever GPU frees first. At that rate 100 epochs
exceed the 2 h walltime the submission gave the 2M rung, and the same holds for other
rungs on slower cards. Every run checkpoints each epoch (every 30 minutes at 200M and
1B, atomically) and resumes, so each chain received extra links (`scripts/continue_v3.sh`,
`results/v3_continuations.txt`): afterany on its current last link, parameters identical
to the original submission -- which they must be, since a checkpoint loads only into the
same architecture. A link that finds its run complete writes any missing final metrics
and exits.

### The 3200-sample targets lift the noise ceiling as extrapolated

`results/noise_ceiling_v2.json` (300 fragments, two seeds, 3200 samples, v2 transforms):

| channel | v1 ceiling (200 samples) | v2 ceiling (3200) | extrapolated before the run |
| --- | --- | --- | --- |
| C_minor_width_A_sd | 0.400 | 0.915 | 0.914 |
| C_major_width_A_sd | 0.562 | 0.954 | 0.954 |
| ten other per-bp spreads | 0.81-0.95 | 0.986-0.996 | |
| two global mean spreads | 0.76-0.80 | 0.985 | |

The extrapolation was made from 1/n scaling of the sampling variance before the v2 targets
existed and matches to three decimals. The worst Family C channels go from a ceiling of
0.4 to above 0.9: the model can now be measured on them rather than on their noise.

---

## First universality data point: no held-out-kingdom gap at 2M (preliminary)

`runs_v3/v3_Ukingdom_2M/final_metrics.json`: trained on bacteria and plants only
(panel windows and plant MPRA), best epoch 99 of 100, every species scored on the shared
held-out test side.

| species | in training? | per-bp median R2 | global median R2 |
| --- | --- | --- | --- |
| Zea mays | yes | 0.9909 | 0.9582 |
| Escherichia coli | yes | 0.9900 | 0.9416 |
| Arabidopsis thaliana | yes | 0.9897 | 0.9481 |
| Thermus thermophilus | yes | 0.9888 | 0.9135 |
| Homo sapiens | **no** | 0.9892 | 0.9496 |
| Mus musculus | no | 0.9902 | 0.9456 |
| Methanocaldococcus jannaschii | no | 0.9909 | 0.9390 |
| Plasmodium falciparum (19% GC) | no | 0.9895 | 0.9124 |

All seven in-filter species: per-bp 0.9888-0.9909, global 0.913-0.962; all nine held-out
species: per-bp 0.9892-0.9910, global 0.912-0.957. At 2M parameters a model that never saw
an animal, fungus, protist or archaeon predicts their physics as well as the kingdoms it
trained on. One size is not a trend: PLAN.md's stated prediction is that this gap opens
with capacity, and the 10M, 50M and 200M encoders test it.

**The per-species metrics also contain the designed libraries**, because the species join
labels them by their library name (`hexamer:polyA` and others). Every fragment in such a
library shares its flanks, so whole-sequence channels are nearly constant within it and
within-library global R-squared is meaningless (-7.9 to -32 here); per-bp is unaffected
(0.974-0.983). The universality report (`src/eval/scaling_report.py`) compares the 16 real
species only.

### The human-only encoder: the gap follows base composition, not phylogeny (2M, preliminary)

`runs_v3/v3_Uhuman_2M/final_metrics.json`: trained on 25,166 human fragments only; human
test fragments per-bp 0.9847, global 0.9079. Held-out species range from per-bp 0.9851 /
global 0.9113 (mouse) down to 0.9764 / 0.8590 (*S. coelicolor*, 72 percent GC) and 0.9778
/ 0.8466 (*M. tuberculosis*). Correlated against the distance of each species' measured
genome GC (RESULTS.md section 1) from human's 40.9 percent:

| encoder | metric | Spearman rho vs GC distance | p | n |
| --- | --- | --- | --- | --- |
| human-only | per-bp median | -0.789 | 0.0008 | 14 |
| human-only | global median | -0.960 | 5e-8 | 14 |
| bacteria + plant | per-bp median | +0.110 | 0.78 | 9 |
| bacteria + plant | global median | -0.895 | 0.001 | 9 |

(For the bacteria-and-plant encoder the distance is from its trained range, 35.9 to 72.0
percent; five of its nine held-out species fall inside it.) At this size, how well the
encoder transfers to an unseen species is governed by whether that species' composition
lies inside the composition range it trained on: the archaeon at 31 percent GC loses less
than the GC-rich bacteria do from a human-only model, which a phylogenetic account would
not predict. That is the reason the plan's panel spans 19 to 72 percent GC, measured. The
capacity trend -- the plan's stated prediction is that the gap widens with size -- comes
from the 10M, 50M and 200M encoders.

### The universality gap: metrics fixed before the 50M and 200M encoders report

Human-only encoders at 2M and 10M (`runs_v3/v3_Uhuman_{2M,10M}`): every species improves
with capacity, held-out ones included (*M. tuberculosis* global 0.847 -> 0.915,
*Plasmodium* 0.870 -> 0.929, per-bp all at 0.988-0.993 at 10M). Whether "universality
degrades with capacity" then depends on the measure, and the measures disagree:

| gap measure (human-only encoder) | 2M | 10M |
| --- | --- | --- |
| per-bp: human median minus held-out median | +0.0013 | +0.0005 |
| global: human median minus held-out median | +0.0120 | +0.0150 |
| global: human minus worst held-out species | +0.0613 | +0.0463 |
| global error ratio, held-out median (1 - R2) / human (1 - R2) | 1.13 | 1.37 |
| Spearman rho, global R2 vs GC distance | -0.960 | -0.820 |

**Fixed now, before the larger encoders report:** the universality result is all five of
these, for each encoder (held-out-kingdom and human-only) at every size, plus the
held-out species against the multi-clade encoder of the same size. No single one is the
headline, so the conclusion cannot be selected by choice of metric. At 2M -> 10M the
absolute gaps shrink or hold while the relative error ratio grows -- the held-out species
improve, but less than the trained one.

The table above was computed by hand over the 15 panel species. The canonical numbers
come from `src/eval/scaling_report.py` (`results/universality_v2.json`), which also
counts *Sorghum bicolor* (an MPRA species, not in the panel), so they differ slightly:
human-only global gap +0.0108 / +0.0140 and error ratio 1.12 / 1.34 at 2M / 10M. The
held-out-kingdom encoder shows essentially no gap at either size (global gap +0.0013 /
+0.0009, error ratio 1.02 / 1.03, worst species +0.029 / +0.027).

---

## The held-out-kingdom series, 2M to 200M: universality holds; the prediction is not borne out

`results/universality_v2.json` (canonical, 16 real species), from
`runs_v3/v3_Ukingdom_{2M,10M,50M,200M}` -- trained on bacteria and plants, every species
scored on the shared test side, all four at 100 epochs:

| size | per-bp error, trained / held-out | global error, trained / held-out | global error ratio | worst-species gap | Spearman vs GC distance |
| --- | --- | --- | --- | --- | --- |
| 2M | 0.0103 / 0.0100 | 0.0584 / 0.0597 | 1.02 | 0.029 | -0.89 |
| 10M | 0.0048 / 0.0046 | 0.0298 / 0.0307 | 1.03 | 0.027 | -0.86 |
| 50M | 0.0026 / 0.0025 | 0.0188 / 0.0229 | 1.22 | 0.030 | -0.77 |
| 200M | 0.0019 / 0.0017 | 0.0189 / 0.0210 | 1.11 | 0.026 | -0.62 |

(errors are 1 - median R2 over species; all five pre-registered measures in the JSON.)

**PLAN.md stated in advance that accuracy would improve while universality degraded with
capacity. For this encoder it does not.** On per-base-pair channels the held-out kingdoms
match or slightly beat the trained ones at every size. On whole-sequence channels the
relative gap rose at 50M and fell back at 200M, where the trained kingdoms' error
plateaued (0.0188 -> 0.0189) while the held-out kingdoms' kept falling (0.0229 ->
0.0210); the worst held-out species is never more than 0.03 behind, and the dependence on
base composition weakens steadily. For an encoder trained across a composition range
(bacteria and plants span 36 to 72 percent GC), universality holds with capacity. The plan
says this "publishes either way"; reported as the result it is. The human-only encoder,
whose training composition is narrow, is the contrasting case and its 200M point is
pending.

### The human-only series, 2M to 200M, and the universality result for both encoders

`runs_v3/v3_Uhuman_{2M,10M,50M,200M}`, all 100 epochs; errors are 1 - median R2:

| size | per-bp error, human / held-out median | global error, human / held-out median / worst held-out | global error ratio | Spearman vs GC distance |
| --- | --- | --- | --- | --- |
| 2M | 0.0153 / 0.0165 | 0.092 / 0.103 / 0.153 | 1.12 | -0.96 |
| 10M | 0.0076 / 0.0081 | 0.041 / 0.055 / 0.087 | 1.34 | -0.82 |
| 50M | 0.0042 / 0.0045 | 0.030 / 0.041 / 0.068 | 1.40 | -0.75 |
| 200M | 0.0032 / 0.0035 | 0.025 / 0.034 / 0.060 | 1.34 | -0.65 |

**The universality result, both encoders, 2M to 200M** (all pre-registered measures in
`results/universality_v2.json`):

1. Every held-out species improves in absolute terms at every size, for both encoders;
   the worst held-out species' error falls monotonically (human-only: 0.153 -> 0.060).
2. On per-base-pair channels the gap is negligible throughout: held-out error is about
   1.07-1.09 times the trained error for the human-only encoder, and at or below it for
   the held-out-kingdom encoder.
3. On whole-sequence channels the relative error ratio rises from 2M into the 10-50M
   range and then plateaus or falls back at 200M (human-only 1.12 -> 1.34 -> 1.40 ->
   1.34; held-out-kingdom 1.02 -> 1.03 -> 1.22 -> 1.11).
4. The dependence on base composition weakens monotonically with capacity for both
   (Spearman against GC distance, human-only -0.96 -> -0.65, held-out-kingdom -0.89 ->
   -0.62): larger models generalise more uniformly across compositions, not less.

**PLAN.md predicted that accuracy would improve while universality degraded. Over 2M to
200M, accuracy improves and universality does not degrade**: the only measure that moves
the predicted way, the whole-sequence error ratio, does so up to 50M and then reverses.
The plan's framing -- "this publishes either way" -- applies, and this is the way it
came out. The comparison of each held-out species against the multi-clade encoder of the
same size follows when the ladder runs finish.

---

## Family L does not scale with parameters; it overfits. A data-scaling test, pre-registered

### The diagnosis

Test results at 2M and 50M put every family but L at 0.998-1.000 per-bp by 50M; Family L
moved from 0.642 to 0.658 per-bp (0.856 to 0.857 global) across a 23-fold increase in
parameters. Its validation trajectory shows why:

| model | L per-bp peak (epoch) | L per-bp at the last epoch read | all families per-bp, same epoch |
| --- | --- | --- | --- |
| 2M | 0.645 (78) | 0.642 (99) | 0.9934, rising |
| 50M | 0.661 (44) | 0.641 (99) | 0.9989, rising |
| 200M | 0.654 (36) | 0.647 (52), falling | 0.9991 |
| 1B | 0.664 (26) | still early | 0.9992 |

The larger the model, the sooner Family L peaks and the further it falls afterwards while
every other family keeps improving: it is overfitting -- memorising the folds of the
training sequences rather than learning to fold -- and the peak barely moves with
capacity. What it fails on is fold detail (positional entropy 0.47-0.51, unpaired
probability 0.80-0.83, hairpin-size statistics about 0.25); ensemble free energies are
learned almost perfectly (0.994). Folding is combinatorial, and 172,210 training
sequences are evidently too few for it. Because the model is selected on the overall
score, it is also selected at an epoch where L has already declined.

### The test

PLAN.md: "Analytic targets are available in unlimited quantity, which is rare and makes
scaling cheap to study." Family L costs about 0.1 s per window, so:

- About 1M new windows from the same 15 panel genomes (new seed), with every window
  dropped that lands in a MinHash similarity group (Jaccard >= 0.5 on 11-mers) with any
  existing library fragment -- the main split's own criterion, by the same code
  (`src/data/sample_aux_windows.py`) -- so no auxiliary window is a near-duplicate of a
  test fragment.
- Family L targets only (`src/data/generate_aux_targets.py`, frozen and self-checked code
  snapshot), placed on the main dataset's exact scale by its own per-column transforms
  (`data.AuxLandscape`) and mixed into every step, all other columns NaN, which the loss
  masks (`--aux-shards`, `--aux-per-step 512`).
- `v3_L50M_auxL`: the 50M configuration exactly as `v3_L50M`, plus the auxiliary windows,
  scored on the identical test side.

**Stated before the run:** if Family L is data-limited, its test R-squared rises well above
the baseline's 0.658 per-bp and 0.857 global, and the overfitting peak moves later or
disappears; every other family must be no worse than in `v3_L50M` (non-inferiority, same
test side). If L does not move, the limit is the architecture's inductive bias for
folding, not data, and that is the result.

The auxiliary sample (`aux_L/aux_windows.tsv.report.json`): 1,048,846 candidate windows,
about 70,000 per panel species; 30,860 (2.9 percent) dropped for landing in a similarity
group with a library fragment -- mostly genomic repeats, which also put 14 percent of
all sequences into multi-member groups (largest 968) -- leaving **1,017,719** auxiliary
windows, 5.9 times the 172,210 main training sequences.

The first submission of `v3_L50M_auxL` failed at argument parsing: the Slurm script
expands its extra arguments unquoted, so the auxiliary glob `aux_L/shard_*.h5` was
expanded by the shell into 256 separate arguments. `--aux-shards` now takes the directory
(no glob character reaches the shell); the chain was resubmitted (34019092-94). Nothing
had run.

---

## The bottleneck sweep: the head-rank estimate of intrinsic dimensionality

`results/rank_sweep_v2.json`: the 10M trunk (width 272) with head rank 4 to full, every
run 100 epochs, held-out test side. The criterion was fixed before the runs: the smallest
rank whose median test R-squared is within 0.001 of the full-rank model's.

| head rank | per-bp median | global median | per-bp > 0.99 | global > 0.99 |
| --- | --- | --- | --- | --- |
| 4 | 0.268 | 0.689 | 0.059 | 0.028 |
| 8 | 0.640 | 0.878 | 0.091 | 0.118 |
| 16 | 0.825 | 0.951 | 0.147 | 0.235 |
| 32 | 0.942 | 0.985 | 0.231 | 0.396 |
| 64 | 0.985 | 0.992 | 0.406 | 0.554 |
| 128 | 0.996 | 0.994 | 0.685 | 0.616 |
| 256 | 0.998 | 0.995 | 0.692 | 0.625 |
| 272 (full) | 0.998 | 0.995 | 0.696 | 0.625 |

**Saturating rank: 128 for whole-sequence channels, 256 for per-base-pair channels**; by
the share of channels above 0.99, 128 for both. PLAN.md calls this "an independent
estimate of the intrinsic dimensionality", and it should be read next to the linear one
from the spectrum (effective rank, v2): participation ratio 11.9 per-bp and 5.3 global,
42 and 49 components for 95 percent of the variance, 72 and 85 for 99. The two disagree
by design rather than by error. The spectrum counts directions of the *total* variance;
predicting every channel to R-squared 0.99 needs each channel's own small-variance
directions too, which a 99-percent-of-total count leaves out. The landscape is low-rank in
bulk -- tens of directions carry almost all of it -- and needs low hundreds for
channel-by-channel fidelity. Both numbers are the answer to "how compact can the
landscape be stored", for two different accuracy targets.

---

## The scaling ladder, four of five rungs: per-bp channels scale, whole-sequence channels saturate

`results/scaling_v2.json` (held-out test side, similarity split; 1 - median R-squared):

| rung | parameters | per-bp error | global error | per-bp > 0.99 | global > 0.99 | coverage at 68.3 / 95.4 |
| --- | --- | --- | --- | --- | --- | --- |
| v3_L2M | 2.1M | 0.00670 | 0.00763 | 0.612 | 0.551 | 0.691 / 0.957 |
| v3_L10M | 9.5M | 0.00228 | 0.00484 | 0.696 | 0.625 | 0.692 / |
| v3_L50M | 48.9M | 0.00121 | 0.00437 | 0.727 | 0.625 | 0.696 / 0.958 |
| v3_L200M | 198.1M | 0.00073 | 0.00504 | 0.727 | 0.607 | 0.688 / 0.955 |

Local scaling exponents, error ~ N^-alpha, between successive rungs: per-bp 0.71, 0.39,
0.37 (0.49 over the whole range); global 0.30, 0.06, -0.10.

**Per-base-pair channels keep improving with capacity, with diminishing returns:** about
nine-fold less error for 94-fold more parameters. **Whole-sequence channels saturate by
10M and slip slightly at 200M** (F global 0.987 -> 0.982, G 0.988 -> 0.985 from 50M to
200M; L flat). The likely reason is the same data limit found for Family L, in a milder
form: each fragment supplies 61 training samples of every per-bp channel but one of every
whole-sequence channel, so the global channels have 61 times less data and overfit first.
That predicts the auxiliary-data experiment should help global Family L channels too,
and it says where the next gain for the global channels lies -- more fragments, not more
parameters. Calibration holds at every size. The 1B rung is pending.

**A 10M variant of the same test.** With GPU contention up and this account's fair-share
priority down after the day's runs, the scheduler put the 50M auxiliary run's start a day
out. `v3_L10M_auxL` -- the 10M rung exactly as `v3_L10M`, plus the same auxiliary windows
-- fits any GPU including 24 GB cards and tests the same hypothesis against a baseline
with the same Family L problem. **Stated before it runs:** baseline `v3_L10M` test Family L
per-bp 0.6581, global 0.8665; if L is data-limited, the auxiliary run beats both clearly,
with every other family non-inferior on the same test side. Both runs stay queued; the
first to finish answers the question and the second replicates it at another size.

---

## The synthetic-composition stress test

`src/eval/evaluate_run.py` -> `runs_v3/<run>/eval_by_group.json`: each run's kept checkpoint
re-scored on its own test fragments grouped by stratum and by GC band. Per-bp / global
median R-squared within each group, 2M -> 200M:

| encoder | genomic panel | composition extremes | random sequence | designed hexamer libraries |
| --- | --- | --- | --- | --- |
| multi-clade | 0.9933->0.9992 / 0.985->0.993 | 0.9920->0.9991 / 0.992->0.996 | 0.9929->0.9991 / 0.989->0.994 | 0.9969->0.9998 / 1.000 |
| bacteria + plants only | 0.9897->0.9983 / 0.957->0.982 | 0.9882->0.9980 / 0.983->0.989 | 0.9894->0.9981 / 0.972->0.985 | 0.9876->0.9959 / 0.718, 0.413, 0.825, 0.877 |
| human only | 0.9838->0.9965 / 0.923->0.976 | 0.9785->0.9954 / 0.929->0.965 | 0.9820->0.9960 / 0.943->0.973 | 0.9610->0.9906 / 0.733, 0.710, 0.794, 0.840 |

1. **Composition extremes (5-15 and 85-95 percent GC) and random sequence** are predicted by
   encoders trained only on natural genomes about as well as held-out genomic windows --
   per-bp within 0.001-0.005, global within 0.01-0.03 -- improving with capacity.
2. **The designed hexamer libraries** -- low-complexity constructs with fixed poly(A) or
   alternating-GC flanks, unlike any genome -- are the one failure: per-bp channels
   transfer (0.96-0.996) but whole-sequence channels do not, and not monotonically with
   capacity (0.41 at 10M for the held-out-kingdom encoder). Within this stratum the three
   sub-libraries' different flanks give the global channels real variance, so this
   R-squared is meaningful, unlike within a single sub-library.
3. **Training on such libraries removes the failure completely**: the multi-clade encoder
   scores 0.9999 on them at every size. The plan's choice to include a designed hexamer
   stratum and composition extremes in the training corpus is what makes the released
   encoder safe off the natural-genome manifold, and this measures it.

The GC-band rows (in the JSON) pool all strata in a band; their global dips for the
natural-only encoders (to 0.46 above 60 percent GC at 10M) are driven by the
alternating-GC hexamer fragments, not by GC-rich genomes, which the panel row shows.

---

## Window outputs: the dedicated heads were the weak link; windows now read from the per-bp output

Every model scored 0.94-0.96 median R-squared on its 10/20/50 bp windows against 0.999 per
base pair. A window target is the window average of the per-bp target
(`pooled_window_targets`), so `src/eval/window_diagnostic.py` scored each run two ways:
the window head, and the same pooling applied to the model's own per-bp predictions.
**Decided on the validation side** (choosing a readout is model selection), confirmed on
test (`runs_v3/<run>/window_diagnostic_{val,test}.json`):

| rung | 10 bp: head -> pooled | 20 bp | 50 bp | 50 bp channels > 0.99 |
| --- | --- | --- | --- | --- |
| 2M | 0.9499 -> 0.9782 | 0.9403 -> 0.9843 | 0.9384 -> 0.9897 | 0.00 -> 0.49 |
| 10M | 0.9592 -> 0.9860 | 0.9552 -> 0.9910 | 0.9446 -> 0.9942 | 0.00 -> 0.57 |
| 50M | 0.9636 -> 0.9890 | 0.9596 -> 0.9931 | 0.9471 -> 0.9954 | 0.00 -> 0.63 |
| 200M | 0.9667 -> 0.9893 | 0.9612 -> 0.9932 | 0.9486 -> 0.9957 | 0.01 -> 0.61 |

(validation side; test agrees to within 0.002 everywhere.) The pooled readout is now a
model option (`window_readout="pooled"`), the primary window readout, and the final
evaluation reports both. Its window means are taken after the reverse-complement merge,
so they are consistent with the per-bp output by construction. They are exactly
equivariant on every interior window (residual 9e-8) but not on the partial windows at
the fragment ends for channels with a reflection shift -- and neither are the window
*targets*: pooling an exactly reflected pair of profiles mismatches by 0.57 at the edges
and 2e-7 inside, because a partial window trims different positions in the two profiles.
The heads had imposed a symmetry the targets do not have there.

---

## Pre-registration: the axial pair module (PairStack) for Family L, 2026-11-20

Family L is the one family the model has not learned: test per-bp median R-squared
0.63-0.66 and whole-sequence 0.86 at every rung from 2M to 200M, while all other
families are at 0.998 or above. Two causes are being separated. Data: the auxiliary-
window runs (1M extra L-only windows) are at L per-bp 0.733 at epoch 41 against 0.642
for the baseline at the same epoch (validation), so L is at least partly data-limited.
Architecture: folding is combinatorial pairing, and the current design only biases
sequence attention with a fixed convolution over the base-pair map.

`PairStack` (`--pair-module axial`) instead keeps a pair representation (built from the
trunk features of both positions, their base identities and their separation), updates it
with attention along the rows and columns of the pair map under a symmetry constraint
(the RNAformer recipe for RNA folding), and returns per-head attention biases plus a
per-position summary added to the trunk features. Its outputs are zero-initialised (at
initialisation the model is exactly the model without it; tested), and each strand pass
builds its own pair map, so every channel's reverse-complement parity is still enforced
exactly by the merge (tested).

Cost, measured on a B200 at batch 512 (`scripts/slurm_bench_pair.sh`): the 10M rung goes
from 96 to 476 ms per step (64 channels x 4 layers; +0.3M parameters), 19.3 GiB peak;
128 x 8 would be 1,142 ms. The 64 x 4 configuration is the one tested.

**Runs.** Same rung (width 272, 12 blocks, rank 272, 4 attention layers, 8 heads, attn_mlp
whole-sequence readout, 100 epochs, the same similarity split and seed), changing only the
pair module:
- `v3_L10M_axial` against `v3_L10M` (no auxiliary data) -- the architecture effect alone;
- `v3_L10M_axial_auxL` against `v3_L10M_auxL` (both with the 1M auxiliary L windows) --
  whether the module adds to more data or only substitutes for it.

**Criterion, fixed now.** The module helps if, on the test side, Family L per-bp median
R-squared rises by at least 0.02 over its matched baseline (baselines: `v3_L10M` 0.6581;
`v3_L10M_auxL` read from its own final metrics when it finishes) and Family L
whole-sequence median does not fall, with every other family non-inferior: per-bp median
no more than 0.001 below and whole-sequence median no more than 0.002 below the matched
baseline. Seed-to-seed variation has not been measured at this rung; a gain smaller than
0.02 would be reported as inconclusive rather than as a gain. Epochs and readouts are
chosen on validation only, as for every run.

---

## Pre-registration: strand tie and base-pair supervision for Family L, 2026-11-20

**What limits Family L.** Per channel on the 200M test side: unpaired probability 0.79-0.81
and positional entropy 0.43-0.47 per base; the worst whole-sequence channels are G-quadruplex
gain (about 0; nearly constant), the co-transcriptional divergence (0.18-0.26), hairpin
sizes and counts, the MFE-ensemble gap, centroid distance and ensemble diversity
(0.26-0.43), while every free energy is at 0.992-0.993. The model has the energies and not
the shape of the Boltzmann ensemble. Every per-base L channel is an exact functional of
the base-pair probability matrix: recomputed from ViennaRNA's matrix,
unpaired = 1 - sum_j p_ij matches the stored target to 3e-8 and
entropy = -sum_j p_ij log2 p_ij - u log2 u matches to 1e-7 (first shard, 20 fragments,
both strands); ensemble diversity is a functional of the same matrix.

**Two changes, each exact.**
1. *Strand tie* (`--strand-tie`). The minus-strand transcript of x is the plus-strand
   transcript of rc(x), and its stored profiles are in its own 5'->3' frame -- the frame
   of the model's reverse pass (verified: the stored minus profile matches the minus
   transcript's own-frame profile to 3e-8, not the reflected one). So L_m(x) = L_p(rc x)
   position by position with no reflection. The model previously left all stranded
   channels to the forward pass alone, so it had to learn the minus strand through an
   internal reversal. Now every L_m output is read from the reverse pass's L_p output:
   exact equivariance, and both strands' targets train one plus-transcript predictor.
2. *Base-pair supervision* (`--bpp-dir`). `src/data/generate_bpp.py` stores ViennaRNA's
   matrix for both transcripts of every fragment with the targets' own model settings
   (uint8, sqrt-quantised; unpaired recomputed after quantisation within 5e-3). The
   axial pair stack predicts its own transcript's matrix (symmetric logits; forward pass
   against the plus strand, reverse pass against the minus strand in the same frame) under
   binary cross-entropy summed over partners and averaged over positions (weight 1), and
   feeds the matrix's exact functionals (paired probability, positional entropy) back
   into the trunk. The matrices are training targets only: no model input at any time,
   only training fragments contribute to the loss, and the auxiliary windows were
   leakage-filtered against the library as before.

**Runs** (10M rung, same hyperparameters, split and seed as `v3_L10M`):
- `v3_L10M_axtie` = axial + strand tie, against `v3_L10M_axial`;
- `v3_L10M_axtiebpp` = axial + strand tie + base-pair supervision, against `v3_L10M_axtie`;
- `v3_L10M_full_auxL` = all three plus the 1M auxiliary L windows (with their matrices),
  against `v3_L10M_auxL` -- the configuration for the best Family L model.

**Criteria, fixed now (test side, read only after training ends).** Each change helps if
Family L per-bp median R-squared rises by at least 0.02 over its matched baseline, Family
L whole-sequence median does not fall, and every other family is non-inferior (per-bp
median no more than 0.001 below, whole-sequence no more than 0.002 below). For the strand
tie the minus-strand channels are reported separately, since that is where it acts. A
smaller gain is inconclusive. Epoch selection and readouts on validation only.

**Addendum (same day, before any of these runs started).** (i) All Family L arms run one
frozen code snapshot (`CODE_ROOT`, `scripts/freeze_code.sh trainL`), so every link of
every chain runs identical code; the first `v3_L10M_axial` and `v3_L10M_axial_auxL`
chains had started on the live tree and were stopped at epochs 11 and 5 and set aside
(`runs_v3/aborted_*_livecode`). Their early numbers, for the record: the unsupervised
axial module was *behind* the convolutional pair bias at matched epochs (L per-bp 0.406
against 0.489 at epoch 11), so the module alone is not expected to clear the criterion;
the base-pair supervision is the arm that tests the idea. (ii) The pair logits start at
the measured prior (mean pairable probability 0.0087, logit -4.7): from logit 0 the
first base-pair gradient was 158 times the landscape loss's, falling to about 1x within
50 steps (`src/model/probe_bpp_weight.py`), and under global clipping that transient
starves the other families. (iii) One more arm: `v3_L10M_tie`, the convolutional pair
bias of `v3_L10M` plus the strand tie, against `v3_L10M` itself (test L per-bp 0.6581),
same criterion -- the strand tie on the unchanged baseline.

---

## Pre-registration: are the whole-sequence channels data-limited? 2026-11-20

Whole-sequence median R-squared sits at 0.995 from 10M to 200M parameters (test: 0.9952,
0.9956, 0.9950), while per-bp keeps rising (0.9977 -> 0.9993). Each fragment gives one
sample of a whole-sequence channel against 61 of a per-bp one, so the plateau may be a
data limit rather than a capacity limit. The weakest families with headroom are F
(0.987 at 10M), C (0.994) and D (0.997); G (0.988) and H have no auxiliary data here and
act as controls.

**Data.** The same 1,017,719 leakage-filtered auxiliary windows as the Family L test,
now with every whole-sequence channel of families B, C (ground-state only; the 34
channels that need the minimised structure or the relaxed ensemble stay missing), D, E,
F and L, plus Family L per-bp (`aux_all/`, `scripts/slurm_aux_all_targets.sh`).

**Run.** `v3_L10M_auxAll`: the `v3_L10M` configuration (convolutional pair bias, no other
change) with the auxiliary windows mixed in exactly as in `v3_L10M_auxL` (512 per step,
missing channels masked).

**Criterion, fixed now (test side).** Whole-sequence channels are data-limited if the
whole-sequence medians of families C, D and F each rise by at least 0.001 over
`v3_L10M` (0.9938, 0.9969, 0.9870), with Family L whole-sequence rising as well; the
control families G and H must not fall by more than 0.002 (whole-sequence) and no
family's per-bp median may fall by more than 0.001. A rise in only one of C, D, F is
reported as partial.

**Seed variance at the 10M rung (added the same day, before any Family L arm finished).**
The criteria above use fixed margins because seed-to-seed variation had not been
measured. `v3_L10M_seed1` and `v3_L10M_seed2` repeat `v3_L10M` exactly (same split,
`--split-seed 0`; new initialisation and batch order, `--seed 1/2`). Their spread is
reported beside every comparison; a pre-registered margin smaller than twice the
observed seed spread will be reported as such rather than silently widened.

**Strand tie and the per-channel standardisation (checked after launch).** Targets are
trained in standardised units, with each channel's mean and spread estimated on the
library separately, so the exact identity L_m(x) = L_p(rc x) holds in standardised units
only if the two channels share their statistics. The library is not exactly
reverse-complement symmetric, so they differ slightly (`landscape_cache_14db...` meta):
per-bp unpaired probability sigma_p/sigma_m = 1.0036, offset +0.0014 sigma; positional
entropy 0.9988, -0.0015 sigma (both log10/linear choices agree between strands). The
largest whole-sequence differences are 2-6 percent in spread on low-R2 channels
(co-transcriptional divergence 1.055, centroid distance 1.028; G-quadruplex gain 0.89,
a rare-event channel). The resulting floor on R-squared is about 1e-5 per base and at
most 0.004-0.012 on those whole-sequence channels, all of which currently score 0-0.4,
so the tie as launched (no affine correction) is kept; an exact version would carry the
two channels' statistics into the model as fixed buffers.

**One more arm: base-pair supervision on the convolutional pair bias.** The axial stack
costs 5x the convolutional pair bias per step; if supervision is what matters, the cheap
module may carry it. `PairBias` gets an optional base-pair head (`--pair-bpp`; off by
default so every earlier checkpoint rebuilds unchanged, tested) and the matrix's
functionals enter the sequence features before the attention layers (zero-initialised).
`v3_L10M_tiebpp` = `v3_L10M_tie` + base-pair supervision (weight 1), against
`v3_L10M_tie`; same criterion as above.

**The 16-snapshot channels (RESULTS.md section 18).** The sugar-pucker and BI
populations are at their binomial sampling ceilings (BI 0.760 against 0.761, C2'-endo
0.465-0.482 against 0.473-0.489, C3'-endo 0 against 0.01-0.02): no model can do better
on those targets. The torsion spreads sit below Gaussian-approximation ceilings that are
upper bounds for rotameric angles; a replicate relaxed ensemble would measure them.

---

## Outcome: Family L data scaling at 10M (`v3_L10M_auxL` against `v3_L10M`), test side

Pre-registered: "if L is data-limited, the auxiliary run beats both clearly, with every
other family non-inferior on the same test side" (no numeric margin was fixed then; the
margins fixed later for the Family L arms, -0.001 per-bp and -0.002 whole-sequence, are
applied here too). `results/family_l_arms.json`, `src/eval/arm_report.py`.

- Family L per-bp 0.6581 -> **0.7427** (+0.085); whole-sequence 0.8665 -> **0.8843** (+0.018).
  Per channel: minus-strand unpaired 0.812 -> 0.861, entropy 0.463 -> 0.581; plus-strand
  unpaired 0.836 -> 0.879, entropy 0.504 -> 0.625. All-channel medians 0.9977 -> 0.9979
  (per-bp), 0.9952 -> 0.9963 (whole-sequence).
- Every other family's per-bp median is unchanged or higher (C +0.0004, H +0.0007), and
  so are whole-sequence B, C, D, E, H; **Family F whole-sequence fell 0.0045** (0.9870
  -> 0.9825; validation shows the same drop, 0.983 -> 0.978), beyond the margin.

**Verdict: L is data-limited (clear, large gain), but the auxiliary L-only data was not
free at 10M** -- the non-inferiority condition fails on one family. The L-only windows
carry no Family F targets, so each step spends part of its gradient on L alone; the
whole-sequence auxiliary set (`aux_all`, which carries F) is the matched follow-up. Seed
spread from `v3_L10M_seed1/2` will be reported beside this when they finish.

---

## Finding: the Watson/Crick structural channels are not strand-symmetric in the targets

Motivated by the strand tie, the identity C(x) = W(rc x) was tested on the targets
themselves (`src/eval/rc_consistency.py`: production code, minimised ground state and
cgNA+ ensemble, six library sequences and their reverse complements;
`results/rc_consistency_v2.json`). Largest |C(x) - W(rc x)| relative to the channel's
spread across the sequences:

| stem | mismatch / spread |
| --- | --- |
| C_mean_phosphate_radius | 0.000 (exact) |
| G_mean_major_edge | 0.05 |
| G_mean_minor_edge | 0.19 |
| G_mean_sasa_base | 0.35 |
| G_mean_sasa_sugar | 0.50 |
| G_mean_sasa_total | 1.76 |
| G_mean_sasa_phosphate | 1.94 |
| C_mean_phosphate_azimuth | 2.0 (a frame convention: W and C azimuths are different quantities) |
| G_mean_radical_target | 2.48 |
| G_cleavage_heterogeneity | 4.78 |
| G_max_cleavage | 6.60 |

The library statistics agree: over the whole corpus the W and C versions of the Family G
channels differ by 0.05-0.23 standard deviations in mean and 0.72-0.97 in spread
(G_max_cleavage: -0.19 sd, spread ratio 0.72), an order of magnitude more than the
library's own strand asymmetry (Family L, whose strand identity is exact by
construction: <= 0.02 sd). A duplex and its reverse complement are the same molecule, so
these differences are artifacts of the structure pipeline or of strand-dependent
definitions, not physics. Consequences: (1) no Watson/Crick tie is applied (it would
impose a symmetry the targets violate); (2) part of the C-strand channels' lower R2
(G_max_cleavage_C 0.30 against _W 0.47) is the model fitting an asymmetric target;
(3) the Family G solvent-accessibility and cleavage channels need a ground-truth fix
before they are used as physics. The per-position check (ends against interior) is
running to locate it.

**Located: the strand asymmetry is at the fragment ends** (`rc_consistency.py`, per-bp
pairs compared position by position, C(x)[k] against W(rc x)[n-1-k]; median over the 57
interior positions against the worst value at the two terminal positions of each end,
both relative to the channel's spread). Interior agreement is good -- 0.000-0.003 for
the purely sequence-defined tracks (CPD sites, cytosine C6, purine N7, DMS/DEPC probes,
beta, epsilon), 0.02-0.09 for the surface areas, 0.13-0.19 for alpha, gamma and the
radical-target track -- while the ends disagree by 0.4-0.7 spreads for every surface
area and up to 1.3-6.6 for backbone torsions and the phosphate axial offset (gamma
5.6, axial offset 6.6, epsilon 2.0). The minimised duplex's terminal residues are
treated differently on the two strands. A whole-sequence maximum is dominated by the
most exposed residue, which is terminal, so G_max_cleavage and G_cleavage_heterogeneity
mostly measure this end artifact (6.6 and 4.8 spreads). For the next target revision:
compute the Family G whole-sequence statistics (max, heterogeneity, means) over interior
positions only -- possible from the stored per-bp tracks without new physics -- and
report interior per-bp R2 (already computed, 5-bp trim) as the primary number for the
structural families. The modest interior mismatch of alpha, gamma and the radical
target (0.13-0.19) is minimiser noise between two runs on the same molecule and joins
the solver noise floor (RESULTS.md section 10).

---

## Outcome: strand tie at 10M (`v3_L10M_tie` against `v3_L10M`), test side

- **Family L per-bp 0.6581 -> 0.7840 (+0.126); whole-sequence 0.8665 -> 0.8934 (+0.027).**
  Per channel: minus-strand unpaired 0.812 -> 0.899, entropy 0.463 -> 0.669; plus-strand
  unpaired 0.836 -> 0.899, entropy 0.504 -> 0.663 -- both strands now equal, and both
  better, as expected when both strands' targets train one predictor. The gain is
  larger than the 1M auxiliary windows' (+0.085) at no data cost.
- All-channel medians: per-bp 0.9977 -> 0.9977, whole-sequence 0.9952 -> 0.9965. Every
  other family's per-bp median within +-0.0001.
- **Family F whole-sequence median fell 0.0135** (0.9870 -> 0.9735; validation 0.9826 ->
  0.9737), beyond the -0.002 margin, so by the rule fixed in advance **the arm does not
  pass**, although L gains far more than the criterion's +0.02. The median overstates
  it -- F's 39 whole-sequence channels have a gap at the median (0.970 against 0.987) --
  but the loss is real: the median per-channel change is 0, with 11 channels down by
  more than 0.005 (G-quadruplex loop and tract channels -0.025 to -0.07, the longest
  tandem unit -0.05, triplex purine fraction -0.04) and 3 up. The same direction
  appeared with the auxiliary L windows (-0.0045). Reading: at 10M, making Family L
  learnable moves trunk capacity toward RNA folding and away from rare-motif detection.
  That is testable: a capacity trade-off should shrink with width.

**Decision.** The tie stays in the architecture (it removes an internal reversal the
model had to learn, and is exact). Its cost is measured across capacity: a scaling
ladder with the tie (`v4_L2M`, `v4_L50M`, `v4_L200M`; the 10M rung is `v3_L10M_tie`,
whose training path is identical) is compared rung by rung with the v3 ladder on every
family, F in particular. Pre-registered reading: if the F loss is a capacity trade-off,
it shrinks monotonically with size and is within the -0.002 margin at 200M.

**Target revision v2.1 (Family G whole-sequence strand statistics over interior
positions).** `InMemoryLandscape(interior_trim_g=5)` / `train.py --g-interior-trim 5`
recomputes G_mean_{sasa_total, sasa_sugar, sasa_phosphate, sasa_base, radical_target,
major_edge, minor_edge}_{W,C}, G_cleavage_heterogeneity_{W,C} and G_max_cleavage_{W,C}
from the stored per-bp tracks over positions [5, 56) -- the same definitions as
`solvation.py` (nanmean, nanstd, nanmax; tested to reproduce the stored channels exactly
at trim 0), in raw units before any transform. The trim is the project's existing
interior definition (`INTERIOR_TRIM = 5` in the evaluation), which also covers the
terminal two positions where the strand asymmetry sits. Nothing else changes; the cache
key carries the option, so v2 and v2.1 caches coexist. v3 and v4 ladders stay on v2 for
comparability; v2.1 is for the final model, and its G whole-sequence numbers are
reported as a separate target set, never mixed with v2's.

**v4 ladder, second pre-registered reading (fixed before any v4 rung reports).** In the
v3 ladder Family L test per-bp was flat with capacity (0.63-0.66 from 2M to 1B) while
every other family rose. If that flatness came from the architecture (the internal
reversal the minus strand required) rather than from a property of folding, then under
the tie L per-bp rises with size: v4 2M < 10M (0.784) < 50M < 200M on the test side.
A flat v4 curve would mean the tie fixed a fixed handicap but folding itself still does
not scale with this trunk. Supporting interim evidence, recorded now so it cannot be
re-read later: at 50M the auxiliary L windows cost Family F nothing on validation
(F whole-sequence 0.9882 against 0.9870 at epoch 70) where at 10M they cost 0.0045 --
the capacity reading of the F trade-off.

---

## Negative result: whole-sequence channels read from the per-bp output (10M, val and test)

`src/eval/derived_globals.py`, `runs_v3/v3_L10M/derived_globals.json`. 104 of the 303
linear-scale whole-sequence channels are exact functionals (R2 >= 0.999 on training
targets) of a per-bp channel -- a mean, maximum, minimum or spread over positions. Read
as the same functional of the model's per-bp output, they score **worse** than the
whole-sequence heads: median R2 0.9939 against 0.9992 (val and test alike), better on 42
of 104. Means come out about even (L mean positional entropy 0.39 -> 0.40, C mean BI
fraction 0.66 -> 0.67); maxima and spreads collapse (max cleavage 0.48 -> 0.21, max
positional entropy 0.44 -> 0.07), because the per-bp output is a conditional mean -- a
smoothed profile whose maximum and spread are biased low -- while the head learns the
conditional mean of the maximum directly. Choosing per channel on val raises the test
median only from 0.9992 to 0.9994. Not adopted; the 200M analysis was cancelled. The
contrast with the windows (a window target is a *mean* of per-bp values, which pooling
the per-bp means estimates without bias) is the general rule: derive linear functionals,
learn nonlinear ones.

**Measured ceilings for the 16-snapshot channels (replicate ensembles, RESULTS 18b).**
Recomputing the relaxed ensemble with an independent seed for 120 fragments puts the
ceilings at -0.03 (C3'-endo) to 0.77 (BI fraction); the 200M model is within 0.05 of
the ceiling on 15 of 18 channels (median gap 0.015), and the remaining gaps (alpha and
gamma spreads, 0.07-0.15) sit under ceilings of 0.23-0.33. The weakest non-L per-bp
channels are therefore target-limited: raising them needs more snapshots per fragment
(the noise variance falls as 1/n), which is a ground-truth decision for the next target
revision, not a model change. The target-only estimate of section 18 is kept for the
populations and retired for the spreads.

---

## Interim: base-pair supervision does not help the convolutional pair bias (validation)

`v3_L10M_tiebpp` against `v3_L10M_tie`, matched epochs on validation: ahead early (L
per-bp 0.571 against 0.441 at epoch 7), level by epoch 15, behind from epoch 19 (0.751
against 0.760 at epoch 20; L whole-sequence 0.871 against 0.877; Family F
whole-sequence 0.915 against 0.939). The pre-registered test comparison is read when the
run ends; the trend is recorded now because it decides the final configuration. With
the axial stack, base-pair supervision is worth +0.14 (0.615 -> 0.761 at epoch 21), but
that only brings the axial model level with the convolutional pair bias plus the tie
(0.764 at epoch 21) at five times the cost. So far the cheapest configuration is also
the best: convolutional pair bias + strand tie.

## Pre-registration: the final encoder of this round (`v4f_L200M`)

Configuration fixed now, from the evidence above: the v3 200M rung (width 1232, 16
blocks, rank 512, 4 attention layers, convolutional pair bias, attention/MLP
whole-sequence readout, same split and seed) plus
- the strand tie (Family L: +0.126 per-bp at 10M);
- the whole-sequence auxiliary windows `aux_all` (1,017,719 windows: whole-sequence
  channels of B, C, D, E, F, L and per-bp L), mixed 1:1 per step as in every auxiliary
  run, with gradient accumulation 4 (same effective batch);
- target revision v2.1 (Family G whole-sequence strand statistics over positions
  [5, 56); `--g-interior-trim 5`);
- windows read from the pooled per-bp output (reported beside the heads).
Base-pair supervision and the axial stack are left out (above). Reported against
`v3_L200M` on the same test side, family by family; Family G whole-sequence channels are
on v2.1 targets and are reported separately from the v2 comparison. Expected, stated
now: Family L per-bp at least 0.80 and whole-sequence at least 0.90; every other
family's per-bp median within 0.0005 of `v3_L200M`; whole-sequence medians of C, D, F at
or above `v3_L200M`'s. Anything short of that is reported as measured.

---

## Three arms stopped early (resource decision, 2026-11-20 15:55)

The GPU partitions are saturated (every GPU type fully allocated, 180-340 higher-priority
jobs pending per partition), and GPU-hours spent now lower this account's fair-share
priority for the runs that matter. Three pre-registered arms whose question the
validation trend had already answered were cancelled; their test comparisons will not
exist and are reported as not completed. Interim validation numbers at the last
evaluated epoch (L per-bp / L whole-sequence / all per-bp / all whole-sequence):

| arm | epoch | values | matched comparison at the same epoch |
| --- | --- | --- | --- |
| `v3_L10M_axial` | 22 | 0.562 / 0.838 / 0.9964 / 0.9883 | conv `v3_L10M` 0.642 L per-bp |
| `v3_L10M_axial_auxL` | 10 | 0.402 / 0.800 / 0.9918 / 0.9608 | conv `v3_L10M_auxL` 0.450 L per-bp |
| `v3_L10M_full_auxL` | 7 | 0.568 / 0.742 / 0.9759 / 0.9164 | -- (superseded by `v4f_L200M`) |

The axial stack without base-pair supervision trails the convolutional pair bias at
every matched epoch, with or without auxiliary data, so neither arm could pass its
criterion; `full_auxL` was the axial route to the best Family L model, which the
convolutional pair bias + tie route reaches at a fifth of the cost and which the final
encoder now carries. `v3_L10M_axtie` and `v3_L10M_axtiebpp` continue at low priority:
together they measure the base-pair supervision effect on a model with a pair
representation (+0.14 L per-bp at epoch 21), a positive result worth completing.

---

## Aim 2 begins on CPU while the GPU queue is saturated (2026-11-20 16:00)

The scheduler put the next GPU starts at Nov 21-22 (fair-share after today's GPU use; every
GPU type fully allocated). Aim 2's machinery does not need GPUs at development scale, so
it is built now with a frozen development encoder (`v3_L10M_tie`, the 10M rung with the
strand tie) on the three cheapest calibration sets, and rerun with the final encoder when
it exists.

- `src/readout/featurize.py`: the encoder's per-bp landscape of any sequence -- 61-bp
  windows every 31 bp, stitched with centre-weighted (triangular) weights, since the
  encoder only knows 61-bp molecules; short sequences completed with the construct's own
  flanks (loop-seq adapters). Standardised units, float16.
- `src/readout/train_readout.py`: the plan's three arms at matched architecture (seq,
  phys, seq+phys), the bottleneck sweep (phys:k, k = 4 ... 128 and full), and the
  decoder test (decode, decode:k: bases predicted from the landscape alone, reported as
  per-base accuracy and bits recovered). One head architecture and one budget for all
  arms (conv stem, 3 dilated residual blocks, mean+max pooling, MLP; AdamW 1e-3, at most
  30 epochs, early stopping on validation, patience 5). Splits: the dataset's own test
  split where it has one (splice sites, Jores plant), similarity-grouped otherwise
  (loop-seq, all libraries pooled); validation always similarity-grouped from training.

**Datasets in this first pass.** loop-seq cyclizability (197k 50-mers; mechanics -- the
closest assay we hold to the plan's positive control, in vitro nucleosome occupancy,
which is not ingested); splice-site class (30k; the plan's negative control; the central
121 bp around the site for every arm); Jores plant promoter enrichment (148k rows; the
smallest of the plan's MPRA-type sets). lentiMPRA, DeepSTARR and yeast need GPU-scale
featurisation (about 90 GB of tracks for lentiMPRA alone) and wait for it.

**Expectations fixed now, from the plan.** (1) Splice sites: the physics head should
fail badly; if it approaches the sequence model, sequence identity is leaking through
the channels. (2) Loop-seq: a mechanics assay, so the physics head should come close to
the sequence model. (3) Decoder: if bases are recoverable at high accuracy, the
compression framing is wrong and is reported as such -- the smoke test (128 sequences,
two epochs) already decoded 96 percent of bases, so this is the expected outcome, and
it decides how (1) must be read: with an invertible landscape a strong physics head on
splice sites would be the decoder at work, not physics. The bottleneck sweep answers how
many landscape dimensions each readout needs and how many carry sequence identity.
Noise ceilings (replicate reproducibility) for these assays come from the source
publications and are not in the ingested tables; they are added before any ceiling
ratio is reported.

**Aim 2 noise ceilings, sourced (2026-11-20).**
- *loop-seq:* the only replicate correlation reported by Basu et al. 2021 (Nature 589:462;
  PMC7855230) is Pearson r = 0.869 (95% CI 0.865-0.872) between **technical** replicates on
  the Cerevisiae Nucleosomal Library (Extended Data Fig. 3c); no biological-replicate
  correlation and none for the Random, Tiling or ChrV libraries. The other correlations in
  that paper (r = 0.73 sequence against reverse complement, Extended Data Fig. 3f; 0.63
  against a 5-base-extended construct, Extended Data Fig. 4l) are design or orientation
  controls, not replicates. Used as the assay ceiling: explainable variance fraction about
  0.87 of a single run's variance (technical replicates only, transferred from one library
  to all four -- both caveats reported with the number).
- *Jores et al. 2021 (Nat Plants 7:842; PMC10246763):* two biological replicates per
  library and system; the replicate R2 and Spearman values are drawn only inside Fig. 1c,d
  and Supplementary Fig. 1, not stated in text, so they are not used here (not transcribed
  from an image by eye). Recorded instead as reference points: the authors' CNN reached test
  R2 0.71 (tobacco leaves) and 0.67 (maize protoplasts) on a 10 percent test set, and
  between-system correlation of promoter strength was R2 0.14-0.40. The ceiling ratio for
  this dataset waits until the replicate values are read from the supplementary PDF or
  recomputed from the authors' raw counts.
- *Splice sites:* a classification with documented class labels; no replicate notion.

---

## Aim 2, first calibration results: the landscape is invertible, so the bottleneck is measured in bits

Development encoder `v3_L10M_tie`; heads and splits as pre-registered above.

- **Decoder test (loop-seq):** bases are recovered from the per-bp landscape with
  validation per-base accuracy **1.0000 after one epoch**. The channel stack carries the
  whole sequence. By the plan's own rule, the compression framing is wrong for the
  per-bp landscape and is reported as such.
- **Negative control (splice sites, test n = 3000, 3 classes):** sequence head 0.830,
  physics head **0.970**, sequence+physics 0.972, and the physics head through a 4-channel
  projection 0.969 (k = 8 ... 128: 0.967-0.972). Read with the decoder result, this is not
  physics explaining splicing: each per-bp channel is a function of a neighbourhood of
  bases, so the landscape is a contextual encoding of the sequence, and 4 projected
  channels per base already carry 2 bits per base. The sequence baseline is also weak
  (its receptive field is 35 bp); a tuned comparison (receptive field 259 bp, width 128,
  two learning rates, 60 epochs) is running for all arms before any reading of 0.83 vs
  0.97 is made.
- **Consequence for the bottleneck sweep.** Narrowing the channels (k = 4 ... 256) does
  not limit sequence information -- four channels per position are a one-hot code. The
  plan's rate-distortion curve therefore needs the rate measured as *bits of sequence
  the representation still carries*, which the decoder test measures directly. New arms
  (`train_readout.py`): `physavg:w` (landscape smoothed over w = 5, 10, 20, 50 bp) and
  `global` (whole-sequence channels only) as information-limited physics
  representations, each paired with its decoder (`decodeavg:w`) so that every readout
  score sits beside the bits of sequence it could have used. A physics claim is
  defensible where a representation carrying little sequence information still reaches
  the assay. This is the change to Aim 2's design (plan amendment A8 follows when the
  first curve exists).

**Splice-site comparison with an adequate sequence head (test n = 3000).** With the head
widened to 128 channels and deepened to a 259-bp receptive field (60 epochs, two learning
rates), the sequence head rises from 0.830 to **0.976** and the physics head stays at
0.976 (seq+physics 0.978; 4-channel physics projection 0.976; both learning rates agree to
0.001). The first-pass "physics beats sequence" (0.970 vs 0.830) was the weak sequence
baseline, which the plan's "best sequence model achievable under the budget" rules out.
With the landscape invertible (decoder: 100 percent of bases, 1.997 of 1.997 bits, also
through a 4-channel projection), physics-only equals sequence-only on the negative
control, which is exactly what an invertible representation predicts. **The tuned head
is now the head for every Aim 2 arm**; the first-pass (35-bp receptive field) numbers are
kept in `results/readout/` for the record and not used.

Loop-seq with the first-pass head (test n = 19,247, similarity-grouped): sequence R2
0.716, physics 0.706, seq+physics 0.723, physics through 4..128 channels 0.699-0.725;
against the sourced ceiling 0.869 the sequence head reaches 0.82 of it and the physics
head 0.81. Decoder 100 percent (1.968 of 1.968 bits), at k = 4 and 16 as well. Tuned
reruns and the information-limited arms are queued.

**Readout queue trimmed (17:20).** The smoothed-landscape arms (`physavg`, `decodeavg`) were
cancelled for loop-seq and the plant set before they started: on splice sites, moving
averages over 5-20 bp left 99.95-100 percent of bases recoverable, so they are not
information-limited representations, and the CPU queue is congested. The binned arms
(`physbin`/`decodebin`) and the whole-sequence arms (`global`/`decodeglobal`) carry the
rate-distortion measurement.

**Bug caught by a dry run before any GPU time (17:26).** `AuxLandscape` refused circular
channels ("circular auxiliary channels are not supported"), and the whole-sequence
auxiliary set carries Family C angles, so `v4f_L200M` and `v3_L10M_auxAll` would both
have failed at startup on their first GPU slot. The main dataset expands a circular
channel (degrees) into sine and cosine columns before standardising; the auxiliary
mapping now does the same (`tests/test_aux_map.py`: the auxiliary mapping reproduces the
main transforms exactly, circular and log10 columns included). A CPU dry run of the
final configuration on the fixed snapshot (`final2_20261120_1726`) loads 191,344 main
fragments (v2.1 targets, 18 Family G channels recomputed) and 1,017,719 auxiliary
windows with 4 per-bp and 268 whole-sequence columns mapped (17 all-missing: the
minimised-structure channels), peak 11 GB. Both chains were resubmitted on it; the
configuration is unchanged from its pre-registration.

**Splice-site rate-distortion (development encoder, tuned head, test n = 3000).** Bits of
sequence recoverable per base (decoder; entropy 1.997) against 3-class accuracy (sequence
head 0.976; chance 0.333):

| physics representation | bits / base | accuracy |
| --- | --- | --- |
| per-bp landscape, or 5-bp bins with all channels | 1.997 | 0.970 |
| 5-bp bins, 1 channel | 0.652 | 0.898 |
| 10-bp bins, 4 channels | 0.637 | 0.949 |
| 20-bp bins, 4 channels | 0.367 | 0.943 |
| 10-bp bins, 1 channel | 0.196 | 0.891 |
| whole-sequence channels only | 0.188 | 0.760 |
| 20-bp bins, 1 channel | 0.150 | 0.832 |

Coarse physics that carries well under half a bit of sequence per base still separates
the classes far above chance (0.83-0.94). The plan's premise for this negative control --
splice identity is base readout, so a physics stack should fail -- does not hold for
this dataset at coarse resolution: the classes carry compositional context (the
pyrimidine tract upstream of acceptors, exon/intron composition), which coarse physics
sees. A strict negative control would need composition-matched decoys. The assay-level
reading therefore comes from comparing curves across assays at matched rates (loop-seq
next), not from this curve alone.

**Readout memory fix (18:45).** The plant-set readout jobs ran out of memory at 40 GB. A
stage-by-stage profile located it: `np.nan_to_num` on the 14.7 GB float16 landscape
array peaked at 49.9 GB from its temporaries, while loading (15.6 GB peak), the grouped
split, the one-hot sequences and batch extraction all stayed at 15.2 GB. The
replacement sets non-finite values to 0 in place, 4,096 rows at a time (non-finite
entries do not occur in featurizer output; 0 is the standardised mean). The plant arms
were resubmitted at 24 GB with the binned bottleneck arms added.

**Interim Aim 2 reading: coarse-graining measures resolution dependence, not physics
(19:20).** Loop-seq with the tuned head (test n = 19,247): sequence R2 0.748 (0.86 of the
0.869 ceiling), per-bp physics 0.742 (0.85 of it). At matched information-limiting
representations, as a fraction of each task's sequence head (splice: accuracy above
chance, 0.643 = 0.976 - 0.333):

| representation | splice (fraction of sequence head) | loop-seq (fraction) |
| --- | --- | --- |
| 5-bp bins, all channels | 0.99 | 0.92 |
| 5-bp bins, 4 channels | 0.98 | 0.88 |
| 5-bp bins, 1 channel | 0.88 | 0.53 |
| whole-sequence channels | 0.66 | 0.43 |

At every coarse representation the negative control keeps *more* of its sequence-model
score than the mechanics assay does -- the reverse of the plan's expected ranking
(mechanics physics-reachable, splicing not). The likeliest reason is resolution, not
physics: intrinsic cyclizability depends on how bends add up in phase along the 10.5-bp
helical repeat, which 5-bp bins and whole-sequence averages destroy, while the splice
classes differ in regional composition, which survives coarse-graining. So an
information bottleneck built by coarse-graining measures a readout's dependence on fine
spatial detail, and cannot by itself separate "physical" from "base readout". A
bottleneck that keeps full resolution but removes base identity would be needed -- and
any rich per-bp physical description is close to invertible (a 4-channel learned
projection recovers every base). What survives as a defensible Aim 2 statement so far:
(i) the landscape is an invertible re-encoding of sequence, so physics-only and
sequence-only heads are equal on every assay tested (splice 0.976/0.976, loop-seq
0.748/0.742); (ii) fixed, non-learned physical channel sets at full resolution are the
remaining candidate for a physics bottleneck, and their decodability must be measured
before any claim; (iii) the three-arm comparison remains useful as an engineering
result (seq+physics against seq). These readings are interim (development encoder,
three calibration sets) and are revisited with the final encoder.

**Next test of reading (ii): fixed physical channel sets at full resolution.** Named
selections with no learned projection (`train_readout.py` `physsel:`/`decodesel:`):
the four classic shape features (minor-groove width, propeller twist, roll, twist), the
thirteen base-pair and step parameters plus minor-groove width, and each per-bp family
alone (B electronic, C geometry, E thermodynamics, G solvation and reactivity, H
methylation). Each is decoded (bits of sequence it carries) and read out on splice
sites and loop-seq.

**Fixed physical channel sets are invertible too (splice-site sequences, test side,
19:40).** Decoder per-base accuracy from named channels at full resolution, no learned
projection: the four classic shape features (minor-groove width, propeller twist, roll,
twist) **1.000** (1.997 of 1.997 bits); the thirteen base-pair and step parameters
1.000; Family C (geometry) 1.000; Family E (thermodynamics) 1.000; Family G (solvation
and reactivity) 1.000. The one exception is Family H (methylation-induced changes, zero
away from CpG): 0.434 accuracy, 0.185 bits, splice readout 0.547. Readouts from the
invertible sets equal the full landscape (0.968-0.974).

Reading: at base resolution any substantive physical description is a one-to-one
re-encoding of sequence -- not only the encoder's predictions but the physics models
themselves, which are deterministic, high-precision functions of the local sequence
(a pentamer-indexed shape table identifies its pentamer). A physics bottleneck at full
resolution therefore does not exist; a bottleneck requires coarse-graining, which
measures resolution dependence (above). The plan's own contingency applies ("the
compression framing is wrong and the paper says so; the project then rests on the
dimensionality and the amortization arguments"). Aim 2 is restated accordingly once the
loop-seq selections and the plant set report: the three-arm table stays (as an
engineering result: does the landscape add to sequence?), the decoder table becomes a
headline result, and the rate-distortion curves are reported as resolution dependence
per assay.

**Loop-seq, fixed selections and the engineering arm (tuned head, test n = 19,247).** The
four shape channels and the thirteen base-pair/step parameters are fully decodable on
loop-seq sequences as well (1.968 of 1.968 bits) and read out at R2 0.729 and 0.733 --
0.97-0.98 of the sequence head (0.748), consistent with shape-based cyclizability
predictors, and, being invertible, not evidence for physics over sequence. Engineering
arm: sequence+physics 0.747 against sequence 0.748 on loop-seq, 0.978 against 0.976 on
splice sites -- the landscape adds nothing measurable to sequence for a head of this
size on these assays. Coarse-grained loop-seq: 10-bp bins with all channels 0.408, 20-bp
0.302, whole-sequence channels 0.325 (0.343 bits), 10-bp single channel 0.232 (0.117
bits) -- cyclizability is lost with resolution far faster than the splice classes.

**v4 ladder, first rung (`v4_L2M` against `v3_L2M`, test side).** Family L per-bp 0.6423 ->
**0.7335** (+0.091), whole-sequence 0.8556 -> 0.8770; all-channel per-bp 0.9933 -> 0.9936,
whole-sequence 0.9924 -> 0.9942; every other family's per-bp within 0.0015 (C +0.0015,
H +0.0005). Family F whole-sequence 0.9147 -> 0.9122 (-0.0025), with the median
per-channel change +0.0001 and 6 channels down, 5 up by more than 0.005 -- balanced,
unlike 10M (11 down, 3 up, -0.0135).

Against the two pre-registered readings so far: (1) L under the tie rises with size --
0.734 at 2M, 0.784 at 10M (+0.050), where v3 rose 0.642 -> 0.658 (+0.016); 50M and 200M
pending. (2) The F cost is *not* larger at the smaller rung, which a plain capacity
trade-off predicts; at 2M it is at the margin and balanced. The 10M drop may be partly
specific to that run -- the 10M seed replicates (pending GPU) measure exactly that -- and
the 50M and 200M rungs decide the trend.

**Plant-set readout redesigned before any result was used (20:35).** The first plant arm
(sequence, tuned head) reached test R2 0.431, with per-system r = 0.761 (tobacco leaf) and
0.665 (maize protoplast) but per-system R2 of only 0.464 and 0.194: one output was
predicting two assay systems whose readouts correlate at R2 0.14-0.40 (Jores et al.),
so a shared output scale and offset cost most of the variance. The dataset now has one
output per system on a shared trunk, targets standardised per system on the training
side, and is scored per system (`DATASETS["jores"]["condition"]`), as the source paper
modelled it. The pooled result is kept in `results/readout/superseded/` and not used;
all plant arms were resubmitted with the per-system design.

**Outcome: base-pair supervision on the convolutional pair bias (`v3_L10M_tiebpp` against
`v3_L10M_tie`, test side).** Family L per-bp 0.7840 -> 0.7739 (-0.010), whole-sequence
0.8934 -> 0.8913 (-0.002); per channel, minus-strand unpaired 0.899 -> 0.894, entropy
0.669 -> 0.656, plus-strand unpaired 0.899 -> 0.892, entropy 0.663 -> 0.645; every other
family within 0.0001. **Does not help**, as the validation trend had shown, and the final
encoder's configuration (no base-pair supervision) stands.

**Seed spread at 10M (`v3_L10M` against `v3_L10M_seed1`: same split, new initialisation and
batch order; test medians).** Per-bp families within 0.001 (H 0.0010, C 0.0002), Family L
per-bp 0.0066; whole-sequence families within 0.0005 except F 0.0023 and L 0.0031.
Individual Family F whole-sequence channels move by up to 0.01-0.04 between seeds (8
down, 3 up by more than 0.005), so per-channel counts alone are noise-level.

Reading the arms against it, as the addendum requires: the -0.002 whole-sequence margin
is *smaller than twice the observed F seed spread* (0.0046) and is reported as too tight
for Family F; it is not widened after the fact. (1) Strand tie: L per-bp +0.126 is 19
times the L seed spread; the F median drop (-0.0135) is about 6 times the F seed spread,
and the largest per-channel drops (G-quadruplex minimum loop -0.070, longest tandem unit
-0.053, tract count -0.045, triplex purine fraction -0.040) are 3-10 times those
channels' own seed-to-seed changes -- a real cost at 10M. (2) Auxiliary L windows: L
+0.085 (13 times the spread); the F median drop (-0.0045) is within twice the seed
spread -- not distinguishable from seed noise. (3) Base-pair supervision on the
convolutional pair bias: L -0.010, about 1.5 times the L seed spread -- no gain.

---

## Outcome: Family L data scaling at 50M (`v3_L50M_auxL` against `v3_L50M`), test side

- Family L per-bp 0.6579 -> **0.7350** (+0.077), whole-sequence 0.8573 -> **0.8851**
  (+0.028); per channel, minus-strand unpaired 0.804 -> 0.864, entropy 0.471 -> 0.576,
  plus-strand unpaired 0.827 -> 0.879, entropy 0.512 -> 0.606.
- Every other family non-inferior: per-bp changes 0 to +0.0008; whole-sequence changes
  -0.0005 (G) to +0.0017 (**F improves**). All-channel medians 0.9988 -> 0.9990 (per-bp),
  0.9956 -> 0.9962 (whole-sequence).

**Verdict: helps** -- the pre-registered criterion is met in full at 50M. With the 10M
result (+0.085, and an F change within twice the seed spread), Family L is data-limited
at both sizes, and the auxiliary windows' cost to other families, if any, disappears with
capacity.

---

## Aim 2 calibration complete (development encoder), and what follows

RESULTS section 25 holds the full table (three assays, every representation, decoder
bits beside each readout); PLAN.md amendment A9 restates Aim 2 on that basis. The plant
set confirms the pattern seen on the other two: per-system sequence head 0.712 (the
source paper's CNN: 0.71 tobacco, 0.67 maize), physics 0.672, physics through 4 channels
0.714, shape channels 0.693, sequence+physics 0.722; coarse representations keep 0.50-0.79
of the sequence score, between splice sites (0.66-0.95) and loop-seq (0.31-0.43). Next
steps for Aim 2, in order: rerun the calibration with the final encoder (`v4f_L200M`);
seeds for the engineering arm; composition-matched decoys for a strict negative control;
then the plan's main datasets (lentiMPRA, DeepSTARR, yeast) with GPU-time featurisation.

---

## 2026-11-21 -- the user restates the plan of record: "never cut back"

The user re-supplied revision 2 in full: "This was the original plan, make sure you're
following it, we will improve upon this plan, but never cut back." `PLAN.md` already holds
that text verbatim. From here, amendments only add; a revision-2 item is never removed or
narrowed, and where a revision-2 stopping rule fires (the negative-control and compression
gates have, RESULTS 25) it is applied as written and the rest of the plan continues.
`PLAN_COVERAGE.md` lists every revision-2 item with its status. The open items include
whole families (A, J, K, M, N, I), the condition and molecule-class axes as encoder inputs,
conformal calibration, composition residualisation, the lookup-table gate, the GC + 5-mer
composition gate, the head capacity sweep, the positive control, most of Aims 3 and 4, and
the compute accounting. Work proceeds in the order listed there, starting with the
missing families' ground truth.

---

## Families K and J implemented (stage 1); the release ladder (2026-11-21)

**Family K, RNA duplex and DNA-RNA hybrid** (`src/physics/duplex_classes.py`; 312 per-bp
and 98 whole-sequence channels). Three molecules per locus -- the RNA duplex, and the two
hybrids with the DNA strand x or rc(x) -- from cgNA+'s RNA and hybrid parameter sets.
Sourced and measured, not assumed: (i) cgNA+ reads a hybrid sequence along the DNA strand,
Watson DNA / Crick RNA (Sharma, Patelli, de Bruin, Maddocks, bioRxiv 2026, Methods), so
hybm(x) = hybp(rc x) exactly (verified to 0); (ii) both sets were trained with G-C ends
only (cgNA+ README), so molecules are computed GC-capped, and "GC" is its own reverse
complement, so the RNA duplex keeps exact reverse-complement consistency (verified to 0 on
every channel); (iii) groove widths at the A-form offsets -3 / +6 measured by the same P-P
scan that gives -4 / +4 in B-DNA, with index maps +3 / -6 (verified); (iv) A-form
signatures reproduced (RNA: slide -1.63 A, roll 8.6 deg, twist 30.2 deg, minor groove
11.2 A; hybrid intermediate); (v) hybrid thermodynamics from Banerjee et al. 2020 Table 2
**as corrected in 2021** (every row self-consistent, dG37 = dH - 310.15 dS); RNA duplex
from Turner 2004 via ViennaRNA (matches its own dimer MFE exactly). Not reused: Family D's
persistence and torsion estimators, which assume B-DNA geometry and give dsRNA 45 nm
against dsDNA 50 nm, the reverse of the cgNA+ authors' finding; the Hoogsteen proxy (a
B-DNA concept, not RC-consistent under the RNA set).

**Family J, single-stranded DNA** (`src/physics/single_strand.py`; 4 per-bp and 50
whole-sequence channels): each strand's self-structure under ViennaRNA's DNA parameter
set (Mathews 2004, the model Family E's hairpins use, matched exactly), at 0.15 M and
37 C with salt and temperature axes; J_m(x) = J_p(rc x) exactly. Not yet: single-strand
stacking, persistence and stretching, electrostatics, hydration, reactivity.

**Corpus integration without regeneration.** Both are generated as side shards aligned
to the 1,024 main shards (`src/data/generate_family_shards.py`) and merged per fragment
by `InMemoryLandscape(extra_dirs=...)`, which refuses a side shard whose sequences differ
from its main shard. The strand tie now covers every declared strand pair (L, K hybrids,
J) through `STRAND_PAIRS`.

**Pre-registration: the release ladder `v5`** (`scripts/submit_release_ladder.sh`): the
2M, 10M, 50M, 200M and 1B rungs (v3 widths, depths, ranks, learning rates) on families
B-H, J, K, L with v2.1 targets, the strand tie, and the whole-sequence auxiliary windows.
Reported on the same test side, family by family, beside the v3 and v4 ladders. Expected,
stated now: every v3 family's per-bp median within the 10M seed spread of its v4 rung or
better, Family L per-bp at or above the v4 rung, and Families J and K -- analytic channels,
the kind the per-bp families already reach 0.999 on -- at per-bp median 0.99 or above
from 10M up. Family A joins in `v6` when its production solves are done.

---

## Outcome: are the whole-sequence channels data-limited? (`v3_L10M_auxAll` against `v3_L10M`, test side)

| family (whole-sequence median) | v3_L10M | auxAll | change | seed spread (10M) |
| --- | --- | --- | --- | --- |
| C | 0.9938 | 0.9955 | +0.0017 | 0.0004 |
| D | 0.9969 | 0.9983 | +0.0014 | 0.0005 |
| F | 0.9870 | 0.9852 | -0.0018 | 0.0023 |
| L | 0.8665 | 0.8822 | +0.0157 | 0.0031 |
| G (control) | 0.9882 | 0.9885 | +0.0003 | 0.0004 |
| H (control) | 0.9992 | 0.9994 | +0.0002 | 0.0001 |

Every per-bp family non-inferior (C +0.0006, H +0.0006, L +0.074, others 0); all-channel
whole-sequence median 0.9952 -> 0.9965.

**Verdict under the pre-registered rule: partial.** C and D rise beyond the +0.001
threshold at three to four times their seed spread; Family L rises; the controls hold;
but Family F's median falls by 0.0018, which is inside its seed spread (0.0023) and so
neither a rise nor a measurable loss. The family median misleads for F, whose 39
whole-sequence channels have a gap near the median (0.970 against 0.987): per channel,
21 of 39 F channels rise by more than 0.002 and none falls by more than 0.002 (seed
replicate: median change 0.0000), so F's channels gain from the data even though its
median does not show it. Reported as partial, with the per-channel evidence beside the
median rather than in place of it. Consequence: the whole-sequence auxiliary windows are
in the final configuration (`v4f`, `v5`), and per-channel change counts are reported
alongside family medians from here on.

**v4 ladder, 50M rung (`v4_L50M` against `v3_L50M`, test side).** Family L per-bp 0.6579 ->
0.7739 (+0.116), whole-sequence 0.8573 -> 0.8834; Family F whole-sequence 0.9873 -> 0.9875
(+0.0002; per channel 8 down, 6 up by more than 0.005); worst other family -0.0008 (G,
whole-sequence); all-channel medians 0.9988 -> 0.9990 per-bp, 0.9956 -> 0.9962
whole-sequence. The arm passes every non-inferiority condition.

Against the two pre-registered readings: (1) **the F cost is not a capacity trade-off.**
It is absent at 2M (-0.0025, balanced per channel) and 50M (+0.0002, balanced) and
present only in the 10M run (-0.0135, 11 down / 3 up), so a monotone shrink with size
does not describe it; it is read as specific to that run, and the 200M rung is still
reported. (2) **L under the tie does not keep rising with size**: 0.734 (2M), 0.784
(10M), 0.774 (50M; 1.5 times the 10M seed spread below 10M). The tie removes a fixed
handicap (+0.09 to +0.13 at every rung) and L then plateaus near 0.78 from 10M; with the
auxiliary-data result (+0.08 at 10M and at 50M), data, not capacity, is the binding
limit for Family L in this regime -- which the release ladder (`v5`, with the auxiliary
windows) addresses. Both readings are recorded as measured; neither is re-worded.

---

## The composition gate: FAIL as computed, on a noise floor from a different construction (2026-11-21)

**As computed** (`src/physics/composition_gate.py`, 644 `compctrl` fragments in 50
composition-matched groups, 150 mM, thresholds as fixed in advance: SNR 5, eta 0.05,
pass fraction 0.5; `results/electrostatics/composition_gate.json`): **FAIL**. Within-group
(base-order) variance is above the floor in 100 percent of channels (median eta 0.65) and
composition explains little (median mononucleotide R-squared 0.29; maximum 0.84), so the
degeneracy the plan warns of -- the potential collapsing into composition -- is not what
happened. What failed is the noise criterion: 31 of 67 channels have within-group SD more
than 5 times the jitter SD (46 percent against 50). By sector, base-order signal / noise
(kT/e): major groove 0.021 / 0.0055, minor groove 0.046 / 0.0046, sector 1 0.041 / 0.0063,
and the backbone-facing sectors 2, 4, 5 at 0.018 / 0.0077, 0.014 / 0.0104, 0.032 / 0.0123.

**Why this does not settle the question.** Three facts found while reading the failing
channels:

1. *The noise floor came from another construction.* The eight jitter replicates were
   solved on 2026-11-19 with the single-start minimiser; the 644 gate fragments on
   2026-11-21 with the two-start minimiser (the alpha/gamma crankshaft fix). The gate
   requires the noise of the construction it tests.
2. *The "grid" replicates also differ in structure.* Moving the grid cannot move atoms,
   yet the replicates' surface radius -- a function of atom positions only -- differs in
   33-62 percent of cells in every sector except the minor groove, base-pair origins by
   up to 0.03 A, final energies by up to 180 kJ/mol.
3. *The minimiser stops on a cap, not at convergence.* `max_iterations=200` against a
   descent from +139,886 to about -35,200 kJ/mol leaves a maximum force of 92.5 kJ/mol/nm
   (RMS 9.0), so the end point depends on the path and small differences between runs
   (hardware rounding) move atoms. Converging it (tolerance 1 kJ/mol/nm) costs 13 s
   against 9 s, reaches maximum force 5.4 (RMS 0.99), changes the restrained-atom RMSD by
   0.013 A (0.311 -> 0.324) and keeps the same start by a 640 kJ/mol margin.

The computed FAIL is recorded as computed and not re-worded. It is not taken as the
plan's verdict either, because its denominator does not belong to its numerator.

**Change to the ground truth, decided on its own merits before any re-run.** Production
minimisation converges: `--min-iterations 20000 --min-tolerance 1.0` (`minimize.py`,
`run_fragment.py`; defaults unchanged, so every earlier result stays reproducible). A
minimiser that stops wherever 200 steps leave it gives labels that depend on the node.

**Pre-registration: the gate re-run.**
- *Binding:* the 644 `compctrl` fragments re-solved with the converged minimiser
  (`apbs/compctrl_v2`), against 16 jitter replicates of the same construction
  (`apbs/jitter_v2_converged`), each with grid jitter 0.175 A *and* the raw build
  displaced by N(0, 0.001 A) (the PDB's precision), so the floor includes the minimiser's
  sensitivity to rounding-sized differences rather than depending on which nodes the
  replicates land on. Thresholds unchanged: SNR 5, eta 0.05, pass fraction 0.5.
- *Diagnostic, reported beside it:* the original 644 against 16 replicates of their own
  (capped) construction made the same way (`apbs/jitter_v2_capped`).
- *If the binding gate fails,* the plan's instruction holds: the ground truth changes
  before production compute. Next in line: averaging the potential over each sector's
  arc rather than reading one ray (the channels are single-point samples today), then
  replicate averaging.
- `src/physics/noise_study.py` separates grid from structure noise on one sequence for
  both minimisers; it is a diagnostic, and its projection of the gate is labelled as one.

**Seed spread at 10M with a third seed (`v3_L10M_seed2`, test side).** Whole-sequence
family medians over three seeds, mean +/- SD: C 0.9935 +/- 0.0002, D 0.9972 +/- 0.0003,
F 0.9853 +/- 0.0051 (0.9870, 0.9893, 0.9796), G 0.9886 +/- 0.0004, H 0.9991 +/- 0.0001,
L 0.8690 +/- 0.0023; per-bp L 0.6633 +/- 0.0047, every other per-bp family SD 0.0006 or
less. Family F's whole-sequence median is the one unstable summary in the model (its 39
channels have a gap near the median), so readings of it are restated against the
three-seed SD: the strand tie's 10M F change (0.9735) is 2.3 SD below the seed mean, not
outside the seed distribution's reach; `auxAll`'s 0.9852 equals the seed mean. Per-channel
counts, which do not move with the median's position in a gap, are the primary F evidence
from here on. The pre-registered thresholds stated against "the seed spread" use this
three-seed SD from now on; the verdicts already recorded are not re-scored.

**Noise decomposition** (`results/electrostatics/noise_study.json`; the gate sequence,
150 mM, 8 replicates per arm). Median SD of the gate's channels, kT/e:

| arm | capped minimiser (as run) | converged minimiser |
| --- | --- | --- |
| grid placement only (one structure, 8 grid jitters) | 0.00033 | 0.00026 |
| structure only (raw build displaced by 0.001 A, one grid) | **0.00455** | 0.00074 |

Grid placement, the thing the gate's noise floor was meant to measure, is 10 to 30 times
below that floor. The floor was the minimiser: a 0.001 A difference in the input moves
the capped minimiser's end point by a median 0.024 A and up to 2.1 A, against 0.006 A
median for the converged one (base-pair origin SD 0.022 A against 0.005 A). Both keep
the same start in 8 of 8.

**Diagnostic gate** (pre-registered above): the original 644 fragments against 16
replicates of their own capped construction, made with the 0.001 A perturbation
(`results/electrostatics/composition_gate_diag_capped_matched.json`): **PASS**, 49 of 67
channels with SNR above 5 (73 percent), median SNR 7.4, eta and composition R-squared
unchanged (0.65, 0.29). With the noise floor of the construction actually tested, the
original fragments pass; the first FAIL was the mismatched denominator. The binding gate
is the converged construction, still running; the labelled projection for it is 98
percent of channels above SNR 5.

---

## Quantum chemistry: the hexamer tier was computed one base off centre (2026-11-21)

The tier-1 hexamer run (GFN2-xTB, 2,080 reverse-complement-canonical 6-mers) was split
from one 20-hour, 32-CPU request that had waited since 2026-11-19 into 52 four-CPU array
tasks, and finished in about 40 minutes. Its first consistency check failed: the
stacking energy averaged over a central tetramer's 16 hexamer flanks did not track the
tetramer tier at all (Pearson r = -0.20; CG steps -19.5 kcal/mol in the tetramer tier and
about -12 in the hexamer tier, GA steps the reverse).

Cause: `build_context_geometries` placed the step at `p = flank + 2` for every context
length -- the centre of a tetramer, but positions 2-3 of a hexamer, one context base on
the left and three on the right. Paired with the tetramer around the step actually
computed, the hexamer tier agrees with the tetramer tier to r = 0.9989 (mean |difference|
0.08 kcal/mol), so the calculations were right and only the centring was wrong. Fixed
(`central_step_index`, tested for 4-, 6- and 8-mers; tetramers unchanged); the centred
hexamer library is rebuilt and recomputed (`qc/hexamer_v2`). The off-centre results are
kept in `results/superseded/` and were never read downstream: Family B's per-fragment
channels use the tetramer tier only (`src/physics/electronic.py`).

---

## 500 mM restored: the plan's own fragment-length rule makes it solvable (2026-11-21)

The 2026-11-19 entries dropped 500 mM because nonlinear PB diverged on every 61 bp
solve, and the linearised substitute was rejected (0.5-0.76 SD error). Under the
"never cut back" instruction that drop has to be undone, and the plan itself gives the
route: "Fragment length is set per ionic condition by the screening length rather than
fixed." At 500 mM the screening length is 4.3 A; the plan's flank estimate is 7 bp at
300 mM and less at 500 mM, so a 61 bp solute is several times longer than needed, and
the divergence grows with solute length (121 bp diverges from 50 mM up where 61 bp does
not).

Probe (`convergence_v1/probe500`, 12 sequences, seed 9000 outside the study's seeds,
each built at 61 bp and trimmed so the central 21 bp keep their full-context geometry):
61 bp **fails 12 of 12**; 41, 31 and 21 bp **converge 12 of 12** (36-42 s each).

Decision: 500 mM is a production condition again, solved on the central 41 bp (10 bp of
flank either side of the scored 21) unless the convergence study, now extended to 500 mM
at 41/31/21 bp over the study's 2,004 sequences, shows 41 bp short of converged. The
nine conditions of the plan are all back: seven NaCl concentrations, KCl and the Na/Mg
mixture. The study also covers 25 mM, KCl and Na/Mg now, since the plan asks for the
convergence curve "at every ionic condition".

**What this opens, stated now.** Per-condition solute length cuts both ways: the 10 mM
pilot left 0.34 SD of error at 61 bp, so low salt needs more context than the corpus's
61 bp fragments carry. Genome-derived fragments can be extended from the genome; the
synthetic strata (random, extremes, hexamer, composition controls) cannot without making
the label depend on sequence the encoder does not see. That design question is settled
when the convergence curves land, before production.

**The first merged cache build failed on a bug in the side-shard merge** (job 34100552:
"shard_0000.h5: channel layout differs from shard_0000.h5"). The merge appended the side
shards' channel names to the metadata before the per-shard layout check, which then
compared every main shard against the merged list -- so it failed for any `extra_dirs`,
on the first shard. No test had built a merged dataset. Fixed (each shard checked against
the layout of its own kind) and covered by `tests/test_extra_dirs.py`: merged values
equal the side shards' per fragment after undoing the standardisation, a misaligned side
shard and a changed side layout are both refused, and the cache round-trips. The v5
ladder, pinned to a snapshot with the bug, was cancelled before any link ran and
resubmitted on a fixed snapshot behind the rebuilt cache (job 34104114).

---

## Family A extraction completed for the map-derived channels; KCl was a duplicate of NaCl (2026-11-21)

**Extraction** (`src/physics/electrostatics_channels.py`, `run_fragment.py --extended
--reference-solve`): per scored base pair and condition, the field (radial, tangential,
axial) and radial field gradient 1 A outside the surface in each groove; log10 Boltzmann
enrichment of monovalent cations, anions and divalent cations at groove sites (the groove
ray +-15 deg, 0-4 A out) and backbone sites (each phosphate's Voronoi cell within 7 A);
the condensed counterion charge per phosphate per strand; the phosphate-cell potential
per strand (strand asymmetry); the most favourable energy of a +1 e, 2 A probe in each
groove; per base pair the local Manning parameter and surface charge density; per
condition the electrostatic solvation free energy against a uniform-dielectric, ion-free
reference on the identical grid. On a real 31 bp map at 150 mM: groove field -0.42 /
-0.47 kT/(e A) (major / minor, pointing inward), cation enrichment 15x / 27x bulk, anion
depletion 13x / 21x, Mg2+ about the square of the monovalent factor, condensed fraction
0.40 per phosphate within 7 A, Manning xi 4.33 (B-DNA textbook 4.2), surface charge
0.0108 e/A^2.

**Reverse-complement check** (x against rc(x), three conditions; rmse over the channel's
SD after the exact mapping -- sector s <-> 6-s, Watson <-> Crick, base pairs reversed,
tangential and axial field negated): surface potentials 0.015, groove enrichment 0.002,
backbone enrichment 0.001, probe 0.016, phosphate potential 0.019, condensed fraction
0.055, field components 0.04-0.08 -- and the radial field gradient **0.52, a failure**:
its second difference at 0.25 A on a 0.32 A trilinear map was interpolation noise. Now
+-1 A (first derivatives +-0.5 A); rerun pending.

**KCl was numerically identical to NaCl at 150 mM** -- solvation energy equal to 0.01
kJ/mol, condensed fraction 0.406 in both. APBS builds one ion-accessible region from the
largest ion radius in the input, which was chloride's 1.67 A in every condition, so the
plan's cation-identity axis carried no information and the Mg mixture did not use
magnesium's size either. Production conditions (`PRODUCTION_CONDITIONS`) now write every
ion line with the counterion's radius: Na+ 1.16 A for the NaCl series and the Na/Mg
mixture, K+ 1.52 A for KCl. KCl then differs from NaCl 150 mM in ion size alone and the
mixture in valence alone. Ion identity enters mean-field PB only through that exclusion
radius -- hydration-specific binding is outside the model -- so KCl joins the Mg mixture
as `approximate`. Earlier studies keep the APBS default (`IONIC_CONDITIONS`) and stay
reproducible.

**Consequence, pre-registered.** The final production construction (converged
minimiser, counterion accessibility, extended extraction, reference solves,
per-condition solute length) is gated once more before production: the 644 `compctrl`
fragments and 16 perturbed jitter replicates re-solved on it, thresholds unchanged. The
convergence study keeps the default accessibility: its question is long-range screening
(the flank needed), which a 0.5 A change in the exclusion radius does not reach; a
48-sequence check at 10 and 150 mM on the production construction confirms that or does
not.

**Still open for Family A:** the electrostatic component of base-pair opening (needs
opened-conformation solves; plan: a first-order estimate validated on a subset of exact
solves), the electrostatic persistence length over 4 windows (the plan's long-fragment
set), the ensemble spread (10 percent x 8 cgNA+ configurations), and the low-salt
long-context design.

## Family N: sourced route for m6A and pseudouridine; 2'-O-methyl unsourced (2026-11-21)

ViennaRNA 2.8, the folding model of Families E, J and L, carries modified-base energy
parameters with published sources (read from the parameter files' metadata on the
ViennaRNA GitHub `misc/` folder): m6A, Kierzek et al. 2022, Nat Commun 13:1271
(10.1038/s41467-022-28817-4); pseudouridine, Hudson, Bloomingdale & Znosko 2013, RNA
19:1474 (10.1261/rna.039610.113); inosine, Wright et al. 2007 Biochemistry 46:4625 and
Wright, Force & Znosko 2018 NAR 46:12099; 7-deaza-adenosine, Richardson & Znosko 2016 RNA
22:934; purine, Jolley & Znosko 2017 NAR 45:1479. 2'-O-methyl has no parameter file there,
so it is open until a source is found. Stage 1 of Family N: the response of the Family L
channels to modifying each modifiable position (m6A at A, pseudouridine at U) on both
transcripts. The installed cgNA+ also ships a methylated-DNA set
(`Di_hmethyl_methylated-hemi_combine`), which makes a DNA-methylation response of the
Family C channels possible, an addition beyond the plan's list.

**Binding gate: PASS** (`results/electrostatics/composition_gate_binding_converged.json`;
644 `compctrl` fragments re-solved with the converged minimiser, against 16 jitter
replicates of the same construction with the 0.001 A input perturbation; thresholds as
fixed in advance). 60 of 66 channels with SNR above 5 (91 percent), median SNR 28.8
(major groove 28.8, minor 43.4, backbone-facing sectors 2 / 4 / 5 at 23.1 / 12.2 / 52.3);
within-group share of variance above the floor in every channel, median 0.68;
mononucleotide composition R-squared median 0.26, maximum 0.83. The six channels below
threshold are four of the outermost surface offset (11 A out, where the potential is
smooth) and two axis-referenced points. The electrostatics ground truth separates
composition-matched sequences, which is the check the plan puts "ahead of training
anything". As pre-registered, the final production construction is gated once more
before production solves begin.

---

## The convergence study as first run measured grid and minimiser differences, not only flanks (2026-11-21)

The 50-250 mM group finished at plan scale (2,004 sequences;
`results/electrostatics/convergence_50_250mM.json`): no length converged within the 0.05 SD
tolerance -- 41 bp against 61 bp left 0.247 SD at 50 mM and 0.144 at 150 mM, 31 against
41 bp 0.188 at 250 mM. The shape was wrong for a flank effect: at the innermost radius the
discrepancy barely moved between 41 and 31 bp (0.113 against 0.126 at 150 mM), where a
missing flank would make the shorter solute clearly worse.

Two confounds in the protocol (`channels_for`), both found by reading the code:

1. *Each length was re-minimised separately* (`solver_probe.prepare_structure` trims the
   raw build, then minimises with the 200-iteration default), so every length comparison
   also compared two minimiser end points -- the path noise measured the same day at a
   median 0.0046 kT/e per fragment-mean channel, more per base pair.
2. *Each length's grid was sized from its own solute.* `dime` serves both focusing
   levels; along the helix axis the 61 bp coarse box needs 289 points and the fine box
   257, so the 61 bp reference was solved at 0.306 A axial spacing and the shorter
   solutes at 0.344 A. A spacing change of that size is known to move channels (0.5 A was
   rejected at 0.305 SD).

The same mechanism reaches production: at every condition whose coarse box outgrows the
fine box (low salt, long solutes) the fine level was finer than specified -- about 0.2 A
at 10 mM -- so labels at different conditions carried different discretisations, which
a model learns as salt dependence.

Changes. (i) `exact_fine_spacing`: when the coarse box forces `dime` up, the fine box is
widened to (dime - 1) x 0.35 A, so every fine level is solved at the plan's 0.35 A on
every axis; production uses it (`--exact-fine-spacing`). (ii) A fixed protocol for the
study (`fixed_grid_sequence`, `--fixed-grid`): one converged minimisation at the longest
length, shorter solutes trimmed from that structure without re-minimisation
(`run_fragment.trim_pdb`), and every length solved on the reference solute's grid at
exactly 0.35 A, so only the removed flank atoms differ. The not-yet-started tasks of the
low-salt, 500 mM and ion-identity groups were cancelled (they used the first protocol).
A 12-sequence validation at 50 and 150 mM on the same sequences as the first protocol's
chunk 1000 (which gives 0.241 and 0.146 at 41 bp) runs before the study is resubmitted.

---

## Families N and I implemented (stage 1) (2026-11-21)

**Family N, RNA modification response** (`src/physics/rna_modification.py`; 12 per-position
and 30 whole-transcript channels, 1.0 s per fragment). For both transcripts, the change in
ensemble free energy and in the site's unpaired probability when one position is modified
-- m6A at A, pseudouridine at U, and inosine at A (A-to-I editing, an addition beyond the
plan's list) -- with ViennaRNA 2.8's sourced modified-base parameters, plus per-transcript
summaries and the fully modified transcript. Checks: N_m(x) = N_p(rc x) exactly; on a
hairpin, m6A destabilises paired A by +0.57 to +1.40 kcal/mol (Kierzek et al. 2022 report
0.5-1.7) and pseudouridine stabilises paired U by -0.77 to -2.50 kcal/mol (the direction
of Hudson et al. 2013); unpaired loop positions are unchanged. Strand pair tied in
`STRAND_PAIRS`. Side shards generating (`targets_v2_famN`); the v5 ladder was requeued
behind them so the release ladder carries Family N. Open: 2'-O-methyl (no sourced
parameters), the hybrid class, modified-RNA geometry.

**Family I, protein readout, held apart** (`src/physics/protein_readout.py`; 12 per-bp and
48 whole-fragment channels -- the plan's 60). Twelve ProBound fits from MotifCentral
v1.0.0 (Rube et al., Nat Biotechnol 40:1520, 2022), one per major structural class: CTCF,
MAX, CREB1, NFKB1, SOX2, FOXA1, GATA3, NR3C1, GABPA, HOXA9, TEAD1, POU5F1; coefficients,
fit ids and source HT-SELEX experiments in `data/probound_panel_v1.json`. Every model's
top-scoring bases were checked against the factor's known site before use (CACGTG for
MAX, the palindromic kB site for NFKB1, TGTTTAC for FOXA1, AGATAAG for GATA3, CCGGAAGT for
GABPA, a GGTACAnnnTGT element for NR3C1, the CTCF core AGGGGGCGC). Per base: the binding
free energy of the best site covering it, relative to the optimal site, on either strand
(exactly reverse-complement symmetric, tested). Per fragment: occupancy at three chemical
potentials calibrated on a fixed sample (1,496 fragments, the first 8 main shards) so the
median fragment sits at 10 / 50 / 90 percent, and occupancy against nucleosome competition
with the fragment's own Family D free energy read from its main shard. Held apart as the
plan requires: its side shards (`targets_v2_famI`) are never passed to the physics
ladders; they enter only the labelled protein-readout variant. Open: "generic amino acid
side chain interaction energies in each groove", which ProBound does not supply.

**GPU priority (2026-11-21).** My GPU fair share admits about one running link at a time,
so the release ladder (v5) goes first. The two exploratory pair-attention runs at 10M
(`v3_L10M_axtie` at epoch 42, `v3_L10M_axtiebpp` at epoch 24; my architecture probes, not
plan items) are deferred, not abandoned: their pending links are cancelled and their
checkpoints kept, to resume when capacity allows. `v4_L200M` (epoch 96 of 100) and the
pre-registered `v4f_L200M` continue.

**Family A channel layout** (`src/physics/electrostatics_family.py`; 470 per-bp + 9
whole-fragment = 479 declared). Per condition (9): surface-referenced potential at 6
sectors x 6 offsets; per groove the radial, tangential and axial field, the radial field
gradient and the best +1 e probe energy; condensed counterion charge per strand; strand
and groove asymmetry; solvation free energy. Condition-independent: Na+, K+, Mg2+, Cl-
enrichment at major, minor and backbone sites (each from its own condition), the local
Manning parameter and surface charge density, and the salt slope of each sector's
potential over the NaCl series. Values sit on the scored central 21 bp; unsolved
conditions stay NaN (anchoring). Paired members (sectors 4 and 5, the Crick phosphate) are
stored in their own frame and tied in `STRAND_PAIRS` exactly like L_m = L_p(rc); groove
quantities carry SYMMETRIC / ANTISYMMETRIC parity. Tested exactly on a synthetic solver
output and its exact reverse-complement image (`tests/test_electrostatics_family.py`, all
identities to 1e-12); on a real x / rc(x) pair the residual is the two independent builds'
structural difference (median rmse/sd 0.13-0.29 against 1.29 for a deliberately wrong
mapping). The plan's 560 is reached with the open items: base-pair opening, the windowed
persistence length, the ensemble spread.

---

## Family M, non-canonical folds as explicit geometry: staged design (2026-11-21)

PLAN.md: "Quadruplexes, i-motifs, intramolecular triplexes and cruciforms treated as real
conformations rather than propensity scores, including the quadruplex central ion
channel" (about 90 channels; ground truth "solved structures, MD ensembles, APBS and
quantum chemistry on the folded geometries"). Family F already carries the propensities
(quadruplex free energy and topology probabilities, i-motif at 3 pH, Z-DNA, cruciform,
triplex); M is the geometry and physics of the folded state itself, conditional on the
fold.

**Construction.** A template library of solved structures per class -- G-quadruplexes
spanning parallel, hybrid and antiparallel topologies with propeller, lateral and diagonal
loops, i-motifs, intramolecular triplexes (H-DNA) and four-way junctions for cruciforms --
from the PDB, annotated by topology and loop lengths. Per fragment and strand: the
candidate fold (the Family F motif calls), the best-matching template (class, topology,
loop lengths), the candidate's sequence threaded onto it (base substitution on fixed
backbone, then the converged restrained minimisation), and on that structure the same
solvers as Family A (APBS at the anchor conditions with explicit K+ in the G4 channel
sites) and the QC tier (stacked tetrads, xTB). Channels: fold present and class;
geometric descriptors (tetrad count, rise and twist between tetrads, tetrad planarity,
the four G4 groove widths, loop lengths and geometry; i-motif C:C+ stack spacing; triplex
Hoogsteen-strand geometry; junction arm angles); electrostatics (central-channel
potential profile and its minimum, coordination-site potentials, groove potentials,
solvation free energy of the fold); stacking energy of tetrads. NaN where the fragment
has no candidate.

**Stages.** (1) G-quadruplexes, parallel topology first (the dominant fold for short
loops), then hybrid and antiparallel; (2) i-motifs; (3) triplexes; (4) cruciforms.
Template threading is validated before use by threading each template's own sequence
back onto a different template of the same class and comparing to the solved structure.

**Fixed-protocol validation** (`results/electrostatics/convergence_v2_validation.json`; the
same 12 sequences as the first protocol's chunk 1000). Per shell, rmse over SD of 41 bp
against 61 bp:

| shell | 50 mM, first protocol (pooled) | 50 mM, fixed | 150 mM, fixed |
| --- | --- | --- | --- |
| axis radii 5 / 7 / 9 / 11 / 14 / 18 A | 0.241 pooled, ~0.12 innermost | 0.012 / 0.043 / 0.048 / 0.064 / 0.13 / 0.47 | 0.002 / 0.008 / 0.011 / 0.014 / 0.031 / 0.135 |
| surface offsets 0 / 1 / 2 / 4 / 7 / 11 A | | 0.054 / 0.063 / 0.077 / 0.13 / 0.27 / 0.63 | 0.012 / 0.014 / 0.018 / 0.033 / 0.086 / 0.272 |

The 0.11 floor at the inner shells is gone (0.002 at 150 mM), confirming it was the
protocol. What remains is a genuine flank effect confined to the outer shells -- 20 A from
the axis the potential feels charges far along the helix -- and growing as salt falls.
Convergence is therefore judged per shell (`shortest_converged_bp_by_shell`), and a
production length is the shortest that converges every shell, with any shell that does not
converge at the longest solvable reference flagged rather than silently kept.

**Solvable references limit the study.** The nonlinear solver diverges on 121 bp solutes
from 50 mM up and on most 61 bp solutes at 250 mM (first protocol: 16 percent), so the
reference is 121 bp at 10 and 25 mM, 61 bp at 50-150 mM, KCl and Na/Mg, 41 bp at 250 and
500 mM. Resubmitted at plan scale on the production construction (`--fixed-grid
--production-conditions`): 2,004 sequences per condition, all nine conditions. Open,
recorded: whether 61 bp is itself converged in the outer shells at 50-150 mM needs a
longer solvable reference; the candidate remedy is a linear coarse level feeding a
nonlinear fine level (the potential at a fine-box boundary 12 A from the surface is in the
linear regime at 150 mM, not at 10 mM, where npbe converges anyway).

**Family J extended: single-strand stacking** (oxDNA2 sequence-dependent stacking,
`data/oxdna2_stacking.json` with the parameter file's sha256; Snodin et al. 2015, Sulc et
al. 2012). Read from oxDNA's own source rather than assumed: the stacking term takes
q = p->n3 and F1_EPS[q][p], so a 5'-YX-3' step uses STCK_X_Y, scaled by
(1 - f + 9 f T / 3000 K) with f = 0.18; that ordering reproduces SantaLucia's (5'-AT-3' >
5'-TA-3', 5'-GC-3' > 5'-CG-3'). Per base the mean strength of the steps touching it at 25,
37 and 55 C (well depths: AA 10.9, TA 9.2 kcal/mol at 37 C) and per strand the mean; 12
channels, both strands, tied. The v5 cache was already built around the first Family J
shards, so these channels enter at v6 (`targets_v2_famJ2`). Still open for J:
persistence and stretching (the published single-strand values are salt-dependent, not
sequence-dependent: Chen et al. PNAS 2012, Bosco et al. NAR 2014), single-strand
electrostatics (no single-strand conformations from cgNA+), hydration, reactivity.

**v4 ladder complete: 200M rung** (`v4_L200M` against `v3_L200M`, test side). Family L
per-bp 0.6309 -> 0.7470 (+0.116), whole-sequence 0.8582 -> 0.8815; Family F whole-sequence
0.9823 -> 0.9838 (+0.0015; per channel 10 up, 3 down by more than 0.005); worst other
family -0.0007 (D, whole-sequence); all-channel medians 0.99927 -> 0.99927 per-bp, 0.9950
-> 0.9962 whole-sequence. Across the ladder the tie adds +0.09 to +0.13 to Family L per-bp
at every rung and costs nothing measurable elsewhere; the 10M F loss does not recur at
50M or 200M. Family L per-bp under the tie does not rise with capacity (0.734, 0.784,
0.774, 0.747 from 2M to 200M), which with the auxiliary-data gains (+0.07 to +0.08)
confirms the data-limit reading; the release ladder carries the auxiliary windows.

**Final gate on the full production construction: PASS** (converged minimiser, counterion
accessibility, exact 0.35 A fine spacing off, extended extraction, reference solves; 644
`compctrl` fragments against 16 perturbed replicates; `composition_gate_final_production.json`):
91 percent of channels with SNR above 5, median SNR 29.3, within-group share 0.69,
composition R-squared median 0.23.

**Per-channel signal against replicate noise** (`src/physics/channel_snr.py`,
`results/electrostatics/channel_snr_production.{json,tsv}`; every Family A channel solved
at 150 mM, pooled over the scored base pairs): all 59 have within-composition-group SNR
above 5. Surface potentials median 21 (minimum 5.7, the outermost offset); groove fields
16-69; field gradients 13-24; condensed charge per strand 27-28; strand and groove
asymmetry 53 and 39; ion enrichment 33-76; probe energies 13.5 (major) and 9.8 (minor);
Manning parameter 59; surface charge 27; solvation free energy 9.2. Every extracted
quantity carries base-order signal well above its own construction noise; the KCl and
Mg channels are measured when their production solves exist.

**Hexamer tier, centred** (`results/qc_hexamer_gfn2.json`): 2,080 of 2,080 contexts, the
step at positions 2-3 of every context. The mean stacking energy over each central
tetramer's 16 hexamer flanks matches the tetramer tier to r = 0.9985 (mean |difference|
0.10 kcal/mol); the outer flanking bases move the central step's stacking by a median
0.10 kcal/mol (between-tetramer SD 2.1), the size of the context effect the hexamer tier
exists to supply.

---

## Incident: the home fileset hit its file-count quota (2026-11-21, 04:45-05:10)

`/home/bjc75/scratch` and `/home/bjc75/project` are directories inside the **home** fileset
(roberts:home, 125 GiB and 500,000 files, per user), not the project or scratch filesets
their names suggest; only `/nfs/roberts/scratch/pi_skr2/bjc75` is group scratch. At
midnight home was at 491,792 files (98 percent); today's code snapshots, quantum-chemistry
geometry libraries and thousands of array-task logs pushed it over. Every job that then
tried to create a file in home or print to a log there failed with EDQUOT: the merged
cache build (its traceback could not even be written), one link of `v4f_L200M` (at a
print; its checkpoint of 04:55 and best model of 05:01 are intact), a v5 link, and 17
convergence tasks.

Recovery: all pending jobs held; the superseded off-centre hexamer library (16,642 files)
and the logs of completed jobs (8,687) moved to group scratch (`pangea/archive_home/`);
`logs/slurm` replaced by a symlink to group scratch (the old directory renamed, so running
jobs kept their open logs); the current hexamer library moved behind a symlink;
`freeze_code.sh` now writes snapshots to group scratch. Holds released; the cache build
and the v5 ladder resubmitted (34110665 and its links); the 17 failed convergence tasks
resubmitted with their original seeds. Training checkpoints and the corpus caches were
never at risk: `runs_v3` and `targets_v2` were already symlinks to group scratch. Rule
from here: anything that creates many files (production solves, array logs, geometry
libraries, snapshots) goes to group scratch, which purges files untouched for 30 days, so
final artifacts are copied to backed-up storage at the end.

---

## Correction: the nonlinear total energy is broken on 61 bp solutes; solvation redefined (2026-11-21)

**Retracted:** the "solvation free energy 9.2" entry in the per-channel SNR table above.
It was computed from `elec_energy_kJ_mol - elec_energy_ref_kJ_mol`, and the nonlinear
solve's total energy is not a number for most fragments: |E| > 1e8 kJ/mol in 618 of 644
`compctrl_v3` fragments and all 16 of their replicates (values to -6e20), and in 620 of
644 `compctrl_v2` fragments at the 1.67 A default radius too, so not caused by the
production exclusion radius. Short solutes are fine (31 bp: -54,329 kJ/mol; the 41 bp
solute at 500 mM: -81,313). The values equal e^40 to the digit on the reproduced case:
the fine box covers the scored region plus margin, so on a 61 bp solute it cuts the helix
along the axis, phosphates sit on the focusing boundary (APBS's own "large potential
values on the focusing boundary" warning), and the nonlinear functional's ionic term
exp(-z phi) overflows there wherever screening is weak. The uniform-dielectric reference,
with no ions, is unaffected (643,513-671,490 kJ/mol). The sampled potentials of the scored
region are not affected: they sit 10 A or more inside the fine box, and the gate and the
per-channel SNR of every potential-derived channel stand.

**Redefinition.** Solvation is taken from per-atom energies (`calcenergy comps`): each
atom's charge-potential share 1/2 q_i (phi_solv - phi_ref) on identical grids, summed per
scored base pair over both nucleotides (`A_bpsolv_<condition>`, a new per-bp channel) and
over the scored 21 bp (`A_solvation_<condition>`, the solvation energy of the central
region in its full context). Scored atoms are far from the boundary. For the nonlinear
equation this is the charge-potential form used in MM-PBSA, which omits the mobile ions'
entropic term; flagged approximate. Checked on a real solve before production.

**Also measured:** APBS's per-atom energies do not sum to its nonlinear total even with
every atom inside the fine box (466,956 against 405,624 kJ/mol on a 31 bp map): the total
is the full free-energy functional, the per-atom values the charge-potential shares.

**Pre-registration: gate on the exact production construction** (converged minimiser,
counterion accessibility, exact 0.35 A fine spacing, extended extraction, per-atom
solvation, reference solves). The earlier "final" gate predates exact spacing and the
solvation redefinition, so it is repeated on what production will run: 644 `compctrl`
fragments and 16 perturbed replicates (`apbs/compctrl_v4`, `apbs/jitter_v4_production`,
now in group scratch), thresholds unchanged, plus per-channel SNR including the new
`A_bpsolv` / `A_solvation`. Production waits for it.

**Linear-coarse / nonlinear-fine focusing does not rescue long solutes** (probe 34110934,
150 mM): at 121 bp the standard mg-auto nonlinear solve overflows (301-324 s) and so does
the two-level scheme (lpbe coarse, npbe fine with focusing boundary; 67-68 s); at 61 bp
both converge (25 / 20 s). The divergence is in the fine-level nonlinear solve where its
box cuts a long helix -- the same boundary mechanism as the broken totals -- not in the
coarse level. The open question it was meant to answer (is 61 bp converged in the outer
shells at 50-150 mM?) is answered from the study's own 31 / 41 / 61 bp trend instead, by
fitting the residual against flank length and extrapolating, rather than from a reference
the solver cannot produce.

**Family M groundwork: G-quadruplex templates** (`src/physics/fold/g4_templates.py`; six PDB
structures in `ext_untrusted/pdb_g4_v1`: 1KF1 and 1XAV parallel, 143D antiparallel basket,
2HY9 / 2JPZ / 2GKU hybrid). Tetrads are found from Hoogsteen contacts alone (each guanine's
N1 and N2 within 3.3 A of its neighbour's O6 and N7, closed four-cycles), not from file
annotations: three tetrads in every template. Parallel stacks rise 3.38-3.47 A with
24.6-29.7 deg twist; the X-ray K+ ions of 1KF1 sit on the axis (radial <= 0.12 A) between
tetrads (axial +1.68 / -1.83 A) with a third at -5.13 A, and 1XAV's two K+ at +-1.9 A; the
NMR structures carry no ions, so channel sites are placed midway between tetrad planes. Next:
threading (loop substitution on fixed backbones), restrained minimisation, and the APBS
central-channel profile.

**Solvation, checked on a real solve** (61 bp fragment; 150 mM and KCl at 61 bp, 500 mM
trimmed to 41 bp): per-base-pair charge-potential solvation median -2,506 / -2,505 /
-2,189 kJ/mol, scored-21 sums -52,583 / -52,572 / -45,699. Definition recorded with its
caveat: the reference is the solute in a uniform dielectric of 2 without ions, whose
long-range Coulomb interactions are unscreened, so the solvation energy of the scored
region grows with solute length (the reason the 41 bp value at 500 mM is less negative
than at 150 mM despite stronger screening). Within a condition every fragment is solved at
that condition's production length, so values are comparable across fragments; across
conditions solved at different lengths they are not, and the salt dependence is read from
the potentials and ion channels instead.

**Correction to the explanation above** (whole-box probe 34111942, 150 mM). With the fine
box containing the entire solute at 0.7 A (no focusing boundary through the helix), the
121 bp solute still overflows (472 s) and the 61 bp solute's nonlinear total is still
broken (-2.8e13 kJ/mol). So the cut boundary is not the cause, and that statement is
withdrawn. The working hypothesis is now the mean-field breakdown itself: grid points that
are ion-accessible but close to phosphate charges reach |phi| of tens of kT/e, where
Boltzmann densities exp(|phi|) x bulk are unphysical; the energy functional overflows on
61 bp solutes and the solver diverges on 121 bp ones, while short solutes stay below the
threshold. Being tested: the ion exclusion radius (1.16 / 2.0 / 2.5 A) on the failing
case. Unchanged: the scored-region potentials (gate, per-channel SNR) and the per-atom
solvation, which reads phi only at atom positions.

**Convergence judged on fragment-specific scatter, not on the common offset** (plan scale,
fixed protocol; KCl 150 mM n = 1,692, NaCl 150 mM n = 996, NaCl 50 mM n = 1,056 so far).
The discrepancy between solute lengths splits into a common offset -- the same for every
fragment, since a shorter solute makes the far field uniformly less negative; removed by
per-channel standardisation at a fixed production length and carrying no sequence
information -- and the fragment-specific scatter, which is the flank sequence changing the
label. 41 against 61 bp, over SD, by shell (axis 5-18 A; surface 0-11 A):

| | offset, 150 mM | scatter, 150 mM | offset, 50 mM | scatter, 50 mM |
| --- | --- | --- | --- | --- |
| inner shells | <= 0.035 | <= 0.014 | <= 0.13 | <= 0.05 |
| axis 18 A | 0.14-0.15 | 0.055-0.060 | 0.51 | 0.16 |
| surface 7 A | 0.08-0.09 | 0.035-0.037 | 0.28 | 0.09 |
| surface 11 A | 0.26-0.28 | 0.12 | 0.64 | 0.22 |

Most of the outer-shell discrepancy is offset. Convergence is now decided on the scatter
(`scatter_within_tolerance`, `shortest_scatter_converged_bp_by_shell`), the offset
reported beside it. At 150 mM a 61 bp solute leaves the inner shells clean and the outer
shells with modest context noise; at 50 mM the outer shells need the study's full answer.

**Long windows prepared** (`src/data/extend_windows.py`): each corpus fragment with F bp of
flank either side (121 bp at F = 30). Panel fragments take their flanks from the source
genome (indexed FASTA; 0-based start, half-open, reverse complement on the minus strand,
as `sample_fragments.py` sampled them) and every window's centre is checked base for base
against the fragment; MPRA and synthetic fragments, and genome edges, get uniform random
flanks seeded by the fragment id, which the encoder sees. Test on 1,500 fragments: 948
genome-flanked, 552 random-flanked, **0 centre mismatches**. Used for whichever Family A
conditions the convergence study shows need context beyond 61 bp; the training side (a
second, long-window data source with its own forward pass) follows that decision.

---

## Production ion exclusion radius: back to the 2 A convention (2026-11-21)

Radius probe (34112824; 150 mM). At 121 bp the nonlinear solve overflows at exclusion radii
1.16, 2.0 and 2.5 A alike, so long-solute divergence is not a pocket effect. On 61 bp the
broken nonlinear total is the same number at every radius (-1.616e16 kJ/mol), so the total
energy's pathology does not come from ions near charges either; it is not used (solvation
is per-atom). What the radius does change is the sampled potential: minor groove 1 A out
-3.118 / -4.215 / -4.872 kT/e and 11 A out -0.928 / -1.073 / -1.172 at 1.16 / 2.0 / 2.5 A.

The bare Na+ radius (1.16 A) was chosen only to separate K+ from Na+; it admits ions closer
than mean-field PB supports and moves the near-surface potential by about 35 percent
against the conventional ~2 A Stern layer of nucleic-acid PB work (APBS and DelPhi
practice). Production now uses 2.0 A for the NaCl series and the Na/Mg mixture and 2.36 A
for KCl (the Shannon K+ - Na+ difference of 0.36 A added), keeping ion identity as a size
effect on a conventional baseline, flagged approximate. The gate is re-run on exactly this
construction (`compctrl_v5`, `jitter_v5_production`; the 1.16 A run cancelled), and a
24-sequence fixed-protocol convergence check at 2.0 A (50 and 150 mM) tests whether the
running study's flank conclusions, made at 1.16 A, carry over -- they should, since the
radius moves the near field. Size-modified PB (APBS 3.4.1 `smpbe`; Chu et al., Biophys J
2007, for DNA ion atmospheres) is recorded as the candidate fix for long-solute divergence
and ion identity, to be evaluated as its own study without blocking production.

**Training on long windows** (`data.LongWindowSet`, `train.py --long-shards --long-per-step
--long-weight`). Family A's labels will live on 121 bp windows (fragment plus 30 bp of
flank either side), every condition solved on the window's structure trimmed to that
condition's production length, so the encoder always sees at least the sequence a label
depends on. Long shards are aligned row for row with the main shards (each window inherits
its fragment's train / validation / test assignment, so the similarity split holds), store
per-bp values for the central 21 positions only (about 3.8 GB for the corpus instead of
21.7 GB at full width), and are checked window-centre against fragment. Their channels are
appended to the model's plan after the main channels with their own standardisation; main
batches carry NaN for them (built per batch, not stored), long windows get their own
forward pass each step scored on the central positions, and evaluation slices predictions
to each block's columns and merges the two into the plan's per-channel arrays. Tested: a
misaligned window is refused, and a full training run on synthetic data scores the
long-window family and the main families together (`tests/test_long_windows.py`; 70 tests
pass).

**Ensemble spread, capability in place** (`run_fragment.py --snapshot k`,
`ensemble_excess`). PLAN.md: "For about 10 percent of fragments, draw 8 configurations from
the cgNA+ Gaussian and solve each, supervising both the ensemble mean and the spread."
Snapshot k is delta ~ N(0, K^-1) from the fragment's own cgNA+ stiffness, composed as
gs o exp(delta), seeded by the fragment id and index, then the same converged restrained
minimisation; outputs are named `<frag_id>__s<k>`. On a 61 bp fragment two snapshots sit
7.4 and 7.8 A RMSD from the ground state after superposition, the size thermal bending of
a ~200 A duplex with a ~50 nm persistence length produces. The production plan solves the
ensemble subset's eight snapshots at the anchor conditions; the assembler reduces them to
ensemble mean and spread channels.

**Family A production design, proposed (fixed when the convergence groups and the gate
land).** Every label is solved on the fragment's 121 bp window (genomic flanks for panel
fragments), each condition on the window's structure trimmed to its production length, so
the conditions share one conformation per fragment. Costed per fragment from measured
times (build and converged minimisation of 121 bp ~30 s; 61 bp solves ~30-60 s; 121 bp at
low salt ~300-450 s; one reference solve per distinct grid), a full-corpus anchor at 25 mM
alone would cost ~24,000 CPU-hours and the whole design ~41,000, above the plan's 15,000-
30,000. Hence:

- full corpus (191,344 fragments) at **50 mM and 500 mM** -- bracketing the range the
  nonlinear solver handles on full-length solutes -- and at **150 mM**, the physiological
  reference;
- **15 percent subsets** (about 28,700 fragments each, deterministic by fragment id) at 10,
  25, 100 and 250 mM, KCl and the Na/Mg mixture, which carry the condition residual;
- the **ensemble subset**, 10 percent of fragments x 8 cgNA+ snapshots at 150 mM.

Estimated ~23,000 CPU-hours. No condition is dropped; the 10 and 25 mM labels are learned
from 28,700 fragments each rather than from the full corpus, and the plan's anchors become
50 and 500 mM with 10 and 25 mM in the subsets, a deviation recorded here.

**The convergence study's conclusions carry over to the production radius.** On the same
24 sequences (seeds 1000-1001), the 41 against 61 bp fragment-specific scatter at 2.0 A
matches 1.16 A in every shell and is slightly smaller in the outer ones: outermost surface
shell 0.118 -> 0.102 at 150 mM and 0.208 -> 0.184 at 50 mM; axis 18 A 0.055 -> 0.046 and
0.149 -> 0.125. The radius moves the near field and leaves the flank dependence alone, so
the plan-scale study (run at 1.16 A) sets production lengths for the 2.0 A construction.

**Family M: a parallel G-quadruplex loop library from the PDB** (`results/g4_structure_classification.json`).
RCSB search for nucleic-acid-only entries mentioning G-quadruplexes: 414 hits, 406 in PDB
format, all classified by our own detector (tetrads from Hoogsteen contacts; columns from
sequence adjacency across consecutive tetrads; parallel when all four columns run the same
way along the stack axis; loop lengths from the sequence gaps between columns): 166
unimolecular, **79 unimolecular parallel** (67 with three tetrads, 7 with two, 5 with four).
Propeller loops available by length: 1 nt 99, 2 nt 57, 3 nt 65, 4 nt 4, 5 nt 9, 6 nt 1
(none of 7; 8 and 9 once each). Known structures come out right (1KF1 [3,3,3], 1XAV
[1,2,1], 2LEE [1,1,1]). This is the loop source for threading: a three-tetrad parallel core,
each loop taken from a solved loop of the same length, superposed on its anchoring
guanines, bases substituted, then the converged restrained minimisation with the K+ ions,
and APBS. Loops of 7+ nt have too few examples and are flagged rather than modelled.

**Size-modified PB is not available in the installed solver.** APBS 3.4.1 parses `smpbe`
but refuses it at setup ("is currently disabled. We apologize for the inconvenience";
probe 34115080), on 61 and 121 bp alike. The long-solute divergence at 50 mM and above
therefore stays a recorded limitation of the solver, and the production design already
works around it: 61 bp solutes at those salts, outer shells flagged where their context
scatter stays large. Same probe, nonlinear PB at 2.0 A on 61 bp: minor groove 1 A out
-4.48 / -4.45 kT/e, major -3.38 / -3.37, 11 A out -1.14 / -1.13 -- consistent with the
radius probe.

**Family M threading, first validation** (`src/physics/fold/g4_build.py`; 65 parallel
three-tetrad structures with loops <= 6 nt, 192 of 195 loops complete). Each solved
structure rebuilt from the 1KF1 core (1XAV for 1KF1 itself) with every loop taken, leave-
one-out, from the best-fitting solved loop of the same length: core backbone RMSD to the
solved core median 1.22 A (max 1.90) -- parallel cores are conserved, so one core template
serves; best-loop anchor superposition median 0.75 A; threaded loop backbone RMSD to the
solved loop median 2.45 A (75th percentile 4.24, 90th 5.96), comparable to the spread of
flexible loops across NMR models. Optimistic in one respect, to be fixed before use:
near-duplicate entries of the same sequence were not excluded as donors (2KZE drew an
anchor RMSD of 0.00); the next validation excludes same-sequence donors and reports by
loop length. Next: base substitution, minimisation with the channel K+ ions, APBS.

**Correction: threaded loops are poorly determined beyond 1 nt.** Re-validated with every
donor of the target's own sequence excluded (near-duplicate entries had made the first
figure optimistic; the 2.45 A median above is retracted): threaded loop backbone RMSD to
the solved loop, median (75th percentile) by length -- 1 nt 2.02 A (2.67), n = 91; 2 nt
3.69 (4.28), n = 30; 3 nt 5.73 (6.14), n = 61; 5 nt 6.00 (6.62), n = 8; 4 and 6 nt have no
other donor. How well a loop's anchors fit the core does not predict the loop's own
conformation, consistent with propeller loops being flexible. Design consequence: each
candidate fold is modelled as an ensemble -- several donor loops per position (the best
anchor fits from distinct structures) -- and its channels are the ensemble mean with the
spread reported beside it, which is also how the plan's "MD ensembles" ground truth reads.
Core-dominated quantities (the central-channel potential, tetrad geometry) remain well
determined, the core being conserved to a median 1.2 A.

**Auxiliary windows get Families J, N and K (2026-11-21).** v5 at 10M, epoch 21
(validation): per-bp family medians J 0.896, L 0.801, N 0.782 against >= 0.988 for every
other family; whole-sequence J 0.862, L 0.887, N 0.876. All three are fold-derived like
Family L, which the auxiliary windows showed to be data-limited (+0.08 per-bp at 10M and
50M), yet the auxiliary windows carry no J, N or K labels. They are cheap (measured per
61 bp window: J 0.09 s, N 0.99 s, K 1.03 s), so all 1,017,719 auxiliary windows get them
as side shards (`aux_all_famJNK/`, job 34115850, ~600 CPU-hours): J and N per-bp and
whole-sequence, K whole-sequence only (its 312 per-bp profiles would take ~40 GB of GPU
memory at this size, and K per-bp already scores 0.997). Loader: `AuxLandscape(...,
extra_dirs=)` and `--aux-extra-dirs`, sequence-verified like the main side shards
(`tests/test_extra_dirs.py`, two new tests). The ensemble-spread channels of Family C
that score lowest (C3'-endo 0.01-0.02, torsion spreads) were already shown to sit at
their measured sampling ceilings (RESULTS.md 18b), so more data cannot help them.

## Gate on the final construction: PASS (2026-11-21, 07:00)

Pre-registered (thresholds unchanged): `compctrl_v5` (644 composition-control fragments, 50
groups, 150 mM) and `jitter_v5_production` (16 replicates) on exactly the production
construction -- converged minimiser, exclusion radius 2.0 A (KCl 2.36 A), exact 0.35 A fine
spacing, extended extraction, per-atom solvation, reference solves.
`results/electrostatics/composition_gate_final_v5.json`: **PASS**. Channels with within-
group SD above 5x the replicate noise: 89.4 percent (threshold 50); within-group variance
above the 0.05 floor in 100 percent; median eta 0.687; median SNR 22.6; composition explains
little (median mononucleotide R-squared 0.26, maximum 0.83), so the potential has not
collapsed into composition. Per channel kind (`results/electrostatics/channel_snr_final_v5.json`):
all 60 kinds clear within-group SNR 5; the lowest are the minor-groove radial-field
gradient 9.7 and **per-base-pair solvation 10.8**; whole-fragment solvation 40.4; surface
potentials median 19.0 over 36 shells/sectors (minimum 6.7).

**Production launched on this construction.** End-to-end smoke test on real data (job
34115192): 187 windows (0 centre mismatches), 4 fragments solved at three conditions in
204-290 s each, one cgNA+ snapshot (418 s), long shards written and loaded by
`LongWindowSet` (479 per-bp channels on the central 21 bp + 9 whole-window), minor-groove
potential 1 A out at 150 mM -4.8 to -6.1 kT/e. Solute length per condition is the longest
the solver handles (121 bp at 10 and 25 mM, 61 bp at 50-150 mM, KCl and the Mg mixture, 41
bp at 250 and 500 mM): at 50 mM and above the overflow limit binds before any convergence
criterion could choose a longer solute, and at 10-25 mM the longest is the conservative
choice. The convergence study, still running, therefore documents the residual flank error
of each condition (outer shells flagged where it stays large) rather than choosing lengths.
Design as proposed above: full corpus at 50, 150 and 500 mM; 15 percent hash subsets at 10,
25, 100, 250 mM, KCl and Na/Mg; ensemble 10 percent x 8 cgNA+ snapshots at 150 mM.

**QC tiers 2-3 launched (2026-11-21).** Timing probe (job 34114976, AAAA context, 8 threads):
B3LYP-D3BJ counterpoise stacking 116 s (def2-SVP) / 514 s (def2-TZVP); the full DFT set
(stacking, two base-pair H-bond energies, dimer HOMO/LUMO and dipole; vertical IP/EA by
delta-SCF UKS at SVP) 348 s / 718 s; SAPT0/jun-cc-pVDZ 558 s. Values: AA/TT stacking -13.4
(SVP) / -13.9 (TZVP) kcal/mol, A.T H-bond -17.5 / -16.9, SAPT0 total -12.6 (electrostatics
-3.5, exchange +12.2, induction -1.5, dispersion -19.8), vertical IP 7.18 eV. The basis
moves stacking by 0.5 kcal/mol, so production uses def2-TZVP: all 136 RC-unique tetramer
contexts with IP/EA and SAPT0 on the cgNA+ ground state (job 34117101, 34 tasks), and stacking and
H-bonds on the +-1 sigma soft-mode geometries (jobs 34117102, 34117103; 17 tasks each), ~700 CPU-hours.
These become Family B quantum-chemistry channels (per base-pair step, mapped from the
tetramer table) once complete; Stage B (transfer integrals, MEPs, polarizability, charges)
follows on the same geometries.

## Family M, quadruplexes: from structures to channels (2026-11-21)

**Model.** `src/physics/structure_folds.py` (Family M, held apart like Family I) with
`fold/g4_select.py` (which quadruplex: Family F's best candidate; four tracts of >= 3 G, the
four consecutive tracts with the shortest total loop length, spare guanines to the loops,
loops 1-6 nt; a C-rich candidate built on the reverse complement), `fold/g4_build.py`
(assembly from the 1KF1 core, donor loops superposed on their anchoring guanines, bases
substituted in the donor's own base frame, AMBER OL15 + GBn2 minimisation with the core
restrained at 2000 kJ/mol/nm^2 and loops free; 41-105 s per model) and
`fold/g4_electrostatics.py` (nonlinear PB at the production radii, 150 mM KCl and NaCl).
Channels (5 per-bp, 46 whole-sequence): the central channel potential at the inner ion sites,
tetrad planes, outer sites (5' and 3' faces), its minimum and 6 A beyond either face; groove
and loop potentials 1 and 3 A outside the molecular surface (surface-referenced rays, as for
Family A, after fixed-radius points were found to land inside the solute: one groove sample
read -68 kT/e); per-nucleotide surface potentials (per-bp); electrostatic solvation (per-atom);
the electrostatic binding free energy of the two inner-site cations by the PB thermodynamic
cycle (complex - quadruplex - ions, plus their Coulomb term in the solute dielectric; Rashin-
Honig Born radii K+ 2.172, Na+ 1.680 A), its desolvation part, K+/Na+ selectivity; dipole,
radius of gyration, groove depth, model spread, energy per nucleotide; ensemble spreads for
the site potential, loop potential, ion binding and solvation. Ensemble: 3 models, model k
taking each loop's k-th best donor. First full model (c-MYC Pu22, loops T/TA/T): sites
-112/-104 kT/e, groove 1 A out -2.5, loops -1.2, ion binding K+ -284 and Na+ -177 kJ/mol
(electrostatics favour K+, the observed selectivity), solvation -7,090 kJ/mol; 170 s for both
conditions. `tests/test_structure_folds.py` (4 tests, one full model) passes.

**Coverage and data.** 799 of 191,344 corpus fragments carry a modelled quadruplex (job
34117983 models them); too few to learn from, and Family F's quadruplex descriptors are rare
events in the corpus for the same reason (F_g4_best_tetrads 0.125 at v5 10M, epoch 21). So
quadruplex-centred genomic windows are drawn from the 15 panel genomes
(`src/data/sample_fold_windows.py`; linear-time tract clustering; leakage-filtered against
the library by the main split's MinHash criterion), up to 1,200 modelled and 1,500 other
G4-motif windows per genome, labelled with the cheap families like the auxiliary windows
and with Family M on the modelled ones (`src/physics/run_family_m.py`, resumable;
`src/data/pack_family_m.py` builds aligned side shards and refuses a modelled fragment with
no solve unless told otherwise).

**Loader.** `AuxLandscape` now takes several auxiliary sets (comma-separated) and side
directories that may lack some shards (`--aux-extra-optional`), uniting channel layouts and
leaving NaN where a set lacks a family (test added). **Test isolation:** `src/data` (a
namespace package) and `src/model/data.py` share the name `data`; in a single pytest process
whichever was imported first won, so the full suite had never collected cleanly. The three
data-pipeline tests now import with the model path hidden and purge the cache afterwards;
the full suite collects, 267 pass in the training environment (the rest need ViennaRNA or
OpenMM) and the 136 physics tests pass in the physics environment.

**Family M, i-motifs: the PDB does not support threading** (`fold/imotif_templates.py`,
`results/imotif_structure_classification.json`). RCSB "i-motif" DNA entries without
protein: 62 hits, 61 downloaded; 29 carry two or more hemiprotonated C.C+ pairs (N3-N3 and
both N4-O2 contacts) and only 4 are unimolecular with four C-tracts (1A83, 1G22, 1EL2/1ELN
-- the last two one sequence), with irregular tract patterns. A donor-loop library like the
quadruplex one (79 parallel structures) cannot be built from that. The i-motif needs a
geometric core instead -- an intercalated C.C+ stack built to the regular geometry of the
tetramolecular structures (rise ~3.1-3.2 A measured here) -- with loops built and minimised
rather than taken from donors; that, and the triplex and cruciform builders, are the next
Family M stage. Until then Family M carries quadruplexes only, and the i-motif, triplex and
cruciform propensities remain Family F channels.

**Family A ensemble channels (2026-11-21).** The ensemble subset (10 percent of fragments x 8
cgNA+ snapshots at 150 mM, production jobs 34118558-65) had no path into training. Now
`electrostatics_family.assemble_ensemble` gives, for a core subset at 150 mM -- the six
sector potentials 1 A out, radial field and probe energy in both grooves, condensed charge
on both phosphates, per-base-pair solvation, groove asymmetry (14 per-bp) and whole-window
solvation -- the mean and spread across snapshots (28 per-bp + 2 whole-window channels; NaN
with fewer than 4 snapshots or for fragments outside the subset). Each snapshot is assembled
exactly like a ground-state solve, so own-frame storage carries over and the strand tie
holds for mean and spread (tested on synthetic reverse-complement solves). The long-shard
generator takes `--ensemble <dir>` and checks each snapshot's window against the fragment's.

**Production restart.** The first preparation job (34116441) died at start-up (exit 127):
its script enabled `set -e -u` before sourcing the environment, the fault already fixed in
the smoke test; Slurm then cancelled the nine dependent arrays. Fixed and resubmitted:
preparation 34118555, main array 34118557 (2,392 tasks), ensemble 34118558-65.

## Pre-registration: the final ladder (v6), written before any v6 run (2026-11-21)

**Configuration** (`scripts/submit_final_ladder.sh`): v5 plus Family A on 121 bp long windows
(central 21 bp labels, ensemble mean/spread on the 10 percent subset), Family M (corpus side
shards and quadruplex windows), Family J revision 2, Families J/N/K on the auxiliary
windows, and the quadruplex and rare-event windows as further auxiliary sets. Same rungs,
widths, schedules and split as v5 (similarity split, seed 0, 5 percent validation, 5 percent
test); Family I held apart. Launched when the Family A long shards, the Family M side
shards and the window-set labels exist; the merged cache is built first (FIRST_DEP).

**Criteria, test side, each rung against the same v5 rung:**
1. *Fold families are data-limited:* per-bp family medians of J and N each rise by >= 0.02,
   whole-sequence medians of J, N and L by >= 0.005 (v5 10M validation at epoch 44 is the
   reference point; seed spread at 10M measured earlier is ~0.003).
2. *Rare events:* for the rare-event channels (modal value >= 99 percent, the trainer's own
   rule), the event-detection AUROC rises on at least 4 of the 5 measured here
   (L_p/L_m_gquad_gain, F_g4_best_tetrads, F_triplet_repeat_instability, F_rloop_score).
3. *Non-inferiority:* no other family's median falls by more than 0.001 per-bp or 0.002
   whole-sequence; per-channel change counts reported beside the medians.
4. *Family A against its own noise ceiling:* ceiling R2 = 1 - 1/SNR_total^2 from the final
   gate (`channel_snr_final_v5.json`): 0.978 for the weakest surface-potential shell, 0.990
   for per-base-pair solvation, 0.995-0.9995 for the rest. Reported: test R2 per channel
   kind and its ratio to the ceiling, by condition; the conditions solved on subsets (10, 25,
   100, 250 mM, KCl, Na/Mg) separately from the anchors (50, 150, 500 mM).
5. *Family M:* test R2 on the corpus fragments with a modelled quadruplex and on held-out
   quadruplex windows, beside the ensemble spread as a noise scale (channels whose spread
   is comparable to their across-sequence SD are reported as uncertain, not as failures).
Any criterion that fails is reported as failing; none is re-worded after the runs.

**QC tiers resubmitted to the plan's full decomposition (before any task started).** The plan
asks for the stacking energy "decomposed into electrostatic, exchange-repulsion, induction
and dispersion at 3 geometries" and the "hydrogen-bond energy decomposed the same way"; the
first submission ran SAPT0 for ground-state stacking only. Now (`run_psi4.py --sapt-hbond`,
SAPT0/jun-cc-pVDZ of each base pair's Watson-Crick interaction): ground state with DFT,
IP/EA, stacking SAPT0 and H-bond SAPT0 (job 34118606); both soft-mode geometries with DFT
stacking and H-bonds, stacking SAPT0 and H-bond SAPT0 (jobs 34118607, 34118608); 34 tasks each.

**Family B quantum chemistry enters as family "Bq"** (`src/physics/qc_channels.py`,
`src/qc/build_qc_table.py`, `tests/test_qc_channels.py`). Every base-pair step takes the
DFT/SAPT values of its own four-base-pair context (through the stored reverse complement
when needed, with the two base pairs exchanged), so the channels are an exact function of
sequence: per step, counterpoise stacking at the three geometries and its SAPT0 components
(18), vertical IP/EA (2), frontier orbitals, gap and dipole at three geometries (12); per base
pair, the H-bond energy and its SAPT0 components at three geometries (18; each pair averages
its two contexts, and the two pairs of a self-complementary context their mean, as symmetry
requires -- which keeps the reverse-complement identity exact, tested); 11 fragment summaries
(including the stacking softness along the soft mode). Step channels carry rc_shift -1. Own
family code so side shards can add them to corpora whose Family B already exists; the table
is built when the three QC arrays finish, after which side shards for the main corpus and the
auxiliary sets cost seconds.

**Production split by memory (before any task ran).** Peak memory of the smoke test (121 bp
build and minimisation, 61 and 41 bp solves) was 4.1 GB; APBS memory follows the grid
dimension, which is set by the solute length, so every 61/41 bp condition fits in ~5 GB and
only the 121 bp solves at 10 and 25 mM need the measured 12.2 GB. Requesting 16 GB for every
task would have let memory, not CPUs, cap concurrency (the day QoS allows 1,500 CPUs and
20 TB per user). Now: `manifest_big.tsv` (53,093 fragments in the 10 or 25 mM subsets; 16 GB,
40 per task, job 34118988, 1,328 tasks) and `manifest_small.tsv` (138,251; 7 GB, 100 per
task, job 34118989, 1,383 tasks); ensemble arrays at 7 GB (34118990-97). Same code, outputs
and manifest rows as before. Preparation output: 191,344 windows (119,934 genomic flanks, 15
with a random edge, 71,395 random for MPRA and synthetic strata; 0 centre mismatches);
746,070 solves (191,344 at each anchor; 28,433-28,882 per subset condition); 19,178 ensemble
fragments.

## The positive control and the primary gate, set up (2026-11-21)

**Positive control ingested** (`ingest_assays.py --only nucleosome_invitro`;
`processed/nucleosome_invitro.tsv`): Kaplan et al. 2009 in vitro map (GEO GSM351491, the
normalised track already downloaded and calibrated: offsets 0-10 bp, used directly), sacCer3
tiled with 147 bp windows every 75 bp on the 16 nuclear chromosomes, target = mean occupancy
over the central 21 bp; 143,462 windows (17,463 skipped for missing coverage or non-ACGT).
Test side = chromosomes XV and XVI (24,688 windows), whole chromosomes because neighbouring
windows overlap by 72 bp; validation drawn by similarity from the rest, as for every provided
split. GC of the window alone: r = 0.738 with the target. Arms queued on the v3_L10M_tie
encoder already used for the other READOUT sets (featurisation 34119581; seq, phys, seq+phys
34119582-84), to be rerun on the final encoder as the plan says.

**The primary gate, GC + 5-mer ridge** (`src/readout/kmer_ridge.py`): GC plus 512 canonical
5-mer frequencies, ridge with the penalty chosen on validation, on the READOUT trainer's own
split (same function, same seed) so every arm sees the same test rows; running on the
positive control, loop-seq, Jores and splice sites (job 34119439).

**Correction to a provenance record.** Running the ingester with `--only` overwrote
`processed/ingest_report.json`, erasing the 2026-11-19 entries for loop-seq, Array Melt and
ORChID. The ingester now merges into an existing report; the three entries were regenerated
from the same raw files, and the regenerated tables are byte-identical to the 2026-11-19
ones (md5), which the restored report notes.

**Family M builder, full validation and two production fixes (2026-11-21).**
`results/g4_build_validation.json` (job 34116210): all 35 distinct parallel three-tetrad
sequences rebuilt leave-one-out from the 1KF1 core with donors of other sequences, through the
production path (assembly, base substitution, AMBER OL15/GBn2 minimisation): 32 built, 2 with
no donor of a needed loop length, 1 failed on a donor nucleotide with only two base atoms.
Core RMSD median 1.19 A before and 1.12 A after minimisation; loop backbone RMSD, median before
-> after: 1 nt 2.15 -> 1.98 A (n = 51), 2 nt 5.13 -> 4.38 (16), 3 nt 5.79 -> 5.72 (22), 5 nt
5.70 -> 4.52 (7); minimisation median 29 s, final maximum force median 11.6 kJ/mol/nm.
Fixes before production: donor loops now need fit-able bases (>= 5 of the six ring atoms;
190 of 195 loops remain usable); and the first production array found no templates, because
frozen code snapshots copy `src/` only -- every fragment failed before writing anything. The
template path now falls back to the project's data folder (and the job script sets it).
Resubmitted: corpus 34119807, quadruplex windows 34119808.

## Pre-registration: rare-event channels under beta = 1 (2026-11-21), before the run

**Observation (v5_L2M, test).** The rare-event channels are detected but not sized: AUROC
0.983-0.999 with R2 -0.006 to 0.002 (F_g4_best_tetrads, F_triplet_repeat_instability,
L_m/L_p_gquad_gain). Under beta-NLL with beta = 0.5 the mean gradient scales as 1/sigma, so a
large residual on a rare input can be absorbed by a larger predicted variance instead of a
larger mean -- the failure Seitzer et al. describe, only partly removed at beta = 0.5.
**Change.** `--rare-beta 1.0`: beta = 1 (mean gradient independent of the predicted spread)
for the channels the trainer's own rule flags rare on the training side (modal value >= 99
percent); every other channel keeps 0.5 (`LandscapeLoss(beta_per_bp=, beta_global=)`, tested).
**Run.** `v5r_L2M`: exactly the v5_L2M configuration plus `--rare-beta 1.0`.
**Criterion (test side, against v5_L2M):** the median R2 of the rare-event channels rises to
>= 0.3 with AUROC not lower than 0.98 on any of them; no family median (per-bp or
whole-sequence) falls by more than 0.002. If it passes, `--rare-beta 1.0` joins the final
ladder; if not, it is reported and left out.

**QC Stage B joins the ground-state tier (before it started).** `run_psi4.py --stage-b` on the
stacked step: in-stack hole/electron site energies of the four bases (frontier orbitals of the
isolated capped bases projected on the dimer Fock matrix), effective transfer integrals
(Valeev et al. 2006, overlap-corrected) for holes and electrons -- intrastrand on both strands,
both interstrand diagonals, within each pair --, MBIS charges (net per base and at the groove
H-bonding atoms), the MEP 1.5 A outside each groove H-bonding atom (purine N7, O6/N6, N3, G N2;
pyrimidine O4/N4, O2) along its local donor/acceptor direction, and the static dipole
polarizability of the step (isotropic, anisotropy, along the stacking axis). Atom names of the
capped bases recovered from the builder's templates and verified on all 544 base files
(`src/qc/data/base_atom_order.json`). Probe at def2-SVP (AAAA, 604 s on 4 threads): site
energies A -5.83 / T -6.42 eV (hole), A -0.26 / T -1.05 eV (electron); hole transfer AA 0.054 eV
and TT 0.074 eV intrastrand, 0.012-0.024 eV interstrand; MBIS charge transfer +-0.04-0.05 e
within each pair; polarizability 292 a.u. isotropic, 169 along the stack. First MEP probe
direction (from the pair centre) put the N7 point beside H8 (+40 kcal/mol/e at an acceptor);
replaced by local directions before production. Ground-state array resubmitted with Stage B:
job 34121015 (68 tasks); pKa shifts remain open.

**Bq channels with Stage B: 80 per-bp + 15 whole-sequence.** Per step (rc_shift -1): stacking
and its SAPT0 components at three geometries, IP/EA, frontier orbitals/gap/dipole at three
geometries, hole and electron transfer integrals as max and min over the two strands
(intrastrand) and over the two diagonals (interstrand), polarizability (isotropic, anisotropy,
axial). Per base pair: H-bond energy and its SAPT0 components at three geometries, purine and
pyrimidine hole/electron site energies, in-pair transfer integrals, purine net MBIS charge
(within-pair charge transfer), MEP and MBIS charge at six groove H-bonding sites. Strand-
specific quantities enter as purine/pyrimidine or strand max/min so every channel is exactly
reverse-complement symmetric (tested); which strand carries the larger integral is not
represented, a recorded limitation.

## Production fault: the energy evaluation overflows at 25 mM (121 bp) and with Mg2+ (2026-11-21)

**Observed** in the first production tasks: 45 fragments failed at NaCl_025mM (121 bp solute)
and 5 at the Na/Mg mixture (61 bp), all with "Detected large potential values in energy
evaluation" after "trapped exp overflows" in the solve -- APBS's mobile-ion energy term
overflowing once the potential has been computed. The smoke test, the gate and the jitter
runs never reached these conditions (150/500 mM and KCl only), and the convergence study did
not request energies, so this is the first time the energy was evaluated there. Because the
fragment runner treated any APBS failure as fatal, each such failure also discarded the
fragment's 50, 150 and 500 mM anchors (no output written; nothing wrong recorded).
**Fix** (`run_fragment.py`, `apbs_run.write_apbs_input(calc_energy=)`): on an energy-evaluation
failure the condition is solved again without the energy -- potentials, fields, ion
occupancies and condensed charge kept, that condition's per-base-pair solvation NaN and the
condition listed in the fragment's `energy_skipped_conditions`; any other failure of one
condition leaves that condition NaN (`failed_conditions`) and keeps the rest; a fragment fails
only when every condition does. Production arrays held (34118988 and the queue of 34118989)
until the fix is verified on two of the failing fragments (job 34123147); finished fragments
are kept by `--resume`.

**Fix verified and production restarted.** Test (job 34124173) on two fragments that had
failed: both solved at every condition; energy skipped only at NaCl_025mM (121 bp) and at the
Na/Mg mixture respectively, with per-base-pair solvation present at every other condition. The
potentials are physical: minor groove 1 A out -6.83 (25 mM), -6.16 (50), -5.50 (100), -5.12
(150), -4.05 kT/e (500 mM), monotonic in salt; 2 mM Mg2+ on top of 150 mM Na+ screens it from
-4.96 to -4.38. Old arrays cancelled; resubmitted on frozen code famAprod2 (big 34124814, small 34124816),
resuming past the fragments already solved.

**Auxiliary J/N/K labels complete** (`aux_all_famJNK/`, job 34115850): 512 side shards,
1,017,719 windows, 0 failed windows, one channel layout (22 per-bp: J 10 + N 12; 184 whole-
sequence: J 56 + N 30 + K 98), sequence alignment with the base shards verified on a sample
of 8 shards, NaN fraction <= 0.02 percent.

**Final-ladder memory: Bq on the main corpus only.** Estimated GPU-resident targets for v6:
main corpus ~16.6 GB (about 714 per-bp channels x 61 bp x 191k fragments, fp16), Family A long
windows ~4.1 GB, auxiliary sets ~4 GB without Bq; adding Bq's 80 per-bp channels to the 1.08M
auxiliary windows would add ~10.5 GB. Bq is a lookup over 136 four-base-pair contexts, each
present millions of times in the main corpus, so the auxiliary copies carry no new
information; they are dropped (`submit_final_ladder.sh`, `finalize_v6_inputs.sh`).

**Scheduling: time limits cut to measured runtimes (pending jobs updated in place).** With
~3,600 CPUs idle on scavenge and ~660 on day, my arrays still waited at "Priority": 12 h
requests do not fit backfill windows. Updated: Family A production 8 h (tasks need 5.6-6.7 h),
ensemble 4 h (~2.5 h), QC arrays 5 h, Family M 4-6 h, window-set labels 3 h. A task that
overruns is retried by resuming the array; no result depends on the limit.

## Outcome: rare-event beta = 1 at 2M (`v5r_L2M` against `v5_L2M`, test side) -- FAILS as pre-registered

Flagged rare by the trainer's rule: whole-sequence F_g4_best_tetrads, F_triplet_repeat_instability,
L_m/L_p_gquad_gain; per-bp C_sugar_C3endo_prob_W/C (at their binomial sampling ceiling, 0.01-0.02).
- Rare-event R2: tetrads -0.006 -> 0.552, triplet-repeat instability 0.000 -> 0.806,
  L_m gquad gain 0.001 -> 0.734, L_p 0.002 -> 0.610; C3'-endo 0.014/0.021 -> 0.016/0.023. Median
  over the six 0.581 (criterion >= 0.3: met).
- AUROC: tetrads 0.999 -> 0.998, triplet 0.983 -> 0.980, gquad gain 0.989/0.987 -> 0.933/0.940,
  C3'-endo 0.837 -> 0.827/0.837 (criterion: none below 0.98 -- **not met**: the gquad channels
  fall to 0.93-0.94; the C3'-endo channels were already below 0.98, which the criterion did not
  anticipate).
- Non-inferiority (no family median down by more than 0.002): **not met** -- whole-sequence N
  0.8511 -> 0.8254 (-0.026), J 0.8605 -> 0.8577 (-0.003); every other family within +-0.0015;
  all-channel medians 0.9942 -> 0.9942 per-bp, 0.9902 -> 0.9905 whole-sequence.
**Verdict: fails; `--rare-beta` stays out of the final ladder, as fixed in advance.** The rare
channels are sized far better (R2 ~0 -> 0.55-0.81), so the idea is not dropped: whether the N and
J losses are an effect or 2M seed variation is unknown (seed spread at 2M was never measured).
Follow-up, run now and not used to revisit this verdict: `v5_L2M_seed1` and `v5r_L2M_seed1`
(same configurations, `--seed 1`), which give the 2M seed spread and a second paired difference.
The final ladder carries the quadruplex and rare-event windows instead, which raise the event
counts without changing the loss.
Follow-up the same hour: the 4-5 h arrays started at once (77 ensemble and 53 QC tasks running,
from 5 and 3), the 8 h production tasks did not; production resubmitted as half-size tasks (20 /
50 fragments, 4.5 h limit; big 34129610, 2,655 tasks; small 34129611, 2,766 tasks), resuming past
solved fragments.

**Second production fault: fused PQR fields on bent structures (2026-11-21).** An ensemble task
lost 8 of 60 snapshots to "Valist_readPQR: Error parsing atom": PDB2PQR writes coordinates in
fixed 8.3f columns, so a coordinate <= -100.000 fills its column and fuses with the previous
field, and APBS reads PQR by whitespace (reproduced: 730 of 3,871 lines of a snapshot PQR had 10
fields instead of 11). Ground-state structures are built along +z near the origin and rarely
reach -100 A, which is why the gate, smoke test and first production fragments never hit it;
bent cgNA+ snapshots do, and so could strongly curved ground states. Fix (`sanitise_pqr`): every
ATOM/HETATM line is re-read by column and rewritten with explicit separators (verified: all
lines 11 fields, coordinates unchanged, total charge -120 e). Frozen as famAprod3; main arrays
resubmitted (big 34131983, small 34131984; nothing had started), ensemble arrays resubmitted to
run after the old tasks still running (34131985-92), resuming past solved fragments.

**Rare-event windows sampled** (`rare_windows/rare_windows.tsv`, jobs 34118194/95): 15 genomes x
400,000 random windows scored by Family F, kept when an event is present and capped at 1,500 per
event type and genome: 79,042 candidates, **75,858 kept** after the library leakage filter
(3,184 dropped). Events (a window can carry several): Z-DNA tract >= 18 bp 22,183; triplet-repeat
instability 17,337; triplex candidate 17,330; R-loop zone 16,913; >= 3-tetrad quadruplex 6,141.
Labels (auxiliary families + J/N/K) running (34118202, 64 tasks).

**Ground-state QC tier, first 24 contexts (12 shards): all succeeded.** Per context: DFT set with
IP/EA 1,465 s median, SAPT0 stacking 578 s, SAPT0 H-bonds 99 s, Stage B 1,073 s (783 s of it
polarizability). Stacking -9.7 to -16.9 kcal/mol (median -12.8); SAPT0 total -7.6 to -16.7;
vertical IP 6.3-7.3 eV; intrastrand hole transfer 0.004-0.112 eV; polarizability ~329 a.u.
(def2-TZVP). The corrected groove MEP is negative at acceptors (purine N7 -45 to -78, pyrimidine
O2 -25 to -56 kcal/mol/e) and changes sign at the purine 6-position between G O6 (acceptor) and
A N6 (donor), as it should.

## Family A brought back to the plan's anchoring, and solvation at every condition (2026-11-21)

**The user: "we need to follow the plan, stop cutting corners".** Two Family A shortcuts are
reversed:
1. **Anchors.** PLAN.md: "solve the full set at two anchor conditions chosen to bracket the
   range, then solve a subset of 10 to 20 percent at the remaining conditions". The design in
   production had moved 10 mM into a 15 percent subset to save ~20,000 CPU-hours; that saving is
   withdrawn. Manifest v2: every fragment at **10 mM and 500 mM** (the bracketing anchors) and at
   50 and 150 mM as well; 25, 100, 250 mM, KCl and Na/Mg on the same hash subsets as before
   (15 percent each); 908,648 solves. Main arrays 34152648 + 34152649 (12,757 tasks of 15
   fragments; the scheduler caps arrays at 10,000 tasks, hence `TASK_OFFSET`). The 5,614
   fragments solved under the old design get a supplementary pass (34152650: 10 mM where
   missing, plus every condition whose solvation was empty -- 4,576 fragments, 4,294 lacking 10
   mM, 1,758 re-solves); `generate_family_a_long_shards.load_merged` joins the passes by
   condition, fills a condition's empty arrays from the later pass, and refuses files whose
   structures differ by more than 0.05 A rms (tested).
2. **Solvation where APBS's energy evaluation fails.** Instead of leaving it empty, per-base-
   pair solvation is computed from the potential maps: 1/2 q_i (phi_solv - phi_ref) at each scored
   atom, the reference potential written on the same grid (`write_apbs_input(reference_maps=)`).
   Validated on two fragments at every condition where APBS's own per-atom form exists (10, 50,
   100, 150, 500 mM): r = 1.00000, mean difference +0.01 kJ/mol (0.00 percent); it supplies 25
   mM and Na/Mg where APBS fails. Stored as `ext_bp_solvation_map_kJ_mol` beside APBS's form on
   every fragment; the channel uses APBS's form where present and the map form otherwise.
   Note for interpretation: the 10 and 25 mM values (about -3,000 kJ/mol per base pair against
   about -2,500 at 50 mM and above) include the longer 121 bp solute those conditions use.
Frozen code famAprod4.

**Family N: 2'-O-methyl added** (`rna_modification.ome_duplex`; tests pass). PLAN.md lists
2'-O-methyl; stage 1 had substituted inosine for want of parameters. Source: Kierzek, Mathews,
Ciesielska, Turner & Kierzek, NAR 34:3609 (2006), Table 3 -- 16 stacks m(5'-XY)/r(3'-X'Y') plus
initiation 3.32 and terminal AU 0.30 kcal/mol, 0.1 M NaCl. Channels per transcript: the duplex
free energy of the fully 2'-O-methylated transcript with its RNA complement, its difference from
the RNA/RNA duplex at the same 0.1 M (Turner 2004 with ViennaRNA's salt correction), and the
per-nucleotide step difference. Caveat recorded with the channel: the paper compares its 0.1 M
set only with RNA/RNA at 1 M (RNA/RNA 0.26 kcal/mol per stack more favourable, "likely because of
higher salt") and gives no salt correction, so the sign of the matched-salt response rests on
ViennaRNA's correction of the reference (a 14-mer: +0.97 kcal/mol). Inosine is kept (an
addition). Still open in N: the hybrid class (no modified-base DNA/RNA parameters located).

**Window flanks with assembly gaps.** One ensemble snapshot failed on "KeyError: 'GS'": 19 of
191,344 windows (178 N characters and single M, Y, R, S codes) carried non-ACGT bases in their
genomic flanks; `extend_windows.py` had checked the fragment centre only. Each such flank
position now takes the base of the fragment's seeded random flank (`fill_non_acgt`, the policy
already used for flanks the genome cannot supply; flank_source "+random_fill"); the 19 rows were
patched in `windows.tsv` and every manifest. Fragments already failed on them are redone by the
final resume pass.

**2M seed spread (`v5_L2M_seed1` against `v5_L2M`, test side).** All-channel medians 0.9942/0.9941
per-bp, 0.9902/0.9903 whole-sequence; largest family differences per-bp N -0.0077, L -0.0076,
whole-sequence F -0.0029, J -0.0026, N -0.0019. The rare-beta run's whole-sequence N loss
(-0.026) is 13 times the seed difference of that family (0.0019), so it is an effect of the
change, not noise -- consistent with the verdict already recorded (beta = 1 stays out). Without the
change the rare channels are unsized at both seeds (R2 -0.006 to 0.002).

**Incident: a wrong package named `oxpy` was installed and removed (2026-11-21).** Looking for
oxDNA's Python bindings I installed `oxpy` 1.5 from PyPI into a new, separate environment
(`conda_envs/oxdna`). It is unrelated to oxDNA: a third-party client for uploading files to the
public host 0x0.st (it shells out to curl). Its full source was read afterwards from the sdist,
downloaded without installing: `__init__.py` only defines four functions after `import
subprocess`, and `setup.py` only reads the README -- nothing runs at install or import. It was
imported once by a benchmark script and none of its functions were called, so nothing was
uploaded; it is uninstalled. oxDNA's real bindings exist only in a source build, which is what is
used from here (official repository); package names alone are no longer trusted.

## Family J extensions from oxDNA2, and Family K electrostatics on A-form structures (2026-11-21)

**Family J: oxDNA2 single-strand simulations** (`src/physics/run_family_j_oxdna.py`, oxDNA 3.7
built from the official repository, sequence-dependent oxDNA2 parameters; 25 C, 0.15 M). Per
fragment, both strands as free single strands: equilibrium (1e5 relaxation, 8e5 sampled every 1e4)
-> end-to-end distance, radius of gyration, short-range persistence (decay of backbone-bond
tangent correlations over 15 nt), the oxDNA2 stacking energy of every step and its unstacked
probability (stacking energy above -0.1 units; per nucleotide as the mean of its two steps); and
constant end-to-end forces of 5, 10, 20 pN -> extension per nucleotide (force spectroscopy, as the
plan names). Test fragment (2e5 production steps): end-to-end 11.8/11.4 nm, Rg 4.6/4.4 nm,
tangent persistence 4.3/4.0 nm (local, stacking-dominated; it is not the force-spectroscopy
worm-like-chain value of ~1 nm, Bosco et al. 2014, and is labelled as the local quantity),
unstacked fraction 0.32/0.23, extension 0.32/0.37/0.47 nm per nt at 5/10/20 pN; ~10 min per
fragment at production length. The oxDNA classic topology lists strands 3'->5'; sequences are
written reversed and results mapped back. Full corpus, the first 20 percent (deterministic hash,
38,317 fragments; job 34154475, nice 200 so Family A runs first) then the remaining 153,027.
Still to add in J: single-strand electrostatics and hydration (APBS/SASA on the stacked strand),
chemical reactivity (sourced intrinsic reactivities x oxDNA unstacked probability).

**Family K: electrostatics on A-form conformations** (`rna_atomic.py`, `run_k_electrostatics.py`).
RNA nucleotide templates from PDB 466D (1.16 A; interior Watson-Crick nucleotides) in their own
base frames, uracil fitted on thymine's ring; the builder places them on cgNA+'s RNA (cgRNA+ OL3)
and hybrid (cgHYB+) frames with phosphates from the phosphate frames, as for DNA. Checks on a 24
bp test: every RNA residue carries its 2'-OH, the hybrid only on its RNA strand; O3'-P 1.61 A,
C1'-N 1.46-1.48 A. Minimised as Family A's structures (restrained, vacuum, two starts, converged;
AMBER RNA.OL3, plus DNA.OL15 for hybrids) and solved with Family A's production construction at
150 mM: RNA major groove 1 A out -4.04 kT/e against minor -1.71 (the A-form inversion of B-DNA's
pattern: the deep, narrow major groove is RNA's electronegative pocket), hybrid -3.63 / -2.91.
Production: every fragment, classes rna, hybp, hybm (Family K's own), capped molecules, central 21
bp scored; ~12,700 CPU-hours; probe on two fragments queued (34154086) before launch.

**Family M, triplexes: template set** (`data/triplex_templates_v1/triplets.json`). RCSB protein-free
DNA "triplex" entries: 27 hits, 26 downloaded; 9 carry Hoogsteen pyrimidine-motif triplets (third-
strand pyrimidine N3 within 3.3 A of the Watson-Crick purine's N7): T*A-T 24, C*G-C 15 (plus a few
mismatched C*A-T / T*G-C). For each type, the medoid triplet's third-strand nucleotide (heavy atoms,
base, sugar and phosphate) in the Tsukuba frame of its purine: T*A-T from 1BCB (median RMSD of the
24 to the medoid 1.14 A), C*G-C from 149D (15; 0.81 A). Builder next: an intramolecular (H-DNA)
triplex places these on the cgNA+ duplex frames of the purine strand; the C*G-C third-strand
cytosine is N3-protonated (the triplet exists only so), which needs a protonated-cytosine charge
set for the electrostatics.

**Triplex builder, first geometry check** (`src/physics/fold/triplex_build.py`, core of an H-DNA
pyrimidine-motif triplex: cgNA+ duplex of the purine arm + third-strand nucleotides from the medoid
triplets in the purine base frames). Test GAAGGAGAAG: third-strand N3 to purine N7 2.82-2.88 A (the
solved Hoogsteen geometry); 1.6 percent of third-strand heavy atoms within 2.5 A of the duplex (min
2.1 A); but consecutive third-strand O3'-P distances 1.1-4.0 A (ideal 1.6): the triplets sit on a
B-DNA cgNA+ duplex whose rise/twist differ from a triplex's, so the independently placed backbone
does not close. Next: restrained minimisation with the third strand bonded (bases held at their
Hoogsteen positions), then electrostatics with the third-strand cytosines N3-protonated (a sourced
protonated-cytosine charge set is needed; not yet located). Spacer loop and the displaced purine
strand are outside the core model.

**Family M MD ensembles, first probe (c-MYC model, GBn2, 300 K).** 120 ps took 2,453 s on one core;
frames 1.1-1.5 A from the minimised model. With the tetrad core restrained weakly (200 kJ/mol/nm^2)
the core drifted: inner-site potential -91 to -104 kT/e across frames against -113 minimised, two-
K+ binding -130 to -261 kJ/mol against -284; one frame's stretched Hoogsteen contacts fell outside
the 3.3 A detection cutoff and its geometry failed. Real cores are held by the channel cations a
continuum solvent cannot keep, so the drift is an artefact. Changed before production: core heavy
atoms held as in the minimisation (2000 kJ/mol/nm^2) so the ensemble samples the loops; tetrads
taken from the minimised model for every frame; 10 ps heating + 50 ps sampling (5 frames). Re-probe
34156698.

**Family A: electrostatic persistence length over 4 window sizes** (PLAN.md item; assembled from
existing solver output, no extra solves). Odijk-Skolnick-Fixman with the PB-renormalised charge:
l = l_B lambda^2 / (4 kappa^2), lambda = (uncompensated charge of both phosphates per base pair)
/ 3.38 A, each phosphate's uncompensated charge = 1 - its condensed counterion charge (APBS cell
around the phosphate; the A_W/C_condensed channels), kappa from each condition's ionic strength,
l_B 7.1 A. Channels: per-bp windows of 3, 5, 7, 11 bp at 150 mM (`A_lpel_w{w}_c150`) and the scored
21 bp at every condition (`A_lpel_w21_{t}`). On a production fragment: 27.95 nm (10 mM), 10.66 (25),
5.09 (50), 2.38 (100), 1.51 (150), 0.34 nm (500 mM); smooth along the 21 bp. Definitional caveat:
the cell-based condensed charge is 0.36 e per phosphate here, not Manning's 0.76, so lambda and
hence l (as lambda^2) are larger than the Manning-renormalised OSF estimate (~0.2 nm at 150 mM);
the channel is the OSF value of the PB atmosphere as this pipeline measures it, so stated.
Still open in A: local dielectric response (a solute-dielectric 4 solve beside 2) and the
electrostatic component of base-pair opening (opened-structure solves) -- supplementary passes.

## Training stopped; every partial-capacity model deleted (user's instruction, 2026-11-22)

User: "do not train until we have the data so our models can train everything, stop training, delete
models that do not have full capacity", and, asked for the scope, "delete everything related, but make
sure we are still able to train the new PANGEA models which will have full capacity".

- **Stopped:** every training job (70 queued or running: v5 50M/200M/1B, v4f 200M, the 2M seed
  replicate). No training of any kind until the final-ladder inputs are complete.
- **Deleted (212 paths, 112.4 GB):** all 46 run directories under `runs_v3/` (v3, v4, v4f, v5 and
  probes, weights and metrics), the 13 earlier runs under `runs/` (stage 1, v2), two smoke-test model
  folders, the READOUT features computed with the v3_L10M_tie encoder (33 GB), and every result file
  computed from a trained model: the 140 READOUT head results, the READOUT summary and submission log,
  ablations_v1targets, family_l_arms, rank_sweep_v2, scaling_v2, universality_v2, ladder_v3_vs_v4,
  their two figures, and v3_submission.txt.
- **Kept, because the final models need them or no model was involved:** all code and frozen
  snapshots, every label set and side shard, the corpus and auxiliary data, environments, the physics
  results (gates, convergence, QC, noise ceilings, structure classifications, validations computed
  from the physics labels), the GC + 5-mer ridge baselines, `results/v3_continuations.txt` (also the
  log of label production), and this project record. The numbers reported earlier for trained models
  (RESULTS.md, sections on v3-v5 models, READOUT, conformal and composition evaluations) remain as
  text, but their files are deleted; they will be re-measured on the final ladder.
- **Checked:** the training code is intact (269 model/data/loss tests pass; the 28 failures in the
  training environment are physics tests needing ViennaRNA/OpenMM, which pass in the physics
  environment); the trainer parses its arguments; the final-ladder inputs present now are Families K,
  J (revision 2), N, M (corpus) side shards, the auxiliary windows with J/N/K, and the quadruplex and
  rare-event window labels; still to come: Bq (QC), Family M on the quadruplex windows, Family A long
  shards, Family N regenerated with 2'-O-methyl, and the further plan items in progress.

**Withdrawn the same day: the in-fragment electrostatic persistence channels.** Two errors.
(1) The plan places this quantity on window sizes of 10 to 100 bp computed on a separate set of
long fragments ("the electrostatic contribution to persistence length and the charge density
integrated over 10 to 100 bp windows ... need fragments long enough to contain the window"), not
on 3-11 bp windows inside the scored 21 bp. (2) The cell-based uncompensated charge (0.64 e per
phosphate at 150 mM) is not the effective charge OSF needs: OSF with it gives 28 nm at 10 mM, while
OSF with Manning-renormalised charge gives 3.2 nm at 10 mM and 17 nm at 1.9 mM -- the measured
salt dependence of DNA persistence length (about 47 nm at 100 mM rising to about 64 nm at 1.9 mM)
fits the latter. The channels are removed; the long-fragment set replaces them (below).

**Long solutes at high salt solve on the current construction (probe 34286873).** A 121 bp
window solved at 50, 150 and 500 mM with the production construction (exclusion radius 2.0 A,
exact 0.35 A fine spacing, map-based solvation; APBS's energy evaluation overflowed and was
skipped, the potential itself converged): against the production solutes (61 bp at 50/150 mM),
every potential channel agrees to RMS 0.021 kT/e at 50 mM and 0.065 at 150 mM (max 0.12 and 0.60)
-- the per-bp values are converged in both, so tiles of production-length solves reconstruct a
long molecule's per-bp quantities at these salts. At 500 mM the 121 bp solve is corrupted (RMS 22,
max 473 kT/e against the 41 bp solute) although APBS does not abort: the nonlinear divergence the
convergence study recorded, still present there.

## 2026-11-22 (evening): labels for every family -- what changed and why

**Family K electrostatics launched (job 34284715, 9,568 tasks x 20 fragments x 3 classes).** Probe
34154086: 6/6 class solves, 138-199 s each, 3.7 GB; potentials 1 A out at 150 mM (RNA) major -3.74,
minor -3.36 kT/e, and the profile outward shows the A-form pocket: major groove still -3.13 at 4 A
and -3.20 at 7 A, minor -2.05 and -1.29 (the deep, narrow major groove holds its potential; the
shallow minor groove opens to bulk). The earlier "major -4.04 / minor -1.71" came from an uncapped
24 bp test whose end effects lowered the minor groove; the capped 65 bp construction is production.

**Family A final pass queued (34285092, after the main array).** 18 fragments failed and 3 lost
their 10 mM condition because 121 bp solves at 10-25 mM exceeded 16 GB (sacct OUT_OF_MEMORY; APBS
killed while its log was buffered, which is why the error text was only the citation banner). The
pass lists every missing .npz (also rows of timed-out tasks) and every recorded failed condition
and re-solves them at 40 GB; the long-shard step reads the supplementary and fix folders too.

**Family J oxDNA v2 (job 34288949, all 191,344 fragments).** Added per nucleotide over the same 80
frames: the oxDNA2 Debye-Hueckel energy (backbone-site charges, 0.15 M; the plan's "electrostatics
without a complementary strand"), the backbone charge spacing (its inverse is the local linear
charge density), the probability of a hydrogen-bonded partner (self-structure), base-site contacts
(burial) and the probability that a base is exposed (unstacked both sides, unpaired; hydration).
Test fragment: DH 0.006-0.02 oxDNA units per nt (ends lower), spacing 0.65 nm, exposed 0.55 at the
5' end. The first pass (39,763 fragments without these) is superseded; everything is rerun.

**Protonated cytosine (C+) charges computed** (`src/qc/resp_cplus.py`, `physics/data/dcp_charges.json`):
RESP (HF/6-31G*, MK points) on the 3',5'-O-dimethyl nucleoside cation, sugar atoms fixed to AMBER
OL15 DC, hydrogens relaxed by GFN2-xTB with heavy atoms fixed (psi4's optking rejected the
frozen-atom list); relative RMS 0.049; N3 -0.291, H3 +0.321, O2 -0.279 (DC: N3 -0.775, O2 -0.655).
OpenMM templates DCP/DCP5/DCP3/DCPN (`fold/dcp_forcefield.py`) for triplex and i-motif models.

**pKa shifts: the PCM discretisation does not converge -- not launched.** Same contexts, GePol
element area 0.5 -> 0.3 A^2: AAAA shifts -0.99/-1.93/-0.85/-1.51 -> -0.21/-0.98/-0.03/-0.56 pKa
units, i.e. the cavity mesh moves the result by as much as the context effect being measured (and
GCGC's SCF failed at 0.3). Testing ddX (domain-decomposition PCM, smooth in its discretisation;
pyddx 1.0.0 in a separate environment so the running DFT tiers are untouched) at lmax 6/8/10.

**Family M v3 design.** (1) Template set v2 (`fold/build_g4_templates_v2.py`): all 166
unimolecular quadruplexes of 2-4 tetrads -- parallel 67, hybrid 30, antiparallel 8 (three-tetrad);
53 two-tetrad -- with each loop's type (propeller, lateral, diagonal) and a fold descriptor (type and
sense around the stack), e.g. hybrid-1 p+l+l+, hybrid-2 l+l+p+, basket l+dl-, chair l-l-l-.
(2) `fold/g4_multi.py`: per topology class a solved core of that class, loops from solved loops of
the same type and length (anchor fit <= 2 A), ensemble of 3 models spanning the class's folds.
Leave-one-out (running): with one model per class, hybrid targets were rebuilt in the wrong hybrid
fold (core RMSD 7 A: sequence alone does not choose hybrid-1 vs hybrid-2); with the fold-diverse
ensemble the target's fold is among the three for 2F8U, 2GKU, 2HY9 and 148D (core 1.2-1.4 A), not
for 143D (basket needs a 3-nt diagonal loop, which only 143D itself has). (3) `g4_select.select_v3`:
two-tetrad cores when no three-tetrad one exists; no loop cap (feasibility is decided by solved loops).
(4) `structure_folds.py` v3: per class p/h/a, 64 channels (ensemble means over models and MD frames,
across-model spread, MD fluctuation, residue potentials and RMSF). Scale: 3,793 corpus fragments have
a three-tetrad and 45,606 a two-tetrad candidate; 47 % / 76 % of a sample have >= 1 feasible class;
with the quadruplex windows about 50,000 fragments and 300,000 models. MD on GPU (CPU GBn2 MD runs
5.6 ns/day for 523 atoms; CUDA failed on B200 nodes: CUDA_ERROR_UNSUPPORTED_PTX_VERSION).

**Family M triplexes** (`structure_triplex.py`): H-DNA core (the purine arm's longest pure purine
run >= 6 between G-C caps, third strand T/C+ from the medoid triplets), minimised with the third
strand bonded (GBn2; duplex and third-strand bases restrained, third-strand backbone free), PB at
10/150/500 mM NaCl: per-bp sector potentials (reverse-complement-symmetric sets), solvation, and the
electrostatic free energy of third-strand binding on one grid. 6,307 corpus fragments carry an
H-DNA candidate. Test 34292025.

**Family A extras** (`run_family_a_extras.py`, test 34290512): electrostatic component of base-pair
opening (central pair opened 60 degrees toward the major groove about each base's C1', the rest of
the molecule fixed; dG = dE_Coulomb(eps 2) + d sum of per-atom map solvation, at 10/50/150/500 mM)
and the local dielectric response (image potential of a unit probe 1 and 4 A outside the surface in
the six sectors of the central pair; Poisson, no ions; probe-centred grid against uniform water).
Coarse-grid follow-up (same probe): a 0.7 A coarse level does not repair 500 mM either (RMS 12.4,
max 307 kT/e against the 41 bp solute; 150 mM unchanged, RMS 0.058), at 2.8x the time and 8.2 GB.
The long-fragment set therefore tiles production-length solves (converged per base pair at every
condition) rather than solving long molecules at high salt.

**Triplex model test passed; production launched (34296222, 7,021 candidates).** panel_7a25d630498913a4
(GAAGG repeat, 30 triplets, 15 C+): third-strand O3'-P gaps closed (largest 1.618 A after the GBn2
minimisation); electrostatic third-strand binding +808/+793/+781 kJ/mol at 10/150/500 mM (+26-27
per triplet: a polyanion binding a polyanion, desolvation and repulsion, less unfavourable as salt
screens it -- the sign of triplex salt dependence); minor groove 1 A out -5.77 kT/e at 150 mM against
-2.84 in the third-strand-filled major groove; 1,289 s per fragment.

**pKa shifts: ddX does not converge in lmax either (lmax 6 / 8 / 10 for AAAA: +0.33/+0.25/+0.07,
-0.18/-0.26/-0.48, +0.18/+0.11/-0.10, -0.09/-0.17/-0.39 pKa units), so production stays on hold.**
The shift is a small difference of large reaction-field energies of a cation and a neutral in two
cavities of different shape (stacked step, free base), and discretisation error does not cancel.
Testing lmax 12/14/16 (Lebedev 770/974/1202); the GC context moves less (GePol 0.7 vs 0.5 A^2: within
0.25), the AT-rich one most.

**Explicit-solvent MD hydration library** (`md_hydration.py`, `run_md_hydration.py`): Family G's
groove hydration occupancy, residence time, spine continuity, water bridge probability and
interfacial water entropy (first-order inhomogeneous solvation theory), and Family K's 2'-OH
hydration and dynamics (occupancy, residence, O2'-O4'(i+1) hydrogen bond, C3'-endo fraction, O2'
RMSF) -- the plan's "MD-derived hydration" / "hexamer-scale RNA MD", which the stage-1 Family G had
declared unreachable. 12-mer GCGC+context+GCGC per RC-unique tetramer and class (DNA, RNA, hybrid;
408 runs), OL15/OL3 + OPC + 0.15 M KCl, HMR 4 fs, 1 ns equilibration + 10 ns production sampled every
2 ps, analysed on the fly. GPU test 34295965.

**Family B trimers** (`build_qc_fragments.py --trimers`, `run_psi4.py --trimer`): the plan's "stacked
dimers and trimers": both base-pair trimers of each tetramer context that contain its central step
(t5, t3), three-body stacking E3 (many-body expansion, each subsystem in its own basis), pair terms,
vertical IP (dSCF UKS), the hole's distribution over the three pairs (cation-minus-neutral Mulliken
charge per pair), HOMO/LUMO and the HOMO's weight per pair (hole localisation; GGG traps). Test at
def2-SVP on AAAA_t5 and GGGG_t5 (34295873) before TZVP production (272 trimers).

**Family N, DFT on modified stacks** (`src/qc/build_qc_rna_mod.py`): every RC-unique tetramer's
central step in A-RNA and in the hybrid (cgRNA+/cgHYB+ geometry), unmodified and with one m6A,
inosine or pseudouridine at each RNA position that can carry it (m6A methyl anti; inosine O6 and N1-H;
Psi as the uracil ring attached through C5, N1-H where U has C5-H), modified atoms and every
hydrogen relaxed by GFN2-xTB with heavy atoms fixed; ~1,100 fragment sets (build 34295699) for the
Family B stacking and hydrogen-bond tier. 2'-O-methyl is a sugar modification, outside base-only
fragments; its thermodynamic effect is Family N's Kierzek 2006 channel.

**Family J oxDNA channels** (`single_strand_oxdna.py`, 46 channels, packer `pack_family_jox.py`):
also chemical reactivity as exposure of each probe's reactive atom -- permanganate (thymine C5=C6
face, protected by stacking: unstacked probability at T), DMS (A N1, C N3), glyoxal (G N1/N2) and
CMCT (T N3, G N1), Watson-Crick-face atoms protected by pairing (1 - paired probability at those
bases). Test fragment: 1.54 e/nm charge density, Debye-Hueckel +0.12 kcal/mol per nucleotide, 0.47 nm/nt
at 20 pN (freely-jointed ssDNA ~0.5).

**i-motif model test passed; production launched.** Telomeric C-strand (CCCTAA x4): 6 C.C+ pairs,
narrow-groove potential 1 A out -2.87 kT/e against -0.71 in the wide grooves at 150 mM (the narrow
grooves carry the close phosphates), solvation -6,913 kJ/mol, 2,559 s on 4 threads. Two descriptors
were redefined before production: the core potential at the pair centres read the shared protons'
own smeared charge (+300 kT/e); it is now on the axis midway between consecutive pairs. The axis
points 6 A beyond the stack can fall inside loops that cap an end (-36 kT/e at one end); the end
potentials are now surface-referenced along the axis (1 A outside the surface). 25,868 candidates
(C-rich strand of a G3+ quadruplex candidate with a feasible core assignment).

**Quadruplex MD frames: potentials and solvation only.** The ion-binding PB cycle (complex and
ions-alone solves, two more per condition) runs on each minimised model, not on its MD frames: the
test fragment needed 4,108 s of APBS for 3 models x 6 structures, and ~590,000 models (sequence-built
loops make nearly every candidate buildable in at least one class) would need ~200k CPU-hours with
it on every frame. Ion binding keeps its across-model spread; its MD fluctuation channel is not
declared (the core is restrained in the MD, so the ion sites barely move).

**Family A extras launched (28,784 fragments, the 15 % "extras" hash subset).** Test 34293674 after the
fix (phosphates, C3' and C5' of the opened nucleotides held; GBn2 relaxation): opening the central
T.A pair 60 degrees costs dE_Coulomb +27.65 kJ/mol (the pair's hydrogen bonds and base-base
electrostatics in the solute dielectric) and gains dG_solv -12.5 (the opened edges hydrated): dG_open,elec
= +15.14 kJ/mol at 10, 50 and 150 mM and +15.04 at 500 mM -- local and salt-independent to 0.1 kJ/mol, so
production computes it at 10 and 150 mM (the solves at 50 and 500 mM would repeat the same number).
Dielectric image potentials 0.9-5.0 kT/e (1 and 4 A out, six sectors); the method reproduces the
Kirkwood image of a dielectric sphere to ~10 % (0.65/0.35/0.14 kT/e at 2.4/3.4/5.4 A from the surface
against 0.57/0.31/0.13). A rebuilt closed structure is not bit-identical to the production solve (per-bp
solvation differs by up to 0.57 kJ/mol), so both states are solved on one grid in this pass.

**Hydration MD test passed (34295965, AATT, 2 ns production each, RTX A5000 OpenCL ~600 ns/day);
library launched (34298686, 408 runs).** DNA: minor-groove first-shell waters 2.05-2.22 per A.T pair
against 3.0-3.5 per G.C pair (more acceptors), residence 25-30 ps in the A-tract minor groove against
20-21 ps at G.C pairs and 8-16 ps in the major groove; translational -TS ~1 kcal/mol per pair and
groove. RNA: 2'-OH 2.0-2.5 first-shell waters, residence 6-7 ps, O2'...O4'(i+1) hydrogen bond 13-51 %,
C3'-endo 83-98 % (A-form), O2' RMSF 0.75-1.1 A; minor-groove residence 8-14 ps (shallow, open groove).
The spine-continuity count used one of the two cross-strand offsets and read ~0 even in the A-tract;
production counts both (fixed before launch).

**Family B trimer tier launched (34298869, 272 trimers, TZVP).** Test (def2-SVP, AAAA_t5, 664 s):
three-body stacking +0.001 kcal/mol (B3LYP-D3BJ has no three-body dispersion; the electronic three-body
term is tiny), nearest pairs -19.5/-20.1 and the 1-3 pair -0.7 kcal/mol (no counterpoise in the trimer
pair terms; the dimer tier carries the counterpoise-corrected stacking), vertical IP 6.83 eV, HOMO 70 %
on the middle pair, the dSCF hole spread 0.37/0.32/0.31 over the three A.T pairs.
Trimer check (same test, GGGG_t5): vertical IP 5.72 eV against 6.83 for AAAA_t5 (the G run is the hole
sink), HOMO 92 % on the middle pair (the central G of GGG, the known oxidation hotspot); the dSCF hole
spreads more (0.36/0.36/0.28), B3LYP's delocalisation error, which is why both the HOMO weight and the
hole fraction are kept as channels.

**Family G rate channels: two sourced, two not.** Depurination: k = 4e-9 s^-1 at 70 C, pH 7.4,
physiological ionic strength, Ea 31 +- 2 kcal/mol (Lindahl & Nyberg, Biochemistry 11:3610, 1972; from
its abstract) -> 3.2e-11 s^-1 at 37 C; denatured DNA ~4x faster (secondary citation of the same
work); "independent of sequence" otherwise -- so the channel is k_ds (1 - p_open) + 4 k_ds p_open per
purine with Family E's opening probability at 37 C. Cytosine deamination: 1e-10 s^-1 single-stranded
and ~7e-13 s^-1 double-stranded at 37 C, pH 7.4, Ea 28 kcal/mol (Frederico, Kunkel & Shaw, Biochemistry
29:2532, 1990; abstract), applied the same way. Not implemented, for want of a retrievable primary
value: depyrimidination (Lindahl & Karlstrom 1973 not retrievable -- ACS refused the fetch and Europe
PMC carries no abstract; only a textbook range "20- to 100-fold lower than depurination" was found,
which spans a factor five) and sequence-dependent abasic-site stability (Gelfand et al., Biochemistry
37:7321, 1998 establishes the context dependence; its parameter table was not retrievable). Both stay
declared unavailable with these reasons rather than carrying an invented number.

**Quadruplex MD on GPUs: batched replicas, full MD length kept.** One 523-atom GBn2 model is
kernel-launch bound on a GPU (OpenCL; CUDA does not load on these drivers): H200 250 ps in 20 s, and
four concurrent processes only 1.15x faster. Packing models as non-interacting replicas in one system
(80 A lattice, GBn2 with a 2 nm non-periodic cutoff, beyond which the 0.15 M-screened interaction is
<8% of its unscreened value; `md_ensemble_batch`, and `minimise_batch` for the minimisation) scales:
1 / 16 / 64 / 128 replicas give 9.9 / 90.5 / 141.7 / 130.9 ps/s on one H200 (test 34299862). Every
model keeps the full protocol (20 ps heating, 100 ps equilibration, 500 ps sampled at 5 frames): ~590,000
models x 620 ps = ~720 GPU-hours at 64 replicas, sequenced over the account's GPU share rather than
shortened (an earlier draft of this entry shortened the MD to 300 ps for compute; withdrawn before any
task ran). Build array: 122 tasks x 600 fragments, 3 workers sharing each GPU; quadruplex windows
first, then corpus three-tetrad, then corpus two-tetrad candidates; the APBS stage follows on CPUs as
records appear.

**Quadruplex model accuracy by class (leave-one-out, `results/g4_multi_validation_v2.json`).** Each of
51 solved unimolecular quadruplexes re-modelled from the other 165 templates, three fold-diverse
models. Best-of-3 core RMSD median: parallel 2-tetrad 0.98 A (6/6 folds in the ensemble), hybrid
3-tetrad 1.27 A (12/19), antiparallel 2-tetrad 1.16 A (10/18), antiparallel 3-tetrad 9.18 A (2/6),
hybrid 2-tetrad 4.69 A (0/1 modelled). Antiparallel three-tetrad folds are the weak class: with one
target left out, too few same-fold templates remain. Loops are less accurate than cores (propeller
1-2 nt 1.8-4.5 A; lateral 3-4 nt 3-14 A). Production keeps every class (the plan models all
topologies); these numbers go with the Family M channels as their structural uncertainty, and the
per-class model spread (`M_g4*_model_spread`) carries it per fragment.

**Correction: xTB did not hold the "fixed" atoms in the RESP fits.** `resp_cplus.py` (DC+) and the first
draft of `resp_modified.py` relaxed hydrogens with GFN2-xTB 6.7.1 under a `$fix` block. Measured on the
m6A nucleoside, the atoms listed there moved 0.38 A in 15 optimisation steps; the DC+ fit's own record
(`tmp/cplus/cplus_resp.json`) shows heavy atoms moved up to 0.48 A, so its geometry was a near-free
vacuum relaxation, not the B-form one with relaxed hydrogens that was logged. On the RNA builder's
m6A geometry (O5'-C5' 1.30 A, C-H 1.19 A) the same run failed (no convergence in 219 steps, gap
0.02 eV) and HF did not converge on it. Both fits now hold the A- or B-form conformation by its
torsions (glycosidic chi, two endocyclic sugar torsions, gamma, delta and the two cap torsions;
harmonic, 0.5 Eh/rad^2) and relax everything else: m6A then converges in 40 steps (gap 3.2 eV) with
O5'-C5' 1.406, C8-H8 1.071, N6-H 1.009 A, chi -159 and delta 90 degrees. The modified-nucleotide
fits also impose the modified group's total exactly (a second Lagrange constraint) instead of
spreading a correction afterwards. DC+ is being refitted the same way (`cplus_resp_v2.json`); the
triplex and i-motif runs used the first DC+ charges, and they are rerun if the refit changes the
base charges materially.

**Family W: the window ladder at 100-500 bp and the wide branch's targets, on 10,240 bp fragments**
(`src/physics/longrange.py`, `src/data/sample_long_fragments.py`, production 34308662). The plan's
window ladder (10-500 bp, "everything defined over an extent") and wide branch (5-10 kb: "persistence
length over long windows, nucleosome free energy, supercoiling response, R-loop formation") need long
fragments, so Family W computes them with Families D, E and F's own models on 21,391 genomic tiles
(min(2,000, all) non-overlapping 10,240 bp ACGT-only tiles per panel genome):
- Mechanics: Family D's quantities need only the 6x6 diagonal blocks of the marginal step covariance,
  and the cgNA+ stiffness is banded (half-bandwidth 41 at 121, 1,000 and 10,240 bp), so they come from
  an exact banded selected inversion (Takahashi recursion, `banded_inverse.py`): equal to the dense
  inverse to 4.8e-14 relative (400 bp; 29 s dense against 0.09 s), Family D's fluctuations and
  windowed bending equal to 3e-13; 7.7 s for a 10,240 bp fragment, where the dense inverse would need
  ~460 GB. Tracks: bending persistence length, anisotropy and soft-axis angle (stored doubled, period
  360) over 10/20/50/100/200/500 and 1,000/2,000/5,000 bp windows; nucleosome wrapping free energy of
  every 147 bp window at Family D's three ionic strengths (on 61 bp fragments Family D can use only 60
  of the 146 steps), with rotational phase and amplitude; cyclizability at Family D's four loop lengths
  and loop-closure free energy at its five lengths for every window; the domain's moduli and
  twist-writhe partition. Both strands, combined by Family D's parities.
- Supercoiling response: Family E's supercoiled helix-coil model with the fragment as one domain at
  its four superhelical densities. The open-count recursion is truncated at k* + 40 widths of the
  stress Gaussian (accepted only when the distribution's last 16 entries hold < 1e-15 of the mass):
  identical to the untruncated recursion at 1 kb (difference 0), 11 s per density at 10,240 bp.
- R-loops: Stolz et al. 2019's model as the authors' code implements it (chedinlab/rlooper): one
  R-loop of any length >= 2 bp or none, G = a + sum B + 2 pi^2 C K (alpha + mA)^2 / (4 pi^2 C + K m),
  a = 10.5 kcal/mol and A = 1/10.5 (paper; the code's defaults are 10 and 1/10.4), C = 1.8 and
  K = 2200 RT/N (code; the SI was not retrievable), 310 K, domain = fragment, sigma -0.07 (the
  authors' default), -0.06, -0.04, -0.02; each strand as the non-template strand.
- Exact reverse-complement equivariance of every track and scalar (residual <= 3e-16 relative on a
  2.2 kb test); 110 channels registered (59 tracks, 51 domain scalars), all matching production.
- Leakage: each long fragment lists the library fragments it contains (>= half of the fragment's
  distinct canonical 11-mers present and >= 26 of them); the trainer joins a long fragment to the
  similarity groups of everything it contains. The first table used half alone, and every
  low-complexity library fragment (one or two distinct 11-mers) was "contained" in 21,359 of 21,391
  tiles; the 26-k-mer floor fixes that (tiles and sequences unchanged, same seed).
Also added for the 61 bp library, family Fr (`rloop_energetics.py`): the plan's "R-loop initiation and
elongation free energy" from the same model without stress (Family F's R-loop channels are the
RIZ/REZ pattern scores): per-step B on each strand, the mean elongation free energy, the cheapest
R-loop (junction plus best hybrid segment) and the ensemble free energy; side shards with Gr (34307335).

**DC+ charges refitted; triplex and i-motif production restarted with them.** The torsion-constrained
refit (`tmp/cplus/cplus_resp_v2.json`) fits its potential better (relative RMS 0.0274 against
0.0488) and moves ring charges by up to 0.23 e (N1 -0.23, O2 -0.19, C6 -0.17, N4 +0.17, C2 +0.16;
base total unchanged, +0.9369). `physics/data/dcp_charges.json` now holds the refit (the first fit
kept as `tmp/cplus/dcp_charges_v1_superseded.json`). Those charges set both the DCP force field used to
minimise protonated cytosines and the PB charges, so the triplex run (168 of 7,021 solved; set aside as
`famM_tx/solves_v1charges_superseded`) and the i-motif run (not started) were relaunched on frozen code
`famMfold2`: triplex 34308933, i-motif 34308934.

**pKa shifts: neither implicit-solvent model in psi4 converges on a stacked step.** ddX at lmax
6/8/10/12 gives AAAA's A N7 shift +0.33/+0.25/+0.07/+0.60 and N3 -0.18/-0.26/-0.48/+0.16 (34291334-6,
34296108). The cause is the stacked step's own solvation energy, which wanders by 1-2 kcal/mol with
lmax without converging (-32.2/-32.6/-34.0/-31.6 kcal/mol), while the single base converges smoothly
(-9.44/-9.48/-9.49/-9.52). GePol PCM was already unconverged in its mesh. So pKa production is held
until a solvation route converges on stacks: the next candidate is gas-phase DFT energies (no
discretisation) plus finite-difference Poisson-Boltzmann solvation of each species' DFT
electrostatic-potential charges, with grid convergence shown explicitly, as Family A's solves are.

**Wide branch widened to the plan's reach.** `WideBranch` was configured with stride 4 and kernel-3
blocks at dilations 1..32: a receptive field of (1 + 2 x 63) x 4 + 9 = 517 bp, narrower than the plan's
"5 to 10 kb" and barely wider than the local trunk's ~300 bp. It now uses stride 16 and dilations
1..128 (8 blocks): (1 + 2 x 255) x 16 + 33 = 8,209 bp (`WideBranch.receptive_field`), 0.33 M
parameters at width 96. No model had trained with it (it was off, the fragments being 61 bp); it
trains on Family W's 10,240 bp fragments.

**Trainer support for Family W (code and a CPU dry run; no training).** `data.WideSet` loads the
Family W shards (registry metadata, the main dataset's own transform rules on Family W's values;
tracks in host memory as float16), and `join_groups` joins each long fragment to the similarity
groups of the library fragments it contains before the three-way split (union-find); unlinked
long fragments are placed by a hash of their id at the same fractions. `train.py --wide-shards`
appends Family W's columns to the plan, switches the wide branch on, gives the extent-defined
window tracks pooled-feature heads (`PANGEA(pb_window_rungs=...)`, box means in O(L), the plan's
"window channels come off pooled representations at each scale"), adds a Family W forward pass to
every step (`--wide-per-step`, `--wide-weight`), pads main-fragment targets and evaluation results
with NaN for Family W's columns, and scores Family W's held-out fragments in `final_metrics.json`
(`evaluate_wide`). Attention and the pair modules are quadratic in length, so on inputs longer than
128 bp they run on overlapping 128 bp blocks blended by triangular weights; inputs up to 128 bp (every
61 bp fragment) take exactly the old path. Reverse-complement equivariance stays exact (residual 0.0
at 61 and 1,000 bp with attention, pair bias, the wide branch and pooled window heads). Dry run: two
main shards + 39 Family W fragments, width 32, one epoch, through final evaluation.

**Family A extras on the whole corpus (was a 15 % subset).** The subset launch (28,784 fragments) was
a cut I made after the no-subsets instruction; the plan's "electrostatic component of base-pair
opening" and "local dielectric response" are Family A channels of every fragment. Measured on the
first 36 production fragments, the opening free energy at 10 and 150 mM differs by at most 0.19
kJ/mol (median 0.05) while it spans 4.6-26.9 kJ/mol between fragments, and the plan asks for it at no
particular salt; the 10 mM solve is the expensive one (121 bp solute, 40 GB, ~39 min per fragment).
So the channel is defined at 150 mM (`a_extras_channels.py`: Ax_open_elec_c150, its Coulomb and
solvation parts, and the dielectric image potential in the six sectors at 1 and 4 A), the pending
subset tasks were cancelled (249 running tasks finish both conditions; their 10 mM values stay as the
salt-independence evidence), and every other fragment (189,351) is solved at 150 mM only (array
34311833, 40 per task, 16 GB; the scheduler caps arrays at 10,000 tasks and starts 200 jobs per user
per hour).

**Modified-RNA QC library rebuilt with heavy atoms held by distance constraints.** The library
(`build_qc_rna_mod.py`, 864 contexts) relaxed each capped base's hydrogens under the same xTB `$fix`
block that does not hold atoms, then superposed the result back, so heavy atoms were xTB-relaxed rather
than "as built", and 7 bases failed ("|grad| > 500"). It now holds the built heavy-atom framework by
xTB distance constraints (`$constrain atoms:`, 2 Eh/bohr^2), refuses a result whose constrained atoms
moved > 0.05 A after superposition, and writes the constrained atoms exactly as built. Both previously
failing contexts (AAGG, CATG) now build with no failures; the full rebuild is 34312183 (the first
build kept as `qc/rna_mod_tetramer_v1_fixbroken`). Its DFT tier had not started. The main DNA QC
libraries never relaxed with xTB and are unaffected.

**pKa route under test: gas-phase DFT plus Poisson-Boltzmann solvation (`src/qc/pka_pb.py`, test
34312851 on AAAA and GCGC).** The same thermodynamic cycle, rigid geometries and proton placement as
`run_psi4.pka_context`, but the solvent enters after the SCF: gas-phase B3LYP-D3BJ/def2-SVP energies
(no discretisation) and, for each species, the electrostatic solvation free energy of its
Merz-Kollman charges (fitted to the gas-phase DFT potential) by finite-difference PB (solute
dielectric 1, water 78.54, no ions, Bondi radii, 1.4 A probe surface) at fine spacings 0.30, 0.20 and
0.15 A, the last step's change carried as each shift's grid error. Not repolarised by the solvent; that
term enters both legs of the cycle. Decided on its measured convergence; the ddX FMM/eta scan
(34309108-10) runs alongside.

**Family N electrostatic response to modification (`modification_electrostatics.py`, family code
"Nes").** Electrostatics is one of Family N's "main channels". For m6A, pseudouridine and 2'-O-methyl
on each base (the plan's list) plus inosine (as in Family N's thermodynamics), with an RNA or a DNA
partner strand, and every 3 nt context: the modified and unmodified 13-mers (GC GCG <ctx> CGC GC,
Family K's caps and build) solved with Family K's construction on identical grids (`solve_conditions`
now takes a prepared PQR and a fixed grid extent). The modified nucleotide carries its RESP charges
(fixed OL3 elsewhere; the residue stays at -1.0000 +- 0.0003) and pdb2pqr's AMBER radii of the same
atom type; new atoms are placed by superposing the relaxed RESP geometry on the site's base (or its
sugar for 2'-O-methyl). The isolated nucleoside's 2'-O-methyl orientation collided with the duplex
(nearest atom 1.6-2.0 A), so the methyl is rotated about C2'-O2' to the rotamer farthest from every
other atom (1.9-2.4 A after, the nearest being its own sugar). Response = modified minus unmodified
at the site pair (major and minor groove potentials 1 and 4 A out, the two strands' backbone sectors,
site and five-pair solvation), looked up per transcript position by context; 224 solves plus 128
unmodified. Inosine is solved in the A-U geometry (an I-U wobble would shift the pair); it is an
addition beyond the plan's list.

**Production arrays reshaped around the scheduler's start limit.** The cluster starts at most 200 of
a user's jobs per hour (`jobs_per_user_per_hour=200`) and caps arrays at 10,000 tasks, so arrays of
short tasks (20 fragments, ~2 h) ran far below the available capacity: Family K had ~7,800 such tasks
pending at 17 % done, Family J v2 ~9,100 at 4 %. Their pending tasks were cancelled and the remaining
work resubmitted on the same frozen code as large tasks (no change to what is computed; every runner is
resumable per fragment): K 156,843 fragments x 100 per task (34318664), J v2 183,349 x 150
(34318878), triplex 6,976 x 30 (34318914), i-motif 25,868 x 20 (34318915), with 16 h limits.

**Quadruplex loops built from sequence: validation (`results/loop_builder_validation_v1.json`, 14 loops
of solved quadruplexes rebuilt on their own cores).** Every rebuilt loop is connected (O3'-P <= 1.68
A) and clash-free (nearest contact >= 2.8 A). Loops of 3-4 nt land 2.1-6.3 A from the solved loop
(median 3.4 A); loops of 6-19 nt 4.5-18 A (median 10.8 A), as expected for loops that are
conformationally heterogeneous and solved as one conformer. Built loops are the fallback where no
solved loop of the right type and length fits (penalty 3 A in model ranking); the MD ensemble and
the model spread carry their uncertainty into the channels.

**Modified-RNA stack library rebuilt cleanly; its DFT tier launched.** With heavy atoms held by
distance constraints (and a stiffer constraint retried where the framework moved > 0.05 A, as one m6A
base did at 2 Eh/bohr^2), all 872 fragment sets built (no failures; `qc/rna_mod_tetramer`). DFT tier on
it, the same quantities and level as the main tier's ground-state geometries (B3LYP-D3BJ/def2-TZVP
counterpoise stacking and the two base-pair H-bond energies): 34319769, 128 shards.

**Family W split: positional containment and repeat hubs.** Positional containment (>= 26 of a
library fragment's 11-mers on one alignment diagonal of the tile, either strand; `--recompute-contains`)
replaced k-mer-set containment, which counted AT-rich fragments as "contained" in any AT-rich tile
(Plasmodium: median 705 contained per tile -> 63; human 52 -> 19). 79,022 library fragments lie in some
tile, 13,476 in two or more (young repeats, satellites, low-complexity tracts; one in 3,796 tiles), and
those chain tiles: one joined component of 12,725 of 21,391 tiles. Without the links of fragments held
by >= 3 tiles the largest component is 41 tiles. So fragments in >= 3 tiles (10,743, 5.6 % of the
corpus) are repeat hubs: they do not link tiles, and their similarity groups are kept on the training
side (`similarity_split_from_labels(force_train=...)`); every other contained fragment still joins its
tile's group. Largest joined group then 770 (tiles plus their fragments). Residual, by nature of
repeats: a held-out tile may contain a repeat whose 61 bp copy trained with main-corpus labels.

**Family N electrostatics: same-protocol references.** Unmodified controls fitted by the RESP protocol
(`resp_modified.py`: A0, U0) give the same fit quality as the modified nucleotides (relative RMS 0.19
for adenosine against 0.19 for m6A; 0.16 for uridine) and reproduce OL3's base charges to 0.075 and
0.081 e RMS (largest: correlated C8-H8 and C6-H6 pairs, 0.17-0.19 e). Differencing a modified fit
against OL3 would mix that protocol difference into the response, so each modification's reference is
the unmodified nucleotide with its base (or 2'-OH group: controls A0om, C0om, G0om, U0om, 34323217)
fitted the same way (`modification_electrostatics.CONTROL`); the queued run on OL3 references was
cancelled before it started.

**ddX converges on stacked steps with a wider switching region.** Scan of the AAAA step's solvation
energy (B3LYP-D3BJ/def2-SVP, `tmp/ddxconv`): with psi4's default eta 0.1 it oscillates by more than
1 kcal/mol over lmax 8/10/12/14 (-32.65/-33.98/-31.64/-33.44), identically with the fast multipole
method off, so the multipole approximation is not the cause. With eta 0.3 it converges monotonically,
-32.43/-32.63/-32.73/-32.79 (steps 0.20, 0.10, 0.06: within ~0.08 kcal/mol of the limit at lmax 14),
and the free base to +-0.02. The pKa shifts themselves are being checked at eta 0.3, lmax 12/14/16
(34325002-4); the gas-phase DFT + PB route (34312851) runs as the comparison. ddX keeps the solvent's
polarisation of the solute, which the PB route does not.

**pKa: the gas-phase DFT + PB route converges worse than ddX at eta 0.3.** Test 34312851 (AAAA,
GCGC; `tmp/pkapb`): each species' PB solvation still moves 0.13-0.29 kcal/mol between fine spacings
0.20 and 0.15 A (e.g. the AAAA step -39.43/-38.84/-38.55 at 0.30/0.20/0.15 A), so the summed
last-step error of the four species in a shift is 0.6-0.7 pKa units (shifts: AAAA A N7 -0.15/-1.04,
N3 -1.39/+0.96; GCGC G N7 -2.50/-2.51, N3 +2.90/+2.95). ddX with eta 0.3 reaches 0.06 kcal/mol per
step at lmax 14 and keeps the solvent's polarisation of the solute; it is the route if its pKa shifts
converge (34325002-4).

**Quadruplex GPU build validated and launched (34349846, 122 tasks x 600 fragments).** Test
34311852 (RTX 5000 Ada): 8 fragments, 72 models (3 classes x 3 models), full protocol (20 ps heat,
100 ps equilibration, 500 ps sampled at 5 frames), batched minimisation and MD: 652 s, all models ok,
frame RMSD to the minimised model 0.5-5.9 A (loops free, core restrained); the CPU solve stage then
took 2,397 s for one fragment (~49k CPU-hours for all 73,191). One model's energy (-426 against
-868/-891 kJ/mol/nt for its siblings) showed the batched minimiser's global stopping rule leaving a
replica unrelaxed; replicas whose own RMS force exceeds 3x the tolerance are now minimised again
alone (`minimise_batch`), in the launched code (`famM3b`).

**numba cache failures in Family W (450 fragments), fixed.** Tasks started on some nodes failed every
fragment in about a second with "cannot cache function '_hat_map_sv': no locator available" from the
cgNA+ SO3 module: numba writes its compiled-function cache next to the package, on the home fileset,
whose file count runs at 97 % of its 500k limit. `scripts/env.sh` (sourced by every task at start) now
points NUMBA_CACHE_DIR at node-local /tmp; the 450 fragments are rerun (34369810) and a final sweep
reruns anything still missing when the Family W array ends. No other array showed the error.

**2026-11-23 16:15: queue order, Family J packing path, Family N launch.** (1) The nice=700 I put on
the big CPU arrays to let tests start first left them below every other pending job of mine and near the
bottom of the cluster queue (about 340 CPUs over the last 12 h against ~1,100 over 48 h); the tests it
served have finished, so the arrays are back at nice 0, and the pKa tests may also start on scavenge
(requeueable). (2) `finalize_v7_inputs.sh` packed Family J from `famJox/solves` (v1, without the
Debye-Hueckel, H-bond and exposure observables the channel code reads); it now reads `solves_v2`.
(3) The 2'-OH controls (A0om 0.236, C0om 0.145, G0om 0.170, U0om 0.194 relative RMS; group sums held
at OL3's -0.0311 e) were checked in place on RNA and hybrid 13-mers (4 atoms change, net charge kept);
Family N electrostatics for all seven modifications is 34436498 (frozen code famNes_20261123_1610).

**2026-11-23 18:55: Family A extras leftovers passes; cruciform probe out of memory.**
The 150 mM extras tasks take 2,025 s per fragment on average (4,979 fragments: median 1,994 s, p90
2,390 s, p99 3,882 s, max 5,411 s), so a 40-row task meets its 14 h limit with about 14 rows unsolved;
63 had timed out by 18:50 and the rest follow as they reach 14 h. `scripts/famAx_leftovers.py` lists
every fragment with no npz that no pending or running task of a source array will compute
(`famA_extras/sources.tsv`: array, library, rows per task, offset; task-to-row mapping checked against
a task log), and each list runs as its own array at 150 mM, 20 rows per task (mean 11.3 h), same code
snapshot. Wave 1: `library_left_w1.tsv`, 1,240 fragments, array 34478525. The 29 requeued pending tasks
of the subset array 34298364 (both salts, 40 GB) were cancelled: their 382-fragment remainder is solved
at 150 mM in the leftovers passes like every other fragment, the channel being defined at 150 mM (the
1,611 subset fragments already solved at both salts stay as the salt-independence evidence).
Cruciform probe 34301681 (2 candidates, 24 GB): the reference solve of the first candidate was
OOM-killed and the job then reached its 5 h limit, so it measured nothing; the cost test 34330658
(64 GB) shows a peak of 26.8 GB and, for one candidate, ~6 min per model build and ~63 min per
condition per model (solvated + reference NPBE on an 81-million-point fine grid, 1089 x 289 x 257 at
0.35 A, covering the whole 121 bp model; APBS runs on one core).

**2026-11-23 21:50: side-chain search rebuilt (Isc v2); cruciform reference solve on APBS's Newton path.**
*Isc.* The 240-start test (34324960, AATT, 7.5 h) settles that the v1 search does not converge: across
three 60-start seeds the best energies differ by up to 10.8 kcal/mol (minor ARG -5.06 / -7.54 /
-15.84), and 240 starts were higher than some 60-start seed in 4 of 14 channels (minor ARG -8.83 vs
-15.84; major ARG, minor ASN, minor THR), while finding values 5-8 kcal/mol lower than any 60-start
seed in others (minor GLN -13.96 vs -5.2 to -5.8; major SER -20.39 vs -15.01). Independent
minimisations of one rigid conformer hit the low basins too rarely, and nothing kept the peptide in
the groove. `sidechain_probes.py` v2: Monte Carlo minimisation (10 chains x 40 moves per groove and
residue; moves: rigid-body about the side-chain tip, or a jump of phi, psi or one chi turning either
side of the bond; Metropolis at 1 kcal/mol on the complex energy), random backbone/side-chain
conformers at each chain start, and a flat-bottom restraint holding the tip within 6 A of a groove
acceptor of the central pairs (4.5 A left a narrow AT minor groove with no admissible serine pose in a
smoke test). `best` is now the interaction energy of the lowest-energy complex found, and `boltz`
weights distinct minima by their complex energy: the search minimises the complex energy, so its
minimum is the quantity that converges with sampling, and the lowest interaction energy over local
minima was not a well-defined target. Every minimum is saved (energies, tip-to-acceptor distance, tip
radius) so convergence is read from the record; results are per (context, residue) files that
`sidechain_channels.load_table` merges, and the packer refuses an incomplete table. Convergence test:
AATT and GCGC x 7 residues x seeds 0 and 1 (34492018, snapshot `code/iscmcm_20261123_2145`); the
production setting is fixed only when the two seeds agree.
*Cruciforms.* In the 64 GB cost test the reference solve (uniform dielectric 2, no ions) of the first
candidate's second model exceeded `run_apbs`'s 1 h limit, so that candidate failed. Cause: for `lpbe`
APBS uses its linear multigrid with a banded direct factorisation of the coarsest level (meth 2,
mgsolv 1); on the cruciform's 1089 x 289 x 257 grid the coarsest level is 137 x 37 x 33 and the
factorisation takes over an hour and 27 GB, while the solvated `npbe` solve (Newton, CG coarse
solver) takes 8-25 min. The reference problem has no mobile ions, so it is linear and `npbe` states
the same equation; on a 129 x 65 x 65 test grid the two paths give the same potential at all 545,025
points to the printed digits and the same energy. Cruciforms now pass `reference_equation="npbe"`
and `write_ndens=False` (the ion-density map was never read) and a 4 h per-solve limit; every other
caller's APBS input is byte-identical to before (checked against the famAx2 snapshot). Old cost test
34330658 cancelled; new test 34489354 (four candidates, stems 5 / 8 / 12 / 28 bp, 1 CPU, 40 GB, snapshot
`code/crcost2_20261123_2113`).

## 2026-11-24 04:00: blocked tests routed through the week partition

After 6 h in the queue, none of the cruciform cost test (34489354), the side-chain convergence test
(34492018), the pKa tests (34325002-4), the long-set/W packing (34433091) or the Nes table (34436498)
had started. Cause: the scheduler's backfill tries at most 100 jobs per user per partition
(`bf_max_job_user_part=100`, `bf_max_job_user=150`), in priority order, and our production arrays
(thousands of pending tasks) are a day older than these jobs, so the jobs were never tried; the start
estimates of the arrays (Nov 27-28) were the earliest they would have run. Niceing the arrays is
irreversible for a user (nice cannot be lowered again) and holding them resets their queue age
(`PriorityFlags` lacks ACCRUE_ALWAYS), so neither was used. Instead the blocked jobs were given the week
partition, where our 128-CPU cap was filled only by production tasks that end within 2-6 h, and the
pending production arrays (34311833, 34318664, 34318878, 34478525, and the new 34526398) lost week
until those jobs start; a partition change keeps a job's AccrueTime (checked on 34492018 and on each
array's pending record). A watcher restores day,week,scavenge to the arrays as soon as none of the
routed jobs is pending. Family A extras leftovers wave 2: 3,957 fragments in 198 tasks (34526398).

## 2026-11-24 07:10: Nes table complete; week partition to the DFT arrays next

*Nes.* 34436498 (16 tasks, 16-22 min each, 2.7 GB) wrote all 224 (modification, partner, context)
entries, 7 modifications x 2 partner strands x 16 contexts. All eight channel quantities are finite in
every entry; total charge is conserved to 3e-4 e. The one non-finite value is the diagnostic
`pqr_min_new_to_other_A` for pseudouridine, which adds no atoms (n_new = 0), so that distance is
undefined; no channel reads it. The medians follow the chemistry: inosine's O6 makes the major-groove
potential more negative (-0.63 kT/e at 1 A), m6A's methyl makes it more positive (+0.27), and the
2'-O-methyls change mainly the minor groove (+0.07 to +0.15).
*Week partition.* When the routed tests are running, week goes to the trimer (34298869, 57 tasks of
~4 h) and modified-RNA stack (34319769, 128 shards; ~7 h each from the main tier's ~1 h per context at
the same level) DFT arrays rather than back to the 1-CPU arrays. Their 8-CPU, 48-64 GB, 24 h tasks
almost never find room in day/scavenge backfill (start estimates Nov 27 and none), while 1-CPU tasks
do; with 16 tasks at a time under our 128-CPU week cap the two arrays need ~2-3 days, after which week
goes back to the 1-CPU arrays.

## 2026-11-24 09:15: week partition is for jobs over 24 h; pKa lmax 16 rerun without fast multipoles; cruciform cost

*Week.* The routed tests did not start when the week slots freed at 08:48. A test submission showed
the reason: the cluster's submit filter admits only jobs requesting more than 24 h to week ("Jobs of
24 hours or less should be submitted to the day partition"). Listing week beside day in the
production submissions had slipped past that filter, so our 14-16 h tasks had been running in week
against the site's rule. Week is now off every pending job of ours (production arrays and tests:
day,scavenge; AccrueTime kept). Jobs already running there finish. The plan to give week to the DFT
arrays is dropped. Our usable share is therefore day and scavenge only, about 128 CPUs fewer than the
measured rates that included week.
*pKa.* lmax 16 (34325004, eta 0.3, 1202 Lebedev points) failed for both contexts at the first SCF
iteration: "Jacobi solver did not converge ... ddCOSMO system". The run used psi4's fast multipoles
with local expansions to lmax 6 against multipoles to lmax 16. `run_psi4.py` gained `--ddx-no-fmm`
and `--ddx-maxiter`, and these settings are applied only when given, so every default run is
unchanged. On water at lmax 8, multipoles on and off give the same energy to 1e-8 Eh. Reruns (snapshot
`code/pkaddx4_20261124_0930`): lmax 16 with multipoles off and maxiter 300 (34547318), plus lmax 14
the same way (34547319), so that a difference between 14 and 16 cannot come from the multipole
switch. lmax 12 and 14 with multipoles on (34325002-3) are still running.
*Cruciform cost (v2).* Two of the four candidates are finished: stem 5 bp in 6,919 s and stem 12 bp
in 5,463 s, with 21.5-21.6 GB peak (sacct MaxRSS) for 2 models x 3 salts x (solvated + reference).
v1 had projected ~6.5 h and ~30 GB; the Newton-path reference removed the slow banded factorisation.
Stem 8 bp is running and stem 28 bp is queued.

## 2026-11-24 09:20: cruciform spread fix; cruciforms with stems >= 8 bp launched; ETA

*Spread channels.* In the third cost-test candidate (stem 8 bp), `M_cr_phi_junction_shell_sd_Na010`
came out 0.0. The shell potential averages solvent points within 6 A of the junction centre, and in
a compact junction one model has none, so its value is NaN. `np.nanstd` over [NaN, x] returns 0.0,
which would have labelled the model spread as zero when it is unknown. The spread now uses the models
with a value and is NaN with fewer than two (`structure_cruciform.py`). The same helper pattern sits
in the Family C ensemble spreads (`geometry._nanstd0`). A scan of all 1,024 main shards found no
exact zero among 148,174,332 finite spread values in 24 channels
(`results/spread_zero_scan_2026-11-24.json`), so no existing label is affected and finished code
is left as is.
*Cost (v2, 3 of 4).* Stem 8 bp took 8,156 s at 25.8 GB peak (MaxRSS), stem 5 bp 6,919 s and stem 12
bp 5,463 s: 1.9 h mean. 8,191 of the 12,559 candidates with stems >= 8 bp have 27-29 bp stems, so
the stem 28 bp case (task 3, queued) is the representative one.
*Launch.* The 12,559 candidates with stems >= 8 bp are needed under either scope the user is choosing
between, so they run now: 34547759, 4,187 tasks x 3 candidates, 12 h, 32 GB, day,scavenge, snapshot
`code/famMcr_20261124_1000` (made 09:18 despite its name; cost-test code plus the spread fix). Candidate lists are
`famM_cr/library_stem_ge8.tsv` and `library_stem_lt8.tsv` (31,048), from `cruciform_build.select`
on the production windows. All 43,607 windows match `famA_prod/windows.tsv`, and each sequence is
its window's [30:91]. The memory request of pending tasks can still be raised or lowered (tested),
and the time limit lowered, once the stem 28 bp cost is in. The cost test is older than this array,
so it should start first.
*ETA* (`results/eta_2026-11-24_0920.json`; our share without week, 607-995 slots; DFT and side-chain
production now included; +1 day packing). Stems >= 8 bp: ~Dec 10 (Dec 7-14), or Dec 9-18 if the
long stems cost twice as much. All 43,607: ~Dec 17 (Dec 12-24). The cheaper cruciforms cut the
difference between the two scopes from about three weeks to about one.

## 2026-11-24 09:50: long-set and W packed; Family A long-shard packer fixed before its first run

*Packed (34433091, 27 min).* Family A long set: 2,002 of 2,002 segments, every tile solved, 579
per-bp channels on the central 105 bp. Its 28 empty channels are the 150 mM ensemble mean and spread.
The long set exists for the window-level channels "alone" (PLAN.md), and the ensemble is drawn for
~10% of the main corpus, so no ensemble was planned for it. Family W: 21,391 fragments, none without a
solve, 214 shards. Content check: 59 tracks (length 10,240) and 51 scalars, none empty, no infinities; median
track 99.3% filled, the lowest the 5,000 bp window tracks (51%, defined only where a 5 kb window fits); scalars 100%.
*Family A long shards (finalize step `a`).* This step had never run. A one-shard test on the login
node refused 66 of 187 windows. `load_merged` joins a fragment's solve files (main, supplementary
10 mM pass, fix pass) by condition, and the per-phosphate `counterion` array differs in length
between passes: 240 phosphates for the 10 mM pass's 121 bp solute, 120 for the trimmed 61 bp solute
at higher salt. Concatenation therefore failed and the whole fragment was dropped. That would have
cost about a third of the Family A labels. No Family A channel reads `counterion`, and within one
file `run_fragment` already writes rows of another length as NaN. The merge now applies the same
rule to that array only; any other shape mismatch raises. After the fix: 187 of 187 windows, none
refused, 23 with an ensemble (12%; the plan's ~10%). 10/50/150/500 mM are present for every
fragment and mg150 for 23% (a subset condition).
Eight more shards across the corpus (128 ... 1023): 1,681 fragments, none refused, 174 (10.4%) with an
ensemble. Step `a` runs as 34549843 (32 tasks, snapshot `code/finalize_20261124_0950`, which has the
fix), and step `nes` as 34549844 (8 tasks) now that the table is complete. The final finalize run will
use a snapshot taken at that time.

## 2026-11-24 09:50: GPU node with a broken OpenCL platform; quadruplex build guarded

The quadruplex GPU build (34349846) started at 09:28. Its task 0 ran on a1112u01n01 (scavenge_gpu, RTX
PRO 6000 Blackwell, driver 580.178.04, compute mode Default, NVIDIA OpenCL ICD installed), where every
OpenMM OpenCL context failed with `clCreateContext (-9999)`. The batched build records a failed
minimisation as the model's status and then saved each fragment's record anyway, so 96 fragments
got records in which all 825 models had failed on the platform error. The resume rule skips any
fragment with a record, so those fragments would never have been rebuilt, and the solve stage
would have labelled them not modelled. Task 0 was cancelled after 18 min, and all 96 records were
deleted. They were the only files in `famM3/out/built`, and each was checked to have only
platform-error statuses. Task 1 then landed on the same node and was cancelled after 29 s without
writing a record. Hydration MD tasks 8-12 (34298686) had also run on this node: each exited 0 in 30 s
with every run failed on the same error. That runner writes nothing for a failed run, so no bad
output exists. OpenCL has worked for our jobs on A40, A5000, H200, RTX 5000 Ada and B200 nodes. The
RTX PRO 6000 Blackwell nodes (8) are now excluded from both GPU arrays' pending tasks
(`ExcNodeList`; AccrueTime kept). Build tasks 0-1 are requeued with the exclusion. Hydration shards
8-12 had left the controller and could not be requeued, so they were resubmitted with the original
script and snapshot (34550015, `code/mdhyd_20261122_1937`, N_SHARDS=34).
Guard (`run_family_m3.py`, live and patched in place in the build's snapshot
`code/famM3b_20261123_0513`, manifest updated and annotated): (1) a build on a GPU platform first
creates a one-particle context and exits 2 before touching any fragment if that fails; (2) the
batched build refuses to save a fragment whose models carry a platform error ("Error initializing
context") and stops the task. Working runs are unaffected. Login-node check: exit 2, no files.

## 2026-11-24 11:17: Family J v2 tasks time out too; leftover waves for J and A extras

Family J v2 (34318878, 150 fragments per task, 16 h): of the 18 tasks finished so far, 4 completed
(10.7-15.7 h) and 14 timed out with 113-139 of their 150 fragments solved (447 missing). The runner
writes one npz per fragment and skips existing ones, so a timeout loses only the fragment in progress;
the rest is a remainder. The same leftovers rule as Family A extras now applies to J
(`scripts/famAx_leftovers.py` is generic: sources TSV, solves folder, new library;
`famJox/sources.tsv` lists 34288949 over `all.tsv` at 20 rows (no live tasks) and 34318878 over
`remaining_v2.tsv` at 150 rows). Wave 1: `famJox/library_left_w1.tsv`, 447 fragments, 60 per task,
12 h (the slowest observed rate, 505 s per fragment, gives 8.4 h), same snapshot
`code/famJox2_20261122_1758` and output `solves_v2`: 34558355 (8 tasks, day,scavenge).
Family A extras: 512 more tasks reached the 14 h limit since 04:00 (none of the 40-row tasks
completes, as expected from 2,025 s per fragment). Wave 3: `library_left_w3.tsv`, 8,576 fragments,
20 per task: 34558351 (429 tasks, day,scavenge, snapshot `code/famAx2_20261122_2154`). Waves are
submitted as soon as a remainder exists, since a task's queue age counts from its submission.

**11:25: Family A extras memory request lowered to what the solves use.** The day QoS caps a user at
1,500 CPUs and 20,000 GB; with 1,235 extras tasks at 16 GB running, our day allocation stood at 1,272
CPUs and 19,584 GB, so memory, not CPUs, limited how many tasks could run. The 150 mM extras solves
(61 bp solute) peak at 3.7 GB (median) and 4.51 GB (max) over 855 finished tasks (sacct MaxRSS), the
same size as the Family A main solves that run at 7 GB. The pending tasks of all four extras arrays
(34311833, 34478525, 34526398, 34558351; 3,257 tasks) now request 8 GB (`scontrol update
MinMemoryNode=8192`; AccrueTime unchanged, checked). Nothing computed changes. Other arrays already
request near their measured peak (K 8 GB vs 4.53; J 2 GB vs 0.13; triplex 16 GB vs 7.59).

**11:20: ETA refreshed (`results/eta_2026-11-24_1120.json`; `scripts/eta_labels.py` now computes it,
CPU partitions without week).** Our CPU allocation rose to 966 CPUs over the last 6 h (590 over 24 h).
Family J is costed at its measured 434 s per fragment (was 294.5 s) and A extras at 8 GB. Training
start: cruciforms >= 8 bp ~Dec 7 (Dec 5-10 over the 6-24 h rates; Dec 7-13 if the 27-29 bp stems cost
twice as much); all 43,607 ~Dec 13 (Dec 10-18). pKa production is not yet included.

## 2026-11-24 14:01: the two open scope questions, settled against the plan (user: "do what you think is best but follow plan")

*Cruciforms: all 43,607 candidates.* PLAN.md Family M treats cruciforms as explicit conformations and
sets no minimum stem; the project's own candidate rule (`noncanonical.find_inverted_repeats`: arm >= 6,
spacer <= 10; `cruciform_build.select`: stem >= 4 after the loop takes stem pairs, flanks >= 3) gives
43,607 windows. The >= 8 bp split was mine, for cost, so the remaining 31,048 candidates
(`famM_cr/library_stem_lt8.tsv`) now run exactly as 34547759 does (snapshot
`code/famMcr_20261124_1000`, 3 per task, 12 h, 32 GB, day,scavenge): 34567521 (tasks 0-5174) and
34567522 (TASK_OFFSET 5175), two arrays because the scheduler's max_array_tasks is 10,000 (a single
10,350-task submission was refused and nothing was queued).
*The 69 provisional channels: excluded from training, as amendment A1 specifies.* A1 (which governs
the body): "Channels without a published parameterisation stay provisional, are excluded from
training and headline numbers, and are listed with their reasons." The literature passes above found
no published parameterisation for any of the 69 (`results/provisional_channels.json` lists each
with its reason), so the final ladder trains without them (the default; `--allow-provisional` is not
used), and the effective channel count is reported net of them. The experimental targets that the
plan's stage 2 names for several of these quantities (loop-seq, in vitro nucleosome occupancy,
quadruplex sequencing, photoproduct maps) are delta-learning targets, not stage-1 analytic channels.
*pKa, lmax 12 vs 14 (34325002-3, eta 0.3, multipoles on; `tmp/pkaddx3`).* The eight shifts move by
0.005-0.115 pKa units between lmax 12 and 14 (median ~0.02; largest AAAA A-N7 of bp 2, 0.229 -> 0.344;
GCGC G-N3 1.796/1.763 and 1.704/1.708). Whether that is converged is decided by lmax 16 (34547318,
multipoles off) against lmax 14 the same way (34547319); both are queued behind the older production
tasks (our jobs start in submission order), so pKa production waits for them.
*ETA with all cruciforms (`results/eta_2026-11-24_1403.json`).* 282,606 CPU-h remain (403,177
slot-h; cruciforms request 32 GB). Over the last 6/12/24/48 h rates: Dec 6 / Dec 10 / Dec 15 / Dec 12;
~Dec 12 is the middle. pKa production not included.

## 2026-11-24 17:33: quadruplex GPU build leaks OpenCL contexts; build throttled to 1 task, fix being tested

Build task 2 (34349846, a1125u31n01, A40) ran 51 min and saved nothing. Each of its three workers built its
first block of fragments on the CPU side, then failed every context at the batched minimisation with
`clCreateContext (-6)`. The guard therefore saved no record, and the task exited 0, so Slurm lists it
COMPLETED. Tasks 3 and 4 (a1125u34n01, A40), inspected live with `srun --overlap`:
- Per process, threads rose from ~80 to ~190 and open `/dev/nvidia*` descriptors to 510-590 within 8 min.
  GPU memory rose ~1 GB/min per process (27 of 46 GB in use at 17:29).
- The job cgroup was at 3.6 of 32 GB with no memory events.
- A new process's first OpenCL context failed with -6.
So OpenCL contexts are being leaked, and the GPU refuses new contexts long before its memory is full.
Every model creates contexts on the default platform, which on a GPU node is OpenCL:
- PDBFixer.addMissingAtoms and Modeller.addHydrogens, each through PDBFixer with no platform set;
- the loop anneal in `loop_builder.fill_and_relax`;
- PDBFixer again, per model, inside `minimise_batch`.
The 8-fragment RTX 5000 Ada validation (34311852) stayed under the limit. The hydration MD, which makes
few contexts per process, is unaffected.
Actions:
- ArrayTaskThrottle 2, then 1 (AccrueTime kept); task 4 cancelled; task 3 is left to end.
- Rows of tasks 2-4 go into a leftovers wave once the fix runs.
- Diagnostic 34584606 tests on a clean GPU: function-scope versus explicit `del` + `gc.collect()`;
  PDBFixer on the default versus the CPU platform; whether a child process's exit returns its contexts;
  and how many contexts one process can create.

## 2026-11-24 18:12: cause of the quadruplex build failure found and fixed (PDBFixer on the CPU platform)

Diagnostic 34584606 (RTX A5000, driver 580.178.04), measured per process:
- **20 OpenMM contexts made explicitly on OpenCL:** no new `/dev/nvidia*` descriptors and no GPU memory
  kept, whether freed at function scope or with `del` + `gc.collect()`. Each leaves one idle thread;
  400 more in one process did not fail.
- **10 PDBFixer runs with no platform set** (its added-atom and hydrogen minimisations): +147 NVIDIA
  descriptors and 2,040 MiB of GPU memory, never returned. Left unset, OpenMM picks the fastest platform,
  which is CUDA (speed 100) on a GPU node.
- **10 PDBFixer runs on the CPU platform:** no change.
- **A child process making 20 contexts:** all returned at its exit.

So every model's PDBFixer call leaked a CUDA context (~200 MB, ~15 descriptors). The loop builder calls it
once per gapped model and `minimise_batch` once per model. Within one block the GPU stopped accepting
contexts, and the OpenCL minimisation then failed with -6.
Fix:
- `loop_builder.fill_and_relax`, `g4_build.minimise_model` and `g4_build.minimise_batch` now build
  PDBFixer with the CPU platform (0.1 s for a 715-atom template).
- Patched live and in the build snapshot `code/famM3b_20261123_0513`; manifest updated and annotated.
- The anneal, batched minimisation and batched MD stay on OpenCL, as validated.
- PDBFixer's own minimisations only place added atoms and hydrogens, which the model minimisation then
  relaxes. Moving them from CUDA to CPU changes rounding, not the protocol.
- The other PDBFixer users run on CPU nodes, or make a few calls per process on a GPU (hydration MD);
  they are unchanged.
Throttle stays at 1 until a patched task has been watched for flat GPU memory and descriptors and
saved records. Rows of tasks 2-4 (consumed, nothing saved) go into a leftovers wave.
Check after the patch: `tests/test_structure_folds.py` 6 passed (610 s, CPU), including the one-model PU22 build through the patched PDBFixer path (18:22).

## 2026-11-24 20:05: heartbeat; leftover waves Ax w4 and J w2

Family A extras: 2,522 fragments that no live task covers (79 tasks hit the 14 h limit since 14:00):
`library_left_w4.tsv`, 20 per task, 8 GB (no memory kills since the 11:25 change), 34597879 (127 tasks,
day,scavenge, `code/famAx2_20261122_2154`). Family J v2: 425 fragments, `library_left_w2.tsv`, 60 per task,
12 h, 34597880 (8 tasks, `code/famJox2_20261122_1758`, output `solves_v2`). Both appended to their
`sources.tsv`. No failed, node-failed or memory-killed pangea task since 14:00. The quadruplex build's first
patched task is waiting for a GPU (Priority).

## 2026-11-24 22:28 — all PANGEA jobs stopped on the user's order
The user said "stop jobs". Cancelled every pangea-* job this project submitted (24 job IDs, 772 queued/running
tasks: famAx incl. waves w1–w4, famJox incl. w1–w2, famK, famMf triplex/i-motif, famMcr cruciform, famMbuild,
trimer DFT, modRNA DFT, mdhyd, packA, packNes, and the pKa/cruciform-cost/Isc probes). The user's own jobs,
including the held pangea-conv, were left alone. Both background watchers stopped. Saved per-row outputs stay on
disk; the rows of the tasks killed mid-run have no output and will be picked up by the leftovers scripts on restart.
The quadruplex PDBFixer-on-CPU fix was not yet verified on a GPU task.
