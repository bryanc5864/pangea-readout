# RESULTS

Data-stage results. Every number here traces to a file under `results/` or to a
provenance record under `data/scratch/`. Nothing is typed by hand.

Status (2026-11-19, night): sections 1-10 are ground truth and validation; sections
11-14 are the first encoder results (on the v1 targets) and the first external validation
against measurement. The v2 targets are being regenerated and the scaling ladder, the
bottleneck sweep and the universality experiments run on them next.

## 1. Species panel

Source: `data/scratch/genomes/genomes_manifest.json`, `.../fa/panel_prepared.json`.

Fifteen RefSeq reference assemblies, each verified against the assembly's own md5.
3.04 GB compressed, 9.6 GB indexed.

| Measured GC | Clade | Species | Accession |
| --- | --- | --- | --- |
| 19.3% | protist | Plasmodium falciparum | GCF_000002765.6 |
| 22.4% | protist | Dictyostelium discoideum | GCF_000004695.1 |
| 31.3% | archaeon | Methanocaldococcus jannaschii | GCF_000091665.1 |
| 35.4% | animal | Caenorhabditis elegans | GCF_000002985.6 |
| 35.9% | plant | Arabidopsis thaliana | GCF_000001735.4 |
| 37.0% | animal | Danio rerio | GCF_052040795.2 |
| 38.2% | fungus | Saccharomyces cerevisiae | GCF_000146045.2 |
| 40.9% | animal | Homo sapiens | GCF_000001405.40 |
| 42.0% | animal | Drosophila melanogaster | GCF_000001215.4 |
| 41.7% | animal | Mus musculus | GCF_000001635.27 |
| 46.9% | plant | Zea mays | GCF_902167145.1 |
| 50.5% | bacterium | Escherichia coli | GCF_000008865.2 |
| 65.6% | bacterium | Mycobacterium tuberculosis | GCF_000195955.2 |
| 69.5% | bacterium | Thermus thermophilus | GCF_000091545.1 |
| 72.0% | bacterium | Streptomyces coelicolor | GCF_000203835.1 |

**The plan's claimed span is confirmed.** Measured composition runs 19.3 to 72.0 percent
GC across bacteria, archaea, fungi, plants, animals and protists, so the panel does
bracket the stated range and the universality certificate has the composition spread it
needs.

## 2. Fragment library

Source: `data/scratch/fragments/library_v1_meta.json`.

139,381 fragments of 61 bp, seed 20261119.

| Stratum | Count | Purpose |
| --- | --- | --- |
| panel | 119,949 | universality certificate and held-out-kingdom test |
| hexamer | 12,288 | all 4,096 hexamers in three fixed flanking contexts |
| random | 3,500 | off-manifold, seven target compositions |
| extremes | 3,000 | composition tails, 5 to 15 and 85 to 95 percent GC |
| compctrl | 644 | the composition gate |

Composition controls verified exact: 50 groups, 49 at full size 13 and one at 7 where
the dinucleotide shuffle could not find six distinct rearrangements. Across all 644
fragments there are **zero mononucleotide composition mismatches** within groups and
**zero dinucleotide mismatches** among seed and dinucleotide-shuffled members. The
Altschul-Erikson shuffle was separately verified exact on 200 of 200 random trials.

## 3. Quantum chemistry, tier 1

Source: `results/qc_tetramer_gfn2.json`. GFN2-xTB single points on capped base-pair
dimers built from cgNA+ conformations. 136 reverse-complement-canonical tetramer
contexts, **136 of 136 converged**, 864 s wall on 16 concurrent workers.

Stacking energy, defined as E(dimer) - E(bp1) - E(bp2):
mean **-13.93**, sd 2.09, range **-20.17 to -10.31 kcal/mol**.

Per central step, ordered by mean stacking energy:

| Step | n contexts | Mean | Spread across flanking contexts |
| --- | --- | --- | --- |
| CG | 10 | -19.62 | 0.92 |
| AG | 10 | -15.29 | 0.91 |
| CT | 6 | -15.28 | 0.94 |
| CA | 10 | -14.81 | 1.33 |
| TG | 6 | -14.46 | 0.67 |
| GC | 10 | -14.28 | 0.87 |
| AC | 10 | -13.98 | 1.17 |
| AA | 10 | -13.96 | 0.73 |
| GT | 6 | -13.88 | 1.10 |
| TT | 6 | -13.83 | 0.54 |
| TA | 10 | -13.65 | 0.85 |
| AT | 10 | -12.81 | 0.60 |
| CC | 10 | -11.55 | 2.01 |
| GG | 6 | -11.51 | 1.17 |
| GA | 10 | -11.37 | 1.14 |
| TC | 6 | -11.21 | 0.70 |

Three things worth stating.

**Context dependence is real and sizeable.** The median spread of a single base-pair
step across its flanking contexts is **0.91 kcal/mol**, and CC reaches 2.01. This is the
quantity a dinucleotide lookup table cannot represent, and it is the reason the plan
computes these on cgNA+ conformations rather than on one idealised step.

**Reverse-complement consistency holds.** A step and its complement are the same
physical object and must agree. Sampled with *different* flanking contexts they do:
AG -15.29 against CT -15.28, AC -13.98 against GT -13.88, AA -13.96 against TT -13.83,
CC -11.55 against GG -11.51, GA -11.37 against TC -11.21. The largest discrepancy is
CA -14.81 against TG -14.46, 0.35 kcal/mol, which is within the context spread. This is
an independent check that the geometry construction and the canonicalisation are right.

**GFN2 overbinds relative to the literature, as expected.** The plan cites published
DFT base-pair-step stacking energies spanning roughly -9.9 to -14.7 kcal/mol. The GFN2
range extends to -20.17, mostly from CG steps. Semi-empirical tight binding is known to
overbind stacked aromatics, and this is exactly why the plan specifies a dispersion
corrected DFT tier that supervises a correction to the cheap tier rather than replacing
it. The tier-1 numbers are therefore usable for ranking and for delta-learning, and are
**not** reported as absolute stacking energies.

Hydrogen-bond energy, E(bp) - E(watson base) - E(crick base): range **-29.34 to -14.74**,
mean -21.92 kcal/mol. The spread corresponds to GC pairs at the strong end and AT pairs
at the weak end, which matches the published gas-phase ordering.

Dimer vertical ionisation potential: **6.510 to 7.010 eV** across the 136 contexts, a
0.5 eV sequence-driven range. This is the quantity that sets where a hole localises, so
it feeds the oxidative-damage channels.

These are total interaction energies, not the electrostatic / exchange-repulsion /
induction / dispersion decomposition the plan calls for. That decomposition requires the
symmetry-adapted perturbation theory tier, which is not implemented.

## 4. Experimental assays ingested

Source: `data/scratch/processed/ingest_report.json`.

**loop-seq cyclizability** (Family D). All four libraries ingested: random 12,472,
tiling 82,368, ChrV 82,404, nucleosomal 19,907, total **197,151 rows, 196,635 unique
50-mers**. The documented construct is 25 bp adapter + 50 bp variable + 25 bp adapter;
the **adapter match fraction is 1.000 for all four libraries**, which confirms the
construct layout rather than assuming it. C0 ranges -3.160 to 2.481, mean -0.170.

**Array Melt thermodynamics** (Family E). 27,732 hairpins, lengths 11 to 24 nt, series
MisMatches 10,725 / Bulges 9,097 / TETRAloop 4,691 / WatsonCrick 2,027 / TRIloop 1,168 /
External 19 / VARloop 5. Both thermodynamic identities were tested row by row:
dG37 = dH - 310.15 dS holds on **100.0 percent** of rows and Tm = dH/dS - 273.15 holds on
**100.0 percent**. The release contains a different file whose entropy column is computed
with Tm in Celsius; this check is what distinguishes the two, and the file in use is
clean.

**ORChID cleavage tables** (Family G). Extracted from the server source: **exactly 256
tetramers and 64 trimers**, with a mean and standard deviation at each position. This
independently confirms the correction recorded in DECISIONS.md, that ORChID is a
tetramer model with a trimer fallback and not the 512-pentamer table the plan describes.

## 5. Code validation

`python -m pytest tests/ -q`: **39 tests, all passing**.

Structural validation of the cgNA+ to all-atom reconstruction, on a 33 bp duplex:

| Quantity | Measured | Canonical B-DNA |
| --- | --- | --- |
| C1'-C1' across pairs | 10.685 +/- 0.070 A | about 10.5 A |
| Glycosidic N-N across pairs | 9.009 +/- 0.026 A | about 8.9 A |
| P-O5' bond | 1.611 A | 1.59 to 1.61 A |
| O3'-P bond | 1.610 A | 1.59 to 1.61 A |
| C1'-N glycosidic bond | 1.508 +/- 0.018 A | 1.47 to 1.48 A |
| Closest inter-residue contact | 1.610 A | the O3'-P bond itself |

Restrained minimisation holds the restrained atoms to **0.265 A RMSD** with a maximum
shift of 0.92 A, so the sequence-dependent conformation survives the backbone cleanup
rather than relaxing toward a canonical duplex.

Groove convention confirmed rather than assumed: **91.8 percent** of phosphates project
onto the negative x axis of the base-pair frame, mean projection -0.301, so +x is the
major groove and an angular sector index means what the channel definition says.

Nearest-neighbour thermodynamics: all **ten published SantaLucia unified step free
energies reproduced to within 0.025 kcal/mol** from the stored enthalpy and entropy
tables, and reverse-complement symmetry is exact to 1e-9 on 50 random sequences.

## 6. Electrostatics

Not yet produced at scale. The pipeline runs end to end and two blocking defects were
found and fixed:

- PDB2PQR's AMBER parameter set assigns **radius 0.000** to the terminal hydroxyl
  hydrogens H5T and H3T. A zero-radius charge 0.96 A from O5' makes the potential
  singular there.
- With only about 20 A of coarse-grid padding the Debye-Huckel boundary condition on a
  -120 e duplex is poor, and the nonlinear solver traps millions of exponential
  overflows and aborts with "Detected large potential values".

Total solute charge is **-120.000 e** for a 61 bp duplex, exactly the 120 internal
phosphates, which confirms charge assignment.

A cost-and-accuracy probe over solute length, fine grid spacing, coarse padding and
equation is running; the production configuration will be chosen from it and any
reduction in scope relative to the plan logged as a deviation.

## 7. Yeast coordinate calibration, and a measurement that bears on the positive control

Source: `results/yeast_coord_calibration.json`.

The in vitro nucleosome occupancy map carries coordinates on a 2006 SGD assembly and no
sequence. Offsets against sacCer3 were determined by scanning trial offsets and
measuring the correlation between 147 bp windowed GC and occupancy, which peaks at the
correct alignment.

| Chromosome | n sampled | r at offset 0 | best offset | r at best |
| --- | --- | --- | --- | --- |
| I | 20,000 | 0.7676 | +0 | 0.7676 |
| VI | 20,000 | 0.7778 | +10 | 0.7807 |
| III | 20,000 | 0.7924 | +5 | 0.7924 |
| IX | 20,000 | 0.7821 | +0 | 0.7821 |

The offset curve is smooth and single-peaked; on chromosome I the correlation falls from
0.7676 at offset 0 to 0.648 at plus or minus 60 bp. Best offsets are 0 to 10 bp, and the
cost of using offset 0 everywhere is 0.0014 in correlation. **No coordinate lift is
needed.**

**GC content in a 147 bp window reaches r = 0.78, so R-squared about 0.61, against in
vitro nucleosome occupancy.** The plan assumes "close to half the variance" and uses this
assay as the positive control. At 61 percent from one scalar feature, the control as
specified would be passed by anything that can see local composition, which is every
channel in the landscape. The control therefore has to be scored against the GC baseline
rather than against the sequence model, and the informative quantity is how much of the
remaining 39 percent physics reaches. See DECISIONS.md for the caveat on comparability
with published figures.

## 8. Genome builds of the coordinate-only assays, determined from the data

Three human assays distribute coordinates without sequence, so each needs its build
established before it can supervise a channel. None was assumed.

| Assay | Accession | Build | How it was determined |
| --- | --- | --- | --- |
| G4-seq | GSE110582, GSE63874 | hg19 | maximum chr1 coordinate exceeds hg38 chr1 |
| S1-END-seq | GSE203632 | hg19 | bigWig chromosome B-tree: chr1 249,250,621, chr2 243,199,373, chrX 155,270,560, 25 contigs |
| KAS-seq | GSE139420 | hg19 | peak set reaches chr1:249,240,722, which fits hg19 (249,250,621) and overflows hg38 (248,956,422) |

All three are hg19, so one genome serves all of them and no liftOver is needed among
them. The yeast assays are handled separately (section 7): sacCer3 coordinates are used
directly, since the measured offset is 0 to 10 bp and costs 0.0014 in correlation.

S1-END-seq was found by reading the GEO file list rather than by title: the series is
titled "END-seq", not "S1-END-seq", which is why a title search misses it. Its files are
named `S1_END_seq_*` explicitly.

## 9. Leakage-controlled splitting

Source: `tests/test_splits.py`.

Splits are made by similarity group, never by row. Sequences are shingled into 11-mers,
MinHash signatures are built, near-duplicate candidates are found by locality-sensitive
hashing and confirmed against the full signature, then whole groups are assigned to
folds and the result is audited for the largest similarity that still crosses a fold
boundary.

Validated on a construction where the answer is known: 60 base sequences of 200 bp, each
with two planted near-duplicates differing by three substitutions.

| Check | Result |
| --- | --- |
| Planted groups recovered intact | 150 of 150 (in the 150-group run) |
| Predicted groups merging unrelated sequences | 0 |
| Planted groups straddling a fold boundary | 0 |
| Maximum similarity crossing a fold boundary after grouping | 0.0156 |
| Planted groups straddling a boundary under a **random** split | majority |

The last row is the control, and it is kept as a test: a random split of the same data
does put near-duplicates on both sides, which is the leakage this machinery prevents and
the reason the extra step is justified.

Test suite: **48 tests, all passing.**

## 10. Solver numerical noise floor

Source: `results/jitter_noise_floor.npz`, from 8 grid-jitter replicates of a single
composition-control sequence at 150 mM NaCl. Each replicate displaces the fine grid
centre by up to half a grid spacing (0.175 A), so the spread across replicates is the
solver's own numerical reproducibility and nothing else.

| Channel set | Median per-channel SD | As a fraction of mean channel magnitude |
| --- | --- | --- |
| Surface-referenced potential | 0.0145 kT/e | 0.67 percent |
| Axis-referenced potential | 0.0090 kT/e | 0.44 percent |
| Surface radius | 0.00000 A | exactly reproducible |

The surface radius is exactly invariant because the marching construction depends only
on atom positions, not on the grid, which is a useful internal check that the two
sampling schemes are independent in the way intended.

**Signal against noise.** The standard deviation of a surface-referenced channel across
the 21 scored base pairs of the same fragment is 0.1379 kT/e, which is **9.5 times** the
jitter noise. Position-to-position structure within a fragment is therefore well
resolved, and the composition gate has a real floor to measure against rather than one
comparable to its signal.

All 8 replicates converged; wall time 103 to 194 s each, inflated by contention with the
quantum chemistry running on the same node.

## 11. The stage-1 encoder: first ablations (v1 targets)

Source: `results/ablations_v1targets.json`, written by `src/eval/compare_runs.py` from
each run's own `history.jsonl`. Validation side of the similarity split; the row is the
epoch each run kept as `best_ema.pt`. The v1 targets carry the backbone-trap artifact and
200-sample ensemble spreads (DECISIONS.md), so these numbers compare designs; the
headline numbers come from the v3 runs on the v2 targets, on a held-out test side.
(The `v2_` prefix of these run names is the second training pipeline, not the targets.)

| run | params | epochs | per-bp median R2 | per-bp > 0.99 | global median R2 | global > 0.99 |
| --- | --- | --- | --- | --- | --- | --- |
| v2_S_base | 2.03M | 100/100 | 0.9883 | 0.439 | 0.9803 | 0.447 |
| v2_S_beta0 | 2.03M | 100/100 | 0.9733 | 0.317 | 0.9340 | 0.300 |
| v2_S_random | 2.03M | 100/100 | 0.9883 | 0.444 | 0.9810 | 0.451 |
| v2_S_r256 | 2.63M | 100/100 | 0.9952 | 0.693 | 0.9854 | 0.478 |
| v2_S_gmlp | 2.59M | 100/100 | 0.9891 | 0.468 | 0.9875 | 0.474 |
| v2_M | 10.11M | 100/100 | 0.9974 | 0.727 | 0.9901 | 0.502 |
| v2_M_gmlp | 12.33M | 50/100 | 0.9970 | 0.712 | 0.9921 | 0.522 |
| v2_M_attn | 14.82M | 100/100 | 0.9984 | 0.732 | 0.9919 | 0.518 |
| v2_M_attn_pair | 14.86M | 74/100 | 0.9982 | 0.732 | 0.9948 | 0.625 |
| v2_M_attn_pair_gmlp | 17.08M | 51/100 | 0.9979 | 0.722 | 0.9947 | 0.640 |
| v2_L | 49.1M | 59/100 | 0.9988 | 0.741 | 0.9916 | 0.502 |

Runs marked short of 100 epochs were stopped by the home-quota event (DECISIONS.md); their
rows are the best epoch reached.

## 12. The sampling-noise ceiling (v1 targets)

Source: `results/noise_ceiling_v1.json`, `src/eval/noise_ceiling.py`: the landscape of
300 fragments recomputed under two seeds. Ceiling = 1 - noise variance / corpus variance.

| channel | ceiling R2 |
| --- | --- |
| C_minor_width_A_sd | 0.400 |
| C_major_width_A_sd | 0.562 |
| C_phosphate_radius_A_W_sd | 0.811 |
| C_phosphate_radius_A_C_sd | 0.826 |
| C_major_depth_A_sd | 0.850 |
| C_phosphate_azimuth_deg_W_sd | 0.854 |
| C_phosphate_axial_offset_A_W_sd | 0.856 |
| C_phosphate_axial_offset_A_C_sd | 0.858 |
| C_phosphate_azimuth_deg_C_sd | 0.864 |
| C_axis_bend_direction_sd | 0.927 |
| C_axis_curvature_sd | 0.929 |
| C_minor_depth_A_sd | 0.945 |
| C_mean_minor_width_sd_A | 0.761 |
| C_mean_major_width_sd_A | 0.797 |

## 13. Effective rank of the channel stack (v1 targets)

Source: `results/effective_rank_v1.json`, `src/eval/effective_rank.py`.

| resolution | channels | samples | participation ratio | components, 95% | components, 99% |
| --- | --- | --- | --- | --- | --- |
| per_bp | 203 | 341016 | 10.8 | 41 | 69 |
| global | 238 | 191344 | 5.6 | 41 | 70 |

## 14. External validation against measurement

### Array Melt (Ke et al. 2025): Family E's DNA fold, 27732 measured hairpins

Source: `results/validation_array_melt.json`. One fitted constant offset; nothing else.

| sodium | Pearson r | Spearman | MAE | MAE after offset | RMSE after offset |
| --- | --- | --- | --- | --- | --- |
| 1M | 0.684 | 0.696 | 2.02 | 0.71 | 0.94 |
| 0.075M | 0.638 | 0.642 | 0.98 | 0.74 | 0.99 |

By construct class at 1 M: Bulges r = 0.523, MisMatches r = 0.660, TETRAloop r = 0.558, TRIloop r = 0.683, WatsonCrick r = 0.893.

### ORChID hydroxyl-radical cleavage, 256 tetramers

Sources: `results/validation_orchid_v1.json` (v1 groove channels, 64 shards) and
`results/validation_orchid_v2_preview.json` (Family G, the first 11 v2 shards). Primary
metric fixed in advance: Pearson r over all aligned positions.

| channel | pooled r | r at positions 1-4 |
| --- | --- | --- |
| C_minor_width_A | +0.012 | +0.03, +0.10, +0.02, -0.38 |
| C_minor_depth_A | -0.007 | +0.01, -0.05, +0.05, +0.02 |
| C_major_width_A | -0.023 | -0.05, -0.09, -0.03, +0.35 |
| C_minor_width_A_sd | +0.113 | +0.12, +0.31, -0.07, -0.21 |
| G_probe_hydroxyl_radical_W | +0.089 | -0.25, +0.35, +0.23, +0.16 |
| G_probe_hydroxyl_radical_C | +0.109 | -0.23, +0.39, +0.23, +0.15 |
| G_radical_target_W | +0.089 | -0.25, +0.35, +0.23, +0.16 |
| G_radical_target_C | +0.109 | -0.23, +0.39, +0.23, +0.15 |
| G_sasa_sugar_W | +0.124 | -0.18, +0.35, +0.29, +0.04 |
| G_sasa_sugar_C | +0.128 | -0.18, +0.36, +0.29, +0.04 |
| C_minor_width_A | +0.010 | +0.03, +0.09, +0.02, -0.38 |

### Loop-seq (Basu et al. 2021): Family D's cyclizability prior, 4000 sequences

Source: `results/validation_loopseq.json`; random library, deterministic subsample,
nothing fitted.

| channel | Pearson r | Spearman |
| --- | --- | --- |
| D_cyclizability_L63 | +0.170 | +0.165 |
| D_cyclizability_L73 | +0.170 | +0.165 |
| D_cyclizability_L84 | +0.170 | +0.165 |
| D_cyclizability_L94 | +0.170 | +0.165 |
| D_persistence_length_nm | -0.022 | -0.044 |
| D_net_bend_deg | +0.190 | +0.169 |
| D_curvature_shortening | +0.187 | +0.177 |
| D_mean_bend_variance_rad2 | +0.020 | +0.040 |

### Effective rank on the v2 targets (all eight families)

Source: `results/effective_rank_v2.json`.

| resolution | channels | samples | participation ratio | components, 95% | components, 99% |
| --- | --- | --- | --- | --- | --- |
| per_bp | 265 | 341016 | 11.9 | 42 | 72 |
| global | 306 | 189622 | 5.3 | 49 | 85 |

By family (per-bp participation ratio): B 2.1, C 12.5, E 1.8, F 1.8, G 6.1, H 4.4, L 3.5.

### The sampling-noise ceiling on the v2 targets (3200 samples)

Source: `results/noise_ceiling_v2.json` against `results/noise_ceiling_v1.json`.

| channel | ceiling, v1 (200 samples) | ceiling, v2 (3200 samples) |
| --- | --- | --- |
| C_minor_width_A_sd | 0.400 | 0.915 |
| C_major_width_A_sd | 0.562 | 0.954 |
| C_phosphate_radius_A_W_sd | 0.811 | 0.986 |
| C_phosphate_radius_A_C_sd | 0.826 | 0.987 |
| C_major_depth_A_sd | 0.850 | 0.989 |
| C_phosphate_axial_offset_A_W_sd | 0.856 | 0.990 |
| C_phosphate_axial_offset_A_C_sd | 0.858 | 0.990 |
| C_phosphate_azimuth_deg_W_sd | 0.854 | 0.990 |
| C_phosphate_azimuth_deg_C_sd | 0.864 | 0.990 |
| C_axis_bend_direction_sd | 0.927 | 0.993 |
| C_axis_curvature_sd | 0.929 | 0.995 |
| C_minor_depth_A_sd | 0.945 | 0.996 |
| C_mean_minor_width_sd_A | 0.761 | 0.985 |
| C_mean_major_width_sd_A | 0.797 | 0.985 |

## 15. The bottleneck sweep (v2 targets, held-out test side)

Source: `results/rank_sweep_v2.json`; the 10M trunk with head rank 4 to full (272).
Saturating rank, criterion fixed in advance (within 0.001 of full rank): per-bp 256, global 128.

| head rank | per-bp median R2 | global median R2 | per-bp > 0.99 | global > 0.99 |
| --- | --- | --- | --- | --- |
| 4 | 0.2680 | 0.6887 | 0.059 | 0.028 |
| 8 | 0.6399 | 0.8782 | 0.091 | 0.118 |
| 16 | 0.8245 | 0.9505 | 0.147 | 0.235 |
| 32 | 0.9417 | 0.9854 | 0.231 | 0.396 |
| 64 | 0.9846 | 0.9918 | 0.406 | 0.554 |
| 128 | 0.9962 | 0.9944 | 0.685 | 0.616 |
| 256 | 0.9976 | 0.9951 | 0.692 | 0.625 |
| 272 | 0.9977 | 0.9952 | 0.696 | 0.625 |

## 16. Synthetic-composition stress test (per-bp / global median R2 within each test stratum)

Source: `runs_v3/<run>/eval_by_group.json` (`src/eval/evaluate_run.py`).

| run | panel | MPRA | extremes | random | hexamer libraries |
| --- | --- | --- | --- | --- | --- |
| v3_L2M | 0.9933 / 0.985 | 0.9932 / 0.977 | 0.9920 / 0.992 | 0.9929 / 0.989 | 0.9969 / 1.000 |
| v3_L10M | 0.9976 / 0.991 | 0.9975 / 0.989 | 0.9974 / 0.995 | 0.9974 / 0.994 | 0.9992 / 1.000 |
| v3_L50M | 0.9987 / 0.992 | 0.9987 / 0.990 | 0.9985 / 0.996 | 0.9986 / 0.994 | 0.9997 / 1.000 |
| v3_L200M | 0.9992 / 0.993 | 0.9992 / 0.991 | 0.9991 / 0.996 | 0.9991 / 0.994 | 0.9998 / 1.000 |
| v3_Ukingdom_2M | 0.9897 / 0.957 | 0.9896 / 0.960 | 0.9882 / 0.983 | 0.9894 / 0.972 | 0.9876 / 0.718 |
| v3_Ukingdom_10M | 0.9950 / 0.978 | 0.9949 / 0.976 | 0.9945 / 0.987 | 0.9949 / 0.981 | 0.9913 / 0.413 |
| v3_Ukingdom_50M | 0.9975 / 0.981 | 0.9974 / 0.979 | 0.9971 / 0.990 | 0.9974 / 0.986 | 0.9955 / 0.825 |
| v3_Ukingdom_200M | 0.9983 / 0.982 | 0.9982 / 0.981 | 0.9980 / 0.989 | 0.9981 / 0.985 | 0.9959 / 0.877 |
| v3_Uhuman_2M | 0.9838 / 0.923 | 0.9843 / 0.908 | 0.9785 / 0.929 | 0.9820 / 0.943 | 0.9610 / 0.733 |
| v3_Uhuman_10M | 0.9916 / 0.958 | 0.9920 / 0.959 | 0.9894 / 0.966 | 0.9907 / 0.964 | 0.9798 / 0.710 |
| v3_Uhuman_50M | 0.9954 / 0.969 | 0.9956 / 0.970 | 0.9942 / 0.967 | 0.9951 / 0.974 | 0.9870 / 0.794 |
| v3_Uhuman_200M | 0.9965 / 0.976 | 0.9966 / 0.974 | 0.9954 / 0.965 | 0.9960 / 0.973 | 0.9906 / 0.840 |

## 17. Window outputs: dedicated heads against the pooled per-bp readout

Source: `runs_v3/<run>/window_diagnostic_{val,test}.json` (`src/eval/window_diagnostic.py`).
The readout was chosen on the validation side; the test side is shown and agrees with
validation to within 0.002 in every cell. Median R-squared over channels; the last
column is the fraction of channels above 0.99 at 50 bp.

| run | 10 bp: head -> pooled | 20 bp | 50 bp | 50 bp channels > 0.99 |
| --- | --- | --- | --- | --- |
| v3_L2M | 0.9501 -> **0.9784** | 0.9408 -> **0.9849** | 0.9392 -> **0.9899** | 0.00 -> 0.49 |
| v3_L10M | 0.9593 -> **0.9864** | 0.9558 -> **0.9911** | 0.9459 -> **0.9942** | 0.00 -> 0.57 |
| v3_L50M | 0.9639 -> **0.9891** | 0.9598 -> **0.9931** | 0.9492 -> **0.9954** | 0.01 -> 0.62 |
| v3_L200M | 0.9664 -> **0.9894** | 0.9618 -> **0.9932** | 0.9502 -> **0.9958** | 0.01 -> 0.62 |

A window target is the window average of the per-bp target, and the model's per-bp
output is already at 0.993-0.999, so pooling it is the better window estimate at every
rung; the dedicated heads were the weak link (0.94-0.97). The pooled readout is now
the primary window readout (`window_readout="pooled"`) and the final evaluation of every
new run reports both.

## 18. What limits the weakest non-L per-bp channels: the 16-snapshot ensemble

Source: `results/snapshot_ceiling_v2.json` (`src/eval/snapshot_ceiling.py`), 200M test R2
from `runs_v3/v3_L200M/final_metrics.npz`. The relaxed-ensemble channels are estimated
from 16 relaxed snapshots on the 10 percent ensemble subset. A population p from 16
snapshots has sampling variance p(1-p)/16, estimated without bias from the targets
themselves; for spreads the Gaussian result s^2/(2(n-1)) is used.

| channel | estimator | ceiling | 200M test R2 |
| --- | --- | --- | --- |
| C_BI_fraction_C | population | 0.761 | 0.760 |
| C_BI_fraction_W | population | 0.751 | 0.746 |
| C_sugar_C2endo_prob_C | population | 0.473 | 0.465 |
| C_sugar_C2endo_prob_W | population | 0.489 | 0.482 |
| C_sugar_C3endo_prob_C | population | 0.014 | -0.006 |
| C_sugar_C3endo_prob_W | population | 0.023 | -0.008 |
| C_alpha_C_sd | spread | 0.769 | 0.206 |
| C_alpha_W_sd | spread | 0.774 | 0.211 |
| C_beta_C_sd | spread | 0.793 | 0.581 |
| C_beta_W_sd | spread | 0.797 | 0.587 |
| C_delta_C_sd | spread | 0.760 | 0.291 |
| C_delta_W_sd | spread | 0.762 | 0.302 |
| C_epsilon_C_sd | spread | 0.913 | 0.652 |
| C_epsilon_W_sd | spread | 0.911 | 0.644 |
| C_gamma_C_sd | spread | 0.900 | 0.104 |
| C_gamma_W_sd | spread | 0.904 | 0.075 |
| C_zeta_C_sd | spread | 0.906 | 0.727 |
| C_zeta_W_sd | spread | 0.905 | 0.706 |

The populations are at their ceilings: C3'-endo is a rare event whose 16-snapshot
estimate is almost pure sampling noise (ceiling 0.01-0.02; model 0), and the C2'-endo
population sits at its ceiling of 0.47-0.49. The torsion spreads score far below their
Gaussian-approximation ceilings, but that approximation assumes a unimodal angle;
alpha and gamma switch between discrete rotamers, and a spread over 16 snapshots then
behaves like a count of rare flips, whose noise the Gaussian formula understates. These
spread ceilings are therefore upper bounds. A replicate relaxed ensemble on a few hundred
fragments (as `noise_ceiling.py` does for the cgNA+ ensemble) would measure them
directly.

## 19. Family L: the strand tie and auxiliary data at 10M (held-out test side)

Source: `results/family_l_arms.json` (`src/eval/arm_report.py`); criteria pre-registered in
DECISIONS.md (2026-11-20). Medians of test R2.

| run | L per-bp | L whole-seq | L_m unpaired | L_m entropy | L_p unpaired | L_p entropy | all per-bp | all whole-seq | F whole-seq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_L10M (baseline) | 0.6581 | 0.8665 | 0.812 | 0.463 | 0.836 | 0.504 | 0.9977 | 0.9952 | 0.9870 |
| v3_L10M_auxL (+1M L windows) | 0.7427 | 0.8843 | 0.861 | 0.581 | 0.879 | 0.625 | 0.9979 | 0.9963 | 0.9825 |
| v3_L10M_tie (strand tie) | **0.7840** | **0.8934** | 0.899 | 0.669 | 0.899 | 0.663 | 0.9977 | 0.9965 | 0.9735 |

The strand tie (L_m(x) read from the reverse pass's L_p, an exact identity) is the
largest single gain on Family L so far, and it costs no data: both strands' targets now
train one predictor, and the minus-strand channels no longer need an internal reversal.
Both arms fail the pre-registered non-inferiority condition on one family: Family F's
whole-sequence median falls (by 0.0045 and 0.0135), driven by 11 of its 39 channels
(G-quadruplex loop and tract channels, longest tandem unit, triplex purine fraction)
while its median per-channel change is 0. Read as a capacity trade-off at 10M, it is
being measured across the ladder (`v4_L*`).

## 20. Whole-sequence channels as functionals of the per-bp output: not better (10M)

Source: `runs_v3/v3_L10M/derived_globals.json`. 104 of 303 whole-sequence channels are
exact functionals of a per-bp channel. Median test R2: head 0.9992, derived 0.9940
(derived better on 43 of 104); chosen per channel on val, 0.9994. Means are even,
maxima and spreads much worse from the derived readout (a smoothed conditional-mean
profile understates its own extremes). The windows' gain (section 17) does not carry
over: linear functionals can be derived from the per-bp output, nonlinear ones are
better learned directly.

### 18b. Measured ceilings by replicate ensemble

Source: `results/ensemble_replicate_ceiling_v2.json` (`src/eval/ensemble_replicate.py`): the
16-snapshot relaxed ensemble recomputed with an independent seed for 120 fragments of
the ensemble subset (7,200-7,320 position pairs per channel); noise variance is half the
mean squared replicate difference, total variance is the corpus variance.

| channel | measured ceiling (replicate) | estimate from targets | 200M test R2 | model / ceiling gap |
| --- | --- | --- | --- | --- |
| C_BI_fraction_C | 0.774 | 0.761 | 0.760 | +0.014 |
| C_BI_fraction_W | 0.756 | 0.751 | 0.746 | +0.011 |
| C_alpha_C_sd | 0.276 | 0.769 | 0.206 | +0.070 |
| C_alpha_W_sd | 0.327 | 0.774 | 0.211 | +0.116 |
| C_beta_C_sd | 0.579 | 0.793 | 0.581 | -0.002 |
| C_beta_W_sd | 0.618 | 0.797 | 0.587 | +0.031 |
| C_delta_C_sd | 0.326 | 0.760 | 0.291 | +0.035 |
| C_delta_W_sd | 0.351 | 0.762 | 0.302 | +0.049 |
| C_epsilon_C_sd | 0.657 | 0.913 | 0.652 | +0.005 |
| C_epsilon_W_sd | 0.667 | 0.911 | 0.644 | +0.024 |
| C_gamma_C_sd | 0.061 | 0.900 | 0.104 | -0.044 |
| C_gamma_W_sd | 0.228 | 0.904 | 0.075 | +0.152 |
| C_sugar_C2endo_prob_C | 0.481 | 0.473 | 0.465 | +0.016 |
| C_sugar_C2endo_prob_W | 0.477 | 0.489 | 0.482 | -0.006 |
| C_sugar_C3endo_prob_C | -0.026 | 0.014 | -0.006 | -0.020 |
| C_sugar_C3endo_prob_W | -0.030 | 0.023 | -0.008 | -0.022 |
| C_zeta_C_sd | 0.732 | 0.906 | 0.727 | +0.005 |
| C_zeta_W_sd | 0.732 | 0.905 | 0.706 | +0.026 |

The model sits at the measured ceiling on the populations and on most spreads (median
gap +0.015); the largest remaining gaps are the alpha and gamma spreads of the
Watson strand (0.12 and 0.15), whose ceilings are themselves low. The estimate from the
targets alone (section 18) was close for the populations but far too optimistic for
the torsion spreads (0.76-0.91 against measured 0.06-0.73), as anticipated for rotameric
angles. These channels are limited by the 16-snapshot targets, not by the model; more
snapshots, not a larger model, would raise them.

## 21. Aim 2 calibration with the development encoder (`v3_L10M_tie`)

Source: `results/readout/`, `results/readout_summary.json` (`src/readout/`). Test side;
splice sites use the dataset's own test split, loop-seq a similarity-grouped 10 percent.

| task | metric | sequence | physics | seq + physics | physics via 4 channels | decoder: bases recovered |
| --- | --- | --- | --- | --- | --- | --- |
| splice-site class (negative control) | accuracy | 0.976 | 0.976 | 0.978 | 0.976 | 100% (1.997 / 1.997 bits) |
| loop-seq cyclizability (first-pass head) | R2 | 0.716 | 0.706 | 0.723 | 0.709 | 100% (1.968 / 1.968 bits) |

The per-bp landscape is invertible: a 4-channel linear projection of it recovers every
base. So a physics-only head can do whatever a sequence head does (negative control
included), and narrowing channels is no bottleneck. Aim 2's rate-distortion curve is
therefore measured against the bits of sequence a representation still carries
(information-limited representations: the landscape smoothed over 5-50 bp, and the
whole-sequence channels alone; runs queued). Loop-seq assay ceiling (technical
replicates, Basu et al. 2021): 0.869; both heads reach about 0.81-0.82 of it.

## 22. The v4 ladder (strand tie) against v3, rung by rung (test side)

Source: `results/ladder_v3_vs_v4.json` (`src/eval/ladder_compare.py`); the 10M v4 rung is
`v3_L10M_tie` (identical training path).

| rung | L per-bp v3 -> v4 | L whole-seq v3 -> v4 | F whole-seq v3 -> v4 | F channels down / up (> 0.005) |
| --- | --- | --- | --- | --- |
| 2M | 0.6423 -> 0.7335 | 0.8556 -> 0.8770 | 0.9147 -> 0.9122 | 6 / 5 |
| 10M | 0.6581 -> 0.7840 | 0.8665 -> 0.8934 | 0.9870 -> 0.9735 | 11 / 3 |
| 50M | 0.6579 -> 0.7739 | 0.8573 -> 0.8834 | 0.9873 -> 0.9875 | 8 / 6 |
| 200M | 0.6309 -> 0.7470 | 0.8582 -> 0.8815 | 0.9823 -> 0.9838 | 3 / 10 |

## 23. Family L data scaling at 50M; seed spread at 10M

| run | L per-bp | L whole-seq | F whole-seq | all per-bp | all whole-seq |
| --- | --- | --- | --- | --- | --- |
| v3_L50M | 0.6579 | 0.8573 | 0.9873 | 0.9988 | 0.9956 |
| v3_L50M_auxL (+1M L windows) | **0.7350** | **0.8851** | 0.9890 | 0.9990 | 0.9962 |

Every other family non-inferior (worst change -0.0005); the pre-registered criterion is
met. Seed spread at 10M (two seeds, same split): per-bp families within 0.001, L 0.0066;
whole-sequence F 0.0023, L 0.0031, others within 0.0005. Against it, the strand tie's L
gain (+0.126) is 19 times the spread and its 10M F cost (-0.0135) about 6 times; the 10M
auxiliary-data F change (-0.0045) is within twice the spread.

## 24. The complete v3 scaling ladder, 2M to 1B (held-out test side)

Source: `results/scaling_v2.json`, `results/figures/scaling_v2.png`
(`src/eval/scaling_report.py`).

| rung | parameters | per-bp median R2 | per-bp error | whole-seq median R2 | per-bp > 0.99 | whole-seq > 0.99 | 68% interval coverage | L per-bp | 50-bp windows (pooled) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2M | 2.1M | 0.9933 | 0.0067 | 0.9924 | 0.612 | 0.551 | 0.691 | 0.6423 | 0.9899 |
| 10M | 9.5M | 0.9977 | 0.0023 | 0.9952 | 0.696 | 0.625 | 0.692 | 0.6581 | 0.9942 |
| 50M | 48.9M | 0.9988 | 0.0012 | 0.9956 | 0.727 | 0.625 | 0.696 | 0.6579 | 0.9954 |
| 200M | 198.1M | 0.9993 | 0.0007 | 0.9950 | 0.727 | 0.607 | 0.688 | 0.6309 | 0.9958 |
| 1000M | 997.0M | 0.9997 | 0.0003 | 0.9948 | 0.741 | 0.588 | 0.682 | 0.6577 | 0.9962 |

Per-bp median error falls as a power law in parameters, (1 - R2) ~ N^-0.47 (fit over the
five rungs), from 0.0067 at 2M to 0.0003 at 1B. The whole-sequence median peaks at 50M
(0.9956) and is flat to slightly lower above it (0.9950, 0.9948): those channels are not
capacity-limited, which is what the whole-sequence auxiliary-data test (`v3_L10M_auxAll`)
measures. Family L per-bp is flat across the whole ladder (0.63-0.66) -- the handicap the
strand tie removes (sections 19, 22). Predicted 68 percent intervals cover 68-70 percent
of test values at every size.

## 25. Aim 2 calibration, complete (development encoder `v3_L10M_tie`, tuned head)

Source: `results/readout/*__tuned.json`, `results/readout_summary.json` (`src/readout/`).
Test sides: splice sites 3,000 (provided split; metric accuracy, chance 1/3); loop-seq
19,247 (similarity-grouped; R2; sourced ceiling 0.869); Jores plant promoters 14,749
(provided split; R2, one output per assay system). In parentheses: the fraction of the
sequence head's score (above chance for splice sites). Bits per base: decoder on the
splice and loop-seq sequences (entropy 1.997 and 1.968).

| physics representation | bits / base (splice; loop-seq) | splice | loop-seq | plant |
| --- | --- | --- | --- | --- |
| sequence head (reference) | 2.0 | 0.976 | 0.748 | 0.712 |
| sequence + physics | - | 0.978 | 0.747 | 0.722 |
| per-bp landscape (286 channels) | 2.00; 1.97 | 0.976 (1.00) | 0.742 (0.99) | 0.672 (0.94) |
| 4 shape channels, per bp | 2.00; 1.97 | 0.970 (0.99) | 0.729 (0.97) | 0.693 (0.97) |
| Family H only (methylation deltas) | 0.19; 0.85 | 0.547 (0.33) | 0.031 (0.04) | - |
| 5-bp bins, 4 channels | 1.58; 1.81 | 0.963 (0.98) | 0.655 (0.88) | - |
| 10-bp bins, 4 channels | 0.64; 0.89 | 0.949 (0.96) | 0.313 (0.42) | 0.560 (0.79) |
| 20-bp bins, 4 channels | 0.37; 0.38 | 0.943 (0.95) | 0.287 (0.38) | 0.510 (0.72) |
| 10-bp bins, 1 channel | 0.20; 0.12 | 0.891 (0.87) | 0.232 (0.31) | 0.425 (0.60) |
| 20-bp bins, 1 channel | 0.15; 0.08 | 0.832 (0.78) | 0.244 (0.33) | 0.365 (0.51) |
| whole-sequence channels only | 0.19; 0.34 | 0.760 (0.66) | 0.325 (0.43) | 0.353 (0.50) |

Three results. (1) **The landscape is an invertible re-encoding of sequence**: every
substantive per-bp channel set, down to the four classic shape features, lets a decoder
recover every base, so physics-only equals sequence-only (0.976 / 0.976, 0.742 / 0.748,
0.672-0.714 / 0.712), and the plant sequence head matches the source paper's CNN (R2
0.71 / 0.67). (2) **Coarse-graining ranks assays by resolution dependence, not by
physics**: splice classes keep 0.66-0.95 of the sequence score in representations that
carry 0.2-0.6 bits per base, plant promoter strength 0.50-0.79, and loop-seq
cyclizability only 0.31-0.43 -- the reverse of the plan's expected physics ranking,
consistent with cyclizability depending on helical phasing that bins destroy. (3) **As
an input on top of sequence, the landscape adds little**: +0.010 R2 on the plant set
(one seed), +0.002 on splice sites, -0.001 on loop-seq.

## 26. Whole-sequence data scaling at 10M (`v3_L10M_auxAll`, 1M auxiliary windows with B-F and L whole-sequence targets)

| family | whole-seq median v3_L10M -> auxAll | per-channel: up / down by > 0.002 |
| --- | --- | --- |
| C | 0.9938 -> 0.9955 | 12 / 9 of 52 |
| D | 0.9969 -> 0.9983 | 7 / 0 of 41 |
| F | 0.9870 -> 0.9852 (within seed spread 0.0023) | 21 / 0 of 39 |
| L | 0.8665 -> 0.8822 | 60 / 1 of 74 |
| G, H (no auxiliary targets) | 0.9882 -> 0.9885, 0.9992 -> 0.9994 | 7 / 0, 2 / 0 |

All-channel whole-sequence median 0.9952 -> 0.9965; per-bp unchanged or higher in every
family (L +0.074). Pre-registered verdict: partial (C and D yes, F median within seed
noise while 21 of its 39 channels improve). The whole-sequence plateau seen across the
ladder (section 24) is at least partly a data limit.

## 27. Seed spread at 10M, three seeds (test side)

| family | per-bp median, mean +/- SD | whole-sequence median, mean +/- SD |
| --- | --- | --- |
| B | 1.0000 +/- 0.0000 | 0.9999 +/- 0.0000 |
| C | 0.9970 +/- 0.0002 | 0.9935 +/- 0.0002 |
| D | - | 0.9972 +/- 0.0003 |
| E | 0.9999 +/- 0.0000 | 0.9998 +/- 0.0001 |
| F | 0.9997 +/- 0.0000 | 0.9853 +/- 0.0051 |
| G | 0.9978 +/- 0.0000 | 0.9886 +/- 0.0004 |
| H | 0.9962 +/- 0.0006 | 0.9991 +/- 0.0001 |
| L | 0.6633 +/- 0.0047 | 0.8690 +/- 0.0023 |

All-channel medians: per-bp 0.9977, 0.9977, 0.9976; whole-sequence 0.9952, 0.9953, 0.9952.
Family F's whole-sequence median is the one unstable summary (range 0.0097 across seeds).

## 28. The electrostatics composition gate and its noise floor

| test | noise floor from | channels with SNR > 5 | median SNR | verdict |
| --- | --- | --- | --- | --- |
| as first computed | 8 replicates of an older construction (single-start minimiser) | 31 / 67 (46%) | 4.6 | FAIL |
| diagnostic | 16 replicates of the gate fragments' own construction, input displaced 0.001 A | 49 / 67 (73%) | 7.4 | PASS |
| binding | 16 replicates of the converged construction (input displaced 0.001 A); 644 fragments re-solved | 60 / 66 (91%) | 28.8 | PASS |

Within-group (base-order) share of variance: median 0.65, above the 0.05 floor in every
channel; mononucleotide composition R-squared: median 0.29, maximum 0.84 (minor groove).
Noise decomposition (median channel SD, kT/e): grid placement 0.0003; minimiser path
0.0046 capped, 0.0007 converged.

## 29. Families added on 2026-11-21 and their checks

| family | channels | check |
| --- | --- | --- |
| J, single-stranded DNA (self-structure; oxDNA2 stacking for v6) | 54 (+12) | J_m(x) = J_p(rc x) exact; MFE equals the Family E DNA fold model |
| K, RNA duplex and DNA-RNA hybrid | 410 | RNA duplex RC-exact; hybm(x) = hybp(rc x) exact; A-form signatures (RNA slide -1.63 A, roll 8.6, twist 30.2, minor groove 11.2 A) |
| N, RNA modification response (m6A, pseudouridine, inosine) | 42 | N_m(x) = N_p(rc x) exact; m6A destabilises paired A by +0.57 to +1.40 kcal/mol; pseudouridine stabilises paired U by -0.77 to -2.50 |
| I, protein readout (held apart) | 60 | best-covering-site energy exactly RC-symmetric; every model's top bases spell the factor's known site; median fragment at 0.49 occupancy for the middle potential |
| A, electrostatics (declared; production pending) | 479 | binding and final composition gates PASS (median SNR 28.8 / 29.3); all 58 potential-derived channels measurable at 150 mM have within-group SNR > 5 (5.7-76); the solvation entry was computed from broken nonlinear totals and is retracted (DECISIONS, correction of 2026-11-21) |

Side shards: J 191,344 fragments, 0 failed; K 191,344, 0 failed; N 191,344, 0
failed; I 191,344, 0 failed.

## 30. Electrostatics convergence, fixed protocol (validation, 12 sequences)

rmse over SD of 41 bp against 61 bp, by shell (axis radii 5-18 A; surface offsets 0-11 A):
150 mM 0.002-0.031 for every shell but the outermost three (0.135, 0.086, 0.272); 50 mM
0.012-0.077 for the inner shells, 0.13-0.63 outer. The first protocol's 0.11 floor at the
inner shells was the protocol (re-minimisation and per-length grids), not physics. Plan
scale (2,004 sequences, nine conditions) running.

## 31. Quantum chemistry tier 1, hexamer contexts (centred)

2,080 of 2,080 contexts (GFN2-xTB). Mean over each central tetramer's 16 flanks against the
tetramer tier: r = 0.9985, mean |difference| 0.10 kcal/mol; flank effect median 0.10
kcal/mol against a between-tetramer SD of 2.1. (The first run, one base off centre, is in
`results/superseded/`.)


## 32. The electrostatics gate on the final construction (production)

Source: `results/electrostatics/composition_gate_final_v5.json`, `..._channels.tsv`,
`results/electrostatics/channel_snr_final_v5.json` (`src/physics/composition_gate.py`,
`src/physics/channel_snr.py`; 644 composition-control fragments in 50 groups and 16
replicate solves, 150 mM, converged minimiser, exclusion radius 2.0 A, exact 0.35 A fine
spacing). Thresholds fixed in advance (SNR 5, eta floor 0.05, pass fraction 0.5).

| quantity | value |
| --- | --- |
| verdict | PASS |
| channels with within-group SD > 5x replicate noise | 89.4 % (threshold 50 %) |
| channels with eta above the floor | 100 % |
| median eta / median SNR | 0.687 / 22.6 |
| composition R2, median / max | 0.26 / 0.83 |
| channel kinds with within-group SNR > 5 | 60 of 60 |
| lowest kinds | minor-groove radial-field gradient 9.7; per-bp solvation 10.8 |
| surface potentials (36 shells/sectors) | median 19.0, minimum 6.7 |

Ceiling R2 implied by the replicate noise (1 - 1/SNR_total^2): 0.978 for the weakest
surface shell, 0.990 for per-base-pair solvation, 0.995-0.9995 for the other kinds.

## 33. Quantum chemistry tiers 2-3, timing and first values (AAAA context)

Source: `tmp/psi4probe` probe (job 34114976), 8 threads, Psi4 1.11.

| quantity | def2-SVP | def2-TZVP |
| --- | --- | --- |
| counterpoise B3LYP-D3BJ stacking, AA/TT step (kcal/mol) | -13.43 | -13.92 |
| A.T H-bond, bp1 / bp2 (kcal/mol) | -17.50 / -17.37 | -16.92 / -16.79 |
| HOMO / LUMO / gap (eV) | -5.76 / -1.17 / 4.59 | -5.91 / -1.44 / 4.47 |
| dipole (D) | 2.92 | 3.08 |
| vertical IP / EA (eV) | 7.18 / -0.29 | -- |
| stacking time / full DFT set (s) | 116 / 348 | 514 / 718 |

SAPT0/jun-cc-pVDZ stacking: total -12.63 kcal/mol = electrostatics -3.53 + exchange 12.23 +
induction -1.50 + dispersion -19.84 (558 s).

## 34. Quadruplex structures and threading (Family M)

Source: `results/g4_structure_classification.json` (`src/physics/fold/g4_templates.py`), and
the self-threading checks in DECISIONS.md (`fold/g4_build.py`).

- RCSB nucleic-acid-only entries mentioning G-quadruplexes: 414; 406 in PDB format; 166
  unimolecular; **79 unimolecular parallel** (three tetrads 67, two 7, four 5). Propeller
  loops by length: 1 nt 99, 2 nt 57, 3 nt 65, 4 nt 4, 5 nt 9, 6 nt 1, 8 and 9 nt 1 each.
- Rebuilding each solved parallel three-tetrad structure from another core with
  leave-one-out donor loops (same-sequence donors excluded): core backbone RMSD median 1.22 A
  (max 1.90); loop backbone RMSD median (75th percentile) 1 nt 2.02 A (2.67), n = 91; 2 nt
  3.69 (4.28), n = 30; 3 nt 5.73 (6.14), n = 61; 5 nt 6.00 (6.62), n = 8.
- Full-atom pipeline (assembly, base substitution, AMBER OL15 + GBn2 minimisation): two
  probe targets, core RMSD 1.48-1.55 A before and after minimisation; minimisation does
  not move loops toward the solved conformation (loop medians 2.1/4.9/3.1 A before, 2.6/
  4.8/3.1 A after), so loop-dependent channels carry their ensemble spread.
- c-MYC Pu22 model, 150 mM KCl: inner-site potential -112.6 / -104.4 kT/e, groove 1 A out
  -2.50, loops 1 A out -1.20, electrostatic binding of two inner-site K+ -283.9 kJ/mol
  against Na+ -176.7 (Coulomb -2336.3; desolvation K+ +2052.4, Na+ +2159.6), solvation
  -7,089.6 kJ/mol.
- Coverage: 799 of 191,344 corpus fragments carry a quadruplex Family M models (four tracts
  of >= 3 G, loops 1-6 nt); i-motifs: 4 unimolecular four-tract structures in the PDB.

## 35. Split conformal calibration of the predicted variances (v4_L2M, test side)

Source: `runs_v3/v4_L2M/conformal.json`, `conformal_scales.npz` (`src/eval/conformal.py`):
per channel, conformal quantiles of |y - mu| / sigma on the validation side (9567 fragments),
coverage measured on the test side (9567 fragments); per-bp positions pooled, at most 200,000
values per channel and side.

| resolution | nominal | raw coverage (median) | raw: channels within 0.02 | conformal coverage (median) | conformal: within 0.02 | scale q/z (median) |
| --- | --- | --- | --- | --- | --- | --- |
| per_bp | 0.5 | 0.5090 | 0.622 | 0.5007 | 0.969 | 0.979 |
| per_bp | 0.683 | 0.6936 | 0.605 | 0.6835 | 0.990 | 0.979 |
| per_bp | 0.9 | 0.9051 | 0.738 | 0.9002 | 1.000 | 0.986 |
| per_bp | 0.954 | 0.9571 | 0.906 | 0.9541 | 1.000 | 0.986 |
| per_bp | 0.99 | 0.9905 | 1.000 | 0.9900 | 1.000 | 0.992 |
| global | 0.5 | 0.5059 | 0.616 | 0.5017 | 0.972 | 0.990 |
| global | 0.683 | 0.6892 | 0.601 | 0.6845 | 0.941 | 0.989 |
| global | 0.9 | 0.9063 | 0.793 | 0.9011 | 0.988 | 0.984 |
| global | 0.954 | 0.9563 | 0.926 | 0.9546 | 0.997 | 0.991 |
| global | 0.99 | 0.9898 | 0.981 | 0.9905 | 1.000 | 1.010 |

The raw Gaussian intervals are slightly wide (scale factors 0.98-0.99) and channel-to-channel
uneven; the conformal scales bring 94-99 percent of channels within two points of nominal at
the central levels. Applied to every release rung when it finishes; the scales ship with the
weights.

## 36. Composition residualisation (v4_L2M, test side)

Source: `runs_v3/v4_L2M/composition_residual.json`, `composition_fits.npz` (`src/eval/composition_residual.py`): ridge regression of every channel on composition (whole-sequence: GC, mono- and dinucleotide frequencies; per-bp: the same in an 11 bp window), fitted on 30000 training fragments, scored on 6000 test fragments (per-bp interior positions). Model R2 on the residual = prediction and target both minus the composition fit.

Medians over channels: per-bp composition R2 0.142, model 0.9970, model on residual 0.9963; whole-sequence composition 0.733, model 0.9942, model on residual 0.9551.

| resolution | family | composition R2 | model R2 | model R2 on residual |
| --- | --- | --- | --- | --- |
| per_bp | B | 0.231 | 0.9999 | 0.9999 |
| per_bp | C | 0.124 | 0.9952 | 0.9934 |
| per_bp | E | 0.457 | 0.9993 | 0.9984 |
| per_bp | F | 0.528 | 0.9997 | 0.9995 |
| per_bp | G | 0.122 | 0.9969 | 0.9965 |
| per_bp | H | 0.107 | 0.9989 | 0.9989 |
| per_bp | L | 0.150 | 0.7289 | 0.6937 |
| global | B | 0.966 | 0.9998 | 0.9953 |
| global | C | 0.877 | 0.9898 | 0.7976 |
| global | D | 0.843 | 0.9969 | 0.9880 |
| global | E | 0.482 | 0.9996 | 0.9976 |
| global | F | 0.546 | 0.9103 | 0.8473 |
| global | G | 0.931 | 0.9890 | 0.7854 |
| global | H | 0.832 | 0.9985 | 0.9912 |
| global | L | 0.619 | 0.8748 | 0.6951 |

Per-bp channels are mostly base-order (composition explains 0.11-0.53 by family) and the model keeps its accuracy on the residual; whole-sequence channels are mostly composition (0.48-0.97), and the residual is where the remaining error lives (C and G whole-sequence 0.80/0.79, L 0.70, F 0.85 on the residual) -- the whole-sequence data-scaling result (section 26) points the same way. Run on every release rung.

## 37. Quadruplex builder, leave-one-out validation (Family M)

Source: `results/g4_build_validation.json` (`src/physics/fold/validate_g4_build.py`): 35
distinct parallel three-tetrad sequences, each rebuilt from the 1KF1 core (1XAV for the 1KF1
sequence) with loops from other sequences, then minimised; 32 built.

| quantity | before minimisation | after |
| --- | --- | --- |
| core RMSD, median (A) | 1.19 | 1.12 |
| loop backbone RMSD, 1 nt (n = 51) | 2.15 | 1.98 |
| 2 nt (n = 16) | 5.13 | 4.38 |
| 3 nt (n = 22) | 5.79 | 5.72 |
| 5 nt (n = 7) | 5.70 | 4.52 |

## 38. Release ladder v5, 2M rung (test side)

Source: `runs_v3/v5_L2M/final_metrics.json` against `runs_v3/v4_L2M/final_metrics.json`. v5 adds Families J, K and N (623 per-bp and 508 whole-sequence channels against 286 and 323) and the whole-sequence auxiliary windows; best epoch 96. All-channel medians: per-bp 0.9942 (v4 0.9936), whole-sequence 0.9902 (v4 0.9942); the medians are over different channel sets.

| family | per-bp v4 | per-bp v5 | whole-seq v4 | whole-seq v5 |
| --- | --- | --- | --- | --- |
| B | 0.9999 | 0.9998 | 0.9998 | 0.9997 |
| C | 0.9905 | 0.9844 | 0.9901 | 0.9836 |
| D | -- | -- | 0.9971 | 0.9961 |
| E | 0.9997 | 0.9996 | 0.9996 | 0.9994 |
| F | 0.9997 | 0.9997 | 0.9122 | 0.9273 |
| G | 0.9973 | 0.9961 | 0.9891 | 0.9833 |
| H | 0.9889 | 0.9900 | 0.9985 | 0.9977 |
| J | -- | 0.8920 | -- | 0.8605 |
| K | -- | 0.9956 | -- | 0.9904 |
| L | 0.7335 | 0.7681 | 0.8770 | 0.8848 |
| N | -- | 0.7585 | -- | 0.8511 |

At 2M the extra 337 per-bp and 185 whole-sequence channels cost C and G about 0.006 (per-bp and whole-sequence) while L and F rise (L per-bp +0.035, F whole-sequence +0.015): capacity sharing at the smallest rung, to be compared at 10M and above. Rare-event channels (test): F_g4_best_tetrads R2 -0.006, AUROC 0.999; F_triplet_repeat_instability R2 0.000, AUROC 0.983; L_m_gquad_gain R2 0.001, AUROC 0.989; L_p_gquad_gain R2 0.002, AUROC 0.987.

Post-hoc (`conformal.json`, `composition_residual.json`, `eval_by_group.json`):
- per_bp at 68.3%: raw coverage median 0.6799 (0.830 of channels within 2 points) -> conformal 0.6835 (0.997)
- global at 68.3%: raw coverage median 0.6880 (0.659 of channels within 2 points) -> conformal 0.6835 (0.992)
- composition: per-bp composition R2 0.138, model 0.9959, on the residual 0.9950; whole-sequence 0.796 / 0.9904 / 0.8446
- by stratum (per-bp / whole-sequence): extremes n=166 0.9939 / 0.9859; hexamer n=659 0.9964 / 0.9993; mpra n=2562 0.9939 / 0.9702; panel n=5989 0.9941 / 0.9780; random n=164 0.9940 / 0.9776

## 39. QC Stage B, probe values (def2-SVP, B3LYP-D3BJ, 4 threads)

Source: `tmp/psi4feat/stageb_svp.json` (job 34119950; `run_psi4.py --stage-b`), contexts AAAA
(central AA/TT) and GCGC (central CG/CG); ~600 s per context, 370 s of it polarizability.

| quantity | AAAA | GCGC |
| --- | --- | --- |
| hole site energy, purine / pyrimidine (eV) | A -5.83 / T -6.42 | G -5.06 / C -6.51 |
| electron site energy, purine / pyrimidine (eV) | A -0.26 / T -1.05 | G +0.77 / C -0.93 |
| hole transfer, intrastrand W / C (eV) | 0.054 / 0.074 | 0.012 / 0.002 |
| hole transfer, interstrand diagonals (eV) | 0.012 / 0.024 | 0.001 / 0.043 (G-G) |
| electron transfer, intrastrand W / C (eV) | 0.054 / 0.024 | 0.057 / 0.049 |
| in-pair hole transfer, bp1 / bp2 (eV) | 0.012 / 0.011 | 0.016 / 0.018 |
| MBIS net charge of the purine (e) | +0.044 | -0.043 (bp1 G) |
| polarizability, isotropic / along the stack (a.u.) | 292 / 169 | 290 / 170 |

Guanine carries the highest hole site energy and the cross-strand G-G coupling of a CG step
exceeds its intrastrand couplings, as in published fragment-orbital studies of B-DNA. The
groove MEP of this probe used the first (centre-based) probe directions and is not reported;
production uses the local directions (DECISIONS).

## 40. Release ladder v5, 10M rung (test side)

Source: `runs_v3/v5_L10M/final_metrics.json` against `runs_v3/v3_L10M_tie/final_metrics.json` (the 10M strand-tie run, 286/323 channels). Best epoch 99; all-channel medians per-bp 0.9976, whole-sequence 0.9950 (623/508 channels; not comparable across channel sets).

| family | per-bp v3 tie | per-bp v5 | whole-seq v3 tie | whole-seq v5 |
| --- | --- | --- | --- | --- |
| B | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| C | 0.9971 | 0.9939 | 0.9941 | 0.9915 |
| D | -- | -- | 0.9975 | 0.9975 |
| E | 0.9999 | 0.9998 | 0.9999 | 0.9998 |
| F | 0.9997 | 0.9997 | 0.9735 | 0.9733 |
| G | 0.9978 | 0.9974 | 0.9881 | 0.9847 |
| H | 0.9968 | 0.9949 | 0.9993 | 0.9990 |
| J | -- | 0.9131 | -- | 0.8772 |
| K | -- | 0.9983 | -- | 0.9939 |
| L | 0.7840 | 0.8409 | 0.8934 | 0.9109 |
| N | -- | 0.8395 | -- | 0.8995 |

Family L gains from the auxiliary windows and the tie (per-bp +0.057, whole-sequence +0.018); C, G and H give back 0.002-0.003 at this size, as at 2M (section 38) but smaller. Rare events at 10M: F_g4_best_tetrads R2 0.952, AUROC 1.000; F_triplet_repeat_instability R2 0.001, AUROC 0.967; L_m_gquad_gain R2 -0.001, AUROC 0.991; L_p_gquad_gain R2 -0.000, AUROC 0.992 -- the tetrad count is sized at 10M (R2 0.952 against -0.006 at 2M); triplet-repeat instability and the RNA quadruplex gain are detected but not sized.

v5_L10M post-hoc:
- per_bp at 68.3%: raw 0.6818 (0.745 within 2 points) -> conformal 0.6835 (0.979)
- global at 68.3%: raw 0.6701 (0.380 within 2 points) -> conformal 0.6839 (0.982)
- composition: per-bp 0.138 / model 0.9983 / residual 0.9979; whole-sequence 0.796 / 0.9949 / 0.9354
- by stratum (per-bp / whole-sequence): extremes 0.9973 / 0.9922; hexamer 0.9989 / 0.9996; mpra 0.9975 / 0.9828; panel 0.9975 / 0.9877; random 0.9974 / 0.9874

## 41. Rare-event channels under beta = 1 (2M, test side; pre-registered, fails)

Source: `runs_v3/v5r_L2M/final_metrics.json` against `runs_v3/v5_L2M/final_metrics.json`.

| channel | R2 v5 | R2 beta=1 | AUROC v5 | AUROC beta=1 |
| --- | --- | --- | --- | --- |
| F_g4_best_tetrads | -0.006 | 0.552 | 0.999 | 0.998 |
| F_triplet_repeat_instability | 0.000 | 0.806 | 0.983 | 0.980 |
| L_m_gquad_gain | 0.001 | 0.734 | 0.989 | 0.933 |
| L_p_gquad_gain | 0.002 | 0.610 | 0.987 | 0.940 |
| C_sugar_C3endo_prob_C (per-bp) | 0.014 | 0.016 | 0.837 | 0.827 |
| C_sugar_C3endo_prob_W (per-bp) | 0.021 | 0.023 | 0.837 | 0.837 |

Family medians unchanged within 0.0015 except whole-sequence N (0.8511 -> 0.8254) and J
(0.8605 -> 0.8577). Fails the pre-registered AUROC and non-inferiority conditions; not used in
the final ladder. Seed replicates at 2M queued to size the noise.

## 42. Family M labels on the corpus (quadruplexes)

Source: `famM/corpus_solves/` (job 34119807; `run_family_m.py`, 3 models per fragment): all 799
corpus fragments with a modelled quadruplex; median 710 s per fragment (max 1,345 s).

| channel (150 mM KCl) | median | SD across fragments | 5-95 % | ensemble noise (SD/sqrt 3) | implied ceiling R2 |
| --- | --- | --- | --- | --- | --- |
| inner-site potential (kT/e) | -110.2 | 1.70 | -113.7 to -108.1 | 0.42 | 0.94 |
| loop potential 1 A out (kT/e) | -1.53 | 0.31 | -2.04 to -1.03 | 0.06 | 0.96 |
| groove potential 1 A out (kT/e) | -2.65 | 0.17 | -2.93 to -2.37 | -- | -- |
| ion binding, two K+ (kJ/mol) | -291.7 | 7.5 | -306.2 to -281.1 | -- | -- |
| K+/Na+ selectivity (kJ/mol) | -108.4 | 0.97 | -110.2 to -107.0 | -- | -- |
| solvation (kJ/mol) | -11,432 | 2,722 | -16,929 to -7,784 | -- | -- |
| model spread (A, pairwise RMSD) | 2.24 | 1.21 | 0.28 to 4.23 | -- | -- |

The ceiling uses the ensemble's own spread as the label noise (three models: SD/sqrt 3); the
core-dominated selectivity varies least across sequences, the loop-dependent channels and the
solvation most.
