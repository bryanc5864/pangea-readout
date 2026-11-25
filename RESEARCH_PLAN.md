# RESEARCH PLAN

Numbered execution plan derived from `PLAN.md`, so later work can cite a section
("executing 4.2"). `PLAN.md` is the plan of record and is unchanged; this document is
the operational decomposition plus a frozen pre-registration.

Deviations from `PLAN.md` are not recorded here. They live in `DECISIONS.md`, with the
reason, and are reported in the paper.

---

## 1. Ground truth

### 1.1 Species panel
Fifteen RefSeq reference assemblies spanning 19.3 to 72.0 percent measured GC across
bacteria, archaea, fungi, plants, animals and protists. **Done.**

### 1.2 Fragment library
61 bp fragments in strata: `panel`, `hexamer` (all 4,096 hexamers in three fixed
flanking contexts), `extremes`, `random`, `compctrl`, `mpra`, and `annotated`
(promoter / enhancer / intergenic / repeat, requires human annotations). **Done except
`annotated`.**

### 1.3 Conformations
cgNA+ ground state for the full 61 bp context, all-atom reconstruction, restrained
minimisation under AMBER OL15. Restraints hold base and phosphate atoms to the cgNA+
positions so the conformation stays sequence dependent. **Done.**

### 1.4 Electrostatics
PDB2PQR with AMBER charges and radii, then focused nonlinear Poisson-Boltzmann in APBS.
Production configuration measured, not assumed: 0.35 A fine grid, 50 A transverse coarse
padding, 40 A axial, 34.9 s per solve. Sampling on cylindrical shells in the base-pair
frame and, as the primary channel, at fixed offsets outward from the molecular surface.
Scored over the central 21 bp only. **Pipeline done; production run pending section 2.1.**

### 1.5 Quantum chemistry
Three tiers with delta-learning between them.
- 1.5.1 GFN2-xTB over tetramer contexts (136 canonical). **Done.**
- 1.5.2 GFN2-xTB over hexamer contexts (2,080 canonical). **Running.**
- 1.5.3 Dispersion-corrected DFT over tetramer contexts at three stacking geometries,
  supervising a correction to 1.5.1. **Not started.**
- 1.5.4 Symmetry-adapted perturbation theory on a stratified subset, giving the
  electrostatic / exchange-repulsion / induction / dispersion decomposition. Until this
  runs, no decomposition is reported, only total interaction energies. **Not started.**
- 1.5.5 Transfer integrals and hole and electron site energies. **Not started.**

### 1.6 Analytic targets
cgNA+ ground state and stiffness; nearest-neighbour duplex thermodynamics with the
Owczarzy sodium correction. **Done.**

### 1.7 Experimental targets
loop-seq cyclizability, Array Melt stabilities, ORChID cleavage tables (all three
sequence-based, **done**); G4-seq, in vitro nucleosome occupancy, CPD-seq and dCPD-seq
(coordinate-only, need lifts; yeast calibrated, **hg19 for G4-seq outstanding**).
S1-END-seq and KAS-seq **unverified, nothing downloaded**.

---

## 2. Gates

### 2.1 Composition gate
From `PLAN.md`: confirm the computed potential separates sequences matched on base
composition, before training anything.

Procedure: solve the `compctrl` stratum (50 groups; within a group every fragment shares
mononucleotide composition exactly, and the dinucleotide-shuffled members share
dinucleotide composition exactly). Decompose each channel's variance into a between-group
part, which composition can explain, and a within-group part, which only base order can
explain. Compare the within-group part against the solver's own numerical noise, measured
from grid-jitter replicates of a single sequence.

### 2.2 Pre-training review
Code correctness, data leakage, experimental design, reproducibility. Must pass before
full training.

### 2.3 Post-results review
Fabrication, leakage, and whether every number traces to `results/`.

---

## 3. Encoder (PANGEA)

### 3.1 Architecture
Local dilated-convolution trunk with a receptive field near 300 bp carrying most
channels; a cheaper strided wide branch seeing 5 to 10 kb feeding only the mesoscale
channels. Conditions and molecule class enter through FiLM layers. Low-rank output head,
so the saturating rank is a measurement of intrinsic dimensionality.

### 3.2 Training
Stage 1 pretraining on analytic targets. Stage 2 delta-learning on expensive and
experimental targets, keeping the analytic prediction as a prior.

### 3.3 Three variants
Human-only, held-out-kingdom (bacteria and plants only), and multi-clade.

### 3.4 Scaling study
2, 10, 50, 200 and 1,000 million parameters; per-channel accuracy and the universality
gap plotted against capacity on the same axis.

### 3.5 Correctness tests
Per-channel reverse-complement parity against the 3DNA convention, as a unit test before
anything downstream runs.

---

## 4. Measurements

### 4.1 Effective rank
Participation ratio and spectral analysis of the channel covariance, per family and
overall, with the component count for 95 percent of variance.

### 4.2 Composition residualization
Every channel regressed on local GC and mono- and dinucleotide frequency; residuals kept
alongside raw channels; downstream analyses run on both.

### 4.3 Table-equivalence
Per-channel comparison against pentamer, heptamer and 9-mer query tables, plus the
randomized-table control. Report the table-equivalence fraction openly.

### 4.4 CEILING
Per dataset: assay replicate reproducibility (the noise ceiling), the fraction of
achievable variance a sequence model reaches, and the fraction a physics head reaches.
All three reported. Three arms at matched capacity: sequence-only, physics-only, and
concatenated. Bottleneck width swept over 4, 8, 16, 32, 64, 128, 256 and full. Decoder
test for sequence reconstructability.

### 4.5 Universality certificate
Human-only encoder evaluated on all fifteen genomes with no retraining; per-channel gap
between best and worst genome; per-genome accuracy regressed on that genome's GC.

### 4.6 Held-out-kingdom and synthetic-composition stress tests
Bacteria-and-plants encoder evaluated on human. Synthetic sequences from 5 to 95 percent
GC with computed labels, where memorization is impossible.

### 4.7 Transfer, composition shift, low-data curves
Cross-species activity transfer against a k-mer ridge baseline at matched capacity;
composition-stratum transfer both directions; subsampling to 100, 1,000, 10,000 and
100,000 examples.

### 4.8 Compute accounting
FLOPs, wall clock on matched hardware for GPU and CPU, peak memory, and the amortised
workload curve as a function of tasks per locus.

---

## 5. Pre-registration (frozen)

Fixed before the corresponding experiments run. Any later change is a deviation, logged
in `DECISIONS.md` with its reason and reported in the paper.

### 5.1 Composition gate
- **Primary quantity:** per channel, the within-group variance as a fraction of total
  variance (eta), and the within-group standard deviation divided by the grid-jitter
  standard deviation (snr).
- **Noise floor:** grid-jitter replicates of a single sequence, 8 seeds, jitter 0.175 A
  (half a fine grid spacing).
- **Pass:** at least 50 percent of electrostatics channels with snr above 5 **and** eta
  above 0.05.
- **Also reported:** R-squared of each channel regressed on mononucleotide counts alone;
  near 1.0 identifies a channel that is a composition feature in physical notation.
- **Stricter variant:** the same statistics restricted to dinucleotide-matched members.
- **Failure action:** the construction is degenerate. Change the ground truth before
  spending compute; do not relax the threshold.

### 5.2 Primary metric for channel prediction
Per-channel R-squared on held-out fragments, split both at random and by sequence
similarity. Reported per family, never pooled into a single headline number.

### 5.3 Universality thresholds (from `PLAN.md`)
- Median per-channel gap in R-squared below **0.05** counts as universal.
- Above **0.15** counts as falsified.
- Between the two is reported as partial, with the responsible channels named.
- Separately, per-genome accuracy is regressed on that genome's GC content; a
  significant slope is the signature of a model that learned composition statistics.
- The headline universality number comes from the **human-only and held-out-kingdom**
  encoders, never the multi-clade one.

### 5.4 Activity-task statistics
- At least 3 seeds per configuration, 5 where affordable, the same seeds for every method.
- Baselines get the same tuning budget as the proposed method, tuned on validation only.
- Test sets used once per final configuration, never for selection.
- Paired tests over seeds, folds or items (Wilcoxon signed-rank or permutation), 95
  percent bootstrap confidence intervals, effect sizes, Holm correction across multiple
  comparisons.
- Splits by sequence similarity and by chromosome, with the overlap quantified.

### 5.5 Stopping rules (from `PLAN.md`, unchanged)
1. **Composition gate.** If physics channels fail to beat GC plus 5-mer composition at
   matched capacity on at least three of five activity datasets, stop the ceiling arm
   and write up the atlas plus the composition result as the finding.
2. **Positive control gate.** If the physics head does not come close to the sequence
   model on in vitro nucleosome occupancy, the landscape is broken; fix before reporting
   anything else.
3. **Negative control gate.** If the physics head performs well on splice site
   recognition, sequence identity is leaking into the channels; find the leak first.
4. **Compression gate.** If a decoder reconstructs the sequence above roughly 95 percent
   per base, drop the compression framing and keep dimensionality and amortization.
5. **Speed gate.** If the crossover in tasks per locus is above real workloads, drop the
   speed claim from the headline.
6. **Universality gate.** If the median per-channel gap exceeds 0.15 in R-squared, or
   per-genome accuracy depends significantly on target GC, the strong claim is dead;
   report partial universality with the responsible families named.
7. **Table gate.** If high-order k-mer tables reproduce most channels at the accuracy
   PANGEA reaches, keep the network only where it beats a table and move the contribution
   to the conditioned, mesoscale and non-B-DNA channels.

### 5.6 Amendment to stopping rule 2, made before the experiment
Measured at the data stage: 147 bp windowed GC correlates with in vitro nucleosome
occupancy at r = 0.78, so R-squared about 0.61
(`results/yeast_coord_calibration.json`). The rule as written would be satisfied by any
model with access to local composition, which every channel has, so passing it would not
show the landscape is sound.

**Amended rule 2:** the positive control is scored against the **windowed-GC baseline**,
not against the sequence model. The reported quantity is the fraction of the variance
*unexplained by GC* that the physics head recovers. The gate passes only if the physics
head beats the GC baseline by a margin exceeding its seed-to-seed spread.

This amendment is made before the control is run and is logged in `DECISIONS.md`.
