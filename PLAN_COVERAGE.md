# PLAN coverage audit

Every item of `PLAN.md` revision 2 (the user's plan, recorded verbatim), with its status.
Amendments A1-A9 only add to the plan; nothing in revision 2 is dropped. Where one of the
plan's own stopping rules has fired, it is applied as the plan writes it and the rest of
the plan continues. Status: **done** (measured, in RESULTS.md), **partial** (part of the
item done, the remainder listed), **open** (not started; next action listed).

Updated 2026-11-22, 19:30. No PANGEA model exists (all deleted at the user's instruction, 2026-11-22); the final 2M-1B ladder trains once every family's labels exist.

## Aim 1 -- PANGEA

| plan item | status | evidence / next action |
| --- | --- | --- |
| 15-genome panel, 19-72% GC, all domains | done | RESULTS 1 |
| Six sampling strata (MPRA, genomic, hexamer library, composition extremes, panel, random) | done | RESULTS 2 (191,344 fragments) |
| Families B, C, D, E, F, G, H, L | done (with provisional channels listed) | v2 targets; provisional channels kept declared, parameters sourced where published (DECISIONS) |
| **Family A, electrostatics (560 channels)** | **production 96.5 % (184,605 of 191,344)** | plan design (full corpus at 10/50/150/500 mM, 15 % subsets at 25/100/250 mM, KCl, Na/Mg; ensemble 10 % x 8 snapshots done); final pass for out-of-memory and timed-out rows queued (34285092, 40 GB). Long-fragment set for the window-level channels (2,002 segments x 5 tiles x 9 conditions, job 34293939; Odijk OSF persistence length and integrated net charge over 10/25/50/100 bp). Base-pair opening and local dielectric response (image potential) implemented, test 34293674 after fixing a vacuum-relaxation artefact |
| Family J, single-stranded DNA (about 80) | production running | self-structure + oxDNA2 stacking (famJ2 side shards); oxDNA2 simulations of both strands of all 191,344 fragments (job 34288949): conformation, persistence length, extension at 5/10/20 pN, stacking, Debye-Hueckel electrostatics and charge density, self-pairing, burial/exposure (hydration), chemical-reactivity exposure proxies (KMnO4, DMS, glyoxal, CMCT) -- 46 channels, `single_strand_oxdna.py`, packer `pack_family_jox.py` |
| Family K, RNA duplex and DNA-RNA hybrid (about 150) | production running | 410 channels in v2 shards; A-form electrostatics on every fragment, 3 classes (job 34284715). Open: 2'-OH hydration and dynamics (plan: hexamer-scale RNA MD) |
| **Family M, non-canonical folds as explicit geometry (about 90)** | **builders done, tests running** | 309 channels: quadruplexes of every class (parallel/hybrid/antiparallel, 2-3 tetrads, 166 solved templates, fold-diverse ensembles, GBn2 MD ensembles on GPU, 192 channels), triplexes (H-DNA, C+ RESP charges, 38), i-motifs (tetramolecular cores + built loops, 45), cruciforms (four-way junction 1DCW/1L4J + cgNA+ arms, 34); loop builder for unmatched loops. 73,191 quadruplex candidates; windows resume 34283494 (v2 static) |
| Family N, RNA modification response (about 40) | partial | m6A, pseudouridine, inosine secondary-structure response and 2'-O-methyl duplex thermodynamics (Kierzek 2006); DFT on modified stacks: library of m6A/inosine/Psi variants in RNA and hybrid contexts building (34295699), then B3LYP-D3BJ/def2-TZVP. Open: modified-base electrostatics, geometry (no cgNA+ set for modified RNA exists) |
| Family I, protein readout (held apart, labeled variant only) | partial | 60 channels from 12 ProBound models (MotifCentral v1.0.0), side shards generating; generic side-chain groove energies open; never in headline numbers |
| Quantum chemistry: tight-binding hexamers, DFT tetramers at 3 geometries, SAPT subset | partial | DFT/SAPT tiers at 3 geometries and Stage B done or finishing; pKa shifts: GePol PCM unconverged in its mesh (1-2 pKa units), ddX converged to ~0.08 units (lmax 6 vs 8) -- production next (`slurm_qc_pka_array.sh`). Open: stacked trimers (plan: 'dimers and trimers') |
| Condition axis as encoder input (FiLM: salt, cation, divalent, T, superhelicity, pH, modification) | partial | conditions are channels at stage 1 (DECISIONS); FiLM inputs exist in the model and are unused -- conditioning to be switched on with the condition-resolved targets |
| Molecule-class axis as encoder input | open | needs J, K, M targets; the class enters through the same FiLM layers |
| Local trunk + wide branch (5-10 kb) for mesoscale channels | partial | local trunk done; wide branch implemented, unused (fragments are 61 bp); needs long-fragment targets |
| Window ladder 10, 20, 50, 100, 200, 500 bp | partial | 10/20/50 done (pooled readout, RESULTS 17); 100-500 bp need longer fragments |
| Three output resolutions with exact cross-scale consistency | done | per-bp, window, whole-sequence; exact relation imposed by construction (pooled windows) |
| Low-rank head; rank as a measurement | done | rank sweep, RESULTS 15 |
| Scaling study 2/10/50/200/1000M + universality gap vs capacity | done | RESULTS 24; universality gaps in `results/universality_v2.json` |
| Heteroscedastic outputs, family-balanced loss | done | beta-NLL; Kendall-Gal weights replaced (DECISIONS) |
| Split conformal calibration of variances | done (method); applied per release rung | `src/eval/conformal.py`; v4_L2M test: channels within 2 points of nominal at 68.3% go from 60% (raw) to 99% (per-bp) and 60% to 94% (whole-sequence) (RESULTS 35); run on each v5/v6 rung as it finishes |
| Reverse-complement equivariance + per-channel parity unit tests | done | tests; plus the exact strand tie for Family L |
| Stage-2 delta-learning on expensive and experimental targets | open | after Family A; experimental supervision (loop-seq, ORChID, Array Melt, nucleosome occupancy, G4-seq, CPD-seq) |
| Three encoders: human-only, held-out-kingdom, multi-clade | done at 2-200M | `v3_U*`; release versions to be retrained with the final configuration |
| Effective rank per family and overall | done | RESULTS 13 and v2 section |
| Composition residualisation of every channel | done (method); applied per release rung | `src/eval/composition_residual.py` (fits kept in `composition_fits.npz`); v4_L2M: per-bp composition R2 0.14, model 0.997 raw / 0.996 on the residual; whole-sequence 0.73 / 0.994 / 0.955 (RESULTS 36) |
| External validation against measurement | partial | Array Melt, ORChID, loop-seq done (RESULTS 14); CPD-seq photoproduct maps open |

## Traps and stopping rules

| item | status | evidence / next action |
| --- | --- | --- |
| Ideal-geometry collapse / composition gate for electrostatics | partial | composition explains median R-squared 0.29; diagnostic PASS with a construction-matched noise floor; binding gate on the converged construction pending (RESULTS 28) |
| **GC + 5-mer ridge at matched capacity (primary gate on every dataset)** | **open** | to add as a READOUT arm on every Aim 2 dataset |
| Similarity leakage | done | MinHash-grouped splits everywhere |
| **Capacity confounding: ridge -> MLP -> CNN sweep on both sides, matched seeds** | **open** | READOUT heads so far are CNNs only; ridge and MLP heads and seeds to add |
| Noise ceiling normalisation | partial | sourced ceilings where stated (loop-seq); the rest open |
| Protein-readout leakage | done by design | Family I absent from every number |
| Correlated-feature attribution (family ablation + Shapley) | open | Aim 3 attribution |
| Strand parity errors | done | unit tests |
| Channel count inflation (effective rank beside count) | done where counts appear | |
| QC fragment-library overfitting | open | held-out hexamer contexts with the hexamer tier |
| Universality by construction | done | headline from human-only and held-out-kingdom encoders |
| **Lookup-table collapse (pentamer/heptamer/9-mer tables + randomised-table control)** | **open** | per-channel table-equivalence fraction against PANGEA |
| Stopping rule: composition gate | open | needs the GC + 5-mer arm |
| Stopping rule: positive control (in vitro nucleosome occupancy) | **open** | dataset not yet ingested; ingest next |
| Stopping rule: negative control (splice sites) | fired | physics head 0.976 = sequence head; leak found -- the landscape is invertible (decoder) |
| Stopping rule: compression gate (decoder > 95% per base) | fired | decoder 100%; compression framing dropped exactly as the plan writes, dimensionality and amortisation kept (A9) |
| Stopping rule: speed gate | open | compute accounting below |
| Stopping rule: universality gate | evaluated | median gaps far below 0.05 for both universality encoders; GC dependence reported (`universality_v2.json`) |
| Stopping rule: table gate | open | with the lookup-table analysis |

## Aim 2 -- CEILING

| item | status | evidence / next action |
| --- | --- | --- |
| Three numbers per dataset (noise ceiling, sequence fraction, physics fraction) | partial | loop-seq complete; others need sourced ceilings |
| Three arms (seq, physics, seq+physics) at matched capacity | done on 3 calibration sets | RESULTS 25 (development encoder) |
| Bottleneck width sweep k = 4 ... full | done (+ information-limited extension) | RESULTS 25, A8 |
| Decoder test | done | RESULTS 25 |
| Positive control: in vitro nucleosome occupancy | **open** | ingest (e.g. Kaplan et al. 2009 in vitro map) |
| Negative control: splice sites | done (fired) | needs composition-matched decoys to be strict (added) |
| Datasets: lentiMPRA (K562, HepG2, WTC11), DeepSTARR, Jores, yeast, Sharpr-MPRA | partial | Jores done; lentiMPRA, DeepSTARR, yeast ingested and waiting for GPU featurisation; Sharpr-MPRA not ingested |
| Reachability spectrum across ~14 functional readouts | open | after the MPRA sets |
| Gap decomposition (motifs, spacing, CpG, repeats, distance) | open | |
| Probing formalism (floors, ceilings, headroom) | open | adopt in the Aim 2 report |
| Rerun with the final encoder | open | after `v4f_L200M` |

## Aim 3 -- universality and the regimes where physics wins

| item | status | evidence / next action |
| --- | --- | --- |
| Universality certificate (human-only encoder, per-genome gaps, GC regression) | done | RESULTS, `universality_v2.json` |
| Held-out-kingdom test | done | |
| Synthetic-composition stress test (5-95% GC, controlled dinucleotides, true labels) | partial | composition tails 5-15 / 85-95% GC and random strata done (RESULTS 16); controlled-dinucleotide libraries open |
| Function-level cross-species transfer (lentiMPRA -> fly, maize, sorghum, Arabidopsis, yeast) with k-mer ridge | open | |
| Composition shift and off-manifold arms | open | |
| Low-data learning curves (100 ... 100,000) | open | |
| Condition axes as predictions (labelled unvalidated) | open | needs the conditioned encoder |
| Attribution across families (ablation + group Shapley) | open | |

## Aim 4 -- design in physical coordinates

| item | status |
| --- | --- |
| Gradient and discrete search through PANGEA against physical targets | open |
| Solver verification, independent oracle, held-out reporter library | open |

## Compute accounting and deliverables

| item | status |
| --- | --- |
| FLOPs, wall-clock (GPU and CPU), memory vs Enformer, Borzoi, Flashzoi, FlashRNA, Enigma, Sei | open |
| Amortised workload curve in M | open |
| Projected landscapes (64 components, 8-bit) per genome + encoder release | open |
| Released encoders with weights; pipelines as workflows; benchmarking scripts | partial (code and frozen snapshots exist) |

## Order of work from here

1. Ground truth for the missing families, largest first: A (electrostatics), K (RNA
   duplex / hybrid), J (single-stranded DNA); then the QC hexamer/DFT/SAPT tiers.
2. The final PANGEA ladder, 2M-1B, on all families, with every improvement that passed
   (strand tie, auxiliary data, v2.1 targets, pooled windows), plus conformal calibration
   and the release human-only / held-out-kingdom / multi-clade encoders.
3. Aim 2 gates on every dataset: GC + 5-mer ridge, ridge/MLP/CNN head sweep with seeds,
   in vitro nucleosome occupancy (positive control), lookup-table analysis, composition
   residualisation; then lentiMPRA, DeepSTARR, yeast, Sharpr-MPRA and the reachability
   spectrum.
4. Aim 3 transfer, low-data and attribution arms; Aim 4 design; compute accounting.
