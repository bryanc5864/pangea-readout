# PANGEA and READOUT: a universal biophysical landscape of nucleic acids, and how much of regulatory function it reaches

Nov 19, 2026 · @Bryan Cheng

> **Transcription note.** This file is the user-supplied plan, revision 2, recorded word for word.
> The only change made in transcription is that tables arrived flattened (one cell per line) and
> are re-rendered here as markdown tables with the same cells in the same order. No word has been
> added, removed, reordered or paraphrased. Revision 1 is archived at `PLAN_v1_archived.md`.
> Corrections and inconsistencies found in this document are recorded in `DECISIONS.md`, never by
> editing this file.
>
> **Revision 3 (Nov 20, 2026).** At the user's direction ("modify the plan as you go"; Nov 21:
> "we will improve upon this plan, but never cut back"), the plan is amended as the work
> proceeds, and amendments only add: no item of revision 2 is removed or narrowed. Where one
> of revision 2's own stopping rules fires, it is applied as revision 2 writes it. Item-by-item
> status is in `PLAN_COVERAGE.md`. Revision 2 is kept verbatim both in the body below, which
> is unchanged, and in `PLAN_v2_archived.md`. Every change is in the section **Amendments
> (revision 3)** at the end of this file, each with its reason and the measurement behind it in
> `DECISIONS.md`.

## Summary

The central claim is universality. A sequence-to-function model learns the motif vocabulary and composition statistics of the genome it was trained on, and it degrades when moved to a species whose vocabulary differs. The Poisson-Boltzmann equation does not know what species it is in. Base stacking energies, duplex mechanics, hydration structure and photochemistry are the same in Plasmodium at 19 percent GC as in Streptomyces at 72 percent. An encoder that has genuinely learned the sequence-to-physics map should therefore predict with flat accuracy across the entire tree of life, and this project is built to test exactly that and to fail loudly if it is false.

The landscape has a second property no sequence-to-function model has. Its ground truth is computable anywhere. Poisson-Boltzmann solutions, coarse-grained mechanics and quantum chemical calculations can be run on any sequence from any organism, so universality here is directly measurable rather than inferred. A sequence-to-function model cannot be tested this way at all, because it needs matched functional assays in every species, and those do not exist.

This project builds a biophysical landscape of nucleic acids: a predicted field of roughly 1,670 physical and chemical observables at every position, under varying ionic strength, cation identity, temperature, superhelical stress, pH and base modification, across duplex DNA, single-stranded DNA, RNA, DNA-RNA hybrids and non-canonical folds. It is computable for any sequence from any organism, or from no organism at all. The project then measures what regulation looks like when a model can see only that landscape.

Stage one is PANGEA, the Pan-Genome Atlas of Energetics and Architecture. It is one encoder that maps a sequence, a molecule class and a vector of physical conditions to the landscape, and it is supervised by three kinds of ground truth. Numerical physics covers electrostatics, mechanics and thermodynamics. Quantum chemistry covers stacking energetics, frontier orbitals, charge transport and site-specific reactivity. High-throughput experiments cover cyclizability, backbone accessibility, duplex stability, non-canonical structure and photochemical damage. Every channel carries a calibrated uncertainty. PANGEA is trained once and then frozen.

Stage two is READOUT, a family of small heads that take PANGEA's predicted channels as their only input and predict function. The name is the term of art: proteins recognize nucleic acids through base readout and shape readout, and these heads read the physical landscape rather than the sequence. A READOUT head is tiny, trains in minutes, and is disposable, so there is one per assay, cell type and species. PANGEA stays frozen while they train, because letting gradients flow back would make it encode activity-relevant sequence features instead of physics and would quietly invalidate every claim in the project. READOUT also trains on PANGEA's predicted channels rather than on true solver values, because predictions are all that exist at inference time, and training on truth while testing on predictions is a mismatch that looks like success right up until deployment. The predicted uncertainties go in alongside the means, so a head can discount channels PANGEA is unsure about.

Because the landscape is a deterministic function of sequence, it cannot carry information the sequence does not already carry, so the honest places to win are compression, compute, sample efficiency and generalization. The project measures all four and reports, per dataset, the fraction of achievable predictive variance that survives the physics bottleneck.

What comes out: a released landscape for a panel of genomes spanning bacteria, archaea, fungi, plants and animals; a measurement of the intrinsic dimensionality of DNA biophysics; a quantitative ceiling on what physics alone explains about regulation; and a universality result that sequence models structurally cannot match.

## What exists, and what is open

No published work builds anything comparable. Two design constraints are worth stating at the front, because they are the failure modes this kind of project falls into. A prediction benchmark that does not clearly improve on anything is the first, and it is answered here by making the headline a measurement rather than a performance claim. A design evaluation that certifies itself is the second, and it is answered by requiring that any designed sequence be verified against the solvers and against oracles built independently of the designer.

What is published is a set of narrow, single-purpose models. Each one solves a slice.

| Prior work | What it does | What it leaves open |
| --- | --- | --- |
| Deep DNAshape (Li, Chiu, Rohs 2024) | Sequence to DNA shape with extended flanking context, trained on molecular dynamics | Trained on simulation, not measurement; geometry only, no electrostatic field, no conditions |
| DNAcycP (2022), MIXBend (2024) | Cyclizability from loop-seq | One assay, one scalar per position |
| ORChID (Greenbaum, Pang, Tullius 2007) | Experimental hydroxyl radical cleavage for all 512 pentamers; structural map of 30 Mb of ENCODE | Pentamer lookup, 30 Mb, no uncertainty, no joint model |
| Rohs et al. 2011 | Minor groove electrostatic potential maps from cleavage data | One scalar per base pair, one implied salt condition |
| cgDNA+, cgNA+, ABC, hexamer-scale MD | Sequence-dependent ground-state geometry and stiffness in closed form | A substrate for this work, not a competitor |
| ProBound (2022) | Biophysical binding free energies from SELEX | Protein readout, which is a separate question from DNA physics |
| PBML, PBGNN, DIS-PB | Neural surrogates for the Poisson-Boltzmann equation | General biomolecules, not genome-scale DNA, not salt-conditioned |
| SENIES, DENIES, iEnhancer-Flow | Shape features added to sequence features for enhancer classification | Classification not quantitative activity, small gains, no ceiling analysis |

Every one of those models predicts a handful of quantities, at one implied set of conditions, trained and validated on one or two genomes. None of them is a landscape, none is conditioned on the physical environment, and none has ever been tested for universality across the tree of life. Five things are open.

Universality, which is the biggest of them and the one this project leads with. No sequence-to-physics model has been evaluated across a species panel spanning the full range of base composition, and nobody has stated a universality criterion for one, let alone measured it. This is open partly because it has not been asked and partly because only a physics target makes it answerable, since the ground truth is computable in any organism while functional assays are not.

Electrostatics as a conditioned field rather than a scalar. Every use of DNA electrostatic potential in regulatory genomics collapses it to one number per base pair in the minor groove at one implied ionic strength. The physical object is a potential field around the duplex varying with groove, angle, radial distance, ion identity and ionic strength, with a counterion atmosphere that has its own structure. Predicting that field genome-wide and conditioned on the environment is new, and it gives a knob no sequence model has.

Quantum chemistry at genome scale. Stacking energetics decomposed into electrostatic, exchange-repulsion, induction and dispersion components, frontier orbital energies, hole and electron site energies and transfer integrals, site-specific molecular electrostatic potentials at the hydrogen bonding atoms proteins actually read, and pKa shifts. These are standard calculations on dinucleotide and trinucleotide fragments, they are strongly sequence dependent, and as far as I can find nobody has ever carried them onto a genome or asked whether they say anything about regulation.

The sufficiency measurement. Nobody has asked how much of regulatory activity is reachable through a physics bottleneck and answered it with a number. The question is well posed, it has a definite answer, and the answer is informative either way. A high number is a compression result. A low number is a real bound on structure-first approaches that the field currently assumes away.

The intrinsic dimensionality of DNA biophysics. Nobody has measured the effective rank of a channel stack like this along a real genome. Measuring it is cheap and it determines how the landscape should be stored and used.

One point about where the speed comes from, because getting it right tells you where to spend effort. The downstream head is cheap because physics is local, so the encoder needs a receptive field of hundreds of base pairs rather than hundreds of kilobases, and because the landscape is computed once per locus and reused across every cell type, assay and species. The encoder is itself a sequence model, so the saving is amortization and locality rather than the absence of sequence.

## The four claims

Each is written so a specific experiment can kill it, and the project is arranged so that killing any one still leaves a result worth publishing.

Claim one, universality. An encoder trained on one genome predicts the biophysical landscape with essentially flat accuracy on every other genome, across the full span of base composition from roughly 19 to 72 percent GC and across bacteria, archaea, fungi, plants and animals. Falsified if per-channel accuracy varies materially with the target genome's composition or clade, which would show the encoder learned composition statistics rather than the sequence-to-physics map. This is the lead claim because it is the one sequence-to-function models structurally cannot make, and because it is cheap to test: the ground truth is computable in any organism.

Claim two, compression. The landscape has a low effective rank, and a bottleneck of a few dozen physical degrees of freedom recovers a large measurable fraction of the regulatory signal a matched sequence model recovers, while the underlying sequence cannot be read back out of it. Falsified if the landscape fails to beat a matched-capacity model on GC content plus k-mer composition, or if a decoder reconstructs the sequence at high accuracy. The second outcome would show the landscape is a re-encoding of sequence rather than a bottleneck, which nobody has checked and which the field should know.

Claim three, generalization into function. Physics heads degrade less than sequence models under cross-species transfer of the activity task, under shifts in base composition, on synthetic sequences away from the genomic manifold, and in the low-data regime. Falsified if sequence models match or beat them in all four regimes.

Claim four, amortization. Over a workload of many loci evaluated against many assays and cell types, the total compute of encoder plus heads falls below the current efficiency frontier, because the encoder is paid once. Falsified if the crossover point sits above workload sizes anyone runs, or if a single efficient sequence model matches the stack at lower total cost.

Claims one and two are the ones most likely to produce clean results and least exposed to a moving field. Claim four is the most fragile, because the efficiency frontier moved substantially in the past year and will keep moving.

## Aim 1: PANGEA, the encoder that produces the landscape

PANGEA takes a sequence, a molecule class and a vector of physical conditions, and returns the per-position landscape with a predicted mean and variance for every channel.

The molecule class axis is what makes this a nucleic acid model rather than a B-DNA model, and it is the main expansion over the original scope. The classes are duplex DNA in B, A and Z forms, single-stranded DNA, duplex RNA, DNA-RNA hybrids, RNA hairpins and loops, and the non-canonical folds treated as explicit geometries rather than as propensity scores: quadruplexes, i-motifs, intramolecular triplexes and cruciforms. The ground truth path already exists for all of these, since cgNA+ parameterizes duplex RNA, hybrids and modified bases, and the solvers and quantum chemistry do not care what molecule they are handed. Adding this axis reaches R-loop biology, guide RNA behavior, antisense oligonucleotide design, aptamers and RNA structure, and it strengthens the central claim, because universality across nucleic acid chemistry is a larger statement than universality across genomes.

The condition axis is the second thing that separates this from existing sequence-to-structure models. Conditions are monovalent ionic strength, cation identity, divalent concentration, temperature, superhelical density, pH, and base modification state. One set of weights serves every condition, so the model learns how the physics responds to the environment rather than memorizing one environment. Almost all of the biology these channels are meant to explain happens under conditions different from the single implied setting at which shape features are normally computed, and the condition axes are where a physical model earns its keep against a statistical one.

Architecture, in two branches. A local trunk of dilated residual convolutions with a receptive field near 300 base pairs carries most of the channels, because local geometry, electrostatics and electronic structure are set within one or two helical turns. A second wide branch, strided and much cheaper per base, sees 5 to 10 kb and feeds only the mesoscale channels that genuinely need it: persistence length over long windows, nucleosome free energy, supercoiling response, R-loop formation. Expanding context uniformly would waste compute on channels that cannot use it. Conditions and molecule class enter through FiLM layers at each block in both branches.

Outputs come at three resolutions, and this is what makes the result a landscape across the sequence rather than a list of per-position numbers. Many of the quantities in the channel set are not defined at a point at all. Persistence length, cyclizability, nucleosome formation free energy, strand separation free energy, stress-induced destabilization and RNA folding free energy are all properties of an extent, and forcing them into a per-base output either discards them or misrepresents them.

The first resolution is per base pair, carrying the local quantities: geometry, stiffness, electrostatic potential at the sampled shells, site-specific potentials, stacking terms, reactivity and hydration. These tile continuously and give the fine-grained track.

The second is a window ladder at 10, 20, 50, 100, 200 and 500 base pairs, carrying everything defined over an extent. Each scale produces its own track, so sliding the window along a chromosome yields a continuous landscape at that scale, and the same quantity can be read at several scales at once. A 20 base pair window shows local mechanical and thermodynamic texture, a 200 base pair window shows the scale nucleosomes and bubbles live at.

The third is per sequence, carrying whole-construct quantities: total electrostatic solvation free energy, net charge, duplex melting temperature, overall fold free energy. These matter for design, where the object being specified is a construct rather than a position.

Architecturally this is a pyramid. Per-base channels come off the local trunk at full resolution. Window channels come off pooled representations at each scale, drawing on the wide branch for the larger windows. Global channels come off a final pooling. Where an exact relation between scales exists, such as integrated charge over a window being the sum of its per-base contributions, it is imposed as a consistency loss rather than left to chance, so the scales agree with each other by construction. Where the relation is emergent rather than exact, as persistence length is with respect to local stiffness, the channels are predicted independently and the approximate relation is used as a diagnostic.

The same structure is what turns fragment-level training labels into genome-wide tracks. A quantity measured once per training fragment becomes a sliding-window track at inference, with no retraining. That extrapolation is only valid where the quantity is local relative to its window, which is exactly what the convergence study in the ground truth section measures, so the two decisions are made together rather than separately.

The output head is deliberately low rank. Rather than projecting the trunk straight to the full channel set, it projects to a bottleneck of rank r and then out. This encodes the expectation that the landscape has few real degrees of freedom, keeps the head small, and makes r a measurement: the smallest rank at which per-channel accuracy saturates is an independent estimate of the intrinsic dimensionality of nucleic acid biophysics, obtained for free while fitting.

A scaling study rather than a size decision. Analytic training targets are available in unlimited quantity, which is rare and makes scaling cheap to study, so train at 2, 10, 50, 200 and 1,000 million parameters and plot per-channel accuracy against capacity. Then plot the universality gap against capacity on the same axis. The prediction worth stating in advance is that accuracy improves while universality degrades, because a larger model has more room to memorize composition statistics, and composition is exactly the confound the project is built to rule out. If that is what happens it is a real and counterintuitive result, namely that smaller models generalize better across genomes for physics prediction. Nobody has produced a scaling law for sequence-to-physics, so this publishes either way, and it answers the question of how large the model should be by measurement rather than by argument.

Outputs are heteroscedastic, with loss weights across families learned through uncertainty weighting rather than set by hand, since channels differ by orders of magnitude in scale and in label noise. Variances are calibrated after training using split conformal intervals. Uncertainty is load-bearing rather than decorative: it lets READOUT down-weight badly predicted channels instead of treating a noisy channel as a measurement, and it makes the universality claim checkable, because a model that is wrong on an unfamiliar genome should say so.

Reverse-complement equivariance is enforced architecturally and is also a correctness test. The channels are not all symmetric under strand exchange. Base-step parameters split into a symmetric set and a set that changes sign, with shift and tilt in the antisymmetric group, and the full assignment should be taken from the 3DNA convention rather than guessed. A channel predicted with the wrong parity is wrong in a way no validation correlation will reveal, so parity is checked channel by channel as a unit test before anything downstream runs.

Training has two stages, which is how the expensive labels get used efficiently.

Pretraining on analytically computable targets, available without limit at near-zero cost for any sequence and any molecule class. These are cgNA+ geometry and stiffness, nearest-neighbor and secondary-structure thermodynamics, stress-induced destabilization profiles, non-canonical propensity calculations, and the quantum chemical quantities looked up from precomputed fragment tables. Tens of millions of windows are easy here, drawn from many genomes. This stage teaches the trunk the local sequence-to-physics map before any expensive label is touched.

Delta-learning on the expensive and experimental targets. The model keeps the analytic prediction as a prior and learns the residual, which is the strategy that works in implicit solvation. Expensive labels are the Poisson-Boltzmann solutions, the converged quantum chemistry, loop-seq, hydroxyl radical cleavage, quadruplex sequencing, Array Melt stabilities, in vitro nucleosome occupancy, and photoproduct maps. Residual learning matters for universality in particular, because on a genome or a molecule class unlike anything in the expensive training set the model degrades toward a known physical approximation rather than toward whatever the network extrapolates.

Three PANGEA variants get trained, and the split is what makes the universality claim testable rather than asserted. A human-only encoder, so cross-species evaluation is genuinely zero-shot. A held-out-kingdom encoder trained on bacterial and plant sequence only and evaluated on human. And a multi-clade encoder trained on the full panel, which is the one released for use.

## The landscape: about 1,670 channels in thirteen families

The right mental model is a tensor rather than a feature list. Five axes: position, physical observable, environmental condition, molecule class, and resolution, where resolution spans per base pair, a window ladder from 10 to 500 base pairs, and whole-sequence. Families A through I below are the duplex DNA core. Families J through N are what the molecule class axis adds, and they are mostly the same observables computed for a different chemistry rather than new quantities invented to raise the count. A flat number understates the structure and overstates the independence, which is why the dimensionality analysis below is part of the deliverable rather than a defensive footnote.

| Family | Contents | Channels | Ground truth |
| --- | --- | --- | --- |
| A. Electrostatics and ionic atmosphere | Potential on cylindrical shells at 6 angular sectors and 6 radial offsets across 9 ionic conditions; field vector components and gradients per groove; condensed counterion fraction per phosphate per strand; ion-specific occupancy for sodium, potassium, magnesium and chloride at groove and backbone sites; electrostatic solvation free energy; electrostatic component of base-pair opening; Manning charge density parameter and surface charge density; electrostatic persistence length over 4 window sizes; strand and groove potential asymmetry; complementarity to a model cationic probe; local dielectric response | 560 | APBS nonlinear Poisson-Boltzmann on sequence-dependent conformations |
| B. Electronic structure and quantum chemistry | Stacking energy decomposed into electrostatic, exchange-repulsion, induction and dispersion at 3 geometries; hydrogen-bond energy decomposed the same way; vertical ionization potential and electron affinity; hole and electron site energies; intrastrand and interstrand transfer integrals; hole localization probability and hopping rate; oxidative damage susceptibility; molecular electrostatic potential at the 8 hydrogen-bonding atoms proteins read in each groove; pKa shifts at four ring nitrogens; polarizability tensor; base-pair and cumulative helical dipole; frontier orbital gap and localization; partial charge summaries | 98 | Dispersion-corrected DFT and symmetry-adapted perturbation theory on stacked dimers and trimers across all tetramer contexts |
| C. Geometry and local stiffness | 12 rigid-base coordinate means; the full 12 by 12 stiffness matrix upper triangle per step; groove widths, depths and their fluctuations; backbone torsions, sugar pucker phase and amplitude per strand with variances; BI and BII populations; the same geometry re-predicted under high salt, methylation and elevated temperature; local helical axis curvature and direction; phosphate position descriptors; sugar pucker state probabilities; stretch and overtwist coupling; Watson-Crick to Hoogsteen opening propensity; conformational substate occupancies | 215 | cgNA+ closed form, ABC and hexamer-scale MD libraries, cross-checked against Deep DNAshape |
| D. Mesoscale mechanics | Cyclizability at 4 loop lengths; isotropic and anisotropic bending stiffness with preferred bend direction over 6 window sizes; intrinsic curvature magnitude and direction; torsional and stretch moduli with twist-stretch coupling; twist-writhe partition at 5 superhelical densities; nucleosome formation free energy, rotational phasing and translational positioning at 3 ionic strengths; loop closure free energy at 5 lengths; persistence length across 5 ionic strengths; kink propensity; response to applied tension at 4 forces; bubble nucleation under torsional stress | 90 | loop-seq, in vitro nucleosome occupancy, coarse-grained mechanics |
| E. Thermodynamics and strand dynamics | Nearest-neighbor free energy, enthalpy and entropy; melting temperature at 5 ionic strengths; base-pair opening probability across 6 temperatures and 4 superhelical densities; opening and closing rates; imino exchange proxy; stress-induced destabilization at 4 superhelical densities; hairpin and cruciform free energy at 3 ionic strengths; triplex and i-motif free energy at 3 pH values; bubble size moments; heat capacity change; hydrogen bond count, length and strength; strand separation free energy over 3 window sizes | 85 | Array Melt equilibrium stabilities, nearest-neighbor models, Poland-Scheraga and stress-induced destabilization calculations |
| F. Non-canonical structure | Quadruplex free energy under potassium, sodium, lithium control and ligand stabilization, with topology class probabilities, loop and bulge descriptors and tetrad count; i-motif at 3 pH; Z-DNA transition free energy at 3 ionic strengths with junction energy and boundary position; intramolecular triplex at 3 pH; cruciform extrusion at 3 superhelical densities; slipped-strand and repeat hairpin propensity; A-phased bend descriptors; R-loop initiation and elongation free energy; single-strandedness; quadruplex-duplex competition; triplet repeat instability; branch migration propensity; transition cooperativity length | 53 | G4-seq, permanganate and S1 footprinting, S1-END-seq, KAS-seq, DRIP-based assays |
| G. Solvation, hydration and chemical reactivity | Hydroxyl radical cleavage per nucleotide per strand; solvent-accessible surface area by atom class; minor and major groove hydration occupancy, residence time, spine continuity and water bridge probability; polar and nonpolar hydration free energy; groove hydrophobicity; cleavage propensity for five nucleases; reactivity to six chemical probes; cyclobutane pyrimidine dimer propensity at four dipyrimidine classes, methylated and unmethylated; 6-4 photoproduct propensity; depurination and depyrimidination rates; 8-oxoguanine susceptibility; cytosine deamination in duplex and single-stranded context; alkylation susceptibility; abasic site stability; generic intercalator and minor groove binder affinity; interfacial water entropy | 85 | ORChID, CPD-seq and dCPD-seq, MD-derived hydration, chemical probing data |
| H. Modification response | For 5mC, 5hmC, 5fC, 5caC and N6-methyladenine, the induced change in groove width, minor groove potential, stacking energy, twist, roll, bending and torsional stiffness, melting temperature, cyclizability, hydration, photoproduct propensity and nucleosome free energy | 60 | cgNA+ with modified bases, DFT on modified stacks, methylation-stratified CPD-seq |
| I. Protein readout, held apart | Occupancy for a transcription factor panel at several chemical potentials, generic amino acid side chain interaction energies in each groove, nucleosome competition terms | 60 | ProBound free energy models fit to SELEX |
| J. Single-stranded DNA | Stacking free energy and stacking propensity, persistence length and stretching elasticity, charge density and electrostatics without a complementary strand, hydration, chemical reactivity, self-structure propensity | about 80 | Single-strand coarse-grained parameters, force spectroscopy, chemical probing |
| K. RNA duplex and DNA-RNA hybrid | A-form geometry and the full stiffness matrix, groove dimensions which differ sharply from B-form, electrostatics solved on A-form conformations, duplex thermodynamics, 2-prime hydroxyl hydration and dynamics | about 150 | cgNA+ for duplex RNA and hybrids, hexamer-scale RNA MD, APBS on A-form geometry |
| L. RNA secondary structure | Folding free energy, base pairing probability, loop entropies, helix stability, salt corrections, co-transcriptional folding proxies | about 60 | Nearest-neighbor RNA models with salt correction, Array Melt, chemical probing |
| M. Non-canonical folds as explicit geometry | Quadruplexes, i-motifs, intramolecular triplexes and cruciforms treated as real conformations rather than propensity scores, including the quadruplex central ion channel, which is an electrostatic object with no counterpart in duplex DNA | about 90 | Solved structures, MD ensembles, APBS and quantum chemistry on the folded geometries |
| N. RNA modification response | Induced change in the main channels for N6-methyladenosine, pseudouridine and 2-prime-O-methyl, applied across the RNA and hybrid classes | about 40 | cgNA+ with modified bases, DFT on modified stacks |

Family B is the one that makes this a chemistry project rather than a structural biology project, and it is the part with no precedent at genome scale. Stacking energies vary strongly with sequence, with reported total step energies spanning roughly minus 9.9 to minus 14.7 kcal per mole, and the balance of dispersion, electrostatic and polarization contributions is sensitive to sequence context in ways a single total energy hides. Charge transport is similarly sequence-determined: holes localize on guanine, pyrimidines act as barriers because of their higher oxidation free energy, and the distance dependence of transfer changes character beyond about three intervening base pairs as the mechanism shifts from tunneling to thermal hopping. None of these quantities has ever been computed along a genome and asked to explain regulation.

Family G carries the other genuinely new axis, which is photochemistry with experimental ground truth. Cyclobutane pyrimidine dimer formation is strongly sequence dependent, is measurable genome-wide at single-nucleotide resolution, and increases close to twofold at methylated cytosines with the magnitude set by flanking context. It is also regulatory-adjacent rather than incidental, since elevated dimer formation at specific promoter positions underlies known mutation hotspots at factor binding sites. That makes it both a supervisable channel and a sanity check, because a landscape that cannot predict measured damage is not describing the real chemistry.

Family I is held apart deliberately, and this is a load-bearing decision rather than tidiness. Occupancy computed from a binding free energy model is a motif model wearing thermodynamic clothing. Folding it into the main stack would smuggle sequence specificity back into a representation meant to be physics and would make the headline measurement uninterpretable. Every headline number is reported for families A through H alone. Family I appears only as a labeled variant, and the difference between the two separates what DNA physics explains from what protein recognition explains.

Two analyses run on the landscape itself before any downstream task touches it.

Effective rank. Participation ratio and spectral analysis of the channel covariance across a large multi-species sample, reported per family and overall, with the component count needed for 95 percent of variance. The working expectation is an effective rank in the tens out of 1,670, with base composition dominating the leading components and the condition axes contributing smooth low-rank structure. If that is the answer it gets reported plainly, and it becomes the justification for a compact bottleneck and for how the landscape is stored.

Composition residualization. Every channel is regressed on local GC content and mononucleotide and dinucleotide frequency, and the residual is kept alongside the raw channel. Downstream analyses run on both. A channel whose residual carries no signal is a composition feature in physical notation, and this project should say which of its own channels those are before a reviewer does.

## Generating the ground truth

This is the part most likely to fail quietly, so it gets specified in detail.

The trap that kills the electrostatics. If every window is built as an ideal fiber model and handed to the solver, the only thing varying with sequence is which atoms sit at which fixed positions. The resulting potential becomes a near-deterministic function of base composition, and the whole electrostatics family collapses into GC content with thousands of CPU-hours of extra steps. A surrogate can reach a very high R-squared against exactly that degenerate target while carrying no information at all. Every window must therefore be built on sequence-dependent conformations, so the geometry responds to sequence before the solver runs. Confirming that the computed potential separates sequences matched on base composition is the first check to run, ahead of training anything.

The electrostatics pipeline, per fragment. Fragment length is set per ionic condition by the screening length rather than fixed, as described below. Generate its sequence-dependent ground state from cgNA+. Reconstruct all-atom coordinates from the rigid-base description and run a short restrained minimization to clean the backbone. Assign charges and radii with PDB2PQR using an AMBER parameter set. Solve the nonlinear Poisson-Boltzmann equation in APBS with two-level focusing, a coarse grid near 1 angstrom over the full fragment plus solvent padding and a fine grid near 0.35 angstroms over the central 21 bp. Sample the potential on cylindrical shells, partitioned into groove and backbone sectors by angular position relative to the helical axis, and integrate the ion densities to get the counterion atmosphere terms. Score only the central 21 bp and tile with overlap, because fragment ends carry large spurious potentials and a model trained on them learns edge artifacts. Report relative to a fixed reference sequence so the uninformative global offset does not dominate.

Fragment length follows the Debye length, which is the lever that makes the solver budget tractable. For a 1:1 electrolyte at 25 degrees the screening length is about 0.304 over the square root of ionic strength in nanometres, and B-DNA rises 3.4 angstroms per base pair, so the flank needed for roughly 2 percent residual contribution is about 36 bp at 10 millimolar, 16 at 50, 11 at 100, 9 at 150 and 7 at 300. A single fixed fragment length is therefore wrong at both ends of the condition range: too short to converge at low salt and several times larger than necessary at physiological salt and above. Fragment length and scored window are both set per condition, which raises labels per solve at high salt and protects convergence at low salt.

This also disposes of the tempting shortcut of training a surrogate on very short fragments. A 20 bp fragment scoring a central 6 bp leaves 7 bp of flank, which is roughly three screening lengths at 150 millimolar and less than one at 10 millimolar. A surrogate trained that way can reach a very high correlation against its own targets while those targets are themselves unconverged, which is fidelity to the wrong number and is invisible to every validation split.

The context requirement gets measured rather than assumed. Take about 2,000 fragments, solve each at 121, 61, 41, 31 and 21 bp, and plot how the central-window potential converges with flank length at every ionic condition. The estimates above come from linearized theory, and DNA sits in the nonlinear counterion condensation regime where that theory is only a guide, so the empirical convergence curve is what sets the production lengths. This costs on the order of tens of CPU-hours and settles a question that otherwise propagates into every downstream number.

Conditions are solved by anchoring and delta-learning, which is the larger saving. Rather than solving every fragment at every condition, solve the full set at two anchor conditions chosen to bracket the range, then solve a subset of 10 to 20 percent at the remaining conditions and learn the condition residual. The salt dependence of the potential is smooth and largely captured by screened-Coulomb scaling, so the residual is an easy learning problem with an analytic prior available, which is the same delta-learning move used elsewhere in the encoder. Against eight conditions this replaces roughly 8 solves per fragment with roughly 2.9, which is a factor near 2.8 on its own, and it composes with the length reduction at high salt.

One class of channel resists all of this and keeps a long-fragment set of its own. The integrated and window-level electrostatic quantities, meaning the electrostatic contribution to persistence length and the charge density integrated over 10 to 100 bp windows, are not reconstructable from tiled local potentials and need fragments long enough to contain the window. Keep a smaller set of long fragments for those alone rather than inflating the whole corpus to serve a handful of channels.

Conformational ensemble. DNA is not rigid and the ensemble-averaged potential is not the ground-state potential. For about 10 percent of fragments, draw 8 configurations from the cgNA+ Gaussian and solve each, supervising both the ensemble mean and the spread. The spread is a channel worth having on its own, since a locus whose electrostatics fluctuate widely is physically different from one rigidly fixed, and no existing feature set captures that.

Divalent conditions are approximate and labeled as such. Mean-field Poisson-Boltzmann treats magnesium poorly, because ion size, ion-ion correlation and site-specific binding all matter and none are in the theory. Magnesium conditions are run with a size-modified treatment, reported as approximate, and excluded from any claim requiring quantitative divalent accuracy.

The quantum chemistry, which works differently and is much cheaper than it sounds. These quantities are properties of short stacked fragments, not of long sequences, so the ground truth is computed once on an exhaustive fragment library and then generalized to flanking context by the encoder, which is the same move Deep DNAshape made for geometry applied to electronic structure. Three tiers, with delta-learning between them. Semi-empirical tight-binding across the broadest context coverage, including hexamer contexts around the central step. Dispersion-corrected density functional theory across all tetramer contexts at three stacking geometries, giving total interaction energies, frontier orbitals, ionization potentials, electron affinities, transfer integrals and site-specific molecular electrostatic potentials. Symmetry-adapted perturbation theory on a stratified subset, giving the decomposition into electrostatic, exchange-repulsion, induction and dispersion terms. The expensive tier supervises a correction to the cheap tier rather than being needed everywhere. The full library is a few thousand CPU-hours, which is modest next to the solver work, and it is reusable forever because it does not depend on any genome.

Sampling design, built for universality from the start. Six strata, about 200,000 fragments. The MPRA library sequences themselves, so every downstream task has exact coverage. Genomic windows stratified across promoters, enhancers, intergenic regions and repeats. A designed library covering all 4,096 hexamers in several fixed flanking contexts, which guarantees local coverage that genomic sampling alone will never give. Composition extremes at both tails. Windows from the full species panel: human, mouse, zebrafish, Drosophila, C. elegans, Arabidopsis, maize, S. cerevisiae, E. coli, Mycobacterium tuberculosis, Streptomyces coelicolor, Plasmodium falciparum, Dictyostelium discoideum, Methanocaldococcus jannaschii and Thermus thermophilus. That panel brackets base composition from roughly 19 to 72 percent GC and spans bacteria, archaea, fungi, plants and animals, and for the universality experiments most of it is held out of training entirely. Random sequence at several target compositions, as an off-manifold stratum.

Compute budget. Roughly 200,000 fragments across 9 ionic conditions is about 1.8 million nonlinear solves, plus the ensemble subset, landing in the range of 15,000 to 30,000 CPU-hours. The quantum chemistry adds a few thousand. This is the dominant cost of the project, it parallelizes trivially, and it is paid once.

External validation, which is the part that distinguishes a physics model from a solver imitator. Internal validation against held-out fragments, split at random and by sequence similarity, with per-channel R-squared and calibration curves. External validation against measurements the solver never saw: hydroxyl radical cleavage, which reports jointly on backbone solvent accessibility, minor groove width and minor groove electrostatic potential; measured photoproduct maps for the photochemistry channels; and measured melting data for the thermodynamic channels. Agreement with held-out solver output only shows the network imitates the solver. Agreement with independent measurement shows the physics is right about the world, and only the second supports a universality claim.

## Aim 2: CEILING, how much of regulation a physics bottleneck reaches

This measurement turns a representation claim into a result, and it is what makes the bottleneck framing falsifiable rather than rhetorical.

The quantity. For each dataset, train the best sequence model achievable under a fixed capacity and tuning budget, and the best READOUT head on the physics channels under the same budget. Report three numbers. The replicate reproducibility of the assay, which is the noise ceiling. The fraction of achievable variance the sequence model reaches. The fraction the physics head reaches. The ratio of the last two is the physics-reachable ceiling.

All three belong in the table, because the ratio alone is misleading. If a sequence model only reaches 40 percent of what the assay allows, then a physics head reaching 90 percent of the sequence model is reaching 36 percent of what the assay allows, and a reader who sees only the ratio will come away with the wrong impression. Reporting all three is also the direct answer to the objection that sank 31665, which was that it was unclear whether anything actually improved.

The bottleneck width sweep. Rather than reporting one number for about 1,670 channels, insert a learned linear projection to k dimensions between the landscape and the head, and sweep k across 4, 8, 16, 32, 64, 128, 256 and the full set. The resulting curve is a rate-distortion relation for regulatory prediction through physics, and it says directly how many physical degrees of freedom the regulatory signal actually needs. It also connects back to the effective rank measurement, since the two should agree, and a disagreement between them would be informative.

The decoder test. Train a decoder that reconstructs the underlying sequence from the channel stack, and report per-base accuracy and bits recovered out of the two bits per base available. This is the sharpest available test of whether the construction is a bottleneck at all, and as far as I can tell nobody has run it on biophysical feature sets. If sequence turns out to be reconstructable at high accuracy, the compression framing is wrong and the paper says so, and the project then rests on the dimensionality and the amortization arguments, which survive that outcome intact.

Calibrating controls. A ceiling number means nothing without knowing what the scale looks like at both ends, so two reference tasks are run alongside every dataset.

The positive control is in vitro nucleosome occupancy. Intrinsic nucleosome preference is known to be largely set by base composition and DNA mechanics, with GC content alone explaining close to half the variance. A physics atlas should nearly saturate this task. If it does not, the landscape is broken and no downstream number is trustworthy.

The negative control is splice site recognition. Splice site identity is base readout in the strictest sense and there is no physical reason a geometry and electrostatics stack should capture it. A physics head should fail badly. If it does well, something is leaking sequence identity into the channels, and the leak has to be found before anything else is reported.

The gap decomposition. The per-locus difference between the sequence model and the physics head is the object of study, not an error term. Stratify held-out loci by that gap and test what explains it: presence of specific transcription factor motifs, motif arrangement and spacing, CpG and methylation context, repeat class, and distance-dependent context. Fit the gap against those annotations and report which ones account for it. This is where the project says something mechanistic rather than numerical, and it connects naturally to the question of what sequence models encode beyond motif identity without re-running that analysis.

Datasets. The lentiMPRA sets in K562, HepG2 and WTC11, DeepSTARR in Drosophila S2, the Jores plant sets in Arabidopsis, maize and sorghum, the yeast promoter set, and Sharpr-MPRA. These are already assembled from prior work, which saves several weeks and means the comparison is against numbers that are already familiar.

The reachability spectrum, which is the headline figure. READOUT heads are cheap, so run one per functional readout and rank the readouts by how much physics reaches: chromatin accessibility, MPRA and STARR-seq activity, transcription factor occupancy, nucleosome positioning, transcription initiation, replication timing, recombination rate, local mutation rate, methylation state, CTCF loop anchors, splicing, RNA binding protein occupancy, translation efficiency and guide RNA efficiency. Normalize each against its own replicate reproducibility, because raw correlations are not comparable across assays. The output is a ranked map of which genomic functions are physics-reachable and which are not, with in vitro nucleosome occupancy expected at the top and splice site recognition at the bottom.

This is adjacent to one existing result and differs from it in a way worth stating. Prediction accuracy from sequence is already known to vary substantially across functional sequencing techniques, which establishes that some assays are more sequence-explainable than others. What has not been done is the same spectrum for physics, and more importantly the gap between the two. An assay where sequence reaches far but physics does not is one whose determinants are base readout. An assay where the two converge is one that physics explains. That per-assay gap is the mechanistic result, and it is what the single-dataset version of this measurement cannot produce.

Three arms, not two. The plan as stated compares physics-only against sequence-only, which answers the scientific question of what physics explains. It does not answer the engineering question of whether physics adds anything on top of sequence, and that is the first thing a reviewer will ask. Run sequence-only, physics-only, and sequence-plus-physics concatenated, all at matched capacity. The third arm can produce a positive engineering result even in the case where physics-only loses badly, which makes it cheap insurance against the main claim failing.

The formalism should be borrowed rather than invented. Treating this as a probing question with floors, ceilings and headroom gives a principled way to report how much a representation reveals about a target, separate from the assay noise ceiling, and it avoids an ad hoc ratio that a reviewer can argue with. Adopt that framing and cite it.

## Aim 3: universality, and the regimes where physics wins

In-distribution accuracy on one genome is the wrong place to look for a win, because the landscape is a lossy function of sequence and a sequence model has access to everything it encodes plus what it discards. Universality is where the argument is actually strong, and it is the first thing to measure.

The universality certificate. Train the human-only encoder. Evaluate per-channel accuracy on each of the fifteen panel genomes with no retraining and no fine-tuning. For each channel, define the universality gap as the spread between the best and worst genome. Report the full distribution of gaps, per family. Pre-register the thresholds: a median gap below 0.05 in R-squared counts as universal, above 0.15 counts as falsified, and the range between is reported as partial with the responsible channels named. Separately, regress per-genome accuracy on that genome's GC content. A significant slope is the signature of a model that learned composition statistics rather than the physics map, and it is the specific failure this test exists to catch.

The held-out-kingdom test. Train an encoder on bacterial and plant sequence only and evaluate it on human. If the sequence-to-physics map is what the model learned, this should barely register. If performance falls, the model learned something clade-specific, and the universality claim is dead in its strong form.

The synthetic-composition stress test, which is the decisive one. Generate sequences at compositions no natural genome reaches, from 5 to 95 percent GC, and at controlled dinucleotide compositions that do not occur in nature. Run the solvers and the quantum chemistry on them to get true labels, and evaluate the encoder. Memorization is impossible here, because there is no organism these could have come from. A model that holds up across this range has learned physics in a sense that admits no other explanation.

The asymmetry worth stating plainly in the paper. All of the above is possible because the ground truth is computable for any sequence from any organism or from no organism at all. A sequence-to-function model cannot be tested this way, because it would need matched functional assays in every species and at every composition, and those do not exist and will not. Universality is not merely a property this approach has and sequence models lack; it is a property only this approach can be measured for.

Function-level cross-species transfer. Train an activity head on human lentiMPRA and evaluate on Drosophila, maize, sorghum, Arabidopsis and yeast with no retraining, then do the same for a sequence model and for a k-mer ridge baseline at matched capacity. The k-mer baseline is essential, because cross-species transfer can be carried by base composition alone and a physics result a k-mer model matches is not a physics result. Sequence-similarity leakage between training and target sets gets quantified rather than assumed absent.

Composition shift and off-manifold sequence. Train on one GC stratum and evaluate on another in both directions. Train on genomic sequence and evaluate on random synthetic libraries, then reverse it. Sequence models degrade off the genomic manifold, and that manifold is where enhancer design operates, so this is practical rather than only a stress test.

Low-data learning curves. Subsample training sets to 100, 1,000, 10,000 and 100,000 examples and plot held-out performance for the physics head against the sequence model and the k-mer baseline. The claim is that the physics head starts higher and the sequence model overtakes it at some crossover, and that the crossover is high enough to matter for the many species and cell types where only small assays exist.

The condition axes as a capability. Because the encoder is conditioned, it can predict what a locus looks like under altered ionic strength, temperature, superhelical stress, pH or methylation, and a head can be asked how activity would be expected to shift. Validation data for most of these does not exist, so they are presented as predictions and as design knobs, labeled clearly as unvalidated, never as results. They are worth including because no sequence model has them and because they give a concrete target to anyone able to test them.

Attribution across families. What fraction of predicted activity rests on electrostatics, on electronic structure, on mechanics, on thermodynamics, on non-canonical structure, on reactivity. Answered with family-wise ablation and retraining, and with family-level Shapley values where the budget allows, rather than with permutation importance, which on a feature set this correlated attributes signal to whichever member of a correlated block happens to be permuted.

## Aim 4: design in physical coordinates

PANGEA and READOUT together give a design loop that sequence models do not have. Instead of optimizing in sequence space against a learned oracle, specify a target landscape in physical coordinates, then search for sequences whose landscape matches it. The design objective becomes a physical specification: this much minor groove negative potential at 150 millimolar, this bending stiffness over 50 base pairs, this duplex opening free energy under negative supercoiling, this quadruplex propensity suppressed. Those are statements about the molecule rather than about a model's opinion of the molecule.

Gradient-based search runs through PANGEA, which is differentiable, with the physical target as the loss. Discrete search runs with straight-through estimators or with a sampler over sequence space scored by the landscape distance. Both are standard, and the interesting part is the objective rather than the optimizer.

The circularity problem has a real solution here, and it is a consequence of using physics as the representation. An oracle that shares training data or architecture with the designer certifies nothing, which is why design results in this area are usually unconvincing. A physical target does not have that problem, because the ground truth for it is computable. Take a designed sequence, run the actual solvers and the actual quantum chemistry on it, and check whether its true landscape matches the specification. That is verification against first principles rather than against another network, and no sequence-space design method can do it.

Three independent checks on every design batch. First, the solver check just described, which asks whether the physical specification was met. Second, a published sequence-to-function model trained by other people on other data, which asks whether the designed sequence looks functional to an outside judge. Third, an entirely held-out reporter library never used at any stage of training, which asks whether the design generalizes beyond anything in the loop. A design result that passes all three is believable in a way that a self-scored one is not.

The design targets worth pursuing span the molecule classes. Regulatory elements with specified activity in a chosen cell type. Sequences that hold a given mechanical profile for nucleosome positioning. Quadruplex-forming and quadruplex-suppressed variants of the same promoter. R-loop prone and R-loop resistant versions of the same region. Guide RNAs and antisense oligonucleotides optimized on hybrid thermodynamics rather than on heuristic rules. Each of these is a physical specification that is checkable, which is the point.

## Compute accounting, done so it survives review

The cost model. Let c-phi be the cost of encoding one locus, c-head the cost of one task head on that locus, and c-seq the cost of one sequence model forward pass. For N loci evaluated against M tasks, the two-stage stack costs N times c-phi plus N times M times c-head, while the sequence model costs N times M times c-seq. The stack wins when M exceeds c-phi divided by the difference between c-seq and c-head.

That inequality is worth reading carefully, because it says the advantage does not depend on how many loci are evaluated. It depends on how many tasks are evaluated per locus. A workload of one assay in one cell type gains nothing from amortization regardless of scale. A workload of hundreds of cell types and assays over the same loci, which is what these models are actually used for, gains a great deal. The paper should present the crossover in M rather than a single speedup figure, because a single speedup figure invites a reviewer to find the workload where it is false.

Three regimes, with honest and different answers.

Genome-wide scanning across many assays and cell types is the best case. The atlas is computed once, cached, and every new task costs only a small head. This is where the amortization argument is strongest and where the claim should be made.

Variant effect prediction is the middle case. Each variant requires recomputing the encoder over a window around it, so the advantage shrinks to roughly the ratio of the encoder window to the sequence model's context, which is still large when the comparison is a megabase-context model, and much smaller against a compact one.

Gradient-based sequence design is the most expensive case, and the compute accounting should say so plainly. Optimizing a sequence requires backpropagating through the encoder, so design runs cost roughly what a sequence model costs per step and the amortization benefit largely disappears. The design aim stands on what it can specify and verify rather than on speed, and the two arguments should be kept apart so that neither is used to prop up the other.

Baselines. The comparison set is Enformer, Borzoi, Flashzoi, FlashRNA, Enigma, and Sei with low-rank approximation and quantization. Measuring against unoptimized Enformer alone is no longer a meaningful statement. For reference, published CPU inference times are about 1.87 seconds per sequence for Sei, 49.9 for Borzoi and 37.8 for Enformer, and low-rank plus quantization gives over fivefold speedup on Sei and more than a hundredfold against the larger models. AlphaGenome is available through an API rather than as weights, so it enters as a wall-clock comparison with that limitation stated, not as a FLOPs comparison.

What gets reported. FLOPs per forward pass, wall-clock on matched hardware for both GPU and CPU, peak memory, and the amortized workload curve as a function of M. Every number from the same machine, with the same batch handling, and with the measurement script released.

Storage, which is the part that is easy to forget and which the channel count makes acute. A full-precision landscape over a 3.1 gigabase genome at about 1,670 channels in 16-bit floats is roughly 10.4 terabytes, which nobody will download. Restricting to annotated candidate regulatory regions, around 2 percent of the genome, still gives about 207 gigabytes. The workable release is the projected form: 64 components in 8-bit covers the whole genome in roughly 198 gigabytes and the regulatory subset in about 4 gigabytes, and if the effective rank comes back in the tens, as expected, that projection loses almost nothing. The full landscape is then recomputed on demand rather than stored, which the encoder is small enough to make practical. The same structure applies per species, so the released multi-genome resource is a set of projected landscapes plus the encoder that regenerates any of them.

## Traps, controls, and stopping rules fixed in advance

Twelve ways this project fails quietly, each with the control that catches it.

| Trap | Why it bites | Control |
| --- | --- | --- |
| Ideal-geometry collapse | Fiber-model windows make the potential a function of composition, and the surrogate still reports a high R-squared | Verify that computed potential varies across sequences matched on base composition, before training anything |
| GC content dominance | Most channels track local composition, and composition alone explains roughly half the variance in intrinsic nucleosome occupancy | GC plus 5-mer ridge at matched capacity as the primary gate on every dataset |
| Similarity leakage | MPRA libraries and cross-species sets share sequence | Splits by sequence similarity and by chromosome, with the overlap quantified and reported |
| Capacity confounding | A weaker head makes the bottleneck look narrower than it is | Capacity sweep from ridge to MLP to CNN on both sides, matched tuning budget, matched seeds |
| Ignoring the noise ceiling | Raw correlations are not comparable across assays | Every result normalized against replicate reproducibility, with all three numbers reported |
| Protein-readout leakage | Occupancy channels reintroduce motif identity | Group G excluded from every headline number and reported only as a labeled variant |
| Correlated-feature attribution | Permutation importance misattributes across correlated blocks, and this was already raised against the earlier work | Group-wise ablation with retraining, plus group Shapley where affordable |
| Strand parity errors | An antisymmetric channel predicted with the wrong sign passes every correlation check | Per-channel parity verified against the 3DNA convention as a unit test |
| Channel count inflation | Condition and molecule class axes multiply the count without adding degrees of freedom, so 1,670 channels can hide a rank of 20 | Effective rank reported next to the count everywhere the count appears, and the bottleneck sweep run to full width |
| Fragment-library overfitting in the quantum chemistry | Context effects learned on short stacked fragments may not extend to genomic flanking sequence | Held-out hexamer contexts, plus validation against measured photoproduct and oxidative damage maps |
| Universality by construction | A model trained on many genomes will look universal on those genomes for trivial reasons | The headline universality number comes from the human-only and held-out-kingdom encoders, never from the multi-clade one |
| Lookup table collapse | The most dangerous one. Deep DNAshape notes that shape features can be approximated from mononucleotide and dinucleotide content, and its own webserver deploys the trained model as a sliding-window query table rather than as a network. If the landscape is table-reproducible there is no model here, and universality reduces to the trivial statement that lookup tables transfer | Per-channel comparison against pentamer, heptamer and 9-mer query tables, plus the randomized-table control used in the shape literature. Report the table-equivalence fraction openly. The defensible ground is the conditioned channels, the mesoscale channels and the molecule class axis, where a table would be combinatorially impossible |

Seven stopping rules, written down before the experiments run, so that a null result is a finding rather than a disappointment.

The composition gate. If the physics channels fail to beat GC plus 5-mer composition at matched capacity on at least three of five activity datasets, stop the ceiling arm. Write up the atlas as a resource and the composition result as the finding, which is a real negative bound that the field currently assumes away.

The positive control gate. If the physics head does not come close to the sequence model on in vitro nucleosome occupancy, the landscape is broken. Fix it before any other number is produced or reported.

The negative control gate. If the physics head performs well on splice site recognition, sequence identity is leaking into the channels. Find the leak before proceeding.

The compression gate. If a decoder reconstructs the underlying sequence from the channels above roughly 95 percent per base, drop the compression framing from the paper and keep the dimensionality and amortization results, which survive that outcome.

The speed gate. If the crossover in the number of tasks per locus lands above what real workloads run, drop the speed claim from the headline and report it as a workload-dependent property in the discussion.

The universality gate. If the median per-channel universality gap exceeds 0.15 in R-squared across the species panel, or if per-genome accuracy shows a significant dependence on target GC content, the strong universality claim is dead. Report it as partial universality with the responsible families named, and move the headline to the ceiling and dimensionality results. A negative universality result is still the first time anyone has measured this, and it is worth publishing.

The table gate. If high-order k-mer query tables reproduce most channels at the accuracy PANGEA reaches, PANGEA is not a model for those channels and should not be presented as one. Report the table-equivalence fraction per channel, keep the network only where it genuinely beats a table, and move the contribution to the channels that tables cannot reach: anything conditioned on the environment, anything mesoscale, and anything outside duplex B-DNA. A project that discovers its own feature set is a lookup table and says so is in far better shape than one that finds out in review.

## Deliverables and datasets

Three papers rather than one. Splitting them keeps each claim standing on its own evidence, and it avoids handing reviewers a long list of separate things to doubt in a single submission.

The first is the landscape and the universality result, which is the stronger of the two and should go out first. A condition-conditioned biophysical encoder covering about 1,670 channels across thirteen physical and chemical families and six molecule classes, validated against solver output and against independent measurement, with the universality certificate across fifteen genomes spanning 19 to 72 percent GC and all three domains of life, the held-out-kingdom test, the synthetic-composition stress test, and the measured intrinsic dimensionality of DNA biophysics. Released as a set of projected landscapes per genome plus the encoder that regenerates any of them. The claim stands without reference to any regulatory activity result. Natural homes are Nature Methods, Genome Research or Nucleic Acids Research, with ISMB as a presentation venue.

The second is the ceiling measurement: how much of regulation a physics bottleneck reaches. The ceiling numbers with their calibrating controls, the bottleneck width sweep, the decoder test, the gap decomposition, the transfer and low-data arms, and the workload-dependent compute analysis. This is a machine learning paper about what a representation can and cannot carry, and it fits a main track or a strong workshop.

The third is design in physical coordinates: specifying a target landscape, searching for sequences that realize it, and verifying the result against the solvers and against independent oracles. The verification protocol is the contribution as much as the designs are, because it is the first design evaluation in this area that can check its own specification from first principles rather than against a learned judge.

Software deliverables are the three encoders with weights, the solver and quantum chemistry generation pipelines as reproducible workflows, the projected landscapes per species, and the benchmarking scripts that produced every timing number.

| Dataset | Use | Stage |
| --- | --- | --- |
| APBS solutions on cgNA+ conformations, 150k fragments, 7 conditions | Electrostatics ground truth | Encoder |
| cgNA+, ABC and hexamer-scale MD libraries | Geometry and stiffness ground truth | Encoder |
| loop-seq cyclizability | Mechanics, experimental | Encoder |
| ORChID hydroxyl radical cleavage | Reactivity and solvent accessibility; also external validation | Encoder |
| Array Melt hairpin stabilities | Thermodynamics, experimental | Encoder |
| G4-seq, S1-END-seq, KAS-seq, permanganate and S1 footprinting | Non-B propensity, experimental | Encoder |
| In vitro nucleosome occupancy | Mechanics channel, and the positive control task | Encoder and ceiling |
| ENCODE4 lentiMPRA, K562 HepG2 WTC11 | Activity | Ceiling |
| DeepSTARR, Drosophila S2 | Activity, cross-kingdom transfer | Ceiling and transfer |
| Jores 2021, Arabidopsis maize sorghum | Activity, cross-species transfer | Ceiling and transfer |
| Yeast promoter library | Activity, synthetic sequence regime | Ceiling and transfer |
| Sharpr-MPRA | Activity, tiling design | Ceiling |
| Splice site annotations | Negative control task | Ceiling |
| HT-SELEX with ProBound free energy models | Group G protein readout, held apart | Variant analysis only |
| Three-tier quantum chemistry fragment library | Electronic structure ground truth: stacking decomposition, orbitals, couplings, site potentials, pKa | Encoder |
| CPD-seq and dCPD-seq photoproduct maps | Photochemistry channels, experimental, and an external validity check on the chemistry | Encoder |
| Fifteen-genome panel, 19 to 72 percent GC, all three domains | The universality certificate and the held-out-kingdom test | Universality |
| Synthetic sequence at 5 to 95 percent GC with computed labels | The composition stress test, where memorization is impossible | Universality |

## References

### Sequence to biophysics

- Li J, Chiu TP, Rohs R. Predicting DNA structure using a deep learning method. Nature Communications 15:1243, 2024. link
- Li K, Carroll M, Vafabakhsh R, Wang XA, Wang JP. DNAcycP: a deep learning tool for DNA cyclizability prediction. Nucleic Acids Research 50:3142, 2022. link
- Greenbaum JA, Pang B, Tullius TD. Construction of a genome-scale structural map at single-nucleotide resolution. Genome Research 17:947, 2007. link
- Bishop EP et al. A map of minor groove shape and electrostatic potential from hydroxyl radical cleavage patterns of DNA. ACS Chemical Biology 6:1314, 2011. link
- Petkeviciute D et al. cgDNA: a software package for the prediction of sequence-dependent coarse-grain free energies of B-form DNA. Nucleic Acids Research 42:e153, 2014. link

### Experimental biophysics at scale

- Ke Y, Sharma E, Wayment-Steele HK, Becker WR, Ho A, Marklund E, Greenleaf WJ. High-throughput DNA melt measurements enable improved models of DNA folding thermodynamics. Nature Communications 16:5572, 2025. link
- Tillo D, Hughes TR. G+C content dominates intrinsic nucleosome occupancy. BMC Bioinformatics 10:442, 2009. link
- Marsico G et al. Whole genome experimental maps of DNA G-quadruplexes in multiple species. Nucleic Acids Research, 2019. G4-seq human data at GEO accession GSE63874.
- Matos-Rodrigues G et al. S1-END-seq reveals DNA secondary structures in human cells.

### Protein readout

- Rube HT et al. Prediction of protein-ligand binding affinity from sequencing data with interpretable machine learning (ProBound). Nature Biotechnology 40:1520, 2022. link

### Activity data and sequence-to-function models

- Agarwal V et al. Massively parallel characterization of transcriptional regulatory elements. Nature, 2025. link
- Avsec Z et al. Advancing regulatory variant effect prediction with AlphaGenome. Nature 649:1206, 2026. link

### The efficiency frontier

- Flashzoi: an enhanced Borzoi for accelerated genomic analysis. link
- FlashRNA: An Efficient Model for Regulatory Genomics. bioRxiv, 2025. link
- Enigma: An Efficient Model for Deciphering Regulatory Genomics. bioRxiv, 2025. link
- When Less Is Enough: Low-Rank Structure in DNA Sequence-to-Function Models. bioRxiv, 2026. link

### Poisson-Boltzmann surrogates

- Jurrus E et al. Improvements to the APBS biomolecular solvation software suite. Protein Science 27:112, 2018.
- Achieving all-atom molecular dynamics accuracy from the Poisson-Boltzmann method through machine learning (DIS-PB). Journal of Chemical Physics, 2025. link
- End-to-End Modeling of Reaction Field Energy Using Data-Driven Geometric Graph Neural Networks (PBGNN). link

### Base stacking energetics and electronic structure

- Sponer J, Riley KE, Hobza P. Nature and magnitude of aromatic base stacking in DNA and RNA: quantum chemistry, molecular mechanics, and experiment. Physical Chemistry Chemical Physics, 2008. Establishes the decomposition into electrostatic, dispersion and short-range repulsion, and the sensitivity of the balance to sequence context.
- Twist-dependent stacking energy of base-pair steps in B-DNA geometry: a density functional theory approach. Reports total step stacking energies spanning roughly minus 9.9 to minus 14.7 kcal per mole across sequence.
- Aida M. An ab initio molecular orbital study on the stacking interaction between nucleic acid bases. International Journal of Quantum Chemistry, 1986. Early demonstration that electrostatic energy drives the sequence dependence while charge transfer and dispersion carry the overall stability.

### Charge transport and oxidative damage

- Giese B, Amaudrut J, Kohler AK, Spormann M, Wessely S. Direct observation of hole transfer through DNA by hopping between adenine bases and by tunnelling. Nature 412:318, 2001. The crossover from superexchange tunnelling to thermal hopping beyond about three intervening base pairs.
- Meggers E, Michel-Beyerle ME, Giese B. Sequence dependent long range hole transport in DNA. Journal of the American Chemical Society, 1998.
- Sequence dependence of charge transport through DNA domains. Journal of the American Chemical Society 127:17445, 2005. Conformationally gated transport through transiently delocalized domains.

### Photochemistry with genome-wide measurement

- Mao P, Smerdon MJ, Roberts SA, Wyrick JJ. Chromosomal landscape of UV damage formation and repair at single-nucleotide resolution. PNAS 113:9057, 2016. link
- Laughery MF, Stark B, Morledge-Hampton B et al. Genome-wide maps of CPD deamination in yeast reveal the impact of DNA sequence context and nucleosome architecture on cytosine deamination rates. Genome Research, 2025. link
- Wilson HE, Wyrick JJ. Genome-wide impact of cytosine methylation and DNA sequence context on UV-induced CPD formation. Environmental and Molecular Mutagenesis 65:14, 2024. Methylation raises CPD formation close to twofold with the magnitude set by flanking context.
- Elliott K, Bostrom M et al. Elevated pyrimidine dimer formation at distinct genomic bases underlies promoter mutation hotspots in UV-exposed cancers. link

### Baselines, ceilings and the table threat

- Li J, Rohs R. Deep DNAshape webserver: prediction and real-time visualization of DNA shape considering extended k-mers. Nucleic Acids Research 52:W7, 2024. link The webserver deploys the trained model as sliding-window query tables rather than as a network, which is the clearest existing evidence that shape channels can collapse to a table.
- Zhou T et al. Quantitative modeling of transcription factor binding specificities using DNA shape. PNAS, 2015. link Source of the randomized-shape-table control, which should be reused here.
- Prediction accuracy of regulatory elements from sequence varies by functional sequencing technique. Frontiers in Cellular and Infection Microbiology, 2023. link The sequence-side version of the reachability spectrum, and the thing this project must differentiate against.
- How High Is 0.6? Floors, Ceilings, and Headroom in Interpretability Probing. arXiv 2610.08544. link A ceiling defined by how much the available information reveals about the target, as distinct from a noise ceiling. This is the formalism the CEILING analysis should adopt instead of an ad hoc ratio.
- Robust Machine Learning for Regulatory Sequence Modeling. arXiv 2601.14969. link Regulatory sequence models under GC-content shift and realistic assay noise. Adjacent to the composition-shift arm and worth citing as the sequence-side comparison.


---

## Amendments (revision 3)

*Made at the user's direction during the stage-1 build; each traces to measurements in
`DECISIONS.md`. Where an amendment and the body disagree, the amendment governs.*

### A1. Ground truth, second generation

The stage-1 corpus is regenerated with all eight implemented families (B, C, D, E, F, G, H,
L), parameters sourced where a published source exists (nearest-neighbour heat capacities,
the complete Z-Hunt table, the DNA fold model, the B-Z junction energy), a two-start
restrained minimisation (the plain start trapped 20 percent of backbones in an alpha/gamma
crankshaft basin 288-790 kJ/mol above the canonical one), a relaxed cgNA+ ensemble of 16
snapshots on 10 percent of fragments, and 3,200-sample ensemble spreads (200 samples put a
sampling-noise ceiling of R-squared 0.40 on the worst channel; 3,200 put it at 0.92).
Channels without a published parameterisation stay provisional, are excluded from
training and headline numbers, and are listed with their reasons.

### A2. Data is a scaling axis, not only parameters

The scaling study is extended from parameters to data. Analytic targets are cheap for most
families, and two channel groups were measured to be data-limited rather than
capacity-limited: Family L (RNA folding), which overfits at every size from 2M to 1B, and
the whole-sequence channels, which saturate near R-squared 0.995 from 10M parameters (each
fragment gives one training sample of a whole-sequence channel against 61 of a per-bp
one). Auxiliary windows -- new genomic windows from the same panel, leakage-controlled by
the main split's own similarity criterion -- carry whatever families are cheap to compute
and are mixed into training with all other channels masked. First test: Family L only, 1M
windows, at 10M and 50M.

### A3. Architecture

- Attention layers are part of every rung; attention heads are biased by a 2-D
  convolution over the base-pair map (Family L per-bp 0.544 -> 0.674 at matched epochs).
- Whole-sequence channels read out through mean, max and attention pools and an MLP
  (extremes and locations, which a linear readout of a mean cannot represent).
- Window outputs are the window average of the per-bp output, not separate heads (median
  R-squared 0.95-0.96 -> 0.989-0.996, decided on validation).
- The head rank is a measured quantity (the bottleneck sweep: saturation at 128 for
  whole-sequence channels and 256 per base pair); ladder rungs use min(width, 512).
- The loss is beta-NLL (beta 0.5), family-balanced; Kendall-Gal family weights are dropped
  (ill-posed on a negative NLL).
- Family L gets a folding-aware pair module: a pair representation updated by attention
  along the rows and columns of the pair map (`--pair-module axial`), supervised by the
  base-pair probability matrix that the targets' own folding engine computes (training
  targets only, never an input), with the matrix's exact functionals (paired probability,
  positional entropy) fed back to the trunk. Folding is combinatorial pairing that a
  sequence trunk with a static pair bias learns only partially, and every per-base L
  channel is a functional of that matrix.
- Stranded channels whose two strands are the same molecule type read the minus strand
  from the reverse-complement pass (`--strand-tie`, Family L): L_m(x) = L_p(rc x) exactly,
  in the stored frame, so both strands train one predictor. (The equivariance merge
  previously handled only symmetric and antisymmetric channels.)

### A4. Evaluation

Every headline number is on a held-out test side (5 percent, similarity-grouped) that no
training or model-selection decision sees; validation alone selects epochs and readouts.
Reported with it: calibration of the predicted variances, rare-event channels separately
(with an event-detection AUROC), interior-position R-squared beside all-position R-squared,
and per-species and per-stratum breakdowns. The universality gap is reported as five
measures fixed before the larger encoders reported, so no single metric can be chosen
after the fact. The synthetic-composition stress test scores natural-genome encoders on
composition extremes, random sequence and designed libraries.

### A5. What the measurements changed in the claims

- The prediction that universality degrades with capacity is not borne out from 2M to
  200M for either universality encoder; the claim is restated as measured (DECISIONS.md).
- Generalisation to unseen species is governed by base composition rather than phylogeny
  at small sizes, and that dependence weakens with capacity.
- Natural-genome encoders fail only on whole-sequence channels of designed low-complexity
  libraries; training on such libraries removes the failure, which is the case for keeping
  them in the released encoder's corpus.

### A6. Decisions from the Family L arms and the ground-truth checks (Nov 20, afternoon)

- **Strand tie in.** Family L's minus strand is read from the reverse-complement pass's
  plus strand (exact). Test L per-bp 0.658 -> 0.784 at 10M, the largest single gain on
  the weakest family, at no data cost; its 10M cost to Family F's rare-motif channels is
  measured across capacity by the v4 ladder.
- **Axial pair stack and base-pair supervision out of the final encoder.** Supervision
  with the folding engine's base-pair matrix rescues the axial stack (+0.14) but only to
  the level of the convolutional pair bias with the tie, at five times the cost, and does
  not help the convolutional pair bias. Both stay as measured options.
- **Whole-sequence channels are read from their own heads; windows from the pooled
  per-bp output.** Linear functionals of per-bp channels can be derived, nonlinear ones
  (maxima, spreads) are better learned directly (RESULTS 17, 20).
- **Ground truth v2.1.** Family G whole-sequence strand statistics are taken over
  interior positions: the minimised duplex's terminal residues differ between strands
  (C(x) != W(rc x) by up to 6.6 spreads at the ends, agreement inside). Strand-pair
  channels in the structural families are not tied, since that symmetry does not hold in
  the targets.
- **Noise-limited channels are named as such.** The 16-snapshot channels sit at their
  measured replicate ceilings (median gap 0.015); improving them requires more
  snapshots per fragment, which is scheduled with the next ground-truth revision.
- **Final encoder of this round:** 200M, convolutional pair bias + strand tie +
  whole-sequence auxiliary windows + v2.1 targets (`v4f_L200M`, pre-registered in
  DECISIONS.md).

### A7. Aim 2 starts before the final encoder (Nov 20, evening)

- The READOUT machinery (`src/readout/`) is built and validated with a frozen development
  encoder (the 10M rung with the strand tie) while the GPU partitions are saturated, and
  every Aim 2 number is recomputed with the final encoder before it is reported as the
  result. Featurisation is CPU-feasible for the calibration sets; lentiMPRA, DeepSTARR and
  the yeast set need GPU-time featurisation (or features computed inside the head's
  training loop) and wait for it.
- Landscapes of sequences longer than 61 bp are defined window by window (61-bp windows,
  31-bp stride, centre-weighted stitching), since the encoder only knows 61-bp molecules.
- In vitro nucleosome occupancy (the plan's positive control) is not ingested; loop-seq
  intrinsic cyclizability (a mechanics readout of the same family) calibrates the
  physics end of the scale until it is.
- The decoder test is run first: if bases are recoverable from the landscape, a strong
  physics head on the splice-site negative control would be the decoder at work, and
  the plan's leak interpretation applies to the whole stack rather than to a channel.
- Assay ceilings are taken only from stated replicate correlations in the source
  publications (loop-seq: technical replicates, r = 0.869); values drawn only in figure
  images are not transcribed.

### A8. Aim 2's bottleneck is measured in bits of sequence, not channels (Nov 20, evening)

The decoder test came first and it decided the design: from the per-bp landscape a
4-channel linear projection recovers every base (100 percent, 2 bits per base), and so
do moving averages of it over 5-20 bp. A physics-only head therefore equals a
sequence-only head on any assay, negative control included (splice sites: 0.976 both),
and the plan's channel-width sweep (k = 4 ... 256) constrains nothing. Changes:
- The bottleneck sweep runs over representations that limit the *numbers per base
  pair* -- non-overlapping bin means of W bp (W = 5, 10, 20) through a learned k-channel
  projection (k = 1, 4, all) -- and over the whole-sequence channels alone.
- Every representation is scored twice: its READOUT score on the assay, and its decoder
  score (bits of sequence recoverable per base). The rate-distortion relation the plan
  asks for is drawn with that measured rate on the horizontal axis.
- The physics claim for an assay is made where a representation that carries little
  recoverable sequence still reaches a large fraction of the assay's ceiling, and the
  negative control must fail at that same rate. The full-landscape comparison is still
  reported, as the plan's three numbers, but it is read as a statement about the
  representation, not about physics.
- The "compression framing" sentence of the plan is answered: the per-bp landscape is
  not a compression of sequence; information-limited summaries of it are, and their
  rate is measured rather than assumed.

### A9. What Aim 2 claims after calibration (Nov 21)

Measured on three calibration assays with a development encoder (RESULTS 25), to be
recomputed with the final encoder before it is reported:
- **The compression framing is replaced by an invertibility result.** Every substantive
  per-bp physical description -- the encoder's 286 channels, any single family except
  the methylation deltas, the four classic shape features -- lets a decoder recover every
  base. Physics-only and sequence-only heads are therefore equal on every assay, and the
  plan's "physics-reachable ceiling" ratio is reported as ~1 for the full landscape with
  that explanation, never as a physics finding.
- **The bottleneck curve is reported as resolution dependence.** Coarse-grained physics
  (bins, few channels, whole-sequence summaries; rate measured in decoded bits per base)
  ranks assays by how much of the readout survives losing fine spatial detail. On the
  calibration set the ranking is splice classes > plant promoter strength > DNA
  cyclizability, the reverse of the plan's expected physics ranking, and is explained by
  resolution (helical phasing) rather than by physics.
- **The engineering arm stays** (sequence + physics against sequence, matched head):
  +0.010 / +0.002 / -0.001 on the three sets so far; reported with seeds before any claim.
- **What the project rests on**, per the plan's own contingency: the dimensionality of
  the landscape (effective rank, interpretable coordinates in which a readout is
  expressed) and amortisation (the encoder reproduces 571 physics computations, some
  taking minutes per fragment, at R2 0.9997 per base in a forward pass). The negative
  control needs composition-matched decoys to be a strict base-readout test.
