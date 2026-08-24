# ReactFlow-Delta Model Rescue v6 design

## 1. Focal question and evidence boundary

**Decision question:** can WT 2A3 reactivity, used as a fixed soft constraint on both WT and exact-mutant thermodynamic ensembles, expose a mutation-response signal large enough to justify a new neural residual model?

**Located evidence:** v5 found a small but unusually consistent sequence-only ensemble signal: signed-delta MAE improved by 0.634%, its paired 95% CI was [0.001396, 0.001994], and all 20 puzzles were positive. It nevertheless failed the frozen 1% eligibility threshold, so no v5 neural model was trained. OpenKnot documents the assay as normalized 2A3 SHAPE-MaP and states that reactivities are normalized so the dataset 90th percentile is 1.0. The 2A3 study reports that 2A3-derived restraints can improve structure prediction. Deigan SHAPE pseudo-energies and ViennaRNA partition-function soft constraints provide a fixed implementation path.

**Assumption:** sequence-only thermodynamics is too weak because the designed pseudoknotted constructs occupy experiment-specific conformational landscapes not fully captured by the nearest-neighbor model. The allowed WT profile can anchor that landscape without using a mutant outcome.

**Prediction:** constrained exact-mutant-minus-WT features add at least 1% signed-delta MAE improvement beyond the already-consumed sequence-only v5 features under 20-puzzle LOPO.

**Falsifier:** the fixed incremental ridge Gate fails, or a later primary residual cannot beat corrected B1 and identical-capacity controls on both point and probabilistic metrics.

## 2. Alternatives considered

### A. WT-2A3-constrained differential ensemble — selected

For each puzzle-method construct, clamp finite negative WT reactivities to zero, encode missing positions as `-999`, and prepend the required one-based dummy value. Apply the same normalized WT field to the WT and every exact-mutant `RNA.fold_compound`. Use ViennaRNA 2.7.2, Deigan `m=1.8`, `b=-0.6`, `RNA.OPTION_PF`, 37°C, no hard constraints and no pseudoknot post-processing. The identical constraint field is essential: removing the edit-site constraint only for mutants would confound mutation effects with constraint removal. The all-missing P20 Eterna cell must reduce exactly to the unconstrained ensemble.

### B. Full-response latent manifold — deferred

A nonlinear decoder could model receiver-receiver covariance more explicitly than the independent point head. It is potentially powerful, but overlaps with prior low-rank work and has no cheap, target-independent eligibility feature. It remains a v7 candidate only if v6 fails.

### C. Hurdle direction/magnitude model — rejected

SparseDelta and MeanAligned already tested closely related sign/magnitude/calibration decompositions. Their CRPS improvements did not translate into reliable signed-delta mean gains. Reopening this family would repeat a falsified mechanism.

## 3. Architecture and attribution

The V6M2 baseline is the fixed v5 ridge feature set: 18 geometric/WT-context covariates plus the 12 unconstrained ensemble-delta features. The v6 probe adds exactly 12 constrained ensemble-delta features. It uses train-only weighted standardization and ridge alpha 1, with position → mutant → puzzle-method cell weighting. A complete 20-fold prediction universe must exist before target join.

Only an exact eligibility PASS opens neural work. The primary model then trains corrected B1 identically, freezes it, and trains a zero-initialized residual head on detached B1 source-receiver features plus 24 structure features. Two mandatory, non-selectable controls use the same head dimensions and budget: a zero-structure capacity null and an unconstrained-only control padded with twelve zero channels. The primary receives unconstrained plus constrained features. This makes a primary-versus-control difference attributable to experimental constraint information rather than head capacity.

The mean stage uses exact method-balanced signed-delta L1. Once frozen, a two-Gaussian zero-mean residual calibrator is fitted by closed-form mixture CRPS; calibration cannot alter the point mean. Candidate selection, feature selection, Deigan parameter search, width search, epoch search, rank search and best-seed reporting are prohibited.

## 4. Adversarial review

**Alternative explanation:** WT reactivity can dominate the constrained ensemble and simply restate the WT input already available to B1. The incremental probe therefore compares against both direct WT covariates and unconstrained physics; a sub-1% result terminates the route.

**Measurement risk:** Deigan parameters were not optimized on this OpenKnot assay. Searching them on development outcomes would create hyperparameter leakage, so v6 uses ViennaRNA defaults and treats failure as informative.

**Physics risk:** ViennaRNA secondary structures do not represent pseudoknots in the partition ensemble. WT 2A3 constraints may still improve local canonical pairing probabilities, but v6 cannot claim pseudoknot mechanism without independent evidence.

**Negative-value risk:** OpenKnot states that negative reactivities arise from experimental error. SHAPE guidance treats negative peaks above the missing-data sentinel as zero, while missing values remain unconstrained. v6 freezes that transformation before outcome access.

**Publication risk:** even a 5% internal gain remains post-hoc development evidence. No v6 result establishes external replication, SOTA, mechanism or publication readiness; a new sealed external amendment would still be required.

## 5. Sources checked

- [OpenKnotAIDesignData official repository](https://github.com/eternagame/OpenKnotAIDesignData)
- [2A3 SHAPE reagent study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8034653/)
- [Deigan et al. SHAPE-directed folding](https://pmc.ncbi.nlm.nih.gov/articles/PMC2629221/)
- [ViennaRNA SHAPE reactivity API](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/probing/SHAPE.html)
- [ViennaRNA Python API](https://viennarna.readthedocs.io/en/latest/api_python.html)
- [SHAPE negative and missing reactivity handling](https://pmc.ncbi.nlm.nih.gov/articles/PMC2941709/)
