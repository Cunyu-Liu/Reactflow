# ReactFlow-Delta prospective-v2: Declaration Statements

> Auto-generated from locked artifacts. Generated: 2026-08-14.

## 1. Data availability

The development dataset is the OpenKnot M2 official release, freely available at
https://github.com/eternagame/OpenKnotAIDesignData. The external confirmatory
dataset is sourced from the RNA Mapping DataBase (RMDB, https://rmdb.stanford.edu/)
Ribonanza M2-style 2A3 datasets (M2SL5_2A3_0000, M3SARS_2A3_0000, 15KLIB_2A3_0000).
All processed data artifacts are available from the authors upon reasonable request.
Raw rdat files are available from RMDB.

## 2. Code availability

All code for data processing, model training, evaluation, and reproducibility
is available at https://github.com/Cunyu-Liu/ReactFlow
(branch: codex/reactflow-delta-prospective-v2-20260813, commit: 13d34ac).
A one-click replay script is provided at scripts/reactflow_delta/run_replay_v1.py.

## 3. Competing interests

The authors declare no competing interests.

## 4. Author contributions (placeholder for human author assignment)

- Conceptualization: [to be assigned]
- Methodology: [to be assigned]
- Software: [to be assigned]
- Validation: [to be assigned]
- Formal analysis: [to be assigned]
- Investigation: [to be assigned]
- Resources: [to be assigned]
- Data curation: [to be assigned]
- Writing (original draft): [to be assigned]
- Writing (review and editing): [to be assigned]
- Visualization: [to be assigned]
- Supervision: [to be assigned]
- Project administration: [to be assigned]
- Funding acquisition: [to be assigned]

## 5. Claim boundary

This study reports a prospective, outcome-blind benchmark of full-spectrum
single-nucleotide mutation response prediction in 2A3-MaP mRNA structures.
The following claims are made:

**Established**:
- Direct learnability of mutation response from WT profile + exact SNV on
  development data (P2, 20-puzzle LOPO).
- LRSO adds no incremental skill over the strongest direct baseline (P3).
- External statistical transportability to development-disconnected 2A3-MaP
  components (P4, 24 components, 3,237 single-SNV, zero sequence overlap).
- The direct-vs-WT-anchor advantage is feature-dependent (negative control
  passes) and replicates across biological regions.

**Not established** (explicitly excluded from claims):
- Edit-site-concentration mechanism (P5: MECHANISM_NOT_ESTABLISHED).
- Practical/material importance (no independent delta_practical).
- Domain SOTA (requires fair direct comparators, qualified external exposure,
  and task-identity alignment not yet completed).
- Experimental ordering utility (requires practical PASS).

## 6. Limitations

- Gaussian predictive with frozen scale 0.3; empirical residual SD (0.61)
  exceeds nominal scale, indicating systematic underdispersion.
- Signed-delta point MAE is negative vs no-change anchor; the CRPS advantage
  is tail-driven, not mean-shift-driven.
- Single seed at deployment; five-seed ensemble is the deployment target.
- Development-only held-out: 20 puzzles from a single public dataset.
  Generalization to other 2A3-MaP datasets, other probes, or other organisms
  is not established.
