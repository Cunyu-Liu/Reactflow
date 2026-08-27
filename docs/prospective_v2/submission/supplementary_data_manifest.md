# ReactFlow-Delta prospective-v2: Supplementary Data Manifest

> Auto-generated from locked artifacts. Generated: 2026-08-14.
> **SUPERSEDED HISTORICAL MANIFEST — NOT SUBMISSION-READY.** Claim boundary:
> `POST_HOC_DEVELOPMENT_ONLY / EXTERNAL_NOT_ESTABLISHED /
> MECHANISM_NOT_ESTABLISHED / SOTA_NOT_ESTABLISHED / PUBLICATION_NOT_READY`.
> V13M3 is terminal FAIL and V14 has no terminal scientific verdict. Legacy P4/P5/P5b
> external scores are alignment-invalid and are retained only as audit artifacts.

## S1. Data availability

### Development dataset
- **Source**: [OpenKnot M2 official release](https://github.com/eternagame/OpenKnotAIDesignData)
- **File**: OK7a_M2_data.v4.5.2.csv
- **Role**: PRIMARY_PUBLIC_DEVELOPMENT
- **Cells**: 160 intended (20 puzzles x 8 methods, same public activity/sequencing release)
- **WT constructs**: 160
- **Registered exact SNV**: 13,976
- **Chemistry**: 2A3-MaP
- **Split**: 20-fold LOPO-puzzle (split_v4_lopo_puzzle)
- **Provenance**: OpenKnot AI Design Data, EternaGames community project

### Historical external exploratory datasets (not confirmatory)
- **Source**: RMDB (https://rmdb.stanford.edu/), Ribonanza M2-style 2A3 datasets
- **Datasets**: M2SL5_2A3_0000 (betacoronavirus SL5), M3SARS_2A3_0000 (coronavirus frameshift elements), 15KLIB_2A3_0000 (diverse: TTR, SAM riboswitch, SARS windows, HDV)
- **Components**: 24 (each = 1 WT anchor + its single-SNV mutant library)
- **Single-SNV mutants**: 3,237
- **Development disconnect**: Zero sequence identity overlap with OK7a_M2 development set
- **Chemistry**: 2A3-MaP (same family as development)
- **Platform**: NovaSeq (Ribonanza) vs Ultima (OpenKnot M2); recorded, not concatenated
- **Normalization**: RNAFramework per-dataset; recorded per component
- **Qualification**: Legacy P4/P5/P5b features and scoring positions were affected
  by a seqpos-alignment defect. The old scores are `LEGACY_ALIGNMENT_INVALID` and
  cannot support external transportability, calibration, or mechanism claims.

### Raw data files (remote archive)
- /mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_20260730/ -- parsed rdat files
- /mnt/cunyuliu/reactflow_delta_raw/rmdb/entries_20260730/ -- raw entries
- Available from RMDB on request

## S2. Primary result artifacts

| Phase | Artifact | Path |
|-------|----------|------|
| P2 | Direct effects (20-puzzle) | docs/prospective_v2/p2_direct_v2_result_20260813.json |
| P2 | Held position rows (raw) | /mnt/cunyuliu/prospective_v2_p2_preds_20260813/p2_held_position_rows.jsonl (975,599 rows) |
| P2 | Secondaries (region/distance/calibration) | /mnt/cunyuliu/prospective_v2_p2_preds_20260813/p2_secondaries_report.json |
| P3 | LRSO effects (spec-compliant v3; v1/v2 retracted) | docs/prospective_v2/p3_lrso_v3_result_20260815.json |
| P4 | Legacy external components (`LEGACY_ALIGNMENT_INVALID / NOT_CITABLE`) | /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_components.json (90 MB) |
| P4 | Legacy external result (`LEGACY_ALIGNMENT_INVALID / NOT_CITABLE`) | /mnt/cunyuliu/prospective_v2_p4_20260813/p4_external_result.json |
| P4 | Legacy calibration (`NOT_CURRENTLY_QUALIFIED`) | /mnt/cunyuliu/prospective_v2_p4_20260813/p4_calibration_result.json |
| P5 | Legacy mechanism result (`MECHANISM_NOT_ESTABLISHED / NOT_CITABLE`) | /mnt/cunyuliu/prospective_v2_p4_20260813/p5_mechanism_result.json |
| P6 | Historical replay report (artifact consistency only) | /mnt/cunyuliu/prospective_v2_p6_20260814/replay_report.json |
| P6 | Main tables | /mnt/cunyuliu/prospective_v2_p6_20260814/out/main_tables.md / .tex |
| P6 | Figures (4) | /mnt/cunyuliu/prospective_v2_p6_20260814/out/figures/fig1..fig4.png |
| P6 | Model/Data/Code cards | /mnt/cunyuliu/prospective_v2_p6_20260814/out/cards.md |
| P6 | Environment spec | /mnt/cunyuliu/prospective_v2_p6_20260814/out/environment.yml |

## S3. Primary results summary

| Estimand | Value | 95% CI | Verdict |
|----------|-------|--------|---------|
| P2: D_p = CRPS(T*) - CRPS(Direct*) | +0.0127 | [+0.0079, +0.0175] | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE |
| P3: LRSO rank 2 vs B* (v3 spec-compliant) | +0.0147 | [+0.0119, +0.0175] | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE |
| P3: LRSO rank 4 vs B* (v3 spec-compliant) | +0.0155 | [+0.0113, +0.0196] | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE |
| P3: LRSO rank 8 vs B* (v3 spec-compliant) | +0.0154 | [+0.0122, +0.0185] | HISTORICAL_POST_HOC_DEVELOPMENT_EVIDENCE |
| P4: Legacy component-macro D vs zero | +0.0410 | [+0.0153, +0.0667] | LEGACY_ALIGNMENT_INVALID / NOT_CITABLE |
| P4: Legacy FWER | historical pass |  | LEGACY_ALIGNMENT_INVALID |
| P4: Legacy calibration cov95 | 0.874 | [0.85, 0.99] | NOT_CURRENTLY_QUALIFIED |
| P5: Legacy edit-site - very-far heterogeneity | -0.0090 | [-0.0199, +0.0019] | MECHANISM_NOT_ESTABLISHED / NOT_CITABLE |
| P5: Legacy permuted negative control | -0.1107 | [-0.159, -0.062] | LEGACY_ALIGNMENT_INVALID / NOT_CITABLE |

## S4. Attrition ledger

| Level | Count | Status |
|-------|-------|--------|
| Intended cells (20x8) | 160 | CONFIRMED_FACT |
| Historical oracle-analyzable | 158 | REPOSITORY_REPORTED_NOT_REPLAYED |
| P2 primary (20 puzzles, all finite D_p) | 20 | CONFIRMED |
| P4 K_preaccess (outcome-blind) | 24 | CONFIRMED |
| P4 K_eff_realized (post-attrition) | 24 | CONFIRMED |

## S5. Code availability

- **Repository**: https://github.com/Cunyu-Liu/ReactFlow
- **Submission branch/commit**: `NOT_FROZEN` pending terminal V14 qualification and
  current claim audit. Historical `13d34ac` is not a submission reference.
- **Entry points**:
  - P2: scripts/reactflow_delta/run_p2_direct_v2.py
  - P3 historical valid implementation: scripts/reactflow_delta/run_p3_lrso_v3.py
  - P4/P5 historical external entry points: audit-only; current external authority denied
  - P6 replay: scripts/reactflow_delta/run_replay_v1.py (default retained P2/P3 only;
    external routes require exact active authority and are currently denied)
  - P6 tables/figures: scripts/reactflow_delta/generate_p6_tables_figures_v1.py
  - P6 cards: scripts/reactflow_delta/build_p6_cards_v1.py
- **Replay**: `python -m scripts.reactflow_delta.run_replay_v1` provides internal
  retained-artifact consistency only; it is not current scientific qualification.
- **Test suite**: pytest tests/reactflow_delta/
- **Environment**: out/environment.yml (conda)
