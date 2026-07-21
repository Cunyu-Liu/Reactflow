# ReactFlow Data Governance

This document is the paper-audit ledger for ReactFlow data.  It maps every
training/evaluation input to a public source, then records the preprocessing
checks needed before a run can enter a SOTA comparison table.

## Public Sources

| Source | Verifiable link | Role in ReactFlow | Required local evidence |
|---|---|---|---|
| RNAndria / eFold Dryad dataset | `https://doi.org/10.5061/dryad.79cnp5j95` | Main public 2D-structure corpus and benchmark tiers. The Dryad page lists `efold_train.json`, `archiveII.json`, `PDB.json`, `viral_fragments.json`, `lncRNA_nonFiltered.json`, `pri_miRNA.json`, `human_mRNA.json`, and its README. | JSONL cache files under `artifacts/full_runs/*/cache/`; `paper_artifact_audit.json/md` must show nonzero row counts. |
| eFold paper | `https://doi.org/10.1126/sciadv.adz4967` | Baseline protocol and cited public benchmark numbers. | README SOTA table keeps cited metrics separate from local ReactFlow metrics. |
| RibonanzaNet2 Kaggle model | `https://www.kaggle.com/models/shujun717/ribonanzanet2/PyTorch/alpha/1` | Frozen encoder only. The model card reports an alpha PyTorch release with roughly 100M parameters trained on RNA DMS/2A3 chemical-mapping profiles. | `frozen/ribonanzanet2_sharded_full/sharded_manifest.json` plus checkpoint SHA256 `c94031719c8a1c70a9068d5de861f65083cdf0555a15570b3724a8d6d7750e35`. |
| Rfam clan metadata | `https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/clan_membership.txt.gz` | Family/clan leakage control. | `metadata/rfam_current_mmseqs_metadata.manifest.json` must show `cluster_method="mmseqs"` and `mmseqs_error=null`. |

The project does not silently download Kaggle assets from inside the import
graph.  Kaggle data and model files require explicit user consent and an
authenticated CLI session; the resulting artifacts are then hashed and consumed
through deterministic local paths.

## Current Full-Scale Evidence

Latest audited full run root:

```text
artifacts/full_runs/full_ablation_20260709_003012
```

Current public cache evidence from `paper_artifact_audit.md`:

| Cache | Rows |
|---|---:|
| `efold_train.jsonl` | 307,641 |
| `archiveII.jsonl` | 2,052 |
| `PDB.jsonl` | 333 |
| `viral.jsonl` | 97 |
| `lncRNA.jsonl` | 289 |
| `human_mRNA.jsonl` | 6,627 |

The final split evidence is `splits/rfam_current_mmseqs_seed0/split_manifest.json`.
The current MMseqs split contains train 228,282, val 16,606, test 16,606 and
novel 46,147 assignments.  The manifest is reloaded by
`reactflow.splits.manifest_from_json`, which reruns leakage validation before an
audit row can pass.

## Preprocessing Contract

The code path lives in `src/reactflow/data.py`, `src/reactflow/train.py`,
`src/reactflow/rfam_metadata.py` and `src/reactflow/splits.py`.

1. Completeness checks

   - RNA sequences must be non-empty and restricted to `A/C/G/U`.
   - Reactivity vectors must match sequence length.
   - eFold structures must provide `structure` or `pairs`.
   - Missing scalar values are kept as `math.nan`; eFold DMS sentinel values
     `<= -999` are converted to missing and excluded from losses.

2. Validity checks

   - Base pairs are range-checked, ordered as `i < j`, and self-pairs are
     rejected.
   - Chemical probing profiles track low reads, low SNR, negative values and
     high positive robust-MAD outliers.
   - DMS masks only A/C positions; 2A3/SHAPE masks all valid RNA bases.
   - Rfam clan IDs and MMseqs clusters are merged into split groups so no group
     can cross train/val/test/novel boundaries.

3. Normalization

   - `normalize_profile(..., method="p90")` divides finite values by the 90th
     percentile, preserving missing positions.
   - `zscore` and `minmax` are implemented for ablations.
   - Optional negative clipping maps finite negative normalized values to zero
     for non-negative reactivity targets.

4. Feature engineering documentation

   - `feature_engineering_report` records length, base counts, GC fraction,
     probe label, effective mask count and finite reactivity range.
   - Training features include base one-hot, diffusion time `t`, current noised
     partner state, legal-pair structure constraints, and optional frozen
     encoder adapter outputs.
   - Frozen encoder features are read by targeted NPZ-member lookup, so full
     sharded export remains auditable without materializing whole shards.

## Audit Commands

Run the paper artifact audit:

```bash
PYTHONPATH=src python scripts/audit_paper_artifacts.py \
  --full-run-root artifacts/full_runs/full_ablation_20260709_003012 \
  --run-glob 'RF-A1-warm_rfam_current_exact_torch_full_data_e1_bs16' \
  --output-json artifacts/full_runs/full_ablation_20260709_003012/paper_artifact_audit.json \
  --output-md artifacts/full_runs/full_ablation_20260709_003012/paper_artifact_audit.md
```

Run the runtime health audit during long jobs:

```bash
bash scripts/refresh_full_run_status.sh
```

Final SOTA tables must additionally rerun `audit_paper_artifacts.py` with
`--require-final-metrics` for every row included in a claim table.  A run with
missing checkpoint, missing final metrics, fallback split, or non-MMseqs
metadata cannot be promoted from "engineering evidence" to "paper result".
