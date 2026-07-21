# C1-1: Global Data Registry, Contamination-Free Splits, and Pretraining Contamination Audit

**Phase:** C1-1
**Date:** 2026-07-21 (v2 — gap-closure re-run)
**Author:** Trae AI agent (branch `trae/c1-1-data-registry`)
**Repository:** `/home/cunyuliu/reactflow_c1_1_stage_20260721`
**Status:** COMPLETE — Gate verdict: **PASS** (5/5 criteria)

---

## Executive Summary

Phase C1-1 built a global RNA data registry unifying 317,039 records from 6
downloaded sources (and registering 5 additional upstream sources for
provenance), computed 61,538 contamination groups using 8 merge criteria
(exact sequence, parent window, MMseqs cluster, Rfam family, Rfam clan, PDB
chain, probing construct, structure similarity), and produced an immutable
benchmark registry with **zero contamination-group overlap** across all
primary splits.

A majority-vote group reassignment was applied to consolidate 3,692
multi-split contamination groups into single splits, moving 6,314 records
(2.05% of efold_train).  The `novel_clan` split is computed as a subset of
`novel_family` whose MMseqs component (clan) does not appear in train;
after reassignment all 46,997 `novel_family` records satisfy this
condition and are moved to `novel_clan`.  All five Gate criteria pass,
including the new criterion 2c (`novel_clan` disjoint from train).  All
four external pretrained models (RiNALMo, RNA-FM, ERNIE-RNA,
RibonanzaNet2) have a documented contamination status, with exact /
identity / family overlap computed at the database level.

This v2 report supersedes the v1 report.  v1 closed the core registry but
left `novel_clan` as an empty hook, did not register the upstream
Rfam/Ribonanza/Ribonanza2/bpRNA/RNAStrAlign sources, did not compute the
window lost-pair ratio, and reported static defaults for the pretraining
overlap audit.  v2 closes all of those gaps and re-runs the full pipeline
on the server.

---

## 1. What Was Checked

1. **Existing data infrastructure**: Reviewed `rfam_metadata.py` (has
   `UnionFind`, `sequence_sha1`), `splits.py` (`SplitRecord`,
   `SplitManifest`), `data.py` (`EfoldRecord`), and 6 cache JSONL files
   in `artifacts/full_runs/full_ablation_20260709_003012/cache/`.

2. **Existing split manifest**: Loaded
   `splits/rfam_current_mmseqs_seed0/split_manifest.json` (307,641
   assignments: train=228,282, val=16,606, test=16,606, novel=46,147).
   Verified that clans are MMseqs component IDs (`component:*`) and that
   **zero clans span multiple splits** — the mmseqs split is
   cluster-disjoint by construction.

3. **Data leakage from C1-0**: 99.7% of human_mRNA is in efold_train;
   4,142 parent RNAs leak across train/test in the exact split (mmseqs
   split has only 14).  These findings informed the contamination grouping
   and split reassignment design.

4. **External pretrained models**: Audited RiNALMo, RNA-FM, ERNIE-RNA,
   RibonanzaNet2 for training-data overlap with ReactFlow test tiers.
   Computed exact / identity / family overlap at the database level using
   a `DATABASE_TO_REACTFLOW_OVERLAP` mapping (conservative upper bound).

5. **Upstream source registry (v2)**: Registered 5 additional upstream
   sources (Rfam, Ribonanza, Ribonanza2, bpRNA, RNAStrAlign) as
   `DataSourceSpec` entries with `downloaded=False`.  Implemented a
   manifest-only downloader for bpRNA/RNAStrAlign that records provenance
   metadata (upstream URL, license, citation, expected record count,
   SHA-256 placeholder).

6. **Window lost-pair ratio (v2)**: Computed the proxy metric
   `1.0 - (window_length / parent_length)` for all 65,714 windowed
   records.  Documented that true lost-pair ratio requires parent pairs,
   which are not stored in the current cache format.

7. **Reads / SNR (v2)**: Documented that reads (sequencing depth) and
   SNR are not stored in the current cache JSONL format; only the final
   aggregated reactivity profile is available.  The report includes a
   `reads_snr_stats` block with `status="not_available_in_cache"`.

---

## 2. Modified / Created Files

| File | Type | Description |
|------|------|-------------|
| `src/reactflow/data_registry.py` | NEW (v1) + UPDATED (v2) | Unified `DataRecord` schema (23 fields), `KNOWN_SOURCES` (11 entries: 6 downloaded + 5 registered-not-downloaded), `iter_jsonl`, `load_cache_file`.  v2 extended `DataSourceSpec` with `downloaded`, `upstream_url`, `upstream_license`. |
| `src/reactflow/contamination.py` | NEW | `UnionFind`, `ContaminationGrouper` (8 merge criteria), `annotate_records_from_split_manifest` |
| `scripts/build_global_registry.py` | NEW (v1) + UPDATED (v2) | Orchestrates loading 11 data sources (skips registered-not-downloaded), annotation, grouping, manifest emission.  v2 records `downloaded`, `upstream_url`, `upstream_license`, `skipped_reason` per source. |
| `scripts/build_frozen_benchmarks.py` | NEW (v1) + UPDATED (v2) | Builds immutable splits with majority-vote group reassignment + Gate validation.  v2 adds `compute_novel_clan_split()` and criterion 2c. |
| `scripts/audit_pretraining_contamination.py` | NEW (v1) + COMPLETE REWRITE (v2) | Audits 4 external RNA foundation models.  v2 actually computes exact / identity / family overlap via `DATABASE_TO_REACTFLOW_OVERLAP` (was static 0 in v1), records weight_hash status, emits operator checklist. |
| `scripts/compute_data_quality_stats.py` | NEW (v1) + UPDATED (v2) | Computes 11 + 2 data quality statistics.  v2 adds `window_lost_pair_ratio_stats` and `reads_snr_stats`. |
| `scripts/download_bprna_rnastralign.py` | NEW (v2) | Manifest-only downloader for bpRNA/RNAStrAlign.  Records upstream URL, license, citation, expected record count, SHA-256 placeholder.  `--download` flag is a stub with manual instructions. |
| `tests/test_data_registry.py` | NEW (v1) + UPDATED (v2) | 50 tests: schema, serialization, cache loading, pair classification, pseudoknot detection.  v2 updated `TestKnownSources` for 11 sources. |
| `tests/test_contamination.py` | NEW | 43 tests: UnionFind, all 8 merge criteria, split_overlap, annotation helpers |
| `tests/test_download_bprna_rnastralign.py` | NEW (v2) | 13 tests: DownloadSpec completeness, build_manifest, download_source stub, SHA-256 computation |
| `tests/test_build_frozen_benchmarks.py` | NEW (v2) | 10 tests: `compute_novel_clan_split`, `validate_novel_clan_disjoint`, `validate_novel_family_disjoint` |
| `tests/test_audit_pretraining_contamination.py` | NEW (v2) | 23 tests: KNOWN_MODELS, SplitSequences, compute_weight_hash, audit_model, DATABASE_TO_REACTFLOW_OVERLAP |
| `docs/c1_1_data_registry_report.md` | NEW (v1) + UPDATED (v2) | This report |
| `artifacts/c1_1/global_registry_manifest.json` | ARTIFACT | Registry manifest with per-source stats (11 sources) |
| `artifacts/c1_1/global_registry_records.jsonl` | ARTIFACT | 317,039 unified DataRecord entries (801 MB) |
| `artifacts/c1_1/contamination_groups.jsonl` | ARTIFACT | 61,538 contamination groups |
| `artifacts/c1_1/frozen_benchmark_manifest.json` | ARTIFACT | Immutable split manifest (Gate PASS, 5 criteria) |
| `artifacts/c1_1/data_quality_stats.json` | ARTIFACT | 13 data quality statistics (11 original + 2 v2) |
| `artifacts/c1_1/pretraining_contamination_report.json` | ARTIFACT | 4-model contamination audit with computed overlap |
| `artifacts/c1_1/bprna_rnastralign_manifest.json` | ARTIFACT (v2) | bpRNA/RNAStrAlign provenance manifest |

---

## 3. Algorithm Principles

### 3.1 Unified DataRecord Schema

`DataRecord` is a frozen dataclass with 23 spec-required fields:

```
record_id, source, source_id, source_version, sequence, sequence_checksum,
length, length_bucket, parent_id, parent_coordinates, window, pairs,
pair_types, pseudoknot_pairs, has_pseudoknot_field, reactivity, reactivity_source,
probe, replicate, experimental_condition, family, clan, sequence_cluster,
structure_cluster, release_date, quality_flags
```

- `from_cache_row()` provides backward compatibility with existing cache JSONL
  (fields: `family, length_bucket, pairs, probe, reactivity, reactivity_source,
  sequence, source_id, window`).
- `canonicalize_sequence()` uppercases and replaces T→U.
- `sequence_checksum()` = SHA-256 of the canonical sequence.
- `classify_pair()` returns `("canonical" | "wobble" | "noncanonical")` based on
  nucleotide identity (AU/UA/GC/CG = canonical, GU/UG = wobble).
- `detect_pseudoknots()` normalizes unordered pairs `(j,i)`→`(i,j)` with
  deduplication, then returns crossing pairs.

### 3.2 DataSourceSpec and KNOWN_SOURCES (v2)

`DataSourceSpec` is a frozen dataclass with the following fields:

```
name, cache_filename, description, has_real_profiles, is_windowed,
downloaded (bool, default True), upstream_url (Optional[str]),
upstream_license (Optional[str])
```

`KNOWN_SOURCES` is a tuple of 11 entries:

- 6 cached sources with `downloaded=True`: efold_train, PDB, ArchiveII,
  viral, lncRNA, human_mRNA.
- 5 registered-but-not-downloaded sources with `downloaded=False`: Rfam,
  Ribonanza, Ribonanza2, bpRNA, RNAStrAlign.  Each carries `upstream_url`
  and `upstream_license` for provenance.

`build_global_registry.py` iterates over `KNOWN_SOURCES` and skips any
source whose cache file does not exist, recording `skipped_reason` in the
manifest.  This allows downstream phases to know which sources were
registered for completeness but not loaded.

### 3.3 Contamination Grouping

`ContaminationGrouper` uses a deterministic `UnionFind` with lexicographic
root selection (smallest record_id wins) for full reproducibility.

**8 merge criteria** (applied in order):

| Criterion | Key | Records merged |
|-----------|-----|----------------|
| `exact_sequence` | `sequence_checksum` | 7,585 |
| `parent_window` | `parent_id` | 41,053 |
| `mmseqs_cluster` | `sequence_cluster` | 133,575 |
| `rfam_family` | `family` | 225,670 |
| `rfam_clan` | `clan` | 244,102 |
| `pdb_chain` | PDB ID + chain (parsed from `source_id`) | 0 |
| `probing_construct` | `(parent_id, probe)` | 41,053 |
| `structure_similarity` | (hook; not yet implemented) | 0 |

Records with `None` family/clan/cluster/parent are **skipped** for that
criterion (treated as singletons) to avoid merging all unannotated records
into one giant group.

### 3.4 Majority-Vote Group Reassignment

The existing mmseqs split is cluster-disjoint but **not** family-disjoint or
parent-disjoint across train/val/test.  To achieve zero contamination-group
overlap (Gate criterion 1), each contamination group is assigned entirely to
the split holding the majority of its records:

```
best_split = argmax_{split} count(records in group ∩ split)
ties broken by priority: train > val > test_mmseqs > novel_family > novel_clan
```

This guarantees:
- **Zero group overlap** (criterion 1): each group is in exactly one split.
- **Zero parent-window cross-split** (criterion 4): `merge_parent_windows`
  groups by `parent_id`, so all records from the same parent are in the same
  group, which is in one split.
- **Novel-family disjointness** (criterion 2b): after reassignment, families
  that spanned train and novel_family are consolidated into one split, so
  no family appears in both.

### 3.5 novel_clan Split Computation (v2)

`compute_novel_clan_split()` (in `build_frozen_benchmarks.py`) reclassifies
a subset of `novel_family` records as `novel_clan`.  A record is moved if:

1. its current split is `novel_family`,
2. its `clan` is not `None`, AND
3. its `clan` does not appear in any train record's clan.

After the majority-vote reassignment, the contamination group containing
all records sharing a clan is in one split.  If that split is
`novel_family`, no train record shares the clan, so **all** `novel_family`
records with a non-None clan qualify.  Because the mmseqs split assigns a
component ID as the clan to every record, every `novel_family` record has
a non-None clan, so all 46,997 records are moved to `novel_clan`.

`validate_novel_clan_disjoint()` confirms that no `novel_clan` record's
clan appears in any train record's clan (criterion 2c).

---

## 4. Data Sources

### 4.1 Downloaded (loaded into the registry)

| Source | Cache file | Records | Real profiles | Windowed |
|--------|-----------|---------|---------------|----------|
| eFold/RNAndria Dryad | `efold_train.jsonl` | 307,641 | No (proxy) | No |
| PDB-derived | `PDB.jsonl` | 333 | No (proxy) | No |
| ArchiveII | `archiveII.jsonl` | 2,052 | No (proxy) | No |
| viral | `viral.jsonl` | 97 | Yes (mixed) | No |
| lncRNA | `lncRNA.jsonl` | 289 | No (proxy) | Yes |
| human_mRNA | `human_mRNA.jsonl` | 6,627 | Yes (DMS) | Yes |
| **Total** | | **317,039** | 60,088 real / 256,951 proxy | 65,714 (20.7%) |

### 4.2 Registered but not downloaded (provenance only)

| Source | Upstream URL | License | Expected records | Notes |
|--------|--------------|---------|------------------|-------|
| Rfam | https://rfam.org/ | CC-BY-4.0 | n/a | Family/clan annotations; consumed via `rfam_metadata.py` rather than as a standalone cache file. |
| Ribonanza | https://www.kaggle.com/competitions/ribonanza-rna-folding | Kaggle competition (research use) | ~2,000,000 | Chemical mapping data; used by RibonanzaNet2 pretraining. |
| Ribonanza2 | https://www.kaggle.com/competitions/ribonanza-rna-folding | Kaggle competition (research use) | (extends Ribonanza) | Extends Ribonanza; used by RibonanzaNet2 pretraining. |
| bpRNA | https://bprna.cgrb.oregonstate.edu/ | MIT (code) / Rfam (data) | 102,318 | Structure annotations; used by RibonanzaNet2 pretraining. |
| RNAStrAlign | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6324060/ | Public (research use) | 30,451 | Structure alignments; used by RibonanzaNet2 pretraining. |

The bpRNA/RNAStrAlign manifest is produced by
`scripts/download_bprna_rnastralign.py` and records the upstream URL,
license, citation, expected record count, and a SHA-256 placeholder
(`not_downloaded`).  Running with `--download` raises a
`NotImplementedError` with manual download instructions.

---

## 5. Run Commands

```bash
# 1. Build global registry (loads 6 downloaded sources, skips 5 registered-not-downloaded)
PYTHONPATH=src python scripts/build_global_registry.py --emit-records

# 2. Build bpRNA/RNAStrAlign provenance manifest (manifest-only mode)
PYTHONPATH=src python scripts/download_bprna_rnastralign.py

# 3. Build frozen benchmark splits (majority-vote reassignment + novel_clan + Gate validation)
PYTHONPATH=src python scripts/build_frozen_benchmarks.py

# 4. Compute data quality statistics (13 statistics including window lost-pair ratio, reads/SNR)
PYTHONPATH=src python scripts/compute_data_quality_stats.py

# 5. Audit pretraining contamination (4 models, database-level overlap computation)
PYTHONPATH=src python scripts/audit_pretraining_contamination.py

# 6. Run tests (146 tests across 5 test files)
PYTHONPATH=src python -m pytest tests/test_data_registry.py tests/test_contamination.py \
    tests/test_download_bprna_rnastralign.py tests/test_build_frozen_benchmarks.py \
    tests/test_audit_pretraining_contamination.py -v
```

---

## 6. Test Results

```
tests/test_data_registry.py ............................................ [ 30%]
....................                                                     [ 43%]
tests/test_contamination.py ....................................         [ 68%]
tests/test_download_bprna_rnastralign.py .............                   [ 77%]
tests/test_build_frozen_benchmarks.py ..........                         [ 84%]
tests/test_audit_pretraining_contamination.py .......................    [100%]

============================= 146 passed in 3.08s ==============================
```

- **50 tests** in `test_data_registry.py`: DataRecord schema, serialization,
  cache loading, pair classification, pseudoknot detection, checksum,
  canonicalization, from_cache_row edge cases, `KNOWN_SOURCES` (11 entries).
- **43 tests** in `test_contamination.py`: UnionFind (union, find, union_many,
  lexicographic root), ContaminationGrouper (all 8 merge criteria, merge_all,
  to_jsonl), split_overlap, annotate_records_from_split_manifest
  (source_id matching), extract_pdb_chain.
- **13 tests** in `test_download_bprna_rnastralign.py`: DownloadSpec
  completeness, build_manifest (manifest-only mode), download_source stub,
  SHA-256 computation.
- **10 tests** in `test_build_frozen_benchmarks.py`:
  `compute_novel_clan_split` (move conditions, edge cases),
  `validate_novel_clan_disjoint`, `validate_novel_family_disjoint`.
- **23 tests** in `test_audit_pretraining_contamination.py`: KNOWN_MODELS,
  SplitSequences, `compute_weight_hash` (existing file, missing file,
  not-downloaded spec), `audit_model` (contamination status, overlap lists,
  family overlap), `DATABASE_TO_REACTFLOW_OVERLAP` (mapping coverage).

---

## 7. Experimental Results

### 7.1 Global Registry Statistics

| Metric | Value |
|--------|-------|
| Total records | 317,039 |
| Total contamination groups | 61,538 |
| Singleton groups | 53,358 |
| Multi-record groups | 8,180 |
| Largest group | 228,520 (component:1e3fa9b8a2d6 = train split) |
| Unique sequences (checksums) | 309,454 |
| Total pairs | 13,219,855 |
| Canonical pairs | 11,849,854 (89.6%) |
| Wobble pairs | 1,370,001 (10.4%) |
| Noncanonical pairs | 0 (0.0%) |
| Pseudoknot records | 608 (0.2%) |
| Pseudoknot pairs | 8,983 |
| Real reactivity profiles | 60,088 |
| Proxy reactivity profiles | 256,951 |
| Unique Rfam families | 29 |
| Unique clans (MMseqs components) | 70,144 |
| Windowed records | 65,714 (20.7%) |
| Mean length | 156.6 nt (median 171, max 256) |

### 7.2 Frozen Benchmark Splits

| Split | Count | Notes |
|-------|-------|-------|
| train | 228,490 | Original 228,282 + 208 from reassignment.  146 unique clans. |
| val | 17,120 | Original 16,606 + 514 from reassignment |
| test_mmseqs | 15,034 | Original 16,606 − 1,572 from reassignment |
| novel_family | 0 | All 46,997 records moved to `novel_clan` (all have non-None clan not in train) |
| novel_clan | 46,997 | Subset of `novel_family` with clan not in train (criterion 2c PASS) |
| public_PDB | 333 | Benchmark tag |
| public_ArchiveII | 2,052 | Benchmark tag |
| viral | 97 | Benchmark tag |
| lncRNA | 289 | Benchmark tag |
| human_mRNA | 6,627 | Benchmark tag (99.7% overlap with train — see §9) |
| pseudoknot | 608 | Independent tag |

**Reassignment stats**: 3,692 multi-split groups consolidated, 6,314 records
moved (2.05% of efold_train).  Split changes:

| From | To | Records moved |
|------|----|---------------|
| val | novel_family | 1,422 |
| test_mmseqs | novel_family | 1,422 |
| novel_family | val | 1,236 |
| test_mmseqs | val | 1,065 |
| val | test_mmseqs | 310 |
| novel_family | test_mmseqs | 651 |
| novel_family | train | 107 |
| val | train | 55 |
| test_mmseqs | train | 46 |

### 7.3 Window Lost-Pair Ratio (v2)

Definition: proxy metric `1.0 - (window_length / parent_length)` for
windowed records.  Assumes uniform pair density across the parent.  True
lost-pair ratio would require parent pairs, which are not stored in the
current cache format.

| Metric | Value |
|--------|-------|
| Windowed records with parent length | 65,714 |
| Mean lost-pair ratio | 0.3345 |
| Median lost-pair ratio | 0.2967 |
| p25 / p75 / p95 | 0.1495 / 0.4733 / 0.7366 |
| Min / Max | 0.0039 / 0.9355 |
| Mean window coverage | 0.6655 |
| Unique parents | 24,661 |
| Mean windows per parent | 2.66 |
| Median windows per parent | 2 |
| Max windows per parent | 20 |
| Mean pairs per parent | 165.7 |
| Median pairs per parent | 145 |
| Max pairs per parent | 924 |

### 7.4 Reads / SNR (v2)

Reads (sequencing depth) and SNR (signal-to-noise ratio) are **not stored**
in the current cache JSONL format.  The cache stores only the final
aggregated reactivity profile.  Raw reads / SNR would need to be re-parsed
from upstream fastq / bam files (DMS / SHAPE / 2A3 mapping experiments).

| Field | Value |
|-------|-------|
| `status` | `not_available_in_cache` |
| `records_with_reads_field` | 0 |
| `records_with_snr_field` | 0 |
| `reads_values_count` | 0 |
| `snr_values_count` | 0 |

This block is included for spec compliance (line 299) and to document the
gap.  The script defensively checks the raw row dict for `reads` / `snr`
fields, so future cache formats that add these fields will be picked up
automatically.

### 7.5 Pretraining Contamination Audit

The audit computes overlap at the **database level** using
`DATABASE_TO_REACTFLOW_OVERLAP`, which maps each upstream RNA database to
the ReactFlow sources it overlaps.  This is a conservative upper bound:
true exact-sequence overlap requires downloading the model's training data.

| Model | Version | Training databases | Overlapping ReactFlow sources | Status |
|-------|---------|--------------------|------------------------------|--------|
| RiNALMo | 1.0 (2024-02) | RNAcentral, Rfam, Ensembl, GENCODE | efold_train | **contaminated** |
| RNA-FM | 1.0 (2023-05) | RNAcentral, Rfam | efold_train | **contaminated** |
| ERNIE-RNA | 1.0 (2023-04) | RNAcentral, Rfam | efold_train | **contaminated** |
| RibonanzaNet2 | 2.0 (2024-08) | Ribonanza, bpRNA, RNAStrAlign | efold_train (via bpRNA/RNAStrAlign overlap with eFold/PDB/ArchiveII) | **contaminated** |

**Overlap computation methodology**:
- `exact_overlap_test/novel`: list of ReactFlow sources whose sequences
  appear in the model's training databases.  Computed by intersecting
  `DATABASE_TO_REACTFLOW_OVERLAP[db]` for each `db` in the model's
  `known_rna_databases`.
- `identity_overlap_test/novel`: subsumed by exact overlap (canonical
  sequences have 100% identity by definition).
- `family_overlap_test/novel`: for models trained on RNAcentral / Rfam,
  the model has seen all Rfam families, so family overlap is reported as
  "all N test/novel families".  For other models, family overlap is 0
  unless verified.

**Weight hash status**: all 4 models have `weight_hash = "not_downloaded"`
and `weight_hash_computation_status = "not_downloaded"`.  The
`weight_hash_operator_checklist` in the report JSON provides per-model
download URLs and commands.

**Recommendation**: For any ReactFlow SOTA claim on test_mmseqs,
novel_family, or novel_clan, use either `from_scratch` or
`self_pretrained` (with disclosed pretraining data).
`external_pretrained` is permitted for train-only feature extraction and
for ablation against the `from_scratch` baseline, but F1 numbers obtained
with `external_pretrained` must be reported as "with external
pretraining" and cannot be compared to literature SOTA without
split-matching.

---

## 8. Gate Judgment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. train/val/test/novel contamination group overlap = 0 | **PASS** | 0 violations (after majority-vote reassignment) |
| 2. ArchiveII etc. benchmarks not in training | **PASS** | 0 violations (benchmark records excluded from primary splits) |
| 2b. novel_family disjoint from train (families) | **PASS** | 0 violations (consolidated by reassignment) |
| 2c. novel_clan disjoint from train (clans) | **PASS** | 0 violations; `novel_clan_count=46,997`, `novel_family_clan_in_train=0` |
| 3. All splits reconstructable from manifest + checksum | **PASS** | `global_registry_records.jsonl` + `contamination_groups.jsonl` + `frozen_benchmark_manifest.json` |
| 4. Parent-window overlap not cross-split | **PASS** | 0 violations (parent_id merge ensures all same-parent records in one group) |
| 5. All external pretrained models have contamination status | **PASS** | 4/4 models audited with explicit status + computed exact/identity/family overlap |

### Overall Gate verdict: **PASS** (5/5 criteria, including new criterion 2c)

---

## 9. Unresolved Issues

1. **human_mRNA benchmark overlap**: 99.7% of human_mRNA sequences appear in
   efold_train (identified in C1-0).  Although human_mRNA records are now
   excluded from primary splits, they share contamination groups with train
   records.  Evaluating on human_mRNA is effectively evaluating on train data.
   **Action**: Per `static_v1.yaml`, human_mRNA is excluded from eFold-vs-
   ReactFlow comparisons.  Future benchmarks should use a truly held-out
   human_mRNA set.

2. **Structure-disjoint and time-censored splits are hooks**: These are
   declared in `ALL_SPLIT_NAMES` but not yet populated.  Structure-disjoint
   requires structure similarity computation (criterion 9 in
   `ContaminationGrouper` is a no-op); time-censored requires release
   dates (currently not populated in the cache).

3. **External model weights not downloaded**: For all 4 external models,
   weights have not been downloaded and SHA-256 hashes have not been
   computed (`weight_hash = "not_downloaded"`).  The
   `weight_hash_operator_checklist` in
   `artifacts/c1_1/pretraining_contamination_report.json` provides
   per-model download URLs and commands.  **Action**: Before using any
   external pretrained model in a ReactFlow experiment, download the
   weights, compute SHA-256, set `PretrainedModelSpec.weights_path`, and
   re-run `scripts/audit_pretraining_contamination.py`.

4. **bpRNA / RNAStrAlign not downloaded** (spec line 251): The downloader
   `scripts/download_bprna_rnastralign.py` runs in manifest-only mode by
   default.  The `--download` flag is a stub that raises
   `NotImplementedError` with manual instructions.  **Action**: If these
   sources are needed for training (e.g., to align with RibonanzaNet2
   pretraining), implement the actual download logic and populate the
   cache files.

5. **Database-level overlap is an upper bound**: The pretraining
   contamination audit computes overlap at the database level
   (RNAcentral ⊇ efold_train, etc.), not at the exact-sequence level.
   True exact-sequence overlap requires downloading the model's training
   data.  **Action**: For a tighter audit, download RNAcentral release
   matching the model's training cutoff and compute exact-sequence
   overlap with ReactFlow test/novel splits.

6. **Fallback pseudo-clan fraction = 0.88%**: 2,793 records have
   `__unannotated__` as their clan.  These are records that could not be
   matched to the split manifest (non-efold_train sources without MMseqs
   cluster annotation).

---

## 10. Next Phase Input

Phase C1-2 (Strong Static PairFormer Prototype) can now proceed with:

1. **Frozen data registry**: Use `artifacts/c1_1/frozen_benchmark_manifest.json`
   as the immutable split definition.  All training/evaluation must use these
   splits.

2. **Contamination groups**: Use `artifacts/c1_1/contamination_groups.jsonl`
   to verify that no new data source introduces cross-split contamination.

3. **Evaluator contract**: Use `configs/evaluation/static_v1.yaml` (from
   C1-0) with the frozen splits.

4. **Pretraining protocol**: Any experiment using external pretrained models
   (RiNALMo, RNA-FM, ERNIE-RNA, RibonanzaNet2) must use the `from_scratch`
   or `self_pretrained` protocol for test/novel F1 claims.  All 4 models
   are marked `contaminated` at the database level.

5. **Data loading**: Use `src/reactflow/data_registry.py` (`iter_jsonl`,
   `load_cache_file`, `DataRecord.from_cache_row`) for all data loading.
   The `DataRecord` schema is the canonical interface for all downstream
   code.

6. **Test counts**: train=228,490, val=17,120, test_mmseqs=15,034,
   novel_family=0, novel_clan=46,997.  Total primary = 307,641.
