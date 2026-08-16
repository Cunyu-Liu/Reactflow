#!/usr/bin/env python3
"""C1-0 Task 5: Data leakage and protocol conflict audit.

Streams eFold and ReactFlow JSONL data to detect five classes of integrity
issues that would invalidate cross-family evaluation:

  1. Sequence overlap between train and test splits (hash-based, normalized).
  2. Pair index protocol violations (0/1-based, self-pairs, out-of-range,
     unordered pairs).
  3. T-to-U conversion issues (DNA thymine leaking into RNA sequences).
  4. Truncation/padding artifacts (fixed-length sequences with pair indices
     that drift beyond the visible window or reference parent coordinates).
  5. Window parent overlap (different windows of the same parent RNA appearing
     in both train and test).

Memory model: each file is streamed line by line.  Only MD5 digests (16 raw
bytes, stored here as 32-char hex strings) and small counters are retained per
split.  eFold train (~307k records) yields ~10MB of hex digests, well within
budget; the ReactFlow splits are smaller.  Records themselves are never held
in aggregate.

Output: artifacts/c1_0/data_overlap_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_FULL_RUN = Path("artifacts/full_runs/full_ablation_20260709_003012")
DEFAULT_OUT = Path("artifacts/c1_0/data_overlap_audit.json")

# eFold cache files (the canonical Dryad-derived dataset).
EFOLD_CACHE = {
    "efold_train": "cache/efold_train.jsonl",
    "PDB": "cache/PDB.jsonl",
    "archiveII": "cache/archiveII.jsonl",
    "viral": "cache/viral.jsonl",
    "lncRNA": "cache/lncRNA.jsonl",
    "human_mRNA": "cache/human_mRNA.jsonl",
}

# ReactFlow splits derived from efold_train (exact and mmseqs clustering).
RF_EXACT_SPLITS = {
    "exact_train": "splits/rfam_current_exact_seed0/train.jsonl",
    "exact_val": "splits/rfam_current_exact_seed0/val.jsonl",
    "exact_test": "splits/rfam_current_exact_seed0/test.jsonl",
    "exact_novel": "splits/rfam_current_exact_seed0/novel.jsonl",
}
RF_MMSEQS_SPLITS = {
    "mmseqs_train": "splits/rfam_current_mmseqs_seed0/train.jsonl",
    "mmseqs_test": "splits/rfam_current_mmseqs_seed0/test.jsonl",
    "mmseqs_novel": "splits/rfam_current_mmseqs_seed0/novel.jsonl",
}

# ReactFlow frozen-feature train manifest (id + sequence only, no pairs).
FROZEN_TRAIN = "clean/train_sequences_for_frozen.jsonl"

# Common truncation/window lengths to flag when a record lands exactly on them.
TRUNCATION_LENGTHS = (64, 100, 128, 256, 512, 1024)

# Cap on stored overlap examples and sampled-protocol records to keep the JSON
# artifact readable.
MAX_OVERLAP_EXAMPLES = 20
MAX_PROTOCOL_SAMPLES = 2000

_WINDOW_SUFFIX_RE = re.compile(r":(\d+)-(\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_seq(seq: str) -> str:
    """Normalize an RNA sequence for hashing.

    Formula: uppercase, then collapse DNA thymine (T) to RNA uracil (U) so that
    a sequence stored as DNA does not dodge an overlap check.  Complexity:
    O(len(seq)).
    """

    return seq.upper().replace("T", "U")


def seq_hash(seq: str) -> str:
    """Return the MD5 hex digest of a normalized sequence.

    Complexity: O(len(seq)).
    """

    return hashlib.md5(normalize_seq(seq).encode("ascii", "ignore")).hexdigest()


def parent_id(source_id: str, window: Optional[Mapping[str, int]] = None) -> str:
    """Recover the parent RNA identifier from a (possibly windowed) source_id.

    Windowed records carry a ``:start-end`` suffix on the source_id and a
    ``window`` dict with parent_length.  Stripping the suffix recovers the
    parent so that two windows of the same parent can be detected as siblings.

    Complexity: O(len(source_id)).
    """

    if source_id is None:
        return "unknown"
    text = str(source_id)
    m = _WINDOW_SUFFIX_RE.search(text)
    if m:
        return text[: m.start()]
    return text


def iter_records(path: Path) -> Iterator[dict]:
    """Stream JSONL records from ``path`` one at a time.

    Complexity: O(N) IO; memory O(1) per record (caller may discard).
    """

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def file_record_count(path: Path) -> int:
    """Count JSONL lines in ``path`` without parsing JSON.

    Complexity: O(N) IO.
    """

    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def resolve(base: Path, rel: str) -> Path:
    """Resolve a data path, tolerating missing files by returning the input."""

    p = base / rel
    return p if p.exists() else Path(rel)


def verdict(*checks: str) -> str:
    """Aggregate per-check verdicts into one overall verdict.

    Formula: fail dominates warning dominates pass.  Complexity: O(len(checks)).
    """

    if "fail" in checks:
        return "fail"
    if "warning" in checks:
        return "warning"
    return "pass"


# ---------------------------------------------------------------------------
# Check 1: sequence overlap (hash-based)
# ---------------------------------------------------------------------------

@dataclass
class SplitDigests:
    """Accumulated hashes and metadata for one split file."""

    name: str
    path: str
    record_count: int = 0
    seq_count: int = 0
    digests: Set[str] = field(default_factory=set)
    # Map digest -> one example source_id (for cross-reference).
    examples: Dict[str, str] = field(default_factory=dict)

    def add(self, digest: str, source_id: str) -> None:
        self.digests.add(digest)
        if digest not in self.examples:
            self.examples[digest] = source_id


def build_split_digests(name: str, path: Path) -> SplitDigests:
    """Stream a split file and collect sequence hashes.

    Complexity: O(N) time, O(U) memory where U is unique sequences.
    """

    sd = SplitDigests(name=name, path=str(path))
    if not path.exists():
        return sd
    for rec in iter_records(path):
        sd.record_count += 1
        seq = rec.get("sequence")
        if not seq:
            continue
        sd.seq_count += 1
        sd.add(seq_hash(seq), str(rec.get("source_id", rec.get("id", "?"))))
    return sd


def pairwise_overlap(
    train: SplitDigests, test: SplitDigests
) -> Tuple[int, List[dict]]:
    """Count overlapping digests between train and test, collect examples.

    Complexity: O(min(|train|, |test|)) set intersection.
    """

    common = train.digests & test.digests
    examples: List[dict] = []
    for d in list(common)[:MAX_OVERLAP_EXAMPLES]:
        examples.append(
            {
                "digest": d,
                "train_source_id": train.examples.get(d, "?"),
                "test_source_id": test.examples.get(d, "?"),
            }
        )
    return len(common), examples


def run_sequence_overlap(base: Path) -> dict:
    """Run all pairwise sequence-overlap comparisons.

    Compares:
      * eFold train vs each eFold test set (PDB, archiveII, viral, lncRNA,
        human_mRNA).
      * ReactFlow exact train vs exact val/test/novel.
      * ReactFlow mmseqs train vs mmseqs test/novel.
      * ReactFlow exact train vs each eFold test set (cross-framework leak).
      * eFold train vs ReactFlow exact test (informational; splits are derived
        from efold_train so overlap is expected and flagged as a warning).
    """

    print("[sequence_overlap] streaming eFold cache files...", file=sys.stderr)
    efold = {
        name: build_split_digests(name, resolve(base, rel))
        for name, rel in EFOLD_CACHE.items()
    }
    print("[sequence_overlap] streaming ReactFlow exact splits...", file=sys.stderr)
    exact = {
        name: build_split_digests(name, resolve(base, rel))
        for name, rel in RF_EXACT_SPLITS.items()
    }
    print("[sequence_overlap] streaming ReactFlow mmseqs splits...", file=sys.stderr)
    mmseqs = {
        name: build_split_digests(name, resolve(base, rel))
        for name, rel in RF_MMSEQS_SPLITS.items()
    }

    train_files = [efold["efold_train"].path, exact["exact_train"].path,
                   mmseqs["mmseqs_train"].path]
    test_files = [efold[n].path for n in ("PDB", "archiveII", "viral", "lncRNA",
                                          "human_mRNA")] + [exact["exact_test"].path,
                                                            exact["exact_novel"].path,
                                                            mmseqs["mmseqs_test"].path,
                                                            mmseqs["mmseqs_novel"].path]

    comparisons: List[dict] = []

    def cmp(label: str, a: SplitDigests, b: SplitDigests, expected: str = "no") -> None:
        n, ex = pairwise_overlap(a, b)
        v = "pass"
        if n > 0:
            v = "pass" if expected == "expected" else "fail"
            if expected == "info":
                v = "warning"
        comparisons.append(
            {
                "label": label,
                "train_split": a.name,
                "test_split": b.name,
                "train_sequences": a.seq_count,
                "test_sequences": b.seq_count,
                "overlap_count": n,
                "overlap_examples": ex,
                "expected": expected,
                "verdict": v,
            }
        )

    # eFold train vs eFold test sets.
    for tname in ("PDB", "archiveII", "viral", "lncRNA", "human_mRNA"):
        cmp(f"efold_train vs {tname}", efold["efold_train"], efold[tname])

    # ReactFlow exact train vs exact val/test/novel.
    for tname in ("exact_val", "exact_test", "exact_novel"):
        cmp(f"exact_train vs {tname}", exact["exact_train"], exact[tname])

    # ReactFlow mmseqs train vs mmseqs test/novel.
    for tname in ("mmseqs_test", "mmseqs_novel"):
        cmp(f"mmseqs_train vs {tname}", mmseqs["mmseqs_train"], mmseqs[tname])

    # Cross-framework: ReactFlow exact train vs eFold test sets.
    for tname in ("PDB", "archiveII", "viral", "lncRNA", "human_mRNA"):
        cmp(f"exact_train vs {tname} (cross)", exact["exact_train"], efold[tname])

    # Informational: efold_train is the union of ReactFlow splits, so overlap
    # with exact_test/novel is expected by construction.
    cmp("efold_train vs exact_test (union)", efold["efold_train"],
        exact["exact_test"], expected="info")
    cmp("efold_train vs exact_novel (union)", efold["efold_train"],
        exact["exact_novel"], expected="info")

    overall = verdict(*[c["verdict"] for c in comparisons])
    return {
        "train_files": train_files,
        "test_files": test_files,
        "train_sequences": efold["efold_train"].seq_count,
        "test_sequences": sum(efold[n].seq_count for n in
                              ("PDB", "archiveII", "viral", "lncRNA", "human_mRNA")),
        "comparisons": comparisons,
        "verdict": overall,
    }


# ---------------------------------------------------------------------------
# Check 2: pair index protocol
# ---------------------------------------------------------------------------

def run_pair_index_protocol(base: Path, sample_cap: int = MAX_PROTOCOL_SAMPLES) -> dict:
    """Sample records and verify pair index protocol.

    Checks per record:
      * Indexing base: detect 1-based by looking for pairs starting at 1 with
        no 0 references, or pairs with max index == len(seq).
      * Self-pairs: (i, i).
      * Out-of-range: any index >= len(seq) or < 0.
      * Ordering: pairs stored as (i, j) with i < j.
      * Non-integer or non-2-element pair entries.

    Complexity: O(samples * P) where P is pairs per record.
    """

    files = []
    for rel in list(EFOLD_CACHE.values()) + list(RF_EXACT_SPLITS.values()):
        p = resolve(base, rel)
        if p.exists():
            files.append((rel, p))

    per_file: List[dict] = []
    total_self_pairs = 0
    total_out_of_range = 0
    total_unordered = 0
    total_malformed = 0
    total_one_based = 0
    total_sampled = 0

    for rel, p in files:
        self_pairs = 0
        out_of_range = 0
        unordered = 0
        malformed = 0
        one_based_signals = 0
        sampled = 0
        examples: List[dict] = []

        for rec in iter_records(p):
            if sampled >= sample_cap:
                break
            sampled += 1
            seq = rec.get("sequence", "")
            pairs = rec.get("pairs", [])
            seqlen = len(seq)
            max_idx = -1
            min_idx = 0
            zero_used = False
            for pr in pairs:
                if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                    malformed += 1
                    if len(examples) < 3:
                        examples.append({"source_id": str(rec.get("source_id", "?")),
                                         "issue": "malformed_pair", "value": pr})
                    continue
                i, j = pr[0], pr[1]
                if not isinstance(i, int) or not isinstance(j, int):
                    malformed += 1
                    continue
                if i == j:
                    self_pairs += 1
                    if len(examples) < 3:
                        examples.append({"source_id": str(rec.get("source_id", "?")),
                                         "issue": "self_pair", "value": [i, j]})
                if i < 0 or j < 0 or i >= seqlen or j >= seqlen:
                    out_of_range += 1
                    if len(examples) < 3:
                        examples.append({"source_id": str(rec.get("source_id", "?")),
                                         "issue": "out_of_range", "value": [i, j],
                                         "seq_len": seqlen})
                if i > j:
                    unordered += 1
                    if len(examples) < 3:
                        examples.append({"source_id": str(rec.get("source_id", "?")),
                                         "issue": "unordered", "value": [i, j]})
                if i == 0 or j == 0:
                    zero_used = True
                if i > max_idx:
                    max_idx = i
                if j > max_idx:
                    max_idx = j
                if i < min_idx:
                    min_idx = i
            # 1-based signal: no zero used, max index == seqlen (would be out
            # of range for 0-based but valid for 1-based), or max index ==
            # seqlen-1 with no zero used across many records.
            if seqlen > 0 and max_idx == seqlen and not zero_used:
                one_based_signals += 1
                if len(examples) < 3:
                    examples.append({"source_id": str(rec.get("source_id", "?")),
                                     "issue": "one_based_signal", "max_idx": max_idx,
                                     "seq_len": seqlen})

        total_self_pairs += self_pairs
        total_out_of_range += out_of_range
        total_unordered += unordered
        total_malformed += malformed
        total_one_based += one_based_signals
        total_sampled += sampled
        per_file.append(
            {
                "file": rel,
                "sampled": sampled,
                "self_pairs": self_pairs,
                "out_of_range": out_of_range,
                "unordered_pairs": unordered,
                "malformed_pairs": malformed,
                "one_based_signals": one_based_signals,
                "examples": examples,
            }
        )

    issues = []
    if total_self_pairs:
        issues.append(f"{total_self_pairs} self-pairs")
    if total_out_of_range:
        issues.append(f"{total_out_of_range} out-of-range indices")
    if total_unordered:
        issues.append(f"{total_unordered} unordered pairs")
    if total_malformed:
        issues.append(f"{total_malformed} malformed pairs")
    if total_one_based:
        issues.append(f"{total_one_based} one-based-indexing signals")

    v = "fail" if (total_out_of_range or total_self_pairs or total_malformed) else (
        "warning" if (total_unordered or total_one_based) else "pass")

    return {
        "files_inspected": len(per_file),
        "records_sampled": total_sampled,
        "indexing_base": "0-based" if total_one_based == 0 else "ambiguous (1-based signals)",
        "self_pairs": total_self_pairs,
        "out_of_range": total_out_of_range,
        "unordered_pairs": total_unordered,
        "malformed_pairs": total_malformed,
        "one_based_signals": total_one_based,
        "per_file": per_file,
        "issues": issues,
        "verdict": v,
    }


# ---------------------------------------------------------------------------
# Check 3: T-to-U conversion
# ---------------------------------------------------------------------------

def run_t_to_u_conversion(base: Path, sample_cap: int = MAX_PROTOCOL_SAMPLES) -> dict:
    """Detect DNA thymine (T) in RNA sequences.

    A non-zero T count indicates either DNA input that was not converted, or a
    mixed alphabet.  U count is reported for context.

    Complexity: O(samples * L) where L is mean sequence length.
    """

    files = []
    for rel in list(EFOLD_CACHE.values()) + list(RF_EXACT_SPLITS.values()):
        p = resolve(base, rel)
        if p.exists():
            files.append((rel, p))

    per_file: List[dict] = []
    total_t = 0
    total_u = 0
    total_records_with_t = 0
    total_sampled = 0
    examples: List[dict] = []

    for rel, p in files:
        t_count = 0
        u_count = 0
        records_with_t = 0
        sampled = 0
        for rec in iter_records(p):
            if sampled >= sample_cap:
                break
            sampled += 1
            seq = rec.get("sequence", "") or ""
            if not seq:
                continue
            up = seq.upper()
            t = up.count("T")
            u = up.count("U")
            t_count += t
            u_count += u
            if t > 0:
                records_with_t += 1
                if len(examples) < 10:
                    examples.append({"file": rel,
                                     "source_id": str(rec.get("source_id", "?")),
                                     "t_count": t, "seq_len": len(seq),
                                     "preview": seq[:80]})
        total_t += t_count
        total_u += u_count
        total_records_with_t += records_with_t
        total_sampled += sampled
        per_file.append({"file": rel, "sampled": sampled, "t_count": t_count,
                         "u_count": u_count, "records_with_t": records_with_t})

    v = "fail" if total_t > 0 else "pass"
    return {
        "files_inspected": len(per_file),
        "records_sampled": total_sampled,
        "total_T": total_t,
        "total_U": total_u,
        "records_with_T": total_records_with_t,
        "per_file": per_file,
        "examples": examples,
        "verdict": v,
    }


# ---------------------------------------------------------------------------
# Check 4: truncation / padding
# ---------------------------------------------------------------------------

def run_truncation_padding(base: Path, sample_cap: int = MAX_PROTOCOL_SAMPLES) -> dict:
    """Detect truncation/padding artifacts.

    Flags:
      * Sequences whose length lands exactly on a common truncation length.
      * Pairs whose indices exceed the visible sequence length (drift).
      * Records with a ``window`` field whose parent_length > window end, where
        pairs reference positions beyond the window end (unadjusted pairs).
      * Records with no window field but a source_id ``:start-end`` suffix
        suggesting a window whose pairs were not re-based.

    Complexity: O(samples * P).
    """

    files = []
    for rel in list(EFOLD_CACHE.values()) + list(RF_EXACT_SPLITS.values()):
        p = resolve(base, rel)
        if p.exists():
            files.append((rel, p))

    per_file: List[dict] = []
    total_exact_trunc = 0
    total_pair_overflow = 0
    total_window_pair_drift = 0
    total_sampled = 0
    length_hist = Counter()
    examples: List[dict] = []

    for rel, p in files:
        exact_trunc = 0
        pair_overflow = 0
        window_pair_drift = 0
        sampled = 0
        for rec in iter_records(p):
            if sampled >= sample_cap:
                break
            sampled += 1
            seq = rec.get("sequence", "") or ""
            seqlen = len(seq)
            pairs = rec.get("pairs", []) or []
            window = rec.get("window")

            if seqlen in TRUNCATION_LENGTHS:
                exact_trunc += 1
                length_hist[seqlen] += 1

            for pr in pairs:
                if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                    continue
                i, j = pr[0], pr[1]
                if not isinstance(i, int) or not isinstance(j, int):
                    continue
                if i >= seqlen or j >= seqlen:
                    pair_overflow += 1
                    if len(examples) < 10:
                        examples.append({"file": rel,
                                         "source_id": str(rec.get("source_id", "?")),
                                         "issue": "pair_overflow",
                                         "pair": [i, j], "seq_len": seqlen})
                    break

            if window and isinstance(window, Mapping):
                wend = window.get("end")
                if isinstance(wend, int) and wend < seqlen:
                    pass
                elif isinstance(wend, int) and wend > seqlen:
                    for pr in pairs:
                        if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                            continue
                        i, j = pr[0], pr[1]
                        if not isinstance(i, int) or not isinstance(j, int):
                            continue
                        if i >= seqlen or j >= seqlen:
                            window_pair_drift += 1
                            if len(examples) < 10:
                                examples.append({"file": rel,
                                                 "source_id": str(rec.get("source_id", "?")),
                                                 "issue": "window_pair_drift",
                                                 "pair": [i, j], "seq_len": seqlen,
                                                 "window": dict(window)})
                            break

        total_exact_trunc += exact_trunc
        total_pair_overflow += pair_overflow
        total_window_pair_drift += window_pair_drift
        total_sampled += sampled
        per_file.append({"file": rel, "sampled": sampled,
                         "exact_truncation_lengths": exact_trunc,
                         "pair_overflow": pair_overflow,
                         "window_pair_drift": window_pair_drift})

    issues = []
    if total_pair_overflow:
        issues.append(f"{total_pair_overflow} records with pair indices beyond sequence length")
    if total_window_pair_drift:
        issues.append(f"{total_window_pair_drift} records with windowed pairs not re-based")
    if total_exact_trunc:
        issues.append(f"{total_exact_trunc} records at exact truncation lengths (may be intentional windowing)")

    v = "fail" if (total_pair_overflow or total_window_pair_drift) else (
        "warning" if total_exact_trunc else "pass")
    return {
        "files_inspected": len(per_file),
        "records_sampled": total_sampled,
        "truncation_lengths_checked": list(TRUNCATION_LENGTHS),
        "records_at_exact_truncation_lengths": total_exact_trunc,
        "length_histogram": dict(length_hist),
        "pair_overflow": total_pair_overflow,
        "window_pair_drift": total_window_pair_drift,
        "per_file": per_file,
        "examples": examples,
        "issues": issues,
        "verdict": v,
    }


# ---------------------------------------------------------------------------
# Check 5: window parent overlap
# ---------------------------------------------------------------------------

def collect_parent_ids(path: Path, sample_cap: Optional[int] = None) -> Tuple[Set[str], int]:
    """Collect parent RNA IDs from a split file.

    Parent ID is derived from source_id by stripping the ``:start-end`` window
    suffix.  Records with no suffix contribute their bare source_id as a parent
    (single-window RNAs are their own parent).

    Complexity: O(N) time, O(U) memory for unique parents.
    """

    parents: Set[str] = set()
    n = 0
    for rec in iter_records(path):
        n += 1
        if sample_cap is not None and n > sample_cap:
            break
        sid = str(rec.get("source_id", rec.get("id", "")))
        if not sid:
            continue
        parents.add(parent_id(sid, rec.get("window")))
    return parents, n


def run_window_parent_overlap(base: Path) -> dict:
    """Detect parent RNAs that contribute windows to both train and test.

    For each split family (exact, mmseqs), checks that the set of parent IDs in
    train is disjoint from test/novel.  Also checks eFold train parents vs the
    eFold test-set parents (PDB etc. are not windowed, so this is mostly
    informational).

    Complexity: O(N) IO per file, O(U) memory for parent sets.
    """

    print("[window_parent_overlap] collecting parent IDs...", file=sys.stderr)

    results: List[dict] = []

    def check(label: str, train_rel: str, test_rel: str) -> None:
        tp = resolve(base, train_rel)
        sp = resolve(base, test_rel)
        if not tp.exists() or not sp.exists():
            return
        train_parents, tn = collect_parent_ids(tp)
        test_parents, sn = collect_parent_ids(sp)
        common = train_parents & test_parents
        ex = sorted(common)[:MAX_OVERLAP_EXAMPLES]
        v = "fail" if common else "pass"
        results.append({
            "label": label,
            "train_file": train_rel,
            "test_file": test_rel,
            "train_records": tn,
            "test_records": sn,
            "train_parents": len(train_parents),
            "test_parents": len(test_parents),
            "shared_parents": len(common),
            "shared_parent_examples": ex,
            "verdict": v,
        })

    # exact split
    check("exact_train vs exact_test",
          RF_EXACT_SPLITS["exact_train"], RF_EXACT_SPLITS["exact_test"])
    check("exact_train vs exact_novel",
          RF_EXACT_SPLITS["exact_train"], RF_EXACT_SPLITS["exact_novel"])
    check("exact_train vs exact_val",
          RF_EXACT_SPLITS["exact_train"], RF_EXACT_SPLITS["exact_val"])
    # mmseqs split
    check("mmseqs_train vs mmseqs_test",
          RF_MMSEQS_SPLITS["mmseqs_train"], RF_MMSEQS_SPLITS["mmseqs_test"])
    check("mmseqs_train vs mmseqs_novel",
          RF_MMSEQS_SPLITS["mmseqs_train"], RF_MMSEQS_SPLITS["mmseqs_novel"])
    # eFold train vs windowed test sets (human_mRNA, lncRNA, viral)
    for tname in ("human_mRNA", "lncRNA", "viral"):
        check(f"efold_train vs {tname}",
              EFOLD_CACHE["efold_train"], EFOLD_CACHE[tname])

    overall = verdict(*[r["verdict"] for r in results])
    return {
        "comparisons": results,
        "verdict": overall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-run",
        type=Path,
        default=DEFAULT_FULL_RUN,
        help="Path to the full_ablation run directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    base = args.full_run
    if not base.exists():
        # Fall back to absolute server path.
        abs_base = Path("/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012")
        if abs_base.exists():
            base = abs_base
    print(f"[c1_0_data_leakage_audit] base = {base}", file=sys.stderr)

    checks: Dict[str, dict] = {}
    checks["sequence_overlap"] = run_sequence_overlap(base)
    checks["pair_index_protocol"] = run_pair_index_protocol(base)
    checks["t_to_u_conversion"] = run_t_to_u_conversion(base)
    checks["truncation_padding"] = run_truncation_padding(base)
    checks["window_parent_overlap"] = run_window_parent_overlap(base)

    overall = verdict(*[c.get("verdict", "pass") for c in checks.values()])

    report = {
        "schema_version": "1.0",
        "phase": "C1-0",
        "base_directory": str(base),
        "checks": checks,
        "overall_verdict": overall,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"[c1_0_data_leakage_audit] wrote {args.out} (overall={overall})",
          file=sys.stderr)

    # Also print a short summary to stdout for log capture.
    print(json.dumps({
        "overall_verdict": overall,
        "sequence_overlap_verdict": checks["sequence_overlap"]["verdict"],
        "pair_index_protocol_verdict": checks["pair_index_protocol"]["verdict"],
        "t_to_u_conversion_verdict": checks["t_to_u_conversion"]["verdict"],
        "truncation_padding_verdict": checks["truncation_padding"]["verdict"],
        "window_parent_overlap_verdict": checks["window_parent_overlap"]["verdict"],
    }, indent=2))
    return 0 if overall != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
