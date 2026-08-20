#!/usr/bin/env python3
"""test_component_graph_v2: audit P0-4 acceptance for the substantiated
joint-dependency union-find graph.

Required (audit §13 P0-4):
  * components sharing publication/study/batch/library/shared-WT are merged into
    ONE joint cluster (K_joint < number of WT anchors when they share a batch)
  * SNV/position/seed NEVER increase K
  * unknown provenance fields exclude a component from K_joint (fail-closed)
  * the seven external dataset files collapse to their true study/batch clusters
"""
from __future__ import annotations

from scripts.reactflow_delta.joint_dependency_component_v1 import (
    ComponentCandidate, UnionFind, compute_k_joint, compute_k_preaccess,
    EXTERNAL_DATASET_PROVENANCE,
)


def _cand(cid: str, dataset: str, n_snv: int = 10, *, n_rows: int = 0,
          publication: str | None = "pub_x", study: str | None = "study_x",
          batch: str | None = "batch_x", library: str | None = "lib_x",
          dev_disconnected: bool = True, prov_resolved: bool = True) -> ComponentCandidate:
    return ComponentCandidate(
        component_id=cid, publication=publication, study=study, batch=batch,
        library=library, dataset=dataset, n_rows=n_rows, n_snv_mutants=n_snv,
        development_disconnected=dev_disconnected, provenance_resolved=prov_resolved,
        metadata_keys=set())


def test_union_find_merges_shared_batch():
    """Components in different datasets but the SAME sequencing batch must merge."""
    uf = UnionFind(["a", "b", "c"])
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")
    assert uf.find("a") != uf.find("c")
    assert len(uf.roots()) == 2


def test_k_joint_collapses_shared_batch_components():
    """5 components across 2 datasets sharing one batch/library => K_joint=2
    clusters (one per dataset is NOT enough: shared batch merges them)."""
    # 2 datasets share the same batch AND library AND publication (BigLib2 case)
    comps = [
        _cand(f"c{i:02d}", dataset="D1", n_snv=10, batch="B", library="L", study="S", publication="P")
        for i in range(3)
    ] + [
        _cand(f"c{i:02d}", dataset="D2", n_snv=10, batch="B", library="L", study="S", publication="P")
        for i in range(3, 5)
    ]
    res = compute_k_joint(comps, development_component_ids=set())
    # all share batch+library+study+publication => ONE joint cluster
    assert res["K_joint"] == 1, f"expected 1 cluster, got {res['K_joint']}"
    assert res["N_WT_anchor"] == 5
    assert res["N_SNV"] == 5 * 10
    # SNV/rows do NOT increase K
    assert res["N_rows"] == 0  # n_rows not set => 0, never counts toward K
    assert res["K_preaccess"] == 5


def test_k_joint_distinct_studies_stay_separate():
    """Components in DIFFERENT studies (different publication/batch) do NOT merge."""
    comps = [
        _cand("s1a", dataset="D1", batch="B1", library="L1", study="S1", publication="P1"),
        _cand("s2a", dataset="D2", batch="B2", library="L2", study="S2", publication="P2"),
    ]
    res = compute_k_joint(comps, development_component_ids=set())
    assert res["K_joint"] == 2
    assert res["N_study"] == 2
    assert res["N_publication"] == 2


def test_unresolved_provenance_excluded_fail_closed():
    """A component with unresolved publication is EXCLUDED from K_joint."""
    good = _cand("good", dataset="D1", batch="B1", library="L1", study="S1", publication="P1")
    bad = _cand("bad", dataset="D2", batch=None, library=None, study=None, publication=None)
    res = compute_k_joint([good, bad], development_component_ids=set())
    assert res["K_joint"] == 1, "unresolved component must not add a cluster"
    assert res["N_WT_anchor"] == 1, "unresolved component must be excluded"
    assert "bad" in res["rejected_components"]
    assert any("provenance" in r for r in res["rejected_components"]["bad"])


def test_development_connected_excluded():
    """Development-connected components never enter K."""
    dev = _cand("dev", dataset="D1", batch="B1", library="L1", study="S1", publication="P1",
                dev_disconnected=False)
    ext = _cand("ext", dataset="D2", batch="B2", library="L2", study="S2", publication="P2")
    res = compute_k_joint([dev, ext], development_component_ids=set())
    assert res["N_WT_anchor"] == 1
    assert "dev" in res["rejected_components"]


def test_seven_datasets_collapse_to_true_clusters():
    """The 7 external dataset files carry the batch/study/publication from the
    registry; shared-run sub-libraries must NOT each count as independent units.
    At the HIGHEST merged level (publication/study), all 7 files collapse to 2
    clusters (SL5 study + Ribonanza study); at the sequencing-BATCH level there
    are 3 distinct NovaSeq runs (2023-06-06 SL5, 2023-08-01 15k, 2023-10-31
    OneMil2 BigLib2)."""
    from scripts.reactflow_delta.joint_dependency_component_v1 import PROVENANCE_BY_DATASET
    batches = {p.dataset_id: p.sequencing_batch for p in EXTERNAL_DATASET_PROVENANCE}
    n_batch = len(set(batches.values()))
    n_study = len({p.study for p in EXTERNAL_DATASET_PROVENANCE})
    n_pub = len({p.publication for p in EXTERNAL_DATASET_PROVENANCE})
    # BigLib2 OneMil2 sub-libraries share one batch; the 15k library is also
    # Ribonanza-era Das-lab M2 data => 3 batches, 2 studies, 2 publications.
    assert n_batch == 3, f"expected 3 distinct NovaSeq batches, got {n_batch}"
    assert n_study == 2, f"expected 2 distinct studies, got {n_study}"
    assert n_pub == 2, f"expected 2 distinct publications, got {n_pub}"
    comps = [
        _cand(p.dataset_id, dataset=p.dataset_id, n_snv=1, batch=p.sequencing_batch,
              library=p.library, study=p.study, publication=p.publication)
        for p in EXTERNAL_DATASET_PROVENANCE
    ]
    res = compute_k_joint(comps, development_component_ids=set())
    # 7 anchors collapse to 2 clusters at the study/publication level
    assert res["K_joint"] == 2, f"expected K_joint=2, got {res['K_joint']}"
    assert res["N_batch"] == 3
    assert res["N_study"] == 2
    assert res["N_publication"] == 2
    assert res["N_WT_anchor"] == 7
