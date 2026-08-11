#!/usr/bin/env python3
"""Coverage check: do the p2_cache profiles cover the benchmark_v3 registry pairs?"""
import pickle, json

d = pickle.load(open("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl", "rb"))
rec_index = d["rec_index"]
avail = set()
for (acc, pidx, aname) in rec_index:
    avail.add((acc, pidx))
print("rec_index profiles available:", len(avail))

pairs = []
for line in open("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl"):
    if not line.strip():
        continue
    pairs.append(json.loads(line))
print("primary pairs:", len(pairs))

cover_wt = 0
cover_both = 0
no_mut = []
for p in pairs:
    acc = p["source_accession"]
    wt = p["wt_profile_index"]
    mut = p["mutant_profile_index"]
    has_wt = (acc, wt) in avail
    has_mut = (acc, mut) in avail
    if has_wt:
        cover_wt += 1
    if has_wt and has_mut:
        cover_both += 1
    else:
        no_mut.append("{}:wt{}:mut{}".format(acc, wt, mut))
print("pairs with wt profile:", cover_wt, "/", len(pairs))
print("pairs with both wt+mut:", cover_both, "/", len(pairs))
print("sample uncovered:", no_mut[:5])