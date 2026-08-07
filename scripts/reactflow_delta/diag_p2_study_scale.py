#!/usr/bin/env python3
"""CPU diagnostic: per-study reactivity/error scale -> cross-study heterogeneity."""
import math
import pickle
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta")
CACHE = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl"

def finite(x): return isinstance(x,(int,float)) and math.isfinite(x)

def main():
    with open(CACHE,"rb") as fh: cache=pickle.load(fh)
    rec_index=cache["rec_index"]; pairs=cache["pairs"]; pool=set(cache["pool"])
    by_study=defaultdict(lambda: {"r":[],"e":[]})
    for rec in rec_index.values():
        sa=rec.get("source_accession") or ""; st=sa.split("_")[0]
        if st not in pool: continue
        tf=rec.get("reactivity_layers",{}).get("train_frozen",{}) or rec.get("reactivity_layers",{}).get("raw",{})
        err=tf.get("error") or []; react=tf.get("reactivity") or []
        for r,e in zip(react,err):
            if finite(r): by_study[st]["r"].append(float(r))
            if finite(e): by_study[st]["e"].append(float(e))
    print(f"{'study':10s} {'n_rec':>6s} {'react_med':>10s} {'react_max':>10s} {'err_med':>10s} {'err_max':>10s}")
    for st in sorted(pool):
        r=np.asarray(by_study[st]["r"]); e=np.asarray(by_study[st]["e"])
        if r.size==0:
            print(f"{st:10s}  {len(rec_index):6d}  no data"); continue
        nr=sum(1 for x in rec_index.values() if (x.get('source_accession') or '').split('_')[0]==st)
        print(f"{st:10s}  {nr:6d}  {np.median(r):10.3f} {np.max(r):10.3f}  {np.median(e):10.4f} {np.max(e):10.3f}")

if __name__=="__main__":
    main()
