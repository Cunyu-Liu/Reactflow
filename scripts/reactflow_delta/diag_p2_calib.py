#!/usr/bin/env python3
"""CPU diagnostic: are reported errors miscalibrated vs actual cross-replicate noise?

For each WT replicate group (same seq+probe+temp), compare the reported per-position
error to the ACTUAL cross-replicate standard deviation of reactivity. If the reported
error is far smaller than the empirical replicate scatter, the caller's z-scores and
null are inflated (miscalibrated errors) -> normalization/error-recalibration fixes it.
"""
import math
import pickle
import sys
import numpy as np
sys.path.insert(0, "/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta")
from run_p2_v1 import build_rep_groups, sanitize_records
CACHE = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl"

def finite(x): return isinstance(x,(int,float)) and math.isfinite(x)

def main():
    with open(CACHE,"rb") as fh: cache=pickle.load(fh)
    rec_index=cache["rec_index"]; pool=set(cache["pool"])
    sanitize_records(rec_index)
    groups=build_rep_groups(rec_index, study_whitelist=pool)
    ratios=[]
    n=0
    for g in groups:
        k=g.n_replicates
        if k<2: continue
        length=min(len(p) for p in g.wt_profiles)
        mask=g.eligibility_mask
        elig=[i for i in range(min(length,len(mask))) if mask[i]]
        for i in elig:
            vals=[g.wt_profiles[r][i] for r in range(k) if finite(g.wt_profiles[r][i])]
            errs=[g.wt_errors[r][i] for r in range(k) if finite(g.wt_errors[r][i])]
            if len(vals)<2: continue
            sd=np.std(vals,ddof=1)
            rep_err=np.sqrt(np.mean(np.square(errs))) if errs else 0.0
            if sd>0 and rep_err>0:
                ratios.append(sd/rep_err)
                n+=1
    a=np.asarray(ratios)
    print(f"[calib] positions_with_rep={n}")
    if a.size:
        print(f"  empirical_sd / reported_err : median={np.median(a):.2f} mean={np.mean(a):.2f} "
              f"p90={np.percentile(a,90):.2f} frac>1.0={np.mean(a>1.0):.3f} frac>5.0={np.mean(a>5.0):.3f} max={np.max(a):.1f}")

if __name__=="__main__":
    main()
