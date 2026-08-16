#!/usr/bin/env python3
"""Safe update of the M0-X window registry: register EPRO_DEV_06 (RUNNING)."""
import json
import hashlib
import datetime

p = "docs/governance/m0x_window_registry_20260804.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

if not any(i.get("iteration_id") == "EPRO_DEV_06" for i in d["iterations"]):
    d["iterations"].append({
        "iteration_id": "EPRO_DEV_06",
        "run_id": "epro_dev_06_20260805",
        "hypothesis_id": "m0x_h06_structure_aware_changer",
        "change_category": "structure_aware_contact_graph_feature_injection_on_proven_classifier",
        "prediction_changing": True,
        "counts_as_iteration": True,
        "status": "RUNNING",
        "registered_at_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "flat MLP changer classifier + structure-aware contact-graph features; "
                "long training with convergence/generalization tracking; single-method "
                "calibration; published ViennaRNA baseline",
    })
# Optionally mark DEV_05 as FAIL if it has wrapped (it is still running; leave RUNNING
# unless already terminal). Only set DEV_05 FAIL if it is not RUNNING.
for it in d["iterations"]:
    if it.get("iteration_id") == "EPRO_DEV_05" and it.get("status") == "RUNNING":
        pass  # still running; do not touch

d["consumed_iterations"] = len(d["iterations"])
d["last_updated_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
    f.flush()
import os
os.replace(tmp, p)

with open(p, "rb") as f:
    b = f.read()
print("iterations:", len(d["iterations"]))
print("bytes:", len(b))
print("sha:", hashlib.sha256(b).hexdigest())