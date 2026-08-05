#!/usr/bin/env python3
"""Safe update of the M0-X window registry: register EPRO_DEV_05 (RUNNING)."""
import json
import hashlib
import datetime

p = "docs/governance/m0x_window_registry_20260804.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

if not any(i.get("iteration_id") == "EPRO_DEV_05" for i in d["iterations"]):
    d["iterations"].append({
        "iteration_id": "EPRO_DEV_05",
        "run_id": "epro_dev_05_20260805",
        "hypothesis_id": "m0x_h05_epro_changer_classifier",
        "change_category": "EPRO_backbone_changer_classifier_end_to_end_long_training",
        "prediction_changing": True,
        "counts_as_iteration": True,
        "status": "RUNNING",
        "registered_at_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "EPRO structure backbone + focal changer head; large capacity; long training; "
                "single-method calibration; published ViennaRNA baseline",
    })

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