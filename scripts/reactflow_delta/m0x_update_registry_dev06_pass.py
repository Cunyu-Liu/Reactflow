#!/usr/bin/env python3
"""Safe update: mark EPRO_DEV_06 PASS (keep DEV_05 RUNNING)."""
import json, hashlib, datetime
p = "docs/governance/m0x_window_registry_20260804.json"
d = json.load(open(p, encoding="utf-8"))
for it in d["iterations"]:
    if it.get("iteration_id") == "EPRO_DEV_06":
        it["status"] = "PASS"
        it["result"] = {
            "study_macro_auprc": 0.7353243279593717,
            "auprc_gain_vs_p2_ci_low": 0.033643179831694514,
            "auprc_gain_vs_p2_point": 0.04172562435626159,
            "auprc_gain_vs_vienna_ci_low": 0.21045022236541128,
            "auprc_gain_vs_vienna_point": 0.28188094356413246,
            "note": "structure-aware features: marginal +0.0009 AUPRC vs DEV_04 (0.7353 vs 0.7344); "
                    "confirms changer-detection saturates; strong vs published ViennaRNA physics (+0.28)",
        }
        it["completed_at_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
d["last_updated_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False); f.flush()
import os; os.replace(tmp, p)
b = open(p, "rb").read()
print("sha:", hashlib.sha256(b).hexdigest())
for it in d["iterations"]:
    print(it.get("iteration_id"), it.get("status"))