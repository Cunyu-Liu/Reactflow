#!/usr/bin/env python3
import json, sys
out = sys.argv[1]
d = json.load(open(f"{out}/run_manifest.json"))
print("PASS:", json.dumps(d["pass"], indent=2))
print("\n=== comparison_table ===")
for k, v in d["comparison_table"].items():
    print(f"  {k}: auprc={v.get('study_macro_auprc')} params={v.get('param_count')}")
ca = d["evaluation"]["structure_aware_changer"]
print("\n=== structure_aware metrics ===")
print("  train_auprc=", ca["train_study_macro_auprc"])
print("  gen_gap=", ca["gen_gap"])
print("  calibrated_auprc=", ca["study_macro_auprc_calibrated"])
print("  calib_selected=", ca.get("calibration_selected_method", "n/a"))
print("  gain_vs_p2=", ca["auprc_gain_vs_p2"])
print("  gain_vs_vienna=", ca["auprc_gain_vs_vienna"])
print("  gain_vs_wt=", ca["auprc_gain_vs_wt"])
print("  calibration:")
for k, v in ca["calibration"].items():
    if isinstance(v, dict):
        print(f"    {k}: brier={v.get('brier')} logloss={v.get('log_loss')} ece={v.get('ece')}")