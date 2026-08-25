# Model Rescue v11 implementation plan

## Scope

Implement the frozen V11 feature41-anchored point representation, exact unanchored null, fixed V10 asymmetric residual calibration, score-blind artifacts, and mechanical gates. No training is authorized until the relevant authority transition is committed.

## V11M1 — implementation and invariants

Files to add:

- `scripts/reactflow_delta/model_rescue_v11.py`
- `scripts/reactflow_delta/run_model_rescue_v11.py`
- `scripts/reactflow_delta/merge_model_rescue_v11.py`
- `scripts/reactflow_delta/score_model_rescue_v11.py`
- `scripts/reactflow_delta/qualify_model_rescue_v11.py`
- `scripts/reactflow_delta/qualify_model_rescue_v11_smoke.py`
- persistent smoke and screen controllers
- focused V11 tests

The failure each focused test must detect and the action if it occurs:

1. Candidate/null parameter or initialization mismatch would confound anchor attribution; block smoke.
2. A model difference beyond the fixed skip multiplier would invalidate the nested comparison; block smoke.
3. Missing FFN, wrong width/depth, or wrong parameter count would mean the implemented model is not the frozen candidate; block smoke.
4. Calibration gradient reaching point parameters would recreate mean/calibration confounding; block smoke.
5. CDF(point) not equal to 0.5 would let calibration move the point median; block smoke.
6. Held target/error/mask changing a prediction would be outcome leakage; block all scientific runs.
7. Pooled-mutant loss replacing cell-balanced loss would change the estimand; block smoke.
8. Duplicate/missing fold merge would make the complete-universe qualifier invalid; block scoring.
9. A scorer accepting target/score fields in prediction artifacts would break prediction/score separation; block scoring.
10. A qualifier accepting any failed headline Gate would permit manual verdict inflation; block authority transition.

V11M1 PASS only after the focused tests and contract validator pass and the exact candidate/null trainable parameter count is recorded in the machine contract and decision ledger.

## V11M2 — real-data engineering smoke

Run folds 0/1, seed 0, 3 point epochs and 3 calibration epochs in a persistent session. It may inspect engineering losses only to diagnose non-finite execution; it must not compute scientific scores. Complete prediction-only artifacts are qualified mechanically. Engineering failures may be fixed and missing folds rerun without changing architecture, Gate, data, seed, or epochs.

## V11M3 — complete seed-0 screen

Run folds 0–19 with non-overlapping fold ownership on any safe GPU0–7. Controllers skip complete artifacts and refuse overwrite. Before 20/20, monitoring is limited to session existence, log mtime/size, artifact filename count, GPU health, and non-metric errors. After 20/20, merge prediction-only files, run the frozen scorer once, then the frozen qualifier once.

The seed-0 feature41-asymmetric comparator is the authoritative terminal V10 same-fold, same-seed, 40-epoch checkpoint and prediction-only distribution. It is loaded and replayed exactly rather than retrained and expected to reproduce a non-deterministic GPU optimization path bitwise. This affects only the comparator; the V11 anchored candidate and exact unanchored null remain freshly trained under V11.

## V11M4 — five-seed formal confirmation

Only exact `V11M3_TOP_JOURNAL_SCREEN_PASS` can open V11M4. Run seeds0–4 on the same 20 folds for candidate, null, and feature41 residual comparators. Assemble equal-seed mixtures and apply all frozen screen gates plus the 4/5 seed direction requirement.

The formal implementation is frozen before V11M3 score access. `run_model_rescue_v11_formal_controller.sh` may only produce the 100 fold×seed prediction artifacts, the complete unscored merge, and the equal-seed prediction-only assembly. `assemble_model_rescue_v11_formal.py` gives every seed one-fifth total mass and therefore creates ten Gaussian components per model. Scientific target access remains a separate authority transition: only then may `score_model_rescue_v11_formal.py` run once, followed by one mechanical call to `qualify_model_rescue_v11_formal.py`. The prediction controller never scores automatically.

## V11M5 — freeze and handoff

Freeze code, contracts, prediction artifacts, score, qualifier, plots/tables, and claim map. Close all training. PASS remains post-hoc development until a separate sealed external amendment; FAIL returns to the benchmark manuscript with the negative result intact.
