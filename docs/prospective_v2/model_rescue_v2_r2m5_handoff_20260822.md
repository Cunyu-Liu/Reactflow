# ReactFlow-Delta Model Rescue v2 R2M5 → M6 handoff

日期：2026-08-22

## 终局裁决

- R2M3：`MODEL_RESCUE_V2_FAIL`
- Mean Gate：`MEAN_GATE_FAIL`；signed-delta MAE gain `+0.00214289`，相对改善
  `0.8833%`，16/20 puzzles 正向，但低于预冻结 `1%` 门槛。
- Calibration Gate：`CALIBRATION_GATE_PASS`；CRPS gain `+0.00547832`，20/20
  puzzles 正向；Candidate A/B point mean 与 signed-delta MAE 完全相同。
- R2M4：`NOT_RUN_PREREQUISITE_FAILED`，不得补跑。
- 模型资格：`CALIBRATION_BASELINE_ONLY`。
- 主合同路线：`M6 / BENCHMARK_ROUTE_LOCKED`。

## 允许进入稿件的表述

在 consumed-development 20-puzzle LOPO screen 中，冻结均值后的严格零均值残差校准改善
了 CRPS，且没有改变 point mean。这说明 residual distribution 中存在可用于概率校准的
信号，但不证明 mean prediction 达到预注册的实用改善门槛。

## 禁止表述

- 不得称 B1-MeanAligned 为通过 Gate 的 mutation-effect predictor improvement；
- 不得声称 Model Rescue v2 成功；
- 不得报告或暗示五 seed R2M4 confirmation；
- 不得把 calibration-only CRPS gain 写成 mean architecture gain；
- 不得声称 external replication、SOTA、mechanism、practical utility 或 publication PASS。

## M6 执行边界

- 关闭全部 model-rescue training；
- 不允许第三次 model-rescue amendment；
- B1 保持 mutation-effect 主基线；CalibratedResidual 只可作为 calibration baseline；
- 新 external outcome 继续锁定；
- 后续工作只进入 benchmark、measurement、复现和稿件 claim requalification。

## 冻结证据

- Qualification：`docs/prospective_v2/audit/r2m3_qualification_20260822.json`
- Human summary：`docs/prospective_v2/audit/r2m3_qualification_20260822.md`
- 完整 merged result：
  `/mnt/cunyuliu/reactflow_delta_model_rescue_v2/r2m3_screen_seed0/v2_result_seed0_merged.json`
- Prediction/checkpoint artifacts：
  `/mnt/cunyuliu/reactflow_delta_model_rescue_v2/r2m3_screen_seed0/`
