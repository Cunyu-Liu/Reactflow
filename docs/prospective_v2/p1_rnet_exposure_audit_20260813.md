# P1 增补：RibonanzaNet checkpoint exposure audit (2026-08-13)

## 结论
RibonanzaNet **本地无冻结 checkpoint** 可用 → 按合同 §8.2 / §7.2 标为
`EXPOSURE_UNKNOWN_DIRECT_REFERENCE`；RNet static-delta 直接基线暂无法纳入正式 direct 集。

## 证据（read-only）
- 源码存在：`/home/cunyuliu/ribonanzanet_src`（`Network.py` / `inference.py` / `run.py` / `configs`）。
- `inference.py` 加载 `models/model{i}.pt`；`run.py` 保存 `models/model{fold}.pt` —— 需要预训练权重。
- `find /home/cunyuliu /mnt/cunyuliu -iname "*.pt|*.ckpt|*.pth" | grep ribo` → **未找到任何 RNet 权重**。
- 无 `EXPOSURE_UNKNOWN_DIRECT_REFERENCE` 之外的 evidence。

## 影响
- 合同 §8.1 direct 集要求 frozen RNet WT-anchored mutant static-delta；当前因缺 checkpoint 未能运行。
- 正式 benchmark 必须使用本地可重放、冻结的 checkpoint（禁止动态 web service 输出）——故在 checkpoint 就绪前 RNet 保持 `EXPOSURE_UNKNOWN_DIRECT_REFERENCE`，不支撑 clean OOD/SOTA。
- 直接后果：P2 v2 / P3 v1 的 Direct*/B* 集合未含 RNet static-delta；需在 P3 后单独获取冻结 checkpoint 补齐该 comparator（或明确记录为无法资格化的 direct 参考）。

## 处置
- 不伪造 RNet 结果；不把源码存在当作可运行 checkpoint。
- 后续若获得冻结 checkpoint，须按 §7.2 分轴审计 sequence / WT profile / mutant profile / OpenKnot M2 outcome exposure 后方可作为 direct comparator。
