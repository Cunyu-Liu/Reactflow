# Target-Identity-Corrected Baseline Rebuild

本阶段与 V7M1 outcome-blind cache 并行，但使用独立 worktree、authority 和 artifact directory。唯一目标是在修复后的 exact puzzle-method-mutation target identity 上从头重建三套固定线性 comparator：`direct18`、`v5_feature30` 和 `v6_feature41`。

它不是模型选择阶段，也不允许读取 RiNALMo dependency、修改 alpha、搜索特征、选择 puzzle、选择 method 或复用旧 prediction/model。三个模型都在相同 corrected outer-train target、相同 method-balanced权重与相同 20-fold LOPO 上拟合。

所有 folds 只输出 registered full-construct prediction ledger。只有 20/20 folds 完整、无重复、schema 合格且 `v5_feature30` 在两条独立构建路径中逐 key replay 到 `1e-12`，才允许一次性 target join。完整评分报告三组预定义比较：direct18→feature30、feature30→feature41、direct18→feature41；不据此改变 V7 候选或 Gate。

本阶段 exact PASS 只表示 corrected comparator 已重建且可以用于后续 V7M2，不表示 SOTA、publication readiness 或任何旧 v5/v6主张恢复。

