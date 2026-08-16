# ReactFlow-Δ EPRO Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task-by-task.  
> 中文名称：ReactFlow-Δ 平衡态扰动响应算子科研合同与分阶段执行 Goal  
> 合同版本：V3.0  
> 冻结日期：2026-07-29  
> 文献检索截止：2026-07-29  
> 适用仓库：`Cunyu-Liu/Reactflow`  
> 当前只读旧现场：`/home/cunyuliu/reactflow_c1_3_stage_20260722`  
> 建议新工作树：`/home/cunyuliu/reactflow_delta_goal_20260729`  
>
> **Goal:** 用公开、可审计的成对 WT–mutant chemical-probing 数据，建立并严格检验一个由 RNA 平衡态物理、扰动响应数学和真实测量过程共同约束的专用模型，预测突变引起的实验 reactivity 响应。  
>
> **Architecture:** 主模型不是普通 Transformer、PairFormer 或 GNN 的拼接，而是 `ReactFlow-Δ EPRO`（Equilibrium Perturbation Response Operator）：端点平衡态表征 → 局部突变强迫 → 稳定易感性传播 → 非线性构象切换 → probe-specific 观测。恒等性、交换反对称、局部强迫和传播稳定性由构造保证。  
>
> **Tech Stack:** Python 3.11、PyTorch、CUDA、ViennaRNA/RNAstructure/LinearPartition、NumPy、SciPy、pandas/pyarrow、pytest、JSON/JSONL/YAML、Git/GitHub。

---

## 0. 文档权威性、继承关系与冻结决策

### 0.1 权威性

本文件完整取代：

- `ReactFlow分阶段执行提示词.md`；
- `ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v2_20260729.md` 中的模型架构、假设、模型阶段、架构消融和执行 Goal；
- 旧 C1-0 至 C1-6 的静态结构 SOTA 主线。

V2 中以下经过讨论的内容继续有效，并已完整吸收到本合同：

- PCCNG 冻结；
- ReactFlow 为唯一主项目；
- 不依赖新湿实验；
- 不依赖外部专家标签；
- 数据审计先于模型；
- 不自动追加多 seed；
- learned training 必须使用 GPU；
- 不频繁轮询训练；
- 等待期间并行做无写冲突工作；
- 每个独立任务结束后测试、聚焦 commit 并 push GitHub；
- 失败只能 fail-forward，不能降低 Gate 或隐藏失败。

### 0.2 冻结决策

1. PCCNG 不再投入新增数据、训练、工程或论文资源。
2. ReactFlow-Δ 是唯一主线。
3. 研究对象是实验 `Δreactivity`，不是 teacher `ΔBPP`。
4. 第一版只做同长度、单碱基 substitution、condition-matched WT–mutant pair。
5. 主 endpoint 只使用未编辑、对齐且 probe eligibility 不变的位置。
6. P1 sequence-only 是主要部署协议；P2 WT-anchored 是实用增强协议。
7. 不以“联合模型更复杂”为创新；必须检验专用物理响应算子是否优于同容量通用模型与独立静态差分。
8. 完成 D0–D2 数据 Gate 前，禁止启动任何 learned training。
9. 旧 StaticPairFormer 可复用工程组件，但不得自动成为新模型 backbone。
10. 不因项目名为 ReactFlow 就强行使用 normalizing flow、diffusion 或 RL。

### 0.3 当前远端事实快照

快照日期：2026-07-29。此段只记录现场，不授权修改旧现场。

- 旧工作树：`/home/cunyuliu/reactflow_c1_3_stage_20260722`
- 分支：`trae/c1-3-static-scale`
- HEAD：`2cdf9faf02f075b6f9289e84411a1ae60ff8d45a`
- 旧工作区存在未提交修改和备份文件。
- 旧 C1-3 v4 以固定 seed 0、3 GPU 运行。
- 旧全量评测约为：
  - validation MEA F1：`0.4013`
  - test MEA F1：`0.4049`
  - novel MEA F1：`0.4004`
  - EternaFold 同协议约：`0.7039 / 0.7036`
- 旧 quick evaluation 约 `0.80` 与全量结果冲突，禁止作为科学证据。
- 旧注册表的 317,039 条记录主要来自静态/proxy 数据，不能代表真实 intervention pair。

旧产物统一标记：

> `historical engineering assets / historical negative evidence`

不得删除，但不得继承为 EPRO 的科学结果。

---

# 一、最终科学问题

## 1.1 任务定义

给定：

- WT 或母本 RNA 序列 \(x_w\)；
- mutant RNA 序列 \(x_m\)；
- 编辑集合 \(e\)，第一版为一个 substitution；
- 条件 \(c\)，包括 probe、温度、配体、缓冲液、体内/体外和必要批次元数据；

预测：

\[
G_\theta(x_w,x_m,e,c)\rightarrow \Delta r
\]

其中：

\[
\Delta r=r(x_m,c)-r(x_w,c)
\]

\(r\) 是真实实验测得的 DMS、SHAPE/2A3 或兼容 chemical-probing reactivity。

### P1：sequence-only

\[
G_\theta(x_w,x_m,e,c)\rightarrow \Delta r
\]

目标：没有目标 RNA 实验 profile 时进行 retrospective variant screening。

### P2：WT-anchored

\[
G_\theta(x_w,x_m,e,c,r_w^{obs})\rightarrow \Delta r
\]

目标：已有一个 WT profile，但无法为每个 mutant 做实验时预测 mutant response。

P1 与 P2 必须：

- 共用同一个物理响应主干；
- 使用独立结果表；
- 不混合指标；
- P2 严禁读取 mutant 实验 profile。

## 1.2 精确科学问题

> 在严格跨研究、跨母本外推下，由平衡态、局部突变强迫、结构易感性传播和 probe 观测共同约束的 EPRO，能否比 zero-change、热力学双折叠、强 sequence-to-reactivity 独立差分、matched Siamese 和同参数通用 paired network 更准确且更校准地预测未编辑位置上的实验结构响应？

## 1.3 价值

静态结构模型回答：

> 每条序列各自可能形成什么结构？

EPRO 回答：

> 一次具体编辑，在具体实验条件下，会把母本的实验可见状态改变多少、改变到哪里、能否传播到远端，以及预测是否可靠？

潜在用途：

- mRNA/UTR 最小编辑结构风险排序；
- riboSNitch 候选优先级；
- riboswitch 与构象切换分析；
- 补偿突变与 rescue 排序；
- 为后续 mRNA-EditFlow 提供带不确定性的实验响应 oracle。

禁止把 retrospective public-data prediction 写成 prospective biological validation。

---

# 二、现有证据、研究空缺与创新边界

## 2.1 已定位证据

1. Mutate-and-map/M2-seq 表明，局部突变可以释放配对伙伴，并在远端产生可测 chemical-probing 响应。
2. Chemical probing 是构象 ensemble、probe chemistry 和实验条件的联合观测，不是唯一二级结构真值。
3. 热力学 partition function 提供 RNA ensemble、BPP、unpaired probability、ensemble free energy 和 entropy。
4. 可微 partition function 已证明可以把 RNA 热力学与梯度学习连接，但当前仍有内存和模型边界。
5. RibonanzaNet 已覆盖“卷积 + self-attention + pair representation + triangle update”的静态 reactivity 路线。
6. BPfold 已覆盖“热力学能量图进入 attention”的静态结构路线。
7. classSNitch 已覆盖“读取 WT/mutant 两条实测 trace 后分类”的路线。
8. VariantFoldRNA 已覆盖“可扩展 genome-wide riboSNitch pipeline”的路线。

## 2.2 本项目不能主张

- 首次研究 RNA 突变结构效应；
- 首次使用 WT–mutant probing；
- 首次用机器学习分析 RNA 变体；
- 首次把热力学先验加入深度模型；
- 首次使用成对或反对称神经网络；
- 首次建立可扩展 riboSNitch 工具；
- 首次从序列预测 chemical reactivity；
- 真实物理能量已被模型识别；
- 预测结果等于真实结构或因果效应。

## 2.3 条件性创新

若数据与结果 Gate 通过，可主张：

1. **任务创新**  
   从两次静态预测差分转向 paired experimental response prediction。

2. **物理架构创新**  
   把突变限制为局部 node/edge forcing，把远端响应限制为经稳定 susceptibility operator 传播得到。

3. **数学架构创新**  
   由构造保证 no-edit identity、endpoint-swap antisymmetry、forcing support 和 solver stability。

4. **观测建模创新**  
   共享潜在 accessibility response，但使用 probe-specific monotone observation model 和显式 measurement noise。

5. **评测创新**  
   以未编辑位置 trans-response、leave-study-out 和 leave-parent-out 为主，而不是突变位点或随机 pair split。

6. **可证伪比较**  
   专用 EPRO 必须同时战胜：
   - strongest executable independent difference；
   - matched generic paired model；
   - local-only perturbation model。

## 2.4 可发表性

| 条件 | 论文形态 | 合理定位 |
|---|---|---|
| Tier A、多研究外测、EPRO 优于全部强基线，物理消融成立 | 方法 + benchmark | NAR、Nature Communications、强 Bioinformatics 方向 |
| Tier B、单一外部研究、EPRO 有稳定限定域增益 | 限定域方法/资源 | PLOS Computational Biology、Bioinformatics、NAR Genomics and Bioinformatics |
| EPRO 不优于静态差分，但形成无泄漏 benchmark 与物理失败分析 | benchmark/negative result | NAR Genomics and Bioinformatics、Database、GigaScience |
| 数据低于 Tier B | 数据审计 | 不训练深度 headline |

投稿前 30 天内必须更新检索。没有系统综述证据不得写 `first`。

---

# 三、从三个世界推导架构

## 3.1 物理世界

### 已知物理事实

对序列 \(x\) 和条件 \(c\)，平衡近似下的构象分布可写为：

\[
p(s\mid x,c)=\frac{\exp[-\beta E(s;x,c)]}{Z(x,c)}
\]

位置 \(i\) 的潜在 accessibility 是构象 observable 的 ensemble average：

\[
\bar a_i(x,c)=\mathbb E_{s\sim p(s\mid x,c)}[a_i(s)]
\]

探针读数不是 \(\bar a_i\) 本身，而是：

\[
r_i\sim \mathcal O_{p,c,b_i}(\bar a_i,\text{noise})
\]

其中 \(p\) 是 probe，\(b_i\) 是碱基身份。

突变改变局部 stacking、pair compatibility 和 ensemble energy，可能：

- 只产生局部变化；
- 释放配对伙伴；
- 沿 helix/contact 传播；
- 在接近多稳态时触发构象切换。

### 物理假设

- 第一版主要研究近平衡、condition-matched 数据。
- in vitro 是主物理域；in vivo 单独报告。
- learned latent 只能称 `energy-like`、`susceptibility-like`，除非与物理能量实验独立校准。
- 热力学预测是 prior，不是实验真值。

### 物理架构要求

1. 突变只在编辑位点及其直接 motif/contact 边产生初始 forcing。
2. 未编辑远端位置不能直接注入 mutation feature。
3. 远端响应必须通过显式传播算子产生。
4. probe chemistry 必须在 observation layer，而不是和结构潜变量混为一体。
5. 结构脆弱性与构象切换必须可被单独输出和消融。

## 3.2 数学世界

### 必然约束 MATH-1：恒等

\[
G(x,x,\varnothing,c)=0
\]

必须由构造满足，不能只靠 identity loss。

### 必然约束 MATH-2：交换反对称

同一 condition 和 normalization domain 下：

\[
G(x_a,x_b,e,c)=-G(x_b,x_a,e^{-1},c)
\]

必须由构造满足。

### 必然约束 MATH-3：保守差分

真实 \(\Delta r\) 是端点状态之差。对同一条件下的三元组：

\[
G(a,b)+G(b,d)\approx G(a,d)
\]

由于实验噪声和归一化，该关系作为 diagnostic 和可选弱正则，不作所有数据上的硬约束。

### 线性响应

小能量扰动下：

\[
\delta \bar a_i
\approx
-\beta\,\mathrm{Cov}(a_i,\delta E)
\]

因此远端响应可理解为局部 forcing 经 susceptibility 传播。

### 稳定传播

定义局部强迫 \(b\) 与背景传播核 \(K\)：

\[
h^{(t+1)}=b+Kh^{(t)}
\]

约束：

\[
\rho(K)<1
\]

则：

\[
h_{\text{lin}}=(I-K)^{-1}b
\]

存在、稳定且可解释。

### 反向一致性

交换端点时：

- 对称背景 \(K\) 不变；
- 局部 forcing \(b\) 变号；
- \(h\) 变号；
- 最终 \(\Delta r\) 变号。

## 3.3 真实世界

真实公开数据具有：

- study/probe/condition 异质性；
- 大量缺失 metadata；
- construct 多但有效 pair 少；
- replicate/no-edit control 稀缺；
- profile missingness、SNR 与测量误差；
- 同一 parent 大量 mutant 造成泄漏风险；
- 外部预训练污染；
- 长度、显存和运行时间限制；
- 无新湿实验、无专家标签。

### 真实世界架构要求

1. primary model 不使用 free study-ID embedding。
2. DMS、SHAPE/2A3 共用潜在响应，但使用独立 observation head。
3. 缺失位置使用 mask，不填 0。
4. measurement noise 和 model uncertainty 分开。
5. 稀疏图复杂度目标为 \(O(LK)\)，不默认 dense \(O(L^2)\) storage 或 \(O(L^3)\) triangle stack。
6. P1 与 P2 共用主干。
7. 模型规模由独立 parent/study 数决定，不由 nucleotide label 总数虚增。

---

# 四、专用模型：ReactFlow-Δ EPRO

## 4.1 总体定义

EPRO 不是“一个 backbone 加一个 head”，而是五个受约束算子的组合：

1. `Endpoint State Operator`
2. `Local Mutation Forcing Operator`
3. `Stable Susceptibility Operator`
4. `Odd Nonlinear Switch Operator`
5. `Probe Observation Operator`

普通卷积、MLP、attention 或 sparse message passing 只允许作为这些算子的数值实现。

## 4.2 模块 A：Endpoint State Operator

对 WT 与 mutant 分别构建端点状态：

\[
z_q=\mathcal E(x_q,c),\quad q\in\{w,m\}
\]

每个端点至少包含：

- sequence local motif state；
- unpaired probability prior；
- sparse BPP/contact candidates；
- ensemble free energy；
- positional pairing entropy；
- total ensemble diversity；
- canonical pair compatibility；
- condition features。

优先使用：

- ViennaRNA/RNAplfold；
- LinearPartition；
- 可选 RNAstructure partition；
- bounded learnable correction。

约束：

- thermodynamic features 离线生成并记录版本/hash；
- learned correction 不得被写成真实 kcal/mol；
- 主模型必须有 `no_thermo_prior` 消融；
- test 不得参与 correction 训练。

## 4.3 模块 B：Local Mutation Forcing Operator

计算有向 forcing：

\[
b=\mathcal B(z_w,z_m,e,c)
\]

forcing 分为：

- node forcing \(b_i\)；
- edge forcing \(b_{ij}\)。

允许输入：

- WT→mutant substitution type；
- 编辑位置局部 motif；
- stacking-energy difference；
- incident pair-compatibility change；
- incident sparse-BPP change；
- local entropy/free-energy change；
- condition。

硬约束：

1. \(x_w=x_m\Rightarrow b=0\)。
2. 交换端点时 \(b\rightarrow-b\)。
3. node forcing 只允许出现在 edit-centered 局部窗口。
4. edge forcing 只允许出现在与 edit 窗口相连的边。
5. 远端位置不得直接读取 mutation token。

`forcing_support_mask` 必须作为可审计 artifact 输出。

## 4.4 模块 C：Stable Susceptibility Operator

构建端点交换不变的背景：

\[
\bar z=\mathrm{Sym}(z_w,z_m)
\]

例如：

\[
\bar z=\left(z_w+z_m,\ z_w\odot z_m,\ |z_m-z_w|\right)
\]

由 \(\bar z\) 构建稀疏传播核：

\[
K=\mathcal K(\bar z,c)
\]

图边只来自：

- 相邻/局部 sequence edges；
- WT/mutant sparse contact union；
- 可选实验独立结构边；
- 明确版本化的 thermodynamic candidates。

稳定约束：

- `K` endpoint-swap invariant；
- `K` 稀疏；
- `K` 行范数或谱范数受控；
- \(\rho(K)\le \rho_{max}<1\)；
- solver residual 必须记录。

响应：

\[
h_{\text{lin}}=(I-K)^{-1}b
\]

允许固定次数 Neumann iteration 或可微稀疏 solver，但必须满足：

\[
\frac{\|h-b-Kh\|}{\|b\|+\epsilon}<\epsilon_{\text{solver}}
\]

## 4.5 模块 D：Odd Nonlinear Switch Operator

线性响应适合小扰动，但 riboSNitch/strand displacement 可能产生有限幅度构象切换。

定义交换不变的 fragility：

\[
f=\mathcal F(\bar z,|b|,c)
\]

fragility 可使用：

- pairing entropy；
- ensemble diversity；
- competing-pair mass；
- WT/mutant contact disagreement；
- \(|b|\)；
- rescue lineage 特征。

switch gate：

\[
\pi=\sigma(f)
\]

非线性响应：

\[
h_{\text{nl}}=\pi\odot \tanh(S(\bar z,c)h_{\text{lin}})
\]

最终潜在 accessibility response：

\[
h=h_{\text{lin}}+h_{\text{nl}}
\]

要求：

- \(\pi\) 交换不变；
- `S` 无方向性 bias；
- \(h(-b)=-h(b)\)；
- 无 bias 的 odd activation；
- `no_switch` 消融必须存在。

若数据不能识别 switch gate，M1 只保留线性 EPRO，不得为复杂而保留。

## 4.6 模块 E：Probe Observation Operator

构建 midpoint accessibility：

\[
\bar a=\mathcal A(\bar z)
\]

端点 accessibility：

\[
a_m=\bar a+\frac{h}{2},\qquad
a_w=\bar a-\frac{h}{2}
\]

对 probe \(p\) 使用单调观测函数：

\[
\hat{\Delta r}_i=
f_{p,c,b_i}(a_{m,i})-
f_{p,c,b_i}(a_{w,i})
\]

要求：

- DMS 与 SHAPE/2A3 使用不同 head；
- monotonicity 由参数化保证；
- edited site 的 allele-specific observation 只作 secondary；
- primary mask 上 WT/mutant 碱基身份一致；
- observation head 不读取 study ID；
- 输出 `probe_domain` 与 calibration status。

## 4.7 不确定性

总方差：

\[
\sigma^2_{\text{total},i}
=
\sigma^2_{\text{measurement},i}
+
\sigma^2_{\text{model},i}
\]

其中：

- measurement variance 来自 replicate、upstream error 或同 study controls；
- model variance 由 EPRO uncertainty head 预测；
- 两者必须分字段保存。

输出：

- `delta_mean`
- `delta_scale`
- `p_abs_delta_above_noise`
- `linear_response`
- `nonlinear_response`
- `fragility`
- `forcing_support`
- `propagation_mass_by_distance`
- `abstain_score`

## 4.8 P2：WT-anchored posterior update

P2 不建立第二套模型。

先根据 WT 观测更新 WT accessibility：

\[
q=r_w^{obs}-\hat r_w
\]

\[
\delta a_w=\mathcal U_{p,c}(q,\sigma_{\text{measurement}})
\]

然后：

- 更新 WT/midpoint state；
- 重新计算 susceptibility；
- 使用相同 mutation forcing；
- 预测 mutant response。

P2 禁止：

- 读取 mutant reactivity；
- 使用整对 profile 的最优缩放；
- 从 frozen test 拟合 update 参数。

## 4.9 模型等级

### EPRO-0：Deterministic response baseline

- 固定 thermodynamic state；
- 手工局部 forcing；
- 固定 stable propagation；
- 无 learned parameters 或只拟合少量 calibration。

用途：验证架构力学，不作 headline。

### EPRO-Lite

- 2–6M 参数；
- fixed thermodynamic prior；
- learned local forcing correction；
- sparse stable susceptibility；
- probe-specific observation；
- 无 nonlinear switch 或只有一个全局 gate。

Tier B 可训练。

### EPRO-Core

- 6–15M 参数；
- bounded endpoint correction；
- node+edge forcing；
- stable sparse susceptibility；
- odd nonlinear switch；
- P1/P2；
- heteroscedastic output。

仅 Tier A 或强 Tier B validation 允许。

### EPRO-DiffPF

- differentiable partition/energy correction；
- 15–30M 参数上限；
- 只在 EPRO-Core 通过且可微 partition 资源 Gate 通过后考虑。

它是可选研究扩展，不是完成项目的必要条件。

## 4.10 禁止架构

- 把 WT 与 mutant token 直接 concatenate 后交给普通 Transformer 作为主模型；
- 用 full dense StaticPairFormer 直接替代 EPRO；
- 以 external foundation model embedding 为唯一科学贡献；
- 把 teacher ΔBPP 当主标签；
- 无支持 mask 的 mutation feature broadcast；
- 无稳定约束的无限 message passing；
- 不可审计的 mixture-of-experts；
- 仅因为模型更大而升级；
- 未通过数据 Gate 就搜索 architecture/hyperparameter。

---

# 五、可证伪假设与架构失败定义

## 5.1 主假设 H1

在冻结 leave-study-out test 上，EPRO-Core 的 study-macro trans-response Skill：

- 大于 0；
- 优于 strongest executable independent difference；
- 优于 matched generic paired network；
- 95% study/parent cluster-bootstrap CI 下界大于 0。

## 5.2 机制假设

- H2：`local forcing + susceptibility` 优于 local-only forcing。
- H3：contact-edge propagation 对 remote response 的贡献高于 sequence-only propagation。
- H4：高 fragility 样本中 nonlinear switch 的增益高于低 fragility 样本。
- H5：P2 WT anchor 改善跨 study calibration，但不损害 identity。
- H6：probe-specific observation 优于把所有 probe 混成一个 head。
- H7：EPRO uncertainty 能识别新 study、低质量 profile 和物理 prior 失配。

## 5.3 数学验收

- identity error：FP32 eval 下 `max_abs < 1e-7`；
- swap error：`max_abs(G(a,b)+G(b,a)) < 1e-6`；
- forcing leakage：support mask 外初始 forcing 严格为 0；
- stability：\(\rho(K)\le \rho_{max}\)；
- solver relative residual `< 1e-5`；
- probe monotonicity：冻结 domain 内导数非负；
- P2 mutant-profile access：静态代码与运行时审计均为 0。

## 5.4 失败定义

任一情况使“专用物理架构”主张失败：

- EPRO 不优于 matched generic paired model；
- 增益只来自 mutation site；
- 远端响应由 mutation feature 泄漏而非 propagation 产生；
- physics prior 消融后相同或更好，且机制指标不成立；
- nonlinear switch 不能被数据识别；
- 稳定约束被训练绕过；
- 增益来自单一 parent/study；
- in-distribution 增益在 leave-study-out 消失；
- uncertainty 与真实误差无关。

失败后可转 benchmark/data/negative-result，不得换成更大黑箱掩盖。

---

# 六、数据与清洗合同

## 6.1 数据优先级

### Tier A 主监督

- RMDB mutate-and-map；
- M2-seq；
- mutate-map-rescue；
- 明确同条件 WT/single-mutant profile；
- 有 replicate/no-edit/error 优先。

### Tier B 扩展

- Ribonanza 中可可靠重建的同批次、同条件单编辑 pair；
- 有设计 lineage 的 Eterna/OpenKnot variants；
- allele-specific PARS riboSNitch；
- 公开 riboswitch/compensatory variant profiling。

### Tier C 辅助

- 单序列 Ribonanza reactivity；
- bpRNA、PDB、Rfam、ArchiveII；
- eFold static data；
- teacher BPP；
- synthetic mutations；
- 功能 assay 而无 matched probing。

Tier C 不得作为主 intervention truth。

## 6.2 原始层

每个文件必须记录：

- source/version/URL；
- publication DOI/PMID；
- license；
- retrieved time；
- SHA256；
- bytes；
- upstream ID；
- raw path；
- parser version；
- download status。

raw 只读，清洗写新层。

## 6.3 Construct schema

至少包含：

- `construct_id`
- `source_entry_id`
- `study_id`
- `publication_id`
- `laboratory_id`
- `parent_id`
- `design_lineage_id`
- `sequence_raw`
- `sequence_normalized`
- `length`
- `probe`
- `probe_protocol`
- `temperature`
- `ligand`
- `ligand_concentration`
- `buffer`
- `in_vivo_in_vitro`
- `batch_id`
- `replicate_id`
- `reactivity_raw`
- `reactivity_upstream`
- `reactivity_error`
- `coverage`
- `snr`
- `valid_mask`
- `probe_eligibility_mask`
- `normalization_method`
- `quality_flags`

缺失字段写 `null + missing_reason`，不得猜测。

## 6.4 Pair schema

至少包含：

- `pair_id`
- `wt_construct_id`
- `mut_construct_id`
- `parent_id`
- `study_id`
- `design_lineage_id`
- `edit_type`
- `edit_positions`
- `wt_alleles`
- `mut_alleles`
- `edit_count`
- `alignment_cigar`
- `condition_match_fields`
- `condition_match_status`
- `delta_reactivity_raw`
- `delta_reactivity_normalized`
- `unchanged_position_mask`
- `changed_position_mask`
- `probe_eligibility_unchanged_mask`
- `local_mask`
- `mid_mask`
- `remote_mask`
- `replicate_noise_estimate`
- `measurement_variance`
- `pair_quality_weight`
- `primary_eligible`
- `exclusion_reasons`

## 6.5 第一版范围

只纳入：

- substitution；
- `edit_count=1`；
- WT/mutant 等长；
- probe 完全一致；
- condition 完全匹配；
- 至少 60% 未编辑位置可比；
- primary physics domain 优先 in vitro。

延后：

- insertion/deletion；
- 多编辑主训练；
- 跨 probe 数值直接回归；
- in vivo/in vitro 混合；
- parent 不明确的近邻；
- 未知 normalization pair。

双突变只进入 rescue 子集。

## 6.6 清洗顺序

1. 解析 raw。
2. 校验 sequence/profile 长度。
3. 规范 T/U 并保留 raw。
4. 验证 WT–mutant 编辑关系。
5. 核对 condition。
6. 构建 alignment。
7. 构建 probe eligibility。
8. 识别 replicate/no-edit/control。
9. 计算 missingness/SNR/coverage。
10. 估计 measurement noise。
11. 冻结 normalization domain。
12. 生成 Δreactivity。
13. 构建 physical features。
14. 最后构建 split。

## 6.7 禁止操作

- test 参与 normalization；
- pair 自身选择最小差异缩放；
- missing 当 0；
- 模型预测填主标签；
- 删除 no-change 或负结果；
- 按 observed effect 选择 test；
- 同一 parent/library 跨 split；
- 不同 probe 无条件合并；
- teacher data 冒充 experimental；
- construct 数冒充 pair 数；
- latent energy 冒充 kcal/mol。

---

# 七、专家标签替代与测量噪声

## 7.1 主监督

\[
\Delta r_i=r_{m,i}-r_{w,i}
\]

连续实验响应就是主标签，不依赖专家。

## 7.2 有重复

- 冻结 dStruct 或等价 replicate-aware caller；
- 输出 differential regions 与 FDR；
- 参数只由 train/validation 决定；
- 统计 label 只作 secondary head/evaluation。

## 7.3 无重复

- 只回归连续响应；
- 使用 upstream error 或同 study control；
- 不声称显著 changer；
- noise threshold 由 controls 冻结。

## 7.4 classSNitch

只用于：

- 历史外部 sanity check；
- no/local/global summary 相关性；
- 与旧文献对齐。

禁止将其作为主训练伪标签或同类 sequence-only predictor。

---

# 八、数据可行性 Gate

## 8.1 Tier A

同时满足：

- ≥5 个独立 study/publication；
- ≥20 个独立 parent；
- ≥5,000 primary-eligible single-mutant pair；
- ≥2 个完整留出 test study，每个 ≥100 pair；
- ≥3 个 study/parent block 有 replicate/no-edit control；
- controls 总计 ≥100 pair；
- 单一 parent 不超过主域 pair 的 40%；
- condition metadata 足以冻结物理域；
- 至少一个 probe domain 可独立成集。

允许：EPRO-Lite、EPRO-Core。

## 8.2 Tier B

同时满足：

- ≥3 个独立 study；
- ≥10 个 parent；
- ≥1,000 primary-eligible pair；
- ≥1 个完整留出 study，≥100 pair；
- 有可审计 noise；
- 至少一个 condition/probe 主域。

允许：EPRO-0、EPRO-Lite。  
EPRO-Core 需 M0 validation 额外批准。

## 8.3 Tier C

任一成立：

- <3 study；
- <5 parent；
- <500 pair；
- condition 无法匹配；
- 无法 parent/study 隔离；
- 标签以 teacher 为主；
- 响应低于可估计 noise。

处理：

- 不训练深度 headline；
- 不追加 seed；
- 转数据/benchmark/negative result；
- 继续 ReactFlow，不返回 PCCNG。

---

# 九、Split、污染与冻结 benchmark

## 9.1 Split 层级

1. exact construct；
2. parent；
3. design lineage/library；
4. study/publication；
5. family/clan；
6. structure similarity。

确认性 test 采用 leave-study-out；validation 至少 parent holdout。

## 9.2 冻结集合

- `train`
- `validation_parent_holdout`
- `test_study_holdout_1`
- `test_study_holdout_2`（Tier A）
- `family_ood`
- `rescue_subset`
- `replicate_control_subset`
- `classic_classsnitch_external`
- `pars_external_stress`
- `physics_diagnostic_trainval`

## 9.3 Test 隔离

test 不得用于：

- normalization；
- distance bins；
- physical feature selection；
- checkpoint；
- solver iteration 数；
- loss weights；
- architecture；
- switch threshold；
- uncertainty calibration。

每个 confirmatory test 只允许一次主解封。

## 9.4 预训练污染

每个外部模型记录：

- version；
- weight SHA256；
- training-data statement；
- RMDB/Ribonanza/Rfam overlap；
- exact/identity/parent/family overlap；
- `clean / contaminated / unknown_contamination`。

主论文必须包含：

- from-scratch，或
- train-only self-pretraining。

RibonanzaNet/RibonanzaNet2 可作强 secondary baseline，但污染未知时不能支撑 clean OOD headline。

---

# 十、强制基线、消融与负控制

## 10.1 非学习基线

- zero-change；
- mutation-type mean；
- distance-decay；
- edit-only；
- nearest train mutant；
- local-release heuristic。

## 10.2 热力学基线

- RNAfold/RNAplfold；
- LinearPartition；
- EternaFold；
- RNAstructure partition；
- RNAsnp；
- SNPfold；
- remuRNA；
- Riprap；
- VariantFoldRNA；
- Rchange 可执行部分。

## 10.3 学习型独立差分

- train-only sequence-to-reactivity；
- RibonanzaNet；
- RibonanzaNet2；
- eFold 可执行 head。

\[
\hat{\Delta r}_{ind}=F(x_m,c)-F(x_w,c)
\]

## 10.4 Matched generic paired baseline

必须实现一个与 EPRO-Lite 参数量、输入和训练预算匹配的通用 paired model：

- shared encoder；
- WT/mutant feature fusion；
- generic cross interaction；
- per-position output；
- 不含 EPRO 的 forcing support、stable solver、odd switch 或 monotone probe constraint。

这是架构创新的关键对照。

## 10.5 EPRO 机制消融

- `no_thermo_prior`
- `local_only_no_propagation`
- `sequence_edges_only`
- `contact_edges_only`
- `unconstrained_K`
- `no_switch`
- `generic_observation_head`
- `no_measurement_variance`
- `P1_without_condition`
- `P2_without_anchor_update`
- `soft_identity_instead_of_hard`
- `generic_paired_matched`

## 10.6 负控制

- shuffled edit position；
- reversed edit direction；
- mismatched condition；
- randomized contact edges；
- study-ID-only；
- mutation-type-only；
- parent memorization probe；
- test-neighbor retrieval probe。

负控制若表现异常高，优先怀疑泄漏或 batch confounding。

---

# 十一、训练目标、模型选择与预训练

## 11.1 主损失

优先 Student-t 或稳健 heteroscedastic NLL：

\[
\mathcal L_{\text{resp}}
=-\sum_i w_i\log p(\Delta r_i\mid\mu_i,\sigma_i,\nu)
\]

只在 primary valid mask 上计算。

## 11.2 允许辅助项

- endpoint reactivity auxiliary；
- solver residual；
- measurement/model variance calibration；
- train-only same-condition cycle diagnostic；
- rescue ranking；
- sparse forcing regularization；
- switch sparsity。

## 11.3 不作为 loss 的硬性质

- identity；
- endpoint-swap antisymmetry；
- forcing support；
- probe monotonicity；
- stability bound。

这些必须由参数化和单元测试保证。

## 11.4 禁止

- teacher ΔBPP 主导；
- 强制全部 remote 为 0；
- test 调权重；
- 用 study ID 捷径；
- 同时堆过多损失而无法消融；
- architecture search 使用 test；
- 看到结果后追加 seed。

## 11.5 预训练顺序

1. from-scratch；
2. train-only masked/self-supervised；
3. 可重建去污染 reactivity pretraining；
4. external foundation model secondary arm。

EPRO 的物理算子不能被 external embedding 替代。

---

# 十二、测评与统计合同

## 12.1 主 endpoint

仅在：

- 未编辑；
- 对齐；
- probe eligibility 不变；
- profile 有效；

的位置计算。

主指标：

\[
\mathrm{Skill}
=1-
\frac{\mathrm{WMAE}(\hat{\Delta r},\Delta r)}
{\mathrm{WMAE}(0,\Delta r)}
\]

聚合顺序：

1. pair 内；
2. parent 宏平均；
3. study 宏平均。

## 12.2 次指标

- WMAE/RMSE；
- Pearson/Spearman；
- sign accuracy；
- affected-position AUPRC；
- local/mid/remote；
- no-change specificity；
- dStruct region overlap；
- uncertainty NLL；
- calibration；
- coverage-risk；
- abstention；
- runtime；
- GPU memory；
- parameter count；
- solver iterations/residual。

## 12.3 机制指标

- forcing support leakage；
- propagation mass by sequence distance；
- propagation mass by predicted contact distance；
- linear/nonlinear contribution ratio；
- fragility vs observed effect；
- contact-edge attribution enrichment；
- swap/identity error；
- cycle residual；
- K stability margin；
- physics-prior agreement/disagreement。

这些是机制证据，不自动等于真实因果机制。

## 12.4 距离

在 validation 前冻结：

- edit：0；
- local：1–10 nt；
- mid：11–50 nt；
- remote：>50 nt。

结构距离使用预测结构时必须标 `predicted`.

## 12.5 统计

- 一个预注册主模型；
- 一个预注册最强主基线；
- paired comparison；
- study/parent cluster bootstrap 95% CI；
- 必要时 study-level sign permutation；
- 次比较 Holm correction；
- effect size + CI；
- 不把 nucleotide 当独立样本伪造样本量；
- 不自动追加 seed。

## 12.6 主 Gate

确认性 test 同时满足：

1. EPRO study-macro Skill > 0；
2. 相对 strongest independent baseline 的差异 CI 下界 > 0；
3. 相对 matched generic paired baseline 的差异 CI 下界 > 0；
4. 增益不由单一 parent/study 驱动；
5. remote 不显著劣于 zero-change；
6. identity/swap/stability 全部通过；
7. 结果可由冻结 commit/config/data/split 重建。

若第 2 通过而第 3 不通过：

- 科学问题可能成立；
- 专用架构创新不成立；
- 论文降级为 paired benchmark/method。

---

# 十三、GPU、监控、时间管理与 GitHub

## 13.1 GPU-only

必须 GPU：

- 任何 learned baseline fitting；
- pilot；
- EPRO training；
- fine-tuning；
- confirmatory inference 的 learned model。

允许 CPU：

- 下载；
- RDAT parsing；
- QC；
- registry/split；
- ViennaRNA/RNAstructure baseline；
- 离线 physical features；
- tests；
- statistics；
- documentation。

训练前：

- `torch.cuda.is_available() == true`
- 记录 GPU/driver/CUDA/PyTorch；
- 参数必须位于 CUDA；
- 禁止自动 CPU fallback。

## 13.2 Run contract

每个 run 预创建：

- unique `run_id`
- config snapshot
- git SHA
- data/split/feature hashes
- structured local log
- metrics JSONL
- system metrics
- checkpoint dir
- manifest
- stop reason
- invariant audit

W&B 不能是唯一证据。

## 13.3 进度检查

- 启动 5–10 分钟健康检查；
- 此后最短 30 分钟；
- >6 小时稳定后 60 分钟；
- NaN/OOM/exit 告警可立即检查；
- 禁止为“看看涨没涨”频繁 full validation。

等待期间并行做：

- data card；
- citation verification；
- evaluator fixtures；
- baseline wrappers；
- contamination audit；
- invariant tests；
- figure scripts；
- paper limitations；
- failure matrix。

不得并行修改同一文件或 artifact。

## 13.4 安全停止

以下任一触发安全停止并保留证据：

- NaN/Inf；
- CUDA/device 异常；
- 数据/split hash 改变；
- label leakage；
- invariant 失败；
- \(\rho(K)\) 越界；
- solver 不收敛；
- 连续 5 次项目级 validation 无推进且资源预算耗尽；
- 磁盘/显存接近安全阈值；
- checkpoint 不可恢复。

## 13.5 GitHub

每个 `T-*` 是一个独立任务。完成后必须：

1. `git status`
2. 审查 diff
3. targeted tests
4. 必要全量 tests
5. artifact/schema validation
6. secret/data/weight/cache audit
7. focused commit
8. push 当前 task branch
9. 记录 SHA/branch/URL

禁止：

- push main；
- 提交 raw RDAT/FASTQ/BAM；
- 提交 checkpoint/weights/cache；
- 提交 token/key/.env；
- 夹带无关脏文件。

push 失败：

- 保留本地 commit；
- 状态 `IMPLEMENTED_NOT_PUSHED`；
- 记录错误；
- 网络/权限恢复后再 push；
- 不伪报完成。

---

# 十四、目标代码与 artifact 布局

## 14.1 新代码

```text
src/reactflow/delta/
  __init__.py
  schema.py
  data.py
  thermo_state.py
  forcing.py
  susceptibility.py
  switch.py
  observation.py
  anchor.py
  model.py
  losses.py
  invariants.py
  evaluate.py
  contamination.py
  manifests.py
```

## 14.2 配置

```text
configs/reactflow_delta/
  data_audit.yaml
  benchmark.yaml
  epro0.yaml
  epro_lite.yaml
  epro_core.yaml
  epro_diffpf.yaml
  generic_paired_matched.yaml
```

## 14.3 脚本

```text
scripts/reactflow_delta/
  preflight.py
  build_source_registry.py
  parse_rmdb.py
  build_pair_registry.py
  build_rsib.py
  build_thermo_features.py
  audit_physics.py
  run_baselines.py
  train.py
  evaluate.py
  audit_invariants.py
  build_final_manifest.py
```

## 14.4 测试

```text
tests/reactflow_delta/
  test_schema.py
  test_pairing.py
  test_probe_masks.py
  test_splits.py
  test_thermo_state.py
  test_forcing.py
  test_susceptibility.py
  test_switch.py
  test_observation.py
  test_anchor.py
  test_invariants.py
  test_losses.py
  test_evaluate.py
  test_contamination.py
  test_manifests.py
  fixtures/
```

## 14.5 Artifacts

```text
artifacts/reactflow_delta/
  r0/
  d0/
  d1/
  d2/
  ph0/
  b0/
  o0/
  m0/
  m1/
  m2/
  e0/
  final/
```

大数据与 weights 不进入 Git。

---

# 十五、完整分阶段执行 Todo

## Phase R0：路线重置与旧资产封存

### 目标

保护旧运行，建立 clean worktree、权威合同和独立 artifact namespace。

### Todo

- [ ] **T-R0.1** 只读记录旧 pwd、HEAD、branch、dirty files、process、GPU、disk、artifact links、remote URL。
- [ ] **T-R0.2** 等旧 v4 自然结束；不 kill、不重启、不追加 seed。
- [ ] **T-R0.3** 生成旧 run archive manifest，标 `historical-only`。
- [ ] **T-R0.4** 从已验证基点建立 clean worktree。
- [ ] **T-R0.5** 建立 `codex/reactflow-delta-r0`。
- [ ] **T-R0.6** 将 V3 合同复制到 `docs/contracts/` 并记录 SHA256。
- [ ] **T-R0.7** 在 V2 添加 supersession notice，不删除旧文档。
- [ ] **T-R0.8** 建立 `src/reactflow/delta/` 和测试空包。
- [ ] **T-R0.9** 运行 import test、commit、push。

### 验收

```bash
pytest -q tests/reactflow_delta/test_manifests.py
git status --short
```

要求：

- 新工作树 clean；
- 旧运行未受干扰；
- 合同 hash 固定；
- 新旧 evidence 隔离。

---

## Phase D0：公开成对数据可行性审计

### 目标

不训练，回答可用 WT–single-mutant pair 到底有多少。

### Todo

- [ ] **T-D0.1** 创建 `source_registry.jsonl` schema 与失败测试。
- [ ] **T-D0.2** 获取 RMDB release/index/metadata 并记录 checksum。
- [ ] **T-D0.3** 识别 mutate-and-map、M2-seq、rescue、variant-library entries。
- [ ] **T-D0.4** 每类抽样 3–5 个 RDAT 建 parser fixtures。
- [ ] **T-D0.5** 实现 RDAT construct parser。
- [ ] **T-D0.6** 审计 WT、single mutant、double mutant、replicate、no-edit。
- [ ] **T-D0.7** 审计 Ribonanza 中同条件单编辑 pair。
- [ ] **T-D0.8** 构建 candidate pair registry，不做最终 normalization。
- [ ] **T-D0.9** 输出 source×study×parent×probe×condition×pair 矩阵。
- [ ] **T-D0.10** 区分真实 pair、designed neighbor、synthetic pair。
- [ ] **T-D0.11** 给出 Tier A/B/C 预判和最大不确定性。
- [ ] **T-D0.12** tests、commit、push。

### 文件

- Create: `src/reactflow/delta/schema.py`
- Create: `src/reactflow/delta/data.py`
- Create: `scripts/reactflow_delta/build_source_registry.py`
- Create: `scripts/reactflow_delta/parse_rmdb.py`
- Test: `tests/reactflow_delta/test_schema.py`
- Test: `tests/reactflow_delta/test_pairing.py`

### 输出

- `data_registry/source_registry.jsonl`
- `data_registry/raw_manifest.json`
- `data_registry/construct_candidates.parquet`
- `data_registry/pair_candidates.parquet`
- `reports/d0_data_feasibility_audit.md`
- `artifacts/reactflow_delta/d0/data_feasibility_summary.json`
- `artifacts/reactflow_delta/d0/parser_fixture_results.json`

### Gate

- 数量来自可解析 artifact；
- 无重复计数；
- 不把 construct 当 pair；
- 不训练。

---

## Phase D1：清洗、配对、噪声与标签

### Todo

- [ ] **T-D1.1** 冻结 construct/pair schema。
- [ ] **T-D1.2** 实现 condition exact matching。
- [ ] **T-D1.3** 实现 substitution verification。
- [ ] **T-D1.4** 实现 alignment 与 unchanged mask。
- [ ] **T-D1.5** 实现 probe eligibility。
- [ ] **T-D1.6** 识别 replicate/no-edit/control。
- [ ] **T-D1.7** 建 raw/upstream/project-normalized 三层。
- [ ] **T-D1.8** 估计 study/probe measurement noise。
- [ ] **T-D1.9** 有重复时运行 frozen differential caller。
- [ ] **T-D1.10** 生成 quality weight 和 exclusion reasons。
- [ ] **T-D1.11** 建手算 fixtures。
- [ ] **T-D1.12** tests、commit、push。

### 文件

- Modify: `src/reactflow/delta/data.py`
- Create: `tests/reactflow_delta/test_probe_masks.py`
- Create: `tests/reactflow_delta/fixtures/pair_cases.json`

### Gate

- fixtures 100%；
- missing 不作 0；
- noise 不用 test；
- normalization 不最小化 pair difference；
- 每个 exclusion 有 machine-readable reason。

---

## Phase D2：RSIB-v1 与数据 Gate

### Todo

- [ ] **T-D2.1** 构建 parent/study/design-lineage graph。
- [ ] **T-D2.2** 审计 exact/identity/family/structure overlap。
- [ ] **T-D2.3** 冻结 leave-study-out test。
- [ ] **T-D2.4** 冻结 validation parent holdout。
- [ ] **T-D2.5** 冻结 rescue/control/external subsets。
- [ ] **T-D2.6** 完成预训练污染审计。
- [ ] **T-D2.7** 冻结 metrics、distance bins 和 statistics。
- [ ] **T-D2.8** 隔离 test labels。
- [ ] **T-D2.9** 计算 detectable effect 与 bootstrap design。
- [ ] **T-D2.10** 判定 Tier A/B/C。
- [ ] **T-D2.11** 生成 dataset card。
- [ ] **T-D2.12** tests、commit、push。

### 文件

- Create: `scripts/reactflow_delta/build_rsib.py`
- Create: `src/reactflow/delta/contamination.py`
- Create: `tests/reactflow_delta/test_splits.py`
- Create: `tests/reactflow_delta/test_contamination.py`

### Gate

- split group overlap = 0；
- ≥Tier B 才允许 learned model；
- test labels 隔离；
- 主指标冻结。

---

## Phase PH0：物理可识别性审计

### 目标

只在 train/validation 上判断 EPRO 的物理分解是否有可识别信号。

### Todo

- [ ] **T-PH0.1** 生成 WT/mutant ViennaRNA/RNAplfold/LinearPartition states。
- [ ] **T-PH0.2** 记录 unpaired/BPP/free-energy/entropy feature provenance。
- [ ] **T-PH0.3** 计算 edit-centered node/edge forcing candidates。
- [ ] **T-PH0.4** 检验 observed response 与 sequence distance。
- [ ] **T-PH0.5** 检验 observed response 与 contact candidates。
- [ ] **T-PH0.6** 检验 Δunpaired/ΔBPP 与 Δreactivity 的相关与失败。
- [ ] **T-PH0.7** 定义 fragility proxy 和 switch-enriched subset。
- [ ] **T-PH0.8** 量化 measurement noise ceiling。
- [ ] **T-PH0.9** 输出物理假设 support/mixed/challenged 状态。
- [ ] **T-PH0.10** tests、commit、push。

### 文件

- Create: `src/reactflow/delta/thermo_state.py`
- Create: `scripts/reactflow_delta/build_thermo_features.py`
- Create: `scripts/reactflow_delta/audit_physics.py`
- Test: `tests/reactflow_delta/test_thermo_state.py`

### Gate

至少满足：

- response 高于 controls/noise；
- 远端或 contact-associated response 有可测信号；
- physical features 可复现；
- feature 计算不使用 test labels。

若不满足：转 benchmark/data，不强做物理模型。

---

## Phase B0：强基线

### Todo

- [ ] **T-B0.1** zero/mutation mean/distance/local-release。
- [ ] **T-B0.2** RNAfold/RNAplfold/LinearPartition。
- [ ] **T-B0.3** EternaFold/RNAstructure。
- [ ] **T-B0.4** RNAsnp/SNPfold/remuRNA/Riprap/VariantFoldRNA/Rchange。
- [ ] **T-B0.5** train-only static reactivity model。
- [ ] **T-B0.6** RibonanzaNet/RibonanzaNet2 difference。
- [ ] **T-B0.7** matched Siamese。
- [ ] **T-B0.8** matched generic paired model。
- [ ] **T-B0.9** 统一 evaluator/runtime/failure table。
- [ ] **T-B0.10** 冻结 strongest executable baselines。
- [ ] **T-B0.11** tests、commit、push。

### Gate

- 同 split/mask/aggregation；
- baseline 失败进入 failure table；
- 参数匹配可审计；
- 不因难跑而挑弱基线。

---

## Phase O0：EPRO 算子力学与不变量

### 目标

先证明算子实现正确，不使用科学 test。

### Todo

- [ ] **T-O0.1** 写 no-edit identity failing test。
- [ ] **T-O0.2** 实现 hard-zero forcing。
- [ ] **T-O0.3** 写 endpoint swap failing test。
- [ ] **T-O0.4** 实现 signed local node/edge forcing。
- [ ] **T-O0.5** 写 forcing support leakage test。
- [ ] **T-O0.6** 实现 sparse symmetric-background K。
- [ ] **T-O0.7** 写 spectral/stability failing test。
- [ ] **T-O0.8** 实现 stable solver 和 residual audit。
- [ ] **T-O0.9** 实现 odd switch 并验证 oddness。
- [ ] **T-O0.10** 实现 monotone probe observation。
- [ ] **T-O0.11** 实现 P2 anchor access guard。
- [ ] **T-O0.12** 建 hairpin release/two-state/no-change synthetic fixtures。
- [ ] **T-O0.13** invariant suite 100%。
- [ ] **T-O0.14** tests、commit、push。

### 文件

- Create: `src/reactflow/delta/forcing.py`
- Create: `src/reactflow/delta/susceptibility.py`
- Create: `src/reactflow/delta/switch.py`
- Create: `src/reactflow/delta/observation.py`
- Create: `src/reactflow/delta/anchor.py`
- Create: `src/reactflow/delta/invariants.py`
- Tests: corresponding `tests/reactflow_delta/test_*.py`

### 验收

```bash
pytest -q tests/reactflow_delta/test_forcing.py
pytest -q tests/reactflow_delta/test_susceptibility.py
pytest -q tests/reactflow_delta/test_switch.py
pytest -q tests/reactflow_delta/test_observation.py
pytest -q tests/reactflow_delta/test_anchor.py
pytest -q tests/reactflow_delta/test_invariants.py
```

所有数学阈值必须 PASS。

---

## Phase M0：EPRO-Lite 单 seed 可学习性

### 前置

- D2 ≥ Tier B；
- PH0 PASS；
- B0 完成；
- O0 invariant PASS。

### Todo

- [ ] **T-M0.1** 实现 `EPRO-0`。
- [ ] **T-M0.2** 实现 `EPRO-Lite`。
- [ ] **T-M0.3** 实现稳健 NLL 与 measurement variance。
- [ ] **T-M0.4** 单样本 GPU overfit。
- [ ] **T-M0.5** 小批量 GPU overfit。
- [ ] **T-M0.6** 固定 seed/预算 train-validation pilot。
- [ ] **T-M0.7** 运行 local-only/sequence-edge/contact-edge 消融。
- [ ] **T-M0.8** 与 strongest independent 和 generic paired 比较。
- [ ] **T-M0.9** 输出 mechanism/failure matrix。
- [ ] **T-M0.10** invariant re-audit。
- [ ] **T-M0.11** tests、commit、push。

### 文件

- Create: `src/reactflow/delta/model.py`
- Create: `src/reactflow/delta/losses.py`
- Create: `configs/reactflow_delta/epro_lite.yaml`
- Create: `scripts/reactflow_delta/train.py`
- Test: `tests/reactflow_delta/test_losses.py`

### Gate

- validation Skill > 0；
- EPRO-Lite ≥预注册 improvement over strongest independent；
- EPRO-Lite 优于 matched generic paired；
- 增益不只在 edit/local；
- invariants 全 PASS；
- 不追加 seed。

失败：进入数据/机制诊断，不扩大模型。

---

## Phase M1：EPRO-Core 与 P2

### 前置

M0 PASS。

### Todo

- [ ] **T-M1.1** 冻结 Core parameter budget。
- [ ] **T-M1.2** 实现 bounded endpoint correction。
- [ ] **T-M1.3** 实现 odd nonlinear switch。
- [ ] **T-M1.4** 实现 P1。
- [ ] **T-M1.5** 实现 P2 anchor posterior update。
- [ ] **T-M1.6** 实现 uncertainty/abstention。
- [ ] **T-M1.7** 运行全部必需机制消融。
- [ ] **T-M1.8** 运行全部负控制。
- [ ] **T-M1.9** parent/study validation 分层。
- [ ] **T-M1.10** 冻结主 checkpoint。
- [ ] **T-M1.11** 生成 model card 草案。
- [ ] **T-M1.12** tests、commit、push。

### Gate

- EPRO-Core > EPRO-Lite 的增益来自预注册 subset；
- switch gate 在 switch-enriched subset 有信息；
- P2 改善 calibration；
- generic paired 不能解释全部增益；
- 不依赖单一 parent/study；
- test 未解封。

若 switch 不可识别：冻结 EPRO-Lite 为主模型，不视为项目失败。

---

## Phase M2：可选 EPRO-DiffPF

仅当：

- Tier A；
- EPRO-Core PASS；
- differentiable partition 可在 GPU 资源内运行；
- memory/length Gate 通过；
- 可微能量 correction 有独立验证计划。

### Todo

- [ ] **T-M2.1** 建可微 partition 最小 fixture。
- [ ] **T-M2.2** 校验离散端点结果与标准 partition 一致。
- [ ] **T-M2.3** 记录 memory/time scaling。
- [ ] **T-M2.4** 接入 bounded energy correction。
- [ ] **T-M2.5** 对比 fixed-prior EPRO-Core。
- [ ] **T-M2.6** 审计 latent-energy 解释边界。
- [ ] **T-M2.7** 决定 keep/drop。
- [ ] **T-M2.8** tests、commit、push。

不满足前置则永久跳过，不视为失败。

---

## Phase E0：一次性冻结外测

### Todo

- [ ] **T-E0.1** 再审计 commit/config/data/split/feature hashes。
- [ ] **T-E0.2** 冻结主模型、主基线和统计脚本。
- [ ] **T-E0.3** 一次性解封 primary test。
- [ ] **T-E0.4** 运行主模型和全部冻结基线。
- [ ] **T-E0.5** cluster bootstrap/permutation。
- [ ] **T-E0.6** local/mid/remote/study/parent/probe 分层。
- [ ] **T-E0.7** 机制指标与负控制。
- [ ] **T-E0.8** negative cases/uncertainty。
- [ ] **T-E0.9** immutable final manifest。
- [ ] **T-E0.10** tests、commit、push。

### 禁止

- 看 test 后改模型；
- 换 seed；
- 改主指标；
- 改 physical features；
- 删除不利 study；
- 覆盖 run ID；
- 再训练后重测同一 confirmatory test。

---

## Phase P0：论文与发布

### 论文主线

1. paired experimental response 与静态结构差不同；
2. RSIB-v1 数据与泄漏审计；
3. 三世界约束如何推导 EPRO；
4. EPRO 是否优于 independent 和 generic paired；
5. local forcing 如何传播；
6. nonlinear switch 何时成立；
7. uncertainty/OOD；
8. 失败域与限制。

### Todo

- [ ] **T-P0.1** 更新投稿前文献检索。
- [ ] **T-P0.2** 冻结 dataset card。
- [ ] **T-P0.3** 冻结 model card。
- [ ] **T-P0.4** 自动生成全部主表/图。
- [ ] **T-P0.5** 写 negative-result/failure section。
- [ ] **T-P0.6** 写 physical-interpretation limitations。
- [ ] **T-P0.7** 写 contamination/reproducibility。
- [ ] **T-P0.8** 核对 paper/README/artifact 数字。
- [ ] **T-P0.9** GitHub release audit。
- [ ] **T-P0.10** tag、push、归档。

## Phase I0：可选 mRNA-EditFlow 集成

只有 E0 主 Gate PASS：

- EPRO 只作带不确定性的结构响应 oracle；
- 高不确定样本拒绝评分；
- 不把 model score 写成实验事实；
- 只做 retrospective public-data evaluation；
- 不重新激活 PCCNG。

---

# 十六、下一阶段立即执行 Goal

```text
你现在只执行 ReactFlow-Δ EPRO 的 Phase R0 + D0。

服务器：
ssh -p 22 cunyuliu@36.137.135.49

只读旧现场：
/home/cunyuliu/reactflow_c1_3_stage_20260722

建议新工作树：
/home/cunyuliu/reactflow_delta_goal_20260729

权威合同：
docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md

最高目标：
在不启动任何 learned training 的前提下，回答公开数据中到底有多少真实、同条件、可按 parent/study 隔离的 WT-single-mutant chemical-probing pair，并判定 Tier A/B/C。

强制边界：
1. 不停止、修改或重启旧 C1-3。
2. 不在旧 dirty checkout 开发。
3. 不训练任何 learned model。
4. 不提前实现 EPRO 主模型。
5. 不把 construct 总数当 pair 数。
6. 不把 designed neighbor/synthetic/teacher 当实验 pair。
7. 原始数据只读并记录 checksum。
8. 不覆盖旧 artifact。
9. 每个 T-R0/T-D0 完成后 targeted tests、focused commit、push。
10. GitHub 不提交 raw data、weights、checkpoints、cache、secret。

先只读 preflight：
- pwd
- repo/HEAD/branch/dirty
- active ReactFlow processes
- GPU processes
- disk
- artifacts symlink
- remote URL

然后依次完成 T-R0.1 至 T-D0.12。

必须输出：
- source_registry.jsonl
- raw_manifest.json
- construct_candidates.parquet
- pair_candidates.parquet
- d0_data_feasibility_audit.md
- data_feasibility_summary.json
- parser_fixture_results.json

报告必须包含：
- entry/construct 总数
- candidate mutational entries
- 明确 WT 数
- single-mutant pair 数
- rescue/double 数
- replicate/no-edit 数
- study 数
- parent 数
- probe/condition/in-vitro-in-vivo 分布
- primary-eligible 预估
- exclusion reason 分布
- Tier A/B/C
- 最大三个不确定性
- 是否允许 D1
- commit SHA / branch / push status

Gate 未通过：
- 不训练
- 不降低阈值
- 保留审计
- 进入数据/benchmark fallback
```

---

# 十七、统一阶段验收

```text
停止下一阶段实现，只验收当前阶段。

1. 核对 V3 合同与 SHA256。
2. 列出新增/修改文件。
3. 核对未触碰旧运行和无关脏文件。
4. 运行 targeted tests。
5. 运行必要全量 tests。
6. 搜索 placeholder/TODO/pass/mock/hard-coded metric。
7. 验证 JSON/JSONL/Parquet/YAML schema。
8. 验证 artifact 非空、可解析、可重建。
9. 验证 source/license/checksum。
10. 验证 parent/study overlap=0。
11. 验证 test 未用于选择。
12. 验证 experimental/proxy/teacher/synthetic 分离。
13. 若有训练，验证 GPU-only；否则 NOT APPLICABLE。
14. 若有 EPRO，验证 identity/swap/support/stability/solver/monotonicity。
15. 每项 Gate 给 PASS/FAIL/NOT RUN。
16. FAIL 时生成 failure matrix，不降低 Gate。
17. 检查 diff 无 secret/data/weight/cache。
18. focused commit 并 push。

最终只汇报：
- contract SHA
- branch
- commit SHA
- push status
- tests
- artifacts
- invariants
- Gate
- blockers
- 下一阶段是否获准
```

---

# 十八、Fail-forward 合同

Gate 失败时先冻结：

- run/config/git/data/split/feature hashes；
- logs；
- last usable checkpoint；
- metrics；
- system metrics；
- invariant audit；
- failure evidence。

然后按层定位：

### A. 数据

- pair 是否真实；
- parent 是否明确；
- condition 是否匹配；
- study/parent 是否足够；
- response 是否高于 noise。

### B. 观测

- probe eligibility；
- normalization；
- missing；
- measurement variance；
- batch confounding。

### C. 泄漏

- exact；
- parent；
- study；
- library；
- physical-feature test leakage；
- pretraining。

### D. 物理算子

- forcing 是否局部；
- K 是否稳定；
- solver 是否收敛；
- contact edges 是否有效；
- switch 是否可识别；
- probe head 是否正确。

### E. 基线

- zero；
- thermodynamic；
- independent reactivity；
- matched generic paired。

### F. 优化

- single-sample overfit；
- small-batch overfit；
- gradient/NaN；
- GPU；
- checkpoint。

### G. 科学假设

- paired response 是否根本不可跨 parent 学习；
- 远端响应是否低于 noise；
- thermodynamic prior 是否失配；
- generic paired 是否已经解释全部增益；
- equilibrium assumption 是否不适用。

每次最多选择三个最高信息增益的最小实验。

只允许：

1. 修复数据/评测错误；
2. 缩小到 Tier B；
3. 删除不能识别的可选模块；
4. 转 benchmark/data/negative result；
5. 安全停止模型路线。

禁止：

- 扩大模型制造 PASS；
- 延长训练无上限；
- 追加 seed；
- 改 test/主指标；
- 删除不利 study；
- 回到 PCCNG；
- 恢复静态 SOTA 主叙事。

---

# 十九、论文审计与参考文献

## 19.1 论文前清单

- [ ] 文献更新至投稿前 30 天。
- [ ] 不使用未经证明的 `first`。
- [ ] 主标签为实验 Δreactivity。
- [ ] reactivity 不写成唯一真实结构。
- [ ] 主 endpoint 排除 edited/eligibility-changed positions。
- [ ] parent/study split 无泄漏。
- [ ] 数据 Gate 可审计。
- [ ] strongest independent baseline 完成。
- [ ] matched generic paired baseline 完成。
- [ ] EPRO invariants 全通过。
- [ ] physics prior、propagation、switch、observation 消融完成。
- [ ] 负控制完成。
- [ ] uncertainty/calibration 完成。
- [ ] cluster bootstrap 单位正确。
- [ ] 未自动追加 seed。
- [ ] GPU 证据完整。
- [ ] negative results 未删除。
- [ ] latent energy 未冒充真实 kcal/mol。
- [ ] README/paper/model card/dataset card 数字一致。
- [ ] GitHub release 无 raw restricted data、weights、secret。
- [ ] 无新湿实验时明确 retrospective。

## 19.2 核心参考文献

1. Sabarinathan R, et al. RNAsnp. *Human Mutation* (2013). [PMID 23315997](https://pubmed.ncbi.nlm.nih.gov/23315997/)
2. Corley M, et al. Genome-wide riboSNitch benchmark. *Nucleic Acids Research* (2015). [PMID 25618847](https://pubmed.ncbi.nlm.nih.gov/25618847/)
3. Woods CT, Laederach A. classSNitch. *Bioinformatics* (2017). [DOI 10.1093/bioinformatics/btx041](https://doi.org/10.1093/bioinformatics/btx041)
4. Cheng CY, et al. M2-seq. *PNAS* (2017). [PMID 28851837](https://pubmed.ncbi.nlm.nih.gov/28851837/)
5. Kladwang W, et al. Mutate-and-map protocol. [PMC4080707](https://pmc.ncbi.nlm.nih.gov/articles/PMC4080707/)
6. Tian S, et al. High-throughput mutate-map-rescue. [PMC4201832](https://pmc.ncbi.nlm.nih.gov/articles/PMC4201832/)
7. Choudhary K, et al. dStruct. *Genome Biology* (2019). [DOI 10.1186/s13059-019-1641-3](https://doi.org/10.1186/s13059-019-1641-3)
8. Calonaci N, et al. Machine learning a model for RNA structure prediction. *NAR Genomics and Bioinformatics* (2020). [DOI 10.1093/nargab/lqaa090](https://pmc.ncbi.nlm.nih.gov/articles/PMC7671377/)
9. Lin J, et al. Riprap/RiboSNitchDB. *NAR Genomics and Bioinformatics* (2020). [DOI 10.1093/nargab/lqaa057](https://doi.org/10.1093/nargab/lqaa057)
10. Miladi M, et al. MutaRNA. *Nucleic Acids Research* (2020). [DOI 10.1093/nar/gkaa331](https://doi.org/10.1093/nar/gkaa331)
11. Matthies MC, et al. Differentiable partition function calculation for RNA. *Nucleic Acids Research* (2024). [Article](https://academic.oup.com/nar/article/52/3/e14/7457012)
12. Ribonanza consortium. Ribonanza/RibonanzaNet (2024). [PMC10925082](https://pmc.ncbi.nlm.nih.gov/articles/PMC10925082/)
13. Kirven KJ, et al. VariantFoldRNA (2025). [DOI 10.1093/nargab/lqaf066](https://doi.org/10.1093/nargab/lqaf066)
14. BPfold: thermodynamic base-pair motif energy attention (2025). [Nature Communications](https://www.nature.com/articles/s41467-025-60048-1)
15. Sacco G, et al. MERGE-RNA: physics-based ensemble and probing model. arXiv (2025). [arXiv:2512.20581](https://arxiv.org/abs/2512.20581)
16. Choi EK, et al. PRIME energetic coupling preprint (2026). [bioRxiv DOI](https://doi.org/10.64898/2026.01.28.702231)
17. de Lajarte AA, et al. eFold (2026). [PMC12935039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935039/)
18. Chen Z, et al. CHANRG (2026 preprint). [arXiv:2603.22330](https://arxiv.org/abs/2603.22330)
19. Cordero P, et al. RMDB. [PMC3496344](https://pmc.ncbi.nlm.nih.gov/articles/PMC3496344/)
20. RNA Mapping Database. [Database](https://rmdb.stanford.edu/)；[About/versioning](https://rmdb.stanford.edu/about/)

---

# 二十、最终执行原则

1. **数据存在性先于模型。**
2. **物理机制先于网络模块。**
3. **数学硬约束先于软损失。**
4. **真实测量过程先于抽象标签。**
5. **局部 forcing 必须通过显式传播产生远端响应。**
6. **热力学是 prior，不是实验 truth。**
7. **latent physical quantity 不冒充可测物理参数。**
8. **EPRO 必须战胜 independent 和 generic paired 两类强对照。**
9. **跨 study/parent 外推先于随机 held-out。**
10. **小而可证伪的算子先于大模型。**
11. **GPU 训练，但不盲跑。**
12. **不频繁查看，但保留可靠告警和证据。**
13. **等待时做无冲突高价值工作。**
14. **每个任务测试、commit、push。**
15. **失败保留证据，只能 fail-forward。**
16. **没有湿实验和专家也能研究，但不能夸大结论。**

最终成功标准不是：

> 训练了一个更大的 RNA 模型。

而是：

> 用公开、可审计、成对实验数据，严格判断一个由 RNA ensemble 物理、扰动响应数学和 probe 观测共同推导的专用 EPRO，是否真的比独立静态差分和同容量通用 paired model 更能预测 RNA 突变的实验响应；如果答案是否定的，也形成可信、可发表、可复用的 benchmark、机制失败分析和负结果。
