# ReactFlow:以化学探针 reactivity 为条件、离散流匹配生成 RNA 2D 结构分布 — 研究规划(v0.1)

> 状态:C1/C2/C3/C4/C5 实现内核已落地(工程实现见 `reactflow/`,合成 pilot 端到端跑通;采样器 100% 合法、guidance `η` 扫描单调性已证并 SymPy 验证;C5 Stage-A warm-start 冻结特征 + 手写线性 adapter + C5.4 评测协议已实现,并已升级到真实 eFold/RNAndria Dryad JSON scale-up 基础设施:`prepare-efold-cache` 物化 JSONL cache 且支持长序列 windowing/length bucketing,`split-efold-cache` 可生成 clan/cluster 无泄漏 train/val/test/novel JSONL artifacts,`train-efold`/`evaluate-efold` 可训练和评估真实结构标签并支持 per-phase profiling 与 JSON checkpoint,RibonanzaNet2 Kaggle-alpha single-only frozen export 成功(64 条 shard 8.8MB),可选 lazy PyTorch base 训练后端已接入;全量 283 passed/2 skipped、coverage 92.79%、`import reactflow` 零重依赖;外部 Rfam/MMseqs 真标签、全量 frozen export、服务器规模化 profiling/torch 对比与同口径 baseline rerun 留待 full real-data run) · 日期 2026-07-08
> 工作名 **ReactFlow** 为占位。它是 `../codonflow/CodonFlow_research_plan.md` 的 **sibling 方向**:CodonFlow 走「密码子=编辑流对齐监督」的核酸-蛋白统一生成;ReactFlow 走「化学探针 reactivity=RNA 2D 结构分布的期望监督」。二者共享 discrete flow matching(DFM)底座与 `../dogmaflow`/`../editflow` 代码资产,但科学题眼不同,互不重叠。
> **诚实约定(沿用 CodonFlow ledger 两级标注):**
> - ✅ **本轮核实** = 本次由主源(arXiv 全文 / OpenReview / 期刊 DOI / 数据集官方页)直接确认。
> - ◻️ **待核实** = 本轮凭既有认知列入,**成文前必须逐条查证卷期页/ID**,暂不得当作已核实引用。
> - ⚠️ **需实验/数值验证** = 依赖 pilot 结果才能定的设计点。
> **禁止带病引用与学术造假。严禁编造数据。** 所有实验数字须区分「in-silico 代理」与「实验真值」。

---

## 0. 执行摘要(abstract-ready)

**一句话定位**:ReactFlow 是首个把**化学探针 reactivity 显式当作 RNA 2D 结构*分布*的一阶矩监督**、用 **discrete flow matching** 学习 `序列 → 碱基配对分布` 映射、并以 **cross-family(Rfam clan)OOD 泛化**为主战场的生成式模型。

**方向与判别式 SOTA 相反**:RibonanzaNet2 学 `序列 → reactivity`(判别、回归标量);ReactFlow 学 `序列 → 结构分布 p_θ(S|x)`,再用「reactivity 是结构 ensemble 的群体平均」这一物理事实,把实测 reactivity 作为**可微一致性监督**回灌到分布上。

**精确护城河(经 2025 同质化体检后收窄,见 §5)** — 不是「生成 ensemble」也不是「把物理塞进速度场」(这两点已被占),而是三者交集:
1. **reactivity 作为分布监督**:用可微前向算子 `f(S)` 把结构映到期望 reactivity,构造矩匹配损失 `E_{S~p_θ}[f(S)] ≈ 实测 profile`(§2.2)——不是把 reactivity 当输入特征(RibonanzaNet 那样),而是当**分布的观测约束**;
2. **分布输出而非点估计**:DFM 天然出 ensemble,恰好匹配 reactivity 的「群体平均」本质,且能表达一条序列的多构象/亚稳态;
3. **cross-family OOD 为一等公民**:按 Rfam clan 切分,严禁家族泄漏,直接量化 eFold/Szikszai 指出的泛化鸿沟(§3)。

**落地策略(warm-start 降算力)**:2D 目标比 3D SE(3) 便宜、不碰稀缺 3D 数据;encoder 从 **RibonanzaNet2 warm-start**(⚠️ 权重可得性待确认,见 §6),Ribonanza2 的 64M 双探针数据撑起训练规模。落地可行性判定:**HIGH**。

---

## 1. 背景与动机:为什么是「现在」「这个」方向

### 1.1 RNAbpFlow 暴露的瓶颈:3D 好坏取决于 2D 配对质量
RNAbpFlow(SE(3)-equivariant flow matching,以碱基配对图为条件生成全原子 3D 构象,◻️ Nature Methods,待核实卷期页)把 RNA 三级结构预测推进了一步,但其**输入是 2D 配对图**——一旦 2D 配对在长 RNA / 伪结上出错,3D 全盘皆错。也就是说:**当前 RNA 3D 的天花板,很大程度上是 2D 碱基配对的天花板。** 改进点前移到 2D 是杠杆率最高的地方。

### 1.2 reactivity 的本质是「ensemble 平均」,判别式回归把它压扁了
DMS / 2A3(SHAPE 类)探针测的是每个核苷酸在**溶液中全体构象**里的平均可及性:某位点若在多数构象里未配对 → 高反应性;多数构象里配对 → 低反应性。**reactivity 本身就是一个分布泛函的观测值。** 判别式模型(RibonanzaNet2:序列→reactivity 标量)把这个「分布的投影」直接回归成点值,丢掉了「哪些构象共同产生了这条 profile」的信息。**用生成分布去反解、并要求其期望 reactivity 对上实测,才是与数据生成过程同构的建模方式。**

### 1.3 数据充足 × 模型欠缺 = 可通过改模型吃红利(本方向的立项判据)
沿用 CodonFlow 的「数据受限 vs 模型受限」框架筛选:
- **数据侧充足**:Ribonanza2(◻️ ~64M 序列、DMS+2A3 双探针、CC-BY 4.0,待核实规模/许可)提供了训练大模型所需的规模;这是判别式 RibonanzaNet2 已验证「数据够喂大模型」的直接证据。
- **模型侧欠缺**:现有 2D 预测器多为**判别式点估计**,不出分布、不显式利用「reactivity=ensemble 平均」、且 cross-family 泛化差(eFold/Szikszai 已量化)。
→ 落在「数据齐全但模型不好」的象限,**改模型能吃到红利**,符合立项判据。

---

## 2. 任务定义与核心方法

### 2.1 任务定义 + 为什么 flow matching 比回归式 2D 预测器更合适

**任务(两种模式,共用同一 `p_θ`):**
- **模式 I — de novo 分布预测**:给定序列 `x`,输出 2D 结构分布 `p_θ(S | x)`(一条序列 → 一组带概率的结构/一个碱基配对概率矩阵)。测试期无需 reactivity;其**预测 reactivity** 可与留出探针数据对照验证泛化。
- **模式 II — reactivity 条件精修(可选)**:推理期若有实测 profile `r`,作为条件 `p_θ(S | x, r)` 引导 ensemble 收敛到与数据一致的构象子集。

**reactivity 的两种角色务必分清(审稿关键):**
- **监督角色(§2.2 一致性损失)**:reactivity 是*目标观测*,用于训练期矩匹配;即使 de novo 也用它教模型 `序列→ensemble` 的映射。**这是护城河。**
- **条件角色(可选输入)**:仅在模式 II 作输入特征。**不能**把「拿 reactivity 当输入」当作创新点(那是 RibonanzaNet 已做的判别式用法)。

**为什么 flow matching > 回归式 2D 预测器:**

| 维度 | 回归式 2D 预测器(RibonanzaNet2 等) | ReactFlow(DFM 生成) |
|---|---|---|
| 输出对象 | 点估计:单个配对矩阵 / reactivity 标量 | **分布** `p_θ(S\|x)`:可采样多构象、给不确定度 |
| 与 reactivity 生成过程的关系 | 把「分布投影」直接回归成点(信息坍缩) | **同构**:reactivity = `E_{S~p}[f(S)]`,天然是分布的一阶矩 |
| 多构象 / 亚稳态 / riboswitch | 无法表达(单点) | 天然表达(ensemble 里多峰) |
| 可控/引导 | 难 | ODE/CTMC 轨迹确定性 → 推理期可注入热力学先验/条件(§2.3) |
| 变长 / 伪结 | 受结构假设限制 | DFM 在配对空间上定义,伪结可通过放开 nested 约束表达(⚠️ 需验证) |

一句话:**回归器学的是「最可能的一张结构」,ReactFlow 学的是「产生这条 reactivity 的那一族结构」——后者才是 reactivity 数据真正编码的东西。**

---

### 2.2 reactivity-consistency loss:完整数学形式 ★核心(审稿人最会挑处)

本节把「期望 reactivity 对上实测」写成可训练目标,分四步:结构表示 → 可微前向算子 `f(S)` → 期望估计的采样方案 → 一致性损失。最后诚实标注可辨识性局限。

#### 2.2.1 结构表示与生成分布
序列 `x ∈ {A,C,G,U}^L`。2D 结构 `S` 用对称配对矩阵 `P(S) ∈ {0,1}^{L×L}` 表示,满足**匹配约束** `Σ_j P_{ij}(S) ≤ 1`(每核苷酸至多一个配对伙伴)。定义**未配对指示**

$$u_i(S) = 1 - \sum_{j} P_{ij}(S) \in \{0,1\}.$$

生成模型 `p_θ(S | x)` 为 discrete flow matching 在配对状态空间上的分布。把每个位点 `i` 的「伙伴选择」建模为在 `{∅, 1, …, L}` 上的类别变量(`∅`=未配对),模型在流的每个时刻 `t` 输出去噪后验 `p^θ_{1|t}(S_1 | S_t, x)` 的按位边缘 `π_i^θ(· | S_t, x)`。

#### 2.2.2 可微 reactivity 前向算子 `f(S)`
对单个结构,定义每核苷酸、每探针的预测反应性:

$$f_i^{(k)}(S) = \phi_k\big(u_i(S),\, e_i(S);\, t_i\big),\qquad k\in\{\mathrm{DMS},\,\mathrm{2A3}\}.$$

其中 `t_i∈{A,C,G,U}` 为核苷酸种类;`e_i(S)` 为**边缘/末端上下文**特征(如螺旋端、发夹小环、fraying——即使配对但邻位未配对时反应性升高),一个可用定义:`e_i(S)=1` 当 `i` 配对且其序列近邻存在未配对位点,否则 0。

**刻意选择「对结构特征仿射」的参数化**,使期望可解析(见 2.2.3):

$$\phi_k(u,e;t) = a_{k,t}\,u \;+\; b_{k,t}\,e \;+\; c_{k,t},\qquad a_{k,t}=\mathrm{softplus}(\tilde a_{k,t})>0.$$

- **物理单调性硬编码**:`a_{k,t}>0` 保证「越未配对 → 反应性越高」,符合探针化学。
- **探针-核苷酸特异掩码**:DMS 只甲基化 A/C 的 WC 面 → 对 `t∈{G,U}` 令其在 DMS 通道权重为 0(不产生监督);2A3(SHAPE 类)酰化 2′-OH,四种碱基皆有信号但主报「局部柔性」,故 `e_i` 在 2A3 通道权重更大。
- `(a,b,c)_{k,t}` 是**少量可学习标量**(每探针×每碱基一组),可从文献先验初始化,受单调性约束,**可解释**。

可选非线性 link `σ`(把输出校准到探针值域):`f_i^{(k)} = σ(a_{k,t_i}u_i + b_{k,t_i}e_i + c_{k,t_i})`。若用非线性 link,则期望需用 2.2.3 的采样估计,或用 delta 近似 `E[σ(·)]≈σ(E[·])`(⚠️ 偏差需验证)。**默认线性,把非线性交给校准层(2.2.4)**,以保住期望的解析性。

#### 2.2.3 期望估计的采样方案(本节是审稿人真正会盯的地方)
核心量:模型下的期望 reactivity

$$\hat r_i^{(k)} \;=\; \mathbb{E}_{S\sim p_\theta(\cdot|x)}\!\big[f_i^{(k)}(S)\big].$$

因 `φ_k` 对 `(u_i,e_i)` 仿射,**期望可交换进去**:

$$\boxed{\;\hat r_i^{(k)} = a_{k,t_i}\,\underbrace{\mathbb{E}[u_i(S)]}_{\displaystyle q_i}\;+\;b_{k,t_i}\,\underbrace{\mathbb{E}[e_i(S)]}_{\displaystyle \bar e_i}\;+\;c_{k,t_i}\;}$$

于是只需估计**边缘未配对概率** `q_i = P_θ(u_i=1)` 与期望边缘特征 `\bar e_i`。给出三种估计器,默认 (A) 为主 + (B) 纠偏:

**(A) DFM 去噪边缘(主用,可微、低方差,无需展开轨迹)。**
DFM 网络在采样时刻 `t~U[0,1]`、噪声态 `S_t` 下直接预测干净数据后验的按位边缘 `π_i^θ(·|S_t,x)`。取

$$q_i(S_t) = \pi_i^\theta(\varnothing\,|\,S_t,x),\qquad
\hat r_i^{(k)}(S_t) = a_{k,t_i}q_i(S_t) + b_{k,t_i}\bar e_i(S_t) + c_{k,t_i},$$

一致性损失在 `t` 上取期望 `E_{t,S_t}[ℓ(\hat r(S_t), r)]`。**完全可微、无 rollout。** 代价:用的是**因子化(mean-field)边缘**,忽略了全局匹配/nested 约束 → 有 mean-field 偏差。作为主信号,便宜且梯度稳。

**(B) 采样估计 + Gumbel 直通(纠偏,捕捉相关性)。**
从流积分 `0→1` 采 `M` 个结构。为保梯度:对每位点伙伴类别用 **Gumbel-Softmax**(温度 `τ`)松弛为软分配 `\tilde π_i∈Δ^L`,取 `\tilde u_i=\tilde π_i[∅]`;对离对角软匹配矩阵做数次 **Sinkhorn** 迭代逼近「对称+至多一个伙伴」。**前向用投影后的合法硬结构**(canonical 配对 + 匹配 + 最小环,见 §2.3.2),**反向用软 `\tilde π`(straight-through)**。经验估计

$$\hat r_i^{(k)} = \frac1M\sum_{m=1}^{M} f_i^{(k)}\!\big(S^{(m)}\big).$$

捕捉 mean-field 漏掉的位点间相关,方差/开销更高 → 只在**部分步/周期性**启用作纠偏;`τ` 退火。

**(C) Score-function / REINFORCE(兜底,无偏、无松弛)。**
`∇_θ \hat r_i = E[f_i(S)∇_θ log p_θ(S)]`,配滑动平均 baseline 降方差。仅当 (B) 的松弛被证有偏时启用;方差高。

> 采样方案总结:**默认 (A) 训练 + (B) 周期纠偏 + eval 用 (B) 的硬采样**。这一「因子化边缘为主、采样为辅纠偏」的两层结构,是应对「期望估计既要可微又要低偏差」的核心工程答案。

#### 2.2.4 一致性损失本体(处理噪声/尺度/掩码/置信度)
实测 profile `r_i^{(k)}` 有四个必须处理的现实问题:**尺度任意**(常做 0–1 或分位归一)、**形状比绝对值更可信**、**部分掩码**(DMS 仅 A/C;capped/低 SNR 位点无效)、**逐位噪声**。因此损失 = 校准幅度项 + 尺度不变形状项 + 置信度加权 + 掩码。

**逐位权重/掩码**:`w_i^{(k)} = m_i^{(k)}·ρ_i^{(k)}`,其中 `m^{(k)}` = 探针-碱基有效掩码(DMS 只 A/C)×实验可用掩码(丢 capped/低 SNR),`ρ_i^{(k)}` = 置信权重(取 Ribonanza 逐位误差的逆方差,⚠️ 字段 schema 待核实)。

**校准**:reactivity 尺度任意,拟合每探针(或每转录本)仿射 `\hat r^{cal}=α_k\hat r+γ_k`(或小单调网络),`Ω(α,γ)` 为其正则。

**幅度项(加权、校准后 MSE):**

$$\ell^{(k)}_{\mathrm{mag}} = \frac{\sum_i w_i^{(k)}\big(\alpha_k\hat r_i^{(k)}+\gamma_k - r_i^{(k)}\big)^2}{\sum_i w_i^{(k)}}.$$

**形状项(尺度/平移不变的加权 Pearson,抓生物学真正关心的 profile 模式):**

$$\ell^{(k)}_{\mathrm{shape}} = 1 - \mathrm{corr}_w\!\big(\hat r^{(k)},\, r^{(k)}\big).$$

(可选 soft-Spearman：用可微排序代理增强对离群点的鲁棒性。)

**reactivity-consistency 总损失:**

$$\boxed{\;L_{\mathrm{react}} = \sum_{k\in\{\mathrm{DMS},\mathrm{2A3}\}}\!\big(\lambda_{\mathrm{mag}}\,\ell^{(k)}_{\mathrm{mag}} + \lambda_{\mathrm{shape}}\,\ell^{(k)}_{\mathrm{shape}}\big) + \lambda_{\mathrm{cal}}\,\Omega(\alpha,\gamma)\;}$$

#### 2.2.5 总训练目标
$$L = \underbrace{L_{\mathrm{DFM}}}_{\text{已知结构上的流匹配}} \;+\; \lambda_r\,L_{\mathrm{react}} \;+\; \lambda_{\mathrm{phys}}\,L_{\mathrm{phys}} \;+\; \lambda_{\mathrm{td}}\,L_{\mathrm{thermo}}.$$

- `L_DFM`:在**有高置信结构**的 RNA(bpRNA / PDB 衍生 / RibonanzaNet2 高置信集,◻️ 待定)上的标准 DFM 损失,监督**联合结构**——锚住模型,防止 reactivity-consistency 把分布塌成「只对边缘、联合乱来」的退化解。
- `L_react`:§2.2 矩匹配(吃 64M 探针数据的规模红利)。
- `L_phys`、`L_thermo`:见 §2.3。
- `λ` 全部待消融(§8)。⚠️

#### 2.2.6 诚实局限(审稿人一定问,先自曝)
reactivity 只约束 ensemble 的**一阶边缘**(逐位 `P(未配对)`)。**许多不同的联合分布共享同一条边缘 profile → 仅凭 reactivity 不可辨识联合结构。** 缓解三招,写进论文:
1. `L_DFM` 用已知结构锚住联合;
2. 若有 **mutate-and-map / MaP 二维**数据,加**二阶矩项**:成对 co-reactivity ≈ `E[u_i u_j]`,直接约束配对相关(把「一阶」升到「二阶」,显著收窄不可辨识性);
3. 热力学先验(§2.3)在无数据处塑形联合。
**这是本方法的诚实边界,不藏。**

---

### 2.3 物理化学约束融合方案 ★核心

先厘清一个易被误解的点:**2D 层面没有键长/键角/原子 clash 这类 3D 立体化学**——那些属于 RNAbpFlow 的 3D 模块。2D 层面的「物理化学约束」是两类:**(i) 结构合法性(组合硬约束)**与 **(ii) 热力学先验(能量软约束)**。ReactFlow 把二者分别以「硬掩码」与「引导/正则」注入,并诚实划出 3D out-of-scope。

#### 2.3.1 2D 层面的「物理」到底是什么
- **合法性(硬)**:每核苷酸至多一个伙伴(匹配);只允许 canonical + wobble 配对(A-U / G-C / G-U),非经典对禁止或重罚;最小发夹环 ≥ 3 nt(空间上环不能太紧);是否允许伪结(nested-only vs pseudoknot)可配置。
- **热力学(软)**:Turner 最近邻自由能给 `ΔG(S|x)`,Boltzmann 分布 `p_Turner(S) ∝ exp(-ΔG/RT)`;ViennaRNA/RNAstructure 的配分函数给配对概率 `P^{Turner}_{ij}`。
- **reactivity 一致性本身就是物理约束**:它是真实物理 ensemble 的实验观测(§2.2 已承担)。

#### 2.3.2 硬约束:约束投影采样(mirror CodonFlow 的 REGION 掩码思想)
在流的**每个积分步**,对速率/logit 矩阵施加硬掩码,保证采样出的每个结构物理合法:

| 约束 | 对 logit/rate 的操作 |
|---|---|
| 至多一个伙伴 | 采样时对每行/列做匹配投影(贪心/Sinkhorn-硬化);冲突对置 `-inf` |
| 仅 canonical+wobble | 非 {A-U,G-C,G-U} 的 `P_{ij}` logit 置 `-inf` |
| 最小环 ≥3 | `|i-j|<4` 的配对 logit 置 `-inf` |
| nested-only(可选) | 与已选配对交叉的候选置 `-inf`(启用则禁伪结) |

这与 CodonFlow 在三头 logits 上加 REGION 掩码同源——**算子级硬约束是 flow/CTMC 框架能干净表达的东西**,判别式回归器只能事后修补。

#### 2.3.3 软先验:热力学能量引导 + 半监督正则(两个注入点)
**(a) 推理期能量引导(training-free steering,类 classifier guidance)。**
在 CTMC 转移速率上按候选编辑的 `ΔΔG` 重加权,或在连续松弛的速度场上加梯度项:

$$\tilde R_\theta(\text{edit}) = R_\theta(\text{edit})\cdot\exp\!\big(-\eta\,\Delta\Delta G_{\text{edit}}/RT\big),\qquad\text{或}\qquad \tilde v = v_\theta + \eta\,\nabla\log p_{\mathrm{Turner}}.$$

`η` 可调:**在「信数据」与「信 Turner」之间连续拨动**——探针数据足的家族靠数据,数据稀的 OOD 家族多靠热力学先验。无需重训。⚠️ `η` 与 guidance 数值稳定性需 pilot。

**(b) 半监督热力学正则(训练期,助 cross-family 泛化)。**
在**无探针标签**的序列上,用 ViennaRNA 配分函数配对概率导出 `q_i^{Turner}=1-Σ_j P^{Turner}_{ij}`,作软目标约束模型边缘:

$$L_{\mathrm{thermo}} = \frac{1}{|\mathcal{U}|}\sum_{i\in\mathcal{U}} \mathrm{KL}\big(q_i \,\|\, q_i^{Turner}\big)\quad\text{或}\quad (q_i - q_i^{Turner})^2 .$$

在探针数据覆盖不到的家族上提供「物理合理」的塑形信号,是 cross-family OOD 的重要支柱。

#### 2.3.4 与 reactivity-consistency 的关系 & 诚实划界
- **分工**:`L_react` 用实验观测约束**有数据处**的边缘;`L_thermo` 用物理先验约束**无数据处**;`L_DFM` 用已知结构约束**联合**;硬掩码保证**每个样本合法**。四者互补,不重叠。
- **诚实标注**:Turner 能量本身是近似,**只作先验不作真值**;若 `η` 过大会把模型拉回热力学 MFE(退化成 ViennaRNA)——所以 guidance 强度是要 ablate 的旋钮,不是越大越好。
- **3D 立体化学 out-of-scope(v0.1)**:键长/键角/no-clash/糖环 pucker 属 3D,明确划为 RNAbpFlow 类下游模块的职责;ReactFlow 只交付「更准、带分布、跨族泛化的 2D 配对」作为其输入。**不夸口做 3D。**

---

## 3. 数据与 cross-family 评测协议

### 3.1 数据源(全部真实可得,◻️ 规模/许可/schema 成文前核实)
| 用途 | 数据源 | 说明 |
|---|---|---|
| 探针监督主干 | **Ribonanza2**(◻️ ~64M 序列、DMS_MaP+2A3_MaP 双探针、逐位 reactivity+error、SN_filter) | `L_react` 的规模来源 |
| 已知结构锚点 | bpRNA / PDB 衍生 2D / RibonanzaNet2 高置信集(◻️) | `L_DFM` 监督联合结构 |
| 热力学先验 | ViennaRNA / RNAstructure 配分函数(◻️ 版本待记) | `L_thermo` 软目标 + guidance |
| 家族标注(切分) | **Rfam / Rfam clan**(◻️ release 待记) | cross-family 切分依据 |
| 泛化标杆对照 | **RNAndria / eFold 数据集**(Rouskin lab,◻️ 待核实) | OOD 协议对齐 |

### 3.2 cross-family 切分协议(硬约束,审稿必查)
- **按 Rfam clan 切分,而非随机、也不仅按 family**:train / val / test 的 clan 集合**不相交**,杜绝「训练集与测试集结构家族重叠」的泄漏(项目硬约束)。
- 追加**序列同一性去冗余**(CD-HIT/MMseqs2 核酸模式)与长度分层,报告不同长度桶的表现(长 RNA 是痛点)。
- 设 **novel-clan holdout**:完全未见 clan 单列,作最严 OOD;报告泛化鸿沟(in-clan vs novel-clan 差值),对齐 eFold/Szikszai 口径。

### 3.3 评测指标
- **结构准确度**:F1 / MCC(配对),分 in-clan / cross-clan / novel-clan 报告。
- **分布质量**:预测 reactivity 与留出探针的 Pearson/Spearman(形状)、校准后 MSE(幅度);ensemble 多样性 vs 过散的权衡。
- **不确定度校准**:模型给的 `q_i` 是否校准(可靠性图 / ECE)。
- **消融**:`L_react` on/off、`L_thermo` on/off、guidance `η` 扫描、估计器 (A) vs (A)+(B)。
- **诚实**:全部 in-silico 代理与实验真值严格分栏;不把没做的写成做了。

---

## 4. Baselines / head-to-head(钉死可比性)

| 对标 | 它是什么 | ReactFlow 的比法与赢点 |
|---|---|---|
| **RibonanzaNet2**(判别式 SOTA,◻️) | `序列→reactivity` 回归,2D SOTA | 反方向:`序列→结构分布`;同数据同切分下比 **cross-family F1/MCC** 与**分布/不确定度**;并证明「用其 encoder warm-start + 加分布头」优于纯判别 |
| **eFold / RNAndria**(泛化标杆,◻️) | 主打 cross-family 泛化的 2D 预测 | 同 novel-clan holdout 下比泛化鸿沟;赢点=分布 + reactivity 一致性带来的额外泛化 |
| **TVAE-RNA**(生成式 2D ensemble,◻️) | 生成式出 2D 分布 | 同数据比 ensemble 质量;赢点=**reactivity 显式监督**(TVAE 未用探针矩匹配)+ cross-family 协议 |
| **RNAbpFlow**(3D 瓶颈验证,◻️) | 2D 图→3D 构象 | 不正面比 3D;而是**把 ReactFlow 的 2D 分布喂给它**,证下游 3D 提升(杠杆率论证) |
| ViennaRNA/RNAstructure(热力学基线) | 配分函数/MFE | 作先验与下界;证「数据+物理」优于「纯物理」 |

---

## 5. 新颖性体检 / 同质化风险(诚实自查,决定成败)

**2025 起 RNA flow-matching / 生成式 2D 已高度拥挤**,原始三卖点里有两个已被实质占据:

| 原始卖点 | 是否已被占 | 占用者(◻️ 待核实主源) | 结论 |
|---|---|---|---|
| (a) 生成式出 ensemble | **已占** | RNAbpFlow / TVAE-RNA / MERGE-RNA | 单独讲「生成 ensemble」不再新颖 |
| (b) 把物理塞进速度场 | **已占** | RNA-EFM(energy-based FM) / GraphaRNA | 单独讲「物理入 velocity」不再新颖 |
| (c) 大规模探针训练 | **已占** | RibonanzaNet2(64M 判别式) | 单独讲「大数据训练」不再新颖 |

**→ 收窄后的可辩护交集(唯一空白格):**
> **以 reactivity 为*分布监督*(可微矩匹配,非输入特征)× 分布输出 × cross-family OOD 为主战场**——方向与 RibonanzaNet2 相反(`序列+reactivity→结构分布` vs `序列→reactivity`),且 ensemble-consistency loss 的**可微前向算子 + 期望采样方案**(§2.2)未见等价物。这一格没被上述任一竞品完整占据。

**活风险(写进论文并防御):**
- 若只强调「生成分布」→ 会被归类为「又一个 RNA flow」。**必须**把叙事钉在 §2.2 的一致性损失与 §3 的 novel-clan 泛化上。
- MERGE-RNA 等「physics-based 探针 ensemble」最接近,需正面区分:它们多为**基于物理模拟/采样**产生 ensemble 再对 reactivity,ReactFlow 是**学习式生成 + 可微矩匹配 + 大规模判别 encoder warm-start**,且以泛化为一等目标(◻️ 需精读区分)。

---

## 6. 落地可行性

- **算力友好**:2D 目标远比 3D SE(3) 便宜;不依赖稀缺 3D 数据。判定 **HIGH**。
- **warm-start**:encoder 拟从 **RibonanzaNet2** 迁入(它已在同类数据上学到强 2D 表征),上接 DFM 分布头 + 前向算子。**⚠️ 待确认**:RibonanzaNet2 权重/许可是否公开可得;若不可得,退回自训 encoder 或用 Ribonanza1 版本。
- **代码资产**:DFM 底座、调度器、耦合、z↔x 机制可复用 `../editflow` / `../dogmaflow`(见 §9 映射);前向算子 `f(S)`、一致性损失、Gumbel/Sinkhorn 松弛、guidance 为**本项目新增**。
- **数据管线**:Ribonanza2 下载 + SN_filter + Rfam clan 映射 + 去冗余,均为确定性脚本(可做成可复用 SOP)。

---

## 7. 风险与诚实局限
- **可辨识性**(最硬):reactivity 只定边缘,不定联合(§2.2.6);靠 `L_DFM`/二阶矩/热力学缓解,但要诚实承认上限。
- **同质化**(§5):赛道拥挤,叙事必须死守「分布监督 + OOD」差异化,否则被淹没。
- **前向算子的物理保真**:`f(S)` 是简化模型(未显式建 fraying/stacking 的全部化学);过简会限制上限,过繁会难训 → `e_i` 特征的丰富度是 ablation 旋钮。⚠️
- **guidance 退化**:热力学 `η` 过大 → 塌回 ViennaRNA;需扫描。
- **数据 schema/许可**:Ribonanza2 规模/许可/逐位误差字段均标 ◻️,成文前核实,**不得编造**。
- **warm-start 可得性**:RibonanzaNet2 权重未必公开(⚠️)。
- **伪结**:DFM 放开 nested 约束可表达伪结,但训练稳定性与评测口径未验证(⚠️)。
- **3D**:明确不做 3D 立体化学,只交付 2D 分布(§2.3.4)。

---

## 8. 落地路线图(按用户偏好的 5–7 天迭代周期)

| 周期 | 时长 | 目标与产出 | 关键验证 |
|---|---|---|---|
| **C1 数据管线** | 5–7d | Ribonanza2 下载/清洗/SN_filter + Rfam clan 切分脚本(SOP)+ novel-clan holdout | 切分无家族泄漏(自动校验) |
| **C2 前向算子 pilot** | 5–7d | 实现 `f(S)`(§2.2.2)+ 估计器 (A);在**已知结构**上验证「`E[f(S)]` 能重现实测 reactivity 的相关性」 | Pearson/形状达阈(证前向算子物理可信) |
| **C3 DFM + 一致性损失** ✅ **已实现(合成 pilot)** | 5–7d | 接 DFM 分布头 + `L_DFM+L_react`;小模型跑通端到端 | ✅ `L_react` 下降且不塌边缘退化解:pilot(40 epoch)total `1.389→1.040`、react_magnitude `0.141→0.077`、mean_F1 `0.500→0.611`,确定性 bit-for-bit;手写反传 FD 校验最大相对误差 `1.8e-9`;symbolic 7 项检查残差全 0。**实现于合成 pilot,真实数据规模化训练留待 C5。** |
| **C4 物理约束融合** ✅ **已实现(合成 pilot)** | 5–7d | 硬掩码合法性 + `L_thermo` 半监督 + 推理 guidance | ✅ 掩码 CTMC 采样器在 pilot 序列 `UAUGAUCUCAUA` 上 500/500 样本 100% 合法;精确最大权 nested 投影(`project_max_weight_nested`,O(L³))使 guidance `η` 扫描的 pair energy **可证单调非增**(交换论证,SymPy 残差 0),而 greedy 投影在同一反例上非单调(能量上升);`L_thermo` 的 MSE/KL logit 梯度经 SymPy 与有限差分双验证。新增 32 个测试,全量 131 passed / 1 skipped,coverage 95.81%。**实现于合成 pilot,真实数据规模化训练留待 C5。** |
| **C5 cross-family 主实验** ✅ **真实 scale-up 基础设施已实现;全量/SOTA run 待规模化** | 5–7d | in-clan / cross-clan / novel-clan 三档 + baselines(**eFold** ✅`10.1126/sciadv.adz4967` / **RibonanzaNet2** ✅ Kaggle MIT / TVAE-RNA)head-to-head。**架构决策(已锁定)**:外部冻结编码器(RibonanzaNet2/eFold 权重离线导出 per-nt+pairwise 表征)+ reactflow 纯 stdlib adapter/DFM 头;复用 eFold 官方测试集(引用其 F1:viral mRNA 0.73 / lncRNA 0.44)+ 叠加自建 Rfam-clan novel split(本地
重算)。 | ✅ 冻结特征 shard + provenance、RibonanzaNet2 Kaggle-alpha real checkpoint export(`weights_sha256=c94031719c8a…`)、手写 adapter + split-gradient、`prepare-efold-cache` / `train-efold` / `evaluate-efold` 真实 Dryad JSON 路径已实现。新增:长序列 windowing(`--window-size/--window-stride`)和 length bucketing(`--bucket-boundaries`)写入 cache metadata 并可直接用于 raw JSON smoke;`split-efold-cache` 从 cache 物化 clan/cluster 无泄漏 split manifest + `train/val/test/novel.jsonl`;训练循环 per-phase profiling(`--profile-path`)输出 JSONL + summary,本地 smoke 定位最慢具体步骤为 `model_backward`(其次 `model_forward`);所有训练/评估训练入口写 `training_checkpoint.json` 以归档 config/parameters/history/metadata;lazy optional PyTorch base 后端(`--backend torch --torch-device ...`)已接入且不破坏 `import reactflow` 零重依赖。服务器 scale-up:Dryad `efold_train.json` cache 64 条短合法记录训练,`archiveII/PDB/viral` cache 分别 32/32/11 条评估;single-only frozen shard(`--d-pair 0 --n-probe 0`)64 条仅 8.8MB;base loss `3.659→3.326`,RibonanzaNet2 warm-start loss `3.660→3.293`,matched `64/64`;短序列 tier mean F1(base:PDB `0.183`,archiveII `0.029`,viral `0.034`;warm:PDB `0.188`,archiveII `0.029`,viral `0.034`)。**这些是 scale-up 诊断,非 SOTA/论文数字。** 剩余:外部 Rfam/MMseqs 真标签接入、全量 frozen export、服务器规模化 profiling/torch 对比、同口径 baseline rerun 与消融。全量 `283 passed / 2 skipped`,coverage `92.79%`,`import reactflow` 零重依赖。 |
| **C6 纠偏 + 消融 + 成文** | 5–7d | 加估计器 (B) 纠偏;`L_react/L_thermo/guidance/estimator` 消融;下游喂 RNAbpFlow 证 3D 提升;文献账本 ◻️→✅ | 消融表 + 杠杆率论证 + 引用核实 |

---

## 9. 文献账本(honest markers,成文前把 ◻️ 全升 ✅)

> 本轮**未逐条重新核主源**,以下按既有认知列入,标 ◻️;成文前须查证 arXiv/DOI/OpenReview ID 与卷期页。**禁止带病引用。**

**竞品(RNA 生成/预测):**
- ◻️ RNAbpFlow — Nature Methods(卷期页/DOI 待核实):SE(3) FM,2D 图→3D 构象。
- ✅ **本轮核实** RibonanzaNet2 — 判别式 2D reactivity 模型,warm-start 编码器来源。Kaggle 模型卡 <https://www.kaggle.com/models/shujun717/ribonanzanet2>(MIT model variation),报告 ~100M 参数、30M RNA 100mer DMS/SHAPE profile、4% validation holdout;架构接口(`ninp=384`、`pairwise_dimension=128`、`nlayers=48`,token 词表 {A:0,C:1,G:2,U:3} pad_id=4)已对照 Shujun-He/RibonanzaNet `Network.py` 核实并落入 `scripts/export_frozen_features.py` 的 torch 后端。
- ✅ **本轮核实** eFold / RNAndria — Rouskin lab,cross-family 泛化标杆。**Science Advances 2026, 12(9):eadz4967**,DOI 10.1126/sciadv.adz4967;数据 Dryad DOI 10.5061/dryad.79cnp5j95、门户 <https://rnandria.org/>。官方报告 F1:viral mRNA `0.73` / lncRNA `0.44`(作为 cited 栏引用,本地重算严格分栏、公共集行保持 `pending`)。
- ◻️ TVAE-RNA — 生成式 2D ensemble(主源待核实)。
- ✅ **本轮核实** RNADiffFold — Wang et al., "RNADiffFold: generative RNA secondary structure prediction using discrete diffusion models", **Briefings in Bioinformatics 2025, 26(1), bbae618**, DOI 10.1093/bib/bbae618。**最接近的 method analog**:同为离散生成式 2D(multinomial diffusion 去噪 contact-map),报告 within/cross-family 且能捕捉多构象;源 Table S2 ArchiveII F1 `0.880`、bpRNA TS0 F1 `0.711`,diffusion 部分 836K 参数。**区分点**:RNADiffFold 去噪 contact-map 像素逼近**单个**结构、无化学探针监督;ReactFlow 加 reactivity 矩匹配分布监督 + Rfam-clan OOD 协议。
- ✅ **本轮核实(纠正)** RNA-EFM — Abir & Zhang, "RNA-EFM: energy-based flow matching for protein-conditioned RNA sequence-structure co-design", **Bioinformatics Advances 2025, 5(1), vbaf258**, DOI 10.1093/bioadv/vbaf258。**实为蛋白条件下的 RNA *3D backbone* 序列-结构 co-design**(flow matching + Lennard-Jones 能量 idempotent refinement),**非 2D energy-FM**。仅共享「能量/物理入 flow」的思路;与 ReactFlow(探针监督的 2D ensemble 预测)不构成直接 2D baseline。
- ◻️ MERGE-RNA — physics-based 探针 ensemble(主源待核实,最需正面区分)。
- ◻️ RiboFlow / RNA-FrameFlow / GraphaRNA — RNA flow/生成同赛道(主源待核实)。
- ◻️ Szikszai et al. — RNA 2D 预测跨族泛化差的量化(主源待核实)。

**方法/基座(well-known,仍需核卷期页):**
- ✅ **本轮核实** Discrete Flow Matching — Gat et al., "Discrete Flow Matching", **NeurIPS 2024**, arXiv:2407.15595(Meta FAIR);提供 mixture/linear 概率路径与 probability-denoiser(x1-prediction)后验,ReactFlow 去噪头直接复用。
- ✅ **本轮核实** Discrete Flow Models — Campbell et al., "Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design", **ICML 2024**(PMLR 235),arXiv:2402.04997;提供 CTMC rate-matrix 视角与 conditional-rate/master-equation,ReactFlow `conditional_rate_matrix` 依此实现并经 SymPy 验证。
- ✅/◻️ Edit Flows — arXiv:2506.09018(NeurIPS'25,CodonFlow ledger 已核;本项目复用底座)。
- ◻️ Gumbel-Softmax — Jang et al. ICLR 2017 / Maddison et al.(Concrete)。
- ◻️ Sinkhorn / Gumbel-Sinkhorn — Cuturi NeurIPS 2013 / Mena et al. ICLR 2018。
- ◻️ Turner 最近邻能量模型 / ViennaRNA(Lorenz et al. 2011)/ RNAstructure(Reuter & Mathews 2010)。
- ◻️ SHAPE-directed folding(Deigan/Weeks)、DMS-MaP(Zubradt/Rouskin)。
- ◻️ Ribonanza(Kaggle/Eterna,He/Das 等) + Ribonanza2 数据集官方页(规模/许可/schema)。

---

## 10. 开放问题(待核实/迭代)
1. **RibonanzaNet2 warm-start 可得性**:权重/许可是否公开;不可得时的退路。
2. **前向算子 `f(S)` 的物理丰富度**:`e_i`(fraying/stacking)要多细?先做最简 A/C-unpaired 版,再按 C2 结果加。
3. **估计器 (A) 的 mean-field 偏差有多大**:用 (B) 采样估计标定;决定是否常开 (B)。
4. **二阶矩数据**:是否有 mutate-and-map/MaP-2D 可用于 co-reactivity 项(收窄可辨识性)。
5. **伪结**:DFM 放开 nested 的训练稳定性与评测口径。
6. **guidance `η` 与半监督 `λ_td` 的联合调参**:避免塌回 ViennaRNA。
7. **cross-family 切分粒度**:clan 是否够严;要不要叠加二级结构类型分层。
8. **与 MERGE-RNA 的精确区分点**:精读后补一张逐维对比表(mirror CodonFlow §9 的 MIMIC 表)。

---

> 下一步可选:① C1 数据管线 SOP(Ribonanza2 清洗 + Rfam clan 切分,可复用脚本);② C2 前向算子 pilot(在已知结构上验证 `E[f(S)]` 重现 reactivity 相关性——这是整个方法的物理可信度前提);③ 精读 MERGE-RNA / TVAE-RNA 主源,补 §5 逐维区分表并把 §9 的 ◻️ 升 ✅。
