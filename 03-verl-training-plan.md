# Harness 约束下的模型强化学习:实验规划

> 版本:1.0 · 日期:2026-10-21
> 上游文档:[`01-research-report.md`](./01-research-report.md)(调研) · [`02-coding-plan.md`](./02-coding-plan.md)(harness 侧实现计划)

---

## 0. 一页速览

**本文档只定义实验规划,不包含实现。** 实现见后续独立文档。

**目标**:用 verl 做强化学习,训练一个在 **mini-swe-agent 约束下**表现更好的 LLM;并与 harness 侧工作(`02-coding-plan.md`)合成一个能回答「harness 改进与 model 训练是互补还是可替代」的联合研究。

**三阶段**:

| 阶段 | 名称 | 产出 | 是否需要 GPU |
|---|---|---|---|
| **P1** | 接线与可行性 | 单条完整轨迹走通 rollout → reward → 梯度;reward 非零率与难度分布基线 | 否(可纯 CPU + 推理服务) |
| **P2** | 训练 | 冷启动 SFT → GRPO;逐代训练台账 | 是 |
| **P3** | 评测与归因 | 2×2 主实验 + 消融 + 跨 harness 泛化 | 是(评测) |

**核心方法论约束**(来自 Agentic RL 环境一文的第 11 章):

> 无论哪种接法,训练与评测的 **prompt 快照**、**观测截断策略**、**token 级记录方式** 必须完全一致。任何一项不一致,你训出来的都是「某个特定 harness 的专用模型」,换一个脚手架就掉分,而 **loss 曲线全程正常**。

**预算量级**:P2 约 3–5 天 GPU 机时 + $8k–25k;P3 约 2–3 天 + $3k–8k。

---

## 1. 研究定位

### 1.1 与 harness 侧工作的关系

本项目与 `02-coding-plan.md` 是同一研究程序的**两条正交轴**:

| | 模型未针对该 harness 训练 | 模型针对该 harness 训练 |
|---|---|---|
| **harness 未优化** | ① 基线 | ② 只训模型 |
| **harness 已优化(gen-K)** | ③ 只做 harness 优化 | ④ 两者都做 |

**② → ④ 的差,就是「harness 特化过拟合」的量。** 这个量只有同时拥有两条轴才测得出来——RHI(只做 harness)和 Athanor(只做 agent 团队)都做不到。

**依赖关系**:

```
02-coding-plan.md  →  gen00.yaml ... genK.yaml   (harness 配置谱系 + 逐代台账)
                            │
                            ▼
03(本文档)         →  训练时把 harness 钉死在 gen00
                       评测时在 gen00..genK 上扫描 → (harness 代 × model) 二维表
```

**⚠️ 关键纪律:训练期间 harness 必须是冻结的 gen00。** 若一边训模型一边改 harness,两者会共同适应,你测到的是一个闭环自洽的数字,换 harness 就崩。harness 的进化只在**评测阶段**扫描。

### 1.2 三个研究问题

| RQ | 问题 | 对照实验 |
|---|---|---|
| **RQ1** | 在固定 harness 下,RL 能否把「通用 instruct 模型」变成「该 harness 的专用模型」?提升幅度多少? | §5.1 ① vs ② |
| **RQ2** | 提升来自**通用能力**还是**harness 特化**? | §5.1 跨 harness 评测(§4.5):在 Claude Code / SWE-agent / OpenHands SDK 下复测 |
| **RQ3** | harness 改进与 model 训练是**互补**还是**可替代**? | §5.1 ②/③/④ 三者对比,及交互项 |

### 1.3 假设与可证伪性

| 假设 | 表述 | 被证伪的条件 |
|---|---|---|
| **H1** | mini-swe-agent 的约束是**可学习的行为**而非能力上限;RL 能显著降低格式错误率并提升解决率 | 训练后解决率提升 < 噪声底噪 δ_min,或格式错误率无下降 |
| **H2** | 训练收益**大部分是 harness 特化**,跨 harness 迁移会显著衰减 | 跨 harness 衰减 < 3pp(则特化不成立) |
| **H3** | harness 改进与 model 训练**部分互补** | 交互项显著为负(两者互相抵消) |

> H2 若被证伪(跨 harness 几乎不掉分),则本项目退化为「普通 agentic RL 有效」,新颖性下降——**这是最需要警惕的负面结果,必须在 P1 阶段就用一个便宜模型预判**。

---

## 2. 实验对象与固定项

### 2.1 Harness 侧:mini-swe-agent 配置快照

**全程只认 `gen00.yaml`**(来自 `02-coding-plan.md` §4.5 adapter 产出的 base config)。所有「固定项」在下表冻结:

| 项 | 冻结值 | 理由 |
|---|---|---|
| `model_class` | `litellm_response`(tool-call 协议) | 与 verl `multi_turn` 的 tool-call 模式同源;训练/评测必须一致 |
| 动作协议 | 原生 tool call,单工具 `BASH_TOOL(command: str)` | 避免文本围栏与 tool call 两套协议混用 |
| `system_template` / `instance_template` | gen00 快照,**字节级冻结** | 文章第 11 章一致性要求 |
| `observation_template` / `format_error_template` | gen00 快照,带版本号 | 同上 |
| `environment_class` | `swerex_docker`(小规模)/ `docker`(单机冒烟) | 执行层与策略层解耦 |
| 容器镜像 | **digest 固定**,不用 tag | 跨周对比必须同环境 |
| `step_limit` | 60 | 轨迹长度上限 |
| `cost_limit` | 2.0 | 与 gen00 一致 |
| `wall_time_limit_seconds` | 1800 | 单条轨迹墙钟上限 |
| `max_consecutive_format_errors` | 3 | 格式纠错预算 |
| 终止协议 | 首行 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且 returncode 0 | mini-swe-agent 原生约定 |

> **⚠️ v2.4.6 字段名更正**:Agentic RL 环境一文用 `environment_type`,但 2.4.6 的 `get_environment()` 取的是 `environment_class`。配置以本仓库 `mini --help` 与 `environments/__init__.py` 为准。

### 2.2 模型侧

| 角色 | 选择 | 说明 |
|---|---|---|
| **被训策略(policy)** | 单个 base model,固定 chat template | P1 用小模型(如 1.7B–4B)验证链路与统计功效;P2 再上目标规模 |
| **对照组(control)** | **同一 base model 的未训练版本** | 唯一合法的对照;不能用别的模型 |
| **评测裁判** | 无 LLM judge,全部用可验证自动评测 | 本文路线不采用 RHI 的成对打分 |
| **可选压力对照** | 同 base 但不同 chat template / 工具集 | 用于 H2 的辅助解释 |

**禁止项**:训练与评测之间任何一次「换个模型版本」「换 chat template」「换 prompt」都作废该代结果。

### 2.3 任务与数据

| 层 | 选择 | 规模 | 用途 |
|---|---|---|---|
| 演化/训练集 | 环境工厂产出(SWE-smith / SWE-Gym / R2E-Gym / SWE-rebench) | 2.4k–50k,按 pass_rate 分桶 | 训练 |
| 演化 dev | 从训练集里按仓库去重划出 | 20% | 选 checkpoint、监控 |
| **评测 held-out** | **SWE-bench Verified 划出 200 题** | 200 | 最终报告 |
| **评测 held-out** | **Terminal-Bench 2.1 划出 145 题** | 145 | 跨任务族泛化 |
| 社区对照 | SWE-bench Verified 全量 / TB 2.1 全量 | 500 / 445 | 一次,最终版 |

**切分纪律**:切分文件入库 + SHA256,**任一代际改变切分即作废全部历史结果**。切分按**仓库**而非按题切,避免同仓库的题跨集合泄漏。

### 2.4 沙箱层

按规模分档,详见 §7.1 的 CPU 账:

| 档 | 后端 | 用途 |
|---|---|---|
| **D0** | 本地 `docker` | P1 冒烟、单机调试 |
| **D1** | `swerex_docker` / `swerex_modal` | P2 训练(执行层与训练机解耦) |
| **D2** | AgentENV(microVM,快照 fork) | P2 大 K 值时(需同题 fork) |

**选择判据(引自原文)**:不是「容器还是 VM」,而是「**我的 rollout 需不需要 fork**」。GRPO 要同题采 K 条 → 需要 fork → 快照机制的价值最高。

---

## 3. 训练协议

### 3.1 五阶段流水线

```
Stage 0  接线冒烟   ──► 一条完整轨迹 + 一次梯度
Stage 1  造题分桶   ──► 难度分布 + pass_rate 统计
Stage 2  验证器自检 ──► 漏网率 0 / 空策略全 0 / 两次一致
Stage 3  冷启动 SFT ──► reward 非零率进入 20~60%
Stage 4  GRPO       ──► 逐代训练台账
```

**阶段依赖是硬的**:Stage 3 未达到「reward 非零率 ≥ 20%」不得进入 Stage 4;Stage 2 三项自检任一不过不得进入 Stage 3。

### 3.2 Stage 0:接线冒烟(P1 核心)

**这是整个项目最重要的一个实验**,因为下游所有坑的症状都是「agent 看起来不工作」。

**通过标准(全部满足才算通过)**:

| # | 检查项 | 判据 |
|---|---|---|
| 1 | 单条完整轨迹 | 从 task 到终止,轨迹非空,终止标记正确 |
| 2 | 动作可解析 | 每步都产出**恰好一个** tool call,`parse_toolcall_actions` 不抛 FormatError |
| 3 | 命令真执行 | 至少一条命令在容器里产生了非空 stdout |
| 4 | 观测回流 | `observation_template` 渲染结果进了下一轮 messages |
| 5 | reward 可算 | 验证器在**新沙箱**里给出 0/1,且人工确认一次正确答案得 1、一次错误答案得 0 |
| 6 | 梯度可回传 | 一次 backward 后 policy 参数发生变化(logprob 非 NaN) |
| 7 | 逐题记录落盘 | 含 finish_reason、原文、每步 tool call、结束原因 |

**必须做的负面测试**(缺一不可):

| # | 负面测试 | 期望行为 | 若不符说明 |
|---|---|---|---|
| N1 | 删掉 `stop_token_ids` | 训练/评测都出现重复刷屏且被截断 | 停 token 未对齐 |
| N2 | 让模型输出自造的结束标记 | 轨迹跑到 step_limit 仍不终止 | 终止协议未对齐 |
| N3 | 故意让模型输出无 tool call | 收到 `format_error_template` 的纠错反馈 | 格式纠错回路未通 |
| N4 | 同一 patch 跑两次验证 | 结果一致 | 测试 flaky |
| N5 | 空策略(不做任何事)跑全量 | reward 全 0 | 存在默认通过分支 |

> N1–N3 直接对应「模型侧一致性」的三条硬要求。若这三条不通,后面所有训练数字都不可信。

**P1 阶段还必须产出的东西**:噪声底噪(§4.4)与难度分布(§3.3)——**这两项决定 P2 的全部超参**。

### 3.3 Stage 1:造题与难度分桶

**难度分桶规则**(不是数据清洗,是保护优势信号):

```
pass_rate = 0.0        → 丢弃(组内无正样本,advantage 恒为 0)
pass_rate ∈ (0, 0.9)   → 训练主力
pass_rate ∈ [0.9, 1]   → 丢弃或降权
```

**分桶按训练步重算**——难度会漂移:训练到中期「难题」变简单,中间桶整体右移。

**必须记录的统计量**(P1 产出):

| 量 | 用途 |
|---|---|
| 每题的 pass_rate(采样 N=8 估计) | 分桶依据 |
| 组内正样本概率 `1-(1-p)^K` | 预测零梯度比例 |
| 零梯度题占比 | **P2 预算的核心依据** |
| 每题平均步数、token 数 | 成本预测 |

**组内正样本概率参考表**:

| 模型 pass_rate p | K=8 | K=16 |
|---|---|---|
| 0.10 | 57%(43% 零梯度) | 81% |
| 0.05 | 34%(66% 零梯度) | 56% |
| 0.03 | 22%(78% 零梯度) | 40% |

> 这是「500 题训到 200 步就刷完了」的定量解释。**如果零梯度题占比 > 40%,加大 K 或扩题量是唯一解**,加 GPU 没用。

### 3.4 Stage 2:验证器与三项自检

**四层门控**(任一步失败即 reward=0,后续昂贵步骤不执行):

| 层 | 动作 | 耗时 | 拦截什么 |
|---|---|---|---|
| 1 | 新开沙箱,不复用 agent 环境 | ~2s | 环境残留 |
| 2 | `git apply --check` 先验补丁 | <0.5s | 恶意/畸形补丁 |
| 3 | 只读挂载测试 + 固定随机种子 | <0.5s | 测试被改写 |
| 4 | 语法 / AST 审计(禁危险调用) | <1s | 危险副作用 |
| 5 | 公开测试 | 10–60s | 主要判据 |
| 6 | **隐藏测试**(换实例复核) | 10–60s | 最终判据 |

**三条硬规则**:

1. 测试**只读挂载**到工作目录之外
2. 每次验证**重开实例**,不复用 agent 跑过的环境
3. 跑之前**校验测试目录哈希**,不一致直接判 0

**三项必做自检**(引自原文,任一不过不得进 Stage 3):

| 自检 | 判据 | 不通过说明 |
|---|---|---|
| **mutation 自检** | 用弱模型专门生成的「看起来对但没解决问题」的补丁,漏网率必须为 0 | 验证器可被攻破 |
| **空策略全量** | reward 必须全 0 | 存在默认通过分支 |
| **两次一致** | 同一补丁跑两次结果一致 | 测试 flaky |

> **⚠️ 已知风险**:审计工作发现,五个终端 benchmark 的近两千个任务里,**16% 能被前沿模型仅凭任务描述骗过**。**自建题库必须先跑 mutation 自检**,否则你不知道自己是不是那 16%。

**reward 定义**:二值(隐藏测试通过=1,否则=0)。不设过程奖励——过程信号容易被刷。

### 3.5 Stage 3:冷启动 SFT(不可省)

**不得从 base model 直接 GRPO。** 理由:格式错误率在冷启动阶段通常 >50%,reward 非零率接近 0,组内无正样本,梯度全零。

**做法(拒绝采样 / STaR)**:

1. 用当前 base model(配合 gen00 harness)对分桶后的训练集采样 N 条轨迹
2. 用 Stage 2 的验证器判分
3. 只保留**通过**的轨迹
4. 用这些轨迹做 SFT(轨迹即标准答案,action/observation 交替序列)

**通过标准**:

| 指标 | 阈值 | 依据 |
|---|---|---|
| reward 非零率 | **20%–60%** | 原文健康区间;低于 20% 是题太难,高于 60% 是验证器漏 |
| 格式错误率 | < 10% | 协议内化是 RL 的主要收益点(H1) |
| 有效 SFT 样本量 | ≥ 5k 条轨迹 | 经验下限;不足则扩大采样 |

> 若 SFT 后 reward 非零率仍 < 20%,**先回查 Stage 2 验证器**(大概率是验证器太严或题太难),不要加 GPU。

### 3.6 Stage 4:GRPO

**接法**:`remote_agent`(harness 自带循环),不选 `tool_agent`。

**理由**:选 `tool_agent` 要在 verl 侧重写 mini-swe-agent 的 system prompt,训出来的是「verl 循环下的模型」,拿到真 mini-swe-agent 评测会有系统性落差——**而 loss 曲线全程正常**。

**必须显式设置、不得用默认值的项**:

| 项 | 要求 | 不设的后果 |
|---|---|---|
| `rollout.mode` | `async` | 多轮退化成串行 |
| `multi_turn.enable` | `True` | — |
| `multi_turn.format` | 与模型 chat template 匹配 | 工具调用永远解析不出,**轨迹长度恒等于 1** |
| `multi_turn.max_assistant_turns` | 显式设大(≥ `step_limit`) | 长任务中途硬截断,**reward 永远是 0** |
| 训练/验证的 tool config | **同一份** | 指标莫名偏低 |
| `function_tool_path` 的 docstring | `Args:` 段不能省 | 缺了直接抛解析异常 |

**超参初值**(以 P1/P2 的难度分布为准调整):

| 参数 | 初值 | 调整依据 |
|---|---|---|
| K(每题采样数) | 8 | 16 → CPU 账单翻倍(§7.1) |
| prompt batch | 4–8 | 受 CPU 约束,非 GPU 约束 |
| 每题 epoch | 1 | pass_rate ≥ 0.9 的题已无信号 |
| KL 系数 | verl 默认 | 不手调,除非出现格式退化 |
| 训练/验证用同一份 tool config | 强制 | — |

**逐代训练台账**(每代一条,字段见 §6.1):

```
代次 / 策略版本哈希 / 训练步 / reward 非零率 / 格式错误率 /
组内正样本比例 / 零梯度题占比 / 平均步数 / 平均 token / 成本 /
评测分数(dev) / 评测分数(held-out,见 §4.2)
```

### 3.7 停止准则

| 条件 | 动作 |
|---|---|
| reward 非零率连续 2 次评估跌破 20% | 停,回查验证器与难度分布 |
| 格式错误率在训练后反而上升 | 停,回查 stop token / 动作协议一致性 |
| dev 分数连续 2 代不升且低于 δ_min | 停,判定为噪声,做失败分析 |
| 策略在 held-out 上不升 | **不停**,但标记为「harness 特化」候选,进入 RQ2 判定 |

---

## 4. 评测协议

### 4.1 训练/评测一致性清单

**这是本项目最容易翻车的地方。** 两张表合起来是 9 条硬要求。

**A. 通用三项**(原文第 11 章):

| # | 项 | 落地方式 |
|---|---|---|
| A1 | prompt 快照 | `system_template` / `instance_template` 从同一文件读出,记录 SHA256 |
| A2 | 观测截断策略 | `observation_template` / `format_error_template` 版本化,训练与评测同版本号 |
| A3 | token 级记录 | 逐 token logprob 落盘,可重放 |

**B. mini-swe-agent 特有五项**:

| # | 项 | 不一致的症状 |
|---|---|---|
| B1 | **动作协议同源**(训练与评测都用 tool call) | 训练能解析、评测解析不出 |
| B2 | **终止 token 原生**(首行 + returncode 0) | 永不终止 → step_limit → reward 全 0 |
| B3 | **stop token 双向对齐**:checkpoint 的 `generation_config.eos_token_id` 含 chat 模板 eos **且** 评测 vLLM 显式传 `stop_token_ids` | 输出后不停止,重复刷屏;**判分器取最后一个数字的兜底规则会让分数虚高** |
| B4 | 容器镜像 digest 一致 | 环境漂移被误记为能力提升 |
| B5 | 温度/采样参数一致 | 方差混入 |

> **B3 的隐蔽性**:截断的响应因为「最后一句看起来像答案」而被判对,分数不降反升。**唯一可靠的发现手段是逐题记录里的 `finish_reason` 与 `truncated` 标记。**

### 4.2 评测集划分与使用纪律

| 集合 | 内容 | 何时跑 | 用途 |
|---|---|---|---|
| `dev` | 训练集划出的 20%(按仓库) | 每个 checkpoint | 选点、监控 |
| `heldout-sb` | SWE-bench Verified 200 题 | **仅最终代 + 少数候选代** | 主报告 |
| `heldout-tb` | Terminal-Bench 2.1 145 题 | 同上 | 跨任务族泛化 |
| `full-sb` / `full-tb` | 全量 | **仅最终一次** | 社区对照 |

**纪律**:held-out 集合的评测结果**不得进入任何优化决策**(不用于选字段、不用于选 checkpoint 的早停点)。一旦用于决策,该集合即降级为 dev。

### 4.3 指标体系:不只看分数

分三组。**第二组是本项目的特色——它度量「约束是否被内化」,比解决率更灵敏。**

**A. 任务层(主指标)**

| 指标 | 定义 |
|---|---|
| 解决率 | 隐藏测试通过率 |
| 配对解决率 | 同一题上两版策略的成功/失败配对(用于 McNemar) |
| 成本 / 题 | USD per solved task |

**B. 约束内化层(本项目特色)**

| 指标 | 定义 | 期望方向 |
|---|---|---|
| **格式错误率** | 每步解析失败的比例 | ↓ |
| **无效动作率** | 命令不存在/语法错/超时 | ↓ |
| **终止正确率** | 用原生终止 token 且 returncode 0 结束的轨迹占比 | ↑ |
| **步数耗尽率** | 撞 `step_limit` 结束的轨迹占比 | ↓ |
| **上下文占用** | 终止时的 prompt token 数分布 | ↓(mini-swe-agent 无压缩) |
| **状态持久化误判率** | 因假设 `cd`/`export` 生效而失败的步占比 | ↓ |
| **文件编辑方式分布** | heredoc / `python - <<` / sed / 其他 的比例 | 收敛到高效方式 |
| **重复命令率** | 连续两步执行等价命令的比例 | ↓ |

> **为什么这一组重要**:解决率是**结果**,约束内化指标是**过程**。SFT→GRPO 期间解决率可能只动 1–2pp(在噪声边缘),但格式错误率可能从 40% 降到 5%——**后者才是 H1 的直接证据**,且样本效率高得多。

**C. 训练期(健康监控)**

| 指标 | 健康区间 | 异常含义 |
|---|---|---|
| reward 非零率 | 20–60% | 过低题太难,过高验证器漏 |
| GPU 等待环境占比 | < 10% | 热池不够深 / 调度串行 |
| setup 延迟 P95 | < 60s | 镜像太大 / 磁盘 IO 不足 |
| episode 失败率 | < 2% | 沙箱不稳 / 网络断连 |
| 快照命中率 | > 70% | 没复用状态,每次重装 |
| 组内正样本比例 | 越高越好 | 零梯度题浪费算力 |
| 零梯度题占比 | < 30% | > 40% 则加 K 或扩题量 |

### 4.4 统计方法

**因为 benchmark 是二值的,底层仍是配对比较。** 设 b = 新过旧不过,c = 反向:

```
Δscore = (b − c)/n        SE ≈ sqrt(b + c)/n        显著判据: |b − c| > 1.96·sqrt(b + c)
```

**噪声底噪(必做,P1 产出)**:同配置跑 5 个不同 seed,估出「配置不变时 Δscore 的标准误」,得到**可检测阈值 δ_min**。

| 判据 | 方法 | 用途 |
|---|---|---|
| 主判据 | 逐题配对 bootstrap(10k 重采样)95% CI | 连续分数差分 |
| 严格判据 | McNemar 检验 | 逐题成败翻转的显著性 |
| 接受规则 | **两者取更严**(CI 下界 > 0 **且** p < 0.05) | 判定新一代是否接受 |
| 多 seed 连续化 | 每题跑 k 个 seed,分数取均值 | 把 {0,1} 变成 {0,1/k,…,1},压低配对差分方差 |
| 归因 | 按字段分组的 Δscore 差 | 逐字段效应量 |

> **多 seed 是必需的**:单 seed 下,相邻两代即使有 2–4pp 的真实改进也测不出(见 `02-coding-plan.md` §6.1 的计算表)。**≥3 seeds,理想 5。**

### 4.5 跨 harness 泛化评测(RQ2 的关键)

**在同一批 held-out 题上,把训练后的策略放进不同的 harness 跑**:

| harness | 角色 | 说明 |
|---|---|---|
| **mini-swe-agent(gen00)** | 训练 harness | 域内 |
| **mini-swe-agent(genK)** | 优化过的 harness | 跨 harness 变异 |
| **SWE-agent** | 异构 harness | YAML 配工具集,协议不同 |
| **OpenHands SDK** | 异构 harness | 完整 SDK,工具面不同 |
| **Claude Code** | 商用 harness | 最强的域外对照 |

**输出**:一张 `harness × 策略版本` 的分数矩阵。**「训练于 mini、只在 mini 上测」与「训练于 mini、跨 harness 测」的落差 = H2 的量化证据。**

**注意**:跨 harness 评测有**提示词落差**问题(不同 harness 的 system prompt 不同)。所以需要同时报告:
- 绝对分数(跨 harness)
- **相对各自 baseline 的提升**(跨 harness)← 这个才是可比的

---

## 5. 实验矩阵

### 5.1 主实验:2×2

| 格 | 配置 | 训练 | 评测 | 回答 |
|---|---|---|---|---|
| **①** | base model + gen00 harness | 无 | 全部 harness | 基线 |
| **②** | **SFT+GRPO 后的 model** + gen00 harness | 有(gen00 冻结) | 全部 harness | RQ1 |
| **③** | base model + genK harness | 无 | 全部 harness | harness 侧收益(来自 `02`) |
| **④** | **SFT+GRPO 后的 model** + genK harness | 有(gen00 冻结) | 全部 harness | RQ3 交互项 |

**关键读法**:

- ② − ① = **model 侧收益**
- ③ − ① = **harness 侧收益**
- ④ − ② − ③ + ① = **交互项**。显著为负 → model 训练与 harness 优化互相抵消;显著为正 → 互补
- ② 在异构 harness 上的衰减 = **H2**

### 5.2 必做基线

| # | 基线 | 作用 |
|---|---|---|
| B1 | **固定 config 跑 K 代**(不训练) | 排除「多跑几轮本身」的假改进 |
| B2 | **随机配置搜索**(仅 harness 侧) | 标定「配置空间本身有多大」 |
| B3 | **仅 SFT,不做 GRPO** | 分离 SFT 与 RL 的贡献 |
| B4 | **无 stop_token / 无终止协议**的消融 | 量化 B3 一致性项的影响 |

> B1、B2 也承接 `02-coding-plan.md` 的同类基线,**两边的基线要能互相复用**,不要各写一套。

### 5.3 消融

| 维度 | 取值 | 回答 |
|---|---|---|
| **K(组大小)** | 4 / 8 / 16 | K 换 CPU 账单还是换梯度质量? |
| **训练方式** | SFT only / SFT+GRPO / GRPO only | 冷启动是否必需 |
| **reward 粒度** | 二值 / 二值+步数惩罚 | 过程信号是否有用还是引入刷分 |
| **难度分桶** | 分桶 / 不分桶 | 零梯度题的代价 |
| **seed 数** | 1 / 3 / 5 | 统计功效的实际收益曲线 |
| **harness 谱系** | gen00 / genK / 混合 | 训练用哪个 harness 影响泛化 |
| **冷启动数据量** | 1k / 5k / 20k 轨迹 | 数据量-收益曲线 |
| **模型规模** | 1.7B / 4B / 8B | 规模是否改变「约束内化」的收益结构 |

---

## 6. 指标定义

供后续实现直接引用(此处只定义,不实现)。

### 6.1 训练台账字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `gen` | int | 训练代次 |
| `policy_hash` | str | 策略权重哈希 |
| `step` | int | 训练步 |
| `reward_nonzero_rate` | float | 健康区间 0.20–0.60 |
| `format_error_rate` | float | 约束内化核心指标 |
| `group_positive_ratio` | float | 组内有正样本的题占比 |
| `zero_grad_task_ratio` | float | 零梯度题占比 |
| `mean_turns` | float | 平均步数 |
| `mean_prompt_tokens` | float | 上下文占用 |
| `cost_usd` | float | 累计成本 |
| `eval_dev` | float | dev 分数 |
| `eval_heldout_sb` | float | held-out SWE-bench(仅最终代) |
| `runtime` | dict | mini-swe-agent / verl / 镜像 digest / 模型版本 |

### 6.2 评测记录字段(逐题)

| 字段 | 说明 |
|---|---|
| `task_id` / `repo` | 题标识与仓库(用于仓库级去重分析) |
| `seed` | 采样 seed |
| `passed` | bool,隐藏测试结果 |
| `finish_reason` | **必存**——`stop` / `length` / `tool_calls` |
| `truncated` | bool,是否被截断 |
| `n_turns` / `n_tool_calls` | 轨迹长度 |
| `format_errors` | 格式错误次数 |
| `terminated_correctly` | 是否用原生终止协议 |
| `ctx_tokens_final` | 终止时上下文占用 |
| `edits` | 编辑方式计数(heredoc / sed / python / 其他) |
| `cost_usd` / `duration_s` | 成本与时长 |
| `trajectory_path` | 原始轨迹 |
| `retries` | 重试次数 |

> **`finish_reason` 与 `truncated` 是必存字段**。只存总分就发现不了 B3 那个「分数虚高」的坑。

### 6.3 归因指标

| 指标 | 定义 |
|---|---|
| 单字段效应量 | 某字段被改动的代际,其 Δscore 减去未改动代际的 Δscore,配对 bootstrap |
| 约束内化增益 | 格式错误率、步数耗尽率、状态持久化误判率的相对下降 |
| 跨 harness 保持率 | 异构 harness 提升 ÷ mini-swe-agent 提升 |
| 成本效率 | USD per solved task 的代际变化 |

---

## 7. 预算与算力

### 7.1 CPU 账(先算这个)

```
并发 episode = prompt 数 × K
CPU 需求     ≈ 并发 episode × 1 核
机器数        = CPU 需求 / (单机核数 × 0.8)
```

| 训练配置 | 并发 episode | 环境侧机器 |
|---|---|---|
| 8 卡 × 4 prompt × 4 | 128 | 2 台 64 核 |
| 16 卡 × 8 prompt × 8 | 1024 | 10 台 128 核 |
| 32 卡 × 8 prompt × 16 | 4096 | **40 台 128 核** |

> **一台 32 卡 GPU 节点通常只有 128 核。** mini-swe-agent 一轮一条命令 = 每步一次容器 exec 往返,60 步轨迹光等命令返回就 ~18 秒。**第一堵墙是 CPU,不是 GPU。**
>
> K 从 16 降到 8,CPU 直接减半,代价是组内优势估计变噪。**多数团队最终停在 8。**

### 7.2 时间与成本

| 阶段 | 时长 | 成本量级 |
|---|---|---|
| P1 冒烟 + 噪声底噪 + 难度分布 | 2–4 天(主要 CPU + 推理) | $500–2,000 |
| P2 Stage 3 冷启动 SFT | 2–4 天 GPU | $3k–8k |
| P2 Stage 4 GRPO(3–5 代) | 3–5 天 GPU | $5k–15k |
| P3 评测与 2×2 + 消融 | 2–3 天 | $3k–8k |
| **合计** | **约 3 周** | **$12k–33k** |

**省钱档**:
- 演化用 1.7B 模型,最终只训练 4B
- K=4,dev 集从 300 题降到 150 题
- 消融只在 `heldout-tb` 上跑,不全量
- → 可压到 **$5k–12k / 2 周**

**锚点参考**(Agentic RL 环境一文,未经独立核实):8×H200、GLM-4.7-Flash、55 rollout、约 3400 episode、64 并发 microVM、episode 失败率约 1%。

### 7.3 环境侧容量监控

按 §4.3 的 C 组指标设告警。**前两个不达标,加再多 GPU 也没用**(GPU 等待环境占比 ≥ 10% 或 setup P95 ≥ 60s 时,先修环境侧)。

---

## 8. 风险登记册

| # | 风险 | 严重度 | 触发信号 | 缓解 |
|---|---|---|---|---|
| **R1** | **训练/评测格式不一致**,训出 harness 专用模型 | 极高 | 训练 loss 正常但评测分数远低于预期 | §4.1 的 9 条;每代记录模板 SHA256 |
| **R2** | **stop token 未对齐**,分数虚高 | 极高 | 分数上升但 `truncated` 比例高 | 逐题存 `finish_reason`;评测显式传 `stop_token_ids` |
| **R3** | **验证器可被攻破** | 极高 | reward 非零率异常高 / mutation 漏网率 > 0 | §3.4 三项自检 + 测试只读 + 哈希校验 |
| **R4** | **零梯度题过多**,算力空转 | 高 | 零梯度题占比 > 40% | 加 K、扩题量;按训练步重算分桶 |
| **R5** | **CPU 瓶颈**,GPU 空转 | 高 | GPU 等待环境占比 > 10% | 快照 fork;SWE-ReX 分离执行层;降 K |
| **R6** | **H2 被证伪**(跨 harness 不掉分) | 中 | 异构 harness 保持率 ≈ 1 | 转为「普通 agentic RL 有效」的定位;强化 RQ1 叙事 |
| **R7** | **model 与 harness 共同适应** | 中 | ④ 的交互项异常 | 训练期冻结 gen00;harness 只在评测扫描 |
| **R8** | **held-out 被污染** | 中 | held-out 结果被用于选点 | 纪律 + 台账记录每次评测的用途 |
| **R9** | **episode 失败率升高**掩盖真实能力 | 中 | 失败率 > 2% | 区分基础设施失败与能力失败(作业分类:failed vs lost) |
| **R10** | **镜像/环境漂移** | 中 | 跨周结果不可比 | digest 固定;每代记 runtime 版本 |
| **R11** | **过拟合到具体题库** | 中 | 训练集分数高、held-out 不升 | 跨任务族评测(TB 2.1)+ 跨 harness 评测 |
| **R12** | **成本失控** | 中 | 每代成本超预算 | 演化用小模型/子集;成本设硬告警 |

> **R9 的分类原则**(引自既有工程经验):**GPU 被平台回收 = lost,配方炸了 = failed,环境不稳 = infra**。三者混在一起会污染所有结论。**控制面坏掉时不能靠 agent 的表现来发现**——环境层要有自己的端到端测试。

---

## 9. 里程碑与验收

| 阶段 | 时长 | 交付物 | 验收标准 |
|---|---|---|---|
| **M0** | W1 | 接线冒烟报告 | §3.2 的 7 项通过 + 5 项负面测试全部符合期望 |
| **M1** | W2 | 噪声底噪 + 难度分布 | δ_min 有数值;零梯度题占比有数值;**据此定 K 与题量** |
| **M2** | W3 | 验证器 | 三项自检全过;mutation 漏网率 = 0 |
| **M3** | W4 | 冷启动 SFT 模型 | reward 非零率 ∈ [20%, 60%];格式错误率 < 10% |
| **M4** | W5–6 | GRPO 训练台账 | ≥ 3 代;每代有完整台账;**至少 1 代在 dev 上显著超过 δ_min** |
| **M5** | W7 | 2×2 主实验 | 四格全部在 heldout 上评测;跨 harness 矩阵完整 |
| **M6** | W8 | 消融 + 报告 | §5.3 的消融至少完成 K / 训练方式 / seed 数 / 模型规模四项 |

**每阶段的 go/no-go**:

| 检查 | 不通过时 |
|---|---|
| M0 的 N1–N3 负面测试 | **不进入 M1**。这三项不通,后面所有数字都不可信 |
| M1 的零梯度题占比 > 40% | 先扩题量或加 K,不要进 M2 |
| M2 的 mutation 漏网率 > 0 | **不进入 M3**。验证器不可信,训什么都是白训 |
| M3 的 reward 非零率 < 20% | 回查验证器与难度,**不加 GPU** |
| M4 连续 2 代不显著 | 停止训练,做失败分析,不进 M5 |

---

## 10. 附录

### 10.1 gen00 harness 快照要点

仅列与训练协议相关的字段(完整定义见 `02-coding-plan.md` §4.5):

| 段 | 关键字段 | 训练期状态 |
|---|---|---|
| `agent` | `agent_class` / `system_template` / `instance_template` / `step_limit` / `cost_limit` / `wall_time_limit_seconds` / `max_consecutive_format_errors` | **冻结** |
| `model` | `model_class=litellm_response` / `model_name` / `model_kwargs` / `observation_template` / `format_error_template` | **冻结**(model_name 指向训练中的策略服务) |
| `environment` | `environment_class` / `image`(digest) / `cwd` / `timeout` | **冻结** |
| `runtime` | `max_iterations` / `context_keep_recent` / `seed` / `temperature` | 温度固定;seed 随实验轮次变化 |

> ⚠️ `context_keep_recent` 在 mini-swe-agent 里**没有对应实现**(它不做上下文压缩)。该字段仅存在于本项目自研的 `HarnessSpec` 中;若落到真实 mini-swe-agent,需由训练侧自行实现裁剪,**并在训练与评测两侧保持同一实现**,否则就是一次隐蔽的 R1 违规。

### 10.2 已知坑清单(来自实战)

| # | 坑 | 症状 | 处置 |
|---|---|---|---|
| T1 | 动作围栏不匹配 | 每步 FormatError | 用宽容正则,仍要求恰好一个块 |
| T2 | 模型未登记价表 | litellm 每次调用抛 RuntimeError | `cost_tracking: ignore_errors`;或 `litellm_model_registry` 登记 |
| T3 | 自造结束标记 | 轨迹跑到 step_limit 仍不终止 | 用原生 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,首行 |
| T4 | 工具不在 PATH | 模型花几步找工具 | 装到 `/usr/local/bin` |
| T5 | **stop token 缺失** | 重复刷屏 + 分数虚高 | checkpoint 的 eos + 评测的 `stop_token_ids` **两边都改** |
| T6 | **vLLM 端口写死** | 服务静默退让一档,训练报 `RemoteDisconnected` | 探测就绪端口 → 回写训练侧;清理要杀**整个候选区间**的监听者 |
| T7 | **SQLite 跨线程** | 作业队列永远停在 `queued` | worker 线程内新建连接,`check_same_thread=False` |
| T8 | **作业失败分类混淆** | 平台回收被记成配方失败 | 区分 `failed` / `lost` / `infra` |
| T9 | **GPU 归属误判** | 抢了别人的卡 | 沿父进程链判归属;`ppid==1` 的引擎 worker 视为孤儿 |
| T10 | **avg@n vs pass@n** | 协议口径错,分数对不上 | 按论文口径算 avg@n;`pass@n` 奖励不稳定,会带偏方向 |
| T11 | **切分/镜像漂移** | 跨周不可比 | 切分与 digest 入库 + SHA256 |

### 10.3 术语

| 术语 | 含义 |
|---|---|
| **harness** | 承载 agent 循环的脚手架;本项目固定为 mini-swe-agent |
| **gen00 / genK** | harness 配置谱系的第 0 代 / 第 K 代(来自 `02-coding-plan.md`) |
| **约束内化** | 策略是否掌握了 mini-swe-agent 的 9 条行为约束,而非是否解出题目 |
| **harness 特化过拟合** | 策略在训练 harness 上涨分、在异构 harness 上不涨 |
| **零梯度题** | 一组 K 条 rollout 全为 reward 0 的题,贡献零梯度 |
| **δ_min** | 在给定任务数与 seed 数下可检测的最小改进幅度 |
| **avg@n / pass@n** | n 次采样的平均正确率 / 至少一次正确的比例;**协议口径必须固定** |

### 10.4 与其他文档的接口

| 来自 | 提供 |
|---|---|
| `01-research-report.md` §3.2 | Athanor 的 dev/held-out 协议范式 |
| `01-research-report.md` §3.2 | RHI 的组件归因结论(哪些字段值得优化) |
| `02-coding-plan.md` §3 | harness 目录结构与 `genNN.yaml` 命名约定 |
| `02-coding-plan.md` §4.1 | `HarnessSpec` / `fields.py` 字段定义 |
| `02-coding-plan.md` §4.6 | 配对 bootstrap / McNemar 的实现口径(训练侧复用同一套统计代码) |
| `02-coding-plan.md` §6.1 | 统计功效计算与多 seed 连续化的依据 |
| `02-coding-plan.md` §7 | 评测管线与「作业排队→执行→日志」的控制面测试 |
