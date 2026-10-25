# Harness 固定、只训练模型:训练计划

## **目标**

在**固定不变的 mini-swe-agent** 下,用 verl 训练一个表现更好的 LLM;并回答这个 harness 的能力上限里有多少能靠后训练挖出来、挖出来的到底是**通用能力**还是**对这个 harness 的特化**。

**三阶段**:

| 阶段 | 名称 | 产出 | 需要 GPU |
|---|---|---|---|
| **P1** | 接线与可行性 | 一条完整轨迹走通 rollout → reward → 梯度;噪声底噪 δ_min;难度分布;**协议选型 A/B** | 否(可纯 CPU + 推理服务) |
| **P2** | 训练 | 冷启动 SFT → GRPO;逐代训练台账 | 是 |
| **P3** | 评测与归因 | 主实验 + 跨 harness 迁移矩阵 + 消融 + 归因分解 | 是(评测) |

**核心方法论约束**(Agentic RL 环境一文第 11 章):

> 无论哪种接法,训练与评测的 **prompt 快照**、**观测截断策略**、**token 级记录方式** 必须完全一致。任何一项不一致,你训出来的都是「某个特定 harness 的专用模型」,换一个脚手架就掉分,而 **loss 曲线全程正常**。

---

## 1. 研究问题

| RQ | 问题 | 对照实验 |
|---|---|---|
| **RQ1** | 在固定 harness 下,后训练能把解决率提升多少?提升来自 SFT 还是 RL? | §7.1 ① vs ②;§7.3 消融「训练方式」 |
| **RQ2** | 提升是**通用能力**还是**harness 特化**? | §7.1 跨 harness 迁移矩阵(保持率 + 行为指纹) |
| **RQ3** | 提升**通过什么机制**发生? | §6.2 约束内化层 8 项过程指标;§7.3 消融「训练集构成」 |

> **RQ2 是本项目的主问题。** 在单轴路线下,「训练有效」是容易得到的结论,「训练出来的东西有没有用」才是难点。

---

## 2. 固定项

### 2.1 harness 常量

**全程只认一份 `harness/frozen.yaml`**。所有字段在下表冻结:

| 项 | 冻结值 | 理由 |
|---|---|---|
| `model_class` | `litellm_textbased`(文本围栏协议) | 与本仓库现有 harness 一致(§2.1.1);且是唯一有同脚手架采集轨迹的公开数据集的协议(§3.4) |
| 动作协议 | 文本围栏 ` ```bash ... ``` `,单动作、单命令 | mini-swe-agent 原生;单 `BASH_TOOL(command: str)` |
| `system_template` / `instance_template` | 冻结,字节级 | 一致性要求 A1 |
| `observation_template` / `format_error_template` | 冻结,带版本号 | 一致性要求 A2 |
| `environment_class` | `swerex_docker`(训练)/ `docker`(单机冒烟) | 执行层与训练机解耦 |
| 容器镜像 | **digest 固定**,不用 tag | 跨周对比必须同环境 |
| `step_limit` | 60 | 轨迹长度上限 |
| `cost_limit` | 2.0 | 单题美元上限 |
| `wall_time_limit_seconds` | 1800 | 单条轨迹墙钟上限 |
| `max_consecutive_format_errors` | 3 | 格式纠错预算 |
| 终止协议 | 首行 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且 returncode 0 | mini 原生约定 |

#### 2.1.1 直接复用本仓库已有资产(强烈建议)

本仓库 `rsi/harbor_agent.py` 已经把 mini-swe-agent 2.4.6 接进 Harbor 0.24.0 并解决了三个真问题,**不要重写**:

| 已有资产 | 位置 | 为什么必须复用 |
|---|---|---|
| `RsiMiniSweAgent` | `rsi/harbor_agent.py` | 已处理 Harbor 的配置注入顺序、`install()`、toolchain 快速路径 |
| **协议保护** | 同上 | Harbor 0.24.0 在 agent option 出现 `reasoning_effort` 且模型名以 `openai/` 开头时,**会注入 `-c model.model_class=litellm_response`**,把本仓库的文本协议顶掉 → **每一步都 FormatError**。症状看起来像「模型变差了」,实为配置问题 |
| **toolchain bundle** | 同上 + `scripts/build_toolchain.sh` | Harbor 默认在**每个** trial 容器里现装 mini-swe-agent(`curl uv` + `uv tool install`),约 4 分钟 / 1 GB 下载**每 trial**。RL 上千 episode 时这是致命的;bundle 把它变成一次性构建 + bind-mount 到 `/root/.local` |
| 冻结切分 | `subsets/*.json`(seed 20260715 + SHA256) | 已按仓库级去重切好,每轮复核 |
| 评测作业模板 | `harbor/job.*.yaml` | `datasets[].task_names` 已冻结,模板跨代字节一致 |

> **注意:训练侧不要用 `rsi.rhi_agent.RhiAgent`。** 那是 harness 进化路线用的多智能体 / contract 驱动 harness,不是本路线。训练侧用 **vanilla `minisweagent.agents.default.DefaultAgent`**。

#### 2.1.2 harness 侧的已知短板(必须测量,不要假装不存在)

公开实测(同一模型 Qwen3.8-27B-FP8、同一数据集 Terminal-Bench 2.1、同一 Harbor):

| harness | 通过率(剔除基础设施错误) | 通过 / 计分 |
|---|---|---|
| terminus-2 | **76.8%** | 53 / 69 |
| **mini-swe-agent** | **51.9%** | 42 / 81 |

差距的一个已定位成因:**mini-swe-agent 每轮重发完整历史**,长任务上撞上下文上限(该次运行 89 题里有 7 题报 `ContextWindowExceededError`)。

> **对本项目的含义**:mini-swe-agent 的分数天花板低于其他 harness。**训练能补上多少,本身就是结果**;但必须在报告里给出这个基线,否则读者会把「训完 55%」误读为「这个模型只有 55% 能力」。
>
> 同时这也是 **H1 的机会**:全历史重发导致的上下文膨胀是**行为问题**,可被训练矫正 —— 约束内化指标里的「上下文占用」直接测它。

### 2.2 模型侧

| 角色 | 选择 | 说明 |
|---|---|---|
| **被训策略(policy)** | 单个 base model,固定 chat template | P1 用小模型(1.7B–4B)验证链路与统计功效;P2 再上目标规模 |
| **对照组(control)** | **同一 base model 的未训练版本** | 唯一合法的对照;不能用别的模型、不能换 checkpoint |
| **评测裁判** | 无 LLM judge,全部可验证自动评测 | 本路线不采用成对打分 |
| **冷启动教师** | 可为大模型;仅用于生成 SFT 数据 | 教师与学生**必须是不同模型**,否则等于蒸馏自己 |

**禁止项**:训练与评测之间任何一次「换个模型版本」「换 chat template」「换 prompt」「换采样参数」都作废该代结果。

### 2.3 沙箱层

| 档 | 后端 | 用途 |
|---|---|---|
| **ENV0** | 本地 `docker` | P1 冒烟、单机调试 |
| **ENV1** | `swerex_docker` / `swerex_modal` | P2 训练(执行层与训练机解耦) |
| **ENV2** | AgentENV(microVM,快照 fork) | P2 大 K 值时(需同题 fork) |

**选择判据**:不是「容器还是 VM」,而是「**我的 rollout 需不需要 fork**」。GRPO 要同题采 K 条 → 需要 fork → 快照机制的价值最高。

---

## 3. 训练集:调研与规范

### 3.1 训练集定稿

**下表是本项目最终要用的训练集。数字均来自数据集卡 / HF API 的公开元数据,已逐条核实;标注「未核实」的除外。**

| # | 数据集 | HF 标识 | 规模 | 仓库数 | 性质 | 许可证 | 轨迹脚手架 | 角色 |
|---|---|---|---|---|---|---|---|---|
| **D1** | SWE-Gym | `SWE-Gym/SWE-Gym` | **2,438** | 11 (Python) | 真实 | MIT | — (仅任务) | GRPO 难样本补充 |
| **D2** | SWE-smith | `SWE-bench/SWE-smith` | **59,136** 行(卡上称 50,137 instances) | 128 | **合成**(注入 bug) | MIT | — (仅任务) | **GRPO 主任务源**(规模最大、自带镜像) |
| **D3** | SWE-rebench | `nebius/SWE-rebench` | `test` **21,336** / `filtered` **6,542** | **3,400+** | 真实 | CC-BY-4.0(逐仓库另计) | — (仅任务) | **GRPO 任务源**(真实、抗污染、预构建镜像) |
| **D4** | **SWE-smith-mini_swe_agent_plus-66k** | `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | **65,994** | 123 | 合成 | MIT | **mini-swe-agent-plus** | **SFT 主数据**(唯一同脚手架采集) |
| **D5** | SWE-smith-trajectories | `SWE-bench/SWE-smith-trajectories` | 三个 split:`tool` 24,100 / `xml` 26,076 / `ticks` 25,826 | 129 | 合成 | MIT | **SWE-agent** | **仅作协议对照**,不作训练源(见 §3.4 方案 B) |

> 另调研过 6 个 OpenHands / SWE-agent 脚手架的轨迹集(SWE-Gym sampled、R2E-Gym SFT、SWE-rebench-openhands、SWE-Hero、Nemotron-SWE-v2、SWE-Play)与 1 个数据分析集(SmolDataEnvs),**均不采用**:前者工具集与 mini 不一致(多工具、有独立编辑工具),逐条重写代价大且极易引入协议污染;后者领域无关。

**各数据集的关键字段差异(决定能不能直接喂进我们的 harness)**:

| 数据集 | 动作表示 | 是否有独立 `tool_calls` 字段 | 终止标记 |
|---|---|---|---|
| **D4** | 文本围栏(mini 格式) | **否**,只有 `messages[{content, role}]` | 见 §3.8.2 实测 |
| D5 `tool` split | tool call | 是(SWE-agent 工具集) | 未核实 |
| D5 `xml` / `ticks` split | XML / tick 格式 | 否 | 未核实 |
| D1 / D2 / D3 | — (只有任务,无轨迹) | — | — |

**两个必须知道的环境事实**:

- **D2 的镜像字段**:SWE-smith 每条实例带 `image_name`;mini-swe-agent 自带的 SWE-bench runner(`minisweagent/run/benchmarks/swebench.py`)的 `DATASET_MAPPING` **原生包含 `smith` → `SWE-bench/SWE-smith`** 与 `rebench` → `nebius/SWE-rebench`,并且 `get_swebench_docker_image_name()` 会优先取实例自带的 `image_name` / `docker_image`,取不到才回落到 `docker.io/swebench/sweb.eval.x86_64.<id>:latest`。**这意味着我们的训练集可以直接用 mini 自己的 runner 跑,不需要另写一套评测适配。**
- **D2 的镜像复用(磁盘关键)**:SWE-smith 的镜像按 **(仓库, commit)** 命名,同一 commit 的不同 bug 变体共享同一个镜像 —— 实测**每个镜像服务 60 – 100 个实例**(见 §9.3)。这是 D2 成为 GRPO 主力任务源的根本原因。
- **D3 的镜像**:SWE-rebench 官方发布了 **7,500 个预构建 Docker 镜像**(Docker Hub 组织 `swerebench`),不必自己 build。但它用的是 SWE-bench 的 fork(基于 Release 4.0.3,改用实例的 `install_config` 字段),**与官方 SWE-bench 的环境安装路径不完全一致** —— 迁移到 `sweb.eval` 镜像时需要验证环境可复现。


### 3.2 选型定稿

```
                    ┌─────────────────────────────────────────────┐
   冷启动 SFT  ───► │ D4 (mini-swe-agent-plus 轨迹, 65,994)        │  ← 同脚手架,协议同源
                    │   + 自采拒绝采样(过滤协议外样本)            │
                    └─────────────────────────────────────────────┘
                                    │
   GRPO 在线 rollout ──► ┌────────────────────────────────────────┐
                    │ D2 (SWE-smith 59,136,合成 bug,自带镜像)      │  ← 任务源,自己采轨迹
                    │ D3 (SWE-rebench 21,336,真实,预构建镜像)      │  ← 任务源,抗污染
                    │ D1 (SWE-Gym 2,438,真实)                     │  ← 难样本补充
                    └────────────────────────────────────────────┘
```

| 角色 | 定稿 | 量级 | 协议风险 | 理由 |
|---|---|---|---|---|
| **SFT 主数据** | **D4** | 1k → 5k → 20k 三档 | **无**(重渲染后 100% 同源) | **唯一用 mini-swe-agent 系脚手架采集的公开轨迹**;字段最简;MIT;实测其 solve rate 随数据量近似线性增长(Qwen3-8B,1k→66k) |
| **SFT 补充** | 自采拒绝采样(base 在 D2/D3 上跑,留成功的) | 3k–10k | **无**(on-policy) | 协议天然同源;可控制难度与长度分布 |
| **GRPO 任务源** | **D2 + D3**(合成 + 真实) | 按 K 与分桶滚动 | **无**(自己采轨迹) | 自己采轨迹,协议无转换风险;D2 规模大、D3 真实且有预构建镜像 |
| **GRPO 难样本** | D1(2,438) | 全量滚动 | **无** | 真实任务补充,规模小但质量高 |
| **协议对照** | D5(三个协议 split) | 各抽 500 条 | 不进训练 | 量化「协议差异」的影响;是 §3.4 方案 B 的唯一现成素材 |

> **磁盘驱动的配比修正(详见 §9.3)**:GRPO 任务源以 **D2** 为主(镜像/实例 = 0.01,磁盘友好);**D3 缩到小规模滚动抽检**,不作为主力 —— 理由不是质量,是 **D3 的镜像数与实例数几乎 1:1**(实测每镜像 2.7 实例,全量约 7,900 个镜像),全量拉取是 TB 量级磁盘。


**D4 的已知风险(必须在 P1 阶段实测,不能想当然)**:

1. `mini-swe-agent-plus` 是 Kwai 的**增强版**,不等于 vanilla mini。它可能含 vanilla mini 没有的动作(如独立文件编辑)。**抽样 200 条统计动作类型分布**,若非 bash 动作占比 > 5%,该数据集不能直接全量使用,必须过滤或转换。
2. D4 是基于 **D2(SWE-smith)** 的 issue 采集的 → **D4 与 D2 共享仓库集合** → D4 与 SWE-bench Verified 的去污染问题与 D2 同源,不能只查一个。
3. D4 的终止标记与 vanilla mini 的原生终止标记是否一致,**未核实**,需抽样确认或定义无歧义转换规则。

### 3.4 动作协议选型:本项目最关键的一个决策

这是**训练集规范里唯一会导致返工的决策**,必须在 P1 阶段用 A/B 实测拍板,不能凭偏好。

| 方案 | 协议 | 可直接消费的公开轨迹 | 风险 |
|---|---|---|---|
| **A** | 文本围栏 `litellm_textbased` | **D4(65,994,mini 同脚手架)** | verl `multi_turn` 对纯文本围栏协议的原生支持形态**未核实**,需在实现前确认 |
| **B** | tool call `litellm_response` | D5 的 `tool` split(但工具集是 SWE-agent 的) | 轨迹与工具集都不匹配,需逐条重写;多工具轨迹与单 `BASH_TOOL` 冲突 |

**默认推荐 A**,理由:唯一存在**同脚手架同协议**的大规模公开轨迹(D4),SFT 阶段的协议污染风险最低;且本仓库现有 harness 已经是 `litellm_textbased`,训练/评测两侧天然同源。

**决策门(M0 的一部分)**:用同一批 200 道题、同一 base model,方案 A 与方案 B 各跑 200 步,比较:

| 指标 | 判据 |
|---|---|
| 动作解析成功率 | A、B 都必须 = 100%;低的直接淘汰 |
| 训练/评测协议同源性 | 训练侧与评测侧必须同协议(跨协议的一律淘汰) |
| 单步往返延迟 | 记录中位数,进 P2 的 CPU 账 |
| 现成数据可直接用比例 | A 应为 ~100%(D4),B 应显著更低 |

**定板后写进 `harness/frozen.yaml` 的 `model_class` 与 `action_regex`,此后不再改**(F1/F2)。

### 3.5 难度分桶与零梯度题

难度分桶**不是数据清洗,是保护优势信号**:

```
pass_rate = 0.0        → 丢弃(组内无正样本,advantage 恒为 0)
pass_rate ∈ (0, 0.9)   → 训练主力
pass_rate ∈ [0.9, 1]   → 丢弃或降权
```

**组内正样本概率**:`P(≥1 正) = 1-(1-p)^K`

| p | K=4 | K=8 | K=16 |
|---|---|---|---|
| 0.05 | 18.5% | 34% | 56% |
| 0.10 | 34% | 57% | 81% |
| 0.20 | 59% | 83% | 97% |

**健康带:reward 非零率 20–60%。** 低于 20% 说明题太难或验证器有问题;高于 60% 说明题太易或**验证器有漏网**(回到 N4)。

**零梯度题占比 < 30%**;超过 40% 时先扩题量或加 K,**不要**直接加 GPU。

### 3.6 去污染协议(必须先做,再做训练)

**污染的三种形式**(都要堵):

| 形式 | 表现 | 堵法 |
|---|---|---|
| **显式污染** | 训练集里直接含有 SWE-bench Verified 的 issue-PR 对 | repo 级黑名单 + instance_id 级精确匹配 |
| **仓库污染** | 训练集含同仓库的**其他** issue(模型见过该仓库的代码风格与历史) | **按 repo 拉黑,不按题** |
| **测试污染** | 模型见过验证用的测试用例 | 测试文件本身不进入训练集 |

**必须拉黑的仓库(SWE-bench Verified 全部 12 个,本地切分已核实)**:

```
astropy · django · matplotlib · mwaskom/seaborn · pallets/flask · psf/requests
pydata/xarray · pylint-dev/pylint · pytest-dev/pytest · scikit-learn · sphinx-doc/sphinx · sympy
```

**已知的高风险项**:

| 数据集 | 风险 | 处置 |
|---|---|---|
| **D1 SWE-Gym** | 11 个 Python 仓库,SWE-Gym 明确「follows SWE-Bench data collection procedure」→ **极可能与 SBV 重叠** | 跑 repo 级交集脚本,重叠部分全部丢弃 |
| **D2 SWE-smith** | 128 个仓库,含大量热门仓库 → **可能命中 SBV 12 仓库** | 同上;**抽样 2,000 条统计命中比例**,> 1% 则按仓库整体剔除 |
| **D3 SWE-rebench** | 3,400+ 仓库,规模大 | 同上;它本身以**去污染**为设计目标,风险最低 |
| **D4** | 源自 D2 | 与 D2 同源,同一个脚本一起查 |

**交付物**:`train_repos_blocked.txt`(拉黑仓库清单)+ `train_repos_clean.txt` + 每次统计的 SHA256,入库。

**反污染的第二个测试**:用 **SWE-bench Verified 的 dev 集(300 题)**在**训练前**跑一次 base model,把 base 的正确题记为「疑似已见题」,在最终报告里单列一栏。如果训练后的模型在这批题上的增益显著高于其他题,就是污染信号。

### 3.7 冷启动 SFT 数据的构造

**冷启动 SFT 不可省**:GRPO 的组内正样本概率要求 base model 有非零的通过率;p=0.10、K=8 时仍有 43% 的题零梯度,p=0.05 时 64%。

**SFT 数据来源(优先级)**:

| 优先级 | 来源 | 量级 | 说明 |
|---|---|---|---|
| 1 | **D4 过滤后** | 1k → 5k → 20k 三档(做规模消融) | 协议同源,零转换成本 |
| 2 | 自采拒绝采样(base 在 D2/D3 上跑) | 3k–10k | 难度与长度可控;教师可以是 base 自己(负样本已被拒) |
| 3 | teacher 蒸馏(更大模型) | 可选 | 成本高,收益未验证;**教师必须 ≠ 学生** |

**SFT 数据的三道门禁**:

| 门禁 | 判据 |
|---|---|
| **协议门** | 动作可被冻结的 `action_regex` 解析,且**恰好一个**动作块 |
| **质量门** | 终止正确率 = 100%,且 `finish_reason` 为 `stop`(非 `length`) |
| **长度门** | 轨迹 token 数 ≤ 上限(否则 SFT 阶段就把长上下文习惯灌进去了) |

> **注意**:SFT 用**成功轨迹**(resolved=1)。用失败轨迹做 SFT 会把错误行为固化;失败轨迹只在 GRPO 里作为负样本存在。

### 3.8 SFT 训练数据管线:标注需求与直用可行性

本节回答一个具体问题:**冷启动 SFT 的训练数据,要不要重新标注?有没有与 mini-swe-agent 兼容、可直接拿来预热的数据?**

**三个结论**:

1. **不需要「重新标注」。** 需要的是一次**确定性重渲染**(纯字符串操作)+ 一个围栏名映射 + 一个工具补齐。**零 LLM 调用、零验证器调用、零容器调用。**
2. **有。** **D4**(`Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`,65,994 条)实测与 mini-swe-agent 2.4.6 的**观测格式逐字节一致**,动作格式只差围栏名。可直接拿来做预热。
3. **但「成功」这件事确实缺标签** —— D4 没有 `resolved` 字段。三个处理选项见 §3.8.3,推荐**选项 2**,因为它**零额外成本**:GRPO 前本来就要统计 base model 的 pass_rate 做难度分桶,那一步顺带就产出 instance 级标签。

**三条路径对比**:

| 路径 | 数据来源 | 协议一致性 | 成功标签 | LLM 调用成本 | 启动速度 |
|---|---|---|---|---|---|
| **A:直用 D4** | 现成 65,994 条 | 重渲染后 100% | ❌ 缺,需补 | **0** | **最快** |
| **B:强模型在 harness 上采** | 任务源 + 强模型跑 harness | **天然 100%** | ✅ 验证器免费给 | 高 | 中 |
| **C:base model 自采** | 任务源 + base model 跑 harness | **天然 100%** | ✅ 验证器免费给 | 推理成本 | 中 |

> **推荐:A 先跑通全链路,B 并行产高质量数据,最后按比例混合。** A 的价值不是「最终数据」,是**用零成本把协议、verl 接口、reward 通路全部打通** —— 这些活儿在 B 路径上一旦出错,排查成本是 A 的十倍。

#### 3.8.1 判定框架:四层兼容性

拿到任何一个候选轨迹集,按这四层逐层判。**任何一层不兼容,要么改(可自动化),要么弃用。**

| 层 | 问什么 | 不兼容的后果 | 可否自动化修复 |
|---|---|---|---|
| **L1 动作表示** | assistant 消息里的动作,能不能被冻结的 `action_regex` 解析? | 每步 FormatError → 轨迹全废 | ✅ 改 regex / 围栏名映射 |
| **L2 观测渲染** | 环境返回的观测,与 `observation_template` 的渲染结果是否一致? | 模型看到的上下文分布与训练时不同 | ⚠️ 需重写观测消息 |
| **L3 prompt 同源** | system / instance 两条消息,与冻结的模板是否同源? | **最隐蔽** —— prompt 变了,模型学到的是另一套指令 | ✅ 重渲染前两条 |
| **L4 终止与工具** | 终止标记一致?额外依赖的工具在容器里存在吗? | 永不终止 / 命令 not found | ✅ 转换 + 装工具 |

> **L2 是最难自动修复的一层。** 观测格式若不一致,就得重写每一条环境消息(几千条 × 每条数 KB),成本与出错率都很高。**L2 一致,L1/L3/L4 就都是便宜的字符串操作。**

#### 3.8.2 路径 A:直用 D4(实测)

**D4 基本信息**:

| 项 | 值 |
|---|---|
| HF 标识 | `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` |
| 规模 | **65,994** 条(train split,单 split) |
| 字段 | **只有两个**:`instance_id`(str)、`messages`(list of `{content, role}`) |
| 采集脚手架 | **mini-swe-agent-plus**(Kwai 版,与 vanilla mini 同族) |
| 采集模型 | claude-3-5-sonnet |
| 任务来源 | SWE-smith 的 issue(`instance_id` 形如 `jd__tenacity.0d40e76f.pr_425`、`Cog-Creators/...`) |
| 许可证 | MIT |
| revision | `750b2c11239fd5e32f97e6cfb9bf80fb9a9a2983`(2025-11-06) |

**实测证据**(抽样 **25 条轨迹 / 576 个 assistant 轮**,逐条核对):

| 维度 | D4 实测 | vanilla mini 2.4.6 | 兼容? | 修复代价 |
|---|---|---|---|---|
| **L2 观测渲染** | `<returncode>N</returncode>` + `<output>`;超长时 `<warning>` + `<output_head>` + `<elided_chars>` + `<output_tail>` | `config/default.yaml` 的 `observation_template` **完全一致** | ✅ | **零** |
| 观测异常样本 | 576 条中 25 条不是观测,而是**重复的 instance prompt**(轨迹重启) | — | ✅ 可识别可丢弃 | 低 |
| **L1 动作围栏** | ` ```bash ` **576/576**(另 3 个 `python` 块是 heredoc 里的代码,非动作) | 默认 `action_regex` 只认 ` ```mswea_bash_command ` | ❌ | **一行 regex** |
| **L3 prompt** | system = "interact multiple times with a computer shell" 版;instance = `<pr_description>` + `<instructions>` | system = "can interact with a computer" 版;instance = `Please solve this issue: {{task}}` + Recommended Workflow | ❌ | **重渲染前两条** |
| **L4 终止标记** | `echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached`(**25/25**) | 首行 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且 returncode 0 | ❌ | 转换 |
| **L4 额外工具** | `edit_via_str_replace` 出现在 **64/576 = 11%** 的 assistant 消息里 | vanilla 容器里**不存在**该 CLI | ⚠️ | 装进镜像 / 过滤 |
| 轨迹长度 | assistant 轮数 min 10 / max 60 / **avg 23** | — | — | — |
| **成功标签** | **无 `resolved` 字段** | — | ⚠️ | 见 §3.8.3 |
| 提交步 | **25/25** 的最后一条 assistant 都是提交命令 | — | — | — |

> **L2 完全一致是这次调研最重要的结论。** 它意味着 D4 的 **assistant 动作消息和全部环境观测消息都可以原样保留**,不需要重写任何一个环境反馈。

**重渲染规格(确定性,无 LLM)**:

| 步骤 | 操作 | 需要 LLM? |
|---|---|---|
| **① 重渲染 system** | 用 `harness/frozen.yaml` 的 `system_template` 替换原 system 消息 | ❌ |
| **② 重渲染 instance** | 用 `instance_template` 渲染 `{{task}}`(从 SWE-smith 的 `problem_statement` 取),替换原 instance 消息 | ❌ |
| **③ 围栏名统一** | assistant 消息里 ` ```bash ` → 冻结协议选定的围栏名 | ❌ |
| **④ 终止标记转换** | `echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached` → 冻结的终止命令 | ❌ |
| **⑤ 轨迹重启剔除** | 丢掉「user 消息是重复的 instance prompt」的那一段 | ❌ |

**产出**:一条与 base model 在同一 harness 下产出的轨迹**逐字节同构**的 SFT 样本。
**成本**:纯字符串处理,66k 条**几十分钟**跑完,零 API 调用、零容器调用。

> **⚠️ 围栏映射的一个陷阱**:D4 的 assistant 消息里除 `bash` 围栏外,还有 3 个 `python` 代码块(heredoc 里的 Python 源码,**不是动作**)。所以 `action_regex` **只能**写成 ```` r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```" ```` 这类**枚举语言**的形式,**绝不能**写成 ```` r"```(\w+)\s*\n(.*?)\n```" ```` —— 后者会把 heredoc 里的 Python 块也当成动作,一条消息解析出 2 个动作 → 每步 FormatError。**「恰好一个动作」这条硬要求在这里第一次真正咬人。**

#### 3.8.3 成功标签的三个选项

D4 没有 `resolved`,而「跑完了提交命令」≠「解题成功」。

| 选项 | 做法 | 成本 | 效果 |
|---|---|---|---|
| **选项 1:不筛,全量用** | 65,994 条直接做 SFT | **0** | 把「哪些真的成功」交给下游 GRPO 的 reward 纠正。这是 RFT(拒绝采样微调)的常规做法,成功样本仍占多数,噪声轨迹被 on-policy RL 洗掉 |
| **选项 2:借 GRPO 的分桶顺带拿到** *(推荐)* | GRPO 前必须先统计 base model 在训练集上的 pass_rate 做难度分桶 → **这一步天然产出 instance 级标签** | **0 额外成本**(本来就要做) | 拿到 D4 与 SWE-smith 任务源按 instance 对齐的成功/失败表,可据此给 D4 每条轨迹打标 |
| **选项 3:抽样外推** | 在「跑完了」的子集里抽 2,000 条重跑验证,估出成功率 r,按 r 加权 | 2,000 次容器执行(约 $200–500) | 得到带置信区间的成功率估计 |

> **选项 2 是最优解**,因为它把「补标签」变成了「本来就要做的一步的副产品」。**前提是 D4 的 instance_id 能对齐到 SWE-smith 的 instance_id** —— 实测 D4 的 `instance_id`(`jd__tenacity.0d40e76f.pr_425`)正是 SWE-smith 的命名格式,join 得上。
>
> ⚠️ **选项 2 的隐含风险**:D4 的 instance_id 有大量**同仓库同 commit 的多个 bug 变体**(`pr_425` / `pr_406` / `lm_rewrite__mc25garo` / `combine_file__otc4vogi`)。分桶时**必须按 instance 而非按仓库聚合**,否则同一仓库的多个变体会把 pass_rate 估歪。

#### 3.8.4 路径 A 的三道门禁(必须先过)

| 门禁 | 判据 |
|---|---|
| **Q1 动作可解析** | 抽样 200 条,用冻结的 `action_regex` 解析 assistant 消息。**必须 100% 解析成功且恰好一个动作**;解析失败率 > 0 = 路径 A 否决 |
| **Q2 工具依赖** | 统计含 `edit_via_str_replace` 的**轨迹级**占比。若 > 20% 的轨迹依赖该工具,**必须**把工具装进训练镜像(不能靠过滤丢数据);若 ≤ 20%,可先过滤这部分轨迹做冷启动,后续再补 |
| **Q3 观测同源** | 抽样 200 条,核对观测消息与 `observation_template` 的渲染结果。已实测 95.7% 一致(其余是轨迹重启),**预期通过**;仍需在真实样本上复核 |

#### 3.8.5 路径 B:强模型在 harness 上采

**接法:只换模型,不换 harness。**

```
harness/frozen.yaml   ← 完全不变(agent / 模板 / 围栏 / 观测渲染)
model.model_name      ← 换成 DeepSeek
```

这样强模型产出的轨迹与 base model 在同一 harness 下产出的轨迹**逐字节同构**,零转换成本;成功标签由验证器给,判定结果同时就是 GRPO 阶段的 reward,两份数据同一个口径。

**成本量级**:

```
SFT 目标规模     3k–10k 条成功轨迹
每条轨迹         ~23 assistant 轮(D4 实测 avg),每轮 ~4–6k tokens(含重发历史)
每条 token 量    ≈ 100k–150k
teacher 成本     按 2M output tokens 估 ≈ $600–1,200(视 DeepSeek 定价)
容器成本         3k–10k 次镜像启动 × ~2–4 min ≈ 10k–40k CPU 分钟
```

**对比路径 A:teacher 成本 = 0,容器成本 = 0。**

**三个必须先验证的坑**:

| 坑 | 现象 | 处置 |
|---|---|---|
| **思维链与围栏协议冲突** | DeepSeek-R1 的 `reasoning_content` 字段:被 `litellm_textbased` 丢掉 → 长思考消失;被保留 → 可能破坏 `THOUGHT:` + 围栏的格式 | **P0 验证**:跑 20 步看解析成功率与推理是否还在;必要时改用无独立 reasoning 字段的模型 |
| **模型不在价表里** | litellm 每次调用抛 RuntimeError,整个 rollout 静默失败 | `cost_tracking: ignore_errors` 或在 `litellm_model_registry` 登记 |
| **轨迹长度膨胀** | 强模型倾向长思考 + 多轮验证,轨迹 token 数远超 base model | SFT 侧加长度门禁(§5.5),并按长度分档采样 |

#### 3.8.6 路径 C:on-policy 自采

**用 base model 自己在 harness 上跑训练集,留成功的做 SFT。**

| 优点 | 缺点 |
|---|---|
| 协议 100% 一致(on-policy) | 质量上限 = base model 自己,**无法超过它** → 冷启动只能到 base 水平,后续 GRPO 的起点也就低 |
| 成功标签免费 | 样本量受限:base 的 pass_rate 决定有多少条成功轨迹;p = 0.10 时,3k 条成功轨迹需要跑 3 万次 |
| 与 GRPO 同分布,无蒸馏偏差 | 算力成本高(每条轨迹都是真实 rollout) |

> **路径 C 通常不是独立选项,而是路径 B 的廉价替代 + GRPO 分桶的副产品。** 若 base model 的 pass_rate 在合理带(10–30%),路径 C 顺手就能拿到几千条 on-policy 成功轨迹;若 pass_rate < 5%,才需要路径 B 补质量。

#### 3.8.7 执行顺序

```
第 1 周   ── 路径 A:重渲染 66k 条 + 过 Q1/Q2/Q3 三道门禁
            ↓
            用 D4 直接跑一次 SFT(1k / 5k 两档)→ 拿到 SFT 模型
            ↓
            ★ 此时协议、verl 接口、reward 通路全部打通,风险基本清零
            ↓
第 1–2 周 ── 并行做路径 B:DeepSeek 接进同一 harness,采 500 条小样本
            ↓
            对比:同一批 instance 上,D4-SFT vs Teacher-SFT 的 dev 分数
            ↓
            若 teacher 显著更好 → 扩量到 3k–10k,做混合配比
            若差不多 → 直接用 D4 全量,省掉 teacher 成本
            ↓
第 2 周   ── GRPO 前的 pass_rate 分桶(选项 2 顺手补齐 D4 的 instance 标签)
```

> **为什么是这个顺序**:路径 A 的边际成本接近零,却能一次性消灭掉「协议不一致」「格式解析失败」「reward 通路不通」这三类**只有在花掉 P2 的钱之后才会暴露**的故障。路径 B 的成本高,但它的产出可以等;**它的正确性必须在 A 打通链路之后才能被廉价验证。**

#### 3.8.8 本节的未核实项与事实来源

**未核实项(P1 阶段实测)**:

| # | 项 | 为什么重要 |
|---|---|---|
| U1 | D4 中含 `edit_via_str_replace` 的**轨迹级**占比(当前只测到消息级 11%) | 决定是「装工具」还是「过滤」(门禁 Q2) |
| U2 | D4 与 SWE-smith 的 instance_id **join 覆盖率** | 决定选项 2 是否可行 |
| U3 | D4 与 SWE-bench Verified 的**仓库交集**实际数字 | D4 源自 SWE-smith,污染风险同源(§3.6) |
| U4 | D4 轨迹中「跑完了但没成功」的比例 | 决定选项 1 的噪声水平是否可接受 |
| U5 | DeepSeek 在 `litellm_textbased` 下的 reasoning 字段行为 | 决定路径 B 是否可行 |

**事实来源**:

| 事实 | 来源 |
|---|---|
| D4 规模 65,994 / 字段仅 `instance_id` + `messages` / MIT / revision | HF API `datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` |
| D4 观测格式 / 围栏名 / 终止标记 / 工具依赖 / 轮数分布 / instance_id 格式 | **本地实测**:HF datasets-server `/rows` 拉 offset=200、length=25,逐条解析 |
| vanilla mini 的 `system_template` / `instance_template` / `observation_template` / `format_error_template` | 本仓库 `.venv312/lib/python3.12/site-packages/minisweagent/config/default.yaml`、`mini_textbased.yaml` |
| 默认 `action_regex` = ```` ```mswea_bash_command\s*\n(.*?)\n``` ```` | `.venv312/.../minisweagent/models/litellm_textbased_model.py:8` |
| 观测模板的 `<warning>/<output_head>/<elided_chars>/<output_tail>` 分支 | `.venv312/.../minisweagent/config/default.yaml` 的 `observation_template` |
| 动作解析实现(`re.findall` + `DOTALL`,**找全部匹配**) | `.venv312/.../minisweagent/models/utils/actions_text.py:24` |

## 4. 测试集(Benchmark):调研与规范

### 4.1 Benchmark 定稿

**Harbor Hub 现有 322 个数据集**(`harbor run -d <org>/<name>` 直接可跑)。下表是本项目最终要用的六个,按角色分工:

| 角色 | # | 数据集 | Harbor 标识 | 题数 | 性质 | 镜像 | 跑频次 | 用途 |
|---|---|---|---|---|---|---|---|---|
| **主指标** | **V1** | SWE-bench Verified | `swe-bench/swe-bench-verified` | **500** | 真实仓库级修复 | `sweb.eval` | 每代 | 核心指标;社区可比 |
| **跨任务族** | **V2** | Terminal-Bench 2.1 | `terminal-bench/terminal-bench-2-1` | **89** | 终端 / 系统 / 环境类任务 | 任务自带 | 每代 | 防止只学会「改 Python 仓库」 |
| **低成本回归** | **V3** | LiveCodeBench | 本地 `livecodebench` v6.0 | 100 | 竞赛编程 | 本地镜像 | 每代 | 便宜,用来高频监控「有没有崩」 |
| **抗污染 held-out** | **V4** | SWE-rebench leaderboard | `swe-rebench/swe-rebench-leaderboard` | **860**(按月切分) | 真实,**抗污染** | **全部预构建** | **仅最终代** | 任务采集于**模型训练截止之后**,污染风险最低 |
| **高难 held-out** | **V5** | SWE-bench Pro | `scale-ai/swe-bench-pro` | **731** | 多文件协同修改 | 有 | **仅最终代** | 更接近真实工程 |
| **长程(可选)** | **V6** | DeepSWE 1.1 | `datacurve/deep-swe-1-1` | 113 | **长程**原创工程任务 | — | 可选 | 长程能力,量少,方差大 |

### 4.2 选型理由与已排除项

**为什么是这六个**:V1 是社区唯一可比的仓库级修复指标;V2 提供与 V1 完全不同形态的任务(终端 / 系统 / 环境),防止模型只学会「改 Python 仓库」;V3 便宜到可以每代都跑,用于第一时间发现「有没有崩」;V4 的任务采集于模型训练截止之后,是抗污染性最强的 held-out;V5 把难度推到多文件协同修改;V6 测长程能力,但样本少、方差大,故列为可选。

**已排除的其余 316 个数据集**,按排除理由归为四类:

- **领域无关**:`tau3-bench`(375,工具型客服对话)、`harveyai/lab`(1,251,合规/法律/金融)、`openthoughts-tblite`(100,表格推理)。
- **角色重复、无增量信息**:`orca-bench`(755)、`apex-agents-1-1`(240)——通用 agent 任务,与 V1/V2 重叠;`aider/aider-polyglot`(225,多语言编辑)与 V3 重复。
- **考的不是终端操作**:`benchflow/skillsbench`(87)考的是 skills 调用,与 harness 能力无关。
- **规模或定位不足**:SWE-Marathon(20)、SWE-Atlas 三件套(124 / 90 / 70,诊断型、非 agentic)、Senior SWE-bench(50)、Harbor Index(80)、TB Pro(200)、TB Science(70)——样本太少或方差大;TB 2.0 / legacy(89 / 66)已被 V2 取代。

**本仓库的 aime / gpqa / arcagi / ineqmath / ifeval / reagym 不在本路线范围内**:它们是 harness 进化路线的**多能力族**评测(数学 / 推理 / 指令遵循)。**本路线只做 agentic SWE 训练**,它们保留给 `02` 路线,本项目不跑。


### 4.3 切分规范(与本仓库现有切分的关系)

**本仓库已有切分**(seed `20260715`,每个文件带 SHA256,每轮复核):

| 切分 | dev | heldout | 状态 |
|---|---|---|---|
| `sbv_dev` / `sbv_heldout` | 300 | 200 | ✅ 无交集,12 个仓库 |
| `tb21_dev` / `tb21_heldout` | **60** | **29** | ⚠️ 见下 |

**⚠️ `tb21` 切分对模型训练路线不可用,原因有二**:

1. **统计功效不足**。heldout 只有 **29 题**。配对比较的判据是 `|b-c| > 1.96·sqrt(b+c)`;29 题上若翻转 4 题,SE ≈ 0.22,需要 2.2pp 的差才勉强显著 —— 这个样本量根本承载不了模型训练常见的 2–5pp 增益。
2. **比例不变量是错的**。`rsi/benchmarks.py` 里 `tb21` 的 `dev_ratio = 300/445`,这个 445 来自「TB 2.1 = 445 题」的旧假设。

**事实更正(已核实)**:

> **Terminal-Bench 2.1 = 89 题**,不是 445。TB 2.0 有 89 题,TB 2.1 修复了其中 **28 题**(外部依赖漂移 9 题、资源预算不足 8 题、指令与测试不一致 11 题),题数仍是 89。
> 本仓库 `harbor/job.tb21.yaml` 指向的正是 `terminal-bench-2-1`,`tb21_dev ∪ tb21_heldout` 恰好 = 89 题,已实测确认。

**本路线的切分规范**:

| 集合 | 内容 | 切法 | 用途 |
|---|---|---|---|
| `dev-sbv` | SWE-bench Verified **全量 500** | **不切分** | 每代评测。500 题单 seed 的配对 SE 已足够小,再切一半是净损失 |
| `dev-tb21` | Terminal-Bench 2.1 **全量 89** | **不切分** | 每代评测。靠 **≥3 seeds(理想 5)** 提统计功效,不靠切分 |
| `heldout-swerebench` | SWE-rebench leaderboard 的**固定月度切分** | 固定到某个 `YYYY_MM`,写死 | 仅最终代。**天然抗污染** |
| `heldout-pro` | SWE-bench Pro | 固定子集,写死 | 仅最终代 |
| `dev-livecode` | LiveCodeBench 全量 100 | 不切分 | 高频低成本回归 |

**切分纪律**:
- **按仓库切,不按题切**(与 `make_real_splits.py` 一致),避免同仓库的题跨集合泄漏。
- 切分文件入库 + SHA256;**任一代改变切分即作废全部历史结果**。
- **held-out 的结果不得进入任何优化决策**(不用于选字段、不用于选 checkpoint 的早停点)。一旦用于决策,该集合即降级为 dev。

### 4.4 各 benchmark 的适配要点与坑

| 数据集 | 坑 | 处置 |
|---|---|---|
| **V1 SWE-bench Verified** | 镜像 `sweb.eval.x86_64.<id>:latest` 共 500 个,磁盘占用巨大 | 预拉 + 分层缓存;或用 `--cache_level=env` 动态清理 |
| | 本仓库 Harbor 适配器有**官方 parity 记录**:mini-swe-agent `2.1.0` + `openai/gpt-5-mini`,官方 0.563±0.000 vs Harbor 0.545±0.007(499 题可比,排除 1 题 Daytona 长期异常) | **我们的分数要对照这个基线解读**,并注明用的是 2.4.6 而非 2.1.0 |
| | 评测环境与 agent 层耦合 | 跑 harbor adapter 的 parity 实验做一次自校准 |
| **V2 Terminal-Bench 2.1** | 任务对**资源/网络/依赖漂移**敏感(2.1 修的 28 题正是这类) | 固定镜像 digest;记录每题的 `reward` 与 `exception_info` |
| | 每题默认 agent timeout 常为 900s,紧 | 长任务用 `--agent-timeout-multiplier` 放大,但**放大值必须全表一致** |
| | mini-swe-agent 在 HF sandbox 下整轮 agent loop 是**一次长 exec**,沙箱 600s 空闲超时会中途杀沙箱 | 显式设 sandbox `job_timeout=2h` |
| | hf-sandbox 用 `/bin/sh`(dash) 执行,Harbor 预置的 `set -o pipefail` 是 bash-only → 全部 trial 秒挂 | 固定 harbor 版本到已修复该问题的版本;**不要**把它记成模型能力问题 |
| **V4 SWE-rebench** | 用 SWE-bench 的 **fork**,按实例 `install_config` 装环境,与 `sweb.eval` 路径不同 | 迁移到 `sweb.eval` 镜像时**逐题验证环境可复现** |
| | 每月更新 → 版本漂移 | 固定到具体月度切分并记录 SHA256 |
| **V5 SWE-bench Pro** | 难度高,base model 可能在 0 附近 → 配对统计退化 | 只作最终代定性报告,不作为主判据 |

### 4.5 评测执行纪律

| # | 纪律 | 说明 |
|---|---|---|
| E1 | **一次评测 = 一份冻结的 job yaml** | `datasets[].task_names` 写死,模板字节一致 |
| E2 | **剔除基础设施错误,不混入能力分** | `exception_info.exception_type` 属 infra 的 trial 单独统计,报告里单列「计分题数 / 总题数」 |
| E3 | **基础设施错误必须重跑到定论** | 重跑时**新起一个 job**,不要用 `job resume` —— 改了模型端点后 resume 的 trial 匹配会失效,会静默重跑全部任务 |
| E4 | **每题记录必存 `finish_reason` 与 `truncated`** | 只存总分就发现不了「stop token 缺失导致分数虚高」 |
| E5 | **不信任 Harbor resume 后的汇总表** | 多轮 resume 会重复计数,最终统计要从每个任务的最新 `result.json` 重新汇总 |
| E6 | **协议字段归训练代所有** | `model_class` / `action_regex` 不被评测作业覆盖(见 §2.1.1 的 Harbor 注入陷阱) |

---

## 5. 训练协议

### 5.1 五阶段流水线

```
Stage 0  接线冒烟   ──► 一条完整轨迹 + 一次梯度 + 协议 A/B 定板
Stage 1  造题分桶   ──► 难度分布 + pass_rate 统计 + 去污染报告
Stage 2  验证器自检 ──► 漏网率 0 / 空策略全 0 / 两次一致
Stage 3  冷启动 SFT ──► reward 非零率进入 20~60%
Stage 4  GRPO       ──► 逐代训练台账
```

**阶段依赖是硬的**:Stage 3 未达到「reward 非零率 ≥ 20%」不得进入 Stage 4;Stage 2 三项自检任一不过不得进入 Stage 3。

### 5.2 Stage 0:接线冒烟

**这是整个项目最重要的一个实验**,因为下游所有坑的症状都是「agent 看起来不工作」。

**通过标准(全部满足才算通过)**:

| # | 检查项 | 判据 |
|---|---|---|
| 1 | 单条完整轨迹 | 从 task 到终止,轨迹非空,终止标记正确 |
| 2 | 动作可解析 | 每步都产出**恰好一个**动作块,解析器不抛 FormatError |
| 3 | 命令真执行 | 至少一条命令在容器里产生了非空 stdout |
| 4 | 观测回流 | `observation_template` 渲染结果进了下一轮 messages |
| 5 | reward 可算 | 验证器在**新沙箱**里给出 0/1,且人工确认一次正确答案得 1、一次错误答案得 0 |
| 6 | 梯度可回传 | 一次 backward 后 policy 参数发生变化(logprob 非 NaN) |
| 7 | 逐题记录落盘 | 含 finish_reason、原文、每步动作、结束原因 |
| 8 | **协议定板** | §3.4 的 A/B 决策完成,结果写入 `harness/frozen.yaml` |

**必须做的负面测试(缺一不可)**:

| # | 负面测试 | 期望行为 | 若不符说明 |
|---|---|---|---|
| N1 | 删掉 `stop_token_ids` | 训练/评测都出现重复刷屏且被截断 | 停 token 未对齐 |
| N2 | 让模型输出自造的结束标记 | 轨迹跑到 step_limit 仍不终止 | 终止协议未对齐 |
| N3 | 故意让模型输出无可解析动作 | 收到 `format_error_template` 的纠错反馈 | 格式纠错回路未通 |
| N4 | 同一 patch 跑两次验证 | 结果一致 | 验证器 flaky |
| N5 | 空策略(不做任何事)跑全量 | reward 全 0 | 存在默认通过分支 |
| N6 | **在评测作业里设 agent option `reasoning_effort`** | Harbor 注入 `litellm_response` → 全员 FormatError | §2.1.1 的协议陷阱未防住 |

> N1–N3 直接对应「模型侧一致性」的三条硬要求。**N6 是本仓库特有的坑**,成本为零,不测等于把整个 P2 押在一个会静默翻转协议的配置上。

### 5.3 Stage 1:造题与难度分桶

见 §3.5(分桶规则)与 §3.6(去污染)。

**P1 阶段还必须产出的东西**:噪声底噪(§6.3)与难度分布 —— **这两项决定 P2 的全部超参**。

### 5.4 Stage 2:验证器与三项自检

**四层门控**:

| 层 | 检查 | 通过标准 |
|---|---|---|
| 1 | 环境自检 | 环境能装起来,测试能跑起来 |
| 2 | **oracle 门** | 施加 gold patch 后 FAIL_TO_PASS 全过、PASS_TO_PASS 不破 |
| 3 | **空策略门** | 不做任何事 → reward 必须为 0 |
| 4 | **mutation 门** | 对 gold patch 做变异 → reward 必须为 0 |

**三项自检**:

| # | 自检 | 判据 |
|---|---|---|
| S1 | mutation 漏网率 | **= 0** |
| S2 | 空策略得分率 | **= 0** |
| S3 | 两次运行一致性 | 同一 patch 两次验证结果一致 |

### 5.5 Stage 3:冷启动 SFT

| 项 | 值 | 理由 |
|---|---|---|
| 数据 | §3.7 的三档规模 | 规模消融 |
| 损失 | 仅在 assistant 动作 token 上 | 不学环境输出 |
| 轮次 | 2–3 epoch | 过多会过拟合到轨迹风格 |
| 通过标准 | reward 非零率 ∈ [20%, 60%];**格式错误率 < 10%** | 格式错误率是 SFT 是否生效的最灵敏指标 |
| 负例 | **不使用失败轨迹** | 会固化错误行为 |

### 5.6 Stage 4:GRPO

| 项 | 值 | 理由 |
|---|---|---|
| 模式 | verl **`remote_agent`**(harness 自带循环) | 用 `tool_agent` 会丢掉 mini 的 system prompt,训出的是「verl 循环下的模型」而非「mini 下的模型」 |
| 组大小 K | 8(消融 4 / 8 / 16) | 见 §3.5 的正样本概率表 |
| reward | 二值(消融「二值 + 步数惩罚」) | 过程奖励易引入刷分 |
| 分桶 | 启用 | 零梯度题的保护 |
| 必设项 | 组内 advantage 标准化;KL 系数;梯度裁剪;**逐 token logprob 落盘** | 可重放 |
| 台账 | 每代一份(§8.1 字段) | 缺台账则结论不可复核 |

### 5.7 停止准则

| 条件 | 动作 |
|---|---|
| reward 非零率连续 2 次评估跌破 20% | 停,回查验证器与难度分布 |
| 格式错误率在训练后反而上升 | 停,回查 stop token / 动作协议一致性 |
| dev 分数连续 2 代不升且低于 δ_min | 停,判定为噪声,做失败分析 |
| 策略在 held-out 上不升 | **不停**,但标记为「harness 特化」候选,进入 RQ2 判定 |

---

## 6. 评测协议

### 6.1 训练/评测一致性清单

**这是本项目最容易翻车的地方。** 两张表合起来是 9 条硬要求。

**A. 通用三项**(Agentic RL 环境一文第 11 章):

| # | 项 | 落地方式 |
|---|---|---|
| A1 | prompt 快照 | `system_template` / `instance_template` 从同一文件读出,记录 SHA256 |
| A2 | 观测截断策略 | `observation_template` / `format_error_template` 版本化,训练与评测同版本号 |
| A3 | token 级记录 | 逐 token logprob 落盘,可重放 |

**B. mini-swe-agent 特有五项**:

| # | 项 | 不一致的症状 |
|---|---|---|
| B1 | **动作协议同源**(训练与评测同一 `model_class` + 同一 `action_regex`) | 训练能解析、评测解析不出 |
| B2 | **终止 token 原生**(首行 + returncode 0) | 永不终止 → step_limit → reward 全 0 |
| B3 | **stop token 双向对齐**:checkpoint 的 `generation_config.eos_token_id` 含 chat 模板 eos **且** 评测 vLLM 显式传 `stop_token_ids` | 输出后不停止,重复刷屏;**判分器取最后一个数字的兜底规则会让分数虚高** |
| B4 | 容器镜像 digest 一致 | 环境漂移被误记为能力提升 |
| B5 | 温度/采样参数一致 | 方差混入 |

> **B3 的隐蔽性**:截断的响应因为「最后一句看起来像答案」而被判对,分数不降反升。**唯一可靠的发现手段是逐题记录里的 `finish_reason` 与 `truncated` 标记。**
>
> **本仓库额外一条**:Harbor 的配置注入顺序会覆盖协议字段(§2.1.1)。这条不在 B 表里,但必须写进 adapter 的不变量。

### 6.2 指标体系:分三组

**第二组是本项目的特色——它度量「约束是否被内化」,比解决率更灵敏。**

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
| **终止正确率** | 用原生终止标记且 returncode 0 结束的轨迹占比 | ↑ |
| **步数耗尽率** | 撞 `step_limit` 结束的轨迹占比 | ↓ |
| **上下文占用** | 终止时的 prompt token 数分布 | ↓(mini 无压缩,全历史重发) |
| **状态持久化误判率** | 因假设 `cd`/`export` 生效而失败的步占比 | ↓ |
| **文件编辑方式分布** | heredoc / `python - <<` / sed / patch 的比例 | 收敛到高效方式 |
| **重复命令率** | 连续两步执行等价命令的比例 | ↓ |

> **为什么这一组重要**:解决率是**结果**,约束内化指标是**过程**。SFT→GRPO 期间解决率可能只动 1–2pp(在噪声边缘),但格式错误率可能从 40% 降到 5% —— **后者才是 H1 的直接证据**,且样本效率高得多。
>
> **「上下文占用」在本项目格外重要**:mini-swe-agent 每轮重发完整历史(§2.1.2),这是它相对 terminus-2 落后 25pp 的一个已定位成因。若训练能把上下文占用显著压下去,那是一个**可归因、可解释、与分数提升同向**的中间变量 —— 换句话说,它是 H3 的最强候选证据。

**C. 训练期(健康监控)**

| 指标 | 健康区间 | 异常含义 |
|---|---|---|
| reward 非零率 | 20–60% | 过低题太难,过高验证器漏 |
| GPU 等待环境占比 | < 10% | 热池不够深 / 调度串行 |
| setup 延迟 P95 | < 60s | 镜像太大 / 磁盘 IO 不足(**toolchain bundle 未生效时是 ~4 分钟**) |
| episode 失败率 | < 2% | 沙箱不稳 / 网络断连 |
| 快照命中率 | > 70% | 没复用状态,每次重装 |
| 组内正样本比例 | 越高越好 | 零梯度题浪费算力 |
| 零梯度题占比 | < 30% | > 40% 则加 K 或扩题量 |

### 6.3 统计方法

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
| 归因 | 按子集分组的 Δscore 差 | 分组效应量 |

> **多 seed 是必需的**:单 seed 下,相邻两代即使有 2–4pp 的真实改进也测不出。**≥3 seeds,理想 5。**
>
> **单轴路线的要求比双轴更严**:双轴路线里 harness 代次可以互相做对照来压噪;单轴路线里**唯一的对照是 base model**,所以 δ_min 估得偏小会直接导致「把噪声当进步」。**噪声底噪实验必须第一个做。**
>
> **协议口径必须固定**:用 avg@n(每题 n 次采样的平均正确率),**不要**用 pass@n(至少一次正确)。pass@n 奖励的是「不稳定」,用它当前沿会把训练引到错方向。

### 6.4 harness 特化度量(RQ2 的核心)

**在同一批 held-out 题上,把训练后的策略放进不同的 harness 跑**:

| harness | 角色 | 说明 |
|---|---|---|
| **mini-swe-agent(冻结)** | 训练 harness | 域内 |
| **terminus-2** | 异构 harness | 有公开对照:同模型同数据集下 76.8% vs mini 51.9% |
| **SWE-agent** | 异构 harness | YAML 配工具集,协议不同 |
| **OpenHands SDK** | 异构 harness | 完整 SDK,工具面不同 |
| **Claude Code / Codex CLI** | 商用 harness | 最强的域外对照 |

#### 6.4.1 迁移保持率

```
保持率 R(h) = Δscore(h) / Δscore(mini)
其中 Δscore(h) = score(trained, h) − score(base, h)
```

| R(h) | 判定 |
|---|---|
| ≥ 0.9 | 几乎无特化,收益基本是通用能力 |
| 0.5 – 0.9 | 部分特化,主收益可迁移 |
| 0.2 – 0.5 | 强特化 |
| < 0.2 | **纯脚手架专用模型** —— H2 被证伪 |

> **必须报告绝对分数与相对提升两列**。不同 harness 的 system prompt 不同,绝对分数不可比;**相对各自 baseline 的提升才是可比的**。

#### 6.4.2 行为指纹(特化的直接证据)

保持率是间接证据。**行为指纹是直接证据**:在 mini 上,「只发单条 bash 命令」「假设 shell 状态不持久」这些行为是**harness 规定的**,模型做对了看不出区别;只有在**异构 harness** 上,这些特化才会暴露。

| 指纹 | 在 mini 上 | 在异构 harness 上的观测 |
|---|---|---|
| **单命令偏置** | 正常 | 异构 harness 有多工具(读文件/写文件/搜索)时,模型仍只发一条 bash 命令的比例 |
| **状态持久化误判** | 正常 | 异构 harness 里 `cd`/`export` 可能持久,模型却仍假设不持久而重复执行的失败率 |
| **自造编辑协议** | 正常 | 模型用 heredoc / `python - <<` 手搓文件编辑,而异构 harness 本来就给了 Edit/Write 工具 —— **该用现成工具而不用**,是特化的直接指纹 |
| **上下文假设** | 正常 | 异构 harness 有上下文压缩时,模型仍表现得像「上下文只增不减」 |

> **这四项必须在异构 harness 上测**,在 mini 上测没有区分度(它们本来就是 mini 的规范)。**指纹命中率与保持率应当同向**;若保持率高但指纹命中率也高,说明「分数没掉」是因为异构 harness 恰好也吃这套行为,不是真的迁移。

---

## 7. 实验矩阵

### 7.1 主实验

| 格 | 策略 | 训练 harness | 评测 harness | 回答 |
|---|---|---|---|---|
| **①** | base model | — | mini(冻结)+ 全部异构 | 基线(含每个异构 harness 的自己的 baseline) |
| **②** | **SFT+GRPO 后的 model** | mini(冻结) | mini(冻结) | **RQ1** |
| **③** | **SFT+GRPO 后的 model** | mini(冻结) | 全部异构 harness | **RQ2** |

**关键读法**:

- ② − ① = **模型侧收益**(RQ1)
- ③ 各列相对 ① 各列的 Δscore 之比 = **迁移保持率**(RQ2)
- ③ 列的行为指纹 = **特化的直接证据**(RQ2)

### 7.2 必做基线

| # | 基线 | 作用 |
|---|---|---|
| B1 | **同一 checkpoint 重复评测 ≥3 seed** | 给出评测侧方差,界定 δ_min |
| B2 | **仅 SFT,不做 GRPO** | 分离 SFT 与 RL 的贡献 |
| B3 | **无 stop_token / 无终止协议**的消融 | 量化一致性项 B3 的影响 |
| B4 | **不同 base model,同配方** | 检验配方是否跨模型通用(而非只对一个 base 有效) |
| B5 | **base model 在异构 harness 上** | 每个异构 harness 自己的 baseline,保持率的必需分母 |

> B1–B3 与 `03` 同名基线的口径一致,可复用统计代码;**B2(随机配置搜索)属于 harness 侧,本路线不做**。

### 7.3 消融

| 维度 | 取值 | 回答 |
|---|---|---|
| **K(组大小)** | 4 / 8 / 16 | K 换 CPU 账单还是换梯度质量? |
| **训练方式** | SFT only / SFT+GRPO / GRPO only | 冷启动是否必需 |
| **reward 粒度** | 二值 / 二值+步数惩罚 | 过程信号是否有用还是引入刷分 |
| **难度分桶** | 分桶 / 不分桶 | 零梯度题的代价 |
| **seed 数** | 1 / 3 / 5 | 统计功效的实际收益曲线 |
| **冷启动数据量** | 1k / 5k / 20k | 数据量-收益曲线(与 D4 卡上报告的近似线性关系对照) |
| **训练集构成** | 全部失败样本 / **仅格式错样本** / **仅补丁错样本** / 混合 | **RQ3:收益从哪来** |
| **合成:真实比** | 100:0 / 50:50 / 0:100 | 合成任务够不够用,真实任务是否必需 |
| **模型规模** | 1.7B / 4B / 8B | 规模是否改变「约束内化」的收益结构 |

> **「训练集构成」这一项是本项目解释力最强的消融**:把 base model 的失败样本按失败原因分成「格式/协议错」与「补丁错」两类,分别只用一类做 SFT,看哪个带来更大的解决率增益。**若「仅格式错」就能带来大部分增益,说明 H3 成立(收益来自约束内化);若必须用「补丁错」样本才有增益,说明收益来自任务知识,H3 被证伪。**

---

## 8. 指标定义

供后续实现直接引用(此处只定义,不实现)。

### 8.1 训练台账字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `gen` | int | 训练代次 |
| `policy_hash` | str | 策略权重哈希 |
| `step` | int | 训练步 |
| `harness_config_sha256` | str | **冻结 harness 配置的哈希**(F1 的执行证据) |
| `harness_version` | str | mini-swe-agent 版本(必须 2.4.6) |
| `protocol` | str | `model_class` + `action_regex` 的指纹(§3.4 定板结果) |
| `reward_nonzero_rate` | float | 健康区间 0.20–0.60 |
| `format_error_rate` | float | 约束内化核心指标 |
| `group_positive_ratio` | float | 组内有正样本的题占比 |
| `zero_grad_task_ratio` | float | 零梯度题占比 |
| `mean_turns` | float | 平均步数 |
| `mean_prompt_tokens` | float | 上下文占用 |
| `cost_usd` | float | 累计成本 |
| `eval_dev_sbv` | float | dev SWE-bench Verified(每代) |
| `eval_dev_tb21` | float | dev Terminal-Bench 2.1(每代) |
| `eval_heldout` | float | held-out(仅最终代) |
| `runtime` | dict | verl 版本 / 镜像 digest / 模型版本 / 数据集 revision |

### 8.2 评测记录字段(逐题)

| 字段 | 说明 |
|---|---|
| `task_id` / `repo` | 题标识与仓库(仓库级去重分析) |
| `seed` | 采样 seed |
| `passed` | bool,隐藏测试结果 |
| `reward` | float,Harbor 的 verifier reward |
| `exception_info.exception_type` | **必存** —— 区分「能力失败」与「基础设施失败」(E2) |
| `finish_reason` | **必存** —— `stop` / `length` / `tool_calls` |
| `truncated` | bool,是否被截断 |
| `n_turns` / `n_tool_calls` | 轨迹长度 |
| `format_errors` | 格式错误次数 |
| `terminated_correctly` | 是否用原生终止协议 |
| `ctx_tokens_final` | 终止时上下文占用 |
| `edits` | 编辑方式计数(heredoc / sed / python / patch / 其他) |
| `harness` | **必存** —— 评测用的 harness 名(跨 harness 矩阵靠它聚合) |
| `cost_usd` / `duration_s` | 成本与时长 |
| `trajectory_path` | 原始轨迹 |
| `retries` | 重试次数 |

> **`finish_reason`、`truncated`、`exception_info`、`harness` 是必存字段**。只存总分就发现不了 B3 那个「分数虚高」的坑,也无法区分 infra 失败。

### 8.3 归因指标

| 指标 | 定义 |
|---|---|
| 约束内化增益 | 格式错误率、步数耗尽率、上下文占用的相对下降 |
| 训练集构成效应 | 「仅格式错 SFT」与「仅补丁错 SFT」的 Δscore 差(RQ3) |
| 跨 harness 保持率 | 异构 harness 提升 ÷ mini-swe-agent 提升 |
| 行为指纹命中率 | 异构 harness 上四项指纹的触发比例 |
| 成本效率 | USD per solved task 的代际变化 |

---

## 9. 预算与算力

### 9.1 CPU 账(先算这个)

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

**toolchain bundle 是这里的杠杆**:Harbor 默认每 trial 现装 mini-swe-agent(约 4 分钟 / 1 GB 下载),RL 上千 episode 时这是纯浪费。本仓库的 bundle 把它降到一次性构建 + bind-mount。**不启用 bundle 的 CPU 账要乘以 1.5–2 倍。**

### 9.2 时间与成本

| 阶段 | 时长 | 成本量级 |
|---|---|---|
| P1 冒烟 + 协议 A/B + 噪声底噪 + 难度分布 + 去污染 | 2–4 天(主要 CPU + 推理) | $500–2,000 |
| P2 Stage 3 冷启动 SFT | 2–4 天 GPU | $3k–8k |
| P2 Stage 4 GRPO(3–5 代) | 3–5 天 GPU | $5k–15k |
| P3 评测 + 跨 harness 迁移矩阵 + 消融 | 2–3 天 | $3k–8k |
| **合计** | **约 3 周** | **$12k–33k** |

**省钱档**:
- 演化用 1.7B 模型,最终只训练 4B
- K=4,dev 集从 500 题降到 250 题
- 消融只在 `dev-tb21` + `dev-livecode` 上跑,不全量
- 跨 harness 矩阵只跑 2 个异构 harness(terminus-2 + SWE-agent)
- → 可压到 **$5k–12k / 2 周**

**锚点参考**(Agentic RL 环境一文,未经独立核实):8×H200、GLM-4.7-Flash、55 rollout、约 3400 episode、64 并发 microVM、episode 失败率约 1%。

---

### 9.3 磁盘账(镜像与容器)

SWE 类任务一题一环境、镜像体积大、训练时并发拉起 —— 磁盘确实是硬约束。但「实例数 × 环境大小」这个直觉算法是**错的**,三个结构性事实把它修正过来。

**事实 1:容器会自动回收,磁盘上留下的是镜像层。**
SWE-bench harness 在每个实例结束都执行 `docker stop` → `docker kill` → `docker rm --force`。磁盘占用的对象是**镜像**,不是「同时活着的环境数」。

**事实 2:镜像数远小于实例数(实测)。**

| 数据集 | 实例数 | 镜像/实例 | 推算镜像数 | 镜像命名 | 单镜像压缩大小 |
|---|---|---|---|---|---|
| **D2 SWE-smith** | 59,136 | **0.010 – 0.016** | **~130 – 600** | `jyangballin/swesmith.x86_64.<repo>_<ver>_<repo>.<commit>` | **1.34 – 1.38 GB**(实测) |
| **D3 SWE-rebench** | 21,336 | **0.37** | **~7,900** | `swerebench/sweb.eval.x86_64.*` | 未实测(SWE-bench 风格) |
| **D1 SWE-Gym** | 2,438 | 11 个仓库 → 约 1:1 | ~11 – 22 | 官方构建 | 未实测 |
| **V1 SWE-bench Verified** | 500 | ~1 | 500 | `sweb.eval.x86_64.*` | Epoch AI 预构建集实测 **≈ 189 GiB** |

> **SWE-smith 的关键结构:镜像按 (仓库, commit) 命名,同一 commit 的不同 bug 变体(`pr_425` / `pr_406` / `lm_rewrite__mc25garo` / `combine_file__otc4vogi`)共享同一个镜像** → 实测**每个镜像服务 60 – 100 个实例**。
>
> **SWE-rebench 恰好相反**:镜像名沿用 SWE-bench 的 `sweb.eval.x86_64.<owner>_<ver>_<repo>-<pr>` 约定,**几乎一个实例一个镜像**(实测每镜像 2.7 个实例),与官方发布的「7,500 个预构建镜像」吻合。

**事实 3:K 条 rollout 共享 1 个镜像。**
GRPO 同题采 K 条 = 1 个镜像 + K 个容器,每个容器只多一个**可写层**(agent 改的文件、pip 安装),量级几十 ~ 几百 MB。**磁盘不随 K 线性增长。**

**事实 4:相同层只存一份。**
本仓库实测(`docker system df`):71 个镜像,`docker images` 显示的虚拟大小和约 57 GB,而实际占用 **22.43 GB**。**镜像数不等于磁盘** —— 这是最容易被高估的一环。

**正确的磁盘公式**:

```
磁盘 ≈ 并发在飞的不同镜像数 × 单镜像唯一增量
     + 共享的 base / env 层(固定)
     + 容器可写层 × 容器数
```

「并发在飞的不同镜像数」≤ 并发 episode 数;通过 image-major 调度可压到 ≈ 镜像池大小。

**SWE-bench 官方的四档 cache_level(直接移植到训练侧)**:

| cache_level | 保留什么 | 磁盘 | 速度 | 本项目用法 |
|---|---|---|---|---|
| `none` / `base` | 仅 base | 运行期仍需 **~120 GB** | 最慢 | **训练侧用这个**:instance 镜像用完即删 |
| `env`(默认) | base + 约 60 个 env | **~100 GB** | 中 | **评测侧用这个** |
| `instance` | 全部 | **~2,000 GB** | 最快 | **不可用于训练** |

官方基线:至少 **120 GB** 空闲磁盘、16 GB RAM、8 核;worker 数取 `min(0.75 × cpu, 24)`。

**六条落地措施**:

| # | 措施 | 做法 | 效果 |
|---|---|---|---|
| **S1** | **image-major 调度** | 训练集按 `image_name` 分组,同一镜像的所有实例连续处理 | 同时在飞的镜像数 ≤ 并发数,而非 ≤ 实例数 |
| **S2** | **K 条 rollout 走同一镜像** | 同题 K 条起 K 个容器共享镜像层;能 fork 就用快照 fork | 磁盘不随 K 增长;fork 把分钟级压到几十毫秒 |
| **S3** | **训练侧按 `cache_level=base` 语义** | instance 镜像用完即删,只留 env 层 | 训练期磁盘从 TB 量级降到百 GB 量级 |
| **S4** | **只 pull 不 build** | SWE-rebench 官方 7,500 个镜像在 Docker Hub `swerebench`;SWE-smith 镜像在 `jyangballin/*`;SWE-bench 用 Epoch AI 预构建集 | 省掉 build 的磁盘与网络双杀;本仓库 `scripts/warm_task_images.py` 的预热思路可直接复用 |
| **S5** | **LRU 驱逐** | 只保留 N 个热镜像(建议 N = 并发数 × 1.5),超出即 `docker image rm` | 磁盘水位恒定 |
| **S6** | **分片到多机** | 训练按 instance 分片,每机只持自己那部分镜像;评测天然可分片 | 磁盘需求线性摊到 N 台机 |

**⚠️ 两个必须处理的运维风险**:

1. **SWE-smith 的镜像挂在个人 Docker Hub 账号 `jyangballin/` 下**,不是官方 org。账号被删或改名 → 训练集镜像全部失效。**必须在 P1 阶段镜像到自己的 registry**(如 `registry.myorg.com/swesmith/*`)。
2. **镜像必须记 digest 不能只记 tag**:`image_name` 用的是 tag,tag 可被重新指向到别的镜像。跨周对比只认 digest。

**P1 阶段必须实测的三项**:

| # | 项 | 怎么测 |
|---|---|---|
| D1 | SWE-smith 全量去重镜像数 | 读 `SWE-bench/SWE-smith` 的 `image_name` 列**全量**去重(目前为跨分片抽样,比例 0.010 – 0.016) |
| D2 | SWE-smith 单镜像解压后实际占用 | `docker pull jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536 && docker images` |
| D3 | SWE-rebench 镜像可达性与大小 | 抽查 20 个 `swerebench/*` 的 `docker pull` + 大小 |

---

## 10. 风险登记册

| # | 风险 | 严重度 | 触发信号 | 缓解 |
|---|---|---|---|---|
| **R1** | **训练/评测协议不一致**,训出 harness 专用模型 | 极高 | 训练 loss 正常但评测分数远低于预期 | §6.1 的 9 条;每代记录模板 SHA256;§3.4 的 A/B 定板 |
| **R2** | **stop token 未对齐**,分数虚高 | 极高 | 分数上升但 `truncated` 比例高 | 逐题存 `finish_reason`;评测显式传 `stop_token_ids` |
| **R3** | **验证器可被攻破** | 极高 | reward 非零率异常高 / mutation 漏网率 > 0 | §5.4 三项自检 + 测试只读 + 哈希校验 |
| **R4** | **零梯度题过多**,算力空转 | 高 | 零梯度题占比 > 40% | 加 K、扩题量;按训练步重算分桶 |
| **R5** | **CPU 瓶颈**,GPU 空转 | 高 | GPU 等待环境占比 > 10% | 快照 fork;执行层分离;降 K;**启用 toolchain bundle** |
| **R6** | **H2 被证伪**(跨 harness 几乎不掉分) | 中 | 保持率 ≈ 1 | 转为「普通 agentic RL 有效」的定位;强化 RQ1/RQ3 叙事 |
| **R7** | **harness 自身的短板成为天花板** | 中 | 训练后仍显著低于 terminus-2 | 显式报出 mini 基线;把「上下文占用」作为可矫正的中间变量 |
| **R8** | **held-out 被污染** | 中 | held-out 结果被用于选点 | 纪律 + 台账记录每次评测的用途;用 SWE-rebench 月度切分做抗污染复核 |
| **R9** | **episode 失败率升高**掩盖真实能力 | 中 | 失败率 > 2% | 区分基础设施失败与能力失败(作业分类:failed / lost / infra) |
| **R10** | **镜像/环境漂移** | 中 | 跨周结果不可比 | digest 固定;每代记 runtime 版本 |
| **R11** | **过拟合到具体题库** | 中 | 训练集分数高、held-out 不升 | 跨任务族评测(TB 2.1)+ 抗污染 held-out(SWE-rebench 月切)+ 跨 harness 评测 |
| **R12** | **成本失控** | 中 | 每代成本超预算 | 演化用小模型/子集;成本设硬告警 |
| **R13** | **训练集协议不兼容**(D4 含非 bash 动作 / 终止标记不符) | 高 | SFT 后格式错误率不降反升 | §3.8.4 的 Q1/Q2 抽样门禁;**P1 阶段先验后用** |
| **R14** | **训练集与评测集仓库重叠** | 高 | 训练后 SBV 增益异常高 | §3.6 的 repo 级去污染 + 「疑似已见题」对照 |
| **R15** | **Harbor 注入协议字段** | 高 | 每步 FormatError,看似模型问题 | §2.1.1 的 adapter 不变量;N6 负面测试 |
| **R16** | **磁盘打满**,训练中断 | 高 | 拉镜像报 `no space left on device`;`docker system df` 可回收量持续下降 | §9.3 的 S1/S3/S5;台账每代记录 `docker system df` 快照;镜像 digest 固定 |
| **R17** | **训练集镜像源失效** | 中 | `jyangballin/*` 拉取失败或 digest 漂移 | P1 阶段镜像到自有 registry 并记 digest(§9.3) |

> **R9 的分类原则**:**GPU 被平台回收 = lost,配方炸了 = failed,环境不稳 = infra**。三者混在一起会污染所有结论。**控制面坏掉时不能靠 agent 的表现来发现** —— 环境层要有自己的端到端测试。

---

## 11. 里程碑与验收

| 阶段 | 时长 | 交付物 | 验收标准 |
|---|---|---|---|
| **M0** | W1 | 接线冒烟报告 | §5.2 的 8 项通过 + 6 项负面测试全部符合期望;**协议 A/B 定板并写入 frozen.yaml** |
| **M1** | W1–2 | 训练集规范报告 | N1–N7 七条门禁逐条有结论;**去污染报告**(拉黑清单 + 命中率数字);**磁盘基线**(D2 全量去重镜像数 + 单镜像实测大小 + 自有 registry 镜像完成,见 §9.3) |
| **M1.5** | W2 | **特化预判** | 用便宜模型跑完整链路 + 一次跨 harness 评测,给出保持率的**初步数值**。**若初判 < 0.2,回到项目定位讨论,不要进 P2** |
| **M2** | W2 | 噪声底噪 + 难度分布 | δ_min 有数值;零梯度题占比有数值;**据此定 K 与题量** |
| **M3** | W3 | 验证器 | 三项自检全过;mutation 漏网率 = 0 |
| **M4** | W4 | 冷启动 SFT 模型 | reward 非零率 ∈ [20%, 60%];格式错误率 < 10% |
| **M5** | W5–6 | GRPO 训练台账 | ≥ 3 代;每代有完整台账;**至少 1 代在 dev 上显著超过 δ_min** |
| **M6** | W7 | 主实验 + 迁移矩阵 | ① ② ③ 三格全部评测;跨 harness 矩阵完整;保持率与行为指纹都有数值 |
| **M7** | W8 | 消融 + 报告 | §7.3 至少完成 K / 训练方式 / seed 数 / **训练集构成** 四项 |

**每阶段的 go/no-go**:

| 检查 | 不通过时 |
|---|---|
| M0 的 N1–N3、N6 负面测试 | **不进入 M1**。这几项不通,后面所有数字都不可信 |
| M1 的 Q1 门禁(动作解析失败率 > 0,见 §3.8.4) | 换数据集或做转换过滤,**不要**先训练再说 |
| M1 的去污染命中率 > 1% | 按仓库整体剔除后重算难度分布 |
| **M1.5 的保持率初判 < 0.2** | **不进 P2**。项目叙事需要重写(这是最贵的错误) |
| M2 的零梯度题占比 > 40% | 先扩题量或加 K,不要进 M3 |
| M3 的 mutation 漏网率 > 0 | **不进入 M4**。验证器不可信,训什么都是白训 |
| M4 的 reward 非零率 < 20% | 回查验证器与难度,**不加 GPU** |
| M5 连续 2 代不显著 | 停止训练,做失败分析,不进 M6 |

> **M1.5 是本路线最关键的一道闸**。双轴路线里,即使做出一个特化模型,2×2 的第三、四格还能救回叙事;单轴路线里**没有第二根柱子** —— 特化就是特化。所以在花掉 P2 的钱之前,必须用小模型先看一眼。