# Harness 自进化(RSI)实现计划

> 版本:1.0 · 日期:2026-10-20 · 上游文档:[`01-research-report.md`](./01-research-report.md)

---

## 0. 目标与范围

### 0.1 一句话目标

用**结构化配置文件的自动搜索**,迭代优化一个 agent harness 的设计,使其在目标 benchmark 上持续提升;每一代的改进由**相邻两代的自动评测绝对分数差**驱动。

### 0.2 范围内

- harness schema(typed config)定义与校验
- 基于 mini-swe-agent 的 `RhiAgent` 实现(支持多 agent hop / contract)
- Harbor 集成:一代 harness = 一份 config
- 双 benchmark 评测管线(SWE-bench Verified / Terminal-Bench 2.1),dev/held-out 切分,多 seed
- 统计模块:噪声底噪、配对 bootstrap、逐字段效应量
- typed config 空间的 optimizer(随机搜索基线 → bandit → CEM)
- 逐代台账:配置快照、SHA256、轨迹归档、跨代 diff

### 0.3 范围外(本期不做)

- 任何形式的 harness **源码**生成或自动重构(只优化配置)
- 端到端自动闭环的「无人值守跑 N 代」模式(先半自动,人工审阅每代 diff)
- 跨 harness 的对比实验(基建就绪即可,实验本身作为下一阶段)

### 0.4 关键设计约束(来自调研,不可违反)

| # | 约束 | 原因 |
|---|---|---|
| C-1 | 评测期间 **job.yaml 跨代完全不变**,只有 `agents[0].kwargs.config` 变 | 否则分数差无法归因于 harness 配置 |
| C-2 | **pin 版本**:mini-swe-agent `==2.4.6`、harbor 固定 tag、Docker 镜像 digest 固定 | runtime 漂移会污染代际差 |
| C-3 | **seed 可控 + common random numbers** | 分离配置效应与采样噪声 |
| C-4 | gen-0 **全字段显式钉死**(不继承 Harbor 默认值) | Harbor 的 `mini` config 会被 layered,继承即引入隐藏变量 |
| C-5 | 单代只改 **1–2 个字段** | 归因需要 |
| C-6 | 每代配置 + 轨迹 **归档 + SHA256** | 可回溯、可复现 |

---

## 1. 技术选型(定稿)

| 层 | 选择 | 理由摘要 |
|---|---|---|
| 执行内核 | **mini-swe-agent `==2.4.6`** | YAML 分层合并;`agent_class` 支持 import path;版本稳定;成本最低 |
| Agent 实现 | 自研 `RhiAgent(minisweagent.agents.default.DefaultAgent)` | 需要多 agent hop / contract,而内核不提供 |
| 适配层 | **Harbor**(pin 版本) | mini-swe-agent 已是预集成 agent;job yaml 天然承载代际 config |
| 主迭代 benchmark | **Terminal-Bench 2.1**(445 题) | 任务量适中、agent-native、终端语义与 mini-swe-agent 的 bash-action 模型一致 |
| 最终验证 benchmark | **SWE-bench Verified**(500 题) | 评测层与 agent 层解耦,scaffold 泄漏更少;跨 benchmark 泛化证据 |
| 模型 | 固定单一模型 + 固定 reasoning effort | 变量控制;不同模型作为后续消融维度 |

**参考实现在 Harbor 中的挂载点**(`examples/configs/agents/mini-swe-agent-job.yaml`):

```yaml
agents:
  - name: mini-swe-agent
    model_name: anthropic/claude-sonnet-4-6
    # Inline mini-swe-agent config overrides are layered over its `mini` config.
    # CLI equivalent: --ak 'config={"agent":{"step_limit":20}}'
    kwargs:
      config:
        agent: {agent_class: mypkg.rhi_agent.RhiAgent}
```

`kwargs.config` 会 layered 到 mini 的 config 上——这正是「一代 harness = 一份 config」的机制。

---

## 2. 总体架构

```
                    ┌──────────────────────────────┐
                    │      optimizer (typed)        │
                    │  bandit / CEM / random search │
                    └───────────┬──────────────────┘
                                │ 写入 gen_N.yaml
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │  harness spec (YAML)                                   │
   │   agents[] / workflow[] / auxiliary_rules / runtime    │
   └───────────┬────────────────────────────────────────────┘
               │ 映射(adapter)
               ▼
   ┌────────────────────────────────────────────────────────┐
   │  RhiAgent  (mini-swe-agent DefaultAgent 子类)           │
   │   + 多 agent hop 执行器 + contract 渲染器               │
   └───────────┬────────────────────────────────────────────┘
               │  config 注入
               ▼
   ┌────────────────────────────────────────────────────────┐
   │  Harbor  (pinned)  →  Docker 环境 → agent → 产物        │
   └───────────┬────────────────────────────────────────────┘
               │ 逐题结果
               ▼
   ┌────────────────────────────────────────────────────────┐
   │  evaluator (TB test script / swebench)  → 逐题 0/1     │
   └───────────┬────────────────────────────────────────────┘
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │  analysis: 配对 bootstrap / 效应量 / 台账 diff          │
   └───────────┬────────────────────────────────────────────┘
               │  相邻代差分 → 接受/拒绝
               └──────────────► 回到 optimizer
```

**关键点**:agent loop 内部(hop 调度、contract 组装、上下文裁剪)全部由 `RhiAgent` 负责,而**它的行为由 spec 决定**;spec 是 optimizer 的唯一操作对象。

---

## 3. 目录结构

```
rsi/
├── pyproject.toml
├── requirements.lock          # pip freeze,含 mini-swe-agent==2.4.6
├── rsi/
│   ├── __init__.py
│   ├── schema.py              # typed spec: dataclass + 校验 + (de)serialize
│   ├── fields.py              # 字段定义: 名称/类型/取值域/默认值/是否可优化
│   ├── rhi_agent.py           # RhiAgent(DefaultAgent)
│   ├── hop_runner.py          # 多 agent hop 调度 + contract 渲染
│   ├── adapter.py             # spec → mini-swe-agent config 的映射
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── base.py            # Optimizer 抽象: propose(spec, history) -> spec
│   │   ├── random_search.py   # 基线
│   │   ├── bandit.py          # 单字段 bandit (UCB / Thompson)
│   │   └── cem.py             # 连续字段的 CMA-ES-lite
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── runner.py          # Harbor 调用 + 结果归一化
│   │   ├── subsets.py         # dev/held-out 切分(固定、带种子)
│   │   ├── stats.py           # 配对 bootstrap、CI、效应量
│   │   └── noise_floor.py     # 同配置多 seed 的噪声底噪估计
│   ├── ledger/
│   │   ├── __init__.py
│   │   ├── ledger.py          # 每代台账: 配置快照/SHA256/分数/元数据
│   │   └── diff.py            # 跨代配置 diff + 效应量归因表
│   └── cli.py                 # rsi init / rsi gen / rsi eval / rsi analyze
├── harness/
│   ├── gen00.yaml             # 基线,全字段显式
│   ├── gen01.yaml
│   └── ...
├── harbor/
│   ├── job.yaml               # 跨代不变(C-1)
│   ├── job.tb.yaml
│   └── job.swebench.yaml
├── subsets/
│   ├── tb21_dev.json
│   ├── tb21_heldout.json
│   ├── sbv_dev.json
│   └── sbv_heldout.json
├── runs/
│   └── genNN/                 # 原始评测产物 + 轨迹
└── ledger/
    ├── genNN.yaml             # 台账条目
    └── report.md              # 自动生成的跨代报告
```

---

## 4. 模块设计

### 4.1 `schema.py` — typed harness spec

**目标**:spec 是唯一被 optimizer 操作的对象;必须 typed、有界、可 diff、可校验。

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

class ContractFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"

@dataclass(frozen=True)
class Contract:
    """RHI 的 contract:subagent → orchestrator 之间传什么。"""
    out_fields: tuple[str, ...]     # 有序;顺序即在 prompt 中的呈现顺序
    in_fields: tuple[str, ...]
    format: ContractFormat = ContractFormat.JSON
    max_tokens: int = 800
    # 允许 subagent 自由文本的额外空间(相对 out_fields 的软约束)
    free_text: bool = True

@dataclass(frozen=True)
class AgentSpec:
    id: str
    role: str
    instruction: str
    contract: Contract
    model: str                    # 允许每个 agent 用不同模型(破除同质化的旋钮)
    reasoning_effort: str = "high"
    tools: tuple[str, ...] = ("read_file", "grep_file", "run_bash")
    max_tokens: int = 8192

@dataclass(frozen=True)
class Hop:
    id: str
    source: str                   # agent id
    target: str                   # agent id
    pass_fields: tuple[str, ...]  # 本跳传递的字段(contract 的具体化)

@dataclass(frozen=True)
class AuxiliaryRules:
    acceptance_gates: tuple[str, ...] = ()
    failure_fallback: tuple[str, ...] = ()
    recall_triggers: tuple[str, ...] = ()

@dataclass(frozen=True)
class Runtime:
    max_iterations: int = 40
    observation_max_chars: int = 10000
    context_keep_recent: int = 20
    seed: int = 1234
    temperature: float = 0.0

@dataclass(frozen=True)
class HarnessSpec:
    version: int
    agents: tuple[AgentSpec, ...]
    workflow: tuple[Hop, ...]
    auxiliary_rules: AuxiliaryRules
    runtime: Runtime
```

**校验规则**(构造时 fail-fast):

1. `agents[].id` 唯一
2. `workflow` 构成有向图,必须存在 orchestrator(唯一入度为 0 的节点)与至少一次回到 orchestrator 的边
3. 每个 hop 的 `pass_fields` 必须是 `source.contract.out_fields ∪ source.contract.in_fields` 的子集
4. 第一个 agent 的 `in_fields` 必须包含 `task`
5. `runtime.max_iterations` 与各 agent 的 `max_tokens` 有上界(避免搜索跑到无效区域)

### 4.2 `fields.py` — 可优化字段登记表

**这是「哪些东西值得优化」的唯一事实来源**,也是归因表的骨架。

```python
@dataclass(frozen=True)
class FieldSpec:
    path: str                       # 点分路径,如 "agents[0].contract.max_tokens"
    kind: Literal["int", "float", "str", "bool", "enum", "list"]
    domain: tuple | None            # 离散域;连续量给 (lo, hi)
    mutable: bool = True            # False = 固定,不允许 optimizer 改
    group: str = ""                 # 归因分组: contract / hop / instruction / role / runtime
    rationale: str = ""
```

**按 RHI §6.2.2 的结论排优先级**:

| 优先级 | group | 可优化字段 | 备注 |
|---|---|---|---|
| P0 | `contract` | `out_fields` / `in_fields` / `max_tokens` / `format` / `free_text` | RHI 实证:contract 变化最大、聚类最清晰 |
| P0 | `hop` | `workflow` 的边集合与 `pass_fields` | RHI:hops 也有任务依赖聚类,仅次于 contract |
| P1 | `runtime` | `max_iterations` / `observation_max_chars` / `context_keep_recent` | 影响成本与上下文,易测 |
| P2 | `instruction` | 指令文本(由模板 + 参数拼装,**不直接自由文本**)| 自由文本难以归因,改用**结构化槽位** |
| P3 | `role` | 角色标签 | RHI:roles 最不可分,几乎不动 |
| P3 | `model` | 各 agent 的模型 / reasoning_effort | 作为**破除同质化**的显式旋钮,默认关闭 |

**重要设计**:`instruction` 不开放自由文本。改为**槽位化**——

```yaml
instruction_template: |
  你是 {role}。
  目标: {goal}
  产出必须包含: {required_outputs}
  不得包含: {forbidden_outputs}
```

槽位(如 `required_outputs` / `forbidden_outputs` / `goal` 措辞变体)才是被优化的对象。这样每次改动语义边界清晰,能映射到 RHI 的 `contract` 概念,而不是无约束的散文改写。

### 4.3 `rhi_agent.py` — RhiAgent

**职责**:把 `HarnessSpec` 变成可执行的 agent 行为。

```python
from minisweagent.agents.default import AgentConfig, DefaultAgent

class RhiConfig(AgentConfig):
    spec_path: Path                       # HarnessSpec 的路径
    max_hops: int = 6
    hop_context_max_chars: int = 6000

class RhiAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class=RhiConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.spec: HarnessSpec = load_spec(config_class_spec_path)
        self.hop_runner = HopRunner(self.spec)

    def step(self) -> list[dict]:
        """被 hop_runner 驱动:每一 hop 是一次 LLM 调用 + 一组工具动作。"""
        ...
```

**关键实现点**:

1. **不使用 `DefaultAgent` 的 `system_template` / `instance_template` 做主 prompt**。改为由 `hop_runner` 按 spec 动态渲染每跳的 system message:
   - orchestrator 的 system = 角色 + 全部 hop 的 workflow 描述 + 当前的 contract 说明 + 可用 agent 名单
   - subagent 的 system = 角色 + **仅本跳的入向 contract 字段** + 工具说明
   - **这一步直接实现 RHI 的核心假设**:subagent 条件于「本跳该看的信息」,而非全部历史
2. **上下文裁剪**:orchestrator 收到的是各 subagent 按 `contract.out_fields` 过滤后的结构化结果,不是原始轨迹。裁剪上限由 `runtime.context_keep_recent` 与 `hop_context_max_chars` 控制
3. **失败回退**:某 hop 抛错时,按 `auxiliary_rules.failure_fallback` 决定重试 / 降级 / 中止
4. **提交信号**:沿用 mini-swe-agent 原生约定——输出首行 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且 returncode 0 → `Submitted`。**不要自造结束标记**,`LocalEnvironment._check_finished` 只认这个
5. **seed 透传**:`model_kwargs` 里带 `seed`,由 adapter 从 `runtime.seed` 注入

### 4.4 `hop_runner.py` — hop 调度与 contract 渲染

```python
class HopRunner:
    def __init__(self, spec: HarnessSpec): ...

    def next_hop(self, state: RunState) -> Hop | None:
        """按 workflow 拓扑选下一跳;None 表示回到 orchestrator 或终止。"""

    def render_system(self, hop: Hop, state: RunState) -> str:
        """渲染本跳的 system message:角色 + 本跳入向字段说明 + 工具。"""

    def extract_payload(self, hop: Hop, raw: str) -> dict:
        """从 subagent 输出里按 contract.out_fields 抽取结构化 payload。
        format=json → 解析 JSON;format=markdown → 按字段名正则抽取。"""

    def assemble_context(self, payloads: list[dict]) -> str:
        """按 contract.in_fields 组装回 orchestrator 的上下文。"""
```

**`extract_payload` 要处理解析失败**:字段缺失或格式错 → 记为该 hop 的失败,按 `failure_fallback` 处理,而不是让整个 run 崩掉。**这一条要在第一周就写好**,否则早期实验会因为 5% 的解析失败浪费大量算力。

### 4.5 `adapter.py` — spec → mini-swe-agent config

职责单一:把 `HarnessSpec` 翻译成 mini-swe-agent 的 config dict,并**把未被 spec 覆盖的字段全部显式写死**(C-4)。

```python
def to_mini_config(spec: HarnessSpec, base: dict) -> dict:
    """base 是 gen-0 冻结的基础 config(环境、模型、限制等)。
    这里只允许 spec 管辖的字段变化;其余字段从 base 原样带出,
    并对 base 做一次「全字段展开」以避免继承 Harbor 默认值。"""
```

产出大致形如:

```yaml
agent:
  agent_class: rsi.rhi_agent.RhiAgent
  system_template: "__RHI_RENDERED__"     # 由 RhiAgent 接管,占位防误用
  instance_template: "__RHI_RENDERED__"
  step_limit: 0
  cost_limit: 3.0
  wall_time_limit_seconds: 3600
  max_consecutive_format_errors: 3
model:
  model_name: anthropic/claude-sonnet-4-6
  model_kwargs: {temperature: 0.0, seed: 1234}
  action_regex: '```(?:mswea_bash_command|bash|sh|shell)?\s*\n(.*?)\n```'
  observation_template: <gen-0 冻结版>
  format_error_template: <gen-0 冻结版>
environment:
  environment_class: docker
  timeout: 120
```

**关于 `action_regex`**:mini-swe-agent 默认要求 ` ```mswea_bash_command ` 围栏,非 Claude 模型通常输出 ` ```bash ` → 每步 FormatError。**采用宽容正则但仍要求恰好一个块**(多块解析失败是正确的)。

### 4.6 `evaluation/` — 评测与统计

#### 4.6.1 切分(`subsets.py`)

- 固定、带种子的划分,**跨代不变**:TB 2.1 的 445 题 → dev 300 / heldout 145;SWE-bench Verified 的 500 题 → dev 300 / heldout 200
- 划分文件入库并带 SHA256;**任一代际改变划分即视为实验作废**

#### 4.6.2 运行器(`runner.py`)

```python
@dataclass
class TaskResult:
    task_id: str
    seed: int
    passed: bool
    duration_s: float
    n_llm_calls: int
    cost_usd: float
    error: str | None
    trajectory_path: Path | None
```

- 调用 Harbor(固定 tag),把逐题结果归一化成 `TaskResult`
- **重试策略**:Harbor 有 `ApiRateLimitError` / `ContextWindowExceededError` 等细粒度异常分类与 `--retry-include`。**重试必须开启且策略固定**,否则跨代的失败率不可比。**重试次数要单独统计**——它是 harness 稳定性的一个指标
- 失败的 run 记为 `passed=False`,**不能丢弃**(丢弃会让分数虚高)

#### 4.6.3 统计(`stats.py`)

```python
def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """逐题配对差分的 bootstrap CI。diff[i] = score_new[i] - score_old[i]"""

def mcnemar(b: int, c: int) -> float:
    """返回 p 值。用于报告「逐代改进是否超过噪声」这一最严格的判据。"""

def effect_by_field(ledger) -> pd.DataFrame:
    """按 fields.py 的 group 汇总逐字段效应量:某字段被改动的那些代际,
    其 Δscore 与未改动代际的 Δscore 之差,配对 bootstrap。"""
```

**主判据**(写进论文的方法节):相邻代被接受的条件 = 配对 bootstrap 的 95% CI 下界 > 0 **且** McNemar p < 0.05。二者取更严者。

#### 4.6.4 噪声底噪(`noise_floor.py`)

**这是 W2 的第一项,必须最先做。**

- 取 gen-0,在 dev 子集上跑 **k=5 个不同 seed**(同配置,只改 seed)
- 计算逐题分数的方差 → 得出「配置不变时,Δscore 的标准误」
- 同时算:相同 seed 不同重复的 Δ(容器/网络抖动)
- **输出:可检测的最小改进阈值 δ_min**。若 δ_min 大于你预期的一代改进幅度,说明 seeds 数或任务数不够,必须先加预算

### 4.7 `optimizer/` — 配置空间搜索

三级递进,每一级都是下一级的对照:

| 级别 | 方法 | 作用 |
|---|---|---|
| L0 | **随机搜索** | 基线。回答「配置空间本身能带来多少提升」。若随机都能涨分,optimizer 不需要复杂 |
| L1 | **单字段 bandit** | 对每个 `fields.py` 里的可优化字段,维护「改这个字段」的收益估计,按 UCB/Thompson 选下一个要试的字段。**天然给出逐字段效应量** |
| L2 | **CEM** | 对连续字段(`max_tokens` / `max_iterations` / `context_keep_recent`)做交叉熵方法 |

**接受规则**:

```python
def accept(new: Generation, prev: Generation, stats) -> tuple[bool, str]:
    ci = paired_bootstrap(new.per_task_score - prev.per_task_score)
    p = mcnemar(new.b, new.c)
    ok = ci["lo"] > 0 and p < 0.05
    return ok, f"Δ={ci['mean']:+.4f} CI95=[{ci['lo']:+.4f},{ci['hi']:+.4f}] mcnemar_p={p:.4f}"
```

**编辑幅度约束(C-5)**:每代只允许改 1–2 个字段,optimizer 的 propose 函数强制执行。

### 4.8 `ledger/` — 台账

每代一条记录:

```yaml
gen: 3
spec_sha256: "3f9a..."
spec: <完整 spec 内容>
parent: 2
changed_fields:
  - path: agents[0].contract.max_tokens
    from: 800
    to: 1200
scores:
  tb_dev: {mean: 0.412, per_task: {...}}
  tb_heldout: {mean: 0.398}
  sb_heldout: {mean: 0.401}
  seeds: [1234, 1235, 1236]
  n_runs: 900
stats:
  vs_gen2: {delta: 0.011, ci95: [0.004, 0.019], mcnemar_p: 0.021}
cost: {usd: 1840.22, gpu_h: 0, llm_calls: 8123}
runtime:
  mini_swe_agent: 2.4.6
  harbor: <pinned tag>
  model: anthropic/claude-sonnet-4-6
  model_snapshot: <日期>
```

`ledger/report.md` 自动生成:分数曲线(带 CI 带)、逐代配置 diff、逐字段效应量表、homogenization 曲线。

### 4.9 `cli.py` — 命令行

```bash
rsi init                       # 生成 gen00.yaml + 冻结 base config + 切分
rsi gen propose --budget 200   # 跑 budget 次采样,产出候选
rsi gen evaluate --gen 3       # 跑 harbor,写 runs/genNN/ + ledger
rsi gen compare 3 2            # 配对统计
rsi analyze fields             # 逐字段效应量 + homogenization
rsi report                     # 生成 ledger/report.md
```

---

## 5. 已知陷阱清单(实现时逐条核对)

| # | 陷阱 | 症状 | 处置 |
|---|---|---|---|
| T-1 | `action_regex` 围栏不匹配 | 每步 FormatError,模型被骂 | 宽容正则,仍要求恰好一个块 |
| T-2 | 未登记的本地模型导致 litellm 抛 RuntimeError | 每次调用报错 | `cost_tracking: ignore_errors`;或用 `litellm_model_registry` 登记价表 |
| T-3 | 自造结束标记 | 模型原样重复直到耗尽 step limit | 用原生 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,**首行**且 returncode 0;其余行放 `SUBMIT: <status>` |
| T-4 | 工具不在 PATH | 模型花几步 `which` / `find` 找工具 | 装到 `/usr/local/bin`;harbor job 里固定 PATH |
| T-5 | Harbor 的 `mini` config 被 layered | 隐藏变量、跨 harbor 版本漂移 | pin harbor;**审计并显式覆盖每个字段** |
| T-6 | prompt 模板被 RhiAgent 接管但 mini 仍尝试渲染 | 报错或空 prompt | `system_template` 设为占位串并在 adapter 里断言 |
| T-7 | contract 字段解析失败 | 整轮 run 崩,浪费算力 | `extract_payload` 必须容错 + 统计解析失败率,超过阈值就告警 |
| T-8 | 重试策略不固定 | 跨代失败率不可比 | 固定 `--max-retries` / `--retry-include`,单独统计重试次数 |
| T-9 | 丢弃失败 run | 分数虚高 | `passed=False` 计入分母 |
| T-10 | Docker 镜像 `:latest` | 跨代环境漂移 | 用 digest 固定 |
| T-11 | 切分文件被覆盖 | 代际不可比 | 切分入库 + SHA256,变更即作废 |
| T-12 | `to_dict` 类序列化丢字段 | 恢复后行为不同 | 不用框架的序列化,用自己的 spec 序列化 |

---

## 6. 评测协议

### 6.1 统计功效(核心)

相邻两代的逐题结果只有四种:都过、都没过、新过旧不过、旧过新没。设 b = 新过旧不过,c = 反向:

```
Δscore = (b - c)/n        SE ≈ sqrt(b+c)/n        显著判据: |b-c| > 1.96·sqrt(b+c)
```

在 SWE-bench Verified(n=500)、假设 40 题表现不同:

| b | c | 净差 | Δscore | SE | 95% CI | 显著 |
|---|---|---|---|---|---|---|
| 25 | 15 | +10 | +2.0pp | 1.27pp | ±2.5pp | ❌ |
| 35 | 15 | +20 | +4.0pp | 1.41pp | ±2.8pp | ❌ |
| 45 | 15 | +30 | +6.0pp | 1.55pp | ±3.0pp | ❌ |

**单 seed 下几乎测不出逐代微改进。** 解法是**多 seed 连续化**:每题跑 k 个 seed,分数从 {0,1} 变成 {0, 1/k, …, 1},逐题差分方差大幅下降(翻转 1/3 的题贡献 1/3 而非 1,该题方差降 9 倍),整体 SE 约降 2 倍。

**协议定稿**:

| 项 | 值 |
|---|---|
| seeds | **≥3,理想 5**(`runtime.seed` 为基准,+0/+1/+2/…) |
| 比较方法 | 逐题配对 bootstrap(10k 次重采样),同时报 McNemar p |
| 接受阈值 | bootstrap CI95 下界 > 0 **且** McNemar p < 0.05(取更严) |
| 演化任务集 | dev(TB 2.1 300 题 / SBV 300 题) |
| 上报任务集 | heldout(TB 2.1 145 题 / SBV 200 题) |
| 泛化实验 | TB 上演化 → SWE-bench 上报(最终代) |
| 社区对照 | 最终 harness 在 TB 2.1 / SBV 全量上的单次结果 |

### 6.2 防过拟合

- 演化只看 dev;heldout 只在最终代与少数候选代上跑
- **每代 diff 必须人工审阅**再接受(半自动,不是全自动)
- 记录并绘制 `cos([i],[i+1])` 的 harness 语义相似度曲线,监测 RHI 报告的 homogenization
- 报告「最终 harness 在 heldout 上的提升 / dev 上的提升」之比 —— 差距大就是过拟合的直接证据

---

## 7. 里程碑与验收标准

| 周 | 里程碑 | 验收标准 |
|---|---|---|
| **W1** | 管线打通 | ① 用 Harbor + mini-swe-agent 在 TB dev 30 题上跑出基线分数 ② **产出 `audit/harbor_mini_swe_agent.md`:Harbor 注入内容的逐字段清单** ③ `gen00.yaml` 全字段冻结 + SHA256 |
| **W2** | 统计地基 | ① 噪声底噪实验完成(gen-0 × 5 seeds × 100 题)② 输出 `δ_min` 与实测 SE ③ 配对 bootstrap + McNemar 管线就绪并用合成数据单测通过 ④ **基于 δ_min 决定 seeds 与任务数** |
| **W3** | spec 与 RhiAgent | ① `schema.py` + `fields.py` 完成,校验规则有单测 ② `RhiAgent` 能跑通单 agent 与两跳 orchestrator→subagent→orchestrator ③ contract 解析失败率 < 2% |
| **W4** | 评测管线 | ① dev/heldout 切分入库 ② runner 产出 `TaskResult` ③ 两次相同配置的完整评测分数差异在 δ_min 内(管线自洽性验证) |
| **W5–6** | optimizer | ① L0 随机搜索跑完,给出「随机配置能涨多少」的基线 ② L1 bandit 跑 5 代,每代 diff 人工审阅 ③ 产出逐字段效应量表 |
| **W7** | 消融 | ① 单字段消融(每个 group 单独改)② 编辑幅度消融(1 字段 vs 2 字段 vs 自由)③ 迭代数消融 ④ homogenization 曲线 ⑤ 跨 benchmark 泛化实验 |
| **W8** | 写作 | 论文初稿 + 自动生成的 `ledger/report.md` 作为实验章节骨架 |

**每周末的 go/no-go 检查**:
- 噪声底噪是否已测出?没有则不进 W3
- δ_min 是否 ≤ 预期单代改进?不是则加 seeds/任务数,不加就换更粗粒度的 benchmark
- 逐代改进是否稳定高于 δ_min?连续两代不显著则暂停演化,做失败分析

---

## 8. 预算估算

| 项 | 计算 | 估算 |
|---|---|---|
| 单次 run | agent 10–20 min + 评测 1–2 min(瓶颈是 agent 执行) | — |
| TB 2.1 全量 445 题 × 3 seeds | 1335 runs | 15–20 h @16 并发 |
| SBV 全量 500 题 × 3 seeds | 1500 runs | 18–24 h @16 并发 |
| 演化阶段每代 | dev 300 题 × 3 seeds = 900 runs | ~11 h @16 并发 |
| **5 代演化 + 2 次全量验证** | 5×900 + 2×(1335+1500) | **~4–5 天纯机时** |
| 成本 | 按 $2–5 / run | **$15k–40k** |

**省钱方案**:演化阶段用 TB dev 150 题 × 3 seeds(每代 ~5.5 h),全量与 SWE-bench 只在最后 2 代跑 → 降至 ~$8k–20k / 3 天。

**省 token 方案**(来自 RHI 的 Claim 3):RHI 证明 harness 改进可降低 **cache read/write** 与总推理成本达 60%。把 **cost / n_llm_calls / prompt token 总量** 作为一等公民指标记录,你的研究可能同时产出成本结论——这本身就是一个有价值的副产物。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 逐代改进低于检测阈值 | 噪声底噪先行;多 seed 连续化;必要时用更粗粒度但更敏感的指标(见下) |
| 过拟合 benchmark | dev/held-out;跨 benchmark 泛化;报告 dev/heldout 提升比 |
| Harbor 默认 config 泄漏 | pin 版本;审计 + 全字段显式覆盖 |
| 有效维度远小于表面维度 | 单字段消融;随机搜索基线;逐字段效应量 |
| 同质化 / 早熟收敛 | `model` / `reasoning_effort` 作为显式破除同质化旋钮;监测 homogenization |
| 成本失控 | 演化用子集;成本指标纳入台账并设预算告警 |
| 新颖性被覆盖 | 明确三点定位;引全部既有工作 |
| **敏感信号(非本文主线,但值得记录)** | harness 改进的**主效应可能体现在「稳定性」而非「均值」**:即 pass/fail 翻转率、解析失败率、重试次数。**建议把这些作为一等指标记录**,它们比均值灵敏得多,可能成为你论文里最有说服力的一组数字 |

---

## 10. 附录:命令速查

```bash
# 依赖(pin 版本)
pip install "mini-swe-agent==2.4.6" "harbor @ git+https://github.com/harbor-framework/harbor.git@<pinned-tag>"

# 单题冒烟(不进 benchmark,只验接线)
mini -c harness/gen00.yaml -t "fix the bug" -y

# 在 Harbor 上跑 mini-swe-agent(Terminal-Bench)
harbor run -d terminal-bench-2-1 --agent mini-swe-agent \
  --agent-kw 'config={"agent":{"agent_class":"rsi.rhi_agent.RhiAgent"}}' \
  --env docker --n-concurrent 16

# SWE-bench(走 mini-swe-agent 自带集成)
mini-extra swebench --subset verified --config harness/gen00.yaml \
  --agent-class rsi.rhi_agent.RhiAgent -n 16
```

**注意**:具体 CLI 参数以 pin 住版本的 `--help` 为准;`harbor run` 的 flag 名称在 harbor 各版本间有变动,接入时先跑 `harbor run --help` 核对。
