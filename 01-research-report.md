# Harness 自进化(RSI)研究调研报告

> 版本:1.0 · 日期:2026-10-20 · 配套文档:[`02-coding-plan.md`](./02-coding-plan.md)

---

## 0. 一页速览

**研究目标**:实现 Harness 的递归自我改进(RSI)。技术路线参考 RHI(Recursive Harness Self-Improvement)——通过比较相邻两代 Harness 在同一 Benchmark 上的表现来驱动迭代优化。**与 RHI 的关键差异**:RHI 用一个外部 LLM 作为优化器改写 Harness 的提示词文本;本工作改为**直接修改结构化配置文件**,以 typed config 空间搜索替代 LLM 文本改写。

**三个核心结论**:

1. **选型**:以 **mini-swe-agent** 作为执行内核。理由不是配置面丰富(五个候选都不提供 workflow 级配置),而是**版本稳定性 + 运行成本 + 已是 Harbor 预集成 agent**。
2. **评测**:放弃 RHI 的 LLM-as-judge 成对比较,直接用 **SWE-bench Verified / Terminal-Bench** 的自动评测绝对分数。但必须解决随之而来的统计功效问题——**benchmark 的二值结果底层仍是配对比较**,需要 **≥3 seeds(理想 5)+ 配对 bootstrap**。
3. **新颖性边界**:"在 Terminal-Bench 上比较 harness"已有既有工作(`boundary-bench`、`harbor-agent-ablation`);本工作的差异必须落在**自动化闭环 + 配置空间搜索 + 递归性** 三点上。

**关键风险**(详见 §6):① 逐代改进幅度可能低于检测阈值;② benchmark 直接当目标会导致过拟合;③ Harbor 侧默认 config 泄漏进实验变量。

---

## 1. 调研范围与方法

### 1.1 覆盖范围

| 类别 | 对象 | 调研深度 |
|---|---|---|
| Harness 生态 | `SWE-agent/mini-swe-agent`、`OpenHands/software-agent-sdk` + `OpenHands/extensions`、`huggingface/smolagents`、`mini-harness/mini-harness`、`QwenLM/Qwen-Agent` | 源码级:抽象模型 / 扩展点 / 裁剪杠杆 / 版本状态 |
| 相关工作 | Athanor(arXiv:2610.21055)、RHI(arXiv:2607.15524) | 机制级拆解:算法、实验协议、可复用结论 |
| 评测基础设施 | Harbor、Terminal-Bench、SWE-bench、mini-swe-agent 的 benchmark 集成 | 接口级:agent 适配面、config 注入点、数据集映射 |
| 既有工作 | boundary-bench、harbor-agent-ablation、sidebutton-harbor-agent 等 | 定位与差异化 |

### 1.2 方法

- 通过 GitHub REST API 拉取仓库文件树,精读关键源文件(不用通读)
- 全部结论标注 `路径:行号` 或文档 URL;未核实项显式标注「未核实」
- **本次调研为静态源码与文档阅读,未实际安装运行任何框架**

---

## 2. Harness 生态调研

### 2.1 判据:「通用 → 专用」的特化能力

一个 harness 要能从通用形态被改造成专用形态,需要满足两类能力:

- **拓展(extension)**:能加自定义工具、模型后端、agent 类、钩子、动作协议
- **裁剪(trimming)**:能减工具集、限制执行环境、限制预算、砍掉不需要的组件

进一步细化为八项,后文各项目均按此打分:

| # | 判据 | 含义 |
|---|---|---|
| C1 | 配置覆盖度 | 配置能表达多少比例的 harness 行为 |
| C2 | 多 agent 结构可配置 | orchestrator-subagent 的 hop / contract 能否进配置 |
| C3 | 可 diff / 可版本化 | 相邻两代能否精确 diff 出语义化 delta |
| C4 | runtime 版本稳定 | 多代对比中,runtime 漂移会不会污染代际差 |
| C5 | 可复现 | 同配置 + 同 seed → 同输出 |
| C6 | 运行成本 | 每代跑 n 任务 × m seeds 的总开销 |
| C7 | 轨迹可读 | 能否读执行过程做 credit assignment |
| C8 | 多 agent 原语 | orchestrator-subagent 能不能真跑起来 |

### 2.2 逐项目分析

#### 2.2.1 mini-swe-agent(SWE-agent/mini-swe-agent,v2.4.6)

「A small, complete Python agent harness」。核心抽象极简且全部是 Protocol(结构化子类型,鸭子类型):

```python
# src/minisweagent/__init__.py
class Environment(Protocol):
    config: Any
    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...
    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...

class Model(Protocol):
    def query(self, messages, **kwargs) -> dict: ...
    def format_message(self, **kwargs) -> dict: ...
    def format_observation_messages(self, message, outputs, template_vars=None) -> list[dict]: ...
    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...
```

- **执行边界**:`Environment.execute(action, cwd)` 是 agent 与世界之间唯一的接口——这正是「环境接管执行」这条设计原则的天然落点
- **配置层**:`agent` / `environment` / `model` / `run` 四段式 YAML。prompt 模板走 Jinja2;`action_regex` 可换动作协议(markdown 代码块 / XML `<action>` / tool call)
- **配置合并**:`get_config_from_spec` 支持三种 spec——文件路径、内置名、`key.subkey=value` 内联;多层递归合并(`src/minisweagent/config/__init__.py`)。查找顺序含 `MSWEA_CONFIG_DIR` 环境变量
- **扩展点**:`agent_class` / `model_class` / `environment_class` **均接受完整 import path**;`MSWEA_DEFAULT_RUN` 可整体替换入口脚本
- **模型层**:`LitellmModelConfig` 含 `model_kwargs`(直传 `litellm.completion`)、`litellm_model_registry`(自登记价表用于成本核算)、`set_cache_control`(Anthropic 缓存控制)、`cost_tracking`(`default` / `ignore_errors`)、`observation_template`、`format_error_template`、`multimodal_regex`
- **内置 agent**:`default` / `interactive`(三种 mode:`human` / `confirm` / `yolo`,靠 `UserInterruption` 异常注入外部消息)
- **内置环境**:`docker` / `singularity` / `local` / `swerex_docker` / `swerex_modal` / `bubblewrap` / `contree`
- **限制语义**:`step_limit` / `cost_limit` / `wall_time_limit_seconds` / `max_consecutive_format_errors`,外加全局 `MSWEA_GLOBAL_COST_LIMIT` / `MSWEA_GLOBAL_CALL_LIMIT`
- **可观测**:`run/utilities/inspector.py` + `config/inspector.tcss` 提供轨迹 TUI;trajectory 落盘为 JSON
- **Benchmark 集成**:`run/benchmarks/swebench.py`(批跑,`concurrent.futures` 并发,逐实例 Docker 镜像 `swebench/sweb.eval.x86_64.<id>:latest`,`run.env_startup_command` 支持 Jinja 模板注入 instance 字段)、`swebench_single.py`(单题)。`DATASET_MAPPING` 覆盖 `full` / `verified` / `lite` / `multimodal` / `multilingual` / `smith` / `rebench`
- **短板**:循环结构(单 agent、一轮一个 action)写死在 Python 里,不可配置;无独立 Write/Edit 工具(只能靠 bash);无显式多 agent 编排

#### 2.2.2 OpenHands 软件代理 SDK(OpenHands/software-agent-sdk,v1.56.0)

**重要背景更正**:`OpenHands/OpenHands` 主仓(91k★)**已不是 harness**。README 标题现在是 "Agent Canvas",是自托管的 TypeScript 控制中心;核实其完整文件树,18 个 `.py` 文件全部位于测试夹具与 CI 脚本,harness 代码为零。真正的 harness 已迁至 `software-agent-sdk`。旧的 MicroAgent 机制仅保留兼容路径(`~/.openhands/microagents/` 读作 legacy skills)。

- **核心抽象**:`AgentBase` 为 ABC,字段含 `llm` / `tools` / `mcp_config` / `agent_context` / `system_prompt` / `system_prompt_filename` / `security_policy_filename` / `condenser` / `critic` / `tool_concurrency_limit`;`Agent` 追加 `persona`
- **Tool 是数据 spec**:`name` + `params`,实现与声明解耦,默认工具集可纯从数据推导
- **扩展三层**:① Python(`register_tool(name, factory)`、`register_agent(name, factory)`) ② 声明式(文件式 subagent,`.agents/agents/*.md` 或 `.openhands/agents/*.md`,frontmatter 支持 `name / description / model / tools / skills / hooks / mcp_servers / permission_mode / condenser` 等) ③ 插件(目录即插件,`plugin.json` + skills + hooks + `.mcp.json`,**兼容 Claude Code plugin 格式**)
- **裁剪**:`tools=[...]` 显式白名单、`filter_tools_regex`、`include_default_tools=[]`
- **包拆分**:`openhands-sdk` / `openhands-tools` / `openhands-workspace` / `openhands-agent-server`
- **摩擦(致命)**:① **第三方 Python 工具没有打包契约**——`register_tool` 只是进程内函数调用,无 entry point、无 `pip install` 后自动发现路径;远程 server 场景只能把工具烤进自定义镜像靠动态 import ② **SDK 日更**(1.56.0),多代对比实验的 runtime 漂移风险 ③ `openhands-tools` 硬依赖 `browser-use>=0.8.0`,裁剪工具集裁不掉依赖树 ④ 三个声明式入口(Markdown subagent / `AgentProfile` JSON / `AgentSettingsConfig`)成熟度不一
- **`OpenHands/extensions`(170★)不是包索引**:是「共享资产仓库 + 目录」——488 个 skill、`plugins/`、`marketplaces/*.json`(当前 78 个扩展)。第三方发布方式是往 `skills/` 或 `plugins/` 提 PR;也可自建 marketplace,registry 不构成发现门槛

#### 2.2.3 smolagents(huggingface/smolagents)

「A library that enables you to run powerful agents in a few lines of code」。

- **核心抽象**:`MultiStepAgent`(ABC)为根,ReAct 循环在 `_run_stream`;`CodeAgent` 生成 Python 交给 executor,`ToolCallingAgent` 走 JSON tool call
- **裁剪杠杆最全**:显式 `tools=[...]`(`add_base_tools` 默认 False)、**`authorized_imports` import 白名单**(`check_import_authorized` 强制)、`executor_type` ∈ local/e2b/docker/modal/blaxel 或注入自定义 `PythonExecutor` ABC、`max_steps`、`planning_interval=None` 关掉规划步、`final_answer_checks`、`managed_agents`(agent 当工具)
- **特化阶梯原生内建**:`ToolCallingAgent`(无代码执行)→ `CodeAgent`(带 executor)→ `MultiStepAgent`(抽象)
- **钩子**:`step_callbacks` 按 step 类注册,`CallbackRegistry` 按 MRO 匹配
- **依赖轻**:`huggingface-hub, requests, rich, jinja2, pillow, python-dotenv`;e2b/docker/mcp/litellm 均为 extras
- **无沙箱**:`LocalPythonExecutor` 明确声明自己不是安全沙箱
- **短板**:无 agent 级 YAML(仅 prompt 模板是 YAML);`to_dict`/`from_dict` 会**静默丢掉** `step_callbacks` 与 `final_answer_checks`;循环内 `ActionStep` / `PlanningStep` 硬编码,新增步骤类型须覆写 `_run_stream`;`CodeAgent` 必构造 executor

#### 2.2.4 mini-harness(mini-harness/mini-harness,209★)

「A small, complete Python agent harness: tools, compaction, retries, and a single-file TUI」,约 1700 行,4 个依赖(`openai` / `httpx` / `pydantic` / `python-dotenv`)。

- **工具集构造注入**:`DeepSeekAgent(tools, cfg=CONFIG)`,`TOOLS` 是模块级列表,内建 9 个工具(`glob_file` / `grep_file` / `read_file` / `write_file` / `edit_file` / `run_bash` / `run_sandbox` / `run_todo` / `run_subagent`)
- **工具定义**:`ToolDefinition(name, description, parameters: type[BaseModel], function, risky)`,schema 由 Pydantic 自动生成,调用前校验
- **配置**:`Config` 为 frozen dataclass,`system_prompt` 是其中一个字段;`build_config()` 仅从 `MINI_HARNESS_*` 环境变量构造,`CONFIG` 是模块级全局单例
- **Profile 机制**:`MINI_HARNESS_PROFILE=bench` 一次性换掉 `system_prompt` + 护栏开关 + 预算(`BENCH_OVERRIDE`)
- **内置子 agent**:`SUBAGENT.agent_table` 硬编码三角色(`explore_agent` / `coding_agent` / `planning_agent`),每个配**工具子集 + 专属 prompt**——是「裁剪成专用 agent」的现成范式,但**不可配置**
- **短板**:无 YAML/JSON 配置文件,特化必须改代码;裁剪工具会导致 prompt 与工具不一致;LLM provider 只支持 `deepseek` / `openai` 两分支;`_load_memory` 需交互式 stdin,不适合无人值守

#### 2.2.5 Qwen-Agent(QwenLM/Qwen-Agent)

「A framework for developing LLM applications based on Qwen」。

- **核心抽象**:`Agent`(ABC)实现 `_run`;`BasicAgent` 裸 LLM 轮;`FnCallAgent` 工具循环;`Assistant(FnCallAgent)` 加 RAG 知识前置
- **扩展**:`@register_tool`(支持 `allow_overwrite=True` 覆盖内置)、`@register_llm(model_type)`、子类化 `Agent`
- **致命短板**:`FnCallAgent.__init__` **无条件构造 RAG 的 `self.mem`**,除非该属性已存在——想要「非 RAG 专用 agent」必须绕过 `super().__init__()`;无执行限制杠杆;步数预算是全局环境变量 `MAX_LLM_CALL_PER_RUN`;无 agent 级声明式配置;`agents/__init__.py` 与 `tools/__init__.py` 急切导入全部 agent 与工具
- **文档过时警告**:网上广为流传的 `QWEN_AGENT_SETTINGS` / `init_assistant` / `register_function` / `MultiAgentRouter` **在当前 main 已不存在**(路由类叫 `Router`,见 `agents/router.py`)

### 2.3 横向对比

| 判据 | mini-swe-agent | OpenHands SDK | smolagents | mini-harness | Qwen-Agent |
|---|---|---|---|---|---|
| C1 配置覆盖度 | ⚠️ 模板/正则/limits/模型参数在配置,循环结构是 Python | ✅✅ 最广 | ❌ agent 配置基本没有 | ❌ dataclass + env | ⚠️ dict + env |
| C2 hop/contract 可配置 | ❌ 单 agent 循环 | ⚠️ 有 subagent 委派,契约靠约定 | ⚠️ `managed_agents` 有,hop 逻辑要写代码 | ❌ 3 个硬编码子 agent | ⚠️ Router 是代码类 |
| C3 可 diff/版本化 | ✅ YAML 分层合并 | ✅ JSON / Markdown+frontmatter | ⚠️ 需自写 loader | ❌ | ⚠️ dict |
| C4 版本稳定 | ✅ 2.4.6 | ❌ 日更(1.56.0) | ✅ | ✅ | ⚠️ |
| C5 可复现 | ✅ `model_kwargs` 直传 litellm,可设 seed | ⚠️ | ✅ | ⚠️ | ⚠️ |
| C6 成本 | ✅ 最低 | ❌ litellm+fastmcp+tree-sitter,tools 带 browser-use | ✅ 低 | ✅ 最低 | ⚠️ 中 |
| C7 轨迹可读 | ✅ trajectory JSON + inspector TUI | ✅ | ✅ | ⚠️ | ⚠️ |
| C8 多 agent 原语 | ⚠️ 需自写子类 | ✅✅ | ✅ | ❌ | ⚠️ |

### 2.4 第一轮调研结论

**「通用 → 专用」的特化顺畅度排序**:

| 排名 | 项目 | 定性 |
|---|---|---|
| 1 | mini-swe-agent | 全 YAML 声明式,配置分层合并,类引用可指向任意 import path——特化是配置行为而非编码行为 |
| 2 | OpenHands SDK | 声明式特化面最强,裁剪只到工具集层,Python 工具无打包契约 |
| 3 | smolagents | 裁剪杠杆最全,但配置全 Python |
| 4 | mini-harness | 极简好读,但特化必须改代码 |
| 5 | Qwen-Agent | 扩展缝隙都有,通用与专用焊死(强制 RAG Memory) |

**但对 RSI 实验床而言,这个排序需要重排**——因为五个候选**都不提供 workflow 级(多 agent hop/contract)配置**,而 RHI 的实证结论恰恰指出 contract 才是核心杠杆。既然 schema 必须自研,选型重心就转移到**执行内核的质量、稳定性与成本**上。重排结果见 §5。

---

## 3. 相关工作拆解

### 3.1 Athanor:多智能体自主后训练系统(arXiv:2610.21055)

ByteDance + Duke + UChicago + VT。核心命题:一个 frontier agent 团队用公开数据把一个 raw base model 后训练成通用 instruct 级模型。

#### 3.1.1 架构:三层角色分离

真正的架构分界**不在 agent 之间**,而在 **agent / 环境 / 操作者**三层:

| 事项 | 谁决定 | 谁执行 |
|---|---|---|
| 数据、方法、超参、合并、何时评估、提交哪个 checkpoint | **Researcher(agent)** | 环境服务:排队、分配 GPU、跑作业 |
| 怎么测量(哪些任务、怎么判分、输出上限、几个 seed) | **Operator**,campaign 前声明 | 冻结 evaluator + 服务 |
| 节点、研究窗口、续期 | **Operator** | operator 命令,每条留痕 |
| held-out suite | **Operator** | 密封评测,researcher 既看不到也不被告知 |
| research record | 无人决定 | 环境自动写 |

设计意图是「empower agents, not imprison them」:agent 保留全部研究决策权,环境只负责执行、测量、恢复。隔离靠软件强制(沙箱、冻结 evaluator、密封 held-out)。

**明确不做角色分工**:遵循 bitter lesson,不把研究流程写进 agent;作者自述早期版本「预设 recipe 形式、每次实验需审批、中央调度器」的结果是**约束了更强的 agent**。

#### 3.1.2 两条并发循环 + 一个共享历史

```
history = persistent_record()          # 共享黑板

# 每个 researcher 独立跑
while research_open():
    evidence = inspect(history)        # 自己的 + 队友的记录
    action   = agent.develop(evidence) # 在自己 workspace 里用 shell/代码分析
    athanor(action)                    # job | eval | data | artifact,立即返回

# 环境服务独立跑
while execution_open():
    updates = collect_job_updates()
    history.record(updates)
    for job in ready_work(history):
        launch_async(job)
```

两条循环彻底解耦:`athanor` 命令立即返回,作业归属于环境而非 agent 的 session。

#### 3.1.3 共享黑板 = research record

四类条目,全部落在 agent 上下文之外:

1. **journal** — 时间序日志,每请求、每状态变更一行。**请求的日志行先写,再启动执行**
2. **lineage** — checkpoint 父子关系;保留产出它的 job、job 的输入(数据集、父 checkpoint、启动时的 workspace commit)
3. **transcript** — 每个 session 的完整对话
4. **workspace history** — workspace 的 git 版本历史,含 agent 笔记

**Agent 间通信:无消息通道。** 协作全靠共享文件系统——队友 workspace 与结果的只读视图、共享 artifact store、workspace 里的普通文件形式的笔记。通信拓扑是 star-through-shared-filesystem。

#### 3.1.4 协作是涌现的,不是设计的

全文最重要的实验(the notice):

- **第 15.8 小时之前**:15.8 小时里 agent 互相读了 37 次文件,但**没给队友写过任何笔记,也没 fetch 过任何 checkpoint**——四台互不相识的机器
- **第 15.8 小时**,operator 告知「最终成绩算团队的提交」(campaign 前就定好、但文档没写的事)
- **几分钟后**每个 agent 写下给队友的第一份笔记;**通知到达后 56 秒**第一个队友 checkpoint 被 fetch
- 之后:共享笔记 160 个版本,fetch 队友 checkpoint 57 次
- **第 21 小时起**,每个刷新最佳分的 checkpoint 要么是多 agent 模型 merge,要么从这种 merge 继续训出;287 条血缘边中 139 条穿过队友 checkpoint;第 84 小时后评测的 32 个 checkpoint 里 28 个同时含四人训练成果

**结论:架构负责让协作「可能且低成本」,是否协作取决于 agent 感知到的目标是个人最优还是团队最优。**

#### 3.1.5 代价:收敛过早

四份同源 agent **一小时内就收敛到同一 recipe**(AM-Thinking-v1-Distilled),之后几乎不试替代方案。论文引 Anthropic 的多智能体研究指出:共享同一模型、同一 harness、同一 context 的 agent 会采取高度相似的行动,**一个 agent 的决定就是所有 agent 的决定**。所以真实瓶颈是**探索多样性不足**。

#### 3.1.6 关键数字

| | |
|---|---|
| 规模 | 4 × Claude Opus 5.5(Claude Code),每个独占 1 节点 × 8 GPU(80GB HBM3)= 32 GPU |
| 时长 / 成本 | 104 小时,约 $11,800(GPU $10,800 @ $3.49/GPU·h)+ 推理 ≈ $12,000 |
| 产出 | 500 个作业,255 次全量评测,174 个完整评测的 checkpoint |
| 成绩 | dev 套件 15.9 → **67.3**;held-out 9.2 → **48.2**(Instruct-2507 为 66.5 / 47.9) |
| 效率 | agent 实际工作占 ~416 agent 小时的 **9%**,纯推理 15 小时(3.6%);87% 时间在等;GPU 空闲 <1% |
| 时延 | agent 单回合中位数 4 秒;GRPO 作业 1.6 小时;全量评测 26 分钟 |
| 工具 | 2,611 条 `athanor` 命令,41 条失败 |

#### 3.1.7 对本工作的可复用结论

1. **dev / held-out 分离**是必需的,并给出可引用的协议范式
2. **跨 benchmark 泛化**是比单 benchmark 更强的证据
3. **同质化是主要瓶颈**——本工作的四个 agent 会共享同一 harness + 同一 prompt + 同一起点,探索多样性预计比论文更差,必须在论文中显式声明并监测
4. **单次实验的时延极不对称**(4 秒 vs 1.6 小时)——但这不适用于 SWE-bench/Terminal-Bench 场景,那边是「agent 跑几分钟 + 自动评测」

### 3.2 RHI:Recursive Harness Self-Improvement(arXiv:2607.15524)

Sakana AI + UC Berkeley(Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Matei Zaharia, Yujin Tang)。

#### 3.2.1 核心思想

把 harness 建模为**agent loop 本身的 prompt 级规格**,用**自身修订历史上的成对反馈**迭代精炼。

**Harness 的四组件分解**(Figure 4):

```
Agent candidates:  Agent_i : {role, instruction, contract(out-to-orchestrator)}
Workflow:          Hop1, Hop2, ..., HopM
Auxiliary rules:   acceptance gates / failure fallback / recall triggers / 通信规则
```

- **agent design** = role + instruction
- **agent workflow** = contract + hop
- RHI **优先优化 workflow**。假设:任务特异的 contract 规定了 agent 间该传什么信息,而非让下游 agent 条件于全部交互历史 → 减少冗余上下文传播、改善 KV cache 效率、降低推理成本
- **信息论解读**:全共享历史 = 稠密注意力,task-specific contract = 任务依赖的**稀疏注意力模式**

#### 3.2.2 目标函数与成本

理想目标(Eq. 1):`H*_x = argmax_H f_x(H)`,`f_x` 是对参考分布 μ 的期望成对胜率。

三种搜索的成本(Table 1):

| 目标 | N_trace | N_pair | 总成本 |
|---|---|---|---|
| 理想目标(Eq. 1) | M | M(M−1)/2 | Θ(M²) |
| 有限总体搜索(Eq. 2) | m | m(m−1)/2 | Θ(m²) |
| **轨迹局部 RHI(Eq. 3)** | **1** | **1** | **Θ(1)** |

RHI 每轮只跟**前一代的缓存输出**比,一次执行 + 一次成对评测。论文论证:在 Bradley-Terry 型偏好模型下,Eq. 3 与 Eq. 1 是同一潜在效用 `u_x` 的单调变换,因此 Eq. 3 保留相同效用序。

**自历史的必要性**:单次成对比较只是噪声较大的局部上升信号,故累积偏好历史

```
D_x^(i) = { L_eval(y^(k), y^(k-1); x_eval) }_{k=1..i}
```

作为「动量语义」的信号引导后续修订。离散文本空间没有梯度,历史充当动量。

#### 3.2.3 算法

```
H^(0) ← 初始 harness
for i = 1, 2, ...:
    for j in 1..n:  y_ij ~ A(H^(i)_j, x_j)
    L_eval 成对比较 y_ij vs y_ij^(i-1) → 更新历史 D_j
    s_i = (1/n) Σ_j 1{y_ij ≻ y_ij^(i-1)}
    if s_i < ε: BREAK
    H^(i+1) ← L_harness( D, H^(i) )     # 优化器 LLM
```

#### 3.2.4 实验

- 30 个合成的开放式 ML 研究任务(量化金融 / 机器人 / 药学,各 10 个),输出是代码仓库
- 模型:Claude Sonnet 4.6 / Opus 4.7 / Opus 4.8,reasoning effort `high`;对照组是 `xhigh` / `max` / `ultracode`
- 评测:LLM-as-judge 成对比较,2–6 个 judge 配置,3 seeds
- 结果:`high + RHI` 的成对胜数超过 `xhigh` / `max` / `ultracode`;**推理成本最多降 60%**
- Claim 2:增益不来自更长的生成;Claim 3:成本效率来自**缓存读写减少**

#### 3.2.5 最有价值的两条结论

**(a) 组件级归因(§6.2.2)**:把 harness 拆成 role / instruction / contract / hop 四类做 embedding 聚类(t-SNE + UMAP,`text-embedding-3-large` 与 `all-mpnet-base-v2` 双 embedding 模型):

> Across all three domains, Figures 10a, 10b, and 10c show that **contracts exhibit the clearest task-dependent clustering**... suggesting the largest semantic changes occur at the information interface between the orchestrator and subagents.

而 **roles 最不可分**——因为这些任务共享「数据准备 / 模型训练 / 验证 / 报告撰写」等高层角色。

→ **对 typed config 的直接启示**:搜索预算应压在 `contract` 与 `hop` 上,`role` 几乎不动。

**(b) 同质化(§6.2.1)**:论文报告了 **harness improvement homogenization**——迭代后期各任务的 harness 变得彼此相似(`cos([i],[i+1])` 随迭代下降),这是退化信号,必须监测。

### 3.3 两篇的交集与差异

| 维度 | Athanor | RHI | 本工作 |
|---|---|---|---|
| 研究对象 | agent 团队做后训练研究 | harness 自身被优化 | harness 自身被优化 |
| 优化器 | 人(operator)只做运维 | **LLM 改写 prompt 文本** | **程序改 typed config** |
| 信号 | 开发集分数 | LLM judge 成对偏好 | **benchmark 自动评测绝对分数** |
| 状态外置 | research record(带血缘) | 偏好历史 D | 配置台账 + 逐代归档 |
| 多 agent | 4 个同构 agent 共享黑板 | orchestrator-subagent(hops/contracts) | 待定(schema 定义) |
| 泛化协议 | dev / held-out 双套件 | 30 任务 + 3 seeds | dev / held-out + **跨 benchmark** |

---

## 4. 本工作的定位与新颖性边界

### 4.1 技术路线

```
基准任务集(dev)  →  第 i 代 harness 配置  →  自动评测  →  绝对分数
                       ↑                                      │
                       └──── typed config 更新(相邻代差分) ←──┘
                                                          held-out 上报
```

三条设计原则的来源:

- **相邻代比较** → RHI 的 Eq. 3(轨迹局部目标),但把信号从成对偏好换成绝对分数
- **状态外置 + 逐代归档** → Athanor 的 research record(带血缘、可回溯)
- **dev / held-out + 跨 benchmark** → Athanor 的 §4.1 协议

### 4.2 相对 RHI 的三点差异(论文贡献点)

1. **搜索空间从无类型散文变成有类型字段**。RHI 的候选是一段自然语言规格;本工作的候选是一个 typed vector。收益:编辑幅度可度量(改了几个字段、每字段什么值),归因从事后 embedding 聚类提升到**逐字段效应量**
2. **消除了「优化器 LLM 能力」这个混淆变量**。RHI 的增益里混着 `L_harness` 本身的强弱;换成 typed config 搜索后可以剥离,并可做**随机配置搜索基线**来标定「配置空间本身有多大」
3. **信号从胜负计数换成连续分数差分**。若 benchmark 支持可验证评分,配对连续差分的方差比胜负计数低一个量级;同时省掉 LLM judge 的成本与偏差

### 4.3 必须引用的既有工作

| 工作 | 内容 | 对本工作的意义 |
|---|---|---|
| **RHI**(arXiv:2607.15524) | RSI 闭环、trajectory-local 目标、组件归因 | 直接前作,核心对照 |
| **boundary-bench** | 在 Terminal-Bench 上对比 Claude Code / Codex / Terminus 2 / Grok 四个 harness,NIST 加固沙箱 | **证明 harness 差异在 Terminal-Bench 上可测** |
| **harbor-agent-ablation** | Harbor Terminal-Bench 上的受控 harness 对比 | **受控对比方法学** |
| **sidebutton-harbor-agent** | Harbor agent adapter,「maintainer-reproducible: bundled skill packs + verify-before-done config」 | **「配置驱动 harness」的既有工程实践** |
| **Athanor**(arXiv:2610.21055) | 4 agent 共享黑板做后训练,dev/held-out,协作涌现 | dev/held-out 与多 agent 组织范式 |
| **mini-swe-agent** v2.4.6 / **Harbor** | 执行内核与适配层 | 基础设施 |

**结论**:"在 Terminal-Bench 上比较 harness" **已被做过**。本工作的新颖性必须落在 **自动化闭环 + 配置空间搜索 + 递归性** 三点上,且这三点的组合是新的。

---

## 5. 执行内核选型(针对 RSI 实验床重排)

### 5.1 选型判据的调整

第一轮排序(§2.4)是按「通用 → 专用的特化顺畅度」排的。**对 RSI 实验床需要重排**,因为:

- 五个候选都不提供 workflow 级配置 → schema 必须自研 → 选型重心转移到执行内核
- 多代对比实验里,**runtime 版本漂移会直接污染代际差**,这是 C4 从「加分项」变成「一票否决」的原因
- 每代要跑 n 任务 × m seeds,成本 × 代数放大,C6 从「加分项」变成「硬约束」

### 5.2 重排结果

**第一:mini-swe-agent**

| 理由 | 证据 |
|---|---|
| 配置层是 YAML 分层合并,天然承载「schema + 任务 + 代际 override」三层 | `config/__init__.py` 的 `get_config_from_spec` / `recursive_merge` |
| 类引用可指向任意 import path → 不 fork 就能挂自己的 `RhiAgent` | `agents/__init__.py` 的 `get_agent` / `get_agent_class` |
| 可整体替换入口 | `MSWEA_DEFAULT_RUN` 环境变量 |
| 版本稳定(2.4.6) | 多代对比不漂移 |
| 运行成本最低 | 依赖极轻 |
| **已是 Harbor 预集成 agent** | `src/harbor/agents/installed/mini_swe_agent.py` + `examples/configs/agents/mini-swe-agent-job.yaml` |
| **自带 SWE-bench 批跑** | `run/benchmarks/swebench.py` 的 `DATASET_MAPPING` 覆盖 verified/lite/full/multimodal/multilingual/smith/rebench |
| seed 可控 | `model_kwargs` 直传 `litellm.completion` |
| Anthropic 支持已处理 | `set_cache_control` + `anthropic_utils._reorder_thinking_blocks` |

**第二:OpenHands SDK** —— 只借 schema 思想,不用作内核。理由:日更版本(实验污染)、依赖重、自带工具层与自定义 schema 冲突、且它是 agent-centric 而非 workflow-centric,**RHI 的 hops/contracts 照样得自己写**。

**第三:smolagents** —— `managed_agents` 是最原生的 agent-as-tool,概念对;但无 agent 级配置、无 session 持久化、`to_dict` 丢回调,且循环封在库里(改 hop 图要动库或大改 → 「配置即 harness」的前提塌掉)。

**第四:mini-harness** —— 配置面最窄,**直接违背「改配置文件」的前提**;但代码最好读,若最终收敛到单 agent,它是最好的起点。

**出局:Qwen-Agent** —— 强制 RAG Memory 引入噪声、无 agent 级配置、Router 是代码类。

### 5.3 一个额外发现

Harbor 的预集成 agent 列表里同时有 `openhands_sdk.py` / `swe_agent.py` / `openhands.py` / `mcode.py` / `claude_code.py` / `codex.py`。**这意味着本工作研究的那几个 harness 能在同一套 Harbor 适配层下直接对比**——同一环境、同一评测、同一超时与重试策略,只有 harness 配置不同。这才是 harness 对比实验该有的样子,也是本工作天然能做而 RHI 做不到的(泛化性实验)。

---

## 6. 风险清单

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | **逐代改进幅度低于检测阈值** | 高 | ≥3 seeds(理想 5)+ 配对 bootstrap;**先测噪声底噪再写 optimizer** |
| R2 | **benchmark 变训练集,过拟合** | 高 | 任务级 dev/held-out 切分;**跨 benchmark 泛化**(TB 演化 → SWE-bench 报) |
| R3 | **Harbor 默认 config 泄漏进实验变量** | 高 | pin Harbor 版本;审计 `mini_swe_agent.py` 注入内容;gen-0 全字段显式钉死 + SHA256 存档 |
| R4 | **同质化 / 早熟收敛** | 中 | 单代只改 1–2 字段;监测 `cos([i],[i+1])`;引入多样化(不同模型 / 不同子 agent) |
| R5 | **有效维度远小于表面维度** | 中 | 单字段消融;逐字段效应量;随机配置搜索基线 |
| R6 | **成本失控** | 中 | 演化阶段用子集(TB 150 题 × 3 seeds),全量 + SWE-bench 只在最后 2 代跑 |
| R7 | **新颖性被既有工作覆盖** | 中 | 明确以「自动化闭环 + 配置空间搜索 + 递归性」定位;引全部既有工作 |
| R8 | **agent 侧泄漏**:benchmark 自带提示词 | 中 | 审计 Harbor 各 agent 的 `render_instruction` / 默认 prompt;固定为 gen-0 常量 |

---

## 7. 参考来源

**Harness**

- `SWE-agent/mini-swe-agent` v2.4.6 — `src/minisweagent/__init__.py`(Protocol 定义)、`config/__init__.py`(`get_config_from_spec` / `get_config_path` / `_key_value_spec_to_nested_dict`)、`agents/__init__.py`(`get_agent` / `get_agent_class`)、`agents/default.py`(`AgentConfig` / `DefaultAgent`)、`agents/interactive.py`(`InteractiveAgentConfig` / `UserInterruption`)、`environments/__init__.py`(`_ENVIRONMENT_MAPPING`)、`environments/local.py`(`LocalEnvironmentConfig` / `_check_finished`)、`models/__init__.py`(`get_model`)、`models/litellm_model.py`(`LitellmModelConfig`)、`run/mini.py`、`run/utilities/config.py`、`run/utilities/inspector.py`、`run/benchmarks/swebench.py`、`run/benchmarks/swebench_single.py`、`config/{mini,mini_textbased}.yaml`、`config/benchmarks/*.yaml`、`docs/advanced/{yaml_configuration,cookbook,environments,global_configuration,control_flow}.md`
- `OpenHands/software-agent-sdk` v1.56.0 — `agent/{base,agent}.py`、`tool/{spec,defaults,registry}.py`、`subagent/{registry,schema,load}.py`、`context/agent_context.py`、`plugin/{plugin,discovery,installed}.py`、`marketplace/registration.py`、`profiles/agent_profile.py`、`conversation/conversation.py`
- `OpenHands/extensions` — `README.md`、`MIGRATION.md`、`plugins/README.md`
- `OpenHands/OpenHands` — README(已核实为 Agent Canvas,18 个 `.py` 全为测试/CI 脚本)
- `huggingface/smolagents` — `src/smolagents/{agents,tools,models,memory,local_python_executor,mcp_client}.py`、`pyproject.toml`、`docs/source/en/tutorials/*`
- `mini-harness/mini-harness` — `README.md`、`src/mini_harness/{config,agent,bench_profile,compact,retry_request}.py`、`src/mini_harness/tool/{box,block,path,tag}.py`、`bench/adapter.py`
- `QwenLM/Qwen-Agent` — `qwen_agent/{agent,settings}.py`、`qwen_agent/agents/{assistant,fncall_agent,router}.py`、`qwen_agent/tools/{base,mcp_manager}.py`、`qwen_agent/memory/memory.py`、`qwen_agent/llm/{__init__,base}.py`、`setup.py`

**论文**

- Athanor: An Agent-Native System for Autonomous Post-Training Research — arXiv:2610.21055v1(2026-10-15),ByteDance / Duke / UChicago / VT。§3.1–3.3(架构与三原则)、§4.1(协议)、§4.3(执行效率)、§5.2(团队进展与 the notice)、§6(局限)
- Recursive Harness Self-Improvement — arXiv:2607.15524v1(2026-07-17),Sakana AI / UC Berkeley。§2.1(问题定义)、§3.1(Eq. 1–3)、§3.2(Algorithm 1)、§3.3(harness 分解)、§5.1–5.5、§6.2.1–6.2.2(组件演化与归因)、§6.3(信息论假设)

**评测基础设施**

- `harbor-framework/harbor`(Apache-2.0)— `src/harbor/agents/base.py`、`src/harbor/agents/installed/{base,mini_swe_agent,openhands_sdk,swe_agent}.py`、`examples/configs/agents/mini-swe-agent-job.yaml`、`docs-mintlify/agents/custom-agents.mdx`
- `laude-institute/terminal-bench` — README(Harbor 迁移公告、Terminal-Bench-Core v0.1.1、beta ~100 tasks)
- SWE-bench(`princeton-nlp/SWE-Bench_Verified`,500 题)

**既有工作**

- `boundary-bench/boundary-bench`、`namanvats/harbor-agent-ablation`、`sidebutton/sidebutton-harbor-agent`、`badlogic/pi-terminal-bench`、`bkrabach/harbor-amplifier`、`Shaopen47/term-harness`
