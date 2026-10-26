# terminal-5task — 用 mini-swe-agent / opencode 双 harness 合成 agent 训练轨迹

> 本文是那次 pilot 的方法学记录。手写的代码（转换器、校验脚本、任务定义）
> 在 `common/pilot/`，生成的产物（抓包、原生轨迹、转换后的样本）在
> `work/pilot/terminal-5task/`，后者不入 git。

用**两个真实 agent harness**（mini-swe-agent 2.4.6、opencode 1.18.33）在本地 vLLM 端点上
跑真实 terminal 任务，把 rollout 转成可直接用于 SFT 的 OpenAI 格式消息序列。

产出：5 个任务 × 2 个 harness = **10 条 SFT 样本，全部通过「与 wire 逐字一致」校验**。

| | |
|---|---|
| 端点 | `http://YOUR-ENDPOINT:PORT/v1`（vLLM，无鉴权），模型 `glm5.3-flash` |
| 样本数 | 10（mini 5 + opencode 5），每条 5–10 条消息 |
| 校验 | 转换结果的前 N 条 == 该 rollout **最后一次请求**的 N 条消息（逐字），且多出恰好 1 条最终 assistant |
| 乱码 | 已检测并修复；`usable_for_sft` 标记当前 10/10 为 True |

---

## 1. 核心原则：训练数据取「模型侧」，不是 harness 侧

这是整套方法的地基。**harness 自己记的轨迹 ≠ 模型实际收到的消息**，
拿 harness 侧反推出来的上下文去训练，模型学到的是「在日志里看起来该怎么做」，
而不是「在真实请求里该怎么做」。

所以每个 harness 前都挂一个**记录代理**，把真实的 OpenAI 请求体原样落盘：

```
agent ──HTTP──▶ record_proxy.py ──HTTP──▶ vLLM 端点
                 └─ 每次请求的 body 追加进 JSONL
```

抓包拿到的东西，harness 自己都不一定有：opencode 的 system prompt（80,330 字符）
和 tool schema 是编译进二进制的，`opencode debug agent` 只给权限和工具列表；
它也**不在 session export 里**。抓包是唯一能拿到模型真实所见字节的地方。

### 1.1 两个 harness 的差异

| | mini-swe-agent 2.4.6 | opencode 1.18.33 |
|---|---|---|
| system prompt | 62 字符 | **80,330 字符** |
| 是否触发模型思考 | 否 | 是（`reasoning_content` 790/94 字符） |
| harness 侧 ↔ 模型侧 | **逐字相同**，只多 `extra`、`function_call` 两个内部键 | 差得远：system prompt 不在 session 里 |
| 推理内容的载体 | wire 上是 `provider_specific_fields: {reasoning, refusal}` | wire 上是独立字段 `reasoning_content` |
| 会话历史 | traj.json 里已有完整 messages | session 是 part 树，要**重建** |
| 抓包里的干扰请求 | 无 | 每个任务多一次标题生成请求 |

**mini 侧**是「剥壳」：harness 记的 messages 逐字就是模型侧收到的，只要
去掉 `role == "exit"` 的消息、去掉尾部那条 `exception_info == "action was not executed"`
的提交占位符观测（模型从未收到过）、再剥掉 `extra` / `function_call` /
值为 `None` 的字段（litellm 重新序列化时会省略 None 字段，wire 上的 assistant
消息干脆没有 `content` 键）。

**opencode 侧**是「重建」：system prompt 从抓包取，历史从 session 的 part 树重建，
重建时 `tool_calls[].function.arguments` 必须用 `separators=(",", ":")`——
JS 的 `JSON.stringify` 不带空格，Python 的 `json.dumps` 默认带，
语义等价但 token 序列不同。

### 1.2 一条样本的精确定义

```
messages = [ 该 rollout 最后一次请求的全部消息 ] + [ 模型最后一次输出 ]
```

最后那次输出从未被发回去过（run 结束就没了），所以它不在任何抓包里，
只能从 harness 侧的轨迹补——这也是唯一允许用 harness 侧补齐的地方。

---

## 2. 目录结构

手写的部分进仓库，生成的产物进 `work/`（不入 git）：

```
common/pilot/                  ← 仓库里
├── README.md               ← 本文
├── tasks.json              5 个任务（id / slug / title / task）
├── to_openai_sft.py        转换器：harness 原生产物 → OpenAI messages
└── build_sft.py            批量转换 + 与 wire 逐字校验（生成 sft/*.json）

work/pilot/terminal-5task/    ← 生成的，不入 git
├── captures/
│   ├── mini-wire.jsonl     mini 侧抓包（19 条请求，全量 body）
│   └── opencode-wire.jsonl opencode 侧抓包（20 条请求，全量 body）
├── mini/
│   ├── raw/01..05.json     mini 的 traj.json 原样存档
│   └── sft/01..05.json     转换后样本（messages + 校验元信息）
└── opencode/
    ├── raw/01..05.json     opencode 的 session export 原样存档
    └── sft/01..05.json     转换后样本
```

`common/pilot/build_sft.py` 默认从 `work/pilot/terminal-5task/` 读产物、写样本；
`tasks.json` 跟脚本放在一起，因为它是输入而不是产物。

---

## 3. 端到端流程

> 下列命令全部在仓库根目录下实跑验证过。

```bash
export REPO=$(pwd)   # the repository root
export DS=$REPO/work/pilot/terminal-5task
export CFG=$REPO/mini_sft/config
```

### Step 0 · 准备

mini-swe-agent 的本地端点配置在 `$CFG/`（`overrides.yaml` → `regen.py` → `local-glm.yaml`），
全局环境变量在 `~/Library/Application Support/mini-swe-agent/.env`：

```bash
MSWEA_CONFIG_DIR=<repo>/mini_sft/config
MSWEA_MINI_CONFIG_PATH=<repo>/mini_sft/config/mini-config.yaml
```

### Step 1 · 起两个**隔离**的抓包代理

两个 harness 必须各用一个端口、各写各的抓包文件——共用会互相污染，
抓出来的对话历史会串。

```bash
python3 $REPO/common/record_proxy.py --port 8897 --host 127.0.0.1 \
  --upstream http://YOUR-ENDPOINT:PORT/v1 \
  --out $DS/captures/mini-wire.jsonl &

python3 $REPO/common/record_proxy.py --port 8898 --host 127.0.0.1 \
  --upstream http://YOUR-ENDPOINT:PORT/v1 \
  --out $DS/captures/opencode-wire.jsonl &
```

**不要加 `--compact`**：compact 模式只留 system + 首条 user + tools，
把对话历史整个丢掉（转换器要靠它做校验）。
**不要加 `--dedupe`**：dedupe 只在 compact 下生效，且保留的是每个 rollout 的
**第一次**请求，不是最后一次——而样本要的是最后一次请求。

### Step 2 · 采集 mini

`mini` 的 `-c` 是**递归合并**的，但**一旦显式给了 `-c`，默认配置就不再被使用**，
所以必须把全量配置显式带上，再叠一层只改 `api_base` 的覆盖：

```bash
sed 's|api_base: http://YOUR-ENDPOINT:PORT/v1|api_base: http://127.0.0.1:8897/v1|' \
  $CFG/local-glm.yaml > /tmp/mini-proxy.yaml

mkdir -p /tmp/run && cd /tmp/run
mini -c $CFG/local-glm.yaml -c /tmp/mini-proxy.yaml \
    -y --exit-immediately -o traj.json -t "<任务文本>" < /dev/null
```

- `-t` 传任务，不是位置参数（用 `--` 会报 "unexpected extra argument"）
- `--exit-immediately` 在非 TTY 环境**必需**，否则 agent 结束时会撞上 `EOFError`
- `traj.json` 存到 `$DS/mini/raw/<任务id>.json`

### Step 3 · 采集 opencode

opencode 的 provider 只能通过环境变量注入：

```bash
export OPENCODE_CONFIG_CONTENT='{"provider":{"probe":{"npm":"@ai-sdk/openai-compatible","name":"probe","options":{"baseURL":"http://127.0.0.1:8898/v1","apiKey":"EMPTY","setCacheKey":true},"models":{"glm5.3-flash":{"name":"glm5.3-flash"}}}}}'

opencode --model=probe/glm5.3-flash run --format=json --auto \
    --dir "$WORKDIR" -- "<任务文本>" < /dev/null > run.jsonl
```

- `--auto` 必须放在 `run` **之后**，放前面 yargs 解析会失败
- `< /dev/null` 必需：opencode 会吃掉 `while read` 的 stdin
- `--format=json` 输出的是 **JSONL**（一行一个事件），不是单个 JSON 对象；
  每行里有 `sessionID`，用它导出 session：

```bash
SID=$(head -1 run.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['sessionID'])")
opencode export "$SID" > $DS/opencode/raw/<任务id>.json
```

（`opencode session list` 也能拿到 id，但按时间倒序，容易拿错。）

### Step 4 · 批量跑（5 个任务）

```bash
mkdir -p $DS/mini/raw $DS/opencode/raw

python3 - $DS/tasks.json > /tmp/tasks.tsv <<'PY'
import json, sys
for t in json.load(open(sys.argv[1]))['tasks']:
    print(t['id'], t['slug'], t['task'], sep='\t')
PY

while IFS=$'\t' read -r ID SLUG TASK; do
  W=/tmp/runs/$ID/work; mkdir -p "$W"
  ( cd "$W" && mini -c "$CFG/local-glm.yaml" -c /tmp/mini-proxy.yaml \
      -y --exit-immediately -o "$DS/mini/raw/$ID.json" -t "$TASK" < /dev/null ) >"$W/mini.log" 2>&1
  ( cd "$W" && opencode --model=probe/glm5.3-flash run --format=json --auto \
      --dir "$W" -- "$TASK" < /dev/null ) >"$W/oc.jsonl" 2>"$W/oc.err"
  SID=$(head -1 "$W/oc.jsonl" | python3 -c "import json,sys; print(json.load(sys.stdin).get('sessionID',''))")
  opencode export "$SID" > "$DS/opencode/raw/$ID.json"
done < /tmp/tasks.tsv
```

### Step 5 · 转换 + 校验

```bash
cd $DS && python3 build_sft.py
```

输出：

```
id  harness   wire校验       条数  wire  备注
01  mini      ✓             5     4  前 4 条与 wire 逐字一致，多出 1 条 role=assistant
01  opencode  ✓             8     7  前 7 条与 wire 逐字一致，多出 1 条 role=assistant
...
共 10 条，全部通过: True
```

---

## 4. 脚本用法

### 4.1 `common/record_proxy.py` —— 记录代理

```
--port PORT        监听端口（默认 8899）
--host HOST        绑定地址；容器要访问时用 0.0.0.0
--upstream URL     上游端点（默认 http://YOUR-ENDPOINT:PORT/v1）
--out PATH         抓包输出 JSONL
--compact          只留 system + 首条 user + tools，丢弃对话历史（**本流程不要用**）
--dedupe           每个 rollout 只记一条（**本流程不要用**，且只在 --compact 下生效）
```

透传代理：不改 body，流式回响应，所以长 rollout 不会被缓冲。
超过 8 MB 的请求体不记录（那 runaway context，不是 rollout）。

### 4.2 `to_openai_sft.py` —— 转换器

单文件转换：

```bash
# mini
python3 to_openai_sft.py mini traj.json > out.json

# opencode（system prompt 从抓包里抠出来存成文本再传进去）
python3 to_openai_sft.py opencode session.json --system system.txt > out.json
```

可用参数：`--task-id`、`--task`、`--meta`（额外 JSON 元信息，会平铺进输出）。

核心 API：

```python
from to_openai_sft import (
    mini_to_openai,      # traj.json -> messages
    opencode_to_openai,  # session.json + system prompt -> messages
    repair_messages,     # 乱码修复 -> (messages, warnings, 是否仍有损坏)
    summarize,           # 条数 / role 分布 / tool_call 数 / 带 reasoning 的 assistant 数
)
```

### 4.3 `build_sft.py` —— 批量转换 + 校验

读 `tasks.json` + `captures/*-wire.jsonl` + `*/raw/*.json`，写 `*/sft/*.json`，
并在终端打出逐条校验表。**没有参数**，直接在 `$DS` 下跑。

产物字段：

```jsonc
{
  "messages": [ ... ],          // OpenAI 格式，role ∈ {system,user,assistant,tool}
  "source": "mini",
  "task_id": "01", "task_slug": "csv-column-sum", "task": "...",
  "n_wire_requests": 2,
  "n_messages": 5,
  "roles": {"system":1,"user":1,"assistant":2,"tool":1},
  "n_tool_calls": 2,
  "n_assistant_with_reasoning": 0,
  "wire_parity": {"ok": true, "note": "前 4 条与 wire 逐字一致，多出 1 条 role=assistant"},
  "warnings": [],               // 非空 = 有乱码修复
  "usable_for_sft": true        // false = 仍有无法修复的损坏，别进训练集
}
```

### 4.4 `mini_sft/config/` —— mini 的本地端点配置

| 文件 | 作用 |
|---|---|
| `overrides.yaml` | 覆盖块（**改这个**） |
| `local-glm.yaml` | 生成物：`内置 mini.yaml + overrides.yaml`，别手改 |
| `regen.py` | 重新生成 `local-glm.yaml`（升级 mini-swe-agent 后要跑） |

```bash
~/.local/share/uv/tools/mini-swe-agent/bin/python $CFG/regen.py
```

`regen.py` 会在写出前断言 YAML 往返无损——长 Jinja 模板一旦被 YAML 折行，
提示词会被静默改掉，这一步不能省。

### 4.5 生产规模（31,600 条）走另一条路

本目录是 **pilot + 方法论**，验证「模型侧取数」这条路走得通。真要批产，
用 `opencode_sft/` 里已有的 Harbor 流水线：

| 脚本 | 作用 |
|---|---|
| `run_rollouts.py` | 批量驱动 Harbor 跑 rollout，带可断点续跑的 ledger（`progress.jsonl`），支持 `--shard i/n` 分机 |
| `build_sft_dataset.py` | Harbor 产出 → 双视图 SFT 数据集 |
| `shared_network.py` | 共享 docker 网络，避免地址池耗尽 |

`build_sft_dataset.py` 产出的是**双视图**：`conversations`（对齐公开数据集
OpenThoughts-Agent-SFT-31.6K 的 `[{role, content}]` 严格 schema）和
`messages`（tool-aware trainer 真正想要的，tool_calls 结构化、tool 结果是
`role:"tool"` + `tool_call_id`）。

**推荐分工**：**用 opencode 采 SFT 轨迹，用 mini 做评测 harness**。
opencode 自带 `reasoning_content`（mini 全为 0），且 mini 的文本围栏协议与
公开数据集一致——同一个模型，跑出两种可读性完全不同的轨迹。

---

## 5. 校验规则：怎么证明数据是真的

光看转换结果「像不像」没用，必须和 wire 逐字比对。规则：

1. 在抓包里按任务定位该 rollout 的**所有**请求（用首条 user 消息匹配，
   跳过 opencode 的标题生成请求）

   两个细节决定了匹配必须用**子串**而不是相等：
   - opencode 的**第一个请求是标题生成**（首条 user 消息 =
     `"Generate a title for this conversation:"`），不是任务本身，必须排除；
   - opencode 真正带任务的请求里，首条 user 消息的内容是
     **被双引号包裹**的任务原文（`"在当前工作目录创建 sales.csv，..."`），
     所以要匹配 `task[:18]` 这种前缀片段。

2. 取**最后一次**请求的 messages 作为基准
3. 转换结果必须满足：**前 N 条 == 基准的 N 条（逐字）**，且**长度恰好 N+1**

第 3 条的「+1」是硬约束：最后那次 assistant 输出模型生成了但从未发回去，
所以它不在抓包里，却必须出现在训练样本里。多一条少了都对不上。

mini 侧是严格逐字比对；opencode 侧额外容忍一处差异——
`tool_calls[].function.arguments` 的 JSON 空格（wire 用 JS `JSON.stringify`，无空格）。

### 5.1 样本末尾有一条「无配对 tool 消息」的 assistant

这是设计使然，不是 bug，但下游训练必须知道：

- **mini**：最后一条 assistant 的 tool_call 是
  `bash: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`（提交指令）。
  mini 剥掉了它对应的 tool 消息——那条观测带
  `exception_info == "action was not executed"`，因为提交指令从未真正执行，
  模型也从未收到过它的结果。所以样本里最后一个 tool_call **没有** tool 消息。
- **opencode**：最后一条 assistant 是纯文本答案（无 tool_calls），配对完整。

实测配对情况（`calls` = assistant 里的 tool_call 数，`tools` = tool 消息数）：

| 样本 | assistant 的 tool_calls | tool 消息 | 配对 |
|---|---|---|---|
| opencode 01/02/03/05 | `[1, 2, 0]` | 3 | ✓ 全配对 |
| opencode 04 | `[1, 1, 0]` | 2 | ✓ 全配对 |
| mini 01/03/04 | `[1, 1]` | 1 | 末尾提交指令差一条（预期） |
| mini 02/05 | `[1, 1, 1]` | 2 | 末尾提交指令差一条（预期） |

如果下游模板要求「每个 tool_call 都有 tool 消息」，mini 的样本需要单独处理
（通常是把末尾那条 assistant 的提交指令从 tool_calls 降级成纯文本）。

---

## 6. 踩坑清单（每条都带实证）

1. **`mini -c` 是整份替换，不是合并基座** —— 一旦给了 `-c`，默认配置就不再用。
   必须 `-c local-glm.yaml -c 覆盖层.yaml` 这样把全量带上。

2. **`mini` 任务用 `-t` 传** —— 用 `--` 位置参数会报 "unexpected extra argument"。

3. **`--exit-immediately` 在非 TTY 环境必需** —— 否则 agent 结束撞 `EOFError`。

4. **opencode 的 `--auto` 必须在 `run` 之后** —— 放前面 yargs 解析失败。

5. **opencode 会吃掉 `while read` 的 stdin** —— 循环里必须 `< /dev/null`，
   否则第一个任务就把循环的输入吃光了。

6. **`opencode run --format=json` 是 JSONL** —— 用 `json.load` 读整个文件会报
   "Extra data: line 2"，得逐行读。

7. **`tool_calls.arguments` 的 JSON 空格** —— JS 无空格、Python 默认带空格。
   语义等价但 token 序列不同，必须用 `separators=(",", ":")`。

8. **值为 `None` 的字段要整个去掉** —— litellm 重新序列化时省略 None 字段，
   wire 上的 assistant 消息没有 `content` 键，traj 里却是 `content: null`。

9. **`provider_specific_fields` 不是错误** —— mini 的 wire 上确实有它
   （`{reasoning, refusal}`），不要当成脏字段删掉。

10. **别把 `reasoning_content` 拼进 `content`** —— Qwen3 的 chat template 会
    渲染成双层 think。必须作为独立字段透传。

11. **连续两条 `user` 消息会破坏严格交替不变量** —— 这条针对**降级视图**
    （生产转换器 `build_sft_dataset.py` 产出的 `conversations`，对齐公开数据集
    OpenThoughts-Agent-SFT-31.6K 的严格 `user`/`assistant` 交替 schema）。
    opencode 一轮 assistant 可以并行发多个工具（实测一轮 2 个），
    降级成观察消息时如果每个工具结果各占一条 user 消息，就会出现连续 user，
    破坏原数据集的不变量。本目录的 `messages` 视图不受影响——并行工具的多个结果
    本来就该是连续多条 `role:"tool"` 消息（这符合 OpenAI 规范）。

12. **抓包代理端口要先腾干净** —— `pkill -f "record_proxy.py --port 8898"` 再起，
    否则 `Address already in use`。

13. **模型输出偶发中文乱码** —— GLM-5.3-flash 经 vLLM 返回的中文会间歇性变成
    「UTF-8 字节被按 latin-1 解码」的形态，**抓包里就是乱的**（不是 harness
    序列化问题）。且损坏是**部分有损**的：多数区段可无损还原，个别字节被替换成
    空格，整串往返会 `UnicodeDecodeError`。修复策略见 `to_openai_sft.repair_mojibake`：
    能整串还原就还原，否则取「最长可解码前缀」，剩下的尾巴标记为不可用。

    判据要严：纯 ASCII 串也能 latin-1→utf-8 往返成功，所以必须加
    「还原后 CJK 数增加」这一条，否则会误改。（第一版检测器字符类写成
    `0x2D-0xBF`，把 ASCII 全框进去了，误报一片。）

14. **别用 harness 侧轨迹反推模型上下文** —— 本文档第 1 节就是这条的展开。

---

## 7. 已知局限

- **只有 5 个任务**，且是自造的 terminal 任务，不是真实 SWE 任务；
  样本的 `system` 消息长度在两个 harness 之间差 1300 倍，分布不真实。
- **mini 侧 `reasoning_content` 全空** —— mini 的推理内容藏在
  `provider_specific_fields.reasoning` 里，要不要提升成统一的
  `reasoning_content` 字段，取决于下游训练模板怎么渲染。
- **opencode 的 tool 结果顺序**依赖 part 树顺序，跨轮并行调用时与 wire 的
  排列是否总是同序，本目录没做穷尽验证。
- **抓包体积**会随对话增长而线性膨胀；本流程要的是全量 body，
  31,600 条规模必须换成「只留最后一次请求」的录制策略，否则会写到几十 GB。
