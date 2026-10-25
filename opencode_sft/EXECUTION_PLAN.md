# 执行计划：两个 harness × 两个数据集

## 目标

把 **Mini** 和 **Opencode** 两个 harness 都用起来，在 **OpenThoughts** 和 **ClawGym** 两个数据集上都产出 harness-agent 模式的 SFT 轨迹。

**模型只有 GLM-5.3-Flash 一个。** 曾接入过 MiniMax-M3.1-Flash 作为第二个标注模型（`opencode_sft/minimax.py` 与两个 driver 的 `--model-id` / `--api-base` 开关都还在），但最终决定停用：全部轨迹由 GLM 产出。停用的原因与经过见下面「第二模型：接入过，又撤回」。

**切分规则**

| 数据集 | 切分粒度 | Mini 拿 | Opencode 拿 |
|---|---|---|---|
| ClawGym | 全部 15,402 个任务 | 一半（7,749） | 一半（7,653） |
| OpenThoughts | 每个家族（issue 8,374 / tezos 7,197 / superuser 7,654） | 每家族一半 | 每家族一半 |

切分必须**确定性**（同一个 task 名永远落在同一侧），且**与行序无关**（ClawGym 前 6,094 行全是 arc 家族，按行序切会切出两个同质集合）。实现：按 task 名的 sha256 取模。

**一个镜像服务两个 harness。** mini-swe-agent 不在镜像里——它是 `.cache/rsi-local` 这个 toolchain bundle **bind-mount** 进容器的（`/root/.local`）。opencode 是镜像里烤进去的二进制。所以：

* 镜像 = ubuntu 24.04 + python 3.12 + PyYAML + build-essential + openclaw workspace 骨架 + **opencode 二进制 1.18.33**；
* Mini 侧：把 toolchain bundle 挂进去即可，镜像不需要 mini；
* Opencode 侧：镜像自带 opencode。

这样两边跑在**完全相同的环境**里，harness 是唯一变量。

---

## 第二模型：接入过，又撤回

**接入了什么。** `opencode_sft/minimax.py` 提供 MiniMax-M3.1-Flash 的 OpenAI 兼容端点（`https://api.minimax.cn/v1`）；`split_tasks.py --minimax-share` 按 task 名哈希切出每个 harness 半边里 30% 的任务给 MiniMax，GLM 流用 `--exclude-manifest` 跳过同一批，于是「四股流」是 mini-glm / mini-mm / oc-glm / oc-mm。

**验证到什么程度。** 两个 harness × MiniMax 都实跑通了：

| 验证项 | 结果 |
|---|---|
| MiniMax × opencode 试点 | 4/4 跑通，reward 1.0 / 1.0 / 0.74 / 1.0，转换 + 校验全过 |
| MiniMax × mini 试点 | 跑通（step 4 已在正常调用 bash 工具），产出 `dataset=MiniMax-M3.1-Flash, harness=Mini` 的行 |
| MiniMax 对 litellm/openai 风格参数 | `tools` / `seed` / `max_tokens` / `extra_body.chat_template_kwargs` 全部接受 |
| MiniMax 的响应形状 | `tool_calls[].function.arguments` 是 JSON 字符串；**没有** `reasoning_content` 字段 |
| 单次调用延迟 | 请求间隔 2–27 s（GLM 的 p90 是 32.6 s、max 235 s），MiniMax 明显更快 |

**为什么撤回。** GLM 端点负载高是切出第二个模型的初衷，但两条流并行会让宿主 CPU 和 Docker 的 10 GB 内存都逼近上限，而 GLM 独占一条流时吞吐并没有被端点卡住。所以最后只保留 GLM。

**撤回时踩到的坑（重要）。** 第一版编排脚本给 GLM 流加了 `--exclude-manifest manifest-<side>-minimax.jsonl`，意思是「GLM 流跳过 MiniMax 那 30%」。撤回 MiniMax 时如果只是删掉那两条流、忘了删 `--exclude-manifest`，那 30% 的任务会被**两个流同时跳过**——永久丢失，而且没有任何报错。所以撤回的正确做法是：**GLM 流必须去掉 `--exclude-manifest`**，让它把整份 manifest 都接过来。已改：两条 GLM 流现在都跑全量 manifest。

**留在树里的东西。** `opencode_sft/minimax.py`、`--model-id` / `--api-base` / `--api-key-file` 开关、`--minimax-share` 都在，将来要再切一个模型回来是改命令行的事。但**当前编排脚本里没有它们**：`opencode_sft/run_all_stages.sh` 只有 mini-glm 和 oc-glm 两条流。

---

## 阶段 0：基础设施（已完成）

- [x] ClawGym 数据勘探：15,402 条、两组（arc 6,094 + persona 9,308）、verifier 契约由 `hook_code` 完全确定
- [x] 全量 verifier 探针：15,402 个 verifier 全部跑通，输出形状分两类（`reward` 6,091 / `flat` 9,302 / `none` 9）
- [x] ClawGym registry：15,402 个任务目录、61,111 个文件、共享网络 + payload 挂载
- [x] ClawGym 镜像 `rsi/oc-base-clawgym:1`（834 MB，含 opencode 1.18.33）
- [x] 修复：toolchain bundle 被容器内 uv 安装写坏
- [x] 修复：Docker 地址池耗尽（共享网络未生效，全批失败）
- [x] 修复：JSONL 行分隔符（`splitlines()` 会按 U+2028 截断记录）
- [x] wire 逐键 diff（Mini）：trajectory = wire + `{extra, function_call}`，零差异
- [x] wire 逐键 diff（Opencode）：重建行 == 最后一次真实请求 + 1 条最终 assistant，**4/4 逐字节一致**
- [x] SFT 独立校验器 + 11 项故障注入全部捕获（两个 harness 各跑一遍）
- [x] OpenThoughts 三家族重建镜像后冒烟 6/6
- [x] **切分器** `opencode_sft/split_tasks.py`：按 task 名哈希切两份 manifest
- [x] **两个 driver 支持 `--manifest`**
- [x] **opencode driver 支持 `--enable-verifier`**（ClawGym 要打分）
- [x] **opencode driver 支持可配置 baseURL**
- [x] **opencode 转换器加 reward 过滤 + reward 篡改过滤 + harness 字段**
- [x] **录制代理 blob 模式**：system prompt / tool schema 内容寻址去重
- [x] **合并脚本** `opencode_sft/clawgym/merge_sft.py`

---

## 阶段 1：ClawGym（先做，15,402 条）

**只有两条流，都是 GLM。** mini 拿 7,749、opencode 拿 7,653，各自跑**全量**自己的 manifest（没有 exclude）。

### 1a. Mini 半边 —— 进行中

- [x] 用切分后的 Mini manifest 重启 driver（**只跑属于 Mini 的那一半**）
- [x] 并发 24，跑满（**进行中**：run-root `/private/tmp/rsi-harbor/clawgym`）
- [x] `build_sft_dataset.py`（Mini 版）+ `validate_sft.py` 已跑通（76 条抽样 → OK）
- [ ] 跑满后全量转换 + 全量校验

### 1b. Opencode 半边 —— 进行中

- [x] 用切分后的 Opencode manifest 跑 opencode driver（**进行中**：run-root `/private/tmp/rsi-harbor/clawgym-opencode`）
- [x] opencode 的轨迹来源是 `agent/opencode/session.json.gz`（不是 mini 的 trajectory.json）
- [x] opencode 一次回复可并行发多个工具 → 工具结果在 session 里挂在**同一个 assistant 消息**的 part 上，
      转换时一个 assistant turn 的多个结果各自成为一条 `role: "tool"` 消息（连续 tool 消息是合法的）；
      严格 user/assistant 交替的 `conversations` 视图把它们**合并成一条 user 观测**
- [x] `build_opencode_sft_dataset.py`（opencode 版）+ 同一套校验规则
- [x] 试点 4/4 成功、mean reward 0.917、转换 + 校验 + wire 逐键验证全过
- [x] opencode driver 补上 `--shuffle-seed` / `--resync-ledger`，ledger 记 `model` / `provider`

**两个 harness 并行跑**：Mini 并发 24、Opencode 并发 10。理由：端点有余量（并发 16 时 p50 1.5s），
内存有余量（opencode 容器 ~500 MB，mini ~100 MB，Docker 上限 9.7 GB），瓶颈在宿主的 compose 进程 churn 而不是端点。

### 1c. 合并

- [x] 合并脚本 `merge_sft.py`：拼接 + 检查「两边不重不漏」「每行都有 harness」「字段齐全」，原子写入
- [ ] 两边各出一份 SFT，合并成 `clawgym-sft.jsonl`
- [ ] 全量校验 + 故障注入复跑

**验收口径**

| 指标 | Mini | Opencode |
|---|---|---|
| 成功率 | ≥ 95% | ≥ 95% |
| reward > 0 占比 | ≥ 85% | ≥ 85% |
| reward = 1.0 占比 | ≥ 35% | 待测 |
| 触碰 reward/ 目录的占比 | 报告（实测 26%） | 报告（试点 1/4） |
| 平均 assistant 轮数 | 报告 | 报告 |

---

## 阶段 2：OpenThoughts 三家族（ClawGym 完成后）

每个家族 7,197–8,374 条，切成两半：

- [ ] issue：一半 Mini / 一半 Opencode
- [ ] tezos：一半 Mini / 一半 Opencode
- [ ] superuser：一半 Mini / 一半 Opencode

**注意**：OpenThoughts 的原始轨迹是 **terminus-2** harness 产出的，不是 opencode。所以这条线是「用 Mini / 用 Opencode 重新执行同一批任务」，产出的是**新 harness 视角**的轨迹；原轨迹是参照物，不是 ground truth（ground truth 是任务本身）。

**这一步特有的坑**

- [ ] **step_limit 必须是 30**。mini 的默认值是 `step_limit: 0`（不限）+ `cost_limit: 3.0`，而 litellm 在这个网络下拉不到价格表 → 成本恒为 0 → 护栏永不触发 → agent 一直跑到超时。实测：不限步时 p50 45 轮 / p90 75 轮，100/100 撞 1800s 超时。
- [ ] **build-essential 必须烤进镜像**。Harbor 的 mini adapter 要 `("curl","bash","build_tools","git")`，缺 gcc/make 就在每个 trial 里现装，Debian 源 0.18 MB/s → trial 起不来。
- [ ] **原轨迹的环境指纹要复原**。issue 家族是 Debian 13 trixie + /usr/local 的 python 3.10（不是 bookworm、不是 ubuntu）；tezos/superuser 是 ubuntu 24.04 + python 3.12。已分别建成 `rsi/oc-base-issue-py310:1` / `rsi/oc-base-plain:1`。
- [ ] **不要动 Opencode 的 MCP 全局配置**。本机全局 config 开着 chrome-devtools MCP，会把 42 个工具塞进每个请求；容器里要用自带 config 的干净 10 工具。
- [ ] **opencode 的 `bash` 工具不是持久 shell**：每次调用都是新进程，`cd` / `export` 不跨调用保留（用 `workdir` 参数控制目录）。描述里写的是 "persistent shell session"，是假的。
- [ ] **opencode 一次回复可并行发 3 个工具**，工具结果各占一条 user 消息 → 连续 user 会破坏严格的 user/assistant 交替不变量。转换器要按 opencode 的实际形状处理，不能照抄 mini 的。

---

## 阶段 3：交付

- [ ] `clawgym-sft.jsonl`（每行带 `dataset` / `model` / `harness` 三个键）
- [ ] `openthought-sft.jsonl`（三家族 × 两个 harness）
- [ ] 校验报告：每份数据的行数、过滤计数、reward 分布、失败样本清单
- [ ] 明确报告**未通过 / 不可用**的样本
- [x] 被丢弃的样本**不删除**，归档到 `invalid/`（数据集旁），另出 `.skipped.jsonl` 原因清单

---

## 关键教训（每一条都是踩出来的）

| 坑 | 后果 | 防呆 |
|---|---|---|
| Docker 地址池只有 16 个 | 全批 trial 起不来，报错像 agent 失败 | 所有 task 带 `environment/docker-compose.yaml` 声明共享网络（`external: true`） |
| toolchain bundle 是 bind-mount，uv 会往里写 | 一个 trial 装不上就把 bundle 写坏，后续全批走慢速安装 | adapter 把 uv 的目录重定向到 `/tmp/rsi-uv`；driver 开跑前预检 bundle |
| `str.splitlines()` 会按 U+2028 / \x0b 断行 | 一条 JSON 记录被读成两半 | 写的时候转义这些字符；读的时候只按 `\n` 切；数据集用临时文件 + rename 原子替换 |
| 端点在并发 32 时 p50 6.4s 但有 65s 长尾 | 长尾任务撞 1800s 超时 | 并发 24，step_limit 30 |
| 任务提示里写着 "Do not write under reward/" | 说明 verifier 对 agent 可见，agent 会去读评分器 | 转换阶段过滤：assistant 轮里碰过 `reward/` 的轨迹一律剔除 |
| `mini -c` 是整份替换 | 配置悄悄丢了 system/instance template | `-c local.yaml -c 覆盖层.yaml` 全量带上 |
| Harbor 输出目录在 macOS 只能是 `/private/tmp` | 项目目录和 `/tmp` 都不行 | run-root 固定 `/private/tmp/rsi-harbor/...` |
| Dockerfile 里 `ARG` 放在 apt 层之前 | ARG 改掉后面所有层的 cache key → apt 层整个重下 → 撞上镜像源瞬时故障 | `ARG` 声明放在用到它的那一层前面 |
| opencode 的 `tool_calls.arguments` 是 JS `JSON.stringify` | Python `json.dumps` 默认在 `:` `,` 后加空格 → 语义等价但 token 序列不同，训练数据不是模型见过的字节 | `separators=(",", ":")` |
| opencode 的 reasoning 是独立字段 | 折进 content 会让样本「读起来一样」但 token 不一样 | `messages` 视图保留 `reasoning_content`；`conversations` 视图才折成 `<think>` |
| opencode 的 system prompt 80 KB / 每条 | 全量抓包要几十 GB | 代理的 blob 模式：内容寻址存一份，日志里只留 120 字节的引用 |
| mini 的 `--api-base` 只写进宿主环境 | 容器里的 `model.model_kwargs.api_base` 没变 → **MiniMax 流静默跑成 GLM**，ledger 却写着 MiniMax | driver 把 `--api-base` 写进 config 的 `model_kwargs`（config 会进 job 文件进容器）；另外加一道硬检查：远程端点 + `--api-key-file` 直接报错退出 |
| mini 在容器里跑，宿主的 `OPENAI_API_KEY` 它看不见 | 密钥要么进 job 文件（落盘、进日志），要么根本到不了 agent | 密钥由录制代理在转发时覆盖注入；job 文件里只有占位符 `EMPTY` |
| 停用第二个模型时忘了删 `--exclude-manifest` | 被排除的那批任务**两条流都不跑**，静默丢失 | 撤回第二模型 = 删流 + **删 exclude**，让 GLM 流接全量 manifest |
| driver 的 `--shuffle-seed` 只有 mini 有、opencode 没有 | 编排脚本给 opencode 传了这个参数 → 两条 opencode 流**启动即崩**（argparse 报错），而监管循环会反复重启它们 | 两个 driver 的参数集对齐；开跑前先 `--dry-run` 验参数 |
| 监管循环逐个 `wait` | 一条流崩了，要等最慢那条跑完（好几天）才会被重启 | 改成轮询所有流：谁没在跑且没有 `COMPLETE` 标记就立刻重启 |
| `kill -0` 对**僵尸进程**也成功 | 崩掉的 driver 被当成「还在跑」，监管循环永远不会重启它 | 先读 `ps -o state=`，`Z` 就算已退出，并顺手 `wait` 回收 |
| 一条 batch 只在**整批跑完后**才写 ledger | 中途重启会把这批已完成的 trial 全重跑一遍 | 加重试前从 `result.json` 反推 ledger（`--resync-ledger`），两个 driver 都支持 |
| mini 并行工具调用 → 一个 assistant 带 N 个 tool_call | 转换时只删最后一条尾部 observation，其余 N-1 条留下 → tool_call 悬空 | 删**全部**尾部 observation；改完校验器立刻抓到 |
| Harbor 的 job 是 driver 的**子进程**，不会随 driver 一起死 | driver 被杀后旧 job 继续起容器（实测 3 次重启后 6 个 job 同时在跑、82 个容器、占掉 8.4/9.7 GB 内存） | driver 开新批前先 `kill_stale_jobs(run_root)`（SIGKILL，Harbor 忽略 SIGTERM）+ `remove_stale_containers(run_root, batch_dir)`；范围只限本 run-root，不碰并发的另一条流 |
| 每个 trial 结束 Harbor 会 `docker compose down` | —— | 正常路径容器是**跑完即释放**的；堆积只来自被杀掉的 job，不需要额外回收逻辑 |
| `cmd &` 跑 **shell 函数** 时 `$!` 是中间 bash 的 pid | 监管脚本「杀 driver」杀的是 wrapper，真正的 python driver 被 reparent 到 init 继续跑 → 同一流**同时跑了 3 份**，各起一个 harbor job，容器和内存翻 3 倍 | 函数体最后一条命令加 `exec`，让 subshell 被 python **替换**，`$!` 就是 driver 自己的 pid |
| 判进程归属不能只做子串匹配 | `clawgym` 是 `clawgym-opencode` 的子串 → mini 的 driver 会误杀 opencode 正在跑的 harbor job，opencode 的 trial 全部 exit 137（看起来像 agent 崩了） | 取 `-c` 后面的 job 文件路径做**路径前缀**比较（`<run-root>/` 开头才算） |
| 判「是不是 harbor 进程」不能拿 `argv[0]` | 真实命令是 `python /…/.venv312/bin/harbor job start …`，`argv[0]` 是 **Python 解释器** → 精确匹配 "harbor" 永远不成立，清理函数静默失效 | 在 argv 列表里找 base name 为 `harbor` 的那一项 |
