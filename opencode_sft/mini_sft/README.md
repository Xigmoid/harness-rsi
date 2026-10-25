# mini_sft — 用 mini-swe-agent 重跑 OpenThoughts-Agent-SFT-31.6K

用 **mini-swe-agent** 作 harness，把
[open-thoughts/OpenThoughts-Agent-SFT-31.6K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-31.6K)
的 31,600 条任务重跑一遍，产出 **OpenAI API 兼容**的 SFT 轨迹数据集。

原数据集是 **terminus-2** 跑的（`agent` 字段），首个 `user` 消息里把
system prompt 和任务指令合在了一起；这里换成 mini-swe-agent，轨迹里
system 与 user 分开，assistant 带原生 OpenAI `tool_calls`。

```
数据集 (parquet, 587 MB, 31,600 行)
    │  extract_user_instructions.py          ① 抽出全部用户指令
    ▼
tasks/user_instructions.jsonl                31,600 条，0 失败
    │  run_mini_rollouts.py                  ② 在 docker 里逐条跑 mini
    ▼
/private/tmp/rsi-harbor/mini-sft-full/       trial 目录 + progress.jsonl
    │  build_mini_sft_dataset.py             ③ 转成 OpenAI 格式
    ▼
mini_sft/dataset/mini-sft.jsonl              每行一条 rollout，messages + conversations 双视图
```

---

## ① 提取全部用户指令

```bash
python opencode_sft/extract_user_instructions.py
```

产出 `opencode_sft/tasks/user_instructions.jsonl`（31,600 行，123.7 MB）：
每行对应**数据集的一行**，带 `instruction` 全文、任务元信息、以及切分用的
`instruction_sha256` / `template_sha256`。

切分规则（与 `extract_tasks.py` 完全一致，指令字节级相同）：

```
<terminus-2 system prompt>
...
Task Description:
<任务指令>            ← 这一段就是 instruction
Current terminal state:
Current Terminal Screen:
<终端屏幕快照>
```

**校验**：脚本拿既有的 `tasks/index.jsonl` 当独立 oracle 对账，
31,600 行**全部匹配，0 失败**。

> 为什么按「数据集行」而不是「唯一任务」统计：有 766 个任务带两个指令变体，
> 数据集的不同行指向不同变体。行才是与已发布轨迹对齐的单位。

---

## ② 在 docker 里跑 mini

### 配置

```bash
python opencode_sft/build_mini_config.py --protocol toolcall
```

产出 `opencode_sft/mini_sft/mini-config.yaml`——**自包含**的 mini 配置：
系统模板、实例模板、观测模板、格式错误模板全部在里面。

三个关键决定：

| 决定 | 原因 |
|---|---|
| 模板取自容器内 `rsi/toolchain:2` 的 `mini.yaml` |  Trials 真正执行的就是这个镜像里的 mini.yaml；用宿主机的副本会让提示词漂移 |
| 协议 = **toolcall**（`model_class: litellm`） | 原生 OpenAI tool call，无需解析围栏；工具链加了 `--with mcp` 后在 Harbor 里可用 |
| `environment.cwd` 留空 | cwd 随 family 变（`/testbed` / `/app`），但它就是镜像的 WORKDIR；mini 的 local 环境在 `cwd` 为空时回落到 `os.getcwd()` |

### 跑

```bash
# 干跑：只报告要做什么
python opencode_sft/run_mini_rollouts.py --dry-run

# 小批量验证
python opencode_sft/run_mini_rollouts.py --batch-size 4 --concurrency 4 --max-batches 1

# 全量
python opencode_sft/run_mini_rollouts.py --batch-size 200 --concurrency 8
```

- 任务来自 `opencode_sft/registry/`（31,600 个，与数据集逐行对应）
- 断点续跑靠 `progress.jsonl`：`ok: false` 的记录不算完成，会在后续批次重试
- `--shard i/n` 可以分机（分片键是任务名的稳定哈希，不会重复）
- Harbor 的输出必须在可挂载路径上（macOS 上是 `/private/tmp`，不是项目目录、也不是 `/tmp`）

### 工具链

Harbor 默认会在**每个 trial 容器里**装一遍 mini-swe-agent（~4 分钟/次）。
`rsi/toolchain:2` 把它预构建成一个 bundle，挂在 `/root/.local`，
`RsiMiniSweAgent` 检测到就直接跳过安装。

bundle 里带了 `mcp`：

```
uv tool install --python 3.12 mini-swe-agent==2.4.6 \
    --with litellm --with orjson --with fastapi --with pyyaml --with mcp
```

这不是可选的：litellm 在 `tools` 被设置时**无条件**导入它的 MCP handler，
缺包会让每一个请求死在 `ModuleNotFoundError: No module named 'mcp'`。
以前框架因此被迫钉死文本协议；有了这个包，工具调用协议在 Harbor 里也能用。

重建：

```bash
bash scripts/rebuild_toolchain_with_mcp.sh
```

---

## ③ 转成 OpenAI API 兼容的 SFT 轨迹

```bash
python opencode_sft/build_mini_sft_dataset.py \
    --run-root /private/tmp/rsi-harbor/mini-sft-full \
    --out opencode_sft/mini_sft/dataset/mini-sft.jsonl
```

每行一条 rollout，**两个视图**：

| 视图 | 用途 | schema |
|---|---|---|
| `messages` | tool-aware trainer（verl / TRL 等） | `role` ∈ {system,user,assistant,tool}；assistant 带 `tool_calls`；观测是 `role:"tool"` + `tool_call_id` |
| `conversations` | 对齐已发布数据集 | 键只有 `content`/`role`，role 只有 `user`/`assistant`；system prompt 折进首条 user（terminus-2 就是这么放的） |

转换只做「剥壳」，不重写：去掉 mini 的两个内部键（`extra`、`function_call`）、
去掉 litellm 序列化时本来就会省略的 `None` 字段、去掉 `role:"exit"` 的
记账消息，以及提交指令那条**从未执行**的观测（`exception_info ==
"action was not executed"`）——模型没收到过它。

---

## ④ 证明数据是真的：与 wire 逐字比对

harness 记的轨迹是不是就是模型收到的？这要抓包验证，不能推断。

```bash
# 1. 起一个独立端口的抓包代理
python3 opencode_sft/record_proxy.py --port 8901 --host 127.0.0.1 \
    --upstream http://YOUR-ENDPOINT:PORT/v1 --out /tmp/mini-wire.jsonl &

# 2. 让几个 trial 走代理（配置指向 host.docker.internal:8901）
python opencode_sft/build_mini_config.py --protocol toolcall \
    --api-base http://host.docker.internal:8901/v1 \
    --out opencode_sft/mini_sft/mini-config-proxy.yaml
python opencode_sft/run_mini_rollouts.py \
    --min-config opencode_sft/mini_sft/mini-config-proxy.yaml \
    --tasks swesmith-00000_copy0001 --batch-size 1 --max-batches 1

# 3. 比对
python opencode_sft/verify_mini_wire.py --capture /tmp/mini-wire.jsonl
```

判定规则（与 pilot 一致，判定函数直接从 pilot 导入，两边不会漂）：

> 转换结果的前 N 条 == 该 rollout **最后一次**请求的 N 条消息（逐字），
> 且转换结果比它多出**恰好 1 条**——那一条是模型最后一次输出，
> 从未被发回去过。

---

## 环境保真度：四个家族不是同一种环境

re-annotation 要和原轨迹同分布，环境就得对齐。**「Task Description:」只给了任务，
没给环境**——环境是从原轨迹的观测里反推出来的。

从数据集的对话内容里能读出四个家族的环境指纹：

| family | 数量 | apt 源 | python | 真实环境 | 镜像 |
|---|---|---|---|---|---|
| swesmith | 8,375 | — | conda `testbed` 3.10.16/3.10.18/3.11.11/3.11.13 | repo checkout | `oc-base-swesmith-py:1` + 运行时 checkout |
| **issue** | 8,374 | **`deb.debian.org`** ×2615 | **`/usr/local/bin/python3.10` 3.10.20** | **Debian 13 trixie + 源码编译的 python 3.10** | `oc-base-issue-py310:1` |
| tezos | 7,197 | `archive.ubuntu.com` ×5307 | 系统 python 3.12.3 | Ubuntu 24.04 | `oc-base-tezos:1` ✓ |
| superuser | 7,654 | `archive.ubuntu.com` ×7895 | 系统 python 3.12.3 | Ubuntu 24.04 | `oc-base-superuser:1` ✓ |

关键区分：`/usr/local/bin/python3.10` 是**源码编译装进 /usr/local** 的 python
（官方 `python:3.10` 镜像就是这样），而 ubuntu 的 `python3` 包装在 `/usr/bin`。
配合 `deb.debian.org` 这个 apt 源，issue 家族的原容器是 **Debian 系**，
和 tezos/superuser 的 **Ubuntu 24.04** 不是一回事。

**Debian 的 suite 也是指纹的一部分，而且很容易看错。** 原轨迹里的
`apt-get update` 输出是 `Hit:1 http://deb.debian.org/debian trixie InRelease`
——trixie（Debian 13），不是 bookworm。第一版镜像用的是
`python:3.10-slim-bookworm`，agent 看到的会是 `bookworm InRelease`，
和原轨迹对不上。现在换成 `python:3.10-slim-trixie`
（它和 floating 的 `3.10-slim` 是同一个 digest）。

顺带一个容易踩的坑：issue 家族的原轨迹里 **`/app` 起始就是空的**
（`ls -la /app/` 只有 `.` 和 `..`），agent 是从零把文件写出来的，
不是「环境缺失、repo 没 checkout」。镜像必须保留这个空目录语义。

所以 issue 家族**不能**用 plain 的 ubuntu 镜像——否则 agent 看到的
`python3 --version`、pytest 的 `platform linux -- Python 3.10.x` 横幅、
apt 源全都不一样，训出来的数据分布会偏。它有自己的镜像：

```
opencode_sft/images/oc-base-issue-py310.Dockerfile   ->  rsi/oc-base-issue-py310:1
```

point release 有漂移（原版 3.10.20，tag 现在解析到 3.10.22），
和 plain 家族 24.04.3→24.04.5 的漂移同理；精确 tag 在这台机器上拉不下来
（`registry-1.docker.io` TLS 握手超时，`ubuntu:24.04.4` 同样拉不下来）。

**构建期换镜像源、运行期换回去。** 容器里 `deb.debian.org` 只有 0.18 MB/s
（构建慢到几十分钟），而 `mirrors.tuna.tsinghua.edu.cn` 有 0.74 MB/s。
但换成清华源就等于改了环境：原轨迹里 agent 自己的 `apt-get update`
命中的是 `deb.debian.org`，agent 看到的主机名必须一致。
所以 Dockerfile 先把源换成清华下完包，**再把
`/etc/apt/sources.list.d/debian.sources` 还原成 Debian 原样**，
并且 `grep` 断言还原成功之后才算构建通过。

**旧镜像跑出来的 issue 批次已隔离。** 第一批 issue trial 跑在
`rsi/oc-base-issue:1`（Ubuntu 24.04 + python 3.12）上——发行版就不对。
这些 trial 目录已移到 `_stale-issue-ubuntu/`，不进 ledger，也不进数据集。

**三个家族都不需要 per-trial 建虚拟环境**：

| 层次 | 状态 |
|---|---|
| 任务环境 | 预构建基础镜像（issue/tezos/superuser 各自一个），`/app` 起始为空——与原轨迹一致，agent 自己在里面装依赖 |
| agent 工具链 | `rsi/toolchain:2` bundle：linux/aarch64 的 uv venv，**一次构建、bind-mount 复用**给所有 trial |
| per-task setup | 这三个家族的 `environment/` 目录**是空的**——零 setup |

只有 swesmith 需要 per-task 环境构建（checkout 特定 commit + 建 conda `testbed` venv），
它靠 150 个 SWE-Smith 镜像仓库（`build_mirror_cache.py`），
一次 clone、全部 trial 共享对象库。

---

## 踩过的坑

1. **`mini -c` 是整份替换** —— 一旦给了 `-c`，打包的 `config/mini.yaml` 就不再用。
   配置必须自带系统/实例模板，否则 agent 拿到空提示词。
2. **冻结的 `observation_template` 在 plain mini 下会炸** ——
   `harness/base_config.yaml` 里那份模板插值 `observation_max_chars`，
   那是 `RhiAgent` 才提供的变量。plain mini 第一次工具调用就
   `UndefinedError: 'observation_max_chars' is undefined`。
   协议（围栏 + action_regex）保留，模板换成 mini 自带的那套。
3. **文本协议下模型会漏 tool-call 残渣** —— GLM 是带 tool call 训过的，
   在围栏协议下偶尔吐出 `</arg_value>` 这类残渣，还会把命令写坏。
   原生 tool call 协议没有这个问题。
4. **Harbor 不装 `mcp`** —— 所以框架以前只能钉文本协议。加进 bundle 即可。
5. **`config` 选项只接受 mapping** —— `config_file` 不在选项模型里，
   想用文件得自己把内容读进来。
6. **Harbor 输出目录不能是项目目录** —— macOS 上 Docker 挂载会卡住；
   `/tmp` 是 `/private` 的软链，也会挂不上。只能用 `/private/tmp`。
7. **每个 trial 泄漏一个 compose 网络** —— Docker 默认地址池只有 16 个，
   漏光之后每个新 trial 都会死在
   `all predefined address pools have been fully subnetted`，
   看起来像 agent 失败，其实是环境层失败。所有 trial 共用一个外部网络。
8. **`docker pull` 在这台机器上基本不work** —— Docker Desktop 的 VM 到不了宿主
   `127.0.0.1:12000` 的代理（那个代理只监听 loopback），直连
   `registry-1.docker.io` 又是 TLS 握手超时。`docker build`（buildkit）同样卡在
   registry 解析。绕过办法写在 `opencode_sft/fetch_image_via_registry.py`：
   **在容器里**（容器没有代理，走的是直连；到 Docker Hub 镜像源只要 0.2s）
   调 registry API 抓 layer，打成 docker-archive，再在宿主上 `docker load`。
   两个细节：镜像源要用**它自己的 realm** 取 token（daocloud 是
   `https://m.daocloud.io/auth/token?service=docker.m.daocloud.io`，拿
   Docker Hub 的 token 会 401）；blob 的预签名重定向 urllib 处理不了（403），
   得用 curl 两步式。
9. **ledger 按批次落盘会丢进度** —— 一批 100 条要跑几十分钟，中途一停，
   整批已完成的 trial 全部作废。现在 `run_mini_rollouts.py` **逐 trial 落盘**
   （harbor 一边跑，驱动一边收割 `result.json`），并提供 `--resync-ledger`：
   启动时从磁盘上的 trial 目录回补 ledger，重启不再重跑。
10. **litellm 每次启动都要等一个 TLS 超时** —— 它会从
    `raw.githubusercontent.com` 拉价格表，这台机器连不上，只能等超时再退回本地副本。
    设 `LITELLM_LOCAL_MODEL_COST_MAP=True`（加在 toolchain 的 shell prelude 里，
    和 mini 同一条命令行）：容器内实测 `import litellm` 从 **9.32s → 4.40s**。
    价格表只喂成本统计，轨迹里本来就记 `instance_cost: 0.0`，模型看到的东西
    一个字没变。

## 跑多快

单 trial 时长（`result.json` 的 started_at→finished_at）：

| 并发 | p50 | p90 | 实测吞吐 |
|---|---|---|---|
| 8 | ~155s | — | ~1.5 条/分钟 |
| 16 | 204s | 584s | **3.27 条/分钟**（batch-0008 的 100 条 / 1834s） |

时长的尾巴比中位数重要得多：100 条里有 71 条在 5 分钟内跑完，
20 条要 5–10 分钟，还有 3 条撞上 1800s 的 agent 超时（这 3 条算失败，
会重试）。**按 p50 估吞吐会乐观一倍**，要按批次的总墙钟算。

瓶颈不在端点：并发 16/24 的端点压测 p50 是 1.4–6.9s、0 错误，
真实负载下单个请求仍是 0.7–1.2s；容器 CPU 占用多在 0.3%。
步与步之间的间隔从并发 8 的 p50 4.2s 只涨到并发 16 的 5.0s，
所以 16 槽滚动时单 trial 变长（155s→204s）不是端点的锅，
是 16 个容器共享 10 个 CPU，命令执行本身变慢了。

按 16 并发 3.27 条/分钟算，三个家族 23,225 条约需 **4.9 天**；
swesmith 家族另有 8,375 条（约 1.8 天），按指示暂缓。
