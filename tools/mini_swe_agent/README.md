# mini-swe-agent → 本地 vLLM (glm5.3-flash)

本机已装好 mini-swe-agent，并把默认模型切到本地端点。

| 项 | 值 |
|---|---|
| mini-swe-agent | 2.4.6 (`uv tool install`，隔离环境) |
| litellm | 1.106.2 |
| 端点 | `http://YOUR-ENDPOINT:PORT/v1` (vLLM, 无鉴权) |
| 模型 | `glm5.3-flash`，`max_model_len=262144` |
| 可执行文件 | `~/.local/bin/mini`、`mini-e`、`mini-extra`、`mini-swe-agent`、`mini-glm` |

## 用法

```bash
# 在目标目录里跑（agent 的工作目录 = 你的当前目录）
cd /path/to/project
mini -t "把 tests/test_x.py 里的 flaky 断言修掉"          # 每条命令都问你要不要执行
mini -t "..." -y                                           # 无人值守
mini -t "..." -y --exit-immediately                        # 无人值守 + 结束时也不问
mini-glm -t "..."                                          # mini 的别名（同上配置）

# 轨迹默认写到：
#   ~/Library/Application Support/mini-swe-agent/last_mini_run.traj.json
# 指定路径：
mini -t "..." -o /tmp/run1.traj.json
```

`--exit-immediately` 在**非 TTY 环境**（CI、脚本、重定向 stdin）里是必需的：
否则 agent 想结束时会在 `confirm_exit` 提示处读到 EOF 而抛 `EOFError`。

## 配置文件

```
~/Library/Application Support/mini-swe-agent/.env     全局环境变量
tools/mini_swe_agent/overrides.yaml                   覆盖块（改这个）
tools/mini_swe_agent/local-glm.yaml                   生成物，别手改
tools/mini_swe_agent/regen.py                         重新生成 local-glm.yaml
```

以上路径相对于本仓库根目录（`~/Documents/CodesProject/harness-rsi`）。
全局 `.env` 里对应的两个变量已经指向仓库内：

```
MSWEA_CONFIG_DIR=<repo>/tools/mini_swe_agent
MSWEA_MINI_CONFIG_PATH=<repo>/tools/mini_swe_agent/local-glm.yaml
```

`local-glm.yaml` = 内置 `mini.yaml` + `overrides.yaml`，由 `regen.py` 生成。
mini-swe-agent 的默认配置文件是**整份替换**而不是合并，所以它必须自带完整模板。
升级 mini-swe-agent 后重新生成：

```bash
~/.local/share/uv/tools/mini-swe-agent/bin/python tools/mini_swe_agent/regen.py
```

`regen.py` 会在写出前断言 YAML 往返无损——长 Jinja 模板一旦被 YAML 折行，
提示词会被静默改掉，所以这一步不能省。

## 踩过的坑（已修）

1. **`ModuleNotFoundError: No module named 'mcp'`**
   litellm 在收到 `tools` 参数时会走 `litellm.responses.mcp.*` 的懒加载导入链，
   而 mini-swe-agent 的依赖里没有 `mcp`。只要用 toolcall 模式就必炸。
   → `uv tool install mini-swe-agent --with mcp`

2. **`chat_template_kwargs` 要放到 wire body 顶层**
   litellm 会把 `extra_body` 里的未知键合并到 body 顶层，所以
   `extra_body` 写法与顶层写法**完全等价**（受控实验：同一 prompt 各 3 次，
   reasoning 长度与 completion_tokens 逐条相同）。
   抓包实证该键确实出现在请求体顶层。
   → 配置里用 `extra_body: {chat_template_kwargs: {reasoning_effort: low}}`
   （早前"顶层会被丢弃"的说法是单样本比较得出的误判，已纠正）

3. **成本计算对所有模型都炸**
   mini-swe-agent 2.4.6 调 `completion_cost(response=...)`，
   但 litellm ≥1.106 的参数名是 `completion_response` → `TypeError`。
   → `cost_tracking: ignore_errors`（本地端点本来也不计费）

4. **litellm 每次启动要花 ~20s 去 GitHub 拉价格表**
   本机连不上 `raw.githubusercontent.com`，连超时 3 次才回落本地备份。
   → `LITELLM_LOCAL_MODEL_COST_MAP=True`

5. **首次运行会卡在交互式初始化向导**
   → `.env` 里 `MSWEA_CONFIGURED=true`

6. **`max_tokens` 给小了会死循环**
   GLM 先思考再答，思考就吃光 token → 回复被截断 → mini 判定 FormatError
   → 重试 → 又是 FormatError。`max_consecutive_format_errors: 3` 之后直接退出。
   → `max_tokens: 8192`，`reasoning_effort: low`

## 端点能力（已实测）

- 原生 OpenAI tool call：`tools=[{"type":"function","function":{"name":"bash",...}}]`
  能正确返回 `tool_calls` + `finish_reason="tool_calls"`，所以用的是官方推荐的
  toolcall 模式（`model_class: litellm`），不是退而求其次的 textbased 模式。
- 若日后服务端关掉了 `--enable-auto-tool-choice`，改用：
  ```yaml
  model_class: litellm_textbased
  model:
    action_regex: '```(?:mswea_bash_command|bash|sh|shell)?\s*\n(.*?)\n```'
  ```
  （textbased 默认只认 ` ```mswea_bash_command ` 围栏，非 Claude 模型一般吐
  ` ```bash `，不改会每步都 FormatError。）

## 冒烟测试记录

| 项 | 结果 |
|---|---|
| `hello.txt` 任务 | 3 次 API 调用，`exit_status=Submitted`，文件 23 字节，`od -c` 字节级校验通过 |
| `data.json` 任务（裸 `mini`） | 2 次 API 调用，`exit_status=Submitted`，JSON 合法且无换行 |
| 格式错误 | 0 |
