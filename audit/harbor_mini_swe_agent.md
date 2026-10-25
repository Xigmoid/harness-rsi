# Audit: what Harbor injects into mini-swe-agent

**Scope.** The field-by-field inventory of everything Harbor 0.24.0 adds to a
mini-swe-agent run, and of which of those additions can change the behaviour
of the harness.  This is the "W1 / constraint C-4" deliverable of
`02-coding-plan.md`: *"产出 `audit/harbor_mini_swe_agent.md`:Harbor 注入内容的逐字段清单"*.

**Sources.** `harbor-framework/harbor` @ `0.24.0`, file
`src/harbor/agents/installed/mini_swe_agent.py` (the adapter) and
`src/harbor/agents/base.py` + `src/harbor/agents/agent_auth_resolution.py`
(connection resolution); `SWE-agent/mini-swe-agent` @ `2.4.6`, file
`src/minisweagent/config/__init__.py` (`get_config_from_spec`) and
`src/minisweagent/config/mini.yaml` (the packaged base config).

---

## 1. How the config is assembled

Harbor writes the generation's config into the container as
`/tmp/mswea-config/custom.yaml` and then assembles a **list of `-c` specs**:

```
mini-swe-agent --yolo --model=<model> --task=<instruction> --output=<traj> \
    -c mini \
    -c model.model_kwargs.timeout=3600 \
    -c /tmp/mswea-config/custom.yaml \
    [-c model.litellm_model_registry=<path>] \
    [-c model.model_kwargs.extra_headers.X-Session-ID=<id>] \
    [-c model.model_kwargs.reasoning_effort=<effort>] \
    [-c model.model_kwargs.max_tokens=<n>] \
    [-c "model.model_kwargs.api_key=$HARBOR_MODEL_API_KEY_JSON"] \
    [-c model.model_kwargs.custom_llm_provider=<provider>] \
    [-c model.model_class=<litellm|litellm_response>] \
    [-c model.model_kwargs.api_base=<url>] \
    --exit-immediately
```

mini-swe-agent's `get_config_from_spec` loads each spec and merges them
**recursively, in order** — later specs win.  Two consequences:

1. `-c mini` is prepended, so the packaged `config/mini.yaml` is always the
   base layer.  Everything in `mini.yaml` is therefore *inherited* unless the
   generation's config overrides it.
2. **Every `-c` flag that appears after `/tmp/mswea-config/custom.yaml`
   overrides the generation's config.**  The flags below are therefore not
   neutral: they are a second, invisible layer of the experiment's
   independent variable.

## 2. The base layer that always leaks in (`-c mini`)

`mini.yaml` (mini-swe-agent 2.4.6) contributes:

| key | value in `mini.yaml` | risk for this study |
|---|---|---|
| `agent.system_template` | the single-agent prompt | **high** — must be overridden, otherwise the RHI prompts never render |
| `agent.instance_template` | the single-agent task prompt | **high** — same |
| `agent.step_limit` | `0` (unlimited) | low |
| `agent.cost_limit` | `3.0` | low, but it is a hard stop → must be explicit |
| `agent.mode` | `confirm` | none (only read by the interactive agent) |
| `model.observation_template` | the **tool-call** flavoured template | **high** — with a text-based model it renders a "bash tool" instruction |
| `model.format_error_template` | the **tool-call** flavoured template | **high** — same |
| `model.model_kwargs.drop_params` | `true` | none |
| `environment.env` | `PAGER`, `MANPAGER`, `LESS`, `PIP_PROGRESS_BAR`, `TQDM_DISABLE` | none |

`rsi/adapter.py` writes **all** of these explicitly (`system_template` and
`instance_template` become the `__RHI_RENDERED__` placeholder, the templates
are replaced by the frozen gen-0 copies).  `assert_full_expansion()` fails the
build if a key of the frozen base config is missing from the generated
config.

## 3. What Harbor injects *after* the generation's config

| # | injection | condition | what it can change | mitigation |
|---|---|---|---|---|
| I-1 | `-c mini` (always first) | always | the *base* config the other flags layer on | intended — see the note below |
| I-2 | `model.model_kwargs.timeout=3600` | always | the HTTP timeout | none needed (constant) |
| I-3 | `model.litellm_model_registry=<path>` | a registry is given | the **price table** → cost accounting | register the model, or leave it out and treat the cost as unknown |
| I-4 | `model.model_kwargs.extra_headers.<name>=<session id>` | `session_id` is set | the request headers | none (constant) |
| I-5 | `model.model_kwargs.reasoning_effort=<eff>` | the agent's `reasoning_effort` option is set | the **reasoning budget** → a different amount of thinking per task | set it in the *job* (constant across generations) or not at all |
| I-6 | `model.model_class=litellm_response` **+** `model.model_kwargs.reasoning.effort=<eff>` | `reasoning_effort` is set **and** the model name starts with `openai/` | **the agent's protocol** | see "I-6 is the dangerous one" |
| I-7 | `model.model_kwargs.max_tokens=<n>` | the agent's `max_tokens` option is set | the **response cap** → truncation → format errors | set it in the job, and keep it identical for every generation |
| I-8 | `model.model_kwargs.max_output_tokens=<n>` | `max_tokens` is set, the model name starts with `openai/`, **and** `reasoning_effort` is set | the response cap (under a different key) | same as I-7 |

### Measured behaviour of `harbor==0.24.0` (not master)

The table above is what the **0.24.0** source in
`.venv312/lib/python3.12/site-packages/harbor/agents/installed/mini_swe_agent.py`
actually does. The injection order, verified against the command line a real
trial wrote into `/logs/agent/`, is:

```
-c mini
-c model.model_kwargs.timeout=3600
-c /tmp/mswea-config/custom.yaml      <- the generation's own config
-c model.litellm_model_registry=...    <- only when a registry is given
-c model.model_kwargs.extra_headers.X-Session-ID=...
-c model.model_kwargs.reasoning_effort=...   (+ model_class=litellm_response
                                               only for openai/* models)
-c model.model_kwargs.max_tokens=...
```

Three corrections to the earlier (master-based) reading of this table:

1. **0.24.0 injects neither `api_key` nor `api_base`.** The credential comes
   from the environment (`MSWEA_API_KEY`, and `OPENAI_BASE_URL` for the
   endpoint); `model.model_kwargs.api_key` is *not* added to the command
   line. A generation's config may still set `api_key` itself — ours sets
   `EMPTY` — and that is what the request carries.
2. **`ResolvedModelConnection` has no `model_api_format` field in 0.24.0.**
   The "resolved connection carries an API format" trigger described in the
   earlier version of this document does not exist here, so the
   `custom_llm_provider` / `model_api_format` injection (old I-7) never
   fires at all.
3. **`model_class` is overridden only by the `openai/` + `reasoning_effort`
   combination** (old I-8). It is *not* driven by an API format. A model
   named `openai/gpt-5.x` with `reasoning_effort` set gets
   `model_class=litellm_response` and `reasoning.effort` (nested), and
   `max_tokens` is renamed to `max_output_tokens`. Our model is
   `openai/glm5.3-flash` — the prefix matches `openai/`, so this *would*
   fire; `rsi/harbor_agent.py` therefore keeps the job's `reasoning_effort`
   unset, so the branch is not taken and the generation's `model_class`
   (`litellm_textbased`) stays authoritative.

### I-6 is the dangerous one

`model.model_class` selects the action protocol:

* `litellm` / `litellm_response` → the model must emit an **OpenAI tool
  call**; the `action_regex` is not even a field of the config class, so it
  is silently ignored;
* `litellm_textbased` → the model must emit a **code block**, parsed with
  `action_regex`.

If Harbor overrides `model_class` while the harness was written for the text
protocol, **every** step fails with a FormatError, and the run looks like a
model-quality problem rather than a configuration problem.

`rsi/harbor_agent.py` (`RsiMiniSweAgent`) records the effective command line
into `/logs/agent/rsi-injected-config.txt` inside the container, so a trial
directory contains the exact flags that were used — which is how the order
above was established.

## 4. Environment variables Harbor sets

| variable | value | effect |
|---|---|---|
| `MSWEA_CONFIGURED` | `true` | skips mini-swe-agent's interactive setup (good) |
| `MSWEA_COST_TRACKING` | `ignore_errors` | an unregistered model yields `cost = 0` instead of a `RuntimeError` |
| `HARBOR_MODEL_API_KEY_JSON` | the credential | read by the injected `-c model.model_kwargs.api_key` |
| `OPENAI_BASE_URL` / `MSWEA_API_KEY` | the endpoint / the key | the connection |

`MSWEA_COST_TRACKING=ignore_errors` means the **cost metric silently becomes
0** for a model that is not in litellm's price table.  The framework therefore
records `n_llm_calls` and the token counts from the trajectory as the primary
cost proxies, and treats `cost_usd` as best-effort.

## 5. The agent installation itself

Harbor installs the agent **inside the task container**:

```
uv python install 3.12
uv tool install --python 3.12 mini-swe-agent[==<version>] --with litellm --with orjson --with fastapi
```

Two consequences for constraint C-2 (pin the runtime):

* the version is pinned with the agent's `version` option
  (`kwargs: {version: "2.4.6"}`) — it is **not** pinned by default, and
  "latest" would change between generations;
* the agent runs under **Python 3.12** inside the container, while the
  framework itself may run under a different Python.  The host-side
  `RhiAgent` class is *not* what executes in the container: the container
  installs mini-swe-agent from PyPI and imports
  `rsi.rhi_agent.RhiAgent` from there.  **The `rsi` package must therefore be
  installed inside the task image** (or the class must be vendored into the
  image) — otherwise the job fails with an import error at agent start.

  In practice: publish the package to the image, or mount the source
  directory and add it to `PYTHONPATH` in the job's environment.  The IFEval
  base image does exactly this (`scripts/build_ifeval_tasks.py`): the `rsi`
  package is copied to `/opt/rsi` and exposed through `PYTHONPATH`, because
  the `uv tool` venv that Harbor creates for mini-swe-agent is *isolated* --
  a `pip install` into the image's system Python would not be visible to it.

### 5.1 The tool-call protocol cannot work inside the container (`mcp`)

Harbor's install line has **no `mcp`**:

```
uv tool install --python 3.12 mini-swe-agent --with litellm --with orjson --with fastapi
```

That matters because litellm 1.106.0 imports the MCP handler **as soon as a
request carries `tools`**:

```
litellm/main.py:5279   if not skip_mcp_handler and tools:
litellm/main.py:5280       from litellm.responses.mcp.chat_completions_handler import acompletion_with_mcp
litellm/main.py:5281       from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
...
litellm/responses/mcp/litellm_proxy_mcp_handler.py:41   from mcp.types import CallToolResult
```

The import is **unconditional** -- the `should_use_litellm_mcp_gateway()` check
that would decide whether MCP is actually needed happens *after* the import.
So inside the container:

```
ModuleNotFoundError: No module named 'mcp'
```

The tool-call protocol (`model_class: litellm`) always sends `tools`, so
**every** request fails.  The failure is reproduced with a 4-minute
exponential-backoff retry against the configured endpoint and shows up as a
run that never leaves the first step -- which is easily mistaken for a weak
model rather than a missing dependency.

**Consequence for this study.**  The default protocol is therefore **text**
(`litellm_textbased` + `action_regex`), the only protocol that works in a
Harbor container.  `toolcall` stays available for local environments where
`mcp` is installed, and `rsi/adapter.py` writes the matching observation and
format-error templates for both.

**Mitigations, if the tool-call protocol is wanted in Harbor:**

1. add `--with mcp` to Harbor's install line (a one-line upstream change), or
2. bake a mini-swe-agent venv that already contains `mcp` into the task image
   and make Harbor's `uv tool install` a no-op, or
3. patch litellm's import to be lazy (upstream `skip_mcp_handler`).

Option 1 is the smallest; until then the framework pins the text protocol.

## 6. The check list for a new job

- [ ] `agents[0].import_path` is `rsi.harbor_agent.RsiMiniSweAgent`
- [ ] `agents[0].kwargs.version` pins mini-swe-agent (`2.4.6`)
- [ ] `agents[0].kwargs.config` contains the **complete** mini-swe-agent
      config (the adapter asserts this; `assert_full_expansion`)
- [ ] the endpoint and the key come from the environment, not from the
      generation's config (or from exactly one of the two)
- [ ] `retry.max_retries` is **not** the Harbor default of 0
- [ ] the `rsi` package is importable inside the task image (and inside the
      isolated `uv tool` venv -- use `PYTHONPATH`, not `pip install`)
- [ ] the protocol is **text** unless `mcp` is installed in the container
      (see §5.1)
- [ ] `/logs/agent/rsi-injected-config.txt` exists in the trial directory
