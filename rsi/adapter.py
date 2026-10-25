"""Translation layer: :class:`~rsi.schema.HarnessSpec` -> mini-swe-agent config.

Two rules drive this module.

**C-1 (attribution).**  The Harbor job file is byte-identical across
generations; the *only* thing that changes is the ``config`` block that is
injected into ``agents[0].kwargs``.  Everything that is not governed by the
spec therefore has to be pinned here.

**C-4 (no hidden variables).**  mini-swe-agent layers a config over a builtin
one, and Harbor layers ``kwargs.config`` over the ``mini`` config.  Any field
that is not explicitly written down can change silently between Harbor
versions.  :func:`assert_full_expansion` makes that failure loud: the
generated config must contain *every* key of the frozen base config.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import HarnessSpec, SpecError

__all__ = [
    "RHI_AGENT_CLASS",
    "TOLERANT_ACTION_REGEX",
    "FROZEN_OBSERVATION_TEMPLATE",
    "FROZEN_FORMAT_ERROR_TEMPLATE",
    "default_base_config",
    "to_mini_config",
    "to_harbor_agent_kwargs",
    "assert_full_expansion",
    "missing_keys",
    "write_mini_config",
]

RHI_AGENT_CLASS = "rsi.rhi_agent.RhiAgent"

#: mini-swe-agent's default fence is ```mswea_bash_command```.  Most models
#: emit ```bash``` instead, which would make *every* step a FormatError.
#: The tolerant regex still demands exactly one block (trap T-1).
TOLERANT_ACTION_REGEX = r"```(?:mswea_bash_command|bash|sh|shell)?\s*\n(.*?)\n```"

#: Frozen at gen-0 and never touched by the optimizer.  ``observation_max_chars``
#: is read from the agent config, so the runtime knob stays effective without
#: the template itself becoming a search variable.
FROZEN_OBSERVATION_TEMPLATE = """\
{%- if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{%- if output.output | length < observation_max_chars -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking for a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you have a lot of output you can redirect it to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - observation_max_chars -%}
<output_head>
{{ output.output[:observation_max_chars // 2] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-observation_max_chars // 2:] }}
</output_tail>
{%- endif -%}
"""

#: The tool-call protocol's format error template.  It must not reference
#: ``has_tool_calls``: that variable only exists for the tool-call model, and
#: Jinja's StrictUndefined would raise when the text-based model renders it.
FROZEN_FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and (finish_reason == "length" or (finish_reason == "tool_calls" and not has_tool_calls)) -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a tool call, so it was cut off. Respond more concisely and finish with exactly one bash tool call. If you need to think more, do so briefly.
{%- else -%}
Tool call error:

<error>
{{error}}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
{%- endif -%}
"""

#: The text-based protocol needs a format error template that talks about code
#: blocks; the tool-call one talks about tool calls.  mini-swe-agent's builtin
#: ``mini.yaml`` ships the *tool-call* variant, so using it with a text-based
#: model would confuse the model on every step.
FROZEN_TEXT_FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and finish_reason in ["length", "tool_calls"] -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before it produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format. If you need to think more, do so briefly.
{%- else -%}
Format error:

<error>
{{error}}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
{%- endif -%}
"""


def default_base_config(
    *,
    model_name: str = "anthropic/claude-sonnet-4-6",
    environment_class: str = "local",
    image: str = "python:3.11-slim",
    cost_limit: float = 3.0,
    wall_time_limit_seconds: int = 3600,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    reasoning_effort: str | None = None,
    action_protocol: str = "text",
    cwd: str = "/app",
) -> dict[str, Any]:
    """The frozen gen-0 base config.

    Every field that is not governed by the harness spec lives here and is
    copied verbatim into every generation.

    ``api_base`` / ``api_key`` / ``extra_body`` exist for OpenAI-compatible
    endpoints (a local vLLM server, for instance).

    ``action_protocol`` selects how the agent talks to the model:

    ``text`` (default)
        The model emits a code block, parsed with the tolerant action regex.
        This is the only protocol that works inside a Harbor task container:
        litellm imports its MCP handler as soon as ``tools`` is set, and
        Harbor installs mini-swe-agent without the ``mcp`` extra.
    ``toolcall``
        The model emits an OpenAI tool call.  mini-swe-agent v2's recommended
        path, and Harbor's default -- but it needs the ``mcp`` package, and
        Harbor will override ``model.model_class`` unless the job uses
        ``rsi.harbor_agent.RsiMiniSweAgent``.
    """
    if action_protocol not in ("toolcall", "text"):
        raise ValueError(
            f"action_protocol must be 'toolcall' or 'text', got {action_protocol!r}"
        )

    model_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "seed": 1234,
        "drop_params": True,
    }
    if api_base:
        model_kwargs["api_base"] = api_base
    if api_key:
        model_kwargs["api_key"] = api_key
    if extra_body:
        model_kwargs["extra_body"] = dict(extra_body)
    if reasoning_effort:
        model_kwargs["extra_body"] = {
            **(model_kwargs.get("extra_body", {})),
            "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
        }

    model_section: dict[str, Any] = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "cost_tracking": "ignore_errors",
        "observation_template": FROZEN_OBSERVATION_TEMPLATE,
        "format_error_template": FROZEN_FORMAT_ERROR_TEMPLATE,
    }
    if action_protocol == "text":
        model_section["model_class"] = "litellm_textbased"
        # mini-swe-agent's default fence is ```mswea_bash_command```.  Most
        # models emit ```bash``` instead, which would make *every* step a
        # FormatError.  The tolerant regex still demands exactly one block
        # (trap T-1).
        model_section["action_regex"] = TOLERANT_ACTION_REGEX
        model_section["format_error_template"] = FROZEN_TEXT_FORMAT_ERROR_TEMPLATE
    else:
        model_section["model_class"] = "litellm"

    return {
        "agent": {
            "agent_class": RHI_AGENT_CLASS,
            # RhiAgent renders the prompts itself; these placeholders exist only
            # because mini-swe-agent's AgentConfig requires them (trap T-6).
            "system_template": "__RHI_RENDERED__",
            "instance_template": "__RHI_RENDERED__",
            "step_limit": 0,
            "cost_limit": cost_limit,
            "wall_time_limit_seconds": wall_time_limit_seconds,
            "max_consecutive_format_errors": 3,
            "output_path": None,
        },
        "model": model_section,
        "environment": {
            # ``local`` on purpose: the agent runs *inside* the Harbor task
            # container, so the commands are executed there, by the container's
            # own shell.  With ``docker`` mini-swe-agent would try to start
            # another container -- and there is no ``docker`` binary inside a
            # task container, so every trial would die with
            # ``FileNotFoundError: 'docker'`` before the model's first step.
            "environment_class": environment_class,
            "image": image,
            "cwd": cwd,
            "timeout": 120,
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        },
    }


def to_mini_config(
    spec: HarnessSpec,
    base: Mapping[str, Any] | None = None,
    *,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the mini-swe-agent config for one generation.

    Only the fields that the spec governs are taken from the spec; everything
    else is copied from the frozen base config.
    """
    base_cfg = copy.deepcopy(dict(base)) if base is not None else default_base_config()
    config = copy.deepcopy(base_cfg)

    agent = config.setdefault("agent", {})
    agent["agent_class"] = RHI_AGENT_CLASS
    agent["system_template"] = "__RHI_RENDERED__"
    agent["instance_template"] = "__RHI_RENDERED__"
    if spec_path is not None:
        agent["spec_path"] = str(spec_path)
    # The spec is inlined as well: the agent runs inside the task container,
    # where the host's spec_path does not exist.  The inlined copy is what
    # the container actually uses, so the generation's configuration cannot
    # silently drift from the spec that was evaluated.
    agent["spec"] = spec.to_dict()

    runtime = spec.runtime
    agent["max_hops"] = runtime.max_hops
    agent["hop_context_max_chars"] = runtime.hop_context_max_chars
    agent["observation_max_chars"] = runtime.observation_max_chars
    agent["context_keep_recent"] = runtime.context_keep_recent
    agent["seed"] = runtime.seed
    agent["temperature"] = runtime.temperature

    model = config.setdefault("model", {})
    model_kwargs = dict(model.get("model_kwargs", {}))
    model_kwargs["temperature"] = runtime.temperature
    model_kwargs["seed"] = runtime.seed
    model["model_kwargs"] = model_kwargs

    # The observation template reads observation_max_chars from the agent
    # config, so the knob stays meaningful while the template stays frozen.
    if "observation_template" not in model:
        model["observation_template"] = FROZEN_OBSERVATION_TEMPLATE
    if "format_error_template" not in model:
        model["format_error_template"] = FROZEN_FORMAT_ERROR_TEMPLATE

    assert_full_expansion(config, base_cfg)
    return config


def to_harbor_agent_kwargs(
    spec: HarnessSpec,
    base: Mapping[str, Any] | None = None,
    *,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """The ``--agent-kw`` payload for Harbor.

    Harbor merges ``kwargs.config`` over the agent's own ``mini`` config, so
    this dictionary has to be self-contained.
    """
    return {"config": to_mini_config(spec, base, spec_path=spec_path)}


def _key_paths(obj: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.add(child)
            paths |= _key_paths(value, child)
    return paths


def missing_keys(produced: Mapping[str, Any], base: Mapping[str, Any]) -> list[str]:
    """Keys present in the frozen base config but missing from ``produced``."""
    return sorted(_key_paths(base) - _key_paths(produced))


def assert_full_expansion(produced: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    """Raise if the generated config does not cover the whole base config (C-4)."""
    missing = missing_keys(produced, base)
    if missing:
        raise SpecError(
            "generated config does not fully expand the frozen base config; "
            f"missing: {missing}. A missing key means the value would be "
            "inherited from a Harbor/mini default and could drift between "
            "versions (constraint C-4)."
        )


def write_mini_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Persist a generation's mini config next to its run artifacts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(json.loads(json.dumps(config, default=str)), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return p
