#!/usr/bin/env python3
"""Build the self-contained mini-swe-agent config for the SFT rollout job.

Why a generated file and not a hand-written one
----------------------------------------------
mini-swe-agent v2 treats ``-c`` as a **spec list**, not a merge base: the
moment any ``-c`` is given, the packaged ``config/mini.yaml`` is *replaced*.
So the config that reaches the container has to carry the system/instance
templates itself, or the agent runs with an empty prompt.

Two sources are merged here, in this order:

1. **the container's own ``mini_textbased.yaml``** -- taken from the
   ``rsi/toolchain:1`` image, because that image is what the trials actually
   run: the toolchain bundle is bind-mounted at ``/root/.local`` and
   ``RsiMiniSweAgent`` skips the per-trial install when it finds it.  Using
   the *host's* config instead would let the prompt drift away from the one
   the container executes.

   ``mini_textbased.yaml`` and not ``mini.yaml``: the agent runs the **text**
   protocol, and the two files differ in more than the protocol setting.
   ``mini.yaml``'s system template asks for "at least one bash *tool call*"
   and its format-error template talks about tool calls -- neither makes
   sense to a model that is answering with a fenced code block.  The
   textbased twin asks for exactly one fenced block and explains the
   submission command.

2. **the endpoint settings** -- the local vLLM endpoint, and
   ``cost_tracking: ignore_errors`` because litellm >= 1.106 renamed the
   argument that mini-swe-agent 2.4.6 still uses (``completion_cost(response=)``),
   which makes cost calculation raise for *every* model.

The protocol is **text**, not tool calls, and that is not a preference:
``model_class: litellm_textbased`` plus the tolerant action regex.  The
tool-call protocol needs the ``mcp`` package -- litellm imports its MCP
handler as soon as ``tools`` is set -- and Harbor installs mini-swe-agent
without it, so every request would die with ``ModuleNotFoundError: No module
named 'mcp'`` and the run would look like a weak model rather than a missing
dependency.

Three deviations from ``base_config.yaml``, all deliberate:

**The observation and format-error templates come from mini, not from the
frozen base.**  The frozen ones are written for this project's ``RhiAgent``:
they interpolate ``observation_max_chars``, a variable that only the RhiAgent
puts into the template context.  Under plain mini-swe-agent the render dies
with ``UndefinedError: 'observation_max_chars' is undefined`` on the first
tool call, after the model has already answered.  The frozen *protocol* (the
text fence and the action regex) is kept; the templates are not.

**``max_tokens: 2560 -> 8192``**
    GLM thinks before it answers, and the thinking eats the budget.  At 2560
    the reply is cut off before the action is emitted, mini reads that as a
    FormatError, retries, and gets truncated again.  The pilot measured this:
    every step fails until ``max_consecutive_format_errors`` stops the run.

**``environment.cwd`` is left unset**
    The cwd differs per family (``/testbed`` for SWE-Smith, ``/app`` for the
    other three) and it is already the image's ``WORKDIR``.  mini's local
    environment falls back to ``os.getcwd()`` when ``cwd`` is empty, and the
    agent process is exec'd in the image's working directory, so leaving it
    unset gets both families right with one config.

Usage::

    python mini_sft/build_config.py
    python mini_sft/build_config.py --out somewhere/else.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: The image whose mini-swe-agent the trials actually execute.
TOOLCHAIN_IMAGE = "rsi/toolchain:2"

#: The exported bundle -- the directory that is bind-mounted into every trial
#: at ``/root/.local``.  Its packaged config is the one the agent really
#: reads, so it is the preferred source.
TOOLCHAIN_BUNDLE = REPO / ".cache" / "rsi-local"
#: Where the toolchain keeps mini-swe-agent's packaged config.  mini ships
#: one file per protocol: ``mini.yaml`` is the tool-call one,
#: ``mini_textbased.yaml`` the fenced-code-block one.  They differ in the
#: system template, the instance template, the observation template and the
#: format-error template -- all four of them.
MINI_CONFIGS_IN_IMAGE = {
    "toolcall": (
        "/root/.local/share/uv/tools/mini-swe-agent/lib/python3.12/"
        "site-packages/minisweagent/config/mini.yaml"
    ),
    "text": (
        "/root/.local/share/uv/tools/mini-swe-agent/lib/python3.12/"
        "site-packages/minisweagent/config/mini_textbased.yaml"
    ),
}
#: The frozen protocol section of the project.
BASE_CONFIG = HERE / "config" / "base_config.yaml"

#: The endpoint the rollouts talk to.
API_BASE = os.environ.get("MODEL_BASE_URL", "http://YOUR-ENDPOINT:PORT/v1")
API_KEY = "EMPTY"
MODEL_NAME = "openai/glm5.3-flash"

#: How many agent steps a rollout may take.  See ``build()`` for why this
#: cannot be left at mini's default.
STEP_LIMIT = 30

#: The ClawGym instance template's head -- the part that is replaced.
#:
#: mini's packaged template opens with "Please solve this issue: {{task}}" and
#: then a workflow about finding a codebase, reproducing an issue and editing
#: source code.  That is a *SWE-bench* framing, and it is wrong for the
#: ClawGym tasks: they are not issues, there is no codebase, and the
#: deliverables are files the task names (``output/report.json`` and
#: friends), not a patch.  A template that contradicts the task teaches the
#: model to work from a framing the task never used.
#:
#: Everything from "## Command Execution Rules" onwards is left byte-identical
#: to mini's own template: that section *is* the protocol (the bash tool, the
#: subshell rule, the submission command), and the protocol is what the
#: harness is.  Only the framing changes.
CLAWGYM_INSTANCE_HEAD = """{{task}}

You are working in a workspace directory. The task's own files are under `input/`, and your deliverables go where the task says -- usually `output/`. Every path in the task is relative to this directory.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Read the task carefully and list the files it gives you under `input/`
2. Inspect the inputs -- their format, their field names, their size -- before producing anything
3. Work out the exact output paths and the exact content or schema the task asks for
4. Produce the deliverables, usually from a small script so that the result is reproducible
5. Verify that every deliverable exists, parses, and carries every field the task requires
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

"""


def builtin_mini_yaml(
    image: str = TOOLCHAIN_IMAGE,
    protocol: str = "toolcall",
    bundle: Path = TOOLCHAIN_BUNDLE,
) -> dict[str, Any]:
    """The packaged config for ``protocol`` that the trials will execute.

    Read from the exported bundle when it is there: the bundle *is* what gets
    bind-mounted, so its copy is the one the agent loads.  The image is the
    fallback, for a host that has not exported the bundle yet.
    """
    name = "mini.yaml" if protocol == "toolcall" else "mini_textbased.yaml"
    in_bundle = (
        bundle
        / "share"
        / "uv"
        / "tools"
        / "mini-swe-agent"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "minisweagent"
        / "config"
        / name
    )
    if in_bundle.exists():
        text = in_bundle.read_text(encoding="utf-8")
        source = str(in_bundle)
    else:
        path = MINI_CONFIGS_IN_IMAGE[protocol]
        result = subprocess.run(
            ["docker", "run", "--rm", image, "sh", "-c", f"cat {path}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"could not read {path} out of {image}:\n{result.stderr}"
            )
        text = result.stdout
        source = f"{image}:{path}"

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"{source}: the config did not parse into a mapping")
    return data


def clawgym_instance_template(builtin_template: str) -> str:
    """mini's instance template, with the SWE-bench framing swapped out.

    The tail -- from ``## Command Execution Rules`` on -- is mini's own text,
    byte for byte.  That section is the protocol, and the protocol is the
    harness; only the framing in front of it changes.
    """
    marker = "## Command Execution Rules"
    _, separator, tail = builtin_template.partition(marker)
    if not separator:
        raise SystemExit(
            "the packaged mini instance template no longer contains the "
            f"{marker!r} section; the clawgym profile has to be updated"
        )
    return CLAWGYM_INSTANCE_HEAD + marker + tail


def build(
    image: str = TOOLCHAIN_IMAGE,
    api_base: str = API_BASE,
    protocol: str = "toolcall",
    profile: str = "swe",
) -> dict[str, Any]:
    builtin = builtin_mini_yaml(image, protocol)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

    frozen_model = base["model"]
    frozen_env = base["environment"]

    agent = dict(builtin["agent"])
    # The packaged templates are the text-protocol ones.  The RhiAgent
    # placeholders were for the research agent, which renders its own
    # prompts; the rollouts run mini's own templates.
    agent["mode"] = "default"
    # **The step limit is what stops the agent, and it has to be set.**
    #
    # mini's default is ``step_limit: 0`` (unlimited) plus
    # ``cost_limit: 3.0``, and the limit fires on
    #
    #     0 < step_limit <= n_calls  or  0 < cost_limit <= cost
    #
    # -- so with step_limit 0 the cost limit is the only guard, and the cost
    # is *always* 0 here: litellm cannot fetch its price table from this
    # network, so every request is billed at nothing.  The agent then runs
    # until the wall-clock timeout, and the rollout is both far too long and
    # a failure.
    #
    # Measured against the published dataset this is not a small difference.
    # The original rollouts' assistant-turn counts: issue p50 9 / p90 14 /
    # max 28, superuser p50 7 / p90 15, tezos p50 6 / p90 16, swesmith
    # p50 17 / p90 27.  Without a step limit the re-annotation produced
    # p50 45 / p90 75 / max 84, and 100 of 100 trials in a batch hit the
    # agent timeout.
    #
    # 30 covers every family's observed maximum (issue 28, tezos 39,
    # swesmith 40) without letting a runaway rollout through.
    agent["step_limit"] = STEP_LIMIT
    # 0 disables the cost limit, which is what it effectively already was.
    # Leaving it at 3.0 would be a trap: a limit that reads as a guard and
    # never fires.
    agent["cost_limit"] = 0.0

    if profile == "clawgym":
        agent["instance_template"] = clawgym_instance_template(
            agent["instance_template"]
        )
    elif profile != "swe":
        raise SystemExit(f"unknown profile {profile!r} (swe, clawgym)")

    model: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_class": "litellm" if protocol == "toolcall" else frozen_model["model_class"],
        "cost_tracking": frozen_model["cost_tracking"],
        # mini's own text templates -- NOT the frozen base ones, which need
        # ``observation_max_chars`` (an RhiAgent-only template variable).
        "observation_template": builtin["model"]["observation_template"],
        "format_error_template": builtin["model"]["format_error_template"],
        "model_kwargs": {
            **frozen_model["model_kwargs"],
            "api_base": api_base,
            "api_key": API_KEY,
            # See the module docstring: 2560 truncates GLM mid-thought.
            "max_tokens": 8192,
        },
    }

    if protocol == "text":
        model["action_regex"] = frozen_model["action_regex"]

    environment: dict[str, Any] = {
        "environment_class": frozen_env["environment_class"],
        # No ``cwd``: the image's WORKDIR is already the family's working
        # directory, and mini falls back to ``os.getcwd()``.
        "timeout": frozen_env["timeout"],
        "env": frozen_env["env"],
    }

    return {
        "agent": agent,
        "model": model,
        "environment": environment,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(HERE / "config" / "mini-config.yaml"),
    )
    ap.add_argument("--image", default=TOOLCHAIN_IMAGE)
    ap.add_argument(
        "--protocol",
        choices=["toolcall", "text"],
        default="toolcall",
        help="the action protocol: native OpenAI tool calls, or a fenced "
        "code block the agent parses",
    )
    ap.add_argument(
        "--api-base",
        default=API_BASE,
        help="the endpoint the agent talks to; point it at "
        "http://host.docker.internal:PORT/v1 to record the wire",
    )
    ap.add_argument(
        "--profile",
        choices=["swe", "clawgym"],
        default="swe",
        help="swe: mini's packaged instance template (an issue to fix in a "
        "codebase).  clawgym: the same template with the SWE-bench framing "
        "replaced, because the ClawGym tasks are not issues and their "
        "deliverables are files, not a patch.  The protocol section -- the "
        "bash tool, the subshell rule, the submission command -- is "
        "byte-identical in both.",
    )
    args = ap.parse_args()

    config = build(args.image, args.api_base, args.protocol, args.profile)
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=4096)

    # The round trip has to be lossless: the observation template is a long
    # Jinja program, and YAML folding it would silently change the prompt.
    reparsed = yaml.safe_load(text)
    if reparsed != config:
        print("FATAL: YAML round trip is lossy", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    print(f"wrote {out}  ({len(text)} bytes)")
    print(f"  base image   : {args.image}")
    print(f"  model        : {config['model']['model_name']} "
          f"({config['model']['model_class']}) @ {config['model']['model_kwargs']['api_base']}")
    print(f"  protocol     : {args.protocol}")
    print(f"  max_tokens   : {config['model']['model_kwargs']['max_tokens']}")
    print(f"  cwd          : unset (the image's WORKDIR)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
