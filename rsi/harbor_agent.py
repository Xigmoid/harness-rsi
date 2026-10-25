"""Harbor agent adapter with the generation's config pinned.

Harbor's ``mini-swe-agent`` adapter assembles the mini-swe-agent command line
from a series of ``-c`` flags, and the generation's own config is only *one*
of them.  The flags that come **after** it win, so a handful of Harbor
injections can silently change the behaviour of the harness.

**This module targets Harbor 0.24.0 specifically** -- the version that is
actually installed.  The injection inventory of that version is:

===================================== ========================================
flag                                  when it is emitted
===================================== ========================================
``-c mini``                           always (the packaged base config)
``-c model.model_kwargs.timeout``     always
``-c /tmp/mswea-config/custom.yaml``  always (the generation's config)
``-c model.litellm_model_registry``   only when a registry is given
``-c model.model_kwargs.extra_headers`` only when a session id is set
``-c model.model_class=litellm_response`` **only when the agent option
                                       ``reasoning_effort`` is set *and* the
                                       model name starts with ``openai/``**
``-c model.model_kwargs.reasoning_effort`` same condition
``-c model.model_kwargs.max_tokens``  only when the agent option is set
===================================== ========================================

The dangerous one is ``model.model_class``: our model is
``openai/glm5.3-flash`` and our harness speaks the **text** protocol
(``model_class: litellm_textbased``), so a job that sets the agent option
``reasoning_effort`` makes Harbor override the protocol and **every step of
every task fails** with a format error.  It looks like a model-quality
problem, not a configuration problem.

Note that 0.24.0 does **not** inject ``model.model_kwargs.api_key`` or
``model.model_kwargs.api_base``; the endpoint travels through the
environment (``MSWEA_API_KEY`` / ``OPENAI_BASE_URL``) and the generation's
config keeps its own explicit values.

The adapter enforces three invariants:

1. the *protocol* fields (``model.model_class``, ``model.action_regex``)
   belong to the generation and are never overridden;
2. the *budget* option ``reasoning_effort`` is removed from the agent
   options, because in 0.24.0 it is the only thing that triggers the
   ``model_class`` injection.  The generation's config carries the same
   setting inside ``model.model_kwargs.extra_body.chat_template_kwargs``,
   so the model's reasoning budget is unchanged;
3. every injected flag is recorded next to the run, so a generation can be
   reproduced from the trial directory alone.

**The toolchain fast path.**  Harbor 0.24.0 installs mini-swe-agent inside
*every* task container (``curl uv`` + ``uv python install 3.12`` + ``uv tool
install mini-swe-agent==2.4.6 --with litellm --with orjson --with fastapi``),
and ``ensure_system_dependencies()`` runs an ``apt-get install
build-essential`` before it.  That is ~4 minutes and ~1 GB of downloads *per
trial*.  On the registry tasks it is not a cost, it is a wall: a 4-task AIME
calibration finished with ``AgentSetupTimeoutError`` on 4/4 trials -- the
agent never ran a single step.

The fix is the *toolchain bundle*: the toolchain is built once in a container
(linux/arm64, the architecture the trials run on) and bind-mounted into every
trial at ``/root/.local`` -- the path uv expects, so the venv's absolute
shebangs keep resolving.  :meth:`RsiMiniSweAgent.install` then detects the
mounted toolchain and returns immediately, and the agent command gets the
bundle's ``bin`` directory and the ``rsi`` package on ``PYTHONPATH``.

The bundle is an *optimisation*, not a change of the harness: the pinned
version (``mini-swe-agent==2.4.6``) is identical to the one the job file
pins, and the fast path is only taken when the tool is actually there --
otherwise the original install still runs.

Usage in the Harbor job file::

    agents:
      - import_path: rsi.harbor_agent.RsiMiniSweAgent
        model_name: openai/glm5.3-flash
        kwargs:
          version: "2.4.6"
          config:
            agent:
              mode: default
"""

from __future__ import annotations

import shlex
from typing import Any

from .paths import TOOLCHAIN_MOUNT_TARGET

try:  # pragma: no cover - Harbor is an optional dependency
    from harbor.agents.installed.mini_swe_agent import (
        _MINI_SWE_AGENT_PYTHON_VERSION,
        MiniSweAgent,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "rsi.harbor_agent requires Harbor (pip install 'harbor==0.24.0'). "
        "The rest of the framework does not need it."
    ) from exc

__all__ = [
    "RsiMiniSweAgent",
    "INJECTION_AUDIT_PATH",
    "GENERATION_OWNED_FIELDS",
    "TOOLCHAIN_DIR",
    "TOOLCHAIN_BIN",
    "RSI_PKG_DIR",
]

#: Where the agent toolchain is bind-mounted inside the task container.  It
#: is uv's default location, so the venv's absolute shebangs keep resolving.
TOOLCHAIN_DIR = TOOLCHAIN_MOUNT_TARGET

#: The mini-swe-agent entry point inside the bundle.
TOOLCHAIN_BIN = f"{TOOLCHAIN_DIR}/bin/mini-swe-agent"

#: Where the ``rsi`` package would live inside the bundle, if it were
#: needed there.  It is not: the rollouts run mini in its default mode, and
#: the adapter itself is imported by the *host* process that runs
#: ``harbor job start``.  The directory is kept on the container's
#: ``PYTHONPATH`` so that a bundle built from an older revision still finds
#: the package it shipped with.
RSI_PKG_DIR = f"{TOOLCHAIN_DIR}/rsi-pkg"

#: Where the adapter records the effective command line inside the container.
INJECTION_AUDIT_PATH = "/logs/agent/rsi-injected-config.txt"

#: The config fields that the *generation* owns.  Harbor must not override
#: them, because they define the agent's protocol.
GENERATION_OWNED_FIELDS = (
    "model.model_class",
    "model.action_regex",
    "agent.agent_class",
    "agent.system_template",
    "agent.instance_template",
    "agent.step_limit",
    "agent.cost_limit",
    "agent.max_consecutive_format_errors",
    "model.observation_template",
    "model.format_error_template",
)


#: Where the *fallback* install puts its toolchain inside the container.
#:
#: uv's default tool and python directories live under ``/root/.local/share/uv``,
#: and the toolchain bundle is bind-mounted at ``/root/.local``.  So a fallback
#: install writes **into the shared mount**: it removes the tool environment,
#: reinstalls it, and in doing so breaks the bundle for every other trial
#: running against the same mount.
#:
#: That is not hypothetical.  It happened: a batch whose fast path did not
#: trigger (the bundle had been damaged by an earlier batch) ran four
#: fallbacks concurrently, the first one removed
#: ``share/uv/tools/mini-swe-agent/bin``, and the bundle stopped working for
#: everybody -- the fast path then failed for the *next* batch too, and the
#: bundle had to be re-exported from the toolchain image.
#:
#: Redirecting uv's directories to a container-local path keeps the fallback
#: install's blast radius inside the trial: the shared bundle is read-only
#: as far as uv is concerned, and a trial that has to install gets its own
#: copy.
UV_REDIRECT_ENV: dict[str, str] = {
    "UV_TOOL_DIR": "/tmp/rsi-uv/tools",
    "UV_TOOL_BIN_DIR": "/tmp/rsi-uv/bin",
    "UV_PYTHON_INSTALL_DIR": "/tmp/rsi-uv/python",
    "UV_CACHE_DIR": "/tmp/rsi-uv/cache",
}

#: The bin directory the redirected install writes its shims to.
UV_REDIRECT_BIN = UV_REDIRECT_ENV["UV_TOOL_BIN_DIR"]


class RsiMiniSweAgent(MiniSweAgent):
    """Harbor's mini-swe-agent adapter, with the generation config pinned.

    Works with Harbor 0.24.0.  It does **not** touch the model connection:
    that API (``model_api_format``) only exists in newer Harbor versions,
    and 0.24.0 has no API-format injection at all.  The only thing that can
    override the protocol in 0.24.0 is the ``reasoning_effort`` agent
    option, so that is what this adapter removes.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # 0.24.0 emits `-c model.model_class=litellm_response` (and the
        # matching reasoning flag) *only* when the reasoning-effort option is
        # set and the model name starts with "openai/".  Both hold for
        # `openai/glm5.3-flash`, and the generation's config selects the
        # text protocol, so the option is dropped here.  The generation's
        # config keeps the same setting inside
        # `model.model_kwargs.extra_body.chat_template_kwargs`, so the
        # model's reasoning budget is unchanged.
        options = getattr(self, "options", None)
        if options is not None and getattr(options, "reasoning_effort", None):
            self._rsi_dropped_reasoning_effort = options.reasoning_effort
            options.reasoning_effort = None
        self._rsi_toolchain_present: bool | None = None

    # -- the toolchain fast path ------------------------------------------- #

    @staticmethod
    def _toolchain_prelude() -> str:
        """The shell prelude that puts the bundle on PATH and PYTHONPATH.

        It is prepended to *every* command the agent runs, because the
        bundle's location is not the container's ``$HOME`` for every task
        image, and the ``rsi`` package has to be importable from inside the
        isolated tool venv.

        ``/opt/rsi`` comes first on purpose: it is the *current* copy of the
        package, while ``/root/.local/rsi-pkg`` may be a stale snapshot that
        the toolchain image carried.

        ``LITELLM_LOCAL_MODEL_COST_MAP`` is set for the same reason: on every
        start litellm tries to fetch its price table from
        ``raw.githubusercontent.com``, which this network cannot reach, so the
        fetch has to time out before litellm falls back to its bundled copy.
        Measured inside a trial container, ``import litellm`` took 9.3s with
        the fetch and 4.4s without it.  The price table only feeds cost
        accounting -- the trajectories already recorded ``instance_cost: 0.0``
        because the fetch failed there too -- so nothing the model sees
        changes; the agent just starts five seconds earlier.
        """
        return (
            f'export PATH="{TOOLCHAIN_DIR}/bin:$PATH"; '
            f'export PYTHONPATH="/opt/rsi:{RSI_PKG_DIR}:${{PYTHONPATH:-}}"; '
            f'export LITELLM_LOCAL_MODEL_COST_MAP=True; '
        )

    async def _toolchain_available(self, environment: Any) -> bool:
        """True when the pinned toolchain is already in the container."""
        if self._rsi_toolchain_present is not None:
            return self._rsi_toolchain_present
        try:
            result = await environment.exec(
                command=f"test -x {TOOLCHAIN_BIN} && echo rsi-toolchain-ok || true",
                user="root",
            )
            present = "rsi-toolchain-ok" in (getattr(result, "stdout", "") or "")
        except Exception:  # pragma: no cover -- fall back to the slow path
            present = False
        self._rsi_toolchain_present = present
        return present

    async def install(self, environment: Any) -> None:
        """Skip the per-trial install when the toolchain bundle is mounted.

        The original install costs ~4 minutes per trial.  When the bundle is
        mounted, the pinned toolchain is already there and the install would
        be a no-op anyway -- so it is not run at all.  When the bundle is
        *not* mounted, the original install still happens: the fast path is
        an optimisation, never a change of what the agent runs.

        The fallback install is **redirected away from the mounted bundle**.
        See :data:`UV_REDIRECT_ENV` for what happens when it is not.
        """
        if await self._toolchain_available(environment):
            return
        await self._install_isolated(environment)

    async def _install_isolated(self, environment: Any) -> None:
        """Harbor's install, with uv's directories pointed at the trial.

        The command is Harbor's own -- same pinned version, same extras, same
        python -- with three things added:

        * uv's tool/python/cache directories point into ``/tmp/rsi-uv``, so a
          fallback install cannot write into the shared bundle;
        * the redirected bin directory is on ``PATH``, because that is where
          the ``mini-swe-agent`` shim lands;
        * the shim directory is prepended, so the shim is found even though
          ``$HOME/.local/bin`` -- the directory the bundle's own shim lives
          in -- is the one Harbor puts on ``PATH``.
        """
        await self.ensure_system_dependencies(
            environment,
            ("curl", "bash", "build_tools", "git"),
        )
        version_spec = f"=={self._version}" if self._version else ""
        python_version = shlex.quote(_MINI_SWE_AGENT_PYTHON_VERSION)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then"
                "  curl -LsSf https://astral.sh/uv/install.sh | sh;"
                " fi && "
                'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then'
                '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc";'
                " fi && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv python install {python_version} && "
                f"uv tool install --python {python_version} "
                f"mini-swe-agent{version_spec} "
                "--with litellm --with orjson --with fastapi && "
                f'export PATH="{UV_REDIRECT_BIN}:$PATH" && '
                "mini-swe-agent --help"
            ),
            env=dict(UV_REDIRECT_ENV),
        )

    async def exec_as_agent(
        self,
        environment: Any,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        """Record the effective agent command line before running it.

        The recorded file is the ground truth for what a generation actually
        ran with: it contains every ``-c`` flag Harbor added, in order.
        Only the *agent* invocation is recorded -- Harbor also runs setup
        commands through the same entry point, and those would overwrite the
        interesting one.
        """
        command = self._toolchain_prelude() + command
        if "mini-swe-agent --yolo" in command:
            audit = (
                "### effective mini-swe-agent invocation\n"
                f"### generation-owned fields: {', '.join(GENERATION_OWNED_FIELDS)}\n"
                + (
                    "### dropped agent option: reasoning_effort="
                    f"{self._rsi_dropped_reasoning_effort}\n"
                    if getattr(self, "_rsi_dropped_reasoning_effort", None)
                    else ""
                )
                + f"{command}\n"
            )
            try:
                await super().exec_as_agent(
                    environment,
                    f"mkdir -p $(dirname {INJECTION_AUDIT_PATH}) 2>/dev/null || true; "
                    f"cat > {INJECTION_AUDIT_PATH} <<'RHI_AUDIT_EOF'\n{audit}RHI_AUDIT_EOF\n"
                    f"chmod 644 {INJECTION_AUDIT_PATH} 2>/dev/null || true",
                    env=env,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                )
            except Exception:
                # The audit must never break the run.
                pass
        return await super().exec_as_agent(
            environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec
        )
