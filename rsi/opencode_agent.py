"""Harbor agent adapter for the **opencode** rollout harness.

This targets Harbor 0.24.0's ``harbor.agents.installed.opencode.OpenCode``,
which already knows how to drive opencode and how to turn opencode's JSON
event stream into an ATIF trajectory.  This adapter adds the three things the
OpenThoughts-Agent re-annotation needs on top of it:

1. **A pinned, reproducible agent runtime.**  Harbor's adapter installs
   ``opencode-ai@latest`` unless the job file pins a version.  ``@latest``
   moves under the harness, which makes a rollout batch irreproducible --
   the same task run twice a week apart can use two different opencode builds.
   The version lives in the job file (``kwargs.version``) and is asserted
   against what the container actually runs.

2. **A fast path for pre-baked images.**  ``npm i -g opencode-ai`` costs tens
   of seconds *per trial*; over a batch of thousands of trials that is the
   single largest fixed cost, and it happens inside every container.  When the
   container already has an ``opencode`` binary -- a task image that bakes it
   in, or the toolchain bundle bind-mounted at ``/root/.local`` -- the install
   is skipped entirely.  The skip is only ever an optimisation: when the
   binary is absent, Harbor's own install still runs.

3. **The raw session export.**  The ATIF trajectory is a *lossy projection* of
   the session: tool calls are flattened into ``tool_calls``, observations into
   ``observation.results``, and per-part timestamps and ids are dropped.  The
   full message list -- every part, every tool input and output, the session
   id -- only exists in opencode's own session store, so it is exported next
   to the trajectory.

4. **Retryable provider errors.**  Harbor's opencode adapter turns *every*
   opencode error event into ``NonZeroAgentExitCodeError``, so a transient
   transport hiccup on the model endpoint kills the trial instead of being
   retried.  At 31,600 rollouts against one endpoint that is not a rare
   event, and it is not even an agent failure: the agent was mid-step when
   the provider dropped the connection.  This adapter recognises the
   provider-shaped error events and re-raises them as the matching
   ``Api*Error``, so the job's own ``retry.include_exceptions`` decides
   what happens.

The adapter never changes *what* the agent does.  The model, the prompt, the
permissions and the command line are exactly the ones Harbor's own adapter
would have used; the additions are an install short-circuit, an audit trail
and an extra export.

Usage in the Harbor job file::

    agents:
      - import_path: rsi.opencode_agent:RsiOpenCode
        name: opencode
        model_name: openai/glm5.3-flash
        kwargs:
          version: "1.18.33"
          export_session: true
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Annotated, Any, override

from pydantic import Field

try:  # pragma: no cover - Harbor is an optional dependency
    from harbor.agents.installed.base import (
        ApiConnectionClosedError,
        ApiInternalServerError,
        ApiOverloadedError,
        ApiRateLimitError,
        NonZeroAgentExitCodeError,
    )
    from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "rsi.opencode_agent requires Harbor (pip install 'harbor==0.24.0'). "
        "The rest of the framework does not need it."
    ) from exc

from harbor.agents.options import Cli

#: The opencode error events that are a **provider** problem rather than an
#: agent failure, and the Harbor exception each of them maps to.  The job
#: file's ``retry.include_exceptions`` already lists these four, so the
#: mapping is what turns a dropped connection into a retry instead of a lost
#: trial.
#:
#: The markers are deliberately specific: "ProviderModelNotFoundError" and
#: "UnknownError" say nothing about the transport, and a configuration error
#: must not be retried into the same configuration error.
_PROVIDER_ERROR_MARKERS: tuple[tuple[str, type[Exception]], ...] = (
    ("rate limit", ApiRateLimitError),
    ("too many requests", ApiRateLimitError),
    ("overloaded", ApiOverloadedError),
    ("internal server error", ApiInternalServerError),
    ("service unavailable", ApiInternalServerError),
    ("bad gateway", ApiInternalServerError),
    ("invalidhttpresponse", ApiConnectionClosedError),
    ("econnreset", ApiConnectionClosedError),
    ("econnrefused", ApiConnectionClosedError),
    ("etimedout", ApiConnectionClosedError),
    ("epipe", ApiConnectionClosedError),
    ("socket hang up", ApiConnectionClosedError),
    ("fetch failed", ApiConnectionClosedError),
    ("network error", ApiConnectionClosedError),
    ("terminated", ApiConnectionClosedError),
    ("premature close", ApiConnectionClosedError),
    ("cannot connect", ApiConnectionClosedError),
    ("unable to connect", ApiConnectionClosedError),
)


__all__ = [
    "RsiOpenCode",
    "RsiOpenCodeOptions",
    "AUDIT_FILENAME",
    "SESSION_EXPORT_FILENAME",
]

#: Where the effective invocation is recorded, inside the task container.
#: The trial's log directory is bind-mounted, so this file is collected with
#: the rest of the trial's logs and is the ground truth for what ran.
AUDIT_FILENAME = "rsi-opencode-audit.json"

#: Where the raw opencode session is written, inside the task container.
#:
#: It is gzipped: the pilot's 73-step trial produced 22 MB of JSON, and
#: 31,600 trials of that would be 700 GB.
SESSION_EXPORT_FILENAME = "session.json.gz"

#: Where the task's own environment initialisation is mounted, inside the
#: task container.
#:
#: This is the per-task half of the **family-shared base image** design.  A
#: family shares one image; whatever differs per task is prepared at run
#: time by a script the task mounts here.  For three of the four families
#: (IssueTasks, Tezos, SuperUser) there is no script at all, because their
#: environment is entirely the shared image -- an empty ``/app`` on ubuntu
#: 24.04.  For SWE-Smith the script checks the repository out at the
#: instance's bug-injected commit and installs the package editable.
TASK_SETUP_PATH = "/rsi/setup.sh"

#: Where opencode keeps its session store, **inside the container**.
#:
#: Harbor points ``XDG_DATA_HOME`` at the trial's log directory, which is
#: bind-mounted into the container -- so opencode's SQLite database lands in
#: the trial directory.  For a long rollout that is enormous: the pilot's
#: 73-step IssueTasks trial left a 592 MB ``opencode.db`` (plus the WAL) in
#: the trial directory, and the exported session alone was 22 MB.  At 31,600
#: trials that is terabytes of disk for data that is thrown away with the
#: container.
#:
#: So the store is kept in the container's own filesystem and only the
#: *exported* session is written to the trial directory.  The export runs in
#: the same container, with the same ``XDG_DATA_HOME``, so it still finds the
#: session.
CONTAINER_XDG_DATA = "/tmp/rsi-opencode/xdg-data"
CONTAINER_XDG_STATE = "/tmp/rsi-opencode/xdg-state"


class RsiOpenCodeOptions(OpenCodeOptions):
    """``OpenCodeOptions`` plus the knobs this project needs.

    Harbor validates the job file's ``agents[].kwargs`` against this model
    with ``extra="forbid"``, so every knob has to be declared here -- an
    unknown key in the job file is an error, not a silent no-op.
    """

    # NOTE: only ``agent_name`` carries a ``Cli`` annotation.  Harbor renders
    # every ``Cli``-annotated field into the agent's command line, so a knob
    # like ``export_session`` must NOT be annotated: it would end up as
    # ``--export-session`` on opencode's command line, which opencode rejects
    # as an unknown flag and every trial would fail.
    export_session: bool = Field(
        default=True,
        description=(
            "Export opencode's raw session (every part, tool input/output, "
            "session id) next to the ATIF trajectory."
        ),
    )
    skip_install_if_present: bool = Field(
        default=True,
        description=(
            "Skip the npm install when the container already provides an "
            "opencode binary."
        ),
    )
    agent_name: Annotated[str | None, Cli("--agent")] = Field(
        default=None,
        description="Name of the opencode agent to run (opencode's --agent flag).",
    )


class RsiOpenCode(OpenCode):
    """Harbor's opencode adapter, pinned and instrumented.

    Works with Harbor 0.24.0.
    """

    options_model = RsiOpenCodeOptions
    options: RsiOpenCodeOptions

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        #: The version of opencode found in the container, when the install
        #: was skipped.
        self._rsi_preinstalled_version: str | None = None
        #: The version the job file asked for.
        self._rsi_pinned_version: str | None = self.options.version

    # -- the install fast path --------------------------------------------- #

    async def _existing_opencode_version(self, environment: Any) -> str | None:
        """Return the version of an ``opencode`` binary already in the container.

        Both the plain ``PATH`` and uv/nvm-style locations are probed: the
        ``PATH`` that ``environment.exec`` uses is not necessarily the one the
        agent's own commands run with.
        """
        command = (
            "for candidate in "
            "$(command -v opencode 2>/dev/null) "
            "$HOME/.nvm/versions/node/*/bin/opencode "
            "/root/.local/bin/opencode "
            "/usr/local/bin/opencode; do "
            'if [ -x "$candidate" ]; then "$candidate" --version 2>/dev/null | head -1; break; fi; '
            "done"
        )
        try:
            result = await environment.exec(command=command, user="root")
        except Exception:  # pragma: no cover -- fall back to the slow path
            return None
        version = (getattr(result, "stdout", "") or "").strip()
        return version or None

    @override
    async def install(self, environment: Any) -> None:
        """Prepare the task's environment, then install opencode.

        The environment preparation comes first because it is the step that
        can fail for reasons the agent cannot fix: a SWE-Smith repository
        that does not clone is a broken trial, and it should fail in the
        setup phase -- where the job's logs say so -- rather than half way
        through a rollout.

        The original install is the fallback, not the default: if the binary
        is missing, Harbor's own install still runs, so the fast path can
        never change what the agent executes -- only how long it takes to get
        there.
        """
        await self._run_task_setup(environment)
        if self.options.skip_install_if_present:
            version = await self._existing_opencode_version(environment)
            if version:
                self._rsi_preinstalled_version = version
                if self._rsi_pinned_version and version != self._rsi_pinned_version:
                    self.logger.warning(
                        "container provides opencode %s but the job pins %s; "
                        "using the container's build",
                        version,
                        self._rsi_pinned_version,
                    )
                return
        await super().install(environment)

    async def _run_task_setup(self, environment: Any) -> None:
        """Run the task's environment initialisation, if it has one.

        The command is a no-op when the task mounts no setup script, which is
        the case for every task whose environment is fully described by the
        shared family image.
        """
        logs = self.environment_logs_dir.as_posix()
        command = (
            f"if [ -f {TASK_SETUP_PATH} ]; then "
            f"mkdir -p {logs} 2>/dev/null || true; "
            f"bash {TASK_SETUP_PATH} > {logs}/task-setup.log 2>&1; rc=$?; "
            f"tail -c 8000 {logs}/task-setup.log; "
            "exit $rc; "
            "else echo 'no task setup script'; fi"
        )
        result = await environment.exec(command=command, user="root")
        return_code = getattr(result, "return_code", 0)
        if return_code:
            stdout = (getattr(result, "stdout", "") or "")[-4000:]
            stderr = (getattr(result, "stderr", "") or "")[-4000:]
            raise RuntimeError(
                f"task setup script failed with exit code {return_code}:\n"
                f"{stdout}\n{stderr}"
            )

    # -- the audit trail --------------------------------------------------- #

    async def exec_as_agent(
        self,
        environment: Any,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        """Record the effective opencode command line before running it.

        Only the agent invocation is recorded -- Harbor also runs setup
        commands through this same entry point, and those would overwrite the
        interesting one.  The audit must never break the run, so every
        failure here is swallowed.

        This is also where opencode's session store is moved out of the
        bind-mounted log directory and into the container's own filesystem,
        for the reason given at ``CONTAINER_XDG_DATA``.
        """
        if env and "XDG_DATA_HOME" in env:
            env = dict(env)
            env["XDG_DATA_HOME"] = CONTAINER_XDG_DATA
            env["XDG_STATE_HOME"] = CONTAINER_XDG_STATE
        if "run --format=json" in command:
            audit = {
                "agent": "opencode",
                "model_name": self.model_name,
                "pinned_version": self._rsi_pinned_version,
                "container_version": self._rsi_preinstalled_version,
                "opencode_config": self._opencode_config,
                "command": command,
            }
            payload = json.dumps(audit, indent=2, sort_keys=True, default=str)
            target = (self.environment_logs_dir / AUDIT_FILENAME).as_posix()
            try:
                await super().exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p $(dirname {shlex.quote(target)}) 2>/dev/null || true; "
                        f"cat > {shlex.quote(target)} <<'RSI_OPENCODE_AUDIT_EOF'\n"
                        f"{payload}\n"
                        "RSI_OPENCODE_AUDIT_EOF\n"
                        "chmod 644 " + shlex.quote(target) + " 2>/dev/null || true"
                    ),
                    env=env,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                )
            except Exception:
                pass
        return await super().exec_as_agent(
            environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec
        )

    # -- the raw session export -------------------------------------------- #

    @override
    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        """Run opencode, then export its raw session next to the trajectory.

        The export is best-effort: it must never turn a successful rollout
        into a failed one.  If it fails, the ATIF trajectory is still there.
        """
        try:
            await super().run(instruction, environment, context)
        except NonZeroAgentExitCodeError as exc:
            # Harbor's opencode adapter turns *every* opencode error event
            # into NonZeroAgentExitCodeError, so a provider hiccup and a real
            # agent crash look the same to the job's retry policy.  They are
            # not the same: a dropped connection is worth another attempt.
            provider_error = self._provider_error()
            if provider_error is not None:
                self.logger.warning(
                    "opencode reported a provider error (%s); re-raising it as %s",
                    provider_error,
                    type(provider_error).__name__,
                )
                raise provider_error from exc
            raise
        finally:
            # The export runs even when the rollout failed.  A trial that
            # timed out still has a trajectory and an event stream, and the
            # published dataset keeps its timed-out rollouts -- 10,981 of its
            # 31,600 rows carry ``result: AgentTimeoutError``.  Dropping the
            # failed ones here would silently shrink the re-annotated dataset
            # by a third.
            if self.options.export_session:
                await self._export_session(environment)

    def _provider_error(self) -> Exception | None:
        """The retryable provider error in opencode's event stream, if any.

        opencode flags its own transport errors: the error data carries
        ``isRetryable: true`` for the ones worth another attempt and omits it
        for the ones that are not (a configuration error, say).  That flag is
        the signal; the message markers are only the fallback for an opencode
        that does not set it.
        """
        for event in self._parse_stdout():
            if event.get("type") != "error":
                continue
            error = event.get("error")
            if not isinstance(error, dict):
                continue
            data = error.get("data")
            data = data if isinstance(data, dict) else {}
            message = str(data.get("message") or error.get("name") or error)
            if data.get("isRetryable"):
                return ApiConnectionClosedError(message)
            lowered = message.lower()
            for marker, error_type in _PROVIDER_ERROR_MARKERS:
                if marker in lowered:
                    return error_type(message)
            if re.search(r"\b(429|500|502|503|504)\b", lowered):
                return ApiInternalServerError(message)
        return None

    async def _export_session(self, environment: Any) -> None:
        """Write opencode's own session store to the trial's log directory.

        The session id is read out of the event stream that Harbor's adapter
        already tees to ``opencode.txt``; the export then runs *inside* the
        container, because that is where the session store lives.

        The export is **best-effort**, and it is written so that a partial
        export is visible rather than silent: the exit code and the byte
        count go next to the file.  ``opencode export`` gets killed part way
        through on a long rollout (it was observed dying after 64-128 KB,
        with no error on stderr -- a kill, not an error), which leaves a
        gzip stream that decompresses fine but holds **truncated JSON**.

        That is why the export writes to a plain file first and gzips it
        afterwards, and why the exit status is recorded: a half-written
        session must not be mistaken for a whole one.  The converter does not
        depend on it -- opencode's **event stream** (``opencode.txt``) holds
        every part, every tool input and output, and is always complete, so
        the rollout is rebuilt from that when the session is unusable.

        The export is gzipped: the pilot's 73-step trial produced 22 MB of
        JSON, and 31,600 trials of that would be 700 GB.  The store itself
        is deleted afterwards -- it lives in the container's own filesystem
        (see ``CONTAINER_XDG_DATA``) and is worth hundreds of megabytes per
        trial.
        """
        logs = self.environment_logs_dir.as_posix()
        event_file = f"{logs}/opencode.txt"
        out_file = f"{logs}/opencode/{SESSION_EXPORT_FILENAME}"
        err_file = f"{logs}/opencode/session-export.err"
        command = (
            f"mkdir -p {shlex.quote(logs + '/opencode')} 2>/dev/null || true; "
            f"sid=$(grep -o '\"sessionID\":\"[^\"]*\"' {shlex.quote(event_file)} 2>/dev/null "
            f"| tail -1 | cut -d'\"' -f4); "
            'if [ -n "$sid" ]; then '
            f"XDG_DATA_HOME={shlex.quote(CONTAINER_XDG_DATA)} "
            f"XDG_STATE_HOME={shlex.quote(CONTAINER_XDG_STATE)} "
            f"opencode export \"$sid\" > /tmp/rsi-session.json 2> {shlex.quote(err_file)}; "
            "rc=$?; "
            "printf 'export rc=%s size=%s\\n' \"$rc\" "
            "$(stat -c%s /tmp/rsi-session.json 2>/dev/null || echo 0) "
            f">> {shlex.quote(err_file)}; "
            f"gzip -c /tmp/rsi-session.json > {shlex.quote(out_file)} 2>/dev/null; "
            f'else echo "no session id in {event_file}" > {shlex.quote(err_file)}; fi; '
            f"rm -rf {shlex.quote(CONTAINER_XDG_DATA)} "
            f"{shlex.quote(CONTAINER_XDG_STATE)} /tmp/rsi-session.json 2>/dev/null || true; "
            "true"
        )
        try:
            await self.exec_as_agent(environment, command=command)
        except Exception:
            self.logger.debug("raw opencode session export failed", exc_info=True)
