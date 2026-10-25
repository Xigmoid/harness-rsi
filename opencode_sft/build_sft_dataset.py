"""Turn the Harbor + opencode rollouts into an SFT dataset.

The published dataset (``open-thoughts/OpenThoughts-Agent-SFT-31.6K``) stores
each rollout as ``conversations``: a list of ``{"role", "content"}`` messages.
Scanning all 31,600 rows gives three hard invariants that the re-annotated
dataset has to keep, otherwise the same training code cannot load it:

============================  =========================================
role vocabulary               ``user`` and ``assistant`` only -- no
                              ``system``, no ``tool``
message keys                  exactly ``content`` and ``role``
shape                         ``user`` (prompt) -> ``assistant`` ->
                              ``user`` (observation) -> ... ; 773 of the
                              31,600 rows have a second ``user`` message
                              at index 1 (the resume episodes)
============================  =========================================

An opencode rollout is not in that shape.  opencode sends a *system* message
with its own prompt, a *user* message with the task, assistant turns that
carry reasoning plus tool calls, and one *tool* message per tool result.  So
this module converts one into the other and writes **two** views of the same
rollout:

``conversations``
    The strict ``[{role, content}]`` schema of the published dataset.  The
    system prompt and the task instruction become the two leading ``user``
    messages; the tool calls are flattened into inline ``<tool_call>``
    text and the tool results become ``user`` observation messages.

``messages``
    The serve-parity view a tool-aware trainer actually wants: the assistant
    turns keep their tool calls *structured*
    (``tool_calls: [{type, function: {name, arguments}}]``), the tool results
    are ``role: "tool"`` messages with a ``tool_call_id``, and the reasoning
    stays verbatim in ``content``.

Both views are derived from the same source -- opencode's own session export
(``agent/opencode/session.json``) -- so nothing is invented on the way.  The
system prompt and the tool schemas are **not** taken from the session: they
are taken from the request that the recording proxy captured, because that is
the only place where the exact bytes the model saw exist.

Two details of the opencode wire format are worth spelling out, because they
are the difference between "the task instruction is unchanged" and "the task
instruction is unchanged":

* opencode sends the user message **wrapped in double quotes**, escaping the
  inner double quotes but *not* the backslashes.  A task that contains three
  literal backslashes arrives at the model with three literal backslashes --
  it is not a JSON round trip.  The dataset records the clean instruction
  (``instruction``) next to the wire form (``instruction_wire``) so both can
  be checked.
* opencode runs a connectivity probe before the real task, asking the model
  to say "OK".  That probe carries the full system prompt and the full tool
  list, so it looks exactly like a real request.  It is skipped because its
  user message never matches a trial.

Usage::

    python opencode_sft/build_sft_dataset.py \
        --run /private/tmp/rsi-harbor/opencode-sft-capture \
        --requests /tmp/oc-requests.jsonl \
        --out opencode_sft/dataset/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

import sys as _sys

_sys.path.insert(0, str(HERE.parent))
from opencode_sft.schema import (  # noqa: E402
    DATASET_OPENTHOUGHT,
    HARNESS_OPENCODE,
    canonical_model,
    provenance,
)

#: The role vocabulary of the published dataset -- the converted view may not
#: use anything else.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"  # only in the ``messages`` view
ROLE_TOOL = "tool"  # only in the ``messages`` view

TAG_TOOL_CALL = "<tool_call>"
TAG_TOOL_RESULT = "<tool_result>"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wire_wrap(text: str) -> str:
    """The form opencode puts a user message in on the wire.

    Double quotes around the text, inner double quotes escaped, backslashes
    left alone.  Verified against the recorded requests of the pilot.
    """
    return '"' + text.replace('"', '\\"') + '"'


# --------------------------------------------------------------------------- #
# the recorded requests
# --------------------------------------------------------------------------- #


def read_lines(path: Path) -> list[str]:
    """The lines of a text file, transparently gunzipping a ``.gz`` one."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


class BlobReader:
    """Puts back the content-addressed blobs the recording proxy wrote.

    The proxy stores the opencode system prompt and the tool schemas **once**,
    by digest, because inside a container they are the same for every
    rollout; the request log carries the digests instead of the bytes.  This
    resolves them, so the rest of the converter sees the shape it has always
    seen -- a body with ``messages`` and ``tools`` in it.
    """

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self._cache: dict[str, Any] = {}

    def _load(self, kind: str, digest: str) -> Any:
        key = f"{kind}:{digest}"
        if key in self._cache:
            return self._cache[key]
        if self.directory is None:
            return None
        path = self.directory / f"{digest}.{kind}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        self._cache[key] = payload
        return payload

    def system_messages(self, digest: str) -> list[dict[str, Any]] | None:
        payload = self._load("system", digest)
        if isinstance(payload, dict):
            messages = payload.get("messages")
            if isinstance(messages, list):
                return messages
        return None

    def tools(self, digest: str) -> Any:
        return self._load("tools", digest)

    def resolve(self, body: dict[str, Any]) -> dict[str, Any]:
        """A body with its blobs inlined again."""
        if not isinstance(body, dict):
            return body
        system_digest = body.get("system_blob")
        tools_digest = body.get("tools_blob")
        if not system_digest and not tools_digest:
            return body
        resolved = dict(body)
        if system_digest:
            messages = self.system_messages(system_digest)
            if messages is not None:
                # The system messages came first in the request, and the
                # compact body kept the user message after them.
                resolved["messages"] = list(messages) + [
                    message
                    for message in (body.get("messages") or [])
                    if message.get("role") != "system"
                ]
        if tools_digest:
            tools = self.tools(tools_digest)
            if tools is not None:
                resolved["tools"] = tools
        return resolved


class RequestIndex:
    """The recorded requests, keyed by the user message they carry.

    opencode sends the task instruction as the first user message, and the
    instruction is unique per task, so it is the key that ties a recorded
    request stream to a trial.  The key is the *wire* form, because that is
    what opencode stores in its session as well -- the two therefore match
    exactly, with no normalisation in between.
    """

    def __init__(self, path: Path, blob_dir: Path | None = None) -> None:
        self.by_user_message: dict[str, list[dict[str, Any]]] = {}
        self.n_requests = 0
        self.n_agent_requests = 0
        self.n_side_requests = 0
        self.n_unresolved_blobs = 0
        self.blobs = BlobReader(blob_dir)
        if not path.exists():
            return
        for line in read_lines(path):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = record.get("body")
            if not isinstance(body, dict):
                continue
            self.n_requests += 1
            if body.get("system_blob") or body.get("tools_blob"):
                body = self.blobs.resolve(body)
                if (
                    body.get("system_blob") and not body.get("messages")
                ) or (body.get("tools_blob") and body.get("tools") is None):
                    self.n_unresolved_blobs += 1
                    continue
            if not body.get("tools"):
                # The title generator: no tools, not an agent request.
                self.n_side_requests += 1
                continue
            self.n_agent_requests += 1
            user_message = self._first_user_message(body)
            if user_message is None:
                continue
            self.by_user_message.setdefault(user_message, []).append(body)

    @staticmethod
    def _first_user_message(body: dict[str, Any]) -> str | None:
        for message in body.get("messages") or []:
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) and content else None
        return None

    def for_user_message(self, user_message: str) -> dict[str, Any] | None:
        """The first recorded agent request for a user message: prompt + tools."""
        bodies = self.by_user_message.get(user_message)
        if not bodies:
            return None
        for body in bodies:
            if body.get("tools"):
                return body
        return bodies[0]


def system_prompt_of(body: dict[str, Any] | None) -> str | None:
    if not body:
        return None
    for message in body.get("messages") or []:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multi-part content
                return "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
    return None


def tools_of(body: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not body:
        return None
    tools = body.get("tools")
    return tools if isinstance(tools, list) else None


# --------------------------------------------------------------------------- #
# one trial
# --------------------------------------------------------------------------- #


def _text_of(parts: list[dict[str, Any]]) -> str:
    """The assistant's visible text of a message (reasoning excluded)."""
    return "\n".join(
        part.get("text", "") for part in parts if part.get("type") == "text"
    ).strip()


def _reasoning_of(parts: list[dict[str, Any]]) -> str:
    return "\n".join(
        part.get("text", "") for part in parts if part.get("type") == "reasoning"
    ).strip()


def _tool_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [part for part in parts if part.get("type") == "tool"]


def _tool_call(part: dict[str, Any]) -> dict[str, Any]:
    """One tool call in the OpenAI shape the trainers expect."""
    state = part.get("state") or {}
    return {
        "type": "function",
        "function": {
            "name": part.get("tool", ""),
            # A JSON *string*, so the row schema stays Arrow-uniform; the
            # training framework parses it back into a dict.
            #
            # **No spaces after the separators.**  opencode serialises the
            # arguments with JavaScript's ``JSON.stringify``, which produces
            # ``{"a":1,"b":2}``; Python's ``json.dumps`` would produce
            # ``{"a": 1, "b": 2}``.  The two parse to the same object, but
            # they are different token sequences, and the training data has
            # to be the bytes the model saw.
            "arguments": json.dumps(
                state.get("input") or {},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        "id": part.get("callID", ""),
    }


def _tool_output(part: dict[str, Any]) -> str:
    state = part.get("state") or {}
    output = state.get("output")
    if output is None:
        output = (state.get("metadata") or {}).get("output", "")
    if output is None:
        return ""
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


def _tool_input(part: dict[str, Any]) -> dict[str, Any]:
    return (part.get("state") or {}).get("input") or {}


def _assistant_body(reasoning: str, text: str) -> str:
    """``<think>`` reasoning plus the visible text, as one blob.

    The published dataset puts the reasoning in a ``<think>`` block right
    before the action, so the converted view does the same.
    """
    if reasoning and text:
        return f"<think>\n{reasoning}\n</think>\n\n{text}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>"
    return text


def turns_from_session(session: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant turns of an opencode session, in order."""
    turns: list[dict[str, Any]] = []
    for message in session.get("messages") or []:
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        parts = message.get("parts") or []
        tools = []
        for part in _tool_parts(parts):
            state = part.get("state") or {}
            tools.append(
                {
                    "tool": part.get("tool", ""),
                    "callID": part.get("callID", ""),
                    "input": state.get("input") or {},
                    "output": _tool_output(part),
                }
            )
        turns.append(
            {
                "reasoning": _reasoning_of(parts),
                "text": _text_of(parts),
                "tools": tools,
            }
        )
    return turns


def turns_from_event_stream(path: Path) -> list[dict[str, Any]]:
    """The assistant turns of an opencode event stream, in order.

    The fallback for a trial whose session export is missing -- a rollout
    that timed out, or a container that was torn down.  The published
    dataset keeps its timed-out rollouts (10,981 of 31,600 rows carry
    ``result: AgentTimeoutError``), so dropping them here would shrink the
    re-annotated dataset by a third.

    The event stream groups its parts by ``messageID``, which is the
    assistant message they belong to; the parts of one message arrive
    consecutively, so the order is preserved by grouping on first sight.
    """
    order: list[str] = []
    grouped: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        message_id = part.get("messageID")
        if not message_id or kind in ("step-start", "step-finish"):
            continue
        turn = grouped.get(message_id)
        if turn is None:
            turn = {"reasoning": "", "text": "", "tools": []}
            grouped[message_id] = turn
            order.append(message_id)
        if kind == "reasoning":
            turn["reasoning"] = "\n".join(
                x for x in (turn["reasoning"], part.get("text", "")) if x
            )
        elif kind == "text":
            turn["text"] = "\n".join(x for x in (turn["text"], part.get("text", "")) if x)
        elif kind == "tool":
            state = part.get("state") or {}
            turn["tools"].append(
                {
                    "tool": part.get("tool", ""),
                    "callID": part.get("callID", ""),
                    "input": state.get("input") or {},
                    "output": _tool_output(part),
                }
            )
    return [grouped[message_id] for message_id in order]


def instruction_of_trial(trial_dir: Path) -> str | None:
    """The instruction the trial ran with, read from the task directory.

    The trial's ``config.json`` points at the Harbor task directory, and that
    directory holds the ``instruction.md`` the agent was given -- the same
    bytes the dataset carries, so the wire form can be rebuilt from it.
    """
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # The trial's own config.json is flat ("task" at the top level), while
    # the job's result.json nests it under "config"; both are accepted.
    task_path = ((config.get("task") or {}).get("path")) or (
        ((config.get("config") or {}).get("task") or {}).get("path")
    )
    if not task_path:
        return None
    instruction_path = Path(task_path) / "instruction.md"
    if not instruction_path.exists():
        return None
    return instruction_path.read_text(encoding="utf-8").strip()


def convert_trial(
    trial_dir: Path,
    requests: RequestIndex,
    task_index: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Convert one Harbor trial directory into a dataset row."""
    # Harbor names a trial directory "<task directory>__<suffix>".  The task
    # directory of a repeated row carries the row index ("<task>__row00042"),
    # so the registry name is everything before the *last* "__".
    registry_name = trial_dir.name.rsplit("__", 1)[0]
    entry = (manifest or {}).get(registry_name, {})
    task = registry_name.split("__row", 1)[0]
    meta = task_index.get(task, {})

    # The session export is gzipped at full scale (the pilot's 73-step trial
    # produced 22 MB of JSON, and there are 31,600 trials), but the
    # uncompressed form is what the pilot wrote.
    session_path = trial_dir / "agent" / "opencode" / "session.json.gz"
    if not session_path.exists():
        session_path = trial_dir / "agent" / "opencode" / "session.json"
    event_path = trial_dir / "agent" / "opencode.txt"
    source = "session"

    wire_user: str | None = None
    instruction: str | None = None
    turns: list[dict[str, Any]] = []
    info: dict[str, Any] = {}

    if session_path.exists():
        session: dict[str, Any] | None = None
        try:
            if session_path.suffix == ".gz":
                import gzip

                with gzip.open(session_path, "rt", encoding="utf-8") as handle:
                    session = json.loads(handle.read())
            else:
                session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # A **truncated** export is not a broken export: the gzip stream
            # is intact but the JSON inside it stops mid-object, because
            # ``opencode export`` was killed while it was writing.  The
            # event stream is complete in that case, so the rollout is not
            # lost -- it is rebuilt from the events instead.
            session = None
        if session is not None:
            messages_raw = session.get("messages") or []
            info = session.get("info") or {}
            # The user message as the agent received it (the wire form).
            for message in messages_raw:
                if (message.get("info") or {}).get("role") == "user":
                    parts = message.get("parts") or []
                    wire_user = "\n".join(
                        part.get("text", "") for part in parts if part.get("type") == "text"
                    ).strip()
                    break
            if wire_user:
                # The clean instruction: the wire form without the quotes
                # opencode adds.
                instruction = (
                    wire_user[1:-1].replace('\\"', '"')
                    if wire_user.startswith('"')
                    else wire_user
                )
                turns = turns_from_session(session)

    if not turns:
        # The session export is missing or unusable -- a rollout that timed
        # out, a container that was torn down, or a truncated export.  The
        # event stream is still there.
        source = "event_stream"
        turns = turns_from_event_stream(event_path) if event_path.exists() else []
        raw_instruction = instruction_of_trial(trial_dir)
        if raw_instruction:
            instruction = raw_instruction
            wire_user = wire_wrap(instruction)

    if not wire_user or not instruction or not turns:
        return None

    body = requests.for_user_message(wire_user)
    system_prompt = system_prompt_of(body)
    tools = tools_of(body)

    # -- the strict {role, content} view ---------------------------------- #
    conversations: list[dict[str, str]] = []
    if system_prompt:
        conversations.append({"role": ROLE_USER, "content": system_prompt})
    conversations.append({"role": ROLE_USER, "content": wire_user})

    # -- the serve-parity view --------------------------------------------- #
    serve_messages: list[dict[str, Any]] = []
    if system_prompt:
        serve_messages.append({"role": ROLE_SYSTEM, "content": system_prompt})
    serve_messages.append({"role": ROLE_USER, "content": wire_user})

    n_tool_calls = 0
    n_assistant_turns = 0

    for turn in turns:
        n_assistant_turns += 1
        reasoning = turn.get("reasoning") or ""
        text = turn.get("text") or ""
        turn_tools = turn.get("tools") or []
        body_text = _assistant_body(reasoning, text)

        # -- the strict {role, content} view ------------------------------- #
        chunks: list[str] = []
        if body_text:
            chunks.append(body_text)
        for tool in turn_tools:
            chunks.append(
                TAG_TOOL_CALL
                + json.dumps(
                    {"tool": tool.get("tool", ""), "input": tool.get("input") or {}},
                    ensure_ascii=False,
                )
                + "</tool_call>"
            )
        if chunks:
            conversations.append(
                {"role": ROLE_ASSISTANT, "content": "\n\n".join(chunks)}
            )

        # -- the serve-parity view ------------------------------------------ #
        # The reasoning is its own field, exactly as the wire had it: on the
        # wire the assistant message carries ``content`` (the visible text,
        # often empty) and ``reasoning_content`` (the thinking).  Folding
        # the thinking into the content would make the sample *read* the
        # same but *tokenise* differently from what the model saw.
        serve: dict[str, Any] = {"role": ROLE_ASSISTANT, "content": text}
        if reasoning:
            serve["reasoning_content"] = reasoning
        if turn_tools:
            serve["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("tool", ""),
                        # No spaces after the separators -- see
                        # ``_tool_call``: opencode serialises with
                        # JavaScript's ``JSON.stringify`` and the training
                        # data has to be the bytes the model saw.
                        "arguments": json.dumps(
                            tool.get("input") or {},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    "id": tool.get("callID", ""),
                }
                for tool in turn_tools
            ]
            n_tool_calls += len(turn_tools)
        serve_messages.append(serve)

        # One assistant turn is one step, so all of its tool results go into
        # a single observation message.  The published dataset alternates
        # strictly -- a turn that produced three observations in a row would
        # break that invariant, and the loader that consumes it expects
        # user / assistant / user / assistant.
        if turn_tools:
            observations: list[str] = []
            for tool in turn_tools:
                observations.append(
                    f'{TAG_TOOL_RESULT} tool="{tool.get("tool", "")}"\n'
                    f"{tool.get('output') or ''}\n"
                    f"{TAG_TOOL_RESULT.replace('<', '</')}"
                )
                serve_messages.append(
                    {
                        "role": ROLE_TOOL,
                        "content": tool.get("output") or "",
                        "tool_call_id": tool.get("callID", ""),
                    }
                )
            conversations.append(
                {"role": ROLE_USER, "content": "\n\n".join(observations)}
            )

    tokens = info.get("tokens") or {}
    audit_path = trial_dir / "agent" / "rsi-opencode-audit.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.exists()
        else {}
    )
    result_path = trial_dir / "result.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    exception = (result.get("exception_info") or {}).get("exception_type")
    agent_info = result.get("agent_info") or {}
    agent_result = result.get("agent_result") or {}
    model_info = agent_info.get("model_info") or {}
    provider = model_info.get("provider") or "rsi_glm"
    model_name = model_info.get("name") or "glm5.3-flash"

    started_at = result.get("started_at")
    finished_at = result.get("finished_at")

    return {
        # -- the columns of the published dataset, same names -------------- #
        "agent": "opencode",
        "conversations": conversations,
        "date": finished_at or started_at,
        "episode": meta.get("episode"),
        "model": f"{provider}/{model_name}",
        "model_id": model_name,
        "model_provider": provider,
        **provenance(
            dataset=DATASET_OPENTHOUGHT,
            harness=HARNESS_OPENCODE,
            model=canonical_model(model_name, provider),
            model_id=model_name,
            model_provider=provider,
        ),
        "result": exception,
        "run_id": result.get("id") or meta.get("run_id"),
        "task": task,
        "trace_source": "main",
        "trial_name": trial_dir.name,
        # -- the provenance of the re-annotation --------------------------- #
        "family": entry.get("family") or meta.get("family"),
        "messages": serve_messages,
        "tools": tools,
        "instruction": instruction,
        "instruction_sha256": sha256(instruction),
        "instruction_wire": wire_user,
        "instruction_wire_sha256": sha256(wire_user),
        "conversion_source": source,
        # -- the join back onto the published dataset ----------------------- #
        # The registry emits one task per dataset row, so the trial answers
        # exactly the rows the manifest names.
        "registry_name": registry_name,
        "dataset_rows": entry.get("source_rows", []),
        "dataset_row": (entry.get("source_rows") or [None])[0],
        "image": entry.get("image"),
        "source_date": meta.get("date"),
        "source_episode": meta.get("episode"),
        "source_model": meta.get("model"),
        "source_model_provider": meta.get("model_provider"),
        "source_result": meta.get("result"),
        "source_run_id": meta.get("run_id"),
        "source_trial_name": meta.get("trial_name"),
        "source_trace_source": meta.get("trace_source"),
        # -- the statistics of the rollout --------------------------------- #
        "n_messages": len(conversations),
        "n_assistant_turns": n_assistant_turns,
        "n_tool_calls": n_tool_calls,
        "prompt_tokens": agent_result.get("n_input_tokens") or tokens.get("input"),
        "completion_tokens": agent_result.get("n_output_tokens") or tokens.get("output"),
        "cwd": info.get("directory"),
        "opencode_version": audit.get("container_version") or agent_info.get("version"),
        "system_prompt_sha256": sha256(system_prompt) if system_prompt else None,
        "has_system_prompt": system_prompt is not None,
        "started_at": started_at,
        "finished_at": finished_at,
        "trial_dir": str(trial_dir),
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

#: The invariants of the published dataset, checked on the converted rows.
EXPECTED_ROLES = {ROLE_USER, ROLE_ASSISTANT}
EXPECTED_KEYS = {"content", "role"}


def validate_row(row: dict[str, Any]) -> list[str]:
    """The problems of one row; an empty list means the row is well formed."""
    problems: list[str] = []
    conversations = row["conversations"]

    for index, message in enumerate(conversations):
        if set(message) != EXPECTED_KEYS:
            problems.append(f"msg{index}: keys are {sorted(message)}, expected {sorted(EXPECTED_KEYS)}")
        if message.get("role") not in EXPECTED_ROLES:
            problems.append(f"msg{index}: role {message.get('role')!r} is outside {EXPECTED_ROLES}")
        if not isinstance(message.get("content"), str):
            problems.append(f"msg{index}: content is not a string")

    if not conversations:
        problems.append("no messages")
        return problems
    if conversations[0]["role"] != ROLE_USER:
        problems.append("msg0 is not a user message")
    if len(conversations) > 1 and conversations[1]["role"] != ROLE_USER:
        problems.append("msg1 is not a user message")

    # After the two leading user messages the roles alternate: the system
    # prompt and the task instruction, then action / observation / action.
    for index in range(2, len(conversations)):
        expected = ROLE_ASSISTANT if index % 2 == 0 else ROLE_USER
        if conversations[index]["role"] != expected:
            problems.append(
                f"msg{index}: role {conversations[index]['role']!r}, expected {expected!r}"
            )

    # The serve-parity view has to pair every tool call with a result.
    calls = [
        call
        for message in row["messages"]
        if message.get("role") == ROLE_ASSISTANT
        for call in (message.get("tool_calls") or [])
    ]
    results = [m for m in row["messages"] if m.get("role") == ROLE_TOOL]
    if len(calls) != len(results):
        problems.append(f"{len(calls)} tool calls but {len(results)} tool results")
    call_ids = {call.get("id") for call in calls}
    result_ids = {message.get("tool_call_id") for message in results}
    if call_ids != result_ids:
        problems.append("tool_call_id / call id mismatch")

    # The clean instruction has to be the wire form without opencode's quotes.
    if row["instruction_wire"] != wire_wrap(row["instruction"]):
        problems.append("instruction / instruction_wire do not round-trip")
    if row["instruction_sha256"] != sha256(row["instruction"]):
        problems.append("instruction_sha256 does not match the instruction")
    if not row.get("has_system_prompt"):
        problems.append("no system prompt was captured for this trial")
    if not row.get("tools"):
        problems.append("no tool schemas were captured for this trial")
    return problems


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def load_task_index(path: Path) -> dict[str, dict[str, Any]]:
    """task id → the row of ``tasks/index.jsonl`` that describes it."""
    index: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return index
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task")
        if task and task not in index:
            index[task] = row
    return index


def find_trials(run_dir: Path) -> list[Path]:
    """The trial directories of a Harbor job output.

    Harbor writes its output as ``<batch dir>/<timestamp>/<trial>/``, so the
    trial directories are **two** levels below the run directory, not one.
    The search therefore walks down until it finds a directory that holds an
    ``agent/`` directory -- that is what makes a directory a trial -- and
    stops there, because everything below a trial belongs to the trial.
    """
    if not run_dir.is_dir():
        return []

    trials: list[Path] = []
    frontier = [run_dir]
    while frontier:
        directory = frontier.pop(0)
        try:
            entries = sorted(entry for entry in directory.iterdir() if entry.is_dir())
        except OSError:
            continue
        for entry in entries:
            if (entry / "agent").exists():
                trials.append(entry)
            else:
                frontier.append(entry)
    return trials


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """registry task name → the registry's record of it.

    The registry emits **one task per dataset row**: the first row of a task
    keeps the task's own name, the further rows carry the row index.  The
    manifest is what ties a finished trial back to the dataset rows it
    answers.
    """
    manifest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return manifest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        name = entry.get("task")
        if name:
            manifest[name] = entry
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Harbor job output directory")
    parser.add_argument(
        "--requests",
        default="/tmp/oc-requests.jsonl",
        help="the recording proxy's request log",
    )
    parser.add_argument(
        "--task-index",
        default=str(HERE / "tasks" / "index.jsonl"),
        help="the extracted task index, for the source-row metadata",
    )
    parser.add_argument(
        "--manifest",
        default=str(HERE / "registry" / "manifest.jsonl"),
        help="the registry manifest, for the trial -> dataset row mapping",
    )
    parser.add_argument(
        "--blob-dir",
        default="",
        help=(
            "the directory the recording proxy wrote its content-addressed "
            "system-prompt and tool-schema blobs to (--blob-dir there)"
        ),
    )
    parser.add_argument("--out", default=str(HERE / "dataset" / "pilot"))
    parser.add_argument(
        "--no-tools-column",
        action="store_true",
        help="do not embed the tool schemas in every row (they are the same for all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = RequestIndex(
        Path(args.requests),
        Path(args.blob_dir) if args.blob_dir else None,
    )
    task_index = load_task_index(Path(args.task_index))
    manifest = load_manifest(Path(args.manifest))

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    problems: dict[str, list[str]] = {}
    for trial in find_trials(Path(args.run)):
        row = convert_trial(trial, requests, task_index, manifest)
        if row is None:
            skipped.append(trial.name)
            continue
        found = validate_row(row)
        if found:
            problems[row["trial_name"]] = found
        rows.append(row)

    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {k: v for k, v in row.items() if k != "tools"}
            if not args.no_tools_column:
                payload["tools"] = row["tools"]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "run": str(args.run),
        "requests_file": str(args.requests),
        "n_recorded_requests": requests.n_requests,
        "n_agent_requests": requests.n_agent_requests,
        "n_side_requests": requests.n_side_requests,
        "n_unresolved_blobs": requests.n_unresolved_blobs,
        "n_distinct_user_messages": len(requests.by_user_message),
        "rows": len(rows),
        "skipped_trials": skipped,
        "rows_with_system_prompt": sum(1 for r in rows if r["has_system_prompt"]),
        "rows_with_tools": sum(1 for r in rows if r.get("tools")),
        "total_tool_calls": sum(r["n_tool_calls"] for r in rows),
        "total_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
        "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
        "rows_by_family": dict(Counter(r["family"] for r in rows)),
        "rows_by_result": dict(Counter(str(r["result"]) for r in rows)),
        "assistant_turns": sorted(r["n_assistant_turns"] for r in rows),
        "distinct_system_prompts": len(
            {r["system_prompt_sha256"] for r in rows if r["system_prompt_sha256"]}
        ),
        "distinct_tool_sets": len(
            {json.dumps(r["tools"], sort_keys=True) for r in rows if r.get("tools")}
        ),
        "validation": {
            "rows_with_problems": len(problems),
            "problems": problems,
        },
        "files": {"dataset": str(dataset_path)},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
