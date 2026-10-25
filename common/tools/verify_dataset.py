"""Check a converted SFT dataset against the published one.

The converter (``build_sft_dataset.py``) validates the rows while it builds
them, but a generator that checks its own output proves nothing: the file on
disk is what a training run will read.  This script therefore reads the
written ``dataset.jsonl`` back and checks it against the **original**
published dataset, which is the only ground truth available for the one
property that matters most -- the task instruction must still be the same
bytes.

Checks, per row:

1.  the instruction still matches a task description of the published
    dataset for that task, byte for byte (sha256);
2.  ``conversations`` keeps the published schema: only ``user`` / ``assistant``
    roles, only ``content`` / ``role`` keys, strict alternation after the two
    leading user messages;
3.  ``messages`` is a faithful serve-parity view: every tool call has a
    matching tool result, the ids pair up, and the arguments parse;
4.  the two views agree -- the tool calls flattened into ``conversations``
    are the tool calls structured in ``messages``;
5.  the system prompt and the tool schemas are the ones the model was
    actually offered (cross-checked against the recorded request);
6.  nothing invisible slipped into the text.

Usage::

    python common/tools/verify_dataset.py \
        --dataset opencode_sft/dataset/pilot/dataset.jsonl \
        --requests /tmp/oc-requests.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from hf_parquet import published_parquet_default
from typing import Any

HERE = Path(__file__).resolve().parent

TASK_MARKER = "Task Description:"
STATE_MARKER = "Current terminal state:"

EXPECTED_ROLES = {"user", "assistant"}
EXPECTED_KEYS = {"content", "role"}

#: The tags the converted view uses for the flattened tool calls.
TAG_TOOL_CALL = "<tool_call>"
TAG_TOOL_CALL_CLOSE = "</tool_call>"

#: Characters that must not appear in the text of a training sample.
INVISIBLE = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE",
    "\u2028": "LINE SEPARATOR",
    "\u2029": "PARAGRAPH SEPARATOR",
    "\u00ad": "SOFT HYPHEN",
    "\u180e": "MONGOLIAN VOWEL SEPARATOR",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wire_wrap(text: str) -> str:
    return '"' + text.replace('"', '\\"') + '"'


def extract_instruction(message: str) -> str | None:
    """Cut the task instruction out of the first message of a published row.

    Mirrors ``extract_tasks.extract_instruction`` exactly -- the two have to
    agree, otherwise the comparison below would be against a different rule
    than the one that produced the task files.
    """
    marker_at = message.find(TASK_MARKER)
    if marker_at < 0:
        return None
    body = message[marker_at + len(TASK_MARKER) :]
    state = re.search(r"\n[ \t]*" + re.escape(STATE_MARKER), body)
    if state is None:
        return None
    instruction = body[: state.start()].strip()
    return instruction or None


def load_published_instructions(path: Path) -> dict[str, set[str]]:
    """task id → the set of task-description hashes the published row has.

    Two sources are accepted, because the ground truth for "the instruction
    the agent was given" is not always the published parquet:

    * a **parquet** file -- the published dataset itself;
    * a **JSONL** file of ``{"task": ..., "instruction_sha256": ...}`` lines.
      That is what the random probe uses: its instructions are generated, so
      the file that holds them is the ground truth.
    """
    if path.suffix == ".jsonl":
        out: dict[str, set[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task = record.get("task")
            digest = record.get("instruction_sha256")
            if task and digest:
                out.setdefault(task, set()).add(digest)
        return out

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    tasks = table.column("task").to_pylist()
    conversations = table.column("conversations").to_pylist()
    out = {}
    for task, conv in zip(tasks, conversations):
        if not conv:
            continue
        first = conv[0].get("content")
        if not isinstance(first, str):
            continue
        instruction = extract_instruction(first)
        if instruction is None:
            continue
        out.setdefault(task, set()).add(sha256(instruction))
    return out


def read_lines(path: Path) -> list[str]:
    """The lines of a text file, transparently gunzipping a ``.gz`` one."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def load_recorded_requests(path: Path) -> dict[str, dict[str, Any]]:
    """wire user message → the system prompt and tools of that request."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in read_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        body = record.get("body")
        if not isinstance(body, dict) or not body.get("tools"):
            continue
        user = next(
            (
                m.get("content")
                for m in body.get("messages") or []
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ),
            None,
        )
        if user is None or user in out:
            continue
        system = next(
            (
                m.get("content")
                for m in body.get("messages") or []
                if m.get("role") == "system" and isinstance(m.get("content"), str)
            ),
            None,
        )
        out[user] = {"system_prompt": system, "tools": body.get("tools")}
    return out


def check_row(
    row: dict[str, Any],
    published: dict[str, set[str]],
    recorded: dict[str, dict[str, Any]],
) -> list[str]:
    problems: list[str] = []
    task = row.get("task")
    conversations = row.get("conversations") or []
    messages = row.get("messages") or []

    # -- 1. the instruction is still the published one -------------------- #
    published_hashes = published.get(task, set())
    if not published_hashes:
        problems.append(f"task {task} is not in the published dataset")
    elif row.get("instruction_sha256") not in published_hashes:
        problems.append(
            f"instruction_sha256 {row.get('instruction_sha256', '')[:16]} matches no "
            f"published task description of {task} "
            f"(published: {sorted(h[:16] for h in published_hashes)})"
        )

    if row.get("instruction_wire") != wire_wrap(row.get("instruction", "")):
        problems.append("instruction / instruction_wire do not round-trip")
    if row.get("instruction_sha256") != sha256(row.get("instruction", "")):
        problems.append("instruction_sha256 does not match the instruction")

    # -- 2. the published schema ------------------------------------------ #
    for index, message in enumerate(conversations):
        if set(message) != EXPECTED_KEYS:
            problems.append(f"msg{index}: keys {sorted(message)} != {sorted(EXPECTED_KEYS)}")
        if message.get("role") not in EXPECTED_ROLES:
            problems.append(f"msg{index}: role {message.get('role')!r} outside {EXPECTED_ROLES}")
        if not isinstance(message.get("content"), str):
            problems.append(f"msg{index}: content is not a string")
    if len(conversations) < 2:
        problems.append("fewer than two messages")
    else:
        if conversations[0]["role"] != "user":
            problems.append("msg0 is not a user message")
        if conversations[1]["role"] != "user":
            problems.append("msg1 is not a user message")
        for index in range(2, len(conversations)):
            expected = "assistant" if index % 2 == 0 else "user"
            if conversations[index]["role"] != expected:
                problems.append(
                    f"msg{index}: role {conversations[index]['role']!r}, expected {expected!r}"
                )

    # -- 3. the serve-parity view ----------------------------------------- #
    calls = [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    results = [m for m in messages if m.get("role") == "tool"]
    if len(calls) != len(results):
        problems.append(f"{len(calls)} tool calls but {len(results)} tool results")
    if {c.get("id") for c in calls} != {r.get("tool_call_id") for r in results}:
        problems.append("tool call ids and tool result ids do not pair up")
    for call in calls:
        if call.get("type") != "function":
            problems.append(f"tool call {call.get('id')} has type {call.get('type')!r}")
        function = call.get("function") or {}
        if not function.get("name"):
            problems.append(f"tool call {call.get('id')} has no name")
        try:
            arguments = json.loads(function.get("arguments") or "")
        except json.JSONDecodeError:
            problems.append(f"tool call {call.get('id')} has unparsable arguments")
        else:
            if not isinstance(arguments, dict):
                problems.append(f"tool call {call.get('id')} arguments are not an object")
    if messages and messages[0].get("role") != "system":
        problems.append("the messages view does not start with a system message")
    if len(messages) < 2 or messages[1].get("role") != "user":
        problems.append("the messages view does not have the task as its second message")

    # -- 4. the two views agree -------------------------------------------- #
    flat_calls = re.findall(
        re.escape(TAG_TOOL_CALL) + r"(\{.*?\})" + re.escape(TAG_TOOL_CALL_CLOSE),
        "\n\n".join(
            m["content"] for m in conversations if m["role"] == "assistant"
        ),
        re.S,
    )
    if len(flat_calls) != len(calls):
        problems.append(
            f"{len(flat_calls)} tool calls flattened into conversations, "
            f"{len(calls)} in messages"
        )
    else:
        for flat, structured in zip(flat_calls, calls):
            try:
                payload = json.loads(flat)
            except json.JSONDecodeError:
                problems.append("a flattened tool call is not valid JSON")
                continue
            if payload.get("tool") != structured["function"]["name"]:
                problems.append("the two views name different tools")
            if payload.get("input") != json.loads(structured["function"]["arguments"]):
                problems.append("the two views carry different tool arguments")

    # -- 5. the prompt and the tools are the ones the model was offered ---- #
    recorded_request = recorded.get(row.get("instruction_wire", ""))
    if recorded_request is None:
        problems.append("no recorded request matches this trial")
    else:
        if recorded_request["system_prompt"] != conversations[0]["content"]:
            problems.append("the system prompt differs from the recorded request")
        if row.get("tools") and recorded_request["tools"] != row.get("tools"):
            problems.append("the tool schemas differ from the recorded request")
        if row.get("system_prompt_sha256") != sha256(conversations[0]["content"]):
            problems.append("system_prompt_sha256 does not match the system prompt")

    # -- 6. nothing invisible --------------------------------------------- #
    for index, message in enumerate(conversations):
        for char, name in INVISIBLE.items():
            if char in message["content"]:
                problems.append(f"msg{index} contains a {name}")
    for column in ("instruction", "instruction_wire"):
        for char, name in INVISIBLE.items():
            if char in (row.get(column) or ""):
                problems.append(f"{column} contains a {name}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="the converted dataset.jsonl")
    parser.add_argument(
        "--published",
        default=published_parquet_default(),
        help="the published parquet, for the ground truth",
    )
    parser.add_argument(
        "--requests",
        default="/tmp/oc-requests.jsonl",
        help="the recording proxy's request log",
    )
    parser.add_argument("--report", default=None, help="write the report as JSON here")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    published = load_published_instructions(Path(args.published))
    recorded = load_recorded_requests(Path(args.requests))

    failures: dict[str, list[str]] = {}
    for row in rows:
        problems = check_row(row, published, recorded)
        if problems:
            failures[row.get("trial_name", "?")] = problems

    report = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "rows_ok": len(rows) - len(failures),
        "rows_with_problems": len(failures),
        "problems": failures,
        "rows_by_family": dict(Counter(r.get("family") for r in rows)),
        "total_tool_calls": sum(r.get("n_tool_calls") or 0 for r in rows),
        "total_messages": sum(r.get("n_messages") or 0 for r in rows),
        "published_tasks_covered": len({r.get("task") for r in rows}),
        "published_task_instructions_known": len(published),
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
