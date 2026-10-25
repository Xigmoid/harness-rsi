#!/usr/bin/env python3
"""Turn the mini-swe-agent rollouts into an OpenAI-format SFT dataset.

Input
-----
The Harbor trials of ``run_mini_rollouts.py``: one directory per trial, each
with ``agent/mini-swe-agent.trajectory.json`` (mini-swe-agent's own
trajectory format) and ``agent/trajectory.json`` (Harbor's ATIF).  The
ledger (``progress.jsonl``) says which trials finished.

Output
------
One JSONL row per rollout, with **two views** of the same conversation:

``messages``
    The OpenAI-native view.  ``role`` is one of ``system`` / ``user`` /
    ``assistant``, the keys are ``content`` / ``role`` plus whatever the
    model actually emitted (``reasoning_content``,
    ``provider_specific_fields``, ``tool_calls`` when the protocol has
    them).  This is the view a tool-aware trainer wants.

``conversations``
    The schema of the published dataset
    (``open-thoughts/OpenThoughts-Agent-SFT-31.6K``): exactly the keys
    ``content`` and ``role``, and a role vocabulary of ``user`` and
    ``assistant`` only -- the system prompt is folded into the first user
    message, which is what terminus-2 did, and what the training code that
    loads the published dataset expects.

Both views are derived from the same source, and nothing is invented on the
way: the messages are the ones mini-swe-agent sent to the model, minus the
two internal keys (``extra``, ``function_call``) and the ``None``-valued
fields that litellm omits when it re-serialises a request.

Usage::

    python mini_sft/build_sft_dataset.py \\
        --run-root /private/tmp/rsi-harbor/mini-sft-full \\
        --out mini_sft/dataset/mini-sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.schema import (  # noqa: E402
    DATASET_OPENTHOUGHT,
    HARNESS_MINI,
    canonical_model,
    provenance,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: mini-swe-agent's internal keys: never on the wire, never in a sample.
INTERNAL_KEYS = ("extra", "function_call")

#: The observation mini writes when the *submission* command was never
#: executed.  The model never saw it, so it is not part of the sample.
NOT_EXECUTED = "action was not executed"


def strip(message: dict[str, Any]) -> dict[str, Any]:
    """Drop the internal keys and the ``None``-valued fields.

    litellm omits a field whose value is ``None`` when it re-serialises the
    conversation into the next request, so the wire's assistant message has
    no ``content`` key at all where the trajectory has ``content: null``.
    """
    return {
        key: value
        for key, value in message.items()
        if key not in INTERNAL_KEYS and value is not None
    }


def to_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """mini-swe-agent's trajectory -> the OpenAI message list.

    Two things are removed:

    * the ``role: "exit"`` message -- it is mini's bookkeeping, not a
      message;
    * the trailing observation of the submission command, which carries
      ``exception_info == "action was not executed"``: the command was never
      run, so the model never saw that observation.  Keeping it would put a
      message in the sample that the model did not receive.
    """
    messages = [
        m for m in (trajectory.get("messages") or []) if m.get("role") != "exit"
    ]
    # Strip the **trailing** observation: the model never received it.
    #
    # mini's loop is "model answers -> the tool runs -> the observation is
    # appended -> the model is called again".  When the episode ends right
    # after an observation, that observation was never sent back.  Both
    # endings land on the same message:
    #   * the submission command: mini intercepts
    #     COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT and does not run it, so the
    #     observation carries exception_info == "action was not executed";
    #   * a timeout: the tool really ran, the observation carries a normal
    #     returncode, and the episode was cut before the next model call --
    #     the model still never saw it.
    #
    # After the strip the sample is exactly "the N messages of the rollout's
    # last request" plus "the model's final output" = wire + 1.
    # **All** of the trailing observations go, not just the last one.  A
    # single assistant message may carry several tool calls -- mini runs
    # them in parallel and appends one observation per call -- so a rollout
    # that ends on a parallel call ends with several observations, and none
    # of them was sent back to the model.  Dropping only the last one
    # would leave the earlier calls answered and the later ones dangling,
    # which is a sample that says the model saw something it did not.
    while messages and messages[-1].get("role") == "tool":
        messages.pop()
    return [strip(m) for m in messages]


def to_conversations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The published-dataset view: ``{content, role}``, ``user``/``assistant``.

    The system prompt is folded into the first user message, exactly the way
    terminus-2 put it there -- the published dataset has no ``system`` role
    at all.  The reasoning stays in the content, wrapped in ``<think>``,
    which is where the published dataset keeps it.
    """
    system = ""
    body: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system = message.get("content") or ""
            continue
        if role == "tool":
            # A tool result is an observation: in the published schema an
            # observation is a user message.
            body.append(
                {
                    "role": "user",
                    "content": message.get("content") or "",
                }
            )
            continue
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        if role == "user" and not body and system:
            content = f"{system}\n\n{content}"
            system = ""
        body.append({"role": role, "content": content})
    return body


def summarize(messages: list[dict[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter(m.get("role") for m in messages)
    return {
        "n_messages": len(messages),
        "roles": dict(roles),
        "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
        "n_assistant_with_reasoning": sum(
            1 for m in messages if m.get("reasoning_content")
        ),
    }


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


#: The characters that break a line-oriented reader.
#:
#: A JSON record is one line, and the line ends at ``\n``.  But Python's
#: ``str.splitlines()`` -- which half the tools that read JSONL use -- also
#: breaks on ``\v``, ``\f``, ``\x1c``, ``\x1d``, ``\x1e``, ``\x85``,
#: ``\u2028`` and ``\u2029``.  The model's output contains such characters
#: (a Hungarian name in a CSV, a line separator in a report), and a record
#: containing one of them is then read as two half-records: the first one
#: fails to parse and the second one is a mystery.
#:
#: They are escaped here, which keeps the record one line *and* keeps the
#: text exactly as the model wrote it -- ``\u2028`` is the same character as
#: U+2028 after the JSON parser has read it.
LINE_BREAK_ESCAPES = {
    codepoint: f"\\u{ord(codepoint):04x}"
    for codepoint in (
        "\x0b",
        "\x0c",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
    )
}


def dumps_row(row: dict[str, Any]) -> str:
    """One record, on one line."""
    return json.dumps(row, ensure_ascii=False).translate(LINE_BREAK_ESCAPES)


def load_instructions(path: Path) -> dict[str, list[dict[str, Any]]]:
    """The extracted instructions, keyed by registry task name."""
    table: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        table.setdefault(record["task"], []).append(record)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        default=str(Path.home() / ".rsi-harbor" / "openthoughts-mini"),
        help="the Harbor output directory (the one with progress.jsonl)",
    )
    ap.add_argument(
        "--instructions",
        default=str(HERE / "tasks" / "user_instructions.jsonl"),
    )
    ap.add_argument(
        "--out",
        default=str(HERE / "mini_sft" / "dataset" / "mini-sft.jsonl"),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--include-failed",
        action="store_true",
        help="also convert the trials that failed (they often hold a usable "
        "partial trajectory)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    ledger_path = run_root / "progress.jsonl"
    if not ledger_path.exists():
        print(f"{ledger_path} does not exist; run the rollouts first", file=sys.stderr)
        return 2

    ledger = load_ledger(ledger_path)
    instructions = load_instructions(Path(args.instructions))

    # The last record per task wins: a task that failed and then succeeded
    # appears twice, and the successful trial is the one to keep.
    latest: dict[str, dict[str, Any]] = {}
    for record in ledger:
        latest[record["registry_name"]] = record

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The dataset is written to a temporary file and renamed into place, so a
    # reader never sees a half-written record: the rollouts keep arriving
    # while this runs, and a torn line in the deliverable is not acceptable.
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    n_rows = 0
    n_missing = 0
    n_no_trajectory = 0
    by_family: Counter[str] = Counter()
    roles_seen: Counter[str] = Counter()
    warnings: Counter[str] = Counter()

    with tmp_path.open("w", encoding="utf-8") as handle:
        for name, record in sorted(latest.items()):
            if not args.include_failed and not record.get("ok"):
                continue
            trial_dir = Path(record["trial_dir"])
            trajectory_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
            if not trajectory_path.exists():
                n_no_trajectory += 1
                continue
            try:
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                n_no_trajectory += 1
                continue

            messages = to_messages(trajectory)
            if not messages:
                n_no_trajectory += 1
                continue
            conversations = to_conversations(messages)

            # The instruction the agent was given, from the extraction.
            task_records = instructions.get(name, [])
            instruction = task_records[0]["instruction"] if task_records else None
            if instruction is None:
                warnings["no_instruction"] += 1

            row = {
                "messages": messages,
                "conversations": conversations,
                "source": "mini-swe-agent",
                **provenance(
                    dataset=DATASET_OPENTHOUGHT,
                    harness=HARNESS_MINI,
                    model=canonical_model(record.get("model")),
                    model_id=record.get("model"),
                    model_provider=record.get("api_base"),
                ),
                "task": name,
                "family": record.get("family"),
                "dataset_rows": record.get("dataset_rows", []),
                "trial_name": record.get("trial_name"),
                "ok": bool(record.get("ok")),
                "exception": record.get("exception"),
                "instruction": instruction,
                **summarize(messages),
            }
            handle.write(dumps_row(row) + "\n")
            n_rows += 1
            by_family[str(record.get("family"))] += 1
            for role, count in row["roles"].items():
                roles_seen[role] += count
            if args.limit and n_rows >= args.limit:
                break

    # The rename is what makes the dataset readable at all times: a reader
    # sees either the old file or the new one, never a half-written record.
    os.replace(tmp_path, out_path)

    print(f"wrote {out_path}  ({n_rows} rows, {out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  trials without a trajectory : {n_no_trajectory}")
    print(f"  by family                   : {dict(by_family)}")
    print(f"  roles across all samples    : {dict(roles_seen)}")
    if warnings:
        print(f"  warnings                    : {dict(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
