"""Write the converted rollouts out as a readable transcript.

The machine form of a rollout is the ``dataset.jsonl``: ``conversations`` for
the strict schema and ``messages`` for the serve-parity view.  That is what a
training run reads.  This script writes the *same* rollouts in a form a
human can read -- one file per rollout -- because "the full trajectory is on
disk" should mean something you can open and follow, not just a column in a
JSONL file.

The transcript is derived from the dataset, never from the trial directory
directly, so it cannot disagree with what a trainer would see.

Usage::

    python common/tools/dump_trajectories.py \
        --dataset runs/random-probe/dataset/dataset.jsonl \
        --out runs/random-probe/trajectories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    try:
        arguments = json.dumps(json.loads(function.get("arguments") or "{}"), indent=2)
    except json.JSONDecodeError:
        arguments = function.get("arguments") or ""
    return f"    -> {function.get('name', '?')}({arguments})"


def render(row: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"task      : {row.get('task')}")
    lines.append(f"trial     : {row.get('trial_name')}")
    lines.append(f"family    : {row.get('family')}")
    lines.append(f"image     : {row.get('image')}")
    lines.append(f"model     : {row.get('model')}")
    lines.append(f"result    : {row.get('result')}")
    lines.append(f"started   : {row.get('started_at')}")
    lines.append(f"finished  : {row.get('finished_at')}")
    lines.append(f"source    : {row.get('conversion_source')}")
    lines.append(f"instruction sha256 : {row.get('instruction_sha256')}")
    lines.append(
        f"turns     : {row.get('n_assistant_turns')} assistant, "
        f"{row.get('n_tool_calls')} tool calls, {row.get('n_messages')} messages"
    )
    lines.append(f"tokens    : prompt={row.get('prompt_tokens')} completion={row.get('completion_tokens')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("--- the instruction the agent was given " + "-" * 47)
    lines.append(row.get("instruction") or "")
    lines.append("")

    # The serve-parity view is the one that keeps the tool calls structured,
    # so the transcript follows it.
    for index, message in enumerate(row.get("messages") or []):
        role = message.get("role")
        if role == "system":
            lines.append("--- [system] " + "-" * 61)
            lines.append(message.get("content") or "")
            lines.append("")
        elif role == "user":
            lines.append("--- [user] " + "-" * 64)
            lines.append(message.get("content") or "")
            lines.append("")
        elif role == "assistant":
            lines.append(f"--- [assistant {index}] " + "-" * 51)
            content = message.get("content") or ""
            if content:
                lines.append(content)
            for call in message.get("tool_calls") or []:
                lines.append(render_tool_call(call))
            lines.append("")
        elif role == "tool":
            name = message.get("name") or "?"
            lines.append(f"--- [tool result: {name}] " + "-" * 48)
            lines.append(message.get("content") or "")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="the converted dataset.jsonl")
    parser.add_argument("--out", required=True, help="where to write the transcripts")
    parser.add_argument(
        "--combined",
        default=None,
        help="also write every transcript into this one file",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    written = 0
    for row in rows:
        name = row.get("task") or row.get("trial_name") or "row"
        path = out_dir / f"{name}.txt"
        path.write_text(render(row), encoding="utf-8")
        written += 1

    if args.combined:
        Path(args.combined).write_text(
            "\n".join(render(row) for row in rows), encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "rows": len(rows),
                "transcripts_written": written,
                "out": str(out_dir),
                "combined": args.combined,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
