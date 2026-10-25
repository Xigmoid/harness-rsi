#!/usr/bin/env python3
"""Check that the converted opencode rows *are* the model-side conversation.

The SFT dataset is rebuilt from opencode's own session export, which is a
*part tree*, not a message list.  That rebuild is only legitimate if it
reproduces the messages the model actually received -- so this script puts a
recording proxy in front of the endpoint for a few trials and compares the
converted rows with the requests that were really sent.

The rule
--------

For every rollout:

    the converted ``messages`` must equal the **last** recorded request's
    messages, byte for byte, and must be exactly one message longer -- the
    extra one being the model's final output, which was never sent back.

"Byte for byte" means the same keys with the same values.  Three things had
to be got right for that to hold, and all three were found by running this
comparison rather than by reading the code:

* ``tool_calls[].function.arguments`` is a JSON *string*, serialised the way
  JavaScript's ``JSON.stringify`` does it -- no spaces after the separators.
  Python's ``json.dumps`` default puts a space after every ``:`` and ``,``,
  which parses to the same object but is a different token sequence.
* the reasoning is its own field (``reasoning_content``), not folded into
  the content.  On the wire the assistant message carries the visible text
  in ``content`` (often empty) and the thinking in ``reasoning_content``.
* the tool result messages carry ``role``, ``content`` and ``tool_call_id``
  -- and nothing else.

Usage::

    # 1. record every request in full
    python3 common/record_proxy.py --port 8911 --host 0.0.0.0 \\
        --upstream http://YOUR-ENDPOINT:PORT/v1 --out /tmp/oc-wire.jsonl &

    # 2. run the trials through the proxy
    python opencode_sft/run_rollouts.py --registry work/registry_clawgym \\
        --manifest /tmp/pilot-manifest.jsonl --run-root /tmp/pilot \\
        --api-base http://host.docker.internal:8911/v1 --enable-verifier

    # 3. convert and check
    python opencode_sft/clawgym/build_sft_dataset.py \\
        --run-root /tmp/pilot --requests /tmp/oc-wire.jsonl --out /tmp/pilot-sft.jsonl
    python3 opencode_sft/verify_wire.py \\
        --capture /tmp/oc-wire.jsonl --dataset /tmp/pilot-sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: The title request opencode makes before every rollout.
TITLE_MARK = "Generate a title for this conversation"


def first_user_message(body: dict[str, Any]) -> str | None:
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) and content else None
    return None


def load_capture(path: Path) -> dict[str, list[dict[str, Any]]]:
    """The recorded agent requests, keyed by the user message they carry.

    The requests of one rollout all carry the same first user message -- the
    task instruction -- so that is what ties a request to a rollout.
    """
    rollouts: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return rollouts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
            if not body.get("tools"):
                # The title generator and the connectivity probe: neither is
                # an agent request.
                continue
            user = first_user_message(body)
            if not user or TITLE_MARK in user:
                continue
            rollouts.setdefault(user, []).append(body)
    return rollouts


def normalize(message: dict[str, Any]) -> dict[str, Any]:
    """The message as the comparison sees it.

    Key order is irrelevant in JSON, so the message is rebuilt with its keys
    sorted; everything else -- the values, the key set -- is compared as it
    is.
    """
    return {key: message[key] for key in sorted(message)}


def compare(
    converted: list[dict[str, Any]], request: dict[str, Any]
) -> list[str]:
    """The differences between the converted row and the last request."""
    problems: list[str] = []
    expected = request.get("messages") or []
    if len(converted) != len(expected) + 1:
        problems.append(
            f"the row has {len(converted)} messages, the request had "
            f"{len(expected)} (the row must be exactly one longer)"
        )
    for index, message in enumerate(expected):
        if index >= len(converted):
            problems.append(f"the row is missing message {index}")
            break
        want = normalize(message)
        got = normalize(converted[index])
        if want == got:
            continue
        # Report the first differing key, so the finding is actionable.
        differing = [
            key
            for key in set(want) | set(got)
            if want.get(key) != got.get(key)
        ]
        problems.append(
            f"message {index} differs in {sorted(differing)}: "
            + "; ".join(
                f"{key}: wire={str(want.get(key))[:120]!r} "
                f"row={str(got.get(key))[:120]!r}"
                for key in sorted(differing)[:3]
            )
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="the full-body request log")
    parser.add_argument("--dataset", required=True, help="the converted dataset")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rollouts = load_capture(Path(args.capture))
    if not rollouts:
        print("the capture holds no agent request", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    with Path(args.dataset).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit:
        rows = rows[: args.limit]

    n_ok = 0
    n_bad = 0
    n_unmatched = 0
    findings: list[str] = []

    for row in rows:
        messages = row.get("messages") or []
        wire_user = row.get("instruction_wire")
        if not wire_user:
            # Fall back to the first user message of the row.
            for message in messages:
                if message.get("role") == "user":
                    wire_user = message.get("content")
                    break
        bodies = rollouts.get(wire_user or "")
        if not bodies:
            n_unmatched += 1
            findings.append(f"{row.get('task')}: no recorded request for this rollout")
            continue
        last = bodies[-1]
        problems = compare(messages, last)
        if problems:
            n_bad += 1
            findings.append(f"{row.get('task')}: " + " | ".join(problems[:3]))
        else:
            n_ok += 1

    print(f"capture   : {args.capture}")
    print(f"dataset   : {args.dataset}")
    print(f"rollouts captured: {len(rollouts)}")
    print(f"rows checked     : {len(rows)}")
    print(f"byte-identical   : {n_ok}")
    print(f"differing        : {n_bad}")
    print(f"unmatched        : {n_unmatched}")
    if findings:
        print("\nfindings:")
        for finding in findings[:20]:
            print(f"  - {finding}")
    if n_bad or n_unmatched:
        print("\nresult    : FAILED -- the conversion is not the wire")
        return 1
    print("\nresult    : OK -- every row is the model's own conversation, plus its final output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
