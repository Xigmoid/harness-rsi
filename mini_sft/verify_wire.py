#!/usr/bin/env python3
"""Check that mini-swe-agent's trajectory *is* the model-side conversation.

The SFT dataset is built from the harness's own trajectory.  That is only
legitimate if the trajectory is what the model actually received -- and for
mini-swe-agent it nearly is: the messages it records are the ones it sent,
plus two internal keys (``extra``, ``function_call``) and the ``None``-valued
fields that litellm omits when it re-serialises a request.

This script proves it for a sample of the Harbor trials, by putting a
recording proxy in front of the endpoint for a few trials and comparing the
recorded request with the trajectory the trial wrote.

The rule is the one the pilot established (see
``dataset/terminal-5task/build_sft.py``):

    the converted messages must equal the **last** recorded request's
    messages, byte for byte, and must be exactly one message longer -- the
    extra one being the model's final output, which was never sent back.

The comparison itself is imported from the pilot's converter, so the two
implementations cannot drift apart.

Usage::

    # 1. record
    python3 common/record_proxy.py --port 8901 --host 127.0.0.1 \\
        --upstream http://YOUR-ENDPOINT:PORT/v1 --out /tmp/mini-wire.jsonl &

    # 2. run the trials through the proxy (the config's api_base points at it)
    python mini_sft/run_rollouts.py \\
        --min-config mini_sft/config/mini-config-proxy.yaml \\
        --tasks swesmith-00000_copy0001 --batch-size 1 --max-batches 1

    # 3. check
    python mini_sft/verify_wire.py \\
        --capture /tmp/mini-wire.jsonl --run-root /private/tmp/rsi-harbor/mini-sft-full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: The reference implementation of the conversion, from the pilot.
#:
#: The pilot's *artefacts* (the captures, the raw harness output) are
#: generated, so they are not in the repository -- but the converter that the
#: wire check compares against is source code, and it is.
sys.path.insert(0, str(REPO / "common" / "pilot"))

from to_openai_sft import mini_to_openai  # noqa: E402

#: The title request opencode makes; mini has no equivalent, but the marker
#: is kept so a mixed capture is filtered the same way.
TITLE_MARK = "Generate a title for this conversation"


def key_of(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True, help="the recorded requests")
    ap.add_argument(
        "--run-root",
        default=str(Path.home() / ".rsi-harbor" / "openthoughts-mini"),
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    capture_path = Path(args.capture)
    records = [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"captured requests: {len(records)}")

    ledger_path = Path(args.run_root) / "progress.jsonl"
    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {record["registry_name"]: record for record in ledger}

    n_ok = 0
    n_checked = 0
    failures = []
    for name, record in sorted(latest.items()):
        if args.limit and n_checked >= args.limit:
            break
        trajectory_path = (
            Path(record["trial_dir"]) / "agent" / "mini-swe-agent.trajectory.json"
        )
        if not trajectory_path.exists():
            continue
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        converted = mini_to_openai(trajectory)

        # The rollout is identified by its session id, which is the only
        # thing that separates two replicas of the same instruction.
        #
        # The dataset contains many copies of one task -- superuser's
        # copy0000 and copy0002 have byte-identical instructions -- so a
        # capture attributed by content alone mixes their conversations and
        # the check fails on a perfectly good trajectory.  mini sends
        # ``X-Session-ID: <trial name>__agent`` with every request, and the
        # recording proxy stores it.
        session_id = f"{record['trial_name']}__agent"
        hits = [
            r["body"]["messages"]
            for r in records
            if r.get("session_id") == session_id
        ]

        if not hits:
            # The whole first user message, not a prefix of its first line.
            # A 40-character prefix is not unique either: two superuser
            # tasks open with the same "Please solve this issue: I'm
            # currently t", and a prefix match attributes one trial's
            # requests to the other.
            first_user = ""
            for message in trajectory.get("messages") or []:
                if message.get("role") == "user":
                    first_user = message.get("content") or ""
                    break
            if not first_user:
                continue
            hits = [
                r["body"]["messages"]
                for r in records
                if key_of(r["body"]["messages"]) == first_user
            ]
            if hits:
                print(
                    f"  ! {name}: no session id in the capture, fell back to "
                    f"the task instruction ({len(hits)} hits)"
                )
        if not hits:
            failures.append((name, "no captured request for this task"))
            n_checked += 1
            continue

        wire = hits[-1]
        n = len(wire)
        if len(converted) != n + 1:
            failures.append(
                (name, f"length {len(converted)}, expected {n + 1} (wire {n} + 1)")
            )
        elif converted[:n] != wire:
            for i, (a, b) in enumerate(zip(converted[:n], wire)):
                if a != b:
                    diff = sorted(
                        {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                    )
                    failures.append(
                        (name, f"[{i}] role={a.get('role')} differs in {diff}")
                    )
                    break
        else:
            n_ok += 1
            print(
                f"  ✓ {name}: 前 {n} 条与 wire 逐字一致，"
                f"多出 1 条 role={converted[-1].get('role')}"
            )
        n_checked += 1

    print(f"\n{n_ok}/{n_checked} 通过")
    for name, why in failures:
        print(f"  ✗ {name}: {why}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
