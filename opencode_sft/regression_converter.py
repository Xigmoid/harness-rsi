#!/usr/bin/env python3
"""Pin the shared opencode converter to the audited pilot implementation.

The opencode conversion exists in two places: the pilot's
``to_openai_sft.py`` (written first, checked against the wire by hand) and
the shared ``opencode_sft/build_sft_dataset.py`` (written for the
re-annotation, used for both datasets).  Two implementations of the same
thing is one too many -- they will drift, and the one that drifts is the one
nobody looks at.

This script makes the shared converter answer to the audited one: for every
pilot rollout, the two must produce **byte-identical** ``messages``.  When
they disagree, the diff says exactly which message and which key, so the
fix is a two-minute job instead of a mystery.

Usage::

    python3 opencode_sft/regression_converter.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PILOT = HERE.parent / "work" / "pilot" / "terminal-5task"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PILOT))

from build_sft_dataset import RequestIndex, convert_trial  # noqa: E402
from to_openai_sft import opencode_to_openai  # noqa: E402

CAPTURE = PILOT / "captures" / "opencode-wire.jsonl"


def load_capture_bodies(path: Path) -> dict[str, dict[str, Any]]:
    """The captured agent requests, keyed by their first user message."""
    bodies: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return bodies
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
            if not isinstance(body, dict) or not body.get("tools"):
                continue
            for message in body.get("messages") or []:
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        bodies.setdefault(content, body)
                    break
    return bodies


def main() -> int:
    raw_dir = PILOT / "opencode" / "raw"
    sft_dir = PILOT / "opencode" / "sft"
    rollouts = sorted(p.name for p in raw_dir.glob("*.json"))
    if not rollouts:
        print(f"{raw_dir} holds no rollout", file=sys.stderr)
        return 2

    bodies = load_capture_bodies(CAPTURE)
    if not bodies:
        print(f"{CAPTURE} holds no agent request", file=sys.stderr)
        return 2

    n_ok = 0
    n_bad = 0
    findings: list[str] = []

    with tempfile.TemporaryDirectory() as temporary:
        for name in rollouts:
            session = json.loads((raw_dir / name).read_text(encoding="utf-8"))
            reference = json.loads((sft_dir / name).read_text(encoding="utf-8"))

            # The system prompt the model saw, from the capture.
            wire_user = None
            for message in session.get("messages") or []:
                if (message.get("info") or {}).get("role") == "user":
                    parts = message.get("parts") or []
                    wire_user = "\n".join(
                        part.get("text", "")
                        for part in parts
                        if part.get("type") == "text"
                    ).strip()
                    break
            body = bodies.get(wire_user or "")
            system = None
            if body:
                for message in body.get("messages") or []:
                    if message.get("role") == "system":
                        content = message.get("content")
                        system = (
                            content
                            if isinstance(content, str)
                            else json.dumps(content, ensure_ascii=False)
                        )
                        break

            # What the audited implementation produces.
            expected = opencode_to_openai(session, system)

            # What the shared converter produces, on the same inputs.
            trial_dir = Path(temporary) / f"trial-{name[:-5]}"
            agent_dir = trial_dir / "agent" / "opencode"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "session.json").write_text(
                json.dumps(session, ensure_ascii=False), encoding="utf-8"
            )
            index = RequestIndex(CAPTURE)
            row = convert_trial(trial_dir, index, {}, {})
            got = (row or {}).get("messages") or []

            if got == expected:
                n_ok += 1
                continue
            n_bad += 1
            detail = []
            if len(got) != len(expected):
                detail.append(
                    f"{len(got)} messages, the audited implementation has {len(expected)}"
                )
            for index_, (a, b) in enumerate(zip(got, expected)):
                if a != b:
                    keys = sorted(set(a) | set(b))
                    differing = [k for k in keys if a.get(k) != b.get(k)]
                    detail.append(
                        f"message {index_} differs in {differing}: "
                        + "; ".join(
                            f"{k}: shared={str(b.get(k))[:80]!r} "
                            f"audited={str(a.get(k))[:80]!r}"
                            for k in differing[:3]
                        )
                    )
                    break
            findings.append(f"{name}: " + " | ".join(detail))

    print(f"rollouts compared : {len(rollouts)}")
    print(f"byte-identical    : {n_ok}")
    print(f"differing         : {n_bad}")
    if findings:
        print("\nfindings:")
        for finding in findings:
            print(f"  - {finding}")
        print("\nresult    : FAILED -- the shared converter drifted from the audited one")
        return 1
    print("\nresult    : OK -- the shared converter is byte-identical to the audited one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
