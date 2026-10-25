#!/usr/bin/env python3
"""Extract **every** user instruction of OpenThoughts-Agent-SFT-31.6K to one JSONL.

The dataset stores each rollout as ``conversations``: a list of
``{"role", "content"}`` messages whose first entry is the **user** message.
terminus-2 puts its system prompt and the task instruction into that one
message::

    <terminus-2 system prompt>
    ...
    Task Description:
    <the task instruction>

    Current terminal state:
    Current Terminal Screen:
    <a dump of the terminal>

So the instruction is what sits between ``Task Description:`` and
``Current terminal state:``.  The split is done by
:func:`extract_tasks.extract_instruction` -- the very same function that built
``tasks/index.jsonl`` and the Harbor registry -- so the instructions here are
byte-identical to the ones the registry hands to the agent.

Output: one record per **dataset row** (31,600), not per unique task.  A task
id is not a unique key: 766 tasks carry two instruction variants, and the
dataset rows point at different variants, so the row is the unit that has to
stay aligned with the published trajectory.

Usage::

    python common/registry/extract_user_instructions.py \\
        --out work/tasks/user_instructions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from extract_tasks import (  # noqa: E402
    REPO_ID,
    extract_instruction,
    family_of,
    first_user_message,
    parquet_files,
    download,
    iter_rows,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def records(repo: str, revision: str | None, limit: int | None) -> Iterator[dict[str, Any]]:
    files = parquet_files(repo, revision)
    print(f"parquet files: {len(files)}", file=sys.stderr)
    paths = download(repo, files, revision)
    for row_index, row in iter_rows(paths, limit):
        conversations = row.get("conversations") or []
        first = first_user_message(conversations)
        instruction, template_sha = (None, None)
        if first is not None:
            instruction, template_sha = extract_instruction(first)
        task = row.get("task") or ""
        yield {
            "row": row_index,
            "task": task,
            "family": family_of(task),
            "trial_name": row.get("trial_name"),
            "run_id": row.get("run_id"),
            "episode": row.get("episode"),
            "date": row.get("date"),
            "model": row.get("model"),
            "model_provider": row.get("model_provider"),
            "agent": row.get("agent"),
            "trace_source": row.get("trace_source"),
            "result": row.get("result"),
            "n_messages": len(conversations),
            "instruction": instruction,
            "instruction_sha256": _sha256(instruction) if instruction else None,
            "instruction_chars": len(instruction) if instruction else 0,
            "template_sha256": template_sha,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out",
        default=str(REPO / "work" / "tasks" / "user_instructions.jsonl"),
    )
    ap.add_argument(
        "--verify-against",
        default=str(REPO / "work" / "tasks" / "index.jsonl"),
        help="the earlier extraction; every instruction must match it byte for byte",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The earlier extraction, used as an independent oracle.
    oracle: dict[int, str] = {}
    verify_path = Path(args.verify_against)
    if verify_path.exists():
        for line in verify_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            oracle[rec["row"]] = rec.get("instruction_sha256")
        print(f"oracle rows: {len(oracle)}", file=sys.stderr)

    n = 0
    n_failed = 0
    n_mismatch = 0
    by_family: dict[str, int] = {}
    by_result: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records(args.repo, args.revision, args.limit):
            n += 1
            if not rec["instruction"]:
                n_failed += 1
                rec["extracted"] = False
            else:
                rec["extracted"] = True
            if oracle:
                want = oracle.get(rec["row"])
                got = rec["instruction_sha256"]
                if want is not None and want != got:
                    n_mismatch += 1
                    rec["oracle_match"] = False
                else:
                    rec["oracle_match"] = True
            by_family[rec["family"]] = by_family.get(rec["family"], 0) + 1
            key = str(rec.get("result"))
            by_result[key] = by_result.get(key, 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 5000 == 0:
                print(f"  {n} rows", file=sys.stderr)

    print(f"\nwrote {out_path}  ({n} rows, {out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  extracted ok   : {n - n_failed}")
    print(f"  failed         : {n_failed}")
    if oracle:
        print(f"  oracle mismatch: {n_mismatch}")
    print(f"  by family      : {by_family}")
    print(f"  by result      : {by_result}")
    return 0 if (n_failed == 0 and n_mismatch == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
