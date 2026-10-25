#!/usr/bin/env python3
"""Drop a family's records from a rollout ledger, so it runs again.

The ledger (``progress.jsonl``) is the source of truth for what the driver
has already done: a task whose name is in it with ``ok: true`` is skipped.
That is exactly what you want when a run is interrupted -- and exactly what
you do **not** want when the reason the family has to run again is that the
*environment* was wrong.

The concrete case: the IssueTasks family was first run against the plain
ubuntu image, which has the system python 3.12, while the published
rollouts ran on Debian with a source-built python 3.10 in ``/usr/local``.
The trajectories from the ubuntu runs are not what the published dataset
looks like, so they have to go, and the family has to run again against
``rsi/oc-base-issue-py310:1``.

Usage::

    # see what would be dropped
    python common/tools/purge_ledger.py --run-root /private/tmp/rsi-harbor/mini-sft-full \\
        --family issue --dry-run

    # drop it
    python common/tools/purge_ledger.py --run-root /private/tmp/rsi-harbor/mini-sft-full \\
        --family issue
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def count_turns(record: dict[str, Any]) -> int | None:
    """How many assistant turns the recorded rollout has.

    Whether a rollout is in distribution is decided by the trajectory, so it
    is read from the trial directory -- the ledger does not know the length.
    """
    trial_dir = record.get("trial_dir")
    if not trial_dir:
        return None
    path = Path(trial_dir) / "agent" / "mini-swe-agent.trajectory.json"
    if not path.exists():
        return None
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return sum(
        1
        for message in trajectory.get("messages") or []
        if message.get("role") == "assistant"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        default="/private/tmp/rsi-harbor/mini-sft-full",
        help="the Harbor output directory (the one with progress.jsonl)",
    )
    ap.add_argument(
        "--family",
        default="",
        help="the family to drop; empty drops every record",
    )
    ap.add_argument(
        "--task",
        default="",
        help="a single registry task to drop; empty drops the whole family",
    )
    ap.add_argument(
        "--drop-failed",
        action="store_true",
        help="drop every record whose trial failed",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="drop every successful record whose rollout took more than this "
        "many assistant turns (0 = keep all)",
    )
    ap.add_argument(
        "--move-trials",
        action="store_true",
        help="also move the dropped trials' directories out of the run root, "
        "into <run-root>/_purged/.  Needed whenever the driver runs with "
        "--resync-ledger: the ledger is rebuilt from the trial directories "
        "on disk, so a dropped record whose directory is still there comes "
        "straight back.",
    )
    ap.add_argument(
        "--keep-only",
        default="",
        help="a manifest file; every record whose task is not in it is "
        "dropped.  The two harnesses each get half of every dataset, and a "
        "run that started before the split was decided has records on the "
        "wrong side -- this is how they are removed.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ledger = Path(args.run_root) / "progress.jsonl"
    if not ledger.exists():
        print(f"{ledger} does not exist", file=__import__("sys").stderr)
        return 2

    records = load(ledger)

    keep_names: set[str] | None = None
    if args.keep_only:
        keep_names = set()
        for line in Path(args.keep_only).read_text(encoding="utf-8").splitlines():
            if line.strip():
                keep_names.add(json.loads(line)["task"])
        print(f"keeping only the {len(keep_names)} tasks of {args.keep_only}")

    reasons: dict[str, int] = {}

    def dropped(record: dict[str, Any]) -> bool:
        if args.task:
            return record.get("registry_name") == args.task
        if args.family:
            return record.get("family") == args.family
        return True

    def unwanted(record: dict[str, Any]) -> str | None:
        """Why this record has to go, or None to keep it."""
        if keep_names is not None and record.get("registry_name") not in keep_names:
            return "not in the kept manifest"
        if args.drop_failed and not record.get("ok"):
            return "failed"
        if args.max_turns:
            turns = count_turns(record)
            if turns is not None and turns > args.max_turns:
                return f"{turns} turns"
        return None

    kept: list[dict[str, Any]] = []
    dropped_names: set[str] = set()
    dropped_dirs: set[str] = set()
    for record in records:
        reason = unwanted(record)
        if reason is None:
            kept.append(record)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            dropped_names.add(record.get("registry_name", "?"))
            if record.get("trial_dir"):
                dropped_dirs.add(record["trial_dir"])

    gone = len(records) - len(kept)
    names = sorted(dropped_names)
    print(f"records: {len(records)}  kept: {len(kept)}  dropped: {gone}")
    if reasons:
        print("  by reason:", dict(sorted(reasons.items())))
    if names:
        print(f"  tasks dropped: {len(names)}")
        for name in names[:10]:
            print(f"    {name}")
        if len(names) > 10:
            print(f"    ... and {len(names) - 10} more")
    if args.dry_run:
        return 0
    if not gone:
        return 0

    backup = ledger.with_suffix(".jsonl.bak")
    shutil.copy2(ledger, backup)
    with ledger.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"rewrote {ledger}  (backup: {backup})")

    if args.move_trials and dropped_dirs:
        # The ledger is not the only source of truth: ``--resync-ledger``
        # rebuilds it from the trial directories on disk.  A dropped record
        # whose directory is still there comes straight back, so the
        # directories have to move with it.
        graveyard = Path(args.run_root) / "_purged"
        graveyard.mkdir(parents=True, exist_ok=True)
        moved = 0
        for trial_dir in sorted(dropped_dirs):
            source = Path(trial_dir)
            if not source.exists():
                continue
            destination = graveyard / f"{source.parent.name}__{source.name}"
            shutil.move(str(source), str(destination))
            moved += 1
        print(f"moved {moved} trial directories into {graveyard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
