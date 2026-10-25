"""Turn the ClawGym trials into an SFT dataset -- with the reward as a filter.

Why this is not the OpenThoughts converter
------------------------------------------

``mini_sft/build_sft_dataset.py`` converts every finished trial into
a sample.  That is the right behaviour when the source data is a *trajectory
dataset*: the published rollouts are the training signal, and re-running them
through another harness produces another view of the same tasks.

ClawGym is the other kind of dataset: it ships **tasks**, so the rollouts
produced here are new, and the only thing that says whether a rollout is
worth training on is the task's own verifier.  A trajectory that scored zero
teaches the model to flail; a trajectory that scored one teaches it the task.
So the reward is the filter, and it is applied here.

Two more filters, and why
--------------------------

``reward`` is a *partial* score for most of the verifiers (the mean of the
checks), so ``--min-reward`` is a knob rather than a boolean: the default of
1.0 keeps only fully solved tasks, and a lower threshold keeps partially
correct ones, which are still valid demonstrations of the task's shape.

``--drop-reward-touched`` removes the rollouts that went into ``reward/``.
This is not a theoretical concern: the verifier is **visible** to the agent,
because 15 of the prompts say "Do not write anything under reward/" and 12
say "Use only relative paths under input/, output/, and reward/" -- those
instructions only make sense if the directory is there.  An agent that reads
``reward/test.py`` learns the expected values and can satisfy the checks
without doing the work, and a rollout like that is actively harmful training
data: it teaches the model to look for the grader.

Usage::

    python mini_sft/clawgym/build_sft_dataset.py \
        --run-root /private/tmp/rsi-harbor/clawgym \
        --registry work/registry_clawgym \
        --out mini_sft/dataset/clawgym-mini-glm-sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mini_sft"))

from build_sft_dataset import (  # noqa: E402
    summarize,
    to_conversations,
    to_messages,
)
from common.schema import (  # noqa: E402
    DATASET_CLAWGYM,
    HARNESS_MINI,
    canonical_model,
    provenance,
)

#: The path the verifier lives at inside the workspace.  A rollout that
#: touches it is either reading the grader or trying to rewrite it.
REWARD_DIR_RE = re.compile(r"reward/test\.py|reward/|/reward\b")


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


def touched_reward(messages: list[dict[str, Any]]) -> bool:
    """Did the agent go into the verifier's directory?

    Only the *assistant* turns are scanned.  The prompt itself mentions
    ``reward/`` in a few hundred tasks ("do not write under reward/"), and
    the instruction is the first user message -- counting that would flag
    every one of them.
    """
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                if REWARD_DIR_RE.search(arguments):
                    return True
            elif isinstance(arguments, dict):
                if REWARD_DIR_RE.search(json.dumps(arguments, ensure_ascii=False)):
                    return True
    return False


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="the run directory whose ledger is read.  Give it more than "
        "once when the half was annotated by more than one model: the GLM "
        "run and the MiniMax run have their own run roots.",
    )
    parser.add_argument(
        "--invalid-dir",
        default="",
        help="where the rollouts that tripped a discard rule are archived.  "
        "They are **not** deleted: a rollout that read the verifier is "
        "still a rollout, and throwing it away loses the evidence of how "
        "often that happens.  Default: an 'invalid' directory next to the "
        "dataset.",
    )
    parser.add_argument(
        "--registry", default=str(REPO / "work" / "registry_clawgym")
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "mini_sft" / "dataset" / "clawgym-mini-glm-sft.jsonl"),
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="keep only the rollouts that scored at least this much "
        "(most ClawGym verifiers report the *fraction* of checks passed)",
    )
    parser.add_argument(
        "--drop-reward-touched",
        action="store_true",
        default=True,
        help="drop the rollouts that read or wrote the verifier (default)",
    )
    parser.add_argument(
        "--keep-reward-touched",
        dest="drop_reward_touched",
        action="store_false",
        help="keep the rollouts that read or wrote the verifier",
    )
    parser.add_argument(
        "--model",
        default="",
        help="the model that produced the rollouts, when the ledger does "
        "not say.  The ledger records it from now on; the runs that were "
        "started before that are filled in from here.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_roots = [Path(r).resolve() for r in (args.run_root or [])]
    if not run_roots:
        run_roots = [Path("/private/tmp/rsi-harbor/clawgym")]
    ledger_paths = [r / "progress.jsonl" for r in run_roots]
    missing = [p for p in ledger_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"{p} does not exist; run the rollouts first", file=sys.stderr)
        return 2

    registry = Path(args.registry).resolve()
    manifest: dict[str, dict[str, Any]] = {}
    manifest_path = registry / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                manifest[record["task"]] = record

    # The last record per task wins: a task that failed and then succeeded
    # appears twice, and the successful trial is the one to keep.
    latest: dict[str, dict[str, Any]] = {}
    for ledger_path in ledger_paths:
        for record in load_ledger(ledger_path):
            latest[record["registry_name"]] = record

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The rollouts that trip a discard rule are archived, not deleted: the
    # dataset is the deliverable, but the evidence of what was filtered and
    # why is worth keeping too.
    invalid_dir = (
        Path(args.invalid_dir).resolve()
        if args.invalid_dir
        else out_path.parent / "invalid"
    )
    invalid_dir.mkdir(parents=True, exist_ok=True)
    invalid_path = invalid_dir / (out_path.stem + ".invalid.jsonl")
    invalid_tmp = invalid_path.with_name(invalid_path.name + ".tmp")
    # The dataset is written to a temporary file and renamed into place, so a
    # reader never sees a half-written record: the rollouts keep arriving
    # while this runs, and a torn line in the deliverable is not acceptable.
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    n_rows = 0
    n_no_trajectory = 0
    n_no_reward = 0
    n_low_reward = 0
    n_failed = 0
    n_reward_touched = 0
    n_no_instruction = 0
    rewards: list[float] = []
    by_skill: Counter[str] = Counter()
    #: Every task that did *not* make it into the dataset, with the reason.
    #: A deliverable that says "the rollouts that touched the verifier were
    #: dropped" without saying *which* ones is not a deliverable.
    skipped: list[dict[str, Any]] = []

    n_invalid = 0
    with tmp_path.open("w", encoding="utf-8") as handle, invalid_tmp.open(
        "w", encoding="utf-8"
    ) as invalid_handle:
        for name, record in sorted(latest.items()):
            # The trajectory is read first, whatever the verdict is going to
            # be: a rollout that is *not* going into the dataset is still
            # archived, and an archived rollout needs its messages.
            messages: list[dict[str, Any]] = []
            trajectory_error = ""
            trial_dir = Path(record.get("trial_dir") or "")
            trajectory_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
            if trial_dir and trajectory_path.exists():
                try:
                    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                    messages = to_messages(trajectory)
                    if not messages:
                        trajectory_error = "empty_trajectory"
                except (OSError, json.JSONDecodeError):
                    trajectory_error = "unreadable_trajectory"
            else:
                trajectory_error = "no_trajectory"

            reward_value = record.get("reward")
            reward = float(reward_value) if reward_value is not None else None

            # -- the verdict ------------------------------------------------ #
            reason = ""
            if not record.get("ok"):
                reason = "trial_failed"
            elif reward is None:
                reason = "no_reward"
            elif reward < args.min_reward:
                reason = "reward_below_threshold"
            elif trajectory_error:
                reason = trajectory_error
            elif args.drop_reward_touched and touched_reward(messages):
                reason = "touched_reward_dir"

            instruction_path = registry / "clawgym" / name / "instruction.md"
            instruction = None
            if instruction_path.exists():
                instruction = instruction_path.read_text(encoding="utf-8")
            else:
                n_no_instruction += 1

            entry = manifest.get(name, {})
            row = {
                "messages": messages,
                "conversations": to_conversations(messages) if messages else [],
                "source": "mini-swe-agent",
                **provenance(
                    dataset=DATASET_CLAWGYM,
                    harness=HARNESS_MINI,
                    model=canonical_model(record.get("model") or args.model),
                    model_id=record.get("model") or args.model,
                    model_provider=record.get("api_base"),
                ),
                "task": name,
                "family": record.get("family"),
                "clawgym_task_id": entry.get("task_id"),
                "clawgym_row": entry.get("row"),
                "primary_skill_id": entry.get("primary_skill_id"),
                "difficulty": entry.get("difficulty"),
                "macro_category": entry.get("macro_category"),
                "trial_name": record.get("trial_name"),
                "ok": bool(record.get("ok")),
                "reward": reward,
                "instruction": instruction,
                **summarize(messages),
            }

            if reason:
                # -- archived, not deleted ---------------------------------- #
                # The rollout is kept, with the reason it did not make it
                # into the dataset.  Deleting it would lose the evidence of
                # how often the discard rules fire, and would make the rule
                # impossible to re-check after the fact.
                skipped.append({"task": name, "reason": reason, "reward": reward})
                if reason == "trial_failed":
                    n_failed += 1
                elif reason == "no_reward":
                    n_no_reward += 1
                elif reason == "reward_below_threshold":
                    n_low_reward += 1
                elif reason == "touched_reward_dir":
                    n_reward_touched += 1
                else:
                    n_no_trajectory += 1
                if messages:
                    row["invalid_reason"] = reason
                    invalid_handle.write(dumps_row(row) + "\n")
                    n_invalid += 1
                continue

            handle.write(dumps_row(row) + "\n")
            n_rows += 1
            if reward is not None:
                rewards.append(reward)
            by_skill[str(entry.get("primary_skill_id"))] += 1
            if args.limit and n_rows >= args.limit:
                break

    # The rename is what makes the dataset readable at all times: a reader
    # sees either the old file or the new one, never a half-written record.
    os.replace(tmp_path, out_path)
    os.replace(invalid_tmp, invalid_path)

    summary = {
        "rows": n_rows,
        "out": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 1) if out_path.exists() else 0,
        "skipped_failed": n_failed,
        "skipped_no_reward": n_no_reward,
        "skipped_low_reward": n_low_reward,
        "skipped_reward_touched": n_reward_touched,
        "skipped_no_trajectory": n_no_trajectory,
        "archived_invalid": n_invalid,
        "invalid_file": str(invalid_path),
        "min_reward": args.min_reward,
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "distinct_skills": len(by_skill),
        "skipped": {
            "trial_failed": n_failed,
            "no_reward": n_no_reward,
            "reward_below_threshold": n_low_reward,
            "touched_reward_dir": n_reward_touched,
            "no_trajectory": n_no_trajectory,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # The summary is written next to the dataset: a dataset without the
    # counts that describe it is not a deliverable.
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    skipped_path = out_path.with_suffix(".skipped.jsonl")
    with skipped_path.open("w", encoding="utf-8") as handle:
        for item in skipped:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {skipped_path}")
    print(f"wrote {invalid_path}  ({n_invalid} archived rollouts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
