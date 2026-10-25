"""Turn the ClawGym **opencode** trials into an SFT dataset.

Why this file exists next to ``opencode_sft/build_sft_dataset.py``
------------------------------------------------------------------

The parent module converts an opencode rollout into a row; that part is
shared, and this module imports it rather than re-implementing it.  What is
ClawGym-specific is everything around it:

* **the reward is the filter.**  ClawGym ships *tasks*, not trajectories, so
  the rollouts produced here are new and the only thing that says whether one
  is worth training on is the task's own verifier.  A rollout that scored
  zero teaches the model to flail.
* **the rollouts that read the grader are dropped.**  ``reward/`` is
  **visible** to the agent -- 15 of the prompts say "Do not write anything
  under reward/" and 12 say "Use only relative paths under input/, output/,
  and reward/", and those instructions only make sense if the directory is
  there.  An agent that reads ``reward/test.py`` learns the expected values
  and can satisfy the checks without doing the work, and such a rollout is
  actively harmful training data: it teaches the model to look for the
  grader.  The environment is left alone -- changing it would change the
  task -- so the filter lives here.
* **the schema is the mini half's schema.**  The two halves are merged into
  one dataset at the end, so both carry the same fields and the same
  ``harness`` marker.

The parent module is imported rather than copied for one more reason: the
opencode session export is a *part tree*, and the reconstruction of it (the
tool calls, their arguments, the observations) is the part that has to match
the wire byte for byte.  There is exactly one implementation of that, and it
is the audited one.

Usage::

    python opencode_sft/clawgym/build_opencode_sft_dataset.py \
        --run-root /private/tmp/rsi-harbor/clawgym-opencode \
        --registry opencode_sft/registry_clawgym \
        --requests /private/tmp/rsi-harbor/logs/oc-requests.jsonl \
        --blob-dir /private/tmp/rsi-harbor/logs/oc-blobs \
        --out opencode_sft/clawgym/dataset/clawgym-opencode-sft.jsonl
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
sys.path.insert(0, str(REPO / "opencode_sft"))

from build_sft_dataset import (  # noqa: E402
    RequestIndex,
    convert_trial,
)
from opencode_sft.schema import (  # noqa: E402
    DATASET_CLAWGYM,
    HARNESS_OPENCODE,
    canonical_model,
    provenance,
)

#: The path the verifier lives at inside the workspace.  A rollout that
#: touches it is either reading the grader or trying to rewrite it.
REWARD_DIR_RE = re.compile(r"reward/test\.py|reward/|/reward\b")

#: The characters that break a line-oriented reader.
#:
#: A JSON record is one line, and the line ends at ``\n``.  But Python's
#: ``str.splitlines()`` -- which half the tools that read JSONL use -- also
#: breaks on ``\v``, ``\f``, ``\x1c``, ``\x1d``, ``\x1e``, ``\x85``,
#: ``\u2028`` and ``\u2029``.  The model's output contains such characters
#: (a Hungarian name in a CSV, a line separator in a report), and a record
#: containing one of them is then read as two half-records: the first one
#: fails to parse and the second one is a mystery.
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


def load_ledger(path: Path) -> list[dict[str, Any]]:
    """The ledger records, skipping the ones that are not valid JSON."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            # Split on "\n" only: ``str.splitlines()`` also breaks on U+2028
            # and nine other characters, and the model's own output contains
            # some of them.
            line = line.strip()
            if not line:
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


def fold_system_prompt(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the system prompt into the first user message.

    The parent module writes the system prompt as its own leading ``user``
    message, because that is what the wire had.  The published dataset has
    no ``system`` role at all and its generator (terminus-2) put the system
    prompt *inside* the first user message -- and so does the mini half of
    this re-annotation.  The two halves are merged into one dataset, so they
    have to agree on the convention.
    """
    if (
        len(conversations) >= 2
        and conversations[0].get("role") == "user"
        and conversations[1].get("role") == "user"
    ):
        folded = list(conversations[1:])
        folded[0] = {
            "role": "user",
            "content": f'{conversations[0].get("content") or ""}\n\n'
            f'{conversations[1].get("content") or ""}',
        }
        return folded
    return conversations


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
        "--registry", default=str(REPO / "opencode_sft" / "registry_clawgym")
    )
    parser.add_argument(
        "--requests",
        default="/private/tmp/rsi-harbor/logs/oc-requests.jsonl",
        help="the recording proxy's request log",
    )
    parser.add_argument(
        "--blob-dir",
        default="/private/tmp/rsi-harbor/logs/oc-blobs",
        help="the proxy's content-addressed system-prompt / tool-schema blobs",
    )
    parser.add_argument(
        "--out",
        default=str(HERE / "dataset" / "clawgym-opencode-sft.jsonl"),
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
        run_roots = [Path("/private/tmp/rsi-harbor/clawgym-opencode")]
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

    requests = RequestIndex(
        Path(args.requests), Path(args.blob_dir) if args.blob_dir else None
    )

    # The last record per task wins: a task that failed and then succeeded
    # appears twice, and the successful trial is the one to keep.
    latest: dict[str, dict[str, Any]] = {}
    for ledger_path in ledger_paths:
        for record in load_ledger(ledger_path):
            latest[record["registry_name"]] = record

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The rollouts that trip a discard rule are archived, not deleted.
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
    n_no_capture = 0
    rewards: list[float] = []
    by_skill: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    skipped_trials: list[str] = []

    n_invalid = 0
    with tmp_path.open("w", encoding="utf-8") as handle, invalid_tmp.open(
        "w", encoding="utf-8"
    ) as invalid_handle:
        for name, record in sorted(latest.items()):
            reward_value = record.get("reward")
            reward = float(reward_value) if reward_value is not None else None

            trial_dir = Path(record.get("trial_dir") or "")
            row = None
            if trial_dir.exists():
                row = convert_trial(trial_dir, requests, {}, manifest)

            # -- the verdict ------------------------------------------------ #
            reason = ""
            if not record.get("ok"):
                reason = "trial_failed"
            elif reward is None:
                reason = "no_reward"
            elif reward < args.min_reward:
                reason = "reward_below_threshold"
            elif row is None:
                reason = "no_usable_rollout"
            elif not row.get("has_system_prompt"):
                # Without the system prompt the row is not the context the
                # model saw, so it is not a training sample.
                reason = "no_captured_request"
            elif args.drop_reward_touched and touched_reward(row["messages"]):
                reason = "touched_reward_dir"

            if reason in ("trial_failed",):
                n_failed += 1
            elif reason == "no_reward":
                n_no_reward += 1
            elif reason == "reward_below_threshold":
                n_low_reward += 1
            elif reason == "no_captured_request":
                n_no_capture += 1
            elif reason == "touched_reward_dir":
                n_reward_touched += 1
            elif reason:
                n_no_trajectory += 1

            if reason:
                skipped_trials.append({"task": name, "reason": reason, "reward": reward})
                # -- archived, not deleted ---------------------------------- #
                if row is not None and row.get("messages"):
                    archived = {
                        "messages": row["messages"],
                        "conversations": fold_system_prompt(row["conversations"]),
                        "source": "opencode",
                        **provenance(
                            dataset=DATASET_CLAWGYM,
                            harness=HARNESS_OPENCODE,
                            model=canonical_model(
                                record.get("model") or args.model or row.get("model"),
                                row.get("model_provider"),
                            ),
                            model_id=row.get("model_id"),
                            model_provider=row.get("model_provider"),
                        ),
                        "task": name,
                        "family": record.get("family"),
                        "trial_name": record.get("trial_name"),
                        "ok": bool(record.get("ok")),
                        "reward": reward,
                        "instruction_wire": row.get("instruction_wire"),
                        "model": row.get("model"),
                        "invalid_reason": reason,
                    }
                    invalid_handle.write(dumps_row(archived) + "\n")
                    n_invalid += 1
                continue

            conversations = fold_system_prompt(row["conversations"])
            entry = manifest.get(name, {})
            instruction_path = registry / "clawgym" / name / "instruction.md"
            instruction = None
            if instruction_path.exists():
                instruction = instruction_path.read_text(encoding="utf-8")
            else:
                n_no_instruction += 1

            out_row = {
                "messages": row["messages"],
                "conversations": conversations,
                "source": "opencode",
                **provenance(
                    dataset=DATASET_CLAWGYM,
                    harness=HARNESS_OPENCODE,
                    model=canonical_model(
                        record.get("model") or args.model or row.get("model"),
                        row.get("model_provider"),
                    ),
                    model_id=row.get("model_id") or record.get("model") or args.model,
                    model_provider=row.get("model_provider"),
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
                "instruction_wire": row.get("instruction_wire"),
                "conversion_source": row.get("conversion_source"),
                "n_assistant_turns": row.get("n_assistant_turns"),
                "n_tool_calls": row.get("n_tool_calls"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "opencode_version": row.get("opencode_version"),
                "model": row.get("model"),
                # The endpoint that produced the rollout.  The run can be
                # served by more than one, and the dataset has to say which
                # one wrote each row.
                "annotation_model": record.get("model") or row.get("model"),
            }
            handle.write(dumps_row(out_row) + "\n")
            n_rows += 1
            rewards.append(reward)
            by_skill[str(entry.get("primary_skill_id"))] += 1
            by_source[str(row.get("conversion_source"))] += 1
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
        "skipped_no_capture": n_no_capture,
        "archived_invalid": n_invalid,
        "invalid_file": str(invalid_path),
        "min_reward": args.min_reward,
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "distinct_skills": len(by_skill),
        "conversion_source": dict(by_source),
        "recorded_requests": requests.n_requests,
        "recorded_agent_requests": requests.n_agent_requests,
        "distinct_captured_sessions": len(requests.by_user_message),
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
        for item in skipped_trials:
            handle.write(json.dumps({"reason": item}, ensure_ascii=False) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {skipped_path}")
    print(f"wrote {invalid_path}  ({n_invalid} archived rollouts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
