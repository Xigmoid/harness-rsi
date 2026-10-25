"""Merge the two ClawGym halves into one dataset.

What is being merged
--------------------

ClawGym is re-annotated twice: once with mini-swe-agent and once with
opencode, each on its own half of the tasks.  The halves are disjoint by
construction -- ``common/split_tasks.py`` assigns every task to one
harness by the sha256 of its name -- so the merged dataset holds one row per
task, and the ``harness`` field says which harness produced it.

The merge is deliberately *dumb*: it concatenates the rows and checks the
invariants that must hold across the seam.  It does not re-derive anything,
because the two halves were already validated on their own, and a merge step
that recomputes is a merge step that can silently disagree with them.

What is checked
---------------

* every row carries a ``harness``;
* no task appears twice -- the split must have kept the halves disjoint, and
  a duplicate means the split was applied to the wrong ledger;
* every row still carries the fields the trainer needs (``messages``,
  ``conversations``, ``reward``, ``instruction``);
* the dataset is written atomically, so a reader never sees half a file.

Usage::

    python common/merge_sft.py \
        --dataset mini_sft/dataset/clawgym-mini-glm-sft.jsonl \
        --dataset opencode_sft/dataset/clawgym-opencode-glm-sft.jsonl \
        --out work/dataset/clawgym-sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

#: The fields a row has to carry for the merged dataset to be usable.
#:
#: The three provenance keys are not optional: a trajectory is only
#: interpretable together with the dataset it answers, the model that wrote
#: it and the harness that ran.
REQUIRED_FIELDS = (
    "messages",
    "conversations",
    "reward",
    "instruction",
    "dataset",
    "model",
    "harness",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSONL file, skipping the lines that do not parse."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # Split on "\n" only: ``str.splitlines()`` also breaks on U+2028
            # and eight other characters.
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="a dataset to merge; give it once per half",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "work" / "dataset"
                   / "clawgym-sft.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    seen: dict[str, str] = {}
    by_harness: Counter[str] = Counter()
    problems: list[str] = []
    n_rows = 0
    by_dataset: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_harness: Counter[str] = Counter()

    with tmp_path.open("w", encoding="utf-8") as handle:
        for dataset_path in args.dataset:
            path = Path(dataset_path)
            rows = read_jsonl(path)
            for row in rows:
                harness = str(row.get("harness") or "")
                if not harness:
                    problems.append(f"{path.name}: a row has no harness")
                    continue
                missing = [
                    field for field in REQUIRED_FIELDS if field not in row
                ]
                if missing:
                    problems.append(
                        f"{path.name}: {row.get('task')!r} is missing {missing}"
                    )
                    continue
                task = str(row.get("task"))
                if task in seen:
                    problems.append(
                        f"{task!r} is in both {seen[task]} and {path.name}"
                    )
                    continue
                seen[task] = path.name
                by_dataset[str(row.get("dataset"))] += 1
                by_model[str(row.get("model"))] += 1
                by_harness[str(row.get("harness"))] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                if args.limit and n_rows >= args.limit:
                    break
            if args.limit and n_rows >= args.limit:
                break

    os.replace(tmp_path, out_path)

    summary = {
        "rows": n_rows,
        "out": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 1) if out_path.exists() else 0,
        "by_harness": dict(by_harness),
        "by_dataset": dict(by_dataset),
        "by_model": dict(by_model),
        "distinct_tasks": len(seen),
        "problems": problems[:50],
        "n_problems": len(problems),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
