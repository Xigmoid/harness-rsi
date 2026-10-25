"""Split a registry's tasks into two halves, one per harness.

The rule
--------

``mini`` and ``opencode`` each get half of every dataset, and the split has
to be two things at once:

* **deterministic** -- the same task name always lands on the same side, so
  a run that is interrupted and resumed does not move a task between
  harnesses, and a dataset built from two runs is still the same dataset;
* **independent of the row order** -- the ClawGym manifest is ordered by
  dataset row, and its first 6,094 rows are all one task family ("arc")
  while the remaining 9,308 are the other ("persona").  Splitting by row
  order would hand one harness 6,094 arc tasks and the other 9,308 persona
  tasks, and the two halves would not be comparable at all.

Both come from one thing: the task name is hashed, and the hash decides the
side.  ``sha256`` over the name, first four bytes, modulo two.  That is
order-independent, reproducible, and needs no state.

Usage::

    python opencode_sft/split_tasks.py \
        --registry opencode_sft/registry_clawgym \
        --out-manifest mini opencode

writes ``<registry>/manifest-mini.jsonl`` and ``<registry>/manifest-opencode.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencode_sft.minimax import share_of  # noqa: E402

#: The two sides.  The names are the harness names, and they are also the
#: suffixes of the manifest files.
SIDES = ("mini", "opencode")


def side_of(name: str) -> str:
    """Which half a task belongs to."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return SIDES[int.from_bytes(digest[:4], "big") % len(SIDES)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="opencode_sft/registry_clawgym",
        help="the registry root; manifest.jsonl is read from it",
    )
    parser.add_argument(
        "--out-manifest",
        nargs="*",
        default=list(SIDES),
        help="the sides to write, as suffixes: manifest-<side>.jsonl",
    )
    parser.add_argument(
        "--report",
        default="",
        help="write a JSON report of the split to this path",
    )
    parser.add_argument(
        "--families",
        default="",
        help="only split these families, comma separated.  The families that "
        "are left out are simply not written to either manifest -- the "
        "OpenThoughts registry carries swesmith, which is not part of this "
        "re-annotation.",
    )
    parser.add_argument(
        "--minimax-share",
        type=float,
        default=0.0,
        help="the share of each half that is annotated with MiniMax instead "
        "of GLM, as a fraction between 0 and 1.  The share is decided per "
        "task by the same hash the harness split uses, so it is "
        "deterministic, independent of the row order, and never overlaps "
        "with anything.  It is written to manifest-<side>-minimax.jsonl.",
    )
    args = parser.parse_args()

    registry = Path(args.registry).resolve()
    manifest_path = registry / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} does not exist")

    entries: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))

    wanted = {f.strip() for f in args.families.split(",") if f.strip()}
    if wanted:
        entries = [entry for entry in entries if entry.get("family") in wanted]

    buckets: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for entry in entries:
        buckets[side_of(entry["task"])].append(entry)

    for side in args.out_manifest:
        path = registry / f"manifest-{side}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in buckets[side]:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(buckets[side])} tasks)")

    # The MiniMax share: the same tasks, decided by the same hash, but
    # carved out of the half so that the GLM run and the MiniMax run never
    # overlap.  The GLM run skips these (--exclude-manifest) and the
    # MiniMax run runs exactly these.
    if args.minimax_share > 0:
        for side in args.out_manifest:
            selected = [
                entry for entry in buckets[side] if share_of(entry["task"], args.minimax_share)
            ]
            path = registry / f"manifest-{side}-minimax.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for entry in selected:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(
                f"wrote {path}  ({len(selected)} tasks, "
                f"{len(selected) / max(len(buckets[side]), 1):.1%} of the half)"
            )

    # The report says what the split *is*, not just how big it is: a split
    # that is balanced overall but puts every arc task on one side is not a
    # split, it is a coincidence.
    report: dict[str, Any] = {"registry": str(registry), "total": len(entries), "sides": {}}
    for side in SIDES:
        bucket = buckets[side]
        by_family = Counter(entry.get("family") for entry in bucket)
        by_skill = Counter(entry.get("primary_skill_id") for entry in bucket)
        report["sides"][side] = {
            "tasks": len(bucket),
            "by_family": dict(by_family),
            "distinct_skills": len([k for k in by_skill if k]),
        }

    # The two sides have to cover the same ground: for every skill that
    # appears in one half, how many of its tasks are in the other?
    skill_counts = {
        side: Counter(entry.get("primary_skill_id") for entry in buckets[side])
        for side in SIDES
    }
    report["skill_overlap"] = {
        "skills_in_both": len(
            {k for k in skill_counts[SIDES[0]] if k} & {k for k in skill_counts[SIDES[1]] if k}
        ),
        "skills_in_one_side_only": len(
            {k for k in skill_counts[SIDES[0]] if k} ^ {k for k in skill_counts[SIDES[1]] if k}
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
