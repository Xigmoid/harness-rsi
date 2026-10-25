"""Map every SWE-Smith task of the dataset to its SWE-Smith instance.

The dataset's SWE-Smith task ids are renumbered indices (``swesmith-00000_copy0001``)
that say nothing about the environment, so the mapping has to be recovered
from the **instruction**: the SWE-Smith instance's ``problem_statement`` is
the instruction the published rollout was given.

The mapping is exact, not approximate.  The published instructions are the
SWE-Smith problem statements with the markdown fences sometimes normalised,
so the match is done on a normalised form of the text.  All 7,693 unique
SWE-Smith tasks match.

The result is what the per-task setup script needs:

* ``clone_url`` -- the SWE-Smith mirror of the repository.  The mirror has
  **one branch per instance**, and the branch is named exactly like the
  instance id, so the instance's state is reachable by name:

      https://github.com/swesmith/<owner>__<repo>.<base_commit>.git
      git checkout <instance_id>

  This is the same recipe the upstream generator uses -- ``data/swesmith/
  task_templates.py`` renders the task Dockerfile as
  ``RUN git fetch && git checkout {instance_id}``.

* ``base_commit`` -- the commit the instance's bug was injected on top of.

One task in 7,693 has an ambiguous instruction: three different SWE-Smith
instances share the same problem statement.  The tie is broken with the
published rollout itself -- the instance whose patch touches a file the
original agent actually looked at is the one that was rolled out.

Usage::

    python opencode_sft/map_swesmith.py
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: The SWE-Smith instance cache.  ``instance_id`` / ``repo`` / ``patch`` /
#: ``problem_statement`` are read from it; the parquet is not downloaded
#: again.
DEFAULT_PARQUET_GLOB = "/tmp/swesmith_parquet/default/train/*.parquet"

#: The published parquet, used only to break an ambiguous instruction.
DEFAULT_PUBLISHED_PARQUET = (
    "/Users/djx/.cache/huggingface/hub/"
    "datasets--open-thoughts--OpenThoughts-Agent-SFT-31.6K/snapshots/"
    "430c23f84bc0616ca8795e02bd9447f9f49c413d/train.parquet"
)

#: The markdown fences are the only difference between the published
#: instruction and the SWE-Smith problem statement; the text inside them is
#: the same.
_FENCE_RE = re.compile(r"```.*?```", re.S)


def normalize(text: str) -> str:
    """The form of the text that is compared.

    NFKC so that the Unicode the instruction happens to use does not matter,
    the markdown fences removed because the published instruction carries
    them and the problem statement does not, and the whitespace collapsed so
    that a re-wrapped paragraph is still the same paragraph.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _FENCE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def changed_files(patch: str) -> set[str]:
    """The files a unified diff touches."""
    files = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
        elif line.startswith("--- a/"):
            files.add(line[6:].strip())
    return files


def load_instances(glob_pattern: str) -> dict[str, list[dict[str, Any]]]:
    """The SWE-Smith instances, keyed by normalised problem statement."""
    import pyarrow.parquet as pq

    by_statement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path("/").glob(glob_pattern.lstrip("/"))):
        table = pq.read_table(
            path, columns=["instance_id", "repo", "problem_statement", "patch"]
        )
        rows = table.to_pydict()
        for instance_id, repo, statement, patch in zip(
            rows["instance_id"], rows["repo"], rows["problem_statement"], rows["patch"]
        ):
            slug = repo.split("/", 1)[1] if "/" in repo else repo
            by_statement[normalize(statement)].append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "slug": slug,
                    "base_commit": slug.rsplit(".", 1)[-1],
                    "clone_url": f"https://github.com/{repo}.git",
                    "patch": patch,
                }
            )
    return by_statement


def resolve_ambiguity(
    task: str, candidates: list[dict[str, Any]], published: Any
) -> tuple[dict[str, Any], list[str]]:
    """Pick the instance the published rollout actually used.

    The tie is broken with the rollout's own text: the instance whose patch
    touches a file the original agent actually opened is the one that was
    rolled out.  When even that does not separate the candidates, the first
    candidate in sorted order wins -- deterministically, and it is reported.
    """
    notes: list[str] = []
    if published:
        text = "\n".join(
            str(message.get("content", "")) for message in published
        )
        scores = []
        for candidate in candidates:
            files = changed_files(candidate["patch"])
            hits = sum(1 for name in files if name and name in text)
            scores.append((hits, candidate["instance_id"]))
        scores.sort(key=lambda item: (-item[0], item[1]))
        notes.append(
            "rollout-overlap scores: "
            + ", ".join(f"{iid}={hits}" for hits, iid in scores)
        )
        if scores[0][0] > 0 and scores[0][0] > scores[1][0]:
            chosen = next(
                c for c in candidates if c["instance_id"] == scores[0][1]
            )
            notes.append(f"chose {chosen['instance_id']} on rollout overlap")
            return chosen, notes
    chosen = sorted(candidates, key=lambda c: c["instance_id"])[0]
    notes.append(
        f"ambiguous and not separable from the rollout; chose {chosen['instance_id']}"
    )
    return chosen, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", default=str(HERE / "tasks" / "instructions"))
    parser.add_argument("--index", default=str(HERE / "tasks" / "index.jsonl"))
    parser.add_argument("--parquet-glob", default=DEFAULT_PARQUET_GLOB)
    parser.add_argument("--published", default=DEFAULT_PUBLISHED_PARQUET)
    parser.add_argument("--out", default=str(HERE / "tasks" / "swesmith_map.json"))
    args = parser.parse_args()

    instructions = Path(args.instructions)
    rows = [
        json.loads(line)
        for line in Path(args.index).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = sorted({row["task"] for row in rows if row["family"] == "swesmith"})

    by_statement = load_instances(args.parquet_glob)
    print(f"SWE-Smith instances: {sum(len(v) for v in by_statement.values())}")

    # The published conversations, for the ambiguity tie-break only.
    published_by_task: dict[str, Any] = {}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(args.published, columns=["task", "conversations"])
        wanted = set(tasks)
        for task, conversation in zip(
            table.column("task").to_pylist(),
            table.column("conversations").to_pylist(),
        ):
            if task in wanted and task not in published_by_task:
                published_by_task[task] = conversation
    except Exception as exc:  # pragma: no cover -- the tie-break is optional
        print(f"(no published parquet for the tie-break: {exc})")

    mapping: dict[str, Any] = {}
    ambiguous = 0
    unmatched: list[str] = []
    for task in tasks:
        text = (instructions / f"{task}.md").read_text(encoding="utf-8")
        candidates = by_statement.get(normalize(text))
        if not candidates:
            unmatched.append(task)
            continue
        if len(candidates) == 1:
            chosen = candidates[0]
            notes = []
        else:
            ambiguous += 1
            chosen, notes = resolve_ambiguity(
                task, candidates, published_by_task.get(task)
            )
        mapping[task] = {
            "instance_id": chosen["instance_id"],
            "repo": chosen["repo"],
            "slug": chosen["slug"],
            "base_commit": chosen["base_commit"],
            "clone_url": chosen["clone_url"],
            "candidates": sorted(c["instance_id"] for c in candidates),
            "notes": notes,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tasks": mapping,
                "unmatched": unmatched,
                "stats": {
                    "tasks": len(tasks),
                    "mapped": len(mapping),
                    "unmatched": len(unmatched),
                    "ambiguous": ambiguous,
                    "distinct_repos": len({m["repo"] for m in mapping.values()}),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mapped": f"{len(mapping)}/{len(tasks)}",
                "unmatched": len(unmatched),
                "ambiguous": ambiguous,
                "distinct_repos": len({m["repo"] for m in mapping.values()}),
                "out": str(out),
            },
            indent=2,
        )
    )
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
