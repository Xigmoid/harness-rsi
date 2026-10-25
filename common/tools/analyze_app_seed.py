"""How much content did the published rollouts find in /app?

The three plain families (IssueTasks, Tezos, SuperUser) start from an empty
/app on ubuntu 24.04 -- that is what the upstream generator builds.  But the
published rollouts sometimes show files in /app, and the question is whether
those files were **in the container already** or whether the agent created
them before the first ``ls -la``.

The distinction matters: files the agent created must not be reproduced (the
agent creates them again), while files that were in the container have to be
reproduced or the re-annotated rollout starts from a different environment.

The evidence is the *first* ``ls -la`` in the rollout.  If the agent created
the files, the rollout shows the creating command in the observation right
before the listing.

Usage::

    python common/tools/analyze_app_seed.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

from hf_parquet import published_parquet as _published_parquet

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from check_env_fidelity import (  # noqa: E402
    _first_ls,
    _observations,
)

def published_parquet() -> Path:
    """The published parquet, found in the HuggingFace cache."""
    return _published_parquet()


def main() -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(published_parquet(), columns=["task", "conversations"])
    tasks = table.column("task").to_pylist()
    conversations = table.column("conversations").to_pylist()

    per_family: dict[str, collections.Counter] = {
        "issue": collections.Counter(),
        "tezos": collections.Counter(),
        "superuser": collections.Counter(),
    }
    file_names: dict[str, collections.Counter] = {
        "issue": collections.Counter(),
        "tezos": collections.Counter(),
        "superuser": collections.Counter(),
    }
    examples: list[dict] = []

    for task, conversation in zip(tasks, conversations):
        family = task.split("-", 1)[0]
        if family not in per_family or not conversation:
            continue
        names = _first_ls(conversation)
        if names is None:
            per_family[family]["no ls at all"] += 1
            continue
        if not names:
            per_family[family]["ls was empty"] += 1
            continue
        per_family[family][f"ls had {len(names)} entries"] += 1
        for name in names:
            file_names[family][name] += 1
        if len(examples) < 8:
            examples.append({"task": task, "files": names})

    report = {
        "counts": {f: dict(c) for f, c in per_family.items()},
        "top_files": {
            f: c.most_common(25) for f, c in file_names.items()
        },
        "examples": examples,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
