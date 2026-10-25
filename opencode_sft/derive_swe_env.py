"""Derive the per-task SWE-Smith environment facts from the published rollouts.

The SWE-Smith family is not one environment but several, and the published
rollouts say which is which.  Two facts decide which image a task runs on and
what its setup script has to do:

**The language of the repository.**  The official SWE-Smith images are built
per repository, and the base image follows the language:

* a Python repository runs in an **ubuntu 22.04 (jammy)** image with an
  environment named ``testbed`` active -- the published rollout shows
  ``(testbed) root@...:/testbed#`` and ``Hit:1 http://archive.ubuntu.com/ubuntu
  jammy InRelease``;
* a Go repository runs in a **debian 12 (bookworm)** image -- the published
  rollout shows ``Hit:1 http://deb.debian.org/debian bookworm InRelease`` --
  with Go installed at ``/usr/local/go`` but *not* on ``PATH``, so the
  rollout starts with ``go: command not found``.

**The python version.**  The conda environment of the Python images is not
always the same build: the published rollouts answered ``python --version``
with 3.10.16 (3,868 rows), 3.10.18 (1,217), 3.11.11 (139) or 3.11.13 (76).

Both facts are read out of the published rollouts -- the language from the
repository's own tree, the python version from the first ``Python x.y.z``
the rollout printed, which is the image's version because nothing has been
installed yet at that point.

Usage::

    python opencode_sft/derive_swe_env.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_MIRRORS = Path("/private/tmp/rsi-swe-mirrors")

DEFAULT_PUBLISHED_PARQUET = (
    "/Users/djx/.cache/huggingface/hub/"
    "datasets--open-thoughts--OpenThoughts-Agent-SFT-31.6K/snapshots/"
    "430c23f84bc0616ca8795e02bd9447f9f49c413d/train.parquet"
)

#: The python versions the family image carries.
KNOWN_PYTHONS = ("3.10.16", "3.10.18", "3.11.11", "3.11.13")

PYTHON_VERSION_RE = re.compile(r"Python (3\.\d+\.\d+)")


def classify_repository(mirror: Path) -> str:
    """The build system of a repository, read from its own tree."""
    result = subprocess.run(
        ["git", "--git-dir", str(mirror), "ls-tree", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    names = set(result.stdout.split())
    if "setup.py" in names or "pyproject.toml" in names:
        return "python"
    if "go.mod" in names:
        return "go"
    if "package.json" in names:
        return "js"
    return "other"


def python_version_of(observations: str) -> str | None:
    """The first python version the rollout printed."""
    match = PYTHON_VERSION_RE.search(observations)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirrors", default=str(DEFAULT_MIRRORS))
    parser.add_argument("--published", default=DEFAULT_PUBLISHED_PARQUET)
    parser.add_argument(
        "--swesmith-map", default=str(HERE / "tasks" / "swesmith_map.json")
    )
    parser.add_argument("--out", default=str(HERE / "tasks" / "swesmith_env.json"))
    args = parser.parse_args()

    mapping = json.loads(Path(args.swesmith_map).read_text(encoding="utf-8"))["tasks"]

    # -- the language, from the mirror cache ------------------------------- #
    kinds: dict[str, str] = {}
    for slug in sorted({v["slug"] for v in mapping.values()}):
        mirror = Path(args.mirrors) / f"{slug}.git"
        kinds[slug] = classify_repository(mirror) if mirror.exists() else "unknown"

    # -- the python version, from the published rollouts ------------------- #
    import pyarrow.parquet as pq

    from check_env_fidelity import _observations  # local import: sibling module

    table = pq.read_table(args.published, columns=["task", "conversations"])
    versions: dict[str, str] = {}
    for task, conversation in zip(
        table.column("task").to_pylist(),
        table.column("conversations").to_pylist(),
    ):
        if task not in mapping or task in versions or not conversation:
            continue
        version = python_version_of("\n".join(_observations(conversation)))
        if version:
            versions[task] = version

    env: dict[str, Any] = {}
    for task, entry in mapping.items():
        kind = kinds.get(entry["slug"], "unknown")
        version = versions.get(task)
        if kind == "python" and version not in KNOWN_PYTHONS:
            # A version the image does not carry: keep the image's default
            # and say so, rather than silently pretending.
            version = None
        env[task] = {
            "kind": kind,
            "slug": entry["slug"],
            "python_version": version,
            "image": (
                "rsi/oc-base-swesmith-py:1"
                if kind == "python"
                else "rsi/oc-base-swesmith-go:1"
            ),
        }

    summary: dict[str, Any] = defaultdict(int)
    for entry in env.values():
        summary[f"kind:{entry['kind']}"] += 1
        summary[f"python:{entry['python_version']}"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"tasks": env, "summary": dict(sorted(summary.items()))},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(dict(sorted(summary.items())), indent=2))
    print(f"out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
