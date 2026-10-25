"""Build the Harbor task registry for ``RUC-AIBOX/ClawGym-Task``.

The shape of the problem
------------------------

ClawGym ships **tasks**, not trajectories: a prompt, the files the task needs,
and a verifier.  So there is no published rollout to reconstruct -- the job is
to *create* trajectories, and to create them in an environment the task
actually fits.

What the dataset pins down
--------------------------

Every one of the 15,402 rows carries the same ``hook_code``::

    python3 reward/test.py /root/.openclaw/workspace

Three things follow from that one line:

1. the verifier lives at ``reward/test.py`` **relative to the working
   directory**, so the working directory is ``/root/.openclaw/workspace``;
2. the verifier takes the workspace root as its argument, and reads the task's
   inputs from ``<workspace>/input/`` and checks deliverables under
   ``<workspace>/output/``;
3. ``reward/`` is *visible to the agent*.  That is not an assumption: 15 of
   the prompts say "Do not write anything under reward/" and 12 say "Use only
   relative paths under input/, output/, and reward/".  Those instructions
   only make sense if the directory is there.  Hiding the verifier would
   change the task.

So the registry is exactly the task's own file tree, rooted at the workspace::

    <task>/
        task.toml            # names the shared image + the workspace
        instruction.md       # the prompt, verbatim
        environment/         # uploaded into the workspace before the agent runs
            input/...        # the task's input files
            reward/test.py   # the verifier
            <other paths>    # tasks whose inputs live outside input/
        tests/test.sh        # runs the verifier and writes the reward

Harbor does the injection for us: a task that names a pre-built
``docker_image`` and has no ``environment/Dockerfile`` gets its
``environment/`` directory uploaded into ``[environment].workdir`` when the
container starts.  One image therefore serves all 15,402 tasks, and the only
per-task cost is the file copy.

The reward
----------

Harbor's verifier reads a single float from ``/logs/verifier/reward.txt``,
but the ClawGym verifiers do not agree on a shape.  Probing all 15,402 of
them against an empty workspace (``common/registry/probe_verifiers.py``)
found two:

* 6,091 print ``{"reward": <float>, "<check>": <bool|float>, ...}`` -- the
  verifier computed its own aggregate;
* 9,302 print a flat ``{"<check>": 0.0, ...}`` with no ``reward`` key, where
  the aggregate is the mean of the checks.

(Probe runs on Python 3.11 also produced nine failures that are not real:
six verifiers use an f-string with a backslash, which is a syntax error
before 3.12, and three read ``input/`` without tolerating its absence.  On
3.12 with ``input/`` present, all 15,402 run clean.)

``tests/test.sh`` therefore prefers the explicit ``reward`` and falls back to
the mean of the numeric checks, and records which rule it used.

Usage::

    python common/registry/build_clawgym_registry.py \
        --jsonl .cache/clawgym/syn_task.jsonl \
        --out work/registry_clawgym
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

#: The one Docker network every trial joins.  See
#: ``common/shared_network.py`` for why.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.shared_network import SHARED_NETWORK  # noqa: E402

#: The family name, which is also the registry subdirectory.
FAMILY = "clawgym"

#: The one image every ClawGym task runs on.
IMAGE = "rsi/oc-base-clawgym:1"

#: The workspace.  It is the container's WORKDIR, it is where the verifier is
#: rooted, and it is where ``[environment].workdir`` points so that Harbor
#: uploads the task's files into it.
WORKSPACE = "/root/.openclaw/workspace"

#: The per-task overlay: the shared Docker network, and the task's payload.
#:
#: **Why the payload is a bind mount and not an upload.**  Harbor uploads a
#: task's ``environment/`` directory into the container's workdir when the
#: task names a pre-built image and has no ``environment/Dockerfile`` or
#: ``environment/docker-compose.yaml``.  But the shared network needs a
#: ``docker-compose.yaml`` (see ``common/shared_network.py``), and the
#: presence of that file switches the upload off.  The two mechanisms cannot
#: both be used, so the payload is mounted instead:
#:
#:   * the payload is mounted read-only at ``/rsi/payload``;
#:   * the container copies it into the workspace when it starts.
#:
#: One mount covers any number of top-level entries -- the ClawGym payloads
#: have up to a couple of hundred distinct top-level directories, and they
#: are not uniform (``input/``, ``config/``, ``scripts/``, ``data/``, and
#: dozens of one-off names, plus root-level files).  Mounting them one by one
#: would mean a compose file per task with a mount per entry.
#:
#: The copy also keeps the workspace ordinary: the agent gets writable
#: directories, exactly as it would from an upload, and the image's own
#: ``output/`` stays writable in the container's layer.
TASK_COMPOSE_YAML = """# The per-task overlay of the ClawGym family.
#
# The image is shared by every ClawGym task; the per-task part is the
# task's own files, and they arrive as a read-only mount that the container
# copies into the workspace when it starts.
services:
  main:
    volumes:
      - ./payload:/rsi/payload:ro
    command:
      - sh
      - -c
      - "cp -a /rsi/payload/. /root/.openclaw/workspace/ ; exec sleep infinity"
{network}
"""

#: The compose fragment every task carries: the one shared Docker network.
#:
#: Harbor gives each trial its own compose project and therefore its own
#: bridge network, and those come out of a pool of sixteen.  A trial that
#: fails leaks its network, and once the pool is empty every later trial
#: fails to start with "all predefined address pools have been fully
#: subnetted" -- which looks like an agent failure and is not one.  See
#: ``common/shared_network.py``.
NETWORK_SNIPPET = """
# Every trial of the re-annotation shares one Docker network, so that the
# per-trial networks cannot exhaust Docker's address pool.
networks:
  default:
    name: {name}
    external: true
"""

#: The payload lands here, inside ``environment/`` so that the relative
#: mount path is a plain ``./payload``.
PAYLOAD_DIR = "payload"

TASK_TOML = """version = "1.0"

[metadata]
author_name = "ClawGym-Task re-annotation"
author_email = "noreply@localhost"
difficulty = "hard"
category = "agent-task"
tags = ["agent", "terminal", "clawgym"]

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 1800.0

# One image for every task.  The task's own files (its inputs and its
# verifier) arrive through ``environment/``, which Harbor uploads into
# ``workdir`` when the container starts -- so the image only has to own the
# skeleton, and no task needs a build.
[environment]
docker_image = "{image}"
workdir = "{workspace}"
"""

#: The verifier driver.  It runs the task's own hook -- the same program,
#: with the same argument, in the same directory the hook was written for --
#: and turns the verifier's JSON into the single float Harbor reads.
TEST_SH = """#!/bin/bash
# Run the task's own verifier and record the reward.
#
# The hook is the one from the dataset:
#
#     python3 reward/test.py /root/.openclaw/workspace
#
# It is executed here from the workspace, so the relative path resolves the
# way it was written to resolve.
set -uo pipefail

WS={workspace}
mkdir -p /logs/verifier
cd "$WS" || {{ echo 0.0 > /logs/verifier/reward.txt; exit 0; }}

python3 reward/test.py "$WS" > /tmp/clawgym-verifier.json 2> /tmp/clawgym-verifier.err
RC=$?

# The verifiers do not agree on a shape: some print an explicit "reward",
# most print a flat mapping of check name -> number.  Prefer the explicit
# aggregate, fall back to the mean of the checks.
python3 - <<'PYEOF' > /logs/verifier/reward.txt
import json

RAW = ""
try:
    with open("/tmp/clawgym-verifier.json", encoding="utf-8", errors="replace") as handle:
        RAW = handle.read()
except OSError:
    RAW = ""

payload = None
decoder = json.JSONDecoder()
index = 0
while index < len(RAW):
    if RAW[index] not in "{{[":
        index += 1
        continue
    try:
        payload, end = decoder.raw_decode(RAW, index)
    except json.JSONDecodeError:
        index += 1
        continue
    index = end


def numbers(values):
    return [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


reward = 0.0
rule = "none"
if isinstance(payload, dict):
    explicit = payload.get("reward")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        reward = float(explicit)
        rule = "explicit"
    else:
        for key in ("scores", "checks", "results"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                values = numbers(inner.values())
                if values:
                    reward = sum(values) / len(values)
                    rule = "nested:" + key
                    break
        else:
            values = numbers(payload.values())
            if values:
                reward = sum(values) / len(values)
                rule = "flat"

print(reward)
with open("/logs/verifier/reward_rule.txt", "w", encoding="utf-8") as handle:
    handle.write(rule)
PYEOF

# The verifier's own output is kept next to the reward: when a trajectory
# looks wrong, this is the first thing to read.
cp /tmp/clawgym-verifier.json /logs/verifier/clawgym-verifier.json 2>/dev/null || true
cp /tmp/clawgym-verifier.err /logs/verifier/clawgym-verifier.err 2>/dev/null || true
exit 0
"""


def slugify(text: str, limit: int = 48) -> str:
    """A filesystem- and Harbor-safe slug."""
    out: list[str] = []
    previous_dash = False
    for char in text.lower():
        if char.isalnum():
            out.append(char)
            previous_dash = False
        elif not previous_dash:
            out.append("-")
            previous_dash = True
    slug = "".join(out).strip("-")
    return (slug[:limit].rstrip("-")) or "task"


def write_file(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        os.chmod(path, 0o755)


def build_task(
    row_index: int,
    record: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    """Materialise one Harbor task directory."""
    meta = record.get("meta_data") or {}
    task_id = meta.get("task_id") or ""
    prompt = record.get("prompt") or ""

    name = f"{FAMILY}-{row_index:06d}"
    if task_id:
        name = f"{name}-{slugify(task_id, 40)}"

    task_dir = out_root / FAMILY / name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    write_file(task_dir / "task.toml", TASK_TOML.format(image=IMAGE, workspace=WORKSPACE))
    write_file(task_dir / "instruction.md", prompt)

    # The task's own file tree, rooted at the workspace.  Every path in
    # ``input_files`` is relative to the workspace root, which is what the
    # verifier reads from and what the prompts refer to.
    written: list[str] = []
    payload_root = task_dir / "environment" / PAYLOAD_DIR
    for entry in record.get("input_files") or []:
        rel = (entry.get("file_path") or "").strip()
        if not rel:
            continue
        rel = rel.lstrip("/")
        target = payload_root / rel
        # Refuse to write outside the task directory.
        try:
            target.resolve().relative_to(task_dir.resolve())
        except ValueError:
            continue
        write_file(target, entry.get("content") or "")
        written.append(rel)

    # The shared network, and the payload mount.
    write_file(
        task_dir / "environment" / "docker-compose.yaml",
        TASK_COMPOSE_YAML.format(network=NETWORK_SNIPPET.format(name=SHARED_NETWORK)),
    )

    write_file(task_dir / "tests" / "test.sh", TEST_SH.format(workspace=WORKSPACE), executable=True)

    return {
        "task": name,
        "family": FAMILY,
        "row": row_index,
        "task_id": task_id,
        "validation_mode": meta.get("validation_mode"),
        "reward_aggregation": meta.get("reward_aggregation"),
        "primary_skill_id": (meta.get("metadata") or {}).get("primary_skill_id"),
        "difficulty": (meta.get("metadata") or {}).get("difficulty"),
        "macro_category": meta.get("macro_category"),
        "category": meta.get("category"),
        "n_files": len(written),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "hook_code": record.get("hook_code"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default=".cache/clawgym/syn_task.jsonl")
    parser.add_argument("--out", default="work/registry_clawgym")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="wipe the output first")
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / FAMILY).mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    entries: list[dict[str, Any]] = []
    with open(args.jsonl, encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if args.limit and row_index >= args.limit:
                break
            record = json.loads(line)
            entries.append(build_task(row_index, record, out_root))

    with manifest_path.open("w", encoding="utf-8") as out:
        for entry in entries:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "family": FAMILY,
        "image": IMAGE,
        "workspace": WORKSPACE,
        "tasks": len(entries),
        "with_task_id": sum(1 for e in entries if e["task_id"]),
        "total_files": sum(e["n_files"] for e in entries),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
