"""Build the Harbor task registry for the full OpenThoughts-Agent re-annotation.

The whole point of this script is the **family-shared base image**.  The
published dataset has 31,600 rollouts over 29,765 tasks in four families, and
the naive way to give each task its own environment -- one Docker image per
task -- would mean 29,765 image builds before the first agent even starts.
Instead:

* each **family** gets one base image (four images, two builds), and
* each **task** gets only the per-task *overlay*: the instruction, a
  ``task.toml`` that names the family image, and -- for SWE-Smith -- a setup
  script that prepares the one thing that differs per task.

What differs per family, and what does not
------------------------------------------

``issue`` / ``tezos`` / ``superuser`` share one image
    The published rollouts started in an **empty** ``/app`` on ubuntu 24.04.
    That is not an inference: the published SuperUser rollout for
    ``superuser-005067_copy0007`` literally prints ``ls -la /app/`` -> ``total 0``
    before it installs anything, and the upstream generator
    (``data/swesmith/task_templates.py``) builds these tasks as
    ``FROM ubuntu:24.04`` + ``WORKDIR /app`` + ``python3 python3-pip`` and
    nothing else.  The apparent "seed content" in ~5% of the rollouts is
    content the **agent** created before the listing (``bench.sh``,
    ``__pycache__``, a downloaded tarball), not content of the container.
    So these three families need **no per-task setup at all**: the shared
    image *is* the environment.

``swesmith`` needs one overlay
    The published SWE-Smith rollouts ran in the official SWE-Smith image with
    the repository checked out at the instance's bug-injected commit.  The
    upstream task Dockerfile is literally::

        FROM {official image}
        RUN git fetch && git checkout {instance_id}

    and the SWE-Smith mirror repository has **one branch per instance**,
    named exactly like the instance id.  So the per-task part is a checkout
    plus the editable install -- a script, not an image.

The unit of work
----------------

The published dataset has 31,600 rows.  They are **31,600 independent
rollouts**, not 29,765: 1,432 tasks were rolled out more than once, and 766
of those repeats used a *different* instruction (a rewritten problem
statement, or one with a ``CONSTRAINT:`` footer added or removed).  Two
rows of the same task with the same instruction still carry two different
conversations -- they are two rollouts, not a duplicated row.

So this script emits **one Harbor task per dataset row**.  The rows beyond
the first of a task get a directory name that carries the row index, so the
output can be joined back to the dataset row by row.  The extra rows of a
task reuse that task's environment, which is why they cost a rollout and not
an environment.

Usage::

    python opencode_sft/build_registry.py --out opencode_sft/registry
    python opencode_sft/build_registry.py --out /tmp/slice --limit-per-family 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from shared_network import SHARED_NETWORK

HERE = Path(__file__).resolve().parent

#: The four families, and the shared base image each one runs on.
#:
#: ``issue``, ``tezos`` and ``superuser`` are three tags of the *same* image:
#: their environments are identical (an empty ``/app`` on ubuntu 24.04), and
#: three tags rather than one because the families are separate upstream
#: datasets and may diverge later.
#:
#: SWE-Smith has **two** images, because the published rollouts ran in two
#: different kinds of official SWE-Smith image: a Python repository ran in
#: ubuntu 22.04 with an environment named ``testbed`` active, and a Go
#: repository ran in debian 12 with Go at ``/usr/local/go`` but not on
#: ``PATH``.  Which one a task needs is decided per task by
#: ``opencode_sft/derive_swe_env.py``.
FAMILY_IMAGE: dict[str, str] = {
    "swesmith": "rsi/oc-base-swesmith-py:1",
    "issue": "rsi/oc-base-issue:1",
    "tezos": "rsi/oc-base-tezos:1",
    "superuser": "rsi/oc-base-superuser:1",
}

#: Where the container starts, per family.
FAMILY_CWD: dict[str, str] = {
    "swesmith": "/testbed",
    "issue": "/app",
    "tezos": "/app",
    "superuser": "/app",
}

#: The host directory that holds the SWE-Smith mirror cache, bind-mounted
#: read-only into the SWE-Smith containers.  It is optional: the setup
#: script falls back to cloning from GitHub when a mirror is missing.
MIRROR_DIR = Path("/private/tmp/rsi-swe-mirrors")

#: The host directory that holds uv's download cache, so the second task on
#: a repository does not download the same wheels again.
UV_CACHE_DIR = Path("/private/tmp/rsi-uv-cache")

TASK_TOML = """version = "1.0"

[metadata]
author_name = "OpenThoughts-Agent re-annotation"
author_email = "noreply@localhost"
difficulty = "hard"
category = "software-engineering"
tags = ["agent", "terminal", "{family}"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 1800.0

# The family image.  Setting ``docker_image`` tells Harbor to use a
# pre-built image and to skip the build entirely -- that is what makes
# 31,600 tasks affordable: there are four images in total, not one per task.
[environment]
docker_image = "{image}"
"""

TEST_SH = """#!/bin/bash
# The re-annotation does not score anything -- the Harbor job runs with the
# verifier disabled, exactly like the upstream datagen job.  This file exists
# so that the task directory is a complete Harbor task.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
"""

COMPOSE_YAML = """# The per-task overlay of the SWE-Smith family.
#
# The image is shared by all {count} SWE-Smith tasks; the only per-task part
# of the environment is the setup script mounted here.  It is mounted rather
# than baked into an image because baking it in would mean one image build
# per task -- 7,693 builds for this family alone.
services:
  main:
    volumes:
      # The task's own initialisation: checkout the instance branch of the
      # SWE-Smith mirror repository.
      - ./setup.sh:/rsi/setup.sh:ro{extra_mounts}
{network}
"""

#: The compose fragment every task carries: the one shared Docker network.
#:
#: Harbor gives each trial its own compose project and therefore its own
#: bridge network, and those come out of a pool of sixteen.  A trial that
#: fails leaks its network, and once the pool is empty every later trial
#: fails to start with "all predefined address pools have been fully
#: subnetted" -- which looks like an agent failure and is not one.  See
#: ``opencode_sft/shared_network.py``.
NETWORK_SNIPPET = """
# Every trial of the re-annotation shares one Docker network, so that the
# per-trial networks cannot exhaust Docker's address pool.
networks:
  default:
    name: {name}
    external: true
"""

#: The overlay of the families whose image *is* the environment.
PLAIN_COMPOSE_YAML = """# The per-task overlay of the {family} family.
#
# The image is shared by every task of the family and the working directory
# starts out empty, so there is nothing to set up per task.  The overlay only
# points the trial at the shared Docker network.
{network}"""

#: The mirror cache is only mounted for the Python tasks: the Go and
#: JavaScript tasks do not use uv, and the Go image does not have it.
PYTHON_MOUNTS = """
      # The SWE-Smith mirror cache, built once on the host.  Optional: the
      # setup script falls back to cloning from GitHub.
      - {mirror_dir}:/rsi/mirrors:ro
      # uv's download cache, shared between the trials so the second task on
      # a repository does not download the same wheels again.
      - {uv_cache_dir}:/var/cache/uv
"""
#: The shared part of the SWE-Smith setup script: get the repository at the
#: instance's bug-injected commit.
_SETUP_COMMON = """#!/bin/bash
# The per-task initialisation of the SWE-Smith family.
#
# The published SWE-Smith rollouts ran in the official SWE-Smith image with
# the repository checked out at the instance's bug-injected commit.  The
# upstream task template renders exactly that:
#
#     FROM {{official image}}
#     RUN git fetch && git checkout {{instance_id}}
#
# The official image is amd64-only and 1.37 GB, so the family image
# reconstructs its contents natively and this script does the checkout.
#
# The SWE-Smith mirror repository has one branch per instance and the branch
# is named exactly like the instance id, so the instance's state is reachable
# by name.
set -euo pipefail

REPO_URL='{clone_url}'
INSTANCE_ID='{instance_id}'
SLUG='{slug}'
MIRROR_DIR='/rsi/mirrors'

mkdir -p /testbed /logs

mirror="${{MIRROR_DIR}}/${{SLUG}}.git"
if [ -d "${{mirror}}" ]; then
    # --shared makes the clone borrow the mirror's object store instead of
    # copying it, so the checkout takes a second even for pandas (a clone
    # from GitHub takes 53).
    git clone --shared "${{mirror}}" /testbed
else
    git clone --filter=blob:none --single-branch --branch "${{INSTANCE_ID}}" \\
        "${{REPO_URL}}" /testbed
fi

cd /testbed
git checkout -f "${{INSTANCE_ID}}" || git checkout -f "origin/${{INSTANCE_ID}}"
git reset --hard
"""

#: The Python variant: the environment named ``testbed`` has to run the
#: python version this task's own rollout ran on, and the package has to be
#: installed editable.
SETUP_SH_PYTHON = (
    _SETUP_COMMON
    + """
# The environment named "testbed" has to run the python version this task's
# own rollout ran on: the published rollouts answered "python --version"
# with 3.10.16 (3,658 rows), 3.10.18 (1,129), 3.11.11 (134) or 3.11.13
# (67), so the image carries all four and the setup script picks the right
# one.  Recreating the environment costs a second, so it is only done when
# the version actually differs.
PYTHON_VERSION='{python_version}'
VENV='/opt/conda/envs/testbed'

current="$(python --version 2>&1 | awk '{{print $2}}')"
if [ "${{current}}" != "${{PYTHON_VERSION}}" ]; then
    uv venv --clear --python "${{PYTHON_VERSION}}" --seed "${{VENV}}"
fi

# The editable install is what puts the package on the path and what leaves
# the <Name>.egg-info in the repository root -- the published rollout's first
# ``ls -la`` in /testbed shows ``PyPika.egg-info`` next to ``pypika/``, and
# the agent's own ``import pypika`` resolves to /testbed/pypika/__init__.py.
uv pip install -e /testbed
"""
)

#: The Go and JavaScript variant: nothing else to do.  The Go toolchain is
#: at /usr/local/go and deliberately **not** on PATH, exactly like the
#: original image, so the rollout starts with "go: command not found" and
#: the agent has to find the toolchain itself.
SETUP_SH_PLAIN = _SETUP_COMMON


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(index: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_task(
    family: str,
    dir_name: str,
    instruction_source: Path,
    target: Path,
    swe: dict[str, Any] | None,
    swe_count: int,
    image: str,
    setup_body: str | None,
    python_mounts: bool,
) -> dict[str, Any]:
    """Write one Harbor task directory."""
    (target / "environment").mkdir(parents=True, exist_ok=True)
    (target / "tests").mkdir(parents=True, exist_ok=True)

    # The instruction is copied byte for byte -- no reformatting, no wrapper.
    shutil.copyfile(instruction_source, target / "instruction.md")
    (target / "task.toml").write_text(
        TASK_TOML.format(family=family, image=image), encoding="utf-8"
    )
    test_sh = target / "tests" / "test.sh"
    test_sh.write_text(TEST_SH, encoding="utf-8")
    test_sh.chmod(0o755)

    # Every task carries the shared-network declaration, and the SWE-Smith
    # tasks additionally mount their per-task setup script.
    network = NETWORK_SNIPPET.format(name=SHARED_NETWORK)
    if setup_body is not None:
        extra_mounts = (
            PYTHON_MOUNTS.format(
                mirror_dir=MIRROR_DIR.as_posix(),
                uv_cache_dir=UV_CACHE_DIR.as_posix(),
            )
            if python_mounts
            else ""
        )
        (target / "environment" / "docker-compose.yaml").write_text(
            COMPOSE_YAML.format(
                count=swe_count, extra_mounts=extra_mounts, network=network
            ),
            encoding="utf-8",
        )
        setup = target / "environment" / "setup.sh"
        setup.write_text(setup_body, encoding="utf-8")
        setup.chmod(0o755)
    else:
        # The image *is* the environment for these families: an empty /app on
        # ubuntu 24.04.  The only thing the overlay adds is the shared
        # network.
        (target / "environment" / "docker-compose.yaml").write_text(
            PLAIN_COMPOSE_YAML.format(family=family, network=network), encoding="utf-8"
        )

    return {
        "task": dir_name,
        "family": family,
        "instruction_sha256": sha256(target / "instruction.md"),
        "instruction_chars": len(
            (target / "instruction.md").read_text(encoding="utf-8")
        ),
        "image": image,
        "cwd": FAMILY_CWD[family],
        "has_setup": setup_body is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "registry"))
    parser.add_argument("--instructions", default=str(HERE / "tasks" / "instructions"))
    parser.add_argument("--index", default=str(HERE / "tasks" / "index.jsonl"))
    parser.add_argument(
        "--swesmith-map", default=str(HERE / "tasks" / "swesmith_map.json")
    )
    parser.add_argument(
        "--swesmith-env",
        default=str(HERE / "tasks" / "swesmith_env.json"),
        help="the per-task SWE-Smith environment facts (language, python version)",
    )
    parser.add_argument(
        "--families",
        default="swesmith,issue,tezos,superuser",
        help="comma separated list of families to build",
    )
    parser.add_argument(
        "--limit-per-family",
        type=int,
        default=0,
        help="keep only the first N tasks of each family (0 = all)",
    )
    parser.add_argument(
        "--one-row-per-task",
        action="store_true",
        help=(
            "emit only the first dataset row of each task instead of one task "
            "per row (used for a quick smoke slice)"
        ),
    )
    parser.add_argument("--clean", action="store_true", help="remove the output first")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    instructions = Path(args.instructions)
    rows = load_rows(Path(args.index))
    swe_map = json.loads(Path(args.swesmith_map).read_text(encoding="utf-8"))["tasks"]
    swe_env = json.loads(Path(args.swesmith_env).read_text(encoding="utf-8"))["tasks"]

    root = Path(args.out)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    # The rows of a task, in dataset order.  The first row of a task keeps the
    # task's own name; the rest get the row index in the name so that the
    # output joins back to the dataset one row to one row.
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task"]].append(row)

    # The number of SWE-Smith tasks the overlay comment talks about.  (The
    # number of *repositories* behind them is a different number -- 150 --
    # and it is what the mirror cache holds.)
    swe_count = len(swe_map)
    manifest: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for task, task_rows in sorted(rows_by_task.items()):
        family = task_rows[0]["family"]
        if family not in families:
            continue
        if args.limit_per_family and counts[family] >= args.limit_per_family:
            continue
        swe = swe_map.get(task) if family == "swesmith" else None
        if family == "swesmith" and swe is None:
            raise KeyError(f"no SWE-Smith instance for {task!r}")

        # The image and the setup script depend on what kind of repository
        # the task is about: a Python repository runs on the ubuntu 22.04
        # image with a python environment named "testbed" and needs the
        # package installed editable; a Go repository runs on the debian 12
        # image and needs nothing but the checkout.
        image = FAMILY_IMAGE[family]
        setup_body: str | None = None
        python_mounts = False
        env_facts: dict[str, Any] = {}
        if family == "swesmith":
            facts = swe_env.get(task)
            if facts is None:
                raise KeyError(f"no SWE-Smith environment facts for {task!r}")
            env_facts = facts
            image = facts["image"]
            common = {
                "clone_url": swe["clone_url"],
                "instance_id": swe["instance_id"],
                "slug": swe["slug"],
            }
            if facts["kind"] == "python":
                setup_body = SETUP_SH_PYTHON.format(
                    **common, python_version=facts["python_version"] or "3.10.16"
                )
                python_mounts = True
            else:
                setup_body = SETUP_SH_PLAIN.format(**common)

        selected = task_rows[:1] if args.one_row_per_task else task_rows
        for position, row in enumerate(selected):
            dir_name = task if position == 0 else f"{task}__row{row['row']:05d}"
            target = root / family / dir_name
            if target.exists():
                shutil.rmtree(target)
            # The instruction of a row is the task's canonical instruction,
            # unless the row carries a variant; the variant file is named
            # after the task and the digest of the instruction.
            if position == 0:
                source = instructions / f"{task}.md"
            else:
                digest = row["instruction_sha256"]
                source = (
                    instructions / "variants" / f"{task}__{digest[:8]}.md"
                )
                if not source.exists():
                    source = instructions / f"{task}.md"

            entry = build_task(
                family,
                dir_name,
                source,
                target,
                swe,
                swe_count,
                image,
                setup_body,
                python_mounts,
            )
            entry.update(
                {
                    "path": str(target.relative_to(root)),
                    "dataset_rows": [r["row"] for r in task_rows]
                    if position == 0
                    else [row["row"]],
                    "source_rows": [row["row"]],
                    "canonical_row": task_rows[0]["row"],
                }
            )
            if swe is not None:
                entry["swesmith"] = {
                    "instance_id": swe["instance_id"],
                    "repo": swe["repo"],
                    "base_commit": swe["base_commit"],
                    "clone_url": swe["clone_url"],
                    **env_facts,
                }
            manifest.append(entry)
            counts[family] += 1

    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "tasks": len(manifest),
        "by_family": dict(sorted(counts.items())),
        "unique_dataset_tasks": len(rows_by_task),
        "dataset_rows": len(rows),
        "rows_covered": sum(len(e["source_rows"]) for e in manifest),
        "families": families,
        "images": {f: FAMILY_IMAGE[f] for f in families},
        "mirror_dir": str(MIRROR_DIR),
        "uv_cache_dir": str(UV_CACHE_DIR),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
