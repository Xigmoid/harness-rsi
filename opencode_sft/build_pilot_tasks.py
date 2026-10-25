"""Build the Harbor task registry for the four-family pilot.

Every pilot task directory is generated from the **extracted instruction**
(``tasks/instructions/<task>.md``) rather than from a hand-written copy, so
the instruction the agent sees is byte-identical to the one in the published
dataset.  The sha256 of the instruction is recorded in the manifest for the
same reason.

The pilot does not score anything -- the job runs with the verifier disabled
-- so ``tests/test.sh`` only exists to make the directory a complete Harbor
task.

Usage::

    python opencode_sft/build_pilot_tasks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTRUCTIONS = HERE / "tasks" / "instructions"
REGISTRY = HERE / "tasks" / "registry"
PILOT_DIR = HERE / "pilot"

#: The four pilot tasks, one per family.
#:
#: The environments are reconstructed from the **published rollouts** rather
#: than from the task names, because the task ids are renumbered indices and
#: say nothing about the environment.  What the original rollouts prove:
#:
#: * every rollout starts in ``/app`` (IssueTasks / Tezos / SuperUser) or in
#:   ``/testbed`` (SWE-Smith), on **ubuntu:24.04**;
#: * the SWE-Smith ``/testbed`` *is* the repository root -- the original
#:   ``ls -la`` shows ``pypika/``, ``setup.py`` and ``PyPika.egg-info``
#:   side by side -- and the package is installed editable in a python 3.10
#:   environment named ``testbed``;
#: * the other three families start with an **empty** ``/app``: the original
#:   IssueTasks rollout found ``/app`` empty and wrote
#:   ``astropy/wcs/wcsapi/fitswcs.py`` from scratch, and the original
#:   SuperUser rollout installed GNU parallel itself.
#:
#: The upstream generator agrees: ``data/swesmith/task_templates.py`` builds
#: the plain tasks as ``FROM ubuntu:24.04`` + ``python3 python3-pip`` in
#: ``/app``, and the repository tasks on the official SWE-smith image.
#:
#: The one thing the pilot cannot reproduce is the official SWE-smith image
#: itself: it is **amd64 only** and 1.37 GB, and emulating it on this arm64
#: host makes every agent command slow.  The SWE-Smith task therefore
#: reconstructs the image's contents -- the same repository at the same
#: commit, the same python 3.10, the same editable install -- on the
#: arm64-native base.
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
"""

TEST_SH = """#!/bin/bash
# The pilot does not score the task -- the Harbor job runs with the verifier
# disabled.  This file exists so that the task directory is a complete Harbor
# task.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_task(
    family: str,
    task: str,
    dockerfile: str,
    registry: Path,
) -> dict[str, object]:
    source = INSTRUCTIONS / f"{task}.md"
    if not source.exists():
        raise FileNotFoundError(f"no extracted instruction for {task!r}: {source}")

    target = registry / family / task
    (target / "environment").mkdir(parents=True, exist_ok=True)
    (target / "tests").mkdir(parents=True, exist_ok=True)

    # The instruction is copied byte for byte -- no reformatting, no wrapper.
    shutil.copyfile(source, target / "instruction.md")
    (target / "task.toml").write_text(TASK_TOML.format(family=family), encoding="utf-8")
    (target / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (target / "tests" / "test.sh").write_text(TEST_SH, encoding="utf-8")
    (target / "tests" / "test.sh").chmod(0o755)

    return {
        "family": family,
        "task": task,
        "path": str(target.relative_to(HERE)),
        "instruction_source": str(source.relative_to(HERE)),
        "instruction_sha256": sha256(target / "instruction.md"),
        "instruction_chars": len((target / "instruction.md").read_text(encoding="utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PILOT_DIR / "registry"))
    parser.add_argument(
        "--swesmith-instance",
        default=str(PILOT_DIR / "swesmith_instance.json"),
        help="JSON file with the SWE-smith instance (repo, commit) to use",
    )
    args = parser.parse_args()

    registry = Path(args.out)
    registry.mkdir(parents=True, exist_ok=True)

    swesmith = json.loads(Path(args.swesmith_instance).read_text(encoding="utf-8"))

    tasks = [
        (
            "swesmith",
            "swesmith-00000_copy0001",
            # A **raw** f-string: in a regular triple-quoted string Python eats
            # a backslash-newline as its own line continuation, which would
            # collapse the Dockerfile's own continuations into one line.
            rf"""FROM rsi/oc-pilot:2

# The SWE-smith environment, reconstructed.
#
# The upstream image is the repository with the bug already introduced, the
# repository sitting at /testbed, and the package installed editable in a
# python 3.10 environment named "testbed".  The original rollout's first
# `ls -la` in /testbed is the proof of the layout:
#
#   (testbed) root@3243371d-...:/testbed# ls -la
#   ...  PyPika.egg-info  pypika  setup.py  tox.ini
#
# so /testbed is the repository root -- not a parent of it.
RUN mkdir -p /testbed /logs
WORKDIR /testbed
RUN git clone --filter=blob:none {swesmith['clone_url']} /testbed \
 && cd /testbed \
 && git fetch --all --tags \
 && git checkout {swesmith['commit']} \
 && git reset --hard {swesmith['commit']} \
 && git log -1 --oneline

# python 3.10, the version the SWE-smith image ships, in an environment named
# "testbed" (the upstream one is a conda env; a venv plays the same role for
# everything the agent can observe: `python` is 3.10 and the package imports).
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# 3.10.16 is the version the SWE-smith images shipped: 6,913 of the
# published rollouts that printed `Python --version` answer 3.10.16,
# and only 2,749 answer 3.10.18.
RUN uv python install 3.10.16 \
 && uv venv --python 3.10.16 --seed /opt/testbed
# --seed puts pip inside the venv.  Without it `pip` resolves to the
# *system* pip, which refuses to install anything ("externally
# managed environment", PEP 668) -- the upstream conda environment has no
# such rule, so an agent that runs `pip install pytest` there succeeds.
ENV VIRTUAL_ENV=/opt/testbed
ENV PATH="/opt/testbed/bin:$PATH"

# The editable install is what puts PyPika.egg-info in the repository, and it
# is what makes `import pypika` work -- the original rollout ran
# `python -c "from pypika import Query, Table"` and got SQL back.
#
# The repository's own requirements-dev.txt is deliberately **not** installed:
# it pins Sphinx 1.6.5 and tox 3.14.3, which do not install on python 3.10,
# and the original rollout never used any of those tools -- it ran `ls`,
# `grep`, `cat`, `git show` and plain `python -c`.  The upstream verifier
# builds its own pytest environment when it scores the task.
RUN uv pip install -e /testbed

WORKDIR /testbed
""",
        ),
        (
            "issue",
            "issue-000000_copy0001",
            """FROM rsi/oc-pilot:2

# The IssueTasks environment: ubuntu 24.04 with python3, and an **empty**
# /app.  The original rollout opened /app, found it empty, and wrote
# astropy/wcs/wcsapi/fitswcs.py from scratch -- so there is no astropy
# checkout to provide, and cloning one would give the agent an environment
# the original never had.
WORKDIR /app
""",
        ),
        (
            "tezos",
            "tezos-0000",
            """FROM rsi/oc-pilot:2

# A question about a baking node's clock service.  The original rollout
# answers `timedatectl: command not found` and `systemctl: command not
# found` -- this is a plain ubuntu 24.04 container, not a systemd one.
WORKDIR /app
""",
        ),
        (
            "superuser",
            "superuser-000000_copy0000",
            """FROM rsi/oc-pilot:2

# The question is about GNU parallel, and the original rollout **installed
# parallel itself** (`apt-get update && apt-get install -y parallel`) after
# finding it missing.  Pre-installing it here would remove the step from the
# trace and change what the agent observes, so it is deliberately not
# installed; the apt sources are ubuntu's own, exactly as in the original.
WORKDIR /app
""",
        ),
    ]

    manifest = []
    for family, task, dockerfile in tasks:
        manifest.append(build_task(family, task, dockerfile, registry))

    out_file = Path(args.out).parent / "manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps({"tasks": manifest}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({"tasks": manifest}, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
