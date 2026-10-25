#!/usr/bin/env python3
"""Build the IFEval task set for Harbor.

IFEval (`google/IFEval`, Apache-2.0) is not in Harbor's registry, so the task
directory is generated here: one Harbor task per prompt, with a verifier that
checks the *verifiable instructions* of that prompt.

    python scripts/build_ifeval_tasks.py --write-base
    python scripts/build_ifeval_tasks.py --build-base
    python scripts/build_ifeval_tasks.py --out datasets/ifeval

**Architecture.**  All 541 tasks share a single base image that carries

* the verifier dependencies (pytest, pytest-json-ctrf, langdetect, nltk +
  the punkt models), so the verifier is deterministic and offline, and
* the ``rsi`` package itself in ``/opt/rsi`` (exposed through
  ``PYTHONPATH``), so the agent class ``rsi.harbor_agent.RsiMiniSweAgent``
  can be imported inside the ``uv tool`` venv that Harbor creates for
  mini-swe-agent -- that venv is isolated, so a ``pip install`` into the
  image's system Python would *not* be visible to it.

Each task therefore only pins ``environment.docker_image`` and ships its
prompt, the verifier and the test script.  The reward is the IFEval *strict
accuracy*: a prompt counts as followed only when **all** of its instructions
are followed.

The checkers come from :mod:`rsi.ifeval_checkers`, which is a faithful port of
Google's reference implementation and is verified against it by
``tests/test_ifeval_differential.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi.ifeval_checkers import INSTRUCTION_DICT  # noqa: E402

HF_RAW = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
HF_SHA256 = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
BASE_IMAGE = "rsi/ifeval-base:1"

TASK_TOML = """\
schema_version = "1.1"

[task]
name = "ifeval/{name}"
description = "IFEval prompt {key} -- verifiable instruction following."
authors = [{{ name = "google/IFEval" }}]
keywords = ["ifeval", "instruction-following"]

[metadata]
ifeval_key = "{key}"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
docker_image = "{image}"
build_timeout_sec = 900.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0

[verifier.env]

[environment.env]

[solution.env]
"""

TEST_SH = """\
#!/bin/bash
# Verifier for one IFEval prompt.  The dependencies are baked into the base
# image, so this runs offline.
set -u

pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
status=$?

if [ $status -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit 0
"""

TEST_OUTPUTS = '''\
"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = {prompt!r}
INSTRUCTION_IDS = {instruction_ids!r}
KWARGS = {kwargs!r}


def test_response_follows_every_instruction():
    try:
        with open("/app/response.txt", encoding="utf-8") as handle:
            response = handle.read()
    except FileNotFoundError:
        raise AssertionError("/app/response.txt was not written by the agent")
    assert response.strip(), "/app/response.txt is empty"
    results = check_prompt(PROMPT, response, INSTRUCTION_IDS, KWARGS)
    failed = [k for k, ok in results.items() if not ok]
    assert not failed, f"instructions not followed: {{failed}} (all: {{results}})"
'''

BASE_DOCKERFILE = """\
# Base image for the IFEval task set.
#
# It carries
#   * the verifier dependencies (pytest + the CTRF reporter, langdetect and
#     nltk with the punkt models), so the verifier is deterministic and does
#     not need the network at verification time, and
#   * the `rsi` package in /opt/rsi, exposed through PYTHONPATH, so that the
#     agent class rsi.harbor_agent.RsiMiniSweAgent can be imported inside the
#     isolated `uv tool` venv that Harbor creates for mini-swe-agent.
#
# It also **pre-installs the agent toolchain**.  Harbor installs
# mini-swe-agent inside every task container, which costs ~4 minutes per
# trial (uv + a Python 3.12 + litellm and friends) and dominates the
# wall-clock of a 541-task benchmark.  Baking the toolchain into the image
# turns that into a no-op check, because the pinned version is already
# there.  The version is pinned in *both* places on purpose: the image and
# the job file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    PYTHONPATH=/opt/rsi \\
    DEBIAN_FRONTEND=noninteractive

# Harbor's ensure_system_dependencies() installs exactly these.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends \\
      build-essential ca-certificates curl git \\
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
      pytest==8.4.1 \\
      pytest-json-ctrf==0.3.5 \\
      langdetect==1.0.9 \\
      nltk==3.10.3 \\
      pyyaml==6.0.3 \\
      numpy==2.4.6

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

# The agent toolchain, pinned to the same version the job file pins.
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \\
 && export PATH="/root/.local/bin:${PATH}" \\
 && uv python install 3.12 \\
 && uv tool install --python 3.12 mini-swe-agent==2.4.6 \\
      --with litellm --with orjson --with fastapi

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
"""

DOCKERIGNORE = """\
.venv
.venv312
.git
.pytest_cache
*.egg-info
runs
ledger
datasets
subsets
harness
audit
__pycache__
"""


def load_dataset(path: str | None) -> list[dict[str, Any]]:
    """Read the IFEval prompts from a local file or from the hub."""
    if path:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        print(f"downloading {HF_RAW}")
        with urllib.request.urlopen(HF_RAW, timeout=120) as response:
            raw = response.read().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write_base(out: Path) -> Path:
    """Write the base-image build context (Dockerfile + .dockerignore)."""
    base = out / "base"
    base.mkdir(parents=True, exist_ok=True)
    (base / "Dockerfile").write_text(BASE_DOCKERFILE, encoding="utf-8")
    (base / ".dockerignore").write_text(DOCKERIGNORE, encoding="utf-8")
    return base


def build_base(base: Path, image: str) -> int:
    """Build the shared base image (build context = the repository root)."""
    command = [
        "docker",
        "build",
        "-f",
        str(base / "Dockerfile"),
        "-t",
        image,
        str(ROOT),
    ]
    print(" ".join(command))
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/ifeval", help="output directory")
    parser.add_argument("--data", default=None, help="local ifeval_input_data.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only build the first N tasks")
    parser.add_argument(
        "--image", default=BASE_IMAGE, help="base image the tasks pin"
    )
    parser.add_argument(
        "--write-base", action="store_true", help="write the base-image build context"
    )
    parser.add_argument(
        "--build-base", action="store_true", help="docker build the base image"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing task directory"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_base or args.build_base:
        base = write_base(out)
        print(f"base build context: {base}")
        if args.build_base:
            status = build_base(base, args.image)
            if status != 0:
                print(f"docker build failed with status {status}")
                return status

    examples = load_dataset(args.data)
    if args.limit:
        examples = examples[: args.limit]

    unknown: set[str] = set()
    written = 0
    for example in examples:
        key = example.get("key") or example.get("id")
        name = f"prompt-{key}"
        task_dir = out / name
        if task_dir.exists() and not args.overwrite:
            continue
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        # Harbor's task validation requires an ``environment/`` directory to
        # exist, even when the task pins a pre-built image.  It must not
        # contain a Dockerfile, or Harbor would build an image per task
        # instead of using ``environment.docker_image``.
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "environment" / "README.md").write_text(
            "This task pins the pre-built image `rsi/ifeval-base:1`\n"
            "(see `datasets/ifeval/base/Dockerfile`).  There is deliberately\n"
            "no Dockerfile here: a Dockerfile would make Harbor build a\n"
            "separate image per task.\n",
            encoding="utf-8",
        )

        (task_dir / "task.toml").write_text(
            TASK_TOML.format(name=name, key=key, image=args.image), encoding="utf-8"
        )
        (task_dir / "tests" / "test.sh").write_text(TEST_SH, encoding="utf-8")
        (task_dir / "tests" / "test_outputs.py").write_text(
            TEST_OUTPUTS.format(
                prompt=example["prompt"],
                instruction_ids=list(example.get("instruction_id_list", [])),
                kwargs=list(example.get("kwargs", [])),
            ),
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text(
            f"{example['prompt']}\n\n"
            "Write your complete answer to the file `/app/response.txt`.\n"
            "The file must contain the answer and nothing else.\n",
            encoding="utf-8",
        )
        unknown |= {i for i in example.get("instruction_id_list", []) if i not in INSTRUCTION_DICT}
        written += 1

    print(f"wrote {written} task(s) to {out}")
    if unknown:
        print(f"! no checker for: {sorted(unknown)}")
        return 1
    print(f"all {len(INSTRUCTION_DICT)} instruction types have a checker")
    print(f"dataset sha256: {HF_SHA256}")
    print(f"base image: {args.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
