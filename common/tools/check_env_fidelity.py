"""Check a pilot environment against the environment of the published rollout.

The pilot environments were reconstructed from the published dataset, so the
dataset is the ground truth -- and it carries the environment facts in two
places:

* the ``Current Terminal Screen:`` block at the end of the first message,
  which is the prompt the original agent started from (the working directory,
  and the conda environment if one was active);
* the first ``ls -la`` of the original rollout, which is the *content* of
  that directory before the agent touched anything.

This script reads those facts out of the published parquet, runs the same
observations inside the built image, and reports the differences.  It does
not look at the pilot's own notes: everything comes from the dataset and from
the container.

Usage::

    python common/tools/check_env_fidelity.py \
        --registry work/pilot/registry \
        --published /path/to/train.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from hf_parquet import published_parquet_default
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The marker the terminus-2 harness used to append the terminal screen.
SCREEN_MARKER = "Current terminal state:\nCurrent Terminal Screen:\n"

#: The marker before the terminal output of an observation.
OUTPUT_MARKER = "New Terminal Output:"

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def family_of(task: str) -> str:
    for prefix in ("swesmith", "issue", "tezos", "superuser"):
        if task.startswith(prefix):
            return prefix
    return "other"


# --------------------------------------------------------------------------- #
# the published facts
# --------------------------------------------------------------------------- #


def published_facts(parquet: Path, tasks: list[str]) -> dict[str, dict[str, Any]]:
    """The environment facts of the published rows of the given tasks."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    all_tasks = table.column("task").to_pylist()
    conversations = table.column("conversations").to_pylist()

    wanted = set(tasks)
    facts: dict[str, dict[str, Any]] = {}
    for task, conversation in zip(all_tasks, conversations):
        if task not in wanted or task in facts or not conversation:
            continue
        first = conversation[0].get("content", "")
        facts[task] = {
            "cwd": _cwd_of(first),
            "conda_env": _conda_env_of(first),
            "first_ls": _first_ls(conversation),
            "os_release": _grep(conversation, r'PRETTY_NAME="([^"]+)"'),
            "python_version": _grep(conversation, r"Python (\d+\.\d+\.\d+)"),
            "missing_commands": _missing_commands(conversation),
        }
    return facts


def _cwd_of(first_message: str) -> str | None:
    index = first_message.find(SCREEN_MARKER)
    if index < 0:
        return None
    screen = first_message[index + len(SCREEN_MARKER) :]
    for line in screen.splitlines():
        line = line.strip()
        if not line or line.startswith("("):
            continue
        match = re.search(r":([^#]+)#\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def _conda_env_of(first_message: str) -> str | None:
    index = first_message.find(SCREEN_MARKER)
    if index < 0:
        return None
    screen = first_message[index + len(SCREEN_MARKER) :]
    for line in screen.splitlines():
        match = re.match(r"^\((\w+)\)\s", line.strip())
        if match:
            return match.group(1)
    return None


def _observations(conversation: list[dict[str, Any]]) -> list[str]:
    outputs = []
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        index = content.find(OUTPUT_MARKER)
        if index >= 0:
            outputs.append(content[index + len(OUTPUT_MARKER) :])
    return outputs


def _first_ls(conversation: list[dict[str, Any]]) -> list[str] | None:
    """The file names of the first ``ls -la`` of the rollout."""
    for output in _observations(conversation):
        match = re.search(r"#\s*ls -la\s*\n(.*?)(?=\n\S*root@|\n\S*\(testbed\) root@|\Z)", output, re.S)
        if not match:
            continue
        names = []
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("total "):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[-1]
            if name in (".", ".."):
                continue
            names.append(name)
        return names
    return None


def _grep(conversation: list[dict[str, Any]], pattern: str) -> str | None:
    for output in _observations(conversation):
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return None


def _missing_commands(conversation: list[dict[str, Any]]) -> list[str]:
    """The commands the original rollout found missing."""
    missing = []
    for output in _observations(conversation):
        for match in re.finditer(r"^(\w[\w.-]*): command not found$", output, re.M):
            missing.append(match.group(1))
    return sorted(set(missing))


# --------------------------------------------------------------------------- #
# the container facts
# --------------------------------------------------------------------------- #

#: What to look at inside the container, per family.  Each entry is a shell
#: snippet; the output is compared with the published facts.
PROBES: dict[str, list[tuple[str, str]]] = {
    "swesmith": [
        ("cwd", "pwd"),
        ("ls", "ls -A"),
        ("git_head", "git log -1 --oneline 2>&1 | head -1"),
        ("python", "python --version 2>&1"),
        ("import", "python -c 'import pypika; print(pypika.__file__)' 2>&1"),
    ],
    "issue": [
        ("cwd", "pwd"),
        ("ls", "ls -A"),
        ("python", "python3 --version 2>&1"),
        ("os", "grep PRETTY_NAME /etc/os-release"),
    ],
    "tezos": [
        ("cwd", "pwd"),
        ("ls", "ls -A"),
        ("python", "python3 --version 2>&1"),
        ("os", "grep PRETTY_NAME /etc/os-release"),
        ("timedatectl", "command -v timedatectl || echo 'MISSING'"),
        ("systemctl", "command -v systemctl || echo 'MISSING'"),
    ],
    "superuser": [
        ("cwd", "pwd"),
        ("ls", "ls -A"),
        ("python", "python3 --version 2>&1"),
        ("os", "grep PRETTY_NAME /etc/os-release"),
        ("parallel", "command -v parallel || echo 'MISSING'"),
    ],
}


def build_task_image(environment_dir: Path, tag: str) -> None:
    """Build a task image the way Harbor does: the ``environment`` directory
    is the build context and the Dockerfile lives inside it."""
    result = subprocess.run(
        ["docker", "build", "-f", str(environment_dir / "Dockerfile"), "-t", tag, str(environment_dir)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker build failed for {tag}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def container_facts(image: str, family: str) -> dict[str, str]:
    """Run the probes inside a container of the image."""
    facts: dict[str, str] = {}
    for key, snippet in PROBES[family]:
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", image, "bash", "-lc", snippet],
            capture_output=True,
            text=True,
            timeout=180,
        )
        facts[key] = (result.stdout or result.stderr).strip()
    return facts


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def compare(task: str, published: dict[str, Any], actual: dict[str, str]) -> list[str]:
    """The differences between the published environment and the container."""
    problems: list[str] = []
    family = family_of(task)

    if published.get("cwd") and actual.get("cwd") != published["cwd"]:
        problems.append(
            f"cwd: published {published['cwd']!r}, container {actual.get('cwd')!r}"
        )

    published_ls = published.get("first_ls")
    if published_ls is not None:
        container_ls = actual.get("ls", "").split()
        # The published listing is the directory *before* the agent touched it;
        # the container is the same directory, so the names have to match.
        missing = [name for name in published_ls if name not in container_ls]
        extra = [
            name
            for name in container_ls
            if name not in published_ls and not name.startswith("__pycache__")
        ]
        if missing:
            problems.append(f"files in the published directory but not in the container: {missing}")
        if extra:
            problems.append(f"files in the container but not in the published directory: {extra}")

    if published.get("os_release"):
        if published["os_release"] not in actual.get("os", ""):
            problems.append(
                f"os: published {published['os_release']!r}, container {actual.get('os')!r}"
            )

    if published.get("python_version") and family != "swesmith":
        if published["python_version"] not in actual.get("python", ""):
            problems.append(
                f"python: published {published['python_version']}, container {actual.get('python')!r}"
            )

    for command in published.get("missing_commands") or []:
        if actual.get(command) != "MISSING" and command in actual:
            problems.append(
                f"{command}: the published rollout found it missing, the container has it"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", default=str(REPO / "work" / "pilot" / "registry")
    )
    parser.add_argument(
        "--published",
        default=published_parquet_default(),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="probe the existing task images instead of building them",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    registry = Path(args.registry)
    tasks: list[str] = []
    for task_toml in sorted(registry.glob("*/*/task.toml")):
        tasks.append(task_toml.parent.name)

    published = published_facts(Path(args.published), tasks)
    report: dict[str, Any] = {"tasks": {}}
    failures = 0
    for task in tasks:
        family = family_of(task)
        environment_dir = registry / family / task / "environment"
        tag = f"rsi/oc-pilot-task:{family}-{task}"
        if not args.skip_build:
            build_task_image(environment_dir, tag)
        actual = container_facts(tag, family)
        problems = compare(task, published.get(task, {}), actual)
        report["tasks"][task] = {
            "family": family,
            "image": tag,
            "published": published.get(task, {}),
            "container": actual,
            "problems": problems,
        }
        failures += bool(problems)

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{len(tasks) - failures}/{len(tasks)} environments match the published rollout")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
