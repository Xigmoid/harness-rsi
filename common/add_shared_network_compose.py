#!/usr/bin/env python3
"""Give every task in a registry the shared-network compose overlay.

Why
---

Harbor gives **every trial** its own compose project, and every compose
project its own bridge network.  Docker hands those networks out of a fixed
address pool that is only sixteen wide by default, so a run with more than
sixteen trials in flight dies with::

    failed to create network ...: all predefined address pools have been
    fully subnetted

which reads like an agent failure and is not one.  The pool cannot be widened
on this host (``default-address-pools`` in ``~/.docker/daemon.json`` stops
Docker Desktop's engine from starting at all), so the leak is removed at the
other end: every trial joins **one** shared network, and the per-trial
networks that leaked are removed by name.

A task joins that network by declaring it in
``environment/docker-compose.yaml``.  Harbor merges the compose files --
base, prebuilt, then the task's own, the task's last, so it can override
scalars -- which makes the task's file an *overlay*: it only has to state
what differs.

The overlay cannot replace the task's own files: a task with a
``docker-compose.yaml`` no longer gets its ``environment/`` directory
uploaded into the container.  For the OpenThoughts registry that costs
nothing -- every one of its 31,600 tasks has an **empty** ``environment/``
directory, because the whole environment is in the family image.  For the
ClawGym registry the payload arrives as a bind mount instead (see
``registry_clawgym``).

Usage::

    python3 common/add_shared_network_compose.py \
        --registry work/registry --families issue,tezos,superuser
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: The overlay.  The comment matters: the file is read by a human six months
#: from now, and "why is there a compose file in a task that needs no
#: services?" is the first question.
COMPOSE = """\
# The per-task overlay of the {family} family.
#
# The image is shared by every task in this family and carries the whole
# environment, so this task needs no services of its own.  What it does need
# is the *network*: Harbor gives every trial its own compose project, and
# every project its own bridge network, and Docker hands those out of a
# fixed pool that is only sixteen wide.  A run with more trials in flight
# than that dies with "all predefined address pools have been fully
# subnetted", which looks like an agent failure and is not one.
#
# So every trial joins one shared network instead.
networks:
  default:
    name: rsi-shared
    external: true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--families",
        default="",
        help="comma separated; empty means every family in the registry",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = Path(args.registry).resolve()
    families = {f.strip() for f in args.families.split(",") if f.strip()}

    counts: Counter[str] = Counter()
    written = 0
    for family_dir in sorted(p for p in registry.iterdir() if p.is_dir()):
        family = family_dir.name
        if families and family not in families:
            continue
        for task_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            environment = task_dir / "environment"
            environment.mkdir(parents=True, exist_ok=True)
            path = environment / "docker-compose.yaml"
            content = COMPOSE.format(family=family)
            if path.exists() and path.read_text(encoding="utf-8") == content:
                counts["unchanged"] += 1
                continue
            if args.dry_run:
                counts["would_write"] += 1
                continue
            path.write_text(content, encoding="utf-8")
            written += 1
            counts["written"] += 1

    print(json.dumps({"registry": str(registry), "written": written, **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
