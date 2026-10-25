"""The containers that a killed trial leaves behind.

Why this exists
---------------
Harbor starts every trial as its own compose project, and a trial that ends
cleanly is torn down with it.  A trial that does **not** end cleanly -- the
job is cancelled, the driver is killed and restarted, the machine reboots --
leaves its container *running*, and a running container is not touched by
``docker container prune``.

On a long run that adds up.  The re-annotation keeps two harnesses busy at
once, and the Docker memory ceiling on this host is 9.7 GB.  Measured:
after a couple of restarts the machine carried 84 containers instead of the
34 that were actually working, and 8.4 GB of the 9.7 GB was in use -- close
enough to the ceiling that the next batch of opencode trials could have been
killed by the OOM killer instead of finishing.

So the containers of the *previous* batches are removed explicitly, at the
start of the next batch.

The scope is deliberately narrow
--------------------------------
The two drivers run at the same time, against the same registry, and the
same image.  A cleanup that keyed on the image, or on "every container that
is not in my current batch", would remove the *other* driver's live trials.
So the rule is:

    a container is removed when its compose project belongs to **this**
    run root and does **not** belong to the batch that is about to run

Both halves matter.  The first keeps the other driver's containers alone; the
second keeps this driver's own running trials alone.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from pathlib import Path


def _projects_of(run_root: Path) -> set[str]:
    """The compose projects of every trial this run root has ever started.

    Harbor names the compose project after the trial directory, lowercased,
    with ``__env`` appended; the container is that name plus ``-main-1``.
    """
    projects: set[str] = set()
    for batch in run_root.glob("batch-*/*/"):
        for trial in batch.iterdir():
            if trial.is_dir():
                projects.add(trial.name.lower() + "__env")
    return projects


def _projects_of_batch(batch_dir: Path) -> set[str]:
    """The compose projects of the trials of one batch."""
    projects: set[str] = set()
    for stamp in batch_dir.glob("*/"):
        for trial in stamp.iterdir():
            if trial.is_dir():
                projects.add(trial.name.lower() + "__env")
    return projects


def remove_stale_containers(run_root: Path, batch_dir: Path) -> int:
    """Remove the containers of this run root's earlier batches.

    Returns the number of containers removed.
    """
    if not run_root.exists() or not batch_dir.exists():
        return 0
    stale = _projects_of(run_root) - _projects_of_batch(batch_dir)
    if not stale:
        return 0

    listed = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    removed = 0
    for name in listed.stdout.split():
        project = name[: -len("-main-1")] if name.endswith("-main-1") else name
        if project not in stale:
            continue
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
        removed += 1
    return removed


def kill_stale_jobs(run_root: Path) -> int:
    """Kill the Harbor job of an earlier batch of this run root.

    Why
    ---
    The driver starts ``harbor job start`` as a **child process**.  When the
    driver itself is killed -- by the supervisor, by a crash, by a reboot --
    that job keeps running: it is not in the driver's process group, and
    Harbor does not die with its parent.  It goes on starting trials that
    nobody will ever harvest, and the containers pile up.

    Measured on this host: after three restarts there were six Harbor jobs
    running at once -- two of them current, four of them orphans from
    drivers that had died -- and 82 containers instead of the 34 that were
    actually working, which is 8.4 GB of the 9.7 GB the Docker VM has.

    ``SIGTERM`` is not enough: Harbor catches it and keeps going, so the
    process has to be killed outright.

    Which job belongs to which run root
    -----------------------------------
    The two drivers run at the same time, against the same registry and the
    same image, and their run roots are *not* disjoint as strings::

        /private/tmp/rsi-harbor/clawgym
        /private/tmp/rsi-harbor/clawgym-opencode

    The second **contains** the first.  A substring test would therefore let
    the mini driver kill the opencode driver's live job -- and the opencode
    trials would die with exit 137 (SIGKILL), which looks like an agent
    crash and is not one.

    So the job file is taken from the ``-c`` option and compared as a
    **path prefix**: ``<run root>/`` has to be the start of the job file's
    path.  That is exact: the opencode job file starts with
    ``/private/tmp/rsi-harbor/clawgym-opencode/``, which does not start with
    ``/private/tmp/rsi-harbor/clawgym/``.

    It is called *before* the new job is started, so there is no job of
    this run root that it could kill by accident.
    """
    listed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    prefix = str(run_root).rstrip("/") + "/"
    killed = 0
    for line in listed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            argv = shlex.split(args)
        except ValueError:
            continue
        # The command is a *Python* interpreter running the console script:
        #
        #   /opt/homebrew/.../Python /Users/.../.venv312/bin/harbor job start \
        #       -c <job file> -o <batch dir> --yes
        #
        # so argv[0] is the interpreter, not "harbor".  The script is looked
        # for by its base name anywhere in the argument list -- an exact
        # match on "harbor" would never fire, and the cleanup would silently
        # do nothing.
        if not any(Path(arg).name == "harbor" for arg in argv):
            continue
        if "job" not in argv or "start" not in argv:
            continue
        job_file = None
        for index, arg in enumerate(argv):
            if arg == "-c" and index + 1 < len(argv):
                job_file = argv[index + 1]
                break
        if not job_file or not job_file.startswith(prefix):
            continue
        try:
            os.kill(int(pid_text), signal.SIGKILL)
            killed += 1
        except (OSError, ValueError):
            continue
    return killed
