#!/usr/bin/env python3
"""Pre-build the task images so that Harbor's per-trial ``docker compose build``
is a cache hit.

Why this exists
---------------
Harbor builds the task image *inside every trial*::

    docker compose ... build          # context = <task>/environment

For the registry tasks that layer is cached by BuildKit, so the build is
cheap -- except for the tasks whose Dockerfile still has an ``apt-get install``
layer (all 101 LiveCode tasks do).  That layer needs the network, and under
concurrency 8 it either takes ~250 s or fails outright with

    E: Failed to fetch http://deb.debian.org/debian/... Unable to connect

which kills the whole trial with a ``RuntimeError`` before the agent ever
runs.  One livecode trial died exactly that way during the first study run.

Building the image once, up front and serially, turns the layer into a cache
hit: the ``apt-get`` layer is produced a single time (network used once,
retried on failure) and every later trial reuses it.

The toolchain is already baked into the base image, so the packages the
``apt-get`` layer installs are not what makes the agent work -- the layer is
pure upstream task boilerplate that we keep verbatim so the task content
stays byte-identical to the registry.

    python scripts/warm_task_images.py --benchmarks livecode
    python scripts/warm_task_images.py --benchmarks livecode ineqmath gpqa --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets" / "registry"


def task_dirs(benchmark: str) -> list[Path]:
    base = REGISTRY / benchmark
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if (d / "environment" / "Dockerfile").is_file():
            out.append(d)
    return out


def build_one(task: Path, benchmark: str, retries: int) -> tuple[str, str, float]:
    """Build one task image.  Returns (name, status, seconds)."""
    env_dir = task / "environment"
    name = f"rsi-warm/{benchmark}/{task.name}"
    cmd = ["docker", "build", "-t", name, str(env_dir)]
    t0 = time.time()
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return task.name, "ok", time.time() - t0
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        detail = " | ".join(tail)
        if attempt < retries:
            print(f"   {task.name}: attempt {attempt} failed, retrying ({detail})", flush=True)
            time.sleep(3 * attempt)
    return task.name, f"FAILED: {detail}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", default=["livecode"])
    ap.add_argument("--jobs", type=int, default=2, help="parallel builds (network-bound; keep low)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--only", nargs="*", default=None, help="explicit task names")
    args = ap.parse_args()

    targets: list[tuple[Path, str]] = []
    for b in args.benchmarks:
        if args.only:
            for name in args.only:
                targets.append((REGISTRY / b / name, b))
        else:
            for d in task_dirs(b):
                targets.append((d, b))
    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        print("nothing to build")
        return 0

    print(f"==> warming {len(targets)} task images with {args.jobs} parallel build(s)", flush=True)
    t0 = time.time()
    failures: list[tuple[str, str]] = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_one, d, b, args.retries): (d, b) for d, b in targets}
        done = 0
        for fut in futures.as_completed(futs):
            name, status, secs = fut.result()
            done += 1
            print(f"[{done}/{len(targets)}] {name}: {status} ({secs:.0f}s)", flush=True)
            if status != "ok":
                failures.append((name, status))

    print(f"\n==> done in {time.time() - t0:.0f}s; {len(failures)} failure(s)")
    for name, status in failures:
        print(f"   FAIL {name}: {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
