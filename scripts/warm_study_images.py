#!/usr/bin/env python3
"""Warm the Docker build cache for exactly the tasks the study runs.

Scope
-----
The study evaluates four home benchmarks (livecode, ineqmath, gpqa, ifeval) on
their 16-task dev splits.  Only those images need to exist; the registry
downloads ~100 tasks per benchmark and warming all of them would waste an
hour of network time.

Why warming matters
-------------------
Harbor runs ``docker compose build`` inside *every* trial.  For a task whose
Dockerfile still has an ``apt-get install`` layer, that layer needs the
network, and with ``n_concurrent_trials: 8`` the apt mirrors either take
~250 s or fail outright::

    E: Failed to fetch http://deb.debian.org/debian/... Unable to connect

A failed build kills the trial before the agent starts, and a 400 s build
stalls it.  Measured on the first study run: trials whose image was already
built spent 3 s in environment setup and 23 s running the agent; trials that
had to build spent 400 s and then died.

Building the image once, serially and with retries, turns the layer into a
cache hit.  The build cache is keyed on the layer content, not on the image
tag, so a warm layer is reused by Harbor's own build.

    python scripts/warm_study_images.py
    python scripts/warm_study_images.py --jobs 3 --benchmarks livecode
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets" / "registry"

#: The benchmarks the study uses as the four home benchmarks.
STUDY_BENCHMARKS = ["livecode", "ineqmath", "gpqa"]


def dev_tasks(benchmark: str) -> list[str]:
    """The frozen dev task names for one benchmark, from the job template."""
    job = ROOT / "harbor" / f"job.{benchmark}.yaml"
    if not job.is_file():
        return []
    doc = yaml.safe_load(job.read_text(encoding="utf-8"))
    for ds in doc.get("datasets", []) or []:
        path = ds.get("path") or ""
        if path.rstrip("/").endswith(f"/{benchmark}"):
            return list(ds.get("task_names") or [])
    return []


def build_one(task: Path, retries: int) -> tuple[str, str, float]:
    """Build one task image; retry because the apt layer needs the network."""
    env_dir = task / "environment"
    name = f"rsi-warm/{task.parent.name}/{task.name}"
    cmd = ["docker", "build", "-t", name, str(env_dir)]
    t0 = time.time()
    detail = ""
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return task.name, "ok", time.time() - t0
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = " | ".join(lines[-4:])
        if attempt < retries:
            print(f"   {task.name}: attempt {attempt} failed, retry ({detail})", flush=True)
            time.sleep(2 * attempt)
    return task.name, f"FAILED: {detail}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", default=STUDY_BENCHMARKS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    targets: list[Path] = []
    for b in args.benchmarks:
        for name in dev_tasks(b):
            d = REGISTRY / b / name
            if (d / "environment" / "Dockerfile").is_file():
                targets.append(d)
            else:
                print(f"   (skip {b}/{name}: no environment/Dockerfile)")

    if not targets:
        print("nothing to warm")
        return 0

    print(f"==> warming {len(targets)} study images with {args.jobs} parallel build(s)", flush=True)
    t0 = time.time()
    failures = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_one, d, args.retries): d for d in targets}
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
