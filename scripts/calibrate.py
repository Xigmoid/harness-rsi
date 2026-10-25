#!/usr/bin/env python3
"""Time a Harbor run on a handful of tasks, to price the real study.

    python scripts/calibrate.py aime 4
    python scripts/calibrate.py tb21 2

It truncates the frozen job template's ``task_names`` to the first N tasks,
runs Harbor, and reports the wall clock and the per-task cost.  The frozen
template itself is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HARBOR = str(ROOT / ".venv312" / "bin" / "harbor")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("n_tasks", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-concurrent", type=int, default=None)
    args = ap.parse_args()

    template = ROOT / "harbor" / f"job.{args.benchmark}.yaml"
    job = yaml.safe_load(template.read_text(encoding="utf-8"))
    names = job["datasets"][0].get("task_names") or []
    job["datasets"][0]["task_names"] = names[: args.n_tasks]
    if args.n_concurrent:
        job["n_concurrent_trials"] = args.n_concurrent
    # A calibration run is a throw-away: never write into the study's runs dir.
    job["jobs_dir"] = "/tmp/calib_runs"

    out_dir = Path(args.out or f"/tmp/calib_{args.benchmark}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    job_path = Path(f"/tmp/calib_job.{args.benchmark}.yaml")
    job_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MSWEA_API_KEY", "EMPTY")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("OPENAI_BASE_URL", "http://YOUR-ENDPOINT:PORT/v1")

    t0 = time.time()
    proc = subprocess.run(
        [HARBOR, "job", "start", "-c", str(job_path), "-o", str(out_dir), "--yes"],
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0

    n_ok = 0
    rewards: list[float] = []
    errors: list[str] = []
    for p in sorted(out_dir.rglob("results.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_ok += 1
        vr = data.get("verifier_result") or {}
        rewards.append(float(vr.get("reward") or 0.0))
        exc = data.get("exception_info") or {}
        if exc:
            errors.append(f"{data.get('task_name')}: {exc.get('exception_type')}")

    print(f"--- {args.benchmark}: {args.n_tasks} tasks ---")
    print(f"exit={proc.returncode} wall={elapsed:.1f}s result_files={n_ok}")
    if n_ok:
        print(f"per_task_wall={elapsed / args.n_tasks:.1f}s mean_reward={sum(rewards)/len(rewards):.3f}")
    for e in errors[:5]:
        print(f"  err {e}")
    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-3000:]
    print(tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
