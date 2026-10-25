"""Build the SWE-Smith mirror cache on the host.

Every SWE-Smith task needs the same repository, at a different
bug-injected commit.  The repository itself is one of **150** distinct
SWE-Smith mirrors (the 7,693 tasks of the family use 150 of them), and the
SWE-Smith mirror repository has **one branch per instance**, so a single
clone of a mirror carries every instance of that repository.

Cloning a mirror once on the host and bind-mounting it read-only into the
containers turns the per-task repository setup from "clone from GitHub" into
"borrow the object store", which is what makes the SWE-Smith family's
per-task initialisation cheap enough to run 8,375 times.

Usage::

    python common/tools/build_mirror_cache.py --jobs 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_MIRRORS = Path("/private/tmp/rsi-swe-mirrors")


def clone_one(slug: str, clone_url: str, target: Path, retries: int) -> dict[str, object]:
    """Clone one SWE-Smith mirror into the cache, if it is not there yet."""
    marker = target / "complete"
    if marker.exists():
        return {"slug": slug, "status": "cached", "seconds": 0.0, "bytes": 0}

    if target.exists():
        shutil.rmtree(target)
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "git",
                "clone",
                "--bare",
                "--quiet",
                clone_url,
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            marker.write_text("ok\n", encoding="utf-8")
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            return {
                "slug": slug,
                "status": "cloned",
                "seconds": round(time.monotonic() - started, 1),
                "bytes": size,
            }
        last_error = (result.stderr or "")[-500:]
        if target.exists():
            shutil.rmtree(target)
        time.sleep(min(30, 3 * attempt))

    return {
        "slug": slug,
        "status": "failed",
        "seconds": round(time.monotonic() - started, 1),
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swesmith-map", default=str(REPO / "work" / "tasks" / "swesmith_map.json")
    )
    parser.add_argument("--out", default=str(DEFAULT_MIRRORS))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    mapping = json.loads(Path(args.swesmith_map).read_text(encoding="utf-8"))["tasks"]
    repos: dict[str, str] = {}
    for entry in mapping.values():
        repos[entry["slug"]] = entry["clone_url"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(repos)} SWE-Smith mirrors to cache into {out}")
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                clone_one, slug, url, out / f"{slug}.git", args.retries
            ): slug
            for slug, url in sorted(repos.items())
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or result["status"] == "failed":
                print(
                    f"[{index}/{len(repos)}] {result['slug']}: {result['status']} "
                    f"{result.get('seconds')}s",
                    flush=True,
                )

    failed = [r for r in results if r["status"] == "failed"]
    total_bytes = sum(int(r.get("bytes", 0)) for r in results)
    summary = {
        "mirrors": len(repos),
        "cloned": sum(1 for r in results if r["status"] == "cloned"),
        "cached": sum(1 for r in results if r["status"] == "cached"),
        "failed": len(failed),
        "bytes": total_bytes,
        "gib": round(total_bytes / 1024**3, 2),
        "failures": failed,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
