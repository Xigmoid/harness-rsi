#!/usr/bin/env python3
"""Move the single-seed transfer results out of the home measurement's path.

Why this exists
---------------
The transfer probe measures the *same* (generation, benchmark) pair as the
home measurement, but with one seed instead of three.  Both were written to
``runs/genNNN/<bench>_dev/result.json``, so the transfer result **overwrote**
the home result -- and with it the three-seed noise floor, which is measured
on the base run.

The fix in the framework is the ``--tag`` option: a tagged measurement is
written to ``runs/genNNN/<bench>_dev.<tag>/``.  This script performs the
same separation for the results that were produced *before* the tag existed,
so the study does not have to be re-run: a result with a single seed is a
transfer measurement by construction, and a result with several seeds is the
home measurement.

    python scripts/separate_transfer_results.py --dry-run
    python scripts/separate_transfer_results.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="transfer")
    args = ap.parse_args()

    moved: list[tuple[str, str]] = []
    for gen_dir in sorted(RUNS.glob("gen[0-9][0-9][0-9]")):
        for subset_dir in sorted(gen_dir.glob("*_dev")):
            if subset_dir.name.endswith(f".{args.tag}"):
                continue
            result = subset_dir / "result.json"
            if not result.is_file():
                continue
            try:
                data = json.loads(result.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            seeds = data.get("seeds") or []
            if len(seeds) >= 2:
                continue  # a home measurement: leave it alone
            target = gen_dir / f"{subset_dir.name}.{args.tag}"
            moved.append((str(subset_dir.relative_to(ROOT)), str(target.relative_to(ROOT))))
            if args.dry_run:
                continue
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(subset_dir, target)

    for src, dst in moved:
        print(f"{src}  ->  {dst}")
    print(f"\n{len(moved)} result(s) separated into the '.{args.tag}' namespace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
