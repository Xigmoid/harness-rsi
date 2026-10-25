"""Probe every ClawGym verifier *before* it ever runs inside a container.

Why this exists
---------------

The re-annotation pipeline needs to turn each verifier's stdout into a single
scalar reward, because Harbor's verifier only reads a float out of
``/logs/verifier/reward.txt``.  The verifiers in ``RUC-AIBOX/ClawGym-Task`` do
not agree on a shape:

* some print ``{"reward": 0.0, "<check>": false, ...}``,
* some print a flat ``{"<check>": 0.0, ...}`` with no ``reward`` key at all,
* some print both, and a handful nest the scores one level deeper.

Deciding the extractor by reading 15,402 scripts by eye is not possible, and
guessing from a ``grep`` is worse: the flat-score scripts do not contain the
literal string ``"scores"`` anywhere, so a static scan classifies 9,284 of them
as "no recognised shape".  The only ground truth is the program's own output.

So this script *runs* every verifier once, against an empty workspace, and
records what it printed.  That answers three questions at once:

1. does the verifier run at all with only the standard library (plus ``yaml``
   for the 101 tasks that import it)?  A verifier that needs a third-party
   package is a hard requirement for the base image, and finding out here
   costs seconds instead of a failed trial batch.
2. what is the exact JSON shape, so the reward extractor can be built from
   observation rather than from a guess?
3. does the verifier write anything *outside* the workspace argument?  A
   verifier that writes to an absolute path would pollute the shared image
   between trials, which is a correctness problem for a re-annotation run
   that reuses one image for every task.

The workspace handed to the verifier is empty and read-only-ish, so a
well-behaved verifier scores 0.  That is the expected baseline and it is
also what lets us detect the "no output at all" case (a crash).

Usage::

    python common/registry/probe_verifiers.py \
        --jsonl .cache/clawgym/syn_task.jsonl \
        --out work/verifier_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

#: Verifiers are plain python programs.  Run them with the interpreter that
#: is on PATH; the image carries the same one.
TIMEOUT_SEC = 60

#: Imports that are not in the standard library.  ``yaml`` is the only one
#: that actually shows up, and it is why the base image needs PyYAML.
_NON_STDLIB_HINT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", re.M)


def _reward_shape(stdout: str) -> str:
    """Classify the JSON shape a verifier printed.

    ``reward``   -- the payload carries an explicit numeric ``reward`` key.
    ``flat``     -- the payload is a flat mapping of check name -> number.
    ``nested``   -- the payload nests the scores (``{"scores": {...}}``).
    ``none``     -- nothing that parses as a JSON object was printed.
    """
    obj = _last_json_object(stdout)
    if obj is None:
        return "none"
    if isinstance(obj, dict):
        if isinstance(obj.get("reward"), (int, float)) and not isinstance(
            obj.get("reward"), bool
        ):
            return "reward"
        for key in ("scores", "checks", "results"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner:
                return "nested"
        if obj and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in obj.values()
        ):
            return "flat"
    return "none"


def _last_json_object(stdout: str) -> Any:
    """The last JSON object printed on stdout.

    Verifiers print exactly one JSON payload, but some of them print a
    human-readable line first, and the object may be indented across several
    lines.  Scanning the whole stdout for a balanced object is the reliable
    way to find it.
    """
    decoder = json.JSONDecoder()
    found: Any = None
    index = 0
    length = len(stdout)
    while index < length:
        char = stdout[index]
        if char not in "{[":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(stdout, index)
        except json.JSONDecodeError:
            index += 1
            continue
        found = value
        index = end
    return found


def _probe_one(payload: tuple[int, str, str]) -> dict[str, Any]:
    """Run one verifier and describe what it did."""
    row_index, task_id, source = payload
    record: dict[str, Any] = {
        "row": row_index,
        "task_id": task_id,
        "rc": None,
        "shape": "none",
        "stderr_tail": "",
        "n_keys": 0,
        "reward_value": None,
        "external_write": False,
    }
    workdir = tempfile.mkdtemp(prefix="clawgym-probe-")
    try:
        workspace = os.path.join(workdir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        script = os.path.join(workdir, "test.py")
        Path(script).write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, script, workspace],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            record["stderr_tail"] = "TIMEOUT"
            record["shape"] = "timeout"
            return record
        record["rc"] = completed.returncode
        record["stderr_tail"] = completed.stderr.strip()[-400:]
        record["shape"] = _reward_shape(completed.stdout)
        obj = _last_json_object(completed.stdout)
        if isinstance(obj, dict):
            record["n_keys"] = len(obj)
            value = obj.get("reward")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                record["reward_value"] = float(value)
        # Anything written outside the workspace and the scratch directory is
        # a verifier that would pollute the shared image.
        leaked = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(workdir)
            for name in files
            if not os.path.join(root, name).startswith(workspace)
            and name != "test.py"
        ]
        record["external_write"] = bool(leaked)
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        default=".cache/clawgym/syn_task.jsonl",
        help="the ClawGym task file",
    )
    parser.add_argument(
        "--out",
        default="work/verifier_probe.jsonl",
        help="where to write one probe record per task",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="probe only the first N rows")
    args = parser.parse_args()

    payloads: list[tuple[int, str, str]] = []
    with open(args.jsonl, encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if args.limit and row_index >= args.limit:
                break
            record = json.loads(line)
            meta = record.get("meta_data") or {}
            task_id = meta.get("task_id") or f"row-{row_index:06d}"
            source = None
            for entry in record.get("input_files") or []:
                if entry.get("file_path") == "reward/test.py":
                    source = entry.get("content")
                    break
            if source is None:
                continue
            payloads.append((row_index, task_id, source))

    print(f"probing {len(payloads)} verifiers with {args.workers} workers", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    failures = 0
    with out_path.open("w", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, record in enumerate(
                pool.map(_probe_one, payloads, chunksize=8), start=1
            ):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                key = record["shape"]
                counts[key] = counts.get(key, 0) + 1
                if record["rc"] != 0 or record["shape"] in {"none", "timeout"}:
                    failures += 1
                if index % 1000 == 0:
                    print(f"  {index}/{len(payloads)}", flush=True)

    print(f"shape counts: {counts}")
    print(f"non-zero exit or unparseable: {failures}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
