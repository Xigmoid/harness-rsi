#!/usr/bin/env python3
"""How much concurrency can the endpoint actually take?

The rollout driver keeps N trials in flight, and each trial is one request
at a time.  So the driver's concurrency is the endpoint's concurrency, and
it decides how many days the 23k rollouts take.

This measures the latency at a few concurrency levels with a *long* prompt
(a few thousand tokens of context plus the tool schemas), because a
one-word prompt would not be representative of what the agent sends.

Usage::

    python common/tools/probe_endpoint_concurrency.py --levels 1,4,8,16
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "http://YOUR-ENDPOINT:PORT/v1/chat/completions"
#: A context big enough to make the prefill cost visible.
FILLER = "The quick brown fox jumps over the lazy dog. " * 200


def one_request(url: str, model: str, n: int) -> tuple[float, int, str]:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{FILLER}\n\nReply with exactly the number {n}.",
                }
            ],
            "max_tokens": 64,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except Exception as error:  # pragma: no cover -- the failure path matters
        return time.monotonic() - started, 0, f"error: {error}"
    elapsed = time.monotonic() - started
    usage = payload.get("usage") or {}
    return elapsed, usage.get("completion_tokens") or 0, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="glm5.3-flash")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--per-level", type=int, default=4, help="requests per level")
    args = ap.parse_args()

    print(f"endpoint: {args.url}")
    for level in [int(x) for x in args.levels.split(",") if x.strip()]:
        latencies: list[float] = []
        errors: list[str] = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(one_request, args.url, args.model, i)
                for i in range(args.per_level)
            ]
            for future in futures:
                elapsed, _, error = future.result()
                latencies.append(elapsed)
                if error:
                    errors.append(error)
        wall = time.monotonic() - started
        ok = [x for x in latencies if x > 0]
        print(
            f"  concurrency {level:>2}: "
            f"p50 {statistics.median(ok):6.1f}s  max {max(ok):6.1f}s  "
            f"wall {wall:6.1f}s for {args.per_level} reqs"
            + (f"  errors={len(errors)}: {errors[0][:60]}" if errors else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
