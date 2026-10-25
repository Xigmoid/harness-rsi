#!/usr/bin/env python3
"""Point the IneqMath verifier's LLM judge at the local endpoint.

Why this exists
---------------
The IneqMath tasks ship a verifier that asks an LLM whether the agent's
answer is mathematically equivalent to the reference:

    JUDGE_MODEL = "gpt-4o-mini"
    ...
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Two things then go wrong on this machine:

1. the endpoint we have (``http://YOUR-ENDPOINT:PORT/v1``) serves exactly
   one model -- ``glm5.3-flash`` -- and answers ``404 NotFoundError`` for
   ``gpt-4o-mini``;
2. ``OpenAI(api_key=...)`` with no ``base_url`` goes to api.openai.com,
   where the key is the placeholder ``EMPTY``, so the call fails with
   ``401 invalid_api_key``.

The verifier catches the failure and writes **-1.0** as the reward, so
every IneqMath trial "fails" for a reason that has nothing to do with the
harness: the measurement was of the endpoint, not of the agent.

What this script changes
------------------------
Exactly one line per task -- the ``JUDGE_MODEL`` constant -- so it reads
the model from the environment and falls back to the published value:

    JUDGE_MODEL = "gpt-4o-mini"
    ->
    JUDGE_MODEL = os.getenv("INEQMATH_JUDGE_MODEL", "gpt-4o-mini")

Nothing else in the verifier is touched: the prompts, the equivalence
check, the parsing, the reward file and the ``-1.0`` failure path are all
byte-identical to the published task.  The manifest records the before/after
hash of every edited file.

The endpoint is then supplied at runtime through the task's own
``[verifier] env`` block, which already exists and already uses Harbor's
``${VAR}`` template syntax:

    [verifier]
    env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

``OPENAI_BASE_URL`` is honoured by the OpenAI SDK itself, so injecting it
is enough to redirect the client -- no code change.

    python scripts/patch_ineqmath_judge.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "datasets" / "registry" / "ineqmath"

OLD = 'JUDGE_MODEL = "gpt-4o-mini"'
NEW = 'JUDGE_MODEL = os.getenv("INEQMATH_JUDGE_MODEL", "gpt-4o-mini")'

#: The plain-text branch of ``query_openai`` hard-codes the model name
#: instead of using the constant, so the constant alone is not enough: the
#: multiple-choice path (which is the one IneqMath actually uses) would
#: still ask the endpoint for ``gpt-4o-mini``.
OLD_CALL = '        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]'
NEW_CALL = '        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}]'

#: The verifier must import ``os`` for the replacement to work.
IMPORT_RE = re.compile(r"^import os$", re.MULTILINE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_one(path: Path) -> dict:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")
    if OLD not in text and OLD_CALL not in text:
        return {"file": str(path.relative_to(ROOT)), "status": "already patched or absent",
                "sha256_before": before, "sha256_after": before}
    if OLD in text:
        text = text.replace(OLD, NEW, 1)
    if OLD_CALL in text:
        text = text.replace(OLD_CALL, NEW_CALL, 1)
    if not IMPORT_RE.search(text):
        # ``import os`` is required by the replacement; add it next to the
        # other stdlib imports, keeping the file otherwise untouched.
        text = re.sub(r"^(import \w+)$", r"\1\nimport os", text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")
    return {
        "file": str(path.relative_to(ROOT)),
        "status": "patched",
        "sha256_before": before,
        "sha256_after": sha256(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(REGISTRY.glob("*/tests/test_outputs.py"))
    if not files:
        print(f"no ineqmath verifiers found under {REGISTRY}")
        return 1

    records = []
    for f in files:
        if args.dry_run:
            text = f.read_text(encoding="utf-8")
            records.append(
                {
                    "file": str(f.relative_to(ROOT)),
                    "status": "would patch" if OLD in text else "already patched or absent",
                }
            )
        else:
            records.append(patch_one(f))

    n_patched = sum(1 for r in records if r["status"] == "patched")
    print(f"patched {n_patched}/{len(files)} ineqmath verifiers")
    if not args.dry_run:
        manifest = {
            "benchmark": "ineqmath",
            "what": "JUDGE_MODEL constant only",
            "old": OLD,
            "new": NEW,
            "n_files": len(records),
            "files": records,
        }
        out = REGISTRY / "JUDGE_PATCH.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
