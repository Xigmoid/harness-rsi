#!/usr/bin/env python3
"""Regenerate local-glm.yaml = built-in mini.yaml + overrides.yaml.

mini-swe-agent's default config file (MSWEA_MINI_CONFIG_PATH) is used as a
*replacement*, not a merge base, so the file it points at has to be
self-contained. This script keeps it in sync with the installed package after
an upgrade: it reads the built-in mini.yaml, merges the local overrides on
top (using mini-swe-agent's own recursive_merge, so the semantics match the
CLI's), and writes the result.

The round-trip through YAML is asserted to be lossless: a long Jinja template
that YAML decides to fold would silently change the prompt, so we compare the
re-parsed result against the in-memory merge and fail loudly if they differ.

Usage:  python3 regen.py
"""

import sys
from pathlib import Path

import yaml
from minisweagent.config import builtin_config_dir
from minisweagent.utils.serialize import recursive_merge

CONFIG_DIR = Path(__file__).resolve().parent
BASE = builtin_config_dir / "mini.yaml"
OVERRIDES = CONFIG_DIR / "overrides.yaml"
OUTPUT = CONFIG_DIR / "local-glm.yaml"


def main() -> int:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    overrides = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8"))
    merged = recursive_merge(base, overrides)

    text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, width=4096)
    reparsed = yaml.safe_load(text)
    if reparsed != merged:
        print("FATAL: YAML round-trip is lossy; refusing to write local-glm.yaml", file=sys.stderr)
        for key in merged:
            if reparsed.get(key) != merged[key]:
                print(f"  differing section: {key}", file=sys.stderr)
        return 1

    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}  ({len(text)} bytes)")
    print(f"  base       = {BASE}")
    print(f"  overrides  = {OVERRIDES}")
    print(f"  model      = {merged['model']['model_name']} "
          f"({merged['model']['model_class']}) @ "
          f"{merged['model']['model_kwargs']['api_base']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
