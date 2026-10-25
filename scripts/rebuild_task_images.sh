#!/usr/bin/env bash
# Rebuild every toolchain-baked task image.
#
# The `rsi` package is baked into the derived images (the agent class and the
# inlined spec run *inside* the container), so any change to rsi/ has to be
# pushed into the images before a run -- otherwise the container would
# execute a stale copy of the harness.
#
#   scripts/rebuild_task_images.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for df in "$ROOT"/datasets/base/tc/*/Dockerfile; do
  name="$(basename "$(dirname "$df")")"
  echo "==> rsi/tc-$name:1"
  docker build -f "$df" -t "rsi/tc-$name:1" "$ROOT" >/dev/null
done

echo "==> done"
docker images --format "{{.Repository}}:{{.Tag}}" | grep "^rsi/" | sort
