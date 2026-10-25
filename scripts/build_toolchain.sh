#!/usr/bin/env bash
# Build the agent toolchain bundle and export it to ./.cache/rsi-local.
#
# Harbor installs mini-swe-agent inside every task container, which costs
# ~4 minutes and ~1 GB of downloads *per trial*.  The bundle is built once
# (linux/arm64, the architecture the trials run on) and bind-mounted into
# every trial at /root/.local -- uv's default location, so the venv's
# absolute shebangs keep resolving.
#
#   scripts/build_toolchain.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="rsi/toolchain:1"
OUT="$ROOT/.cache/rsi-local"

echo "==> building $IMAGE"
docker build -f "$ROOT/datasets/base/toolchain/Dockerfile" -t "$IMAGE" "$ROOT"

echo "==> exporting /root/.local -> $OUT"
rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
CID="$(docker create "$IMAGE")"
docker cp "$CID:/root/.local" "$OUT"
docker rm "$CID" >/dev/null

echo "==> bundle: $(du -sh "$OUT" | cut -f1)"
ls "$OUT"
test -x "$OUT/bin/mini-swe-agent" || { echo "!! mini-swe-agent shim missing"; exit 1; }
echo "==> ok"
